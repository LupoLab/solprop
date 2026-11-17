# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The building blocks the reference's session composes (SEP 47 section 5).

The reference's session (:class:`solprop.backends.numpy_ref.session.HostSession`) runs
its own plane schedule over its own march, and composes the parts below, none of which
names a march; the native session keeps the same bookkeeping in ``session.cpp``:

- :class:`SessionLedger` is the lifecycle bookkeeping: receipts, the pending barrier,
  closing and poisoning; it holds no numerical state;
- :class:`AttemptCounter` forwards a march's attempts to a progress listener and
  counts them, and :class:`YieldDue` reads that count and the clock for a bounded
  advance (SEP 47 section 5.1);
- :class:`SnapshotLatch` coalesces snapshot requests until one is serviced (section
  5.5);
- :class:`ObservedField` observes a committed field once per ticket and shares it
  among its exports (section 5.4);
- :func:`merge_summaries`, :func:`bind` and :data:`EMPTY_SUMMARY` assemble events.

The refusals a preparation makes before any session starts are both backends'
(:func:`solprop.backends.session.check_preparation`).
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from solprop.backends.events import Event, ExportTicket, Outcome, Receipt
from solprop.backends.export import ExportRecipe, apply_recipe
from solprop.backends.protocol import (
    BackendHistory,
    Observation,
    StepStats,
    StepSummary,
)
from solprop.backends.session import (
    AttemptListener,
    SessionError,
    SessionState,
    SnapshotReceipt,
)
from solprop.log import NullReporter
from solprop.translate.linop import power_weight_at
from solprop.translate.problem import PropagationProblem

if TYPE_CHECKING:
    # Under TYPE_CHECKING, so that the runtime type checks of the test suite do not
    # check a refused export destination's shape before the session refuses it.
    from jaxtyping import Complex

__all__ = [
    "EMPTY_SUMMARY",
    "AttemptCounter",
    "ObservedField",
    "SessionLedger",
    "SnapshotLatch",
    "YieldDue",
    "bind",
    "export_weight",
    "merge_summaries",
]

EMPTY_SUMMARY = StepSummary(
    z=0.0,
    n_steps=0,
    n_rejected=0,
    max_norm=0.0,
    mean_norm=0.0,
    wall_time=0.0,
    met_tolerance=True,
)
"""The summary of a launch plane, which no propagation reached."""


class AttemptCounter(NullReporter):
    """Counts a driver's attempts and forwards them and its start, and nothing else.

    The service loop reports planes itself; :class:`~solprop.log.NullReporter` drops
    the rest.

    Parameters
    ----------
    listener : callable or None, optional
        Called with each attempt; ``None`` only counts it.
    started : callable or None, optional
        Called once, when the driver first propagates; ``None`` drops it.

    Attributes
    ----------
    count : int
        The attempts heard so far.
    """

    def __init__(
        self,
        listener: AttemptListener | None = None,
        started: Callable[[float, float, int], None] | None = None,
    ) -> None:
        """Hold the listeners."""
        self._listener = listener
        self._started = started
        self.count = 0

    def started(self, z_start: float, z_end: float, n_planes: int) -> None:
        """Forward the driver's start, made when it first propagates.

        Parameters
        ----------
        z_start : float
            Where propagation starts, m.
        z_end : float
            Where it ends, m.
        n_planes : int
            The number of save planes.
        """
        if self._started is not None:
            self._started(z_start, z_end, n_planes)

    def step(self, z: float, dz: float, accepted: bool) -> None:
        """Count one attempt and forward it.

        Parameters
        ----------
        z : float
            Where the attempt ended, m.
        dz : float
            Its length, m.
        accepted : bool
            Whether it was accepted.
        """
        self.count += 1
        if self._listener is not None:
            self._listener(z, dz, accepted)


class YieldDue:
    """Says whether a bounded advance has reached its attempt bound or its deadline.

    A class rather than a closure in a session's ``advance``, so that no function
    object is made per call (each one costs the coverage lane's import hook a compile).

    Parameters
    ----------
    counter : AttemptCounter
        The session's attempt count.
    limit : int
        The attempt count at which the march halts.
    started : float
        The advance's start, in :func:`time.perf_counter` seconds.
    deadline : float
        The seconds after `started` at which the march halts.
    """

    def __init__(
        self, counter: AttemptCounter, limit: int, started: float, deadline: float
    ) -> None:
        self._counter = counter
        self._limit = limit
        self._started = started
        self._deadline = deadline

    def __call__(self) -> bool:
        """Return whether the attempt bound or the deadline has been reached.

        Returns
        -------
        bool
            True once the march should halt at its next accepted endpoint.
        """
        return (
            self._counter.count >= self._limit
            or time.perf_counter() - self._started >= self._deadline
        )


class SnapshotLatch:
    """Holds one outstanding snapshot request until it is serviced (SEP 47 5.5).

    Attributes
    ----------
    pending : int or None
        The outstanding request's number, or ``None``.
    """

    def __init__(self) -> None:
        """Start with nothing requested."""
        self._requests = 0
        self.pending: int | None = None

    def request(self) -> SnapshotReceipt:
        """Latch a request, or join the outstanding one.

        Returns
        -------
        solprop.backends.session.SnapshotReceipt
            The request's receipt, ``coalesced`` when it joined an outstanding one.
        """
        if self.pending is not None:
            return SnapshotReceipt(self.pending, coalesced=True)
        self._requests += 1
        self.pending = self._requests
        return SnapshotReceipt(self.pending, coalesced=False)

    def clear(self) -> None:
        """Mark the outstanding request serviced."""
        self.pending = None


def merge_summaries(before: StepSummary | None, step: StepSummary) -> StepSummary:
    """Merge a propagation's summary with the yields before it on the same plane.

    Parameters
    ----------
    before : solprop.backends.protocol.StepSummary or None
        The plane so far, or ``None`` when `step` is its first propagation.
    step : solprop.backends.protocol.StepSummary
        The latest propagation.

    Returns
    -------
    solprop.backends.protocol.StepSummary
        The summary of the plane so far.
    """
    if before is None:
        return step
    accepted_before = before.n_steps - before.n_rejected
    accepted_now = step.n_steps - step.n_rejected
    accepted = accepted_before + accepted_now
    return StepSummary(
        z=step.z,
        n_steps=before.n_steps + step.n_steps,
        n_rejected=before.n_rejected + step.n_rejected,
        max_norm=max(before.max_norm, step.max_norm),
        mean_norm=0.0
        if accepted == 0
        else (before.mean_norm * accepted_before + step.mean_norm * accepted_now)
        / accepted,
        wall_time=before.wall_time + step.wall_time,
        met_tolerance=before.met_tolerance and step.met_tolerance,
    )


def bind(event: Event, receipt: Receipt) -> Event:
    """Return an event with its ticket bound to the batch that delivers it.

    Parameters
    ----------
    event : solprop.backends.events.Event
        An event, with an unbound ticket or none.
    receipt : solprop.backends.events.Receipt
        The batch's receipt.

    Returns
    -------
    solprop.backends.events.Event
        The event, its ticket bound; the same event when it carries none.
    """
    ticket = getattr(event, "ticket", None)
    if ticket is None:
        return event
    return dataclasses.replace(
        event,  # pyright: ignore[reportArgumentType]
        ticket=dataclasses.replace(ticket, receipt=receipt),
    )


def export_weight(
    problem: PropagationProblem, z: float, recipe: ExportRecipe
) -> tuple[ExportRecipe, np.ndarray | None]:
    """Return the recipe an export applies at `z`, and its true-power weight.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The run's problem.
    z : float
        The observation's position, m.
    recipe : solprop.backends.export.ExportRecipe
        The conversion asked for.

    Returns
    -------
    tuple
        The recipe, without the power conversion on a spatial route, whose field
        measure is already the stored quantity, and the weight, or ``None`` when no
        power conversion is made.
    """
    if recipe.power and problem.transform.geometry is not None:
        recipe = dataclasses.replace(recipe, power=False)
    return recipe, power_weight_at(problem, z) if recipe.power else None


class ObservedField:
    """One committed field per ticket, shared by every export of it (SEP 47 5.4).

    A session observes its committed field once per state generation, when a consumer
    first asks for it; every export of that ticket reads the same observation. An
    observation's arrays are read-only and nothing writes to them again (the next step
    replaces the state rather than writing into it), so an export with nothing
    to convert is that array on a spatial grid, and every other export a new array (or
    the caller's destination). Attempts a field fetch drains are kept for the next
    event's chunk.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The run's problem, for the true-power conversion.
    drain : callable
        ``drain(field=True)`` observes the committed state, draining the attempts.
    """

    def __init__(
        self, problem: PropagationProblem, drain: Callable[..., Observation]
    ) -> None:
        """Hold the problem and the session's drain; nothing is observed yet."""
        self._problem = problem
        self._drain = drain
        self._observed: Observation | None = None
        self._generation = -1
        # Attempts a field fetch drained that no event has delivered yet.
        self._undelivered: list[tuple[BackendHistory, StepStats]] = []

    def release(self) -> None:
        """Forget the observation: the next export of any ticket observes again."""
        self._observed = None

    def undelivered(self) -> list[tuple[BackendHistory, StepStats]]:
        """Return, once, the attempts field fetches drained since the last call.

        Returns
        -------
        list of tuple
            The attempts and rows of each fetch that drained any, in order.
        """
        chunks, self._undelivered = self._undelivered, []
        return chunks

    def export(
        self,
        ticket: ExportTicket,
        recipe: ExportRecipe,
        out: Complex[np.ndarray, "n_modes n_pol n_omega"] | None = None,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Fill a destination with a valid ticket's converted field.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            A ticket the session's ledger has checked.
        recipe : solprop.backends.export.ExportRecipe
            The conversion.
        out : numpy.ndarray or None, optional
            The destination; a new array by default.

        Returns
        -------
        numpy.ndarray
            The converted field.
        """
        spatial = self._problem.transform.geometry is not None
        recipe, weight = export_weight(self._problem, ticket.z, recipe)
        # A view of its own, as every fetch has handed out: a destination a caller
        # passes back is then never taken for the source and is checked as one.
        state = self.observation(ticket).require_field().view()
        identity = not recipe.optics and (weight is None or bool(np.all(weight == 1.0)))
        if identity and out is None and spatial:
            # An export with nothing to convert is the observation's array. A modal
            # field the backend may still hold is detached by a copy, as the
            # true-power conversion always detached it.
            return state
        return apply_recipe(state, recipe, weight=weight, out=out)

    def noise_phase(self, ticket: ExportTicket) -> np.ndarray | None:
        """Return the noise phase at a valid ticket.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            A ticket the session's ledger has checked.

        Returns
        -------
        numpy.ndarray or None
            The phase, shape ``(n_modes, n_pol, n_omega)``, or ``None`` without noise.
        """
        phase = self.observation(ticket).noise_phase
        return None if phase is None else np.array(phase)

    def observation(self, ticket: ExportTicket) -> Observation:
        """Observe the field for a ticket once, sharing it among its consumers.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            A valid ticket.

        Returns
        -------
        solprop.backends.protocol.Observation
            The driver's observation with its field.
        """
        if self._observed is None or self._generation != ticket.generation:
            observed = self._drain(field=True)
            if observed.n_steps or observed.step_stats.n_rows:
                # Attempts a field fetch drained belong to the next event's chunk.
                self._undelivered.append((observed.step_history, observed.step_stats))
            self._observed = observed
            self._generation = ticket.generation
        return self._observed


_SESSION_IDS = itertools.count(1)


class SessionLedger:
    """The lifecycle bookkeeping of a Python session; no numerical state.

    It issues receipts, records the pending barrier, refuses stale or foreign
    receipts and advancing across an unacknowledged barrier, and remembers whether the
    session finished, failed or closed.

    Examples
    --------
    >>> from solprop.backends.numpy_ref.session_parts import SessionLedger
    >>> ledger = SessionLedger()
    >>> receipt = ledger.issue()
    >>> ledger.ticket_valid(receipt)
    True
    >>> ledger.acknowledge(receipt)
    >>> ledger.ticket_valid(receipt)
    False
    """

    def __init__(self) -> None:
        """Start a ready session's ledger."""
        self.session = next(_SESSION_IDS)
        self._sequence = 0
        self._pending: Receipt | None = None
        self.state = SessionState.READY
        self.outcome: Outcome | None = None

    def check_advance(self) -> None:
        """Refuse to advance a session that is pending, finished or closed.

        Raises
        ------
        SessionError
            Naming the state that forbids it.
        """
        if self.state is SessionState.PENDING:
            msg = "acknowledge the pending batch before advancing the session again"
            raise SessionError(msg)
        if self.state is SessionState.FINISHED:
            msg = "the session has ended and cannot advance"
            raise SessionError(msg)
        self.check_open()

    def check_open(self) -> None:
        """Refuse a closed session.

        Raises
        ------
        SessionError
            If the session was closed.
        """
        if self.state is SessionState.CLOSED:
            msg = "the session is closed"
            raise SessionError(msg)

    def issue(self, *, outcome: Outcome | None = None) -> Receipt:
        """Issue the receipt of a new pending batch.

        Parameters
        ----------
        outcome : solprop.backends.events.Outcome or None, optional
            The batch's terminal outcome, when it ends the session.

        Returns
        -------
        solprop.backends.events.Receipt
            The batch's receipt.
        """
        self._sequence += 1
        receipt = Receipt(self.session, self._sequence)
        self._pending = receipt
        self.state = SessionState.PENDING
        if outcome is not None:
            self.outcome = outcome
        return receipt

    def acknowledge(self, receipt: Receipt) -> None:
        """Accept the pending batch's receipt.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The receipt to acknowledge.

        Raises
        ------
        SessionError
            If the receipt is not the pending one: foreign, stale or repeated.
        """
        self.check_open()
        if receipt != self._pending:
            msg = (
                f"receipt {receipt} is not the pending batch's "
                f"({self._pending}): a foreign, stale or repeated acknowledgment"
            )
            raise SessionError(msg)
        self._pending = None
        self.state = (
            SessionState.FINISHED if self.outcome is not None else SessionState.READY
        )

    def ticket_valid(self, receipt: Receipt) -> bool:
        """Return whether tickets of a batch can still be exported.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The batch's receipt.

        Returns
        -------
        bool
            True while that batch is the pending one.
        """
        return self.state is not SessionState.CLOSED and receipt == self._pending

    def check_ticket(self, ticket: ExportTicket) -> None:
        """Refuse an export with an expired or foreign ticket.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The ticket presented.

        Raises
        ------
        SessionError
            If its batch is no longer pending.
        """
        if not self.ticket_valid(ticket.receipt):
            msg = (
                f"the export ticket for z = {ticket.z!r} m has expired: its batch was "
                "acknowledged, or it belongs to another session"
            )
            raise SessionError(msg)

    def close(self) -> bool:
        """Mark the session closed.

        Returns
        -------
        bool
            True the first time, False when it was already closed.
        """
        if self.state is SessionState.CLOSED:
            return False
        self.state = SessionState.CLOSED
        self._pending = None
        return True

    @property
    def pending(self) -> Receipt | None:  # numpydoc ignore=RT01
        """Pending batch's receipt, or ``None``."""
        return self._pending

    @property
    def finished(self) -> bool:  # numpydoc ignore=RT01
        """Whether a terminal batch was issued."""
        return self.outcome is not None
