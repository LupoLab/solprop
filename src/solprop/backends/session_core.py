# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The building blocks the Python execution sessions share (SEP 47 sections 5 and 6.7).

The reference's sessions (:class:`solprop.backends.host.session.HostSession` and
:class:`solprop.backends.host.group_session.HostGroupSession`) run their own plane
schedule over their own driver, and compose the parts below, none of which names a
driver; the native backend's prepared execution (``solprop_native.session``) makes the
same preparation refusals:

- :class:`AttemptCounter` forwards a driver's attempts to a progress listener and
  counts them, and :class:`YieldDue` reads that count and the clock for a bounded
  advance (SEP 47 section 5.1);
- :class:`SnapshotLatch` coalesces snapshot requests until one is serviced (section
  5.5);
- :class:`ObservedField` observes a committed field once per ticket and shares it
  among its exports, converting it in place where nothing else holds it (section 5.4);
- :func:`merge_summaries`, :func:`bind` and :data:`EMPTY_SUMMARY` assemble events;
- :func:`foreign_request` and :func:`check_preparation` are the refusals a
  preparation makes before any session starts (section 5.2).

The lifecycle bookkeeping -- receipts, the pending barrier, closing and poisoning -- is
:class:`solprop.backends.session.SessionLedger`.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from solprop.backends.events import Event, ExportTicket, Receipt
from solprop.backends.export import ExportRecipe, apply_recipe
from solprop.backends.protocol import (
    BackendHistory,
    Observation,
    StepStats,
    StepSummary,
)
from solprop.backends.session import (
    SessionError,
    SnapshotReceipt,
    statistics_unavailable,
)
from solprop.exceptions import BackendError
from solprop.log import NullReporter
from solprop.translate.linop import power_weight_at
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    # Under TYPE_CHECKING, so that the runtime type checks of the test suite do not
    # check a refused export destination's shape before the session refuses it.
    from jaxtyping import Complex

__all__ = [
    "EMPTY_SUMMARY",
    "AttemptCounter",
    "AttemptListener",
    "ObservedField",
    "SnapshotLatch",
    "YieldDue",
    "bind",
    "check_preparation",
    "export_weight",
    "foreign_request",
    "merge_summaries",
]

AttemptListener = Callable[[float, float, bool], None]
"""Called with each attempt's end, length and acceptance, for a progress reporter."""

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


def bind(event: Event, receipt: Receipt, member: int | None = None) -> Event:
    """Return an event with its ticket bound to the batch that delivers it.

    Parameters
    ----------
    event : solprop.backends.events.Event
        An event, with an unbound ticket or none.
    receipt : solprop.backends.events.Receipt
        The batch's receipt.
    member : int or None, optional
        The group member the ticket names; ``None`` for a lone session.

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
        ticket=dataclasses.replace(ticket, receipt=receipt, member=member),
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

    A session observes its driver's field once per state generation, when a consumer
    first asks for it; every export of that ticket reads the same observation. An
    observation's arrays are read-only and nothing writes to them again
    (:meth:`solprop.backends.protocol.Backend.observe`), so an export with nothing to
    convert is that array; one the driver made for the session alone is converted in
    place when it was not handed out unconverted. Attempts a field fetch drains are
    kept for the next event's chunk.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The run's problem, for the true-power conversion.
    drain : callable
        ``drain(field=True)`` observes the driver, draining its attempts.
    """

    def __init__(
        self, problem: PropagationProblem, drain: Callable[..., Observation]
    ) -> None:
        """Hold the problem and the driver's drain; nothing is observed yet."""
        self._problem = problem
        self._drain = drain
        self._observed: Observation | None = None
        self._generation = -1
        self._converted: ExportRecipe | None = None
        # Whether the observation's own array was handed out unconverted (export).
        self._lent = False
        # Attempts a field fetch drained that no event has delivered yet.
        self._undelivered: list[tuple[BackendHistory, StepStats]] = []

    def release(self) -> None:
        """Forget the observation: the next export of any ticket observes again."""
        self._observed = None
        self._converted = None
        self._lent = False

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
        observed = self.observation(ticket)
        converted = self._converted
        if converted is not None and (converted != recipe or out is not None):
            # Converted in place for another recipe, and handed out so: observe the raw
            # field again rather than write to an array a consumer holds.
            self._observed = None
            self._converted = None
            observed = self.observation(ticket)
        # A view of its own, as every fetch has handed out: a destination a caller
        # passes back is then never taken for the source and is checked as one.
        state = observed.require_field().view()
        if self._converted is not None:
            # Converted in place for this recipe already: the same detached result.
            return state
        identity = not recipe.optics and (weight is None or bool(np.all(weight == 1.0)))
        if identity and out is None and (observed.field_owned or spatial):
            # An export with nothing to convert is the observation's array. A modal
            # field the backend may still hold is detached by a copy, as the
            # true-power conversion always detached it. Once lent, the array is never
            # converted in place.
            self._lent = True
            return state
        if out is None and observed.field_owned and not self._lent:
            # The driver made this array for this session alone, which hands out only
            # the export: convert it where it is (SEP 47 section 5.4).
            self._converted = recipe
            # The flag is the observation's own array's, which `state` is.
            whole = observed.require_field()
            whole.setflags(write=True)
            try:
                apply_recipe(whole, recipe, weight=weight, out=whole)
            finally:
                whole.setflags(write=False)
            return whole
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
            self._converted = None
            self._lent = False
        return self._observed


def foreign_request(own: RunRequest, request: RunRequest) -> str | None:
    """Return why `request` is not a run of `own`, or ``None`` when it is.

    A backend lowers its method, error control, replay and itinerary where it is
    built (SEP 47 section 5.2), so a run it serves may differ from its own request
    only where :meth:`~solprop.translate.run_request.RunRequest.for_run` changes it:
    the start, retention, statistics, monitor, history and snapshots. ``for_run``
    keeps the other sections by identity, which is what is checked.

    Parameters
    ----------
    own : solprop.translate.run_request.RunRequest
        The request the backend was built for.
    request : solprop.translate.run_request.RunRequest
        The request one run asks it to serve.

    Returns
    -------
    str or None
        The first section that differs, named, or ``None``.

    Examples
    --------
    >>> from solprop.stepper import DOPRI5, PIController
    >>> from solprop.translate.problem import StepperSettings
    >>> from solprop.translate.run_request import describe_run
    >>> settings = StepperSettings(DOPRI5, PIController(rtol=1e-8, atol=1e-14),
    ...                            1e-9, 0.1, 1e-3, 1.1, None, [0.5, 1.0], "auto")
    >>> own = describe_run(settings)
    >>> foreign_request(own, own.for_run(start_plane=1)) is None
    True
    >>> foreign_request(own, describe_run(settings))
    "the run's integration is not the backend's own"
    """
    sections = (
        ("integration", own.integration, request.integration),
        ("error control", own.error_control, request.error_control),
        ("replay", own.replay, request.replay),
        ("planes", own.observation.save_planes, request.observation.save_planes),
    )
    for name, mine, theirs in sections:
        if mine is not theirs:
            return f"the run's {name} is not the backend's own"
    return None


def check_preparation(
    name: str, own: RunRequest, problem: PropagationProblem, request: RunRequest
) -> None:
    """Refuse a run a constructed backend cannot prepare, by name (SEP 47 5.2).

    Parameters
    ----------
    name : str
        The backend's registered name, for the message.
    own : solprop.translate.run_request.RunRequest
        The request the backend was built for.
    problem : solprop.translate.problem.PropagationProblem
        The run's problem.
    request : solprop.translate.run_request.RunRequest
        The run's request.

    Raises
    ------
    solprop.backends.session.SessionError
        If the request is not one run of the backend's own (:func:`foreign_request`).
    solprop.exceptions.BackendError
        If the request asks for statistics the problem does not carry
        (:func:`~solprop.backends.session.statistics_unavailable`).
    """
    foreign = foreign_request(own, request)
    if foreign is not None:
        msg = f"{name} cannot prepare this run: {foreign}"
        raise SessionError(msg)
    stats = request.observation.stats
    unavailable = statistics_unavailable(problem, (*stats.names, *stats.periodic))
    if unavailable is not None:
        # Refused by name before any session starts, on every executor.
        msg = f"{name} cannot prepare this run: {unavailable}"
        raise BackendError(msg)
