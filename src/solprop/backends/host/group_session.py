# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference's group session (SEP 47 section 6.7).

:class:`HostGroupSession` is the :class:`~solprop.backends.session.GroupSession` that
serves every ``numpy_ref`` group, and :class:`HostPreparedGroup` is
the prepared group a reference backend returns from ``prepare_group``. The group loads
each member from its start and hands the loaded members to the reference's group runner
(:class:`~solprop.backends.host.group.HostGroup`, or its member-axis arithmetic
:class:`~solprop.backends.numpy_ref.batch.NumpyBatchBackend`, the choice
:func:`solprop.backends.registry.select_batch` makes),
which advances every member through each step of the common fixed schedule in turn.
The session presents that march through the group protocol: one window per save plane,
each member's own events, tickets, counters and outcome, and the failure distinction of
SEP 47 section 6.7.

**Why the runner, rather than one** :class:`~solprop.backends.host.session.HostSession`
**per member.** The native group composes one native session per member
(``solprop_native.group.NativeGroupSession``), each advancing a whole window before the
next. The reference group keeps its runner because what it must preserve is the
runner's, and composed member sessions would change both:

- *Its arithmetic.* A group evaluates its stages on arrays with a leading
  member axis (``NumpyBatchBackend``), whose reductions are ordered differently from a
  lone run's: a member and its lone run agree to rounding, not bitwise
  (``tests/backends/test_batch_lockstep.py``). Member sessions would run the lone path
  and change every group's result.
- *Its failure positions.* Members take each common step in turn, so a failure in one
  member ends the group with the members ahead of it in flight at that step's end
  (poisoned) and those behind it at the step's start (SEP 47 section 6.7). Member
  sessions advancing whole windows in turn would leave the members ahead committed at
  the window's end instead.

Each member's boundary monitor is still its own: a stop ends that member alone while
the others go on (:attr:`~solprop.backends.host.group.HostGroup.member_stops`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from solprop.backends.events import (
    Eligibility,
    Event,
    ExportTicket,
    GroupBatch,
    ObservationKind,
    ObservationReady,
    Outcome,
    Receipt,
    Terminal,
    join_attempts,
)
from solprop.backends.export import ExportRecipe, apply_recipe
from solprop.backends.host.driver import HostDriver
from solprop.backends.protocol import (
    BackendHistory,
    Observation,
    StepStats,
    StepSummary,
    stepping,
)
from solprop.backends.session import (
    SessionError,
    SessionLedger,
    SessionState,
    SessionStatus,
    Start,
    check_group,
    check_member_ticket,
)
from solprop.backends.session_core import EMPTY_SUMMARY, bind, export_weight
from solprop.exceptions import StopRequested
from solprop.log import StopFlag
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    # Under TYPE_CHECKING, so that the runtime type checks of the test suite do not
    # check a refused export destination's shape before the group refuses it.
    from jaxtyping import Complex

    from solprop.backends.protocol import Backend

__all__ = ["HostGroupSession", "HostPreparedGroup"]


class HostGroupSession:
    """The reference's :class:`~solprop.backends.session.GroupSession`.

    As the batched scan always has, the first advance gives every member its boundary
    monitor before the launch plane is reported, so that a launch a monitor refuses
    ends that member at once.

    Parameters
    ----------
    members : collections.abc.Sequence of solprop.backends.protocol.Backend
        One constructed reference backend per member, all of one backend. Once the
        group is constructed it owns them and closes them with itself; a construction
        that fails leaves them to the caller.
    problems : collections.abc.Sequence of solprop.translate.problem.PropagationProblem
        Each member's independently translated problem.
    requests : collections.abc.Sequence of solprop.translate.run_request.RunRequest
        Each member's run request, with its own fresh boundary monitor; their save
        planes and statistics are the group's common schedule.
    starts : collections.abc.Sequence of solprop.backends.session.Start
        Each member's start: its entrance, with no method history or monitor to
        continue. The group takes each start's field.
    stop : solprop.log.StopFlag or None, optional
        A group-wide stop token, polled before every common step.

    Raises
    ------
    solprop.backends.session.SessionError
        If the sequences differ in length or are empty, or if a start is not a fresh
        entrance.
    solprop.exceptions.BackendError
        If the runner refuses the members (``select_batch``): another backend, or
        another schedule, planes or periodic statistics.
    """

    def __init__(
        self,
        members: Sequence[Backend],
        problems: Sequence[PropagationProblem],
        requests: Sequence[RunRequest],
        starts: Sequence[Start],
        *,
        stop: StopFlag | None = None,
    ) -> None:
        """Load every member from its start and build the group runner."""
        from solprop.backends.registry import select_batch  # noqa: PLC0415 - cycle

        count = len(members)
        check_group(count, problems, requests, starts)
        observation = requests[0].observation
        # The reference's members step on the host (`HostStepping`).
        hosted = tuple(stepping(member) for member in members)
        for member, start in zip(hosted, starts, strict=True):
            member.set_initial(
                np.asarray(start.take_field()),
                start.noise,
                z=float(start.z),
                resume=start.resume,
            )
        self._members = hosted
        # The same members, as the reference's marches whose position and attempt
        # record the group reads: the runner below steps host drivers alone.
        self._hosts = cast("tuple[HostDriver[Any], ...]", self._members)
        self._problems = tuple(problems)
        self._requests = tuple(requests)
        self._planes = np.asarray(observation.save_planes, dtype=np.float64)
        self._runner = select_batch(self._members)
        self._runner.set_stop(stop)
        self._ledger = SessionLedger()
        self._launched = False
        self._index = 0
        self._generation = 0
        self._z = float(starts[0].z)
        self._outcomes: list[Outcome | None] = [None] * count
        self._attempts = [0] * count
        self._rejected = [0] * count
        self._observed: dict[int, Observation] = {}
        self._observed_generation = -1
        # Set once a group-wide stop or failure ended every running member.
        self._broken = False
        # Each member's attempts drained since its previous event and not yet delivered.
        self._drained: dict[int, tuple[BackendHistory, StepStats]] = {}

    # -- protocol -------------------------------------------------------------------

    @property
    def n_members(self) -> int:
        """Return the number of members.

        Returns
        -------
        int
            The members.
        """
        return len(self._members)

    @property
    def finished(self) -> bool:
        """Return whether the group's terminal batch was issued.

        Returns
        -------
        bool
            True once every member has ended.
        """
        return self._ledger.finished

    @property
    def closed(self) -> bool:
        """Return whether the group was closed.

        Returns
        -------
        bool
            True once closed.
        """
        return self._ledger.state is SessionState.CLOSED

    def advance(self) -> GroupBatch:
        """Run every running member through the next common window, and report it.

        Returns
        -------
        solprop.backends.events.GroupBatch
            Each member's events in the window.
        """
        self._ledger.check_advance()
        if not self._launched:
            self._launched = True
            return self._launch()
        if self._index >= self._planes.size:
            return self._complete()
        target = float(self._planes[self._index])
        try:
            summary = self._runner.propagate_to(target)
        except StopRequested as error:
            return self._ended(Outcome.STOPPED, error)
        except Exception as error:
            return self._ended(Outcome.FAILED, error)
        self._z = float(summary.z)
        self._generation += 1
        plane = self._index
        self._index += 1
        return self._window(ObservationKind.PLANE, plane, summary)

    def acknowledge(self, receipt: Receipt) -> None:
        """Release the pending batch and allow advancing past it.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The pending batch's receipt.
        """
        self._ledger.acknowledge(receipt)
        self._observed = {}

    def export(
        self,
        member: int,
        ticket: ExportTicket,
        recipe: ExportRecipe,
        out: Complex[np.ndarray, "n_modes n_pol n_omega"] | None = None,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Fill a destination with one member's converted field at a pending ticket.

        Parameters
        ----------
        member : int
            The member's position.
        ticket : solprop.backends.events.ExportTicket
            A ticket of the pending batch, for this member: its position is the
            member's own.
        recipe : solprop.backends.export.ExportRecipe
            The conversion.
        out : numpy.ndarray or None, optional
            The destination; a new array by default.

        Returns
        -------
        numpy.ndarray
            The member's converted field.
        """
        self._ledger.check_ticket(ticket)
        check_member_ticket(ticket, self._member(member))
        recipe, weight = export_weight(self._problems[member], ticket.z, recipe)
        field = self._fielded(ticket, member).require_field()
        return apply_recipe(field, recipe, weight=weight, out=out)

    def noise_phase(self, member: int, ticket: ExportTicket) -> np.ndarray | None:
        """Return one member's noise phase at a pending ticket.

        Parameters
        ----------
        member : int
            The member's position.
        ticket : solprop.backends.events.ExportTicket
            A ticket of the pending batch.

        Returns
        -------
        numpy.ndarray or None
            The phase, shape ``(n_modes, n_pol, n_omega)``, or ``None`` without noise.
        """
        self._ledger.check_ticket(ticket)
        check_member_ticket(ticket, self._member(member))
        phase = self._fielded(ticket, member).noise_phase
        return None if phase is None else np.array(phase)

    def member_status(self, member: int) -> SessionStatus:
        """Return one member's committed position, counters and outcome.

        Parameters
        ----------
        member : int
            The member's position.

        Returns
        -------
        solprop.backends.session.SessionStatus
            Its status; a member that ended reports its own outcome while the group
            goes on.
        """
        index = self._member(member)
        outcome = self._outcomes[index]
        state = self._ledger.state
        if outcome is not None and state is not SessionState.CLOSED:
            state = SessionState.FINISHED
        return SessionStatus(
            z=self._hosts[index].position,
            plane_index=self._index,
            attempts=self._attempts[index],
            rejected=self._rejected[index],
            state=state,
            outcome=outcome,
        )

    def execution_record(self, member: int) -> Mapping[str, object]:
        """Return one member's execution facts, with the group's strategy.

        Parameters
        ----------
        member : int
            The member's position.

        Returns
        -------
        collections.abc.Mapping
            Its backend provenance.
        """
        return self._runner.member_metadata()[self._member(member)]

    def monitor_record(self, member: int) -> Mapping[str, object] | None:
        """Return one member's boundary-monitor record.

        Parameters
        ----------
        member : int
            The member's position.

        Returns
        -------
        collections.abc.Mapping or None
            Its record as it closed or as it stands, or ``None`` when unwatched.
        """
        return self._runner.member_records[self._member(member)]

    def close(self) -> None:
        """Release every member and the runner; idempotent."""
        if self._ledger.close():
            self._runner.set_stop(None)
            self._runner.close()

    def __enter__(self) -> "HostGroupSession":
        """Return the group.

        Returns
        -------
        HostGroupSession
            This group.
        """
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Close the group.

        Parameters
        ----------
        kind : type or None
            The exception type, if one is propagating.
        error : BaseException or None
            The exception, if one is propagating.
        trace : types.TracebackType or None
            Its traceback.
        """
        del kind, error, trace
        self.close()

    # -- internals ------------------------------------------------------------------

    def _member(self, member: int) -> int:
        """Return a valid member position."""
        if not 0 <= member < len(self._members):
            msg = f"the group has {len(self._members)} members, not a member {member}"
            raise SessionError(msg)
        return member

    def _running(self) -> list[int]:
        """Return the members that have not ended."""
        return [index for index, ended in enumerate(self._outcomes) if ended is None]

    def _chunks(self) -> list[tuple[BackendHistory, StepStats]]:
        """Drain every member's attempts and rows since the previous window.

        - Returns: each member's attempts and rows (without a batch axis), in member
          order.
        """
        chunks: list[tuple[BackendHistory, StepStats]] = []
        for index in range(len(self._members)):
            # One member at a time, each kept until an event delivers it: a later
            # member's failure, or the statistics', ends the group with the attempts
            # already drained delivered by the terminals. Each event drains once.
            observation = self._members[index].observe(field=False)
            self._attempts[index] += observation.n_steps
            self._rejected[index] += observation.n_rejected
            self._drained[index] = (
                observation.step_history,
                observation.step_stats,
            )
            chunks.append(self._drained[index])
        return chunks

    def _final_chunk(self, index: int) -> tuple[BackendHistory, StepStats] | None:
        """Drain one member's attempts after a group-wide stop or failure.

        A member left in flight cannot be observed; the attempts it recorded before
        the abandoned step are drained from its record instead
        (:meth:`~solprop.backends.host.driver.HostDriver.drain_attempts`).

        - Returns: its attempts and rows, or ``None`` when its state cannot be read at
          all.
        """
        member = self._hosts[index]
        try:
            observed = (
                member.drain_attempts()
                if member.in_flight is not None
                else member.observe(field=False)
            )
        except Exception:
            return None
        self._attempts[index] += observed.n_steps
        self._rejected[index] += observed.n_rejected
        return observed.step_history, observed.step_stats

    def _ticket(self, z: float | None = None) -> ExportTicket:
        """Return an unbound ticket for a committed state of the current generation.

        - ``z``: the member's committed position, m; the window's end by default.
        - Returns: the ticket, its receipt filled in when the batch is issued.
        """
        return ExportTicket(
            Receipt(self._ledger.session, 0),
            self._generation,
            self._z if z is None else z,
        )

    def _member_stopped(
        self, index: int, chunk: tuple[BackendHistory, StepStats]
    ) -> Terminal | None:
        """Return a stopped member's terminal event, or ``None`` if it runs on.

        - ``chunk``: its attempts and rows since its previous event.
        """
        stop = self._runner.member_stops[index]
        if stop is None:
            return None
        self._outcomes[index] = Outcome.FAILED
        cursor = self._hosts[index].position
        return Terminal(
            Outcome.FAILED,
            cursor,
            Eligibility.ELIGIBLE,
            stop,
            self._ticket(cursor),
            *chunk,
        )

    def _launch(self) -> GroupBatch:
        """Give each member its monitor and report the launch planes."""
        try:
            self._runner.attach_monitors(
                [request.boundary.monitor for request in self._requests]
            )
        except Exception as error:
            return self._ended(Outcome.FAILED, error)
        return self._window(ObservationKind.LAUNCH, -1, EMPTY_SUMMARY)

    def _window(
        self, kind: ObservationKind, plane: int, summary: StepSummary
    ) -> GroupBatch:
        """Report a window's end: each running member's plane, or its boundary stop."""
        try:
            chunks = self._chunks()
            stats = self._runner.stats(self._requests[0].observation.stats)
        except Exception as error:
            return self._ended(Outcome.FAILED, error)
        events: list[tuple[Event, ...]] = []
        for index, chunk in enumerate(chunks):
            if self._outcomes[index] is not None:
                events.append(())
                continue
            stopped = self._member_stopped(index, chunk)
            if stopped is not None:
                events.append((stopped,))
                continue
            events.append(
                (
                    ObservationReady(
                        z=self._z,
                        kind=kind,
                        plane_index=plane,
                        history=chunk[0],
                        step_stats=chunk[1],
                        stats={
                            name: np.asarray(value[index])
                            for name, value in stats.items()
                        },
                        summary=summary,
                        ticket=self._ticket(),
                    ),
                )
            )
        if not self._running():
            # Every member ended at its own boundary: nothing is left to advance.
            return self._issue(events, outcome=Outcome.COMPLETED)
        return self._issue(events)

    def _complete(self) -> GroupBatch:
        """Run each watched member's final check and report every outcome."""
        try:
            chunks = self._chunks()
            self._runner.finish_monitors()
        except Exception as error:
            return self._ended(Outcome.FAILED, error)
        events: list[tuple[Event, ...]] = []
        for index, chunk in enumerate(chunks):
            if self._outcomes[index] is not None:
                events.append(())
                continue
            stopped = self._member_stopped(index, chunk)
            if stopped is not None:
                events.append((stopped,))
                continue
            self._outcomes[index] = Outcome.COMPLETED
            events.append(
                (
                    Terminal(
                        Outcome.COMPLETED,
                        self._hosts[index].position,
                        Eligibility.ELIGIBLE,
                        None,
                        None,
                        *chunk,
                    ),
                )
            )
        return self._issue(events, outcome=Outcome.COMPLETED)

    def _ended(self, outcome: Outcome, error: BaseException) -> GroupBatch:
        """End every running member with a group-wide stop or failure.

        Members take each common step in turn, so after a failure partway through one
        a member ahead of the failing one stands at the step's end, its step neither
        validated nor recorded -- in flight, and so poisoned -- and the others at the
        step's start; each reports its own position.
        """
        self._generation += 1
        self._broken = True
        chunks = [
            join_attempts(
                [
                    self._drained.get(index, (None, None)),
                    self._final_chunk(index) or (None, None),
                ]
            )
            for index in range(len(self._members))
        ]
        events: list[tuple[Event, ...]] = []
        for index, chunk in enumerate(chunks):
            if self._outcomes[index] is not None:
                events.append(())
                continue
            self._outcomes[index] = outcome
            member = self._hosts[index]
            poisoned = member.in_flight is not None
            history, rows = chunk
            cursor = member.position
            events.append(
                (
                    Terminal(
                        outcome,
                        cursor,
                        Eligibility.POISONED if poisoned else Eligibility.ELIGIBLE,
                        error,
                        None if poisoned else self._ticket(cursor),
                        history,
                        rows,
                    ),
                )
            )
        return self._issue(events, outcome=outcome, error=error)

    def _issue(
        self,
        members: list[tuple[Event, ...]],
        *,
        outcome: Outcome | None = None,
        error: BaseException | None = None,
    ) -> GroupBatch:
        """Issue a window's events under a new receipt.

        - ``outcome``: the group's outcome, when the batch ends it.
        """
        receipt = self._ledger.issue(outcome=outcome)
        # Every drained attempt is delivered by this batch.
        self._drained = {}
        bound = tuple(
            tuple(bind(event, receipt, index) for event in events)
            for index, events in enumerate(members)
        )
        return GroupBatch(receipt, bound, outcome, error)

    def _fielded(self, ticket: ExportTicket, member: int) -> Observation:
        """Observe a member's field for a ticket once, shared by its exports.

        A window's fields are observed together, once; after a group-wide failure each
        member is observed alone, since a member left in flight refuses observation.
        """
        if self._observed_generation != ticket.generation:
            self._observed = {}
            self._observed_generation = ticket.generation
        if member not in self._observed:
            # The window's attempts were drained when it ended; nothing ran since.
            if self._broken:
                self._observed[member] = self._members[member].observe(field=True)
            else:
                self._observed.update(
                    enumerate(self._runner.observe_members(field=True))
                )
        return self._observed[member]


class HostPreparedGroup:
    """A prepared group of reference members (SEP 47 section 6.7).

    Each :meth:`start` opens one :class:`HostGroupSession` over the members.

    Parameters
    ----------
    members : collections.abc.Sequence of solprop.backends.protocol.Backend
        One constructed reference backend per member, all of one backend; they stay
        the caller's until a started group owns them.
    problems : collections.abc.Sequence of solprop.translate.problem.PropagationProblem
        Each member's independently translated problem.
    requests : collections.abc.Sequence of solprop.translate.run_request.RunRequest
        Each member's run request.
    """

    def __init__(
        self,
        members: Sequence[Backend],
        problems: Sequence[PropagationProblem],
        requests: Sequence[RunRequest],
    ) -> None:
        """Hold the members, their problems and their requests."""
        self._members = tuple(members)
        self._problems = tuple(problems)
        self._requests = tuple(requests)

    @property
    def backend(self) -> str:
        """Return the registered name of the backend that prepared it.

        Returns
        -------
        str
            The first member's registered name.
        """
        return self._members[0].name

    def start(
        self, starts: Sequence[Start], *, stop: StopFlag | None = None
    ) -> HostGroupSession:
        """Start the group over the reference's group runner.

        Parameters
        ----------
        starts : collections.abc.Sequence of solprop.backends.session.Start
            Each member's entrance; the group takes their fields.
        stop : solprop.log.StopFlag or None, optional
            A group-wide stop token.

        Returns
        -------
        HostGroupSession
            The started group.
        """
        return HostGroupSession(
            self._members, self._problems, self._requests, starts, stop=stop
        )
