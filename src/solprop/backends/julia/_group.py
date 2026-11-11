# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Assemble common windows from a Julia-owned fixed-schedule member group."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np

from solprop._interrupts import committing
from solprop.backends.events import (
    Eligibility,
    Event,
    ExportTicket,
    GroupBatch,
    ObservationReady,
    Outcome,
    Progress,
    Receipt,
    ServiceKind,
    ServiceRequired,
    Terminal,
    emit_warning,
    join_attempts,
)
from solprop.backends.julia import _calls, _group_calls
from solprop.backends.julia._descriptors import transfer_memory
from solprop.backends.julia._ownership import require_main_thread
from solprop.backends.julia._stop import StopLease
from solprop.backends.session import (
    SessionError,
    SessionLedger,
    SessionState,
    SessionStatus,
    Start,
    StopToken,
    check_group,
    check_member_ticket,
)
from solprop.exceptions import BackendError, ResponseBudgetError, StopRequested

if TYPE_CHECKING:
    from jaxtyping import Complex

    from solprop.backends.export import ExportRecipe
    from solprop.backends.julia.backend import JuliaBackend
    from solprop.backends.protocol import BackendHistory, StepStats
    from solprop.log import StopFlag
    from solprop.translate.problem import PropagationProblem
    from solprop.translate.run_request import RunRequest

__all__ = ["JuliaGroupSession", "check_compatible"]

_OUTCOMES = {0: None, 1: Outcome.COMPLETED, 2: Outcome.STOPPED, 3: Outcome.FAILED}


def check_compatible(
    owners: Sequence[JuliaBackend],
    problems: Sequence[PropagationProblem],
    requests: Sequence[RunRequest],
    starts: Sequence[Start],
) -> None:
    """Check the common request before constructing or consuming any member state.

    Parameters
    ----------
    owners : sequence of JuliaBackend
        Distinct execution owners, already checked by their factory.
    problems : sequence of PropagationProblem
        Independently translated member descriptors.
    requests : sequence of RunRequest
        Member requests, each with one fixed itinerary.
    starts : sequence of Start
        Fresh entrance fields, retained unchanged on refusal.

    Raises
    ------
    BackendError
        Naming an incompatible method, shape, norm mask, statistics or execution choice.
    SessionError
        Naming a missing or reused entrance.
    """
    check_group(len(owners), problems, requests, starts)
    first, reference = requests[0], problems[0]
    for index, (owner, problem, request, start) in enumerate(
        zip(owners, problems, requests, starts, strict=True)
    ):
        owner._check()
        if request.replay is None:
            raise BackendError("julia groups require an explicit fixed schedule")
        if request.observation.step_planes or request.observation.dense:
            raise BackendError("julia groups currently require landed save planes")
        if start.z != problem.initial.z or start.ledgers is not None or start.resume:
            raise SessionError("julia groups require an unadvanced fresh entrance")
        if first.replay is None:
            raise BackendError("julia groups require an explicit fixed schedule")
        schedule, expected = request.replay.schedule, first.replay.schedule
        spec, expected_spec = request.observation.stats, first.observation.stats
        same = (
            request.integration.kind == first.integration.kind
            and request.integration.method == first.integration.method
            and all(
                np.array_equal(
                    getattr(request.integration.tableau, name),
                    getattr(first.integration.tableau, name),
                )
                for name in ("matrix", "nodes", "solution_weights", "error_weights")
            )
            and problem.shape == reference.shape
            and problem.precision == reference.precision
            and np.array_equal(problem.grids.norm_mask, reference.grids.norm_mask)
            and schedule.z_start == expected.z_start
            and np.array_equal(schedule.steps, expected.steps)
            and np.array_equal(schedule.edges, expected.edges)
            and np.array_equal(
                request.observation.save_planes, first.observation.save_planes
            )
            and (spec.names, spec.periodic, spec.period)
            == (expected_spec.names, expected_spec.periodic, expected_spec.period)
            and np.array_equal(spec.energy_windows, expected_spec.energy_windows)
            and spec.same_tables(expected_spec)
            and owner._setup_id == owners[0]._setup_id
            and owner._wide == owners[0]._wide
        )
        if not same:
            raise BackendError(
                f"julia group member {index} differs in method, layout, schedule, "
                "statistics or execution identity"
            )


@dataclass
class _Window:
    """Retain unpublished member chunks and its latest closed event.

    Attributes
    ----------
    chunks : list
        Detached attempt histories and statistic rows, in delivery order.
    event : Event or None
        Latest observation or terminal in this common window.
    """

    chunks: list[tuple[BackendHistory | None, StepStats | None]] = field(
        default_factory=list
    )
    event: Event | None = None


class JuliaGroupSession:
    """Own independent member sessions while Julia selects every numerical turn.

    Parameters
    ----------
    owners : sequence of JuliaBackend
        Distinct owners transferred only after successful construction.
    problems : sequence of PropagationProblem
        Independently lowered member problems.
    requests : sequence of RunRequest
        Compatible fixed-schedule requests.
    starts : sequence of Start
        Fresh entrances; failure retains every original field.
    stop : solprop.log.StopFlag or None, optional
        Common cancellation, polled by Julia at accepted common round endpoints.
    """

    def __init__(
        self,
        owners: Sequence[JuliaBackend],
        problems: Sequence[PropagationProblem],
        requests: Sequence[RunRequest],
        starts: Sequence[Start],
        *,
        stop: StopFlag | None = None,
    ) -> None:
        check_compatible(owners, problems, requests, starts)
        self._owners = tuple(owners)
        self._ledger = SessionLedger()
        self._busy = False
        self._stop = stop
        self._token = stop if isinstance(stop, StopToken) else StopToken()
        self._report: _group_calls.GroupReport | None = None
        self._batch: GroupBatch | None = None
        self._tickets: dict[int, ExportTicket] = {}
        self._chunks = [0] * len(owners)
        self._events = [0] * len(owners)
        self._outcomes: list[Outcome | None] = [None] * len(owners)
        self._errors: dict[int, BaseException] = {}
        self._primary: BaseException | None = None
        self._windows: list[_Window] | None = None
        self._recovering = False
        with ExitStack() as cleanup:
            sessions = []
            for owner, problem, request, start in zip(
                owners, problems, requests, starts, strict=True
            ):
                session = owner.prepare(problem, request).start(
                    replace(start), owns=False
                )
                cleanup.callback(session.close)
                sessions.append(session)
            self._members = tuple(sessions)
            self._runtime = sessions[0]._runtime
            if any(session._runtime is not self._runtime for session in sessions):
                raise BackendError(
                    "julia group members require the same runtime identity"
                )
            self._lease = StopLease(self._runtime, self._token)
            cleanup.callback(self._lease.close)
            self._engine = self._runtime.own_call(
                "start_group_session",
                self._lease.flag,
                *(session._engine for session in sessions),
                owners=(*sessions, self._lease),
            )
            cleanup.callback(self._engine.close)
            fields = tuple(start.field for start in starts)
            try:
                with committing():
                    for owner in owners:
                        owner._groups.add(self)
                    for start in starts:
                        start.take_field()
            except BaseException:
                with committing():
                    for owner in owners:
                        owner._groups.discard(self)
                    for start, original in zip(starts, fields, strict=True):
                        start.field = original
                raise
            cleanup.pop_all()

    @property
    def n_members(self) -> int:
        """Return the number of independently owned members.

        Returns
        -------
        int
            Positive member count.
        """
        return len(self._members)

    @property
    def finished(self) -> bool:
        """Return whether the group's terminal batch has been issued.

        Returns
        -------
        bool
            Whether every running member has ended.
        """
        return self._ledger.finished

    @property
    def closed(self) -> bool:
        """Return whether numerical ownership has been released.

        Returns
        -------
        bool
            True after closure.
        """
        return self._ledger.state is SessionState.CLOSED

    def _check(self) -> None:
        """Refuse closed, recursive or non-main-thread group entry."""
        require_main_thread()
        self._ledger.check_open()
        if self._busy:
            raise SessionError("recursive Julia group entry is not allowed")

    def _index(self, member: int) -> int:
        """Validate a zero-based member ordinal.

        Parameters
        ----------
        member : int
            Requested member.

        Returns
        -------
        int
            Validated ordinal.
        """
        if (
            isinstance(member, bool)
            or not isinstance(member, int)
            or not 0 <= member < self.n_members
        ):
            raise SessionError(f"the Julia group has no member {member!r}")
        return member

    def advance(self) -> GroupBatch:
        """Run Julia-owned transports through the next complete common window.

        Returns
        -------
        solprop.backends.events.GroupBatch
            Each member's own observations or terminal, with exactly-once chunks.
        """
        self._check()
        if not self._recovering:
            self._ledger.check_advance()
            self._windows = [_Window() for _ in self._members]
        assert self._windows is not None
        windows = self._windows
        self._busy = True
        for session in self._members:
            session._busy = True
        try:
            if self._recovering:
                with committing():
                    self._collect(windows)
                    batch = self._publish(windows)
                self._windows, self._recovering = None, False
                return batch
            while True:
                if self._stop is not None and self._stop.is_set():
                    self._token.set()
                with committing():
                    self._report = _group_calls.advance(self._runtime, self._engine)
                    self._collect(windows)
                    if self._report.kind in (2, 3):
                        batch = self._publish(windows)
                    else:
                        batch = None
                        _calls.call_void(
                            self._runtime,
                            "acknowledge_group!",
                            self._engine,
                            self._report.generation,
                        )
                if batch is not None:
                    self._windows = None
                    return batch
        except BaseException as error:
            failed = self._failed_window(windows, error)
            if isinstance(error, Exception):
                self._windows, self._recovering = None, False
                return failed
            raise
        finally:
            self._busy = False
            for session in self._members:
                session._busy = False

    def _failed_window(
        self, windows: list[_Window], error: BaseException
    ) -> GroupBatch:
        """Retain copied work and publish a shared external failure.

        Parameters
        ----------
        windows : list of _Window
            Durable chunks already copied for the common window.
        error : BaseException
            Cause retained unless an earlier primary cause exists.

        Returns
        -------
        GroupBatch
            Failed window, without another numerical advance.
        """
        self._recovering = True
        if self._primary is None:
            self._primary = error
        with committing():
            self._report = _group_calls.fail(self._runtime, self._engine)
            self._collect(windows)
            return self._publish(windows)

    def _collect(self, windows: list[_Window]) -> None:
        """Copy each new member event and service warnings at the held barrier.

        Parameters
        ----------
        windows : list of _Window
            Durable per-member assembly, extended only for new chunks.
        """
        report = self._report
        assert report is not None
        slots = _group_calls.members(
            self._runtime, self._engine, report.generation, self.n_members
        )
        events: dict[int, Event] = {}
        if report.outcome == 2 and self._primary is None:
            self._primary = StopRequested(
                self._token if self._stop is None else self._stop
            )
        elif report.outcome == 3 and self._primary is None and report.primary >= 0:
            slot = slots[report.primary]
            primary = self._members[report.primary]._group_event(
                slot.report, slot.chunk
            )
            assert isinstance(primary, Terminal)
            self._primary = primary.error
            events[report.primary] = primary
        for index, slot in enumerate(slots):
            if (
                not slot.available
                or not slot.new_event
                or self._events[index] == slot.report.generation
            ):
                continue
            error = self._errors.get(index)
            if report.outcome in (2, 3) and self._outcomes[index] is None:
                error = self._primary
            event = events.get(index)
            if event is None:
                event = self._members[index]._group_event(
                    slot.report, slot.chunk, error=error
                )
            self._receive(index, slot, event, windows[index])
        # A raised warning changed just that member's event without draining its chunk.
        updated = _group_calls.report(self._runtime, self._engine, report.generation)
        self._report = updated
        if any(
            index in self._errors and self._outcomes[index] is None
            for index in self._errors
        ):
            self._collect(windows)

    def _receive(
        self, index: int, slot: _group_calls.MemberReport, event: Event, window: _Window
    ) -> None:
        """Commit one copied chunk and retain its closed member event.

        Parameters
        ----------
        index : int
            Zero-based member ordinal.
        slot : MemberReport
            Numerical generation, chunk and eligibility facts.
        event : Event
            Detached event with owned arrays.
        window : _Window
            This member's durable common-window assembly.
        """
        if self._chunks[index] != slot.chunk:
            window.chunks.append((event.history, event.step_stats))
            self._chunks[index] = slot.chunk
        self._events[index] = slot.report.generation
        if isinstance(event, ServiceRequired):
            self._serve(index, event)
        elif isinstance(event, ObservationReady | Terminal):
            if isinstance(event, Terminal):
                if not slot.eligible and event.outcome is not Outcome.COMPLETED:
                    event = replace(
                        event, eligibility=Eligibility.POISONED, ticket=None
                    )
                self._outcomes[index] = event.outcome
            window.event = event
        elif not isinstance(event, Progress):
            raise BackendError("Julia group returned an unknown member event")

    def _serve(self, index: int, event: ServiceRequired) -> None:
        """Format a warning and retain a warning-filter failure for its member.

        Parameters
        ----------
        index : int
            Zero-based member ordinal.
        event : ServiceRequired
            Closed service facts; non-warning services need no Python work here.
        """
        if event.kind is not ServiceKind.WARNING:
            return
        try:
            emit_warning(event, stacklevel=4)
        except Warning as error:
            assert self._report is not None
            _calls.call_void(
                self._runtime,
                "fail_group_member!",
                self._engine,
                self._report.generation,
                index + 1,
            )
            self._errors[index] = error

    def _publish(self, windows: list[_Window]) -> GroupBatch:
        """Commit a receipt only after all window allocations succeed.

        Parameters
        ----------
        windows : list of _Window
            Durable assembled chunks and final member events.

        Returns
        -------
        GroupBatch
            Complete common window with its newly committed receipt.
        """
        report = self._report
        assert report is not None
        outcome = (
            Outcome.FAILED
            if self._primary is not None and report.outcome == 1
            else _OUTCOMES[report.outcome]
        )
        batch, tickets = self._ledger.publish(
            lambda receipt: self._publication(windows, receipt, outcome),
            outcome=outcome,
        )
        self._batch, self._tickets = batch, tickets
        return batch

    def _publication(
        self, windows: list[_Window], receipt: Receipt, outcome: Outcome | None
    ) -> tuple[GroupBatch, dict[int, ExportTicket]]:
        """Build a detached batch and private tickets before committing ownership.

        Parameters
        ----------
        windows : list of _Window
            Complete per-member assembly.
        receipt : Receipt
            Prospective receipt, still uncommitted during allocation.
        outcome : Outcome or None
            Group terminal outcome, or None while running.

        Returns
        -------
        GroupBatch
            Fully allocated common window.
        dict
            Member ordinals mapped to their held private tickets.
        """
        report = self._report
        assert report is not None
        result: list[tuple[Event, ...]] = []
        tickets: dict[int, ExportTicket] = {}
        for index, window in enumerate(windows):
            event = window.event
            if event is None:
                result.append(())
                continue
            assert isinstance(event, ObservationReady | Terminal)
            history, rows = join_attempts(window.chunks)
            if event.ticket is not None:
                tickets[index] = event.ticket
                event = replace(
                    event,
                    ticket=ExportTicket(
                        receipt, report.generation, event.z, member=index
                    ),
                )
            result.append((replace(event, history=history, step_stats=rows),))
        return GroupBatch(receipt, tuple(result), outcome, self._primary), tickets

    def acknowledge(self, receipt: Receipt) -> None:
        """Expire the public window's tickets without releasing member barriers.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            Current public window receipt.
        """
        self._check()
        if not self._ledger.ticket_valid(receipt):
            raise SessionError("the Julia group receipt is stale or foreign")
        assert self._report is not None
        with committing():
            _calls.call_void(
                self._runtime,
                "acknowledge_group!",
                self._engine,
                self._report.generation,
            )
            self._ledger.acknowledge(receipt)
            self._tickets.clear()
            self._batch = None

    def _ticket(self, member: int, ticket: ExportTicket) -> ExportTicket:
        """Resolve a valid public group ticket to its held member ticket.

        Parameters
        ----------
        member : int
            Zero-based member ordinal.
        ticket : ExportTicket
            Public ticket to validate against this generation and coordinate.

        Returns
        -------
        ExportTicket
            Private member-session ticket.
        """
        self._check()
        index = self._index(member)
        self._ledger.check_ticket(ticket)
        check_member_ticket(ticket, index)
        private = self._tickets.get(index)
        if (
            private is None
            or self._report is None
            or ticket.generation != self._report.generation
            or ticket.z != private.z
        ):
            raise SessionError("the Julia group ticket does not name this held member")
        return private

    def export(
        self,
        member: int,
        ticket: ExportTicket,
        recipe: ExportRecipe,
        out: Complex[np.ndarray, "n_modes n_pol n_omega"] | None = None,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Export one held member into a validated owned complex128 destination.

        Parameters
        ----------
        member : int
            Zero-based member index.
        ticket : ExportTicket
            Current group ticket for that member.
        recipe : solprop.backends.export.ExportRecipe
            Power normalization and Fourier convention.
        out : numpy.ndarray or None, optional
            Destination of shape ``(n_modes, n_pol, n_omega)``.

        Returns
        -------
        numpy.ndarray
            Converted complex128 field, independent of numerical owners.
        """
        private = self._ticket(member, ticket)
        return self._members[member].export(private, recipe, out)

    def noise_phase(self, member: int, ticket: ExportTicket) -> np.ndarray | None:
        """Read the held member's noise phase when its composition admits noise.

        Parameters
        ----------
        member : int
            Zero-based member index.
        ticket : ExportTicket
            Current group ticket.

        Returns
        -------
        numpy.ndarray or None
            Detached phase, or None without input noise.
        """
        private = self._ticket(member, ticket)
        return self._members[member].noise_phase(private)

    def member_status(self, member: int) -> SessionStatus:
        """Read one member's actual committed counters and independent outcome.

        Parameters
        ----------
        member : int
            Zero-based member index.

        Returns
        -------
        SessionStatus
            Committed position and counts; no common-cursor rollback is invented.
        """
        require_main_thread()
        index = self._index(member)
        status = self._members[index].status()
        state = (
            SessionState.FINISHED
            if self._outcomes[index] is not None
            else self._ledger.state
        )
        return replace(
            status,
            state=SessionState.CLOSED if self.closed else state,
            outcome=self._outcomes[index],
        )

    def execution_record(self, member: int) -> Mapping[str, object]:
        """Return detached member provenance and the Julia group strategy.

        Parameters
        ----------
        member : int
            Zero-based member index.

        Returns
        -------
        collections.abc.Mapping
            Actual member execution metadata.
        """
        require_main_thread()
        return dict(self._members[self._index(member)].execution_record()) | {
            "batch_size": self.n_members,
            "batch_strategy": "julia_group_session",
            "fused_batch_kernels": False,
        }

    def monitor_record(self, member: int) -> Mapping[str, object] | None:
        """Return one member's detached temporal-monitor continuation record.

        Parameters
        ----------
        member : int
            Zero-based member index.

        Returns
        -------
        collections.abc.Mapping or None
            Owned monitor metadata, retained after closure.
        """
        require_main_thread()
        return self._members[self._index(member)].monitor_record()

    def check_close(self) -> None:
        """Refuse owner closure while a group transport or service is in flight."""
        require_main_thread()
        if not self.closed:
            self._check()

    def _memory(self) -> dict[str, int]:
        """Count group and delivery buffers outside the members' own ledgers.

        Returns
        -------
        dict
            Named retained byte counts, including unpublished recovery windows.
        """
        self._check()
        result = _group_calls.memory(self._runtime, self._engine)
        assembly = (
            ()
            if self._windows is None
            else tuple((tuple(window.chunks), window.event) for window in self._windows)
        )
        diagnostic = (
            self._primary.field
            if isinstance(self._primary, ResponseBudgetError)
            else None
        )
        shared = tuple(
            (member._payload, member._accounting, member._monitor, member._exports)
            for member in self._members
        )
        result.update(
            (f"delivery.{name}", size)
            for name, size in transfer_memory(
                (self._batch, assembly, diagnostic), exclude=shared
            ).items()
        )
        return result

    def close(self) -> None:
        """Release the group, every member and every adopted owner, idempotently."""
        require_main_thread()
        if self.closed:
            return
        self._check()
        self._ledger.close()
        with ExitStack() as cleanup:
            for owner in self._owners:
                owner._groups.discard(self)
                cleanup.callback(owner.close)
            for session in self._members:
                cleanup.callback(session.close)
            cleanup.callback(self._lease.close)
            cleanup.callback(self._engine.close)
            _calls.call_void(self._runtime, "close_group!", self._engine)
        self._tickets.clear()
        self._batch = None
        self._windows = None
        self._recovering = False

    def __enter__(self) -> JuliaGroupSession:
        """Return this open group for scoped ownership.

        Returns
        -------
        JuliaGroupSession
            This group.
        """
        self._check()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Close the owned group when leaving its scope.

        Parameters
        ----------
        kind : type or None
            Propagating exception type.
        error : BaseException or None
            Propagating exception.
        trace : types.TracebackType or None
            Propagating traceback.
        """
        del kind, error, trace
        self.close()
