# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Service closed Julia events without driving numerical trials from Python."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np

from solprop._interrupts import committing
from solprop.backends.events import (
    Eligibility,
    Event,
    EventBatch,
    ExportTicket,
    ObservationKind,
    ObservationReady,
    Outcome,
    Progress,
    Receipt,
    ServiceKind,
    ServiceRequired,
    Terminal,
    join_attempts,
)
from solprop.backends.export import ExportRecipe, destination_refusal
from solprop.backends.julia import _calls
from solprop.backends.julia._history import (
    CALL_COUNTERS,
    CallRestore,
    HistoryIdentity,
    export_history,
    history_info,
    lower_call_restore,
    restore_history,
)
from solprop.backends.julia._lower import RunTransfer
from solprop.backends.julia._ownership import JuliaHandle, require_main_thread
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.julia._stop import StopLease
from solprop.backends.protocol import BackendHistory, StepStats, StepSummary
from solprop.backends.session import (
    DEFAULT_DEADLINE,
    DEFAULT_WORK_LIMIT,
    SessionError,
    SessionLedger,
    SessionState,
    SessionStatus,
    SnapshotReceipt,
    Start,
    StopToken,
    positive_count,
)
from solprop.exceptions import (
    BackendError,
    ConvergenceError,
    ResponseBudgetError,
    StopRequested,
)
from solprop.log import StopFlag
from solprop.output.schema import MarchSnapshot

if TYPE_CHECKING:
    # Destination admission must run before the optional runtime array-type hook.
    from jaxtyping import Complex, Int

__all__ = ["JuliaSession"]

_OBSERVATIONS = {
    0: ObservationKind.LAUNCH,
    1: ObservationKind.PLANE,
    2: ObservationKind.STEP,
}
_OUTCOMES = {5: Outcome.COMPLETED, 6: Outcome.STOPPED, 7: Outcome.FAILED}


@dataclass(frozen=True)
class _ExecutionRecord:
    """Publish counters, compact orders and their generation as one owned value.

    Attributes
    ----------
    record : dict
        Private detached cumulative execution metadata.
    orders : numpy.ndarray
        Compact accepted ``[order, count]`` pairs, shape ``(n_runs, 2)``.
    generation : int
        Last copied numerical generation; zero precedes the first event.
    """

    record: dict[str, object]
    orders: Int[np.ndarray, "n_runs 2"]
    generation: int


class JuliaSession:
    """Own one Julia march and translate its bounded events into the shared contract.

    Parameters
    ----------
    runtime : JuliaRuntime
        Verified process runtime, entered only from Python's main thread.
    preparation : JuliaHandle
        Immutable engine preparation; its owner outlives this session.
    transfer : RunTransfer
        Resolved numerical policy copied once at session construction.
    start : solprop.backends.session.Start
        Initial state or field restart. Consumed after engine construction succeeds.
    shape : tuple of int
        Host field shape ``(n_modes, n_pol, n_omega)``.
    record : mapping
        Detached execution metadata.
    released : callable
        Called with this session after its resources are released.
    stop : solprop.log.StopFlag or None, optional
        A StopToken attaches an atomic setter; another flag is polled at each advance.
    listener : callable or None, optional
        Observational attempt listener; forces an external yield after every attempt.
    started : callable or None, optional
        Announcement before numerical work, after an owed launch observation.

    Raises
    ------
    BackendError
        For an unsupported restart or an invalid field, before consuming the start.
    SessionError
        For an already consumed start.
    """

    def __init__(  # noqa: PLR0913 - the prepared numerical owner and service context
        self,
        runtime: JuliaRuntime,
        preparation: JuliaHandle,
        transfer: RunTransfer,
        start: Start,
        shape: tuple[int, int, int],
        *,
        record: Mapping[str, object],
        released: Callable[[JuliaSession], None],
        stop: StopFlag | None = None,
        listener: Callable[[float, float, bool], None] | None = None,
        started: Callable[[float, float, int], None] | None = None,
    ) -> None:
        """Validate the start, construct its independent engine, then take the field."""
        runtime.check()
        self._validate_start(start, shape)
        self._runtime = runtime
        self._preparation = preparation
        self._shape = shape
        initial = dict(record)
        initial["source_evaluations"] = 0
        self._adams = transfer.adams is not None
        self._history_identity = HistoryIdentity(transfer.method, transfer.history_key)
        self._march_refusal: str | None = None
        restored = lower_call_restore(start.ledgers, adams=self._adams)
        if self._adams:
            initial["response_calls"] = (
                dict.fromkeys(CALL_COUNTERS, 0)
                if restored is None
                else dict(zip(CALL_COUNTERS, restored.counts, strict=True))
            )
        if transfer.low_storage:
            initial["attempted_source_evaluations"] = 0
            initial["norm_evaluations"] = 0
        self._accounting = _ExecutionRecord(
            initial,
            np.empty((0, 2), dtype=np.int64) if restored is None else restored.orders,
            0,
        )
        self._released = released
        self._ledger = SessionLedger()
        self._listener = listener
        self._started = started
        self._launch_owed = start.plane_pending
        self._start_z = start.z
        self._end_z = float(transfer.planes[-1])
        self._n_planes = len(transfer.planes)
        self._stop = stop
        self._token = stop if isinstance(stop, StopToken) else StopToken()
        self._lease = StopLease(runtime, self._token)
        self._engine = self._start_engine(transfer, start, restored)
        start.take_field()
        self._busy = False
        self._held = False
        self._report: _calls.EngineReport | None = None
        self._last_status = SessionStatus(
            start.z, start.plane_index, 0, 0, SessionState.READY
        )
        self._exports: dict[ExportRecipe, np.ndarray] = {}
        self._abandoned: tuple[float, float] | None = None

    def _start_engine(
        self, transfer: RunTransfer, start: Start, restored: CallRestore | None
    ) -> JuliaHandle:
        """Construct and restore a numerical owner before taking the start's field.

        Parameters
        ----------
        transfer : RunTransfer
            Resolved method and control policy.
        start : Start
            Unconsumed field and optional portable history.
        restored : CallRestore or None
            Validated cumulative ledger, installed before numerical history.

        Returns
        -------
        JuliaHandle
            Independently owned engine; failures release all temporary resources.
        """
        runtime = self._runtime
        engine: JuliaHandle | None = None
        try:
            engine = runtime.own_call(
                "start_low_storage"
                if transfer.low_storage
                else "start_adams"
                if self._adams
                else "start_runge_kutta",
                self._preparation,
                transfer,
                start.field,
                float(start.z),
                int(start.plane_index),
                bool(start.plane_pending),
                self._lease.flag,
                owners=(self._lease,),
            )
            if restored is not None:
                _calls.call_void(
                    runtime, "restore_session_counts!", engine, restored.counts
                )
            if start.march is not None:
                self._march_refusal = restore_history(
                    runtime, engine, start.march, transfer, self._shape
                )
        except BaseException:
            try:
                if engine is not None:
                    _calls.call_void(runtime, "close_session!", engine)
            finally:
                try:
                    if engine is not None:
                        engine.close()
                finally:
                    self._lease.close()
            raise
        return engine

    @staticmethod
    def _validate_start(start: Start, shape: tuple[int, int, int]) -> None:
        """Check the initial field and currently admitted restart records.

        Parameters
        ----------
        start : Start
            Unconsumed field and continuation data.
        shape : tuple of int
            Required ``(n_modes, n_pol, n_omega)`` shape.

        Raises
        ------
        BackendError
            For invalid field data or unsupported continuation records.
        SessionError
            If the start was already consumed.
        """
        if start.field is None:
            raise SessionError("this start was already used")
        if (
            start.field.shape != shape
            or start.field.dtype != np.complex128
            or not np.all(np.isfinite(start.field))
            or not math.isfinite(start.z)
        ):
            raise BackendError("julia requires a finite complex128 starting field")
        if start.noise is not None or start.monitor:
            raise BackendError(
                "this Julia composition does not yet restore noise or monitors"
            )

    def _check(self) -> None:
        """Require an open session on main without recursive entry.

        Raises
        ------
        BackendError
            If called from another Python thread.
        SessionError
            If the session is closed or busy.
        """
        require_main_thread()
        self._ledger.check_open()
        if self._busy:
            raise SessionError("recursive Julia session entry is not allowed")

    @property
    def finished(self) -> bool:
        """Return whether a terminal event has been issued.

        Returns
        -------
        bool
            True once the session reached a terminal barrier.
        """
        return self._ledger.finished

    @property
    def closed(self) -> bool:
        """Return whether the session's resources have been released.

        Returns
        -------
        bool
            True after closure.
        """
        return self._ledger.state is SessionState.CLOSED

    def status(self) -> SessionStatus:
        """Return detached committed-state counters and lifecycle facts.

        Returns
        -------
        solprop.backends.session.SessionStatus
            Position in metres, next plane, counts and lifecycle state.
        """
        require_main_thread()
        if not self.closed:
            self._check()
            z, plane, attempts, rejected, _, _ = _calls.status(
                self._runtime, self._engine
            )
            self._last_status = SessionStatus(
                z, plane, attempts, rejected, self._ledger.state, self._ledger.outcome
            )
        return self._last_status

    def advance(
        self,
        *,
        work_limit: int = DEFAULT_WORK_LIMIT,
        deadline: float = DEFAULT_DEADLINE,
    ) -> EventBatch:
        """Advance Julia to one closed event and deliver its attempt chunk once.

        Parameters
        ----------
        work_limit : int, optional
            Positive attempt bound, reduced to one for an attempt listener.
        deadline : float, optional
            Finite nonnegative numerical work duration, seconds.

        Returns
        -------
        solprop.backends.events.EventBatch
            One event, with detached arrays retained safely across later advances.

        Raises
        ------
        SessionError
            Across an unacknowledged barrier, after closure or terminal completion.
        BackendError
            For an invalid work limit or deadline.
        """
        self._check()
        self._ledger.check_advance()
        limit = positive_count("work_limit", work_limit)
        if not math.isfinite(deadline) or deadline < 0.0:
            raise BackendError("the Julia deadline must be finite and nonnegative")
        if self._listener is not None:
            limit = 1
        self._busy = True
        try:
            self._announce()
            if self._stop is not None and self._stop.is_set():
                self._token.set()
            batch = self._advance(limit, float(deadline))
            self._held = False
            return batch
        finally:
            self._busy = False

    def _advance(self, limit: int, deadline: float) -> EventBatch:
        """Bind one engine event before translating or delivering its owned chunk.

        Parameters
        ----------
        limit : int
            Positive bounded attempt count.
        deadline : float
            Nonnegative numerical work duration, seconds.

        Returns
        -------
        EventBatch
            One receipt-bound event; failed delivery leaves its chunk recoverable.
        """
        failure: BaseException | None = None
        translated: tuple[BackendHistory, Event] | None = None
        try:
            with committing():
                report = _calls.advance(self._runtime, self._engine, limit, deadline)
                receipt = self._hold(report)
                try:
                    chunk = _calls.attempts(
                        self._runtime, self._engine, report.generation
                    )
                    translated = chunk, self._event(report, receipt, chunk)
                except BaseException as error:
                    failure = error
        except BaseException as signalled:
            if failure is not None and signalled is not failure:
                raise signalled from failure
            raise
        if failure is not None:
            raise failure
        assert translated is not None
        chunk, event = translated
        if self._listener is not None:
            for z, dz, accepted in zip(
                chunk.z.tolist(),
                chunk.dz.tolist(),
                chunk.accepted.tolist(),
                strict=True,
            ):
                self._listener(z, dz, bool(accepted))
        return EventBatch(receipt, (event,))

    def _hold(self, report: _calls.EngineReport) -> Receipt:
        """Record a closed numerical event before any fallible delivery work.

        Parameters
        ----------
        report : EngineReport
            Detached engine scalars and export generation.

        Returns
        -------
        Receipt
            The matching Python event receipt.
        """
        self._report = report
        self._held = True
        if report.poisoned:
            _, _, in_flight, z, dz = _calls.failure(self._runtime, self._engine)
            self._abandoned = (z, dz) if in_flight else None
        receipt = self._ledger.issue(outcome=_OUTCOMES.get(report.kind))
        self._capture_execution(report.generation)
        return receipt

    def _capture_execution(self, generation: int) -> None:
        """Retain cumulative counts and encode a generation's accepted orders once.

        Parameters
        ----------
        generation : int
            A bound numerical event, whose acknowledgment has not drained its chunk.
        """
        current = self._accounting
        if generation == current.generation:
            return
        record = _calls.execution(self._runtime, self._engine, generation)
        updated = dict(current.record)
        runs = [(int(order), int(count)) for order, count in current.orders]
        if record.response_calls is not None:
            for order in record.orders.tolist():
                if order == 0:
                    continue
                if runs and runs[-1][0] == order:
                    runs[-1] = (order, runs[-1][1] + 1)
                else:
                    runs.append((order, 1))
            updated["response_calls"] = {
                **dict(zip(CALL_COUNTERS, record.response_calls, strict=True)),
            }
        updated["source_evaluations"] = record.source_evaluations
        if record.low_storage_counts is not None:
            committed, norms = record.low_storage_counts
            updated["source_evaluations"] = committed
            updated["attempted_source_evaluations"] = record.source_evaluations
            updated["norm_evaluations"] = norms
        # Publish only a complete encoding, so a failed allocation can retry without
        # counting a prefix twice. No Julia object or per-step Python callback remains.
        encoded = np.asarray(runs, dtype=np.int64).reshape(-1, 2)
        self._accounting = _ExecutionRecord(updated, encoded, generation)

    def _event(
        self, report: _calls.EngineReport, receipt: Receipt, chunk: BackendHistory
    ) -> Event:
        """Translate closed engine facts without making numerical decisions.

        Parameters
        ----------
        report : EngineReport
            Numerical event scalars and generation.
        receipt : Receipt
            Its bound Python receipt.
        chunk : BackendHistory
            Detached attempt data since the preceding event.

        Returns
        -------
        Event
            The shared observation, progress, service or terminal event.
        """
        rows = StepStats.empty()
        ticket = ExportTicket(receipt, report.generation, report.z)
        if report.kind in _OBSERVATIONS:
            return ObservationReady(
                report.z,
                _OBSERVATIONS[report.kind],
                report.plane,
                chunk,
                rows,
                {},
                StepSummary(
                    report.z,
                    report.plane_attempts,
                    report.plane_rejected,
                    report.max_norm,
                    report.mean_norm,
                    report.wall_time,
                    report.met_tolerance,
                ),
                ticket,
            )
        if report.kind == 3:
            return Progress(report.z, chunk, rows)
        if report.kind == 4:
            return ServiceRequired(
                ServiceKind.SNAPSHOT,
                report.z,
                "snapshot at an accepted endpoint",
                ticket=ticket,
                history=chunk,
                step_stats=rows,
            )
        outcome = _OUTCOMES[report.kind]
        error: BaseException | None = None
        if outcome is Outcome.STOPPED:
            error = StopRequested(self._token if self._stop is None else self._stop)
        elif outcome is Outcome.FAILED:
            code, message, _, _, _ = _calls.failure(self._runtime, self._engine)
            if code == "AdamsBudgetError":
                error = self._budget_error(report)
            elif code in ("StepFloorError", "DomainError", "NonfiniteSourceError"):
                error = ConvergenceError(message)
            else:
                error = BackendError(f"Julia {code}: {message}")
        return Terminal(
            outcome,
            report.z,
            Eligibility.POISONED if report.poisoned else Eligibility.ELIGIBLE,
            error,
            None if report.poisoned or outcome is Outcome.COMPLETED else ticket,
            chunk,
            rows,
        )

    def _budget_error(self, report: _calls.EngineReport) -> ResponseBudgetError:
        """Attach the accepted field and exact charges to the shared budget error.

        Parameters
        ----------
        report : EngineReport
            The pending terminal budget refusal at its committed position.

        Returns
        -------
        ResponseBudgetError
            A shared diagnostic owning a detached internal-convention field.
        """
        from solprop.backends.multistep import (  # noqa: PLC0415 - failure formatting only
            budget_error,
        )

        maximum, spent, required = _calls.budget(
            self._runtime, self._engine, report.generation
        )
        field = np.empty(self._shape, dtype=np.complex128)
        _calls.call_void(
            self._runtime,
            "export_session!",
            field,
            self._engine,
            report.generation,
            self._preparation,
            False,
            False,
        )
        return budget_error(maximum, spent, report.z, required, field)

    def _announce(self) -> None:
        """Deliver the start notification once after the launch barrier."""
        if self._launch_owed:
            self._launch_owed = False
            return
        announce, self._started = self._started, None
        if announce is not None and self._start_z < self._end_z:
            announce(self._start_z, self._end_z, self._n_planes)

    def acknowledge(self, receipt: Receipt) -> None:
        """Release exactly the pending event, expiring its export generation.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The pending batch's receipt.

        Raises
        ------
        SessionError
            For a stale, foreign or repeated receipt.
        """
        self._check()
        if not self._ledger.ticket_valid(receipt):
            raise SessionError("the receipt is stale, foreign or already acknowledged")
        assert self._report is not None
        with committing():
            self._capture_execution(self._report.generation)
            _calls.call_void(
                self._runtime,
                "acknowledge_session",
                self._engine,
                self._report.generation,
            )
            self._ledger.acknowledge(receipt)
            self._exports.clear()
            self._held = False

    def _check_ticket(self, ticket: ExportTicket) -> _calls.EngineReport:
        """Validate a ticket before reading or writing exported data.

        Parameters
        ----------
        ticket : ExportTicket
            Requested state, event receipt and numerical generation.

        Returns
        -------
        EngineReport
            The matching pending event's detached facts.

        Raises
        ------
        SessionError
            If the receipt, generation, position or member is invalid.
        """
        self._check()
        self._ledger.check_ticket(ticket)
        report = self._report
        if (
            report is None
            or report.poisoned
            or ticket.member is not None
            or ticket.generation != report.generation
            or ticket.z != report.z
        ):
            raise SessionError(
                "the Julia export ticket does not name this committed state"
            )
        return report

    def export(
        self,
        ticket: ExportTicket,
        recipe: ExportRecipe,
        out: Complex[np.ndarray, "n_modes n_pol n_omega"] | None = None,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Fill a validated complex128 destination with Julia's fused export recipe.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            Current committed-state ticket.
        recipe : solprop.backends.export.ExportRecipe
            Power and Fourier-convention conversion, in shared operation order.
        out : numpy.ndarray or None, optional
            Writable C-contiguous complex128 destination of the field's shape.

        Returns
        -------
        numpy.ndarray
            The supplied destination, or a detached read-only cached array.

        Raises
        ------
        SessionError
            For a stale, foreign or poisoned ticket.
        BackendError
            For an invalid destination, before it is written.
        """
        report = self._check_ticket(ticket)
        if out is not None and (reason := destination_refusal(out, self._shape)):
            raise BackendError(reason)
        held = self._exports.get(recipe)
        if held is not None:
            if out is None:
                return held
            np.copyto(out, held)
            return out
        destination = np.empty(self._shape, dtype=np.complex128) if out is None else out
        _calls.call_void(
            self._runtime,
            "export_session!",
            destination,
            self._engine,
            report.generation,
            self._preparation,
            recipe.power,
            recipe.optics,
        )
        if out is None:
            destination.setflags(write=False)
            self._exports[recipe] = destination
        return destination

    def noise_phase(self, ticket: ExportTicket) -> np.ndarray | None:
        """Return no noise phase for this noiseless composition.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            Current committed-state ticket.

        Returns
        -------
        numpy.ndarray or None
            None; noise is refused before this composition starts.
        """
        self._check_ticket(ticket)
        return None

    def request_snapshot(self) -> SnapshotReceipt:
        """Latch a snapshot at an eligible accepted endpoint.

        Returns
        -------
        solprop.backends.session.SnapshotReceipt
            Request sequence and whether it coalesced with a pending request.
        """
        self._check()
        if self.finished:
            raise SessionError("the finished session cannot accept a snapshot request")
        request, coalesced = _calls.snapshot(self._runtime, self._engine)
        return SnapshotReceipt(request, coalesced)

    def march_snapshot(self, ticket: ExportTicket) -> MarchSnapshot | None:
        """Copy a pending endpoint's portable Adams history losslessly.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            Current committed-state ticket.

        Returns
        -------
        solprop.output.schema.MarchSnapshot or None
            Detached raw history, or None for RK, before startup, or behind the march.

        Raises
        ------
        SessionError
            For a stale, foreign or otherwise ineligible ticket.
        """
        report = self._check_ticket(ticket)
        if self._adams:
            return export_history(
                self._runtime,
                self._engine,
                report.generation,
                ticket.z,
                self._shape,
                self._history_identity,
            )
        return None

    def march_snapshot_nbytes(self) -> int | None:
        """Read the pending endpoint's history retention size without copying arrays.

        Returns
        -------
        int or None
            Raw state and full store bytes, or None without an eligible pending history.
        """
        self._check()
        ticket = self.pending_ticket()
        if self._adams and ticket is not None:
            info = history_info(
                self._runtime, self._engine, ticket.generation, ticket.z
            )
            return None if info is None else info.nbytes
        return None

    def march_export_refusal(self) -> str | None:
        """Return no refusal for the admitted portable history implementations.

        Returns
        -------
        str or None
            None; RK methods have an explicit field reset, and Adams exports history.
        """
        return None

    @property
    def march_refusal(self) -> str | None:
        """Return why an input snapshot restarted from its field, when applicable.

        Returns
        -------
        str or None
            Shared restoration-record reason, absent after compatible installation.
        """
        return self._march_refusal

    def execution_record(self) -> Mapping[str, object]:
        """Return this session's detached execution choices.

        Returns
        -------
        collections.abc.Mapping
            Backend, device, precision, runtime identity and effective thread budget.
        """
        # The nested compact order record is caller-owned, as are the top-level
        # execution options. A caller cannot corrupt a later checkpoint's counts.
        import copy  # noqa: PLC0415 - only the metadata service needs recursive copying

        if self._ledger.pending is not None and self._report is not None:
            self._capture_execution(self._report.generation)
        current = self._accounting
        record = copy.deepcopy(current.record)
        calls = record.get("response_calls")
        if isinstance(calls, dict):
            calls["orders"] = current.orders.tolist()
        return record

    def monitor_record(self) -> Mapping[str, object] | None:
        """Return no monitor record for this unwatched composition.

        Returns
        -------
        collections.abc.Mapping or None
            None; monitored requests are refused at preparation.
        """
        return None

    def continue_monitor(self, continuation: Mapping[str, object]) -> None:
        """Refuse a nonempty continuation for a composition with no monitor.

        Parameters
        ----------
        continuation : collections.abc.Mapping
            The earlier execution's monitor continuation.

        Raises
        ------
        BackendError
            For a nonempty continuation, without changing numerical state.
        """
        self._check()
        if continuation:
            raise BackendError(
                "this Julia composition has no boundary monitor to continue"
            )

    def undelivered(self, ticket: ExportTicket) -> tuple[BackendHistory, StepStats]:
        """Deliver a failed consumer's held chunk once for its terminal checkpoint.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            Pending committed-state ticket.

        Returns
        -------
        tuple
            Attempts and periodic rows not already returned by an event.
        """
        report = self._check_ticket(ticket)
        if not self._held:
            return join_attempts(())
        self._capture_execution(report.generation)
        chunk = _calls.attempts(self._runtime, self._engine, report.generation)
        self._held = False
        return chunk, StepStats.empty()

    def pending_ticket(self) -> ExportTicket | None:
        """Return a pending committed-state ticket after an external service failure.

        Returns
        -------
        solprop.backends.events.ExportTicket or None
            None without a pending batch, or after closure or poisoning.
        """
        require_main_thread()
        receipt, report = self._ledger.pending, self._report
        if self.closed or receipt is None or report is None or report.poisoned:
            return None
        return ExportTicket(receipt, report.generation, report.z)

    def failure_ticket(self) -> ExportTicket | None:
        """Bind an external failure between batches without advancing numerically.

        Returns
        -------
        solprop.backends.events.ExportTicket or None
            The pending state, or a new failed barrier after acknowledgment; None
            before the first batch, after terminal completion, closure or poisoning.
        """
        require_main_thread()
        if self.closed or self._report is None or self._report.poisoned:
            return None
        if self._ledger.pending is not None:
            return self.pending_ticket()
        if self.finished:
            return None
        self._check()
        with committing():
            self._hold(_calls.fail(self._runtime, self._engine))
        return self.pending_ticket()

    @property
    def in_flight(self) -> tuple[float, float] | None:
        """Return the start and length of an abandoned in-place step, metres.

        Returns
        -------
        tuple or None
            None unless an in-place failure poisoned the state.
        """
        return self._abandoned

    def memory_ledger(self) -> dict[str, int]:
        """Return every retained session buffer's accounted bytes.

        Returns
        -------
        dict
            Julia numerical storage and cached detached exports by name.
        """
        self._check()
        ledger = _calls.memory(self._runtime, self._engine)
        ledger["execution.response_orders"] = self._accounting.orders.nbytes
        for recipe, values in self._exports.items():
            ledger[f"export_power_{recipe.power}_optics_{recipe.optics}"] = (
                values.nbytes
            )
        return ledger

    def close(self) -> None:
        """Detach cancellation and release the session on main, idempotently."""
        require_main_thread()
        if self.closed:
            return
        self._check()
        status = self.status()
        try:
            _calls.call_void(self._runtime, "close_session!", self._engine)
        finally:
            self._ledger.close()
            self._last_status = SessionStatus(
                status.z,
                status.plane_index,
                status.attempts,
                status.rejected,
                SessionState.CLOSED,
                status.outcome,
            )
            self._exports.clear()
            try:
                self._engine.close()
            finally:
                self._lease.close()
                self._released(self)

    def check_close(self) -> None:
        """Refuse a recursive close before an execution owner changes any lifetime.

        Raises
        ------
        SessionError
            If a numerical advance or external callback is in progress.
        """
        require_main_thread()
        if not self.closed:
            self._check()

    def __enter__(self) -> JuliaSession:
        """Return the open session for scoped ownership.

        Returns
        -------
        JuliaSession
            This session.
        """
        self._check()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Close the session when its scope ends.

        Parameters
        ----------
        kind : type or None
            The exception's type, if any.
        error : BaseException or None
            The exception, if any.
        trace : types.TracebackType or None
            Its traceback, if any.
        """
        self.close()
