# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Service closed Julia events without driving numerical trials from Python."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
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
from solprop.backends.julia._descriptors import transfer_memory
from solprop.backends.julia._fft_cache import FFTCache, publish_fft_cache
from solprop.backends.julia._fft_runtime import FFTPlanRecord, read_planning
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
from solprop.backends.julia._monitor import (
    MonitorLog,
    MonitorPayload,
    monitor_error,
    monitor_finished,
    monitor_warning,
    parse_resume,
    read_monitor,
)
from solprop.backends.julia._noise import InputNoiseTransfer
from solprop.backends.julia._observations import ObservationTransfer
from solprop.backends.julia._ownership import JuliaHandle, require_main_thread
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.julia._services import ObservationPayload, observations
from solprop.backends.julia._stop import StopLease
from solprop.backends.julia._transverse import TransverseFormat
from solprop.backends.julia._wide import DF32StartBinding
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
from solprop.monitor import BoundaryCheck
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
_OUTCOMES = {
    5: Outcome.COMPLETED,
    6: Outcome.STOPPED,
    7: Outcome.FAILED,
    9: Outcome.FAILED,
}


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
    observation : ObservationTransfer or None, optional
        Immutable shapes and offsets for Julia-owned observation services.
    monitor : solprop.monitor.BoundaryCheck or None, optional
        Immutable policy for formatting Julia-decided monitor records and notices.
    transverse : TransverseFormat or None, optional
        Host-only spatial message context, without a live edge monitor.
    noise : InputNoiseTransfer or None, optional
        Owned per-start draw, copied into Julia during construction and not retained
        by Python. Its active model was checked by the prepared execution.
    fft_cache : FFTCache or None, optional
        Verified offers for a private per-start planning context. Publication follows
        successful construction, history/monitor restoration and start adoption.
    wide : DF32StartBinding or None, optional
        Owned actual-start operands for DF32 execution, absent for binary64-wide.
        Julia copies its geometry and certifies every actual trial independently.

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
        observation: ObservationTransfer | None = None,
        monitor: BoundaryCheck | None = None,
        transverse: TransverseFormat | None = None,
        noise: InputNoiseTransfer | None = None,
        fft_cache: FFTCache | None = None,
        wide: DF32StartBinding | None = None,
    ) -> None:
        """Validate the start, construct its independent engine, then take the field."""
        runtime.check()
        self._validate_start(start, shape)
        self._runtime = runtime
        self._observation = observation
        self._fft_cache = fft_cache
        self._planning_record: FFTPlanRecord | None = None
        self._monitor = (
            None
            if monitor is None
            else MonitorLog(monitor, start.z, transverse=transverse)
        )
        self._declared_logged = False
        if self._monitor is not None and start.monitor is not None:
            self._monitor = self._monitor.continued(
                parse_resume(
                    start.monitor,
                    faces=0 if transverse is None else len(transverse.faces),
                )
            )
        self._payload: ObservationPayload | None = None
        self._preparation = preparation
        self._shape = shape
        initial = dict(record)
        initial["source_evaluations"] = 0
        self._adams = transfer.adams is not None
        self._history_identity = HistoryIdentity(
            transfer.method, transfer.history_key, transfer.arithmetic.name
        )
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
        self._released: Callable[[JuliaSession], None] | None = released
        self._ledger = SessionLedger()
        self._listener = listener
        self._started = started
        self._launch_owed = start.plane_pending
        self._start_z = start.z
        self._end_z = float(transfer.planes[-1])
        self._n_planes = len(transfer.planes)
        self._stop = stop
        self._token = stop if isinstance(stop, StopToken) else StopToken()
        self._busy = self._servicing = self._held = False
        self._report: _calls.EngineReport | None = None
        self._last_status = SessionStatus(
            start.z, start.plane_index, 0, 0, SessionState.READY
        )
        self._exports: dict[ExportRecipe, tuple[ExportTicket, np.ndarray]] = {}
        self._has_noise = noise is not None
        self._phases: dict[ExportTicket, np.ndarray] = {}
        # Copies retain their local cache until publication. Invalidating an event
        # replaces these dictionaries, so callbacks cannot republish into a new
        # event or a closed session after returning from the foreign copy.
        self._field_exports = 0
        self._abandoned: tuple[float, float] | None = None
        self._group_chunk = 0
        self._lease = StopLease(runtime, self._token)
        self._adopt_engine(transfer, start, restored, noise, wide)

    def _adopt_engine(
        self,
        transfer: RunTransfer,
        start: Start,
        restored: CallRestore | None,
        noise: InputNoiseTransfer | None,
        wide: DF32StartBinding | None,
    ) -> None:
        """Clean numerical ownership if final host adoption fails.

        Parameters
        ----------
        transfer : RunTransfer
            Resolved numerical and control policy.
        start : Start
            Unconsumed input and optional continuation.
        restored : CallRestore or None
            Validated cumulative accounting.
        noise : InputNoiseTransfer or None
            Selected input draw.
        wide : DF32StartBinding or None
            Bound DF32 actual-start operands, copied only during construction.
        """
        runtime = self._runtime
        self._engine = self._start_engine(transfer, start, restored, noise, wide)
        try:
            self._finish_start(start)
        except BaseException:
            try:
                _calls.call_void(runtime, "close_session!", self._engine)
            finally:
                try:
                    self._engine.close()
                finally:
                    self._lease.close()
            raise

    def _finish_start(self, start: Start) -> None:
        """Adopt a fully constructed start before publishing its planning facts.

        Parameters
        ----------
        start : Start
            Validated unconsumed input whose numerical construction has succeeded.
        """
        fft_cache = self._fft_cache
        planning = self._planning_record
        if fft_cache is not None and planning is not None:
            self._accounting.record.update(
                {
                    "fft_plan_fingerprint": fft_cache.fingerprint.digest,
                    "fft_plan_effort": fft_cache.effort,
                    "fft_cache_supplied": fft_cache.file_supplied,
                    "fft_plans_applied": len(planning.applied),
                    "fft_plans_measured": len(planning.measured),
                    "fft_plans_reused": len(planning.reused),
                    "fft_plan_rejections": dict(planning.rejected),
                }
            )
        field = start.take_field()
        try:
            if fft_cache is not None and planning is not None:
                self._accounting.record["fft_cache_written"] = publish_fft_cache(
                    fft_cache, dict(planning.measured)
                )
        except BaseException:
            # Unexpected publication failures cannot orphan numerical ownership
            # or consume a start whose factory call never returned.
            start.field = field
            raise
        # Numerical plans retain their context. Host wisdom copies are preparation
        # facts, not session buffers or a growing propagation cache.
        self._planning_record = None
        self._fft_cache = None

    def _start_engine(
        self,
        transfer: RunTransfer,
        start: Start,
        restored: CallRestore | None,
        noise: InputNoiseTransfer | None,
        wide: DF32StartBinding | None,
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
        noise : InputNoiseTransfer or None
            The selected input draw, copied before launch observation or history work.
        wide : DF32StartBinding or None
            Bound DF32 operands; binary64-wide calls retain their argument list.

        Returns
        -------
        JuliaHandle
            Independently owned engine; failures release all temporary resources.
        """
        runtime = self._runtime
        engine: JuliaHandle | None = None
        planning: JuliaHandle | None = None
        try:
            if self._fft_cache is not None:
                planning = runtime.own_call("prepare_fft_planning", self._fft_cache)
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
                noise,
                bool(start.resume),
                None
                if self._monitor is None or self._monitor.resume is None
                else self._monitor.resume.transverse,
                planning,
                *((wide,) if wide is not None else ()),
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
            self._restore_monitor(engine)
            if planning is not None:
                self._planning_record = read_planning(runtime, planning)
                planning.close()
                planning = None
        except BaseException:
            try:
                if engine is not None:
                    _calls.call_void(runtime, "close_session!", engine)
            finally:
                try:
                    if engine is not None:
                        engine.close()
                finally:
                    try:
                        if planning is not None:
                            planning.close()
                    finally:
                        self._lease.close()
            raise
        return engine

    def _restore_monitor(self, engine: JuliaHandle) -> None:
        """Restore the selected continuations before publishing initial monitor facts.

        Parameters
        ----------
        engine : JuliaHandle
            Newly constructed numerical owner, not yet adopted by the caller.
        """
        if self._monitor is None:
            return
        resume = self._monitor.resume
        if resume is not None:
            _calls.call_void(
                self._runtime,
                "restore_session_monitor!",
                engine,
                resume.numerical,
                resume.plane,
                resume.transverse,
            )
        payload = read_monitor(self._runtime, engine, 0)
        if payload is None:
            raise BackendError("Julia monitor preparation is missing")
        self._monitor = self._monitor.captured(payload)

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

    def _check(self, *, allow_control: bool = False) -> None:
        """Require an open session on main without recursive entry.

        Parameters
        ----------
        allow_control : bool, optional
            Permit a snapshot request from an external service callback, where Julia
            has returned. Recursive numerical advancement and closure still refuse.

        Raises
        ------
        BackendError
            If called from another Python thread.
        SessionError
            If the session is closed or busy.
        """
        require_main_thread()
        self._ledger.check_open()
        if self._busy and not (allow_control and self._servicing):
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
            self._servicing = True
            try:
                for z, dz, accepted in zip(
                    chunk.z.tolist(),
                    chunk.dz.tolist(),
                    chunk.accepted.tolist(),
                    strict=True,
                ):
                    self._listener(z, dz, bool(accepted))
            finally:
                self._servicing = False
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

    def _group_event(
        self,
        report: _calls.EngineReport,
        chunk: int,
        *,
        error: BaseException | None = None,
    ) -> Event:
        """Translate a group-owned barrier without advancing or acknowledging it.

        A forced terminal changes generation but preserves the numerical buffers.
        Rebind already published caches to that generation before retrying any copy;
        each independently completed publication remains exactly once after failure.

        Parameters
        ----------
        report : EngineReport
            Held member scalars selected by the Julia group coordinator.
        chunk : int
            Positive chunk identity, unchanged by forced terminal replacement.
        error : BaseException or None, optional
            Shared group primary cause or member warning-filter failure.

        Returns
        -------
        Event
            Detached event retaining the independently published member arrays.
        """
        previous = self._report
        if previous is not None and self._group_chunk == chunk:
            if self._accounting.generation == previous.generation:
                self._accounting = replace(
                    self._accounting, generation=report.generation
                )
            if (
                self._payload is not None
                and self._payload.generation == previous.generation
            ):
                self._payload = replace(self._payload, generation=report.generation)
            if (
                self._monitor is not None
                and self._monitor.payload is not None
                and self._monitor.payload.generation == previous.generation
            ):
                self._monitor = replace(
                    self._monitor,
                    payload=replace(
                        self._monitor.payload, generation=report.generation
                    ),
                )
        self._group_chunk = chunk
        self._exports = {}
        self._phases = {}
        receipt = self._hold(report)
        history = _calls.attempts(self._runtime, self._engine, report.generation)
        if error is None:
            return self._event(report, receipt, history)
        self._capture_monitor(report.generation)
        payload = self._capture_observations(report.generation)
        return Terminal(
            _OUTCOMES[report.kind],
            report.z,
            Eligibility.POISONED if report.poisoned else Eligibility.ELIGIBLE,
            error,
            None
            if report.poisoned
            else ExportTicket(receipt, report.generation, report.z),
            history,
            payload.rows,
        )

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
        monitored = self._capture_monitor(report.generation)
        payload = self._capture_observations(report.generation)
        rows = payload.rows
        ticket = ExportTicket(receipt, report.generation, report.z)
        if report.kind in _OBSERVATIONS:
            return ObservationReady(
                report.z,
                _OBSERVATIONS[report.kind],
                report.plane,
                chunk,
                rows,
                payload.plane,
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
        if report.kind == 8:
            if self._monitor is None or monitored is None or monitored.notice is None:
                raise BackendError("Julia warning barrier has no temporal notice")
            warning = monitor_warning(
                self._monitor.check, monitored.notice, self._monitor.transverse
            )
            return ServiceRequired(
                ServiceKind.WARNING,
                report.z,
                str(warning),
                type(warning),
                ticket,
                chunk,
                rows,
                warning,
            )
        outcome = _OUTCOMES[report.kind]
        error: BaseException | None = None
        if outcome is Outcome.STOPPED:
            error = StopRequested(self._token if self._stop is None else self._stop)
        elif report.kind == 9:
            if self._monitor is None or monitored is None:
                raise BackendError("Julia boundary stop has no temporal monitor")
            error = monitor_error(
                self._monitor.check, monitored, self._monitor.transverse
            )
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

    def _capture_observations(self, generation: int) -> ObservationPayload:
        """Publish a complete copied payload once, leaving failed copies retryable.

        Parameters
        ----------
        generation : int
            Pending numerical generation.

        Returns
        -------
        ObservationPayload
            Complete detached values for this generation.
        """
        current = self._payload
        if current is None or current.generation != generation:
            current = observations(
                self._runtime, self._engine, generation, self._observation
            )
            self._payload = current
        return current

    def _capture_monitor(self, generation: int) -> MonitorPayload | None:
        """Publish a complete monitor record and its new rows once per generation.

        Parameters
        ----------
        generation : int
            Pending numerical generation, or zero before numerical attachment.

        Returns
        -------
        MonitorPayload or None
            Owned numerical monitor data, absent for an unwatched session.
        """
        current = self._monitor
        if current is None:
            return None
        previous = current.payload
        if previous is not None and previous.generation == generation:
            return previous
        payload = read_monitor(self._runtime, self._engine, generation)
        if payload is None:
            raise BackendError("Julia temporal monitor preparation is missing")
        self._monitor = current.captured(payload)
        transverse = None if self._observation is None else self._observation.transverse
        if (
            not self._declared_logged
            and payload.transverse is not None
            and transverse is not None
            and transverse.declared
            and transverse.policy.has_design
        ):
            self._declared_logged = payload.transverse.log_declared()
        if payload.finished and (previous is None or not previous.finished):
            monitor_finished(payload)
        return payload

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
            self._servicing = True
            try:
                announce(self._start_z, self._end_z, self._n_planes)
            finally:
                self._servicing = False

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
            self._capture_monitor(self._report.generation)
            self._capture_observations(self._report.generation)
            _calls.call_void(
                self._runtime,
                "acknowledge_session",
                self._engine,
                self._report.generation,
            )
            self._ledger.acknowledge(receipt)
            self._exports = {}
            self._phases = {}
            self._payload = None
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
            raise SessionError("the Julia export ticket does not name this held state")
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
            Current ticket for the selected observation or committed endpoint.
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
        cache = self._exports
        held = cache.get(recipe)
        if held is not None and held[0] == ticket:
            if out is None:
                return held[1]
            np.copyto(out, held[1])
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
        self._field_exports += 1
        if out is None:
            destination.setflags(write=False)
            # A main-thread callback may acknowledge or replace the event after
            # the foreign copy returns. Bind publication to the complete ticket,
            # including a terminal replacement within one numerical generation.
            cache[recipe] = (ticket, destination)
        return destination

    def noise_phase(self, ticket: ExportTicket) -> np.ndarray | None:
        """Copy the input draw's absolute unitary phase at the selected coordinate.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            Current ticket for the selected observation or committed endpoint.

        Returns
        -------
        numpy.ndarray or None
            Read-only binary64 radians with the field's shape, detached from Julia;
            None when the problem has no active input noise. Returned arrays remain
            valid after acknowledgment and closure.
        """
        report = self._check_ticket(ticket)
        if not self._has_noise:
            return None
        cache = self._phases
        held = cache.get(ticket)
        if held is None:
            phase = np.empty(self._shape, dtype=np.float64)
            _calls.call_void(
                self._runtime,
                "copy_session_noise_phase!",
                phase,
                self._engine,
                report.generation,
            )
            phase.setflags(write=False)
            cache[ticket] = phase
            return phase
        return held

    def request_snapshot(self) -> SnapshotReceipt:
        """Latch a snapshot at an eligible accepted endpoint.

        Returns
        -------
        solprop.backends.session.SnapshotReceipt
            Request sequence and whether it coalesced with a pending request.
        """
        self._check(allow_control=True)
        if self.finished:
            raise SessionError("the finished session cannot accept a snapshot request")
        request, coalesced = _calls.snapshot(self._runtime, self._engine)
        return SnapshotReceipt(request, coalesced)

    def march_snapshot(self, ticket: ExportTicket) -> MarchSnapshot | None:
        """Copy a pending endpoint's portable Adams history losslessly.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            Current ticket for the selected observation or committed endpoint.

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
        """Return the last published monitor metadata, including after closure.

        Returns
        -------
        collections.abc.Mapping or None
            Shared record with all delivered rows; None for an unwatched session.
        """
        require_main_thread()
        if (
            not self.closed
            and self._ledger.pending is not None
            and self._report is not None
        ):
            self._capture_monitor(self._report.generation)
        return None if self._monitor is None else self._monitor.summary()

    def continue_monitor(self, continuation: Mapping[str, object]) -> None:
        """Restore the first continuation before Julia attaches its temporal monitor.

        Parameters
        ----------
        continuation : collections.abc.Mapping
            The earlier execution's monitor continuation.

        Raises
        ------
        SessionError
            If the numerical monitor already watches the run.
        """
        self._check()
        if self._ledger.pending is not None and self._report is not None:
            self._capture_monitor(self._report.generation)
        current = self._monitor
        if current is None:
            return
        if current.payload is not None and current.payload.attached:
            raise SessionError(
                "a boundary monitor is continued before it watches the run"
            )
        resume = current.resume
        if resume is None:
            resume = parse_resume(
                continuation,
                faces=0
                if current.transverse is None
                else len(current.transverse.faces),
            )
        published = current.continued(resume)
        pending = self._ledger.pending is not None
        if not pending and self._report is not None:
            # Between the launch acknowledgment and attachment no engine generation
            # can be read. Keep its last complete scalar publication until the next
            # barrier, without manufacturing Julia-derived totals on the host.
            published = replace(published, payload=current.payload)
        with committing():
            _calls.call_void(
                self._runtime,
                "restore_session_monitor!",
                self._engine,
                resume.numerical,
                resume.plane,
                resume.transverse,
            )
            self._monitor = published
            if pending or self._report is None:
                generation = 0 if self._report is None else self._report.generation
                self._capture_monitor(generation)

    def undelivered(self, ticket: ExportTicket) -> tuple[BackendHistory, StepStats]:
        """Deliver a failed consumer's held chunk once for its terminal checkpoint.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            Pending ticket for the selected observation or committed endpoint.

        Returns
        -------
        tuple
            Attempts and periodic rows not already returned by an event.
        """
        report = self._check_ticket(ticket)
        if not self._held:
            return join_attempts(())
        self._capture_execution(report.generation)
        self._capture_monitor(report.generation)
        chunk = _calls.attempts(self._runtime, self._engine, report.generation)
        payload = self._capture_observations(report.generation)
        self._held = False
        return chunk, payload.rows

    def pending_ticket(self) -> ExportTicket | None:
        """Return the held observation or endpoint ticket, even after service failure.

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
        if self._phases:
            ledger["input_noise.phase_export"] = sum(
                phase.nbytes for phase in self._phases.values()
            )
        if self._payload is not None:
            ledger.update(
                (f"observation.{name}", size)
                for name, size in transfer_memory(self._payload).items()
            )
        if self._monitor is not None:
            ledger.update(
                (f"monitor.{name}", size)
                for name, size in transfer_memory(self._monitor).items()
            )
        for recipe, (_, values) in self._exports.items():
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
            self._exports = {}
            self._phases = {}
            self._payload = None
            try:
                self._engine.close()
            finally:
                self._preparation = None
                self._observation = None
                self._lease.close()
                released, self._released = self._released, None
                if released is not None:
                    released(self)

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
