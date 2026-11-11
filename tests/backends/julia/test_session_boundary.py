# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise Python session transactions around a scripted, nonnumerical engine.

The script supplies closed engine events and already-computed fields. It performs
no integration or control decisions. Real Julia numerics and runtime qualification
are separate; these tests exercise the actual handle registry, call wrappers,
session adapter, cancellation attachment and execution-owner lifecycle.
"""

import ctypes
import gc
import signal
from collections import deque
from dataclasses import astuple, dataclass, fields, replace
from types import SimpleNamespace
from weakref import WeakSet, ref

import numpy as np
import pytest

from backends.julia.cases import cpu_case
from solprop import _interrupts
from solprop.backends.events import (
    Eligibility,
    ObservationKind,
    ObservationReady,
    Outcome,
    Progress,
    ServiceKind,
    ServiceRequired,
    Terminal,
)
from solprop.backends.export import ExportRecipe
from solprop.backends.julia import _calls, _runtime
from solprop.backends.julia import _session as session_module
from solprop.backends.julia._lower import RunTransfer
from solprop.backends.julia._observations import lower_observations
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.julia._services import ObservationPayload
from solprop.backends.julia._session import JuliaSession
from solprop.backends.julia.backend import JuliaBackend, JuliaPrepared
from solprop.backends.protocol import BackendHistory, StepStats
from solprop.backends.session import SessionError, SessionState, Start, StopToken
from solprop.exceptions import BackendError, ConvergenceError, StopRequested
from solprop.output.schema import MarchSnapshot
from solprop.stats import StatsSpec
from solprop.stepper.controller import PIController
from solprop.translate.run_request import ErrorControlSpec

SHAPE = (1, 1, 2)
IDENTITY = ExportRecipe(power=False, optics=False)


def test_closed_observation_session_releases_request_preparation(boundary):
    case = cpu_case()
    request = replace(
        case.request,
        observation=replace(
            case.request.observation, stats=StatsSpec(names=("energy",))
        ),
    )
    transfer = lower_observations(case.problem, request)
    handle = boundary.runtime.owners.retain(object())
    handle_ref, transfer_ref = ref(handle), ref(transfer)
    prepared = JuliaPrepared(
        boundary.owner,
        boundary.runtime,
        handle,
        _transfer(),
        SHAPE,
        observation=transfer,
    )
    held = ref(prepared)
    session = prepared.start(
        Start(np.ones(SHAPE, dtype=np.complex128), 0.0), owns=False
    )
    del prepared, handle, transfer
    gc.collect()
    assert held() is not None
    session.close()
    gc.collect()
    assert held() is None and handle_ref() is None and transfer_ref() is None


def test_close_releases_observation_payload_and_preserves_delivered_rows(boundary):
    session = boundary.prepared.start(
        Start(np.ones(SHAPE, dtype=np.complex128), 0.0), owns=False
    )
    rows = StepStats(np.array([0.1]), np.array([0]), {"energy": np.array([[[2.0]]])})
    session._payload = ObservationPayload(1, rows, {})
    session.close()
    assert session._payload is None
    np.testing.assert_array_equal(rows.values["energy"], [[[2.0]]])


def _history(*, z=None):
    if z is None:
        return BackendHistory(
            np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=bool), np.empty(0)
        )
    return BackendHistory(
        np.array([z]),
        np.array([0.1]),
        np.array([0.25]),
        np.array([True]),
        np.array([0.001]),
    )


@dataclass
class _Frame:
    report: _calls.EngineReport
    history: BackendHistory
    field: np.ndarray
    failure: tuple[str, str, bool, float, float]


def _frame(kind, *, z=0.0, plane=0, attempts=0, poisoned=False, code="StepFloorError"):
    history = _history(z=z) if attempts else _history()
    return _Frame(
        _calls.EngineReport(
            kind,
            z,
            -1 if kind == 0 else plane,
            0,
            attempts,
            0,
            int(bool(attempts)),
            0,
            0.25 if attempts else 0.0,
            0.25 if attempts else 0.0,
            0.001 if attempts else 0.0,
            True,
            poisoned,
        ),
        history,
        np.full(SHAPE, 1.0 + z + 0.5j, dtype=np.complex128),
        (code, "scripted engine failure", poisoned, z, 0.1),
    )


def _transfer():
    controller = PIController(rtol=1e-6, atol=0.0, weights=np.ones(2))
    control = ErrorControlSpec(controller, 1e-12, 1.0, 0.1, 1.0, False, False)
    return RunTransfer(
        control,
        np.array([0.0, 1.0]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([0.5, 0.5]),
        np.array([-0.5, 0.5]),
        np.ones(2, dtype=bool),
        np.ones(2, dtype=bool),
        np.ones(2),
        np.array([1.0]),
        np.array([0.0, 1.0]),
        np.empty(0),
        np.empty(0),
        np.empty(0),
        0,
        10.0,
    )


class _Engine:
    def __init__(self):
        self.frames = deque()
        self.sessions = []
        self.advances = []
        self.acknowledgments = []
        self.exports = []
        self.closed = []
        self.construct_error = None
        self.close_error = None
        self.after_ack = None
        self.module = SimpleNamespace(
            StopFlag=lambda address: SimpleNamespace(address=address),
            start_runge_kutta=self.start,
            advance_session=SimpleNamespace(_jl_call_nogil=self.advance),
            acknowledge_session=self.acknowledge,
            session_status=self.status,
            session_failure=lambda session: session.frame.failure,
            session_execution=lambda session, generation: (0, None, None),
            fail_session=self.fail,
        )
        for name, operation in (
            ("export_session!", self.export),
            ("session_attempts!", self.attempts),
            ("request_snapshot!", self.snapshot),
            ("close_session!", self.close),
        ):
            setattr(self.module, name, operation)

    def start(  # noqa: PLR0913, PLR0917 - mirrors the fixed Julia bridge entry
        self, preparation, transfer, field, z, plane, plane_pending, stop
    ):
        if self.construct_error is not None:
            raise self.construct_error
        session = SimpleNamespace(
            preparation=preparation,
            initial=np.array(field, copy=True),
            stop=stop,
            frames=deque(self.frames),
            pending=False,
            finished=False,
            closed=False,
            generation=0,
            snapshot_pending=False,
            snapshot_sequence=0,
            frame=_frame(0, z=z, plane=plane),
            attempts=SimpleNamespace(count=0),
        )
        self.sessions.append(session)
        return session

    def _issue(self, session, frame):
        assert not session.pending and not session.closed and not session.finished
        session.generation += 1
        session.frame = replace(
            frame, report=replace(frame.report, generation=session.generation)
        )
        session.pending = True
        session.finished = frame.report.kind in (5, 6, 7)
        session.attempts.count = frame.history.n_steps
        if frame.report.kind == 4:
            session.snapshot_pending = False
        return astuple(session.frame.report)

    def advance(self, session, work_limit, deadline):
        self.advances.append((work_limit, deadline))
        return self._issue(session, session.frames.popleft())

    def acknowledge(self, session, generation):
        assert session.pending and generation == session.generation
        self.acknowledgments.append(generation)
        session.pending = False
        session.attempts.count = 0
        if self.after_ack is not None:
            self.after_ack()

    def attempts(  # noqa: PLR0913, PLR0917 - mirrors the fixed Julia bridge entry
        self, positions, lengths, norms, accepted, elapsed, session, generation
    ):
        assert session.pending and generation == session.generation
        history = session.frame.history
        for destination, source in zip(
            (positions, lengths, norms, accepted, elapsed),
            (
                history.z,
                history.dz,
                history.error_norm,
                history.accepted,
                history.wall_time,
            ),
            strict=True,
        ):
            np.copyto(destination, source)

    def export(self, destination, session, generation, preparation, power, optics):
        assert session.pending and generation == session.generation
        assert preparation is session.preparation
        self.exports.append((generation, power, optics))
        # Recipe arithmetic is deliberately absent: this is a precomputed field.
        np.copyto(destination, session.frame.field)

    def status(self, session):
        report = session.frame.report
        return (
            report.z,
            report.plane,
            report.attempts,
            report.rejected,
            session.generation,
            report.poisoned,
        )

    def snapshot(self, session):
        assert not session.closed and not session.finished
        coalesced = session.snapshot_pending
        if not coalesced:
            session.snapshot_sequence += 1
            session.snapshot_pending = True
        return session.snapshot_sequence, coalesced

    def fail(self, session):
        assert not session.pending
        previous = session.frame.report
        return self._issue(
            session,
            _frame(7, z=previous.z, plane=previous.plane, code="ExternalFailure"),
        )

    def close(self, session):
        session.closed = True
        session.pending = False
        self.closed.append(session)
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def boundary(monkeypatch, setup_record):
    monkeypatch.setattr(_runtime, "_STATE", _runtime._State())
    monkeypatch.setattr(_interrupts, "_GUARD", _interrupts._Guard())
    engine = _Engine()
    stopped = []

    def store(address):
        ctypes.cast(address, ctypes.POINTER(ctypes.c_uint8))[0] = 1
        stopped.append(address)

    # This local callback tests cancellation attachment only, never Julia's LLVM leaf.
    setter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(store)
    address = ctypes.cast(setter, ctypes.c_void_p).value
    assert address is not None
    runtime = JuliaRuntime(setup_record, 1, None, engine.module, address)
    preparation = runtime.owners.retain(object())
    # Scientific admission is tested separately. Supply only the actual owner's
    # lifetime state needed by JuliaPrepared and its real close/release methods.
    owner = object.__new__(JuliaBackend)
    owner._runtime = runtime
    owner._preparation = preparation
    owner._closed = False
    owner._sessions = WeakSet()
    owner._groups = WeakSet()
    owner._prepared = WeakSet()
    monkeypatch.setattr(owner, "_lowered", (), raising=False)
    owner._options = {}
    owner._wide = "binary64"
    owner._node_chunk = 1
    owner._observation = None
    prepared = JuliaPrepared(owner, runtime, preparation, _transfer(), SHAPE)
    harness = SimpleNamespace(
        engine=engine,
        owner=owner,
        prepared=prepared,
        runtime=runtime,
        preparation=preparation,
        stopped=stopped,
    )
    yield harness
    owner.close()
    runtime.owners.close()


def _start(boundary, *frames, **kwargs):
    boundary.engine.frames.extend(frames)
    start = Start(np.ones(SHAPE, dtype=np.complex128), 0.0)
    session = boundary.prepared.start(start, owns=False, **kwargs)
    assert start.field is None
    return session


def _launch(session):
    batch = session.advance()
    (event,) = batch.events
    assert isinstance(event, ObservationReady)
    assert event.kind is ObservationKind.LAUNCH
    session.acknowledge(batch.receipt)


def test_prepared_ledgers_count_each_owned_run_array_once(boundary, monkeypatch):
    monkeypatch.setattr(_calls, "memory", lambda *args, **kwargs: {})
    first = boundary.prepared
    second = JuliaPrepared(
        boundary.owner, boundary.runtime, boundary.preparation, _transfer(), SHAPE
    )
    expected = {}
    for descriptor in fields(first._transfer):
        one = getattr(first._transfer, descriptor.name)
        two = getattr(second._transfer, descriptor.name)
        if isinstance(one, np.ndarray):
            assert not np.shares_memory(one, two)
            expected[descriptor.name] = one.nbytes
    ledger = first.memory_ledger()
    for index in range(2):
        assert {
            name.removeprefix(f"run.{index}."): size
            for name, size in ledger.items()
            if name.startswith(f"run.{index}.")
        } == expected
    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert sum(first.memory_ledger().values()) == sum(expected.values())


def test_started_session_retains_execution_owner_without_run_transfer(boundary):
    prepared = JuliaPrepared(
        boundary.owner, boundary.runtime, boundary.preparation, _transfer(), SHAPE
    )
    prepared_ref = ref(prepared)
    array_ref = ref(prepared._transfer.weights)
    boundary.engine.frames.extend((_frame(0), _frame(5, z=1.0)))
    session = prepared.start(
        Start(np.ones(SHAPE, dtype=np.complex128), 0.0), owns=False
    )
    del prepared
    gc.collect()
    assert prepared_ref() is None and array_ref() is None
    assert tuple(boundary.owner._prepared) == (boundary.prepared,)
    assert not boundary.owner.closed
    _launch(session)
    terminal = session.advance()
    assert terminal.terminal is not None
    assert terminal.terminal.outcome is Outcome.COMPLETED
    session.acknowledge(terminal.receipt)
    session.close()
    assert not boundary.owner._sessions


def test_scripted_events_bind_receipts_and_export_generations(boundary):
    session = _start(
        boundary,
        _frame(0),
        _frame(3, z=0.1, attempts=1),
        _frame(1, z=1.0, attempts=2),
        _frame(5, z=1.0, attempts=2),
    )
    launch = session.advance(work_limit=3, deadline=0.25)
    (event,) = launch.events
    assert isinstance(event, ObservationReady)
    ticket = event.ticket
    assert ticket is not None
    with pytest.raises(SessionError, match="acknowledge"):
        session.advance()
    assert boundary.engine.advances == [(3, 0.25)]
    assert session.status().state is SessionState.PENDING
    kept = session.export(ticket, IDENTITY)
    assert not kept.flags.writeable
    assert session.export(ticket, IDENTITY) is kept
    destination = np.empty(SHAPE, dtype=np.complex128)
    assert session.export(ticket, IDENTITY, out=destination) is destination
    assert np.array_equal(destination, kept)
    assert len(boundary.engine.exports) == 1
    session.acknowledge(launch.receipt)
    with pytest.raises(SessionError, match=r"stale|expired"):
        session.export(ticket, IDENTITY)
    with pytest.raises(SessionError, match="stale"):
        session.acknowledge(launch.receipt)
    progress = session.advance(work_limit=2, deadline=0.0)
    (progress_event,) = progress.events
    assert isinstance(progress_event, Progress)
    assert progress_event.history.z.tolist() == [0.1]
    session.acknowledge(progress.receipt)
    plane = session.advance()
    (observed,) = plane.events
    assert isinstance(observed, ObservationReady)
    assert observed.kind is ObservationKind.PLANE
    assert observed.summary.n_steps == 1
    assert session.status().attempts == 2
    assert kept.tobytes() == _frame(0).field.tobytes()
    assert progress_event.history.z.tolist() == [0.1]
    session.acknowledge(plane.receipt)
    terminal = session.advance()
    (outcome,) = terminal.events
    assert isinstance(outcome, Terminal)
    assert outcome.outcome is Outcome.COMPLETED
    assert session.finished and outcome.ticket is None
    session.acknowledge(terminal.receipt)
    with pytest.raises(SessionError, match="ended"):
        session.advance()


@pytest.mark.parametrize("change", ["receipt", "generation", "position", "member"])
def test_invalid_tickets_refuse_before_export_writes(boundary, change):
    session = _start(boundary, _frame(0))
    session.advance()
    ticket = session.pending_ticket()
    assert ticket is not None
    if change == "receipt":
        ticket = replace(ticket, receipt=replace(ticket.receipt, session=-1))
    elif change == "generation":
        ticket = replace(ticket, generation=ticket.generation + 1)
    elif change == "position":
        ticket = replace(ticket, z=0.5)
    else:
        ticket = replace(ticket, member=0)
    destination = np.full(SHAPE, 9.0 + 2.0j, dtype=np.complex128)
    with pytest.raises(SessionError):
        session.export(ticket, IDENTITY, out=destination)
    assert np.all(destination == 9.0 + 2.0j)
    assert boundary.engine.exports == []


def test_invalid_destination_is_checked_before_a_cached_export(boundary):
    session = _start(boundary, _frame(0))
    session.advance()
    ticket = session.pending_ticket()
    assert ticket is not None
    session.export(ticket, IDENTITY)
    wrong = np.full(SHAPE, 9.0)
    with pytest.raises(BackendError, match="complex128"):
        session.export(ticket, IDENTITY, out=wrong)
    assert np.all(wrong == 9.0)
    assert len(boundary.engine.exports) == 1


@pytest.mark.parametrize("where", ["listener", "translation", "attempt_copy"])
def test_failed_delivery_keeps_exactly_one_undelivered_chunk(
    boundary, monkeypatch, where
):
    def broken_listener(*unused):
        raise RuntimeError("consumer failed")

    session = _start(
        boundary,
        _frame(0),
        _frame(3, z=0.1, attempts=1),
        listener=broken_listener if where == "listener" else None,
    )
    _launch(session)
    with monkeypatch.context() as injected:
        if where == "translation":
            injected.setattr(JuliaSession, "_event", broken_listener)
        elif where == "attempt_copy":
            injected.setattr(_calls, "attempts", broken_listener)
        with pytest.raises(RuntimeError, match="consumer failed"):
            session.advance()
    ticket = session.failure_ticket()
    assert ticket is not None and ticket.z == 0.1
    assert session.status().attempts == 1
    history, rows = session.undelivered(ticket)
    assert history.z.tolist() == [0.1]
    assert history.accepted.tolist() == [True]
    assert not history.z.flags.writeable
    assert rows.z.size == 0
    assert session.undelivered(ticket)[0].n_steps == 0
    assert len(boundary.engine.advances) == 2
    session.acknowledge(ticket.receipt)


def test_returned_chunk_is_not_delivered_again_for_a_service_failure(boundary):
    session = _start(boundary, _frame(0), _frame(3, z=0.1, attempts=1))
    _launch(session)
    batch = session.advance()
    (event,) = batch.events
    assert isinstance(event, Progress) and event.history.n_steps == 1
    ticket = session.failure_ticket()
    assert ticket is not None
    assert session.undelivered(ticket)[0].n_steps == 0
    session.acknowledge(batch.receipt)
    ticket = session.failure_ticket()
    assert ticket is not None and ticket.generation == 3
    assert session.finished
    assert session.undelivered(ticket)[0].n_steps == 0
    assert len(boundary.engine.advances) == 2


@pytest.mark.parametrize("failed_copy", [False, True])
def test_observation_chunks_retry_atomically_and_remain_owned(
    boundary, monkeypatch, failed_copy
):
    monkeypatch.setattr(_calls, "memory", lambda *args, **kwargs: {})
    case = cpu_case()
    spec = StatsSpec(names=("energy", "peak_power"), periodic=("peak_power", "energy"))
    request = replace(
        case.request, observation=replace(case.request.observation, stats=spec)
    )
    transfer = lower_observations(case.problem, request)
    session = _start(boundary, _frame(0), _frame(3, z=0.1, attempts=1))
    _launch(session)
    session._observation = transfer
    copies = []

    def copy_observations(positions, steps, values, plane, engine, generation):
        assert engine.pending and generation == engine.frame.report.generation
        copies.append(generation)
        positions[:] = 0.1
        steps[:] = 0
        values[:] = [11.0, 22.0]
        assert plane.size == 0
        if failed_copy and len(copies) == 1:
            raise RuntimeError("interrupted observation copy")

    boundary.engine.module.session_observation_status = lambda engine, generation: (
        1,
        2,
        0,
    )
    setattr(boundary.engine.module, "copy_session_observations!", copy_observations)
    if failed_copy:
        with pytest.raises(BackendError, match="observation copy"):
            session.advance()
        assert session._payload is None
        ticket = session.pending_ticket()
        assert ticket is not None
        attempts, rows = session.undelivered(ticket)
        assert attempts.n_steps == 1
        assert session.undelivered(ticket)[1].z.size == 0
        receipt = ticket.receipt
    else:
        batch = session.advance()
        (event,) = batch.events
        assert isinstance(event, Progress)
        rows = event.step_stats
        receipt = batch.receipt
    assert rows is not None
    assert tuple(rows.values) == ("peak_power", "energy")
    assert rows.values["peak_power"].shape == (1, 1)
    assert rows.values["energy"].shape == (1, 1, 1)
    np.testing.assert_array_equal(rows.values["peak_power"], [[11.0]])
    np.testing.assert_array_equal(rows.values["energy"], [[[22.0]]])
    assert not rows.z.flags.writeable and not rows.values["energy"].flags.writeable
    assert (
        sum(
            value
            for name, value in session.memory_ledger().items()
            if name.startswith("observation.")
        )
        == 32
    )
    session.acknowledge(receipt)
    assert session._payload is None
    assert len(copies) == (2 if failed_copy else 1)
    np.testing.assert_array_equal(rows.values["peak_power"], [[11.0]])


def test_snapshot_coalescing_and_close_release_owned_resources(boundary):
    token = StopToken()
    session = _start(
        boundary,
        _frame(0),
        _frame(1, z=0.1, attempts=1),
        _frame(4, z=0.1, attempts=1),
        stop=token,
    )
    first = session.request_snapshot()
    second = session.request_snapshot()
    assert first.request == second.request == 1
    assert not first.coalesced and second.coalesced
    _launch(session)
    plane = session.advance()
    session.acknowledge(plane.receipt)
    snapshot = session.advance()
    (event,) = snapshot.events
    assert isinstance(event, ServiceRequired)
    assert event.kind is ServiceKind.SNAPSHOT and event.ticket is not None
    assert session.march_snapshot(event.ticket) is None
    assert session.noise_phase(event.ticket) is None
    token.set()
    assert len(boundary.stopped) == 1
    session.close()
    session.close()
    assert session.closed and session.status().state is SessionState.CLOSED
    assert session.pending_ticket() is None and session.failure_ticket() is None
    assert len(boundary.engine.closed) == 1
    token.set()
    assert len(boundary.stopped) == 1
    with pytest.raises(SessionError, match="closed"):
        session.export(event.ticket, IDENTITY)
    with pytest.raises(SessionError, match="closed"):
        session.request_snapshot()


@pytest.mark.parametrize("callback", ["started", "listener"])
@pytest.mark.parametrize("operation", ["advance", "close", "owner_close"])
def test_callbacks_cannot_reenter_session_or_close_its_owner(
    boundary, callback, operation
):
    refused = []

    def attempt(*unused):
        action = (
            boundary.owner.close
            if operation == "owner_close"
            else getattr(session, operation)
        )
        with pytest.raises(SessionError, match="recursive"):
            action()
        refused.append(operation)
        assert not boundary.owner.closed and not session.closed
        assert boundary.preparation.borrow() is not None

    session = _start(
        boundary, _frame(0), _frame(3, z=0.1, attempts=1), **{callback: attempt}
    )
    _launch(session)
    batch = session.advance(work_limit=12, deadline=0.5)
    assert refused == [operation]
    assert len(boundary.engine.advances) == 2
    assert boundary.engine.advances[-1] == (1 if callback == "listener" else 12, 0.5)
    session.acknowledge(batch.receipt)


@pytest.mark.parametrize("callback", ["started", "listener"])
def test_external_callbacks_can_latch_a_snapshot(boundary, callback):
    requests = []

    def observe(*unused):
        requests.extend((session.request_snapshot(), session.request_snapshot()))

    session = _start(
        boundary, _frame(0), _frame(3, z=0.1, attempts=1), **{callback: observe}
    )
    _launch(session)
    batch = session.advance()
    assert len(requests) == 2
    assert requests[0].request == requests[1].request
    assert not requests[0].coalesced and requests[1].coalesced
    assert not session._servicing
    session.acknowledge(batch.receipt)


def test_interrupted_acknowledgment_keeps_both_ledgers_consistent(boundary):
    session = _start(boundary, _frame(0), _frame(3, z=0.1, attempts=1))
    launch = session.advance()
    ticket = session.pending_ticket()
    assert ticket is not None
    cached = session.export(ticket, IDENTITY)
    boundary.engine.after_ack = lambda: _interrupts._handle(signal.SIGINT, None)
    with pytest.raises(KeyboardInterrupt):
        session.acknowledge(launch.receipt)
    boundary.engine.after_ack = None
    assert not boundary.engine.sessions[0].pending
    assert session.status().state is SessionState.READY
    assert session.pending_ticket() is None
    with pytest.raises(SessionError, match="expired"):
        session.export(ticket, IDENTITY)
    next_batch = session.advance()
    assert next_batch.receipt.sequence == launch.receipt.sequence + 1
    assert cached.tobytes() == _frame(0).field.tobytes()
    session.acknowledge(next_batch.receipt)


@pytest.mark.parametrize("kind", [5, 6, 7])
def test_terminal_reports_preserve_outcome_and_ticket_eligibility(boundary, kind):
    session = _start(boundary, _frame(0), _frame(kind, z=0.1, attempts=1))
    _launch(session)
    batch = session.advance()
    (event,) = batch.events
    assert isinstance(event, Terminal)
    assert (
        event.outcome
        is {5: Outcome.COMPLETED, 6: Outcome.STOPPED, 7: Outcome.FAILED}[kind]
    )
    assert event.eligibility is Eligibility.ELIGIBLE
    if kind == 5:
        assert event.ticket is None and event.error is None
    else:
        assert event.ticket is not None
        assert isinstance(event.error, StopRequested if kind == 6 else ConvergenceError)
    assert session.finished
    with pytest.raises(SessionError, match="finished"):
        session.request_snapshot()
    session.acknowledge(batch.receipt)


def test_poisoned_failure_never_exports_the_abandoned_field(boundary):
    session = _start(
        boundary,
        _frame(0),
        _frame(7, z=0.1, attempts=1, poisoned=True, code="OverflowError"),
    )
    _launch(session)
    batch = session.advance()
    (event,) = batch.events
    assert isinstance(event, Terminal)
    assert event.eligibility is Eligibility.POISONED and event.ticket is None
    assert isinstance(event.error, BackendError)
    assert session.in_flight == (0.1, 0.1)
    assert session.pending_ticket() is None and session.failure_ticket() is None
    session.acknowledge(batch.receipt)


def test_failed_engine_construction_keeps_start_and_detaches_stop(boundary):
    token = StopToken()
    start = Start(np.ones(SHAPE, dtype=np.complex128), 0.0)
    original = start.field
    boundary.engine.construct_error = RuntimeError("scripted construction failure")
    with pytest.raises(BackendError, match="scripted construction failure"):
        boundary.prepared.start(start, stop=token, owns=False)
    assert start.field is original and not boundary.owner.closed
    assert boundary.engine.sessions == []
    token.set()
    assert boundary.stopped == []


@pytest.mark.parametrize("interrupted", [False, True])
def test_failed_history_restore_releases_engine_without_consuming_start(
    boundary, monkeypatch, interrupted
):
    token = StopToken()
    field = np.ones(SHAPE, dtype=np.complex128)
    snapshot = MarchSnapshot(
        z=0.0,
        state=field.copy(),
        proposal=0.1,
        method="test",
        order=1,
        rejections=0,
        last_rejected=False,
        positions=np.array([0.0]),
        identities=("accepted",),
        slots=np.array([0], dtype=np.int64),
        rows={"store": np.zeros((3, 1, *SHAPE), dtype=np.complex128)},
    )
    start = Start(field, 0.0, march=snapshot)

    def failure(*_):
        if interrupted:
            raise KeyboardInterrupt
        raise BackendError("scripted history copy failure")

    monkeypatch.setattr(session_module, "restore_history", failure)
    with pytest.raises(KeyboardInterrupt if interrupted else BackendError):
        boundary.prepared.start(start, stop=token, owns=False)
    assert start.field is field and not boundary.owner.closed
    assert len(boundary.engine.sessions) == 1
    assert boundary.engine.sessions[0].closed
    assert not boundary.owner._sessions
    token.set()
    assert boundary.stopped == []


def test_engine_close_failure_still_releases_handles_and_stop_listener(boundary):
    token = StopToken()
    session = _start(boundary, _frame(0), stop=token)
    boundary.engine.close_error = RuntimeError("scripted close failure")
    with pytest.raises(BackendError, match="scripted close failure"):
        session.close()
    assert session.closed
    assert not boundary.owner._sessions
    assert not boundary.owner.closed
    with pytest.raises(SessionError, match="closed"):
        session._engine.borrow()
    token.set()
    assert boundary.stopped == []
    session.close()


def test_invalid_advance_bounds_refuse_before_announcement_or_engine_entry(boundary):
    announced = []
    session = _start(boundary, _frame(0), started=lambda *args: announced.append(args))
    for keywords in (
        {"work_limit": 0},
        {"work_limit": True},
        {"deadline": -1.0},
        {"deadline": np.inf},
    ):
        with pytest.raises(BackendError):
            session.advance(**keywords)
    assert not announced and not boundary.engine.advances


@pytest.mark.parametrize("fail_copy", [False, True])
def test_execution_orders_survive_failed_copy_and_encode_each_chunk_once(
    boundary, monkeypatch, fail_copy
):
    def counters(session, generation):
        assert session.pending and session.generation == generation
        count = session.frame.report.attempts
        startup = int(count > 0)
        return startup + count, (startup, count, 0, 0, count, 0, 0, 0, 0), None

    failed = False

    def orders(output, session, generation):
        nonlocal failed
        assert session.pending and session.generation == generation
        if fail_copy and output.size and not failed:
            failed = True
            raise RuntimeError("injected order-copy failure")
        output.fill(2)

    monkeypatch.setattr(boundary.engine.module, "session_execution", counters)
    monkeypatch.setattr(
        boundary.engine.module, "session_orders!", orders, raising=False
    )
    monkeypatch.setattr(_calls, "memory", lambda *_: {})
    session = _start(
        boundary, _frame(0), _frame(3, z=0.1, attempts=1), _frame(1, z=1.0, attempts=2)
    )
    session.acknowledge(session.advance().receipt)
    if fail_copy:
        with pytest.raises(BackendError, match="accounting read failed"):
            session.advance()
        ticket = session.pending_ticket()
        assert ticket is not None
        history, rows = session.undelivered(ticket)
        assert history.n_steps == 1 and rows.z.size == 0
        assert session.undelivered(ticket)[0].n_steps == 0
        receipt = ticket.receipt
    else:
        receipt = session.advance().receipt
    first = session.execution_record()
    assert first["source_evaluations"] == 2
    assert first["response_calls"]["orders"] == [[2, 1]]
    first["response_calls"]["orders"][0][1] = 99
    assert session.execution_record()["response_calls"]["orders"] == [[2, 1]]
    session.acknowledge(receipt)
    plane = session.advance()
    expected = session.execution_record()
    assert expected["source_evaluations"] == 3
    assert expected["response_calls"]["accepted_steps"] == 2
    assert expected["response_calls"]["orders"] == [[2, 2]]
    assert (
        session.memory_ledger()["execution.response_orders"]
        == 2 * np.dtype(np.int64).itemsize
    )
    session.acknowledge(plane.receipt)
    assert session.execution_record() == expected
    session.close()
    assert session.execution_record() == expected


@pytest.mark.parametrize("interrupt_at", ["before", "after"])
def test_accounting_publication_is_atomic_under_interruption(
    boundary, monkeypatch, interrupt_at
):
    import inspect  # noqa: PLC0415 - trace the one publication boundary precisely
    import sys  # noqa: PLC0415 - isolated synthetic interruption, no runtime signals

    session = _start(boundary, _frame(0))
    accounting = _calls.EngineExecution(
        2, (1, 1, 0, 0, 1, 0, 0, 0, 0), np.array([2], dtype=np.int64)
    )
    monkeypatch.setattr(_calls, "execution", lambda *_: accounting)
    capture = inspect.unwrap(JuliaSession._capture_execution)
    lines, first = inspect.getsourcelines(capture)
    publication = first + next(
        index for index, line in enumerate(lines) if "self._accounting =" in line
    )
    interrupted = False

    def trace(frame, event, arg):
        nonlocal interrupted
        before = event == "line" and frame.f_lineno == publication
        after = event == "return"
        if frame.f_code is capture.__code__ and (
            before if interrupt_at == "before" else after
        ):
            interrupted = True
            raise KeyboardInterrupt
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt):
            session._capture_execution(1)
    finally:
        sys.settrace(previous)
    assert interrupted
    assert session._accounting.generation == (0 if interrupt_at == "before" else 1)
    session._capture_execution(1)
    session._capture_execution(1)
    record = session.execution_record()
    assert record["source_evaluations"] == 2
    assert record["response_calls"]["orders"] == [[2, 1]]
    assert record["response_calls"]["accepted_steps"] == 1
