# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the common benchmark boundary without inventing numerical stepping."""

import importlib.util
import subprocess
import sys
import threading
import weakref

import numpy as np
import pytest
from benchmarks import worker
from benchmarks.builders import build
from benchmarks.records import SuiteError
from benchmarks.registry import CASES

from solprop.backends import select
from solprop.backends.protocol import Backend, ExecutionBackend
from solprop.backends.session import PreparedExecution, RunSession, Start
from solprop.driver import run as driver_run
from solprop.output.schema import FieldRetention


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def spend(self, nanoseconds):
        self.now += nanoseconds


class Session:
    """A slotted lifecycle double; its work is explicitly scripted by the caller."""

    __slots__ = (
        "clock",
        "close_calls",
        "close_error",
        "closed",
        "entered",
        "exited",
        "finished",
        "march_refusal",
        "record",
        "record_error",
        "records_read",
    )

    def __init__(self, clock):
        self.clock = clock
        self.record = {"source_evaluations": 3}
        self.records_read = []
        self.closed = self.finished = False
        self.close_calls = 0
        self.close_error = self.record_error = None
        self.entered = self.exited = False
        self.march_refusal = "scripted optional extension"

    def execution_record(self):
        assert not self.closed, "the record must be read before release"
        self.clock.spend(5)
        if self.record_error is not None:
            raise self.record_error
        self.records_read.append(dict(self.record))
        return self.record

    def close(self):
        self.clock.spend(40)
        self.close_calls += 1
        self.closed = True
        # Simulate resources whose execution metadata disappears on close.
        self.record.clear()
        if self.close_error is not None:
            raise self.close_error

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_):
        self.exited = True
        self.close()

    def _unexpected(self, *_args, **_keywords):
        raise AssertionError("the benchmark must not drive extra numerical work")

    status = advance = acknowledge = export = noise_phase = _unexpected
    request_snapshot = march_snapshot = march_snapshot_nbytes = _unexpected
    march_export_refusal = monitor_record = continue_monitor = _unexpected
    undelivered = pending_ticket = _unexpected


class Prepared:
    __slots__ = ("closed", "owner", "starts")

    def __init__(self, owner):
        self.owner = owner
        self.starts = []
        self.closed = False

    @property
    def backend(self):
        return self.owner.name

    def start(self, start, **keywords):
        self.starts.append((start, keywords))
        self.owner.clock.spend(30)
        if self.owner.start_error is not None:
            raise self.owner.start_error
        return self.owner.session

    def metadata(self):
        return self.owner.metadata()

    def memory_ledger(self):
        return {"prepared": 8}

    def close(self):
        self.closed = True


class Owner:
    __slots__ = (
        "clock",
        "closed",
        "factory",
        "preparations",
        "problem",
        "ready",
        "request",
        "session",
        "start_error",
    )
    name = "session_only"

    def __init__(self, clock, problem, request):
        self.clock = clock
        self.problem = problem
        self.session = Session(clock)
        self.factory = Prepared(self)
        self.request = request
        self.preparations = []
        self.start_error = None
        self.ready = self.closed = False

    def prepare(self, problem, request):
        self.clock.spend(20)
        self.preparations.append((problem, request))
        self.ready = True
        return self.factory

    def metadata(self):
        # An owner counter must never replace the returned session's counter.
        record = {"source_evaluations": 1000, "backend": self.name}
        if self.ready:
            record.update(setup_id="verified", provider="chosen", julia_threads=2)
        return record

    def close(self):
        self.closed = True

    def _unexpected(self, *_args, **_keywords):
        raise AssertionError("the benchmark must only prepare and start its owner")

    march_export_refusal = edge_memory = timing_probe = _unexpected
    available_device_memory = memory_ledger = start_group = _unexpected


@pytest.fixture
def scripted(monkeypatch, case):
    clock = Clock()
    monkeypatch.setattr(worker.time, "perf_counter_ns", clock)
    simulation, problem = case
    owner = Owner(clock, problem, simulation._run_request())
    timing = worker._Started(owner)
    return clock, owner, timing


def _start(timing):
    owner = timing.engine._target
    prepared = timing.engine.prepare(owner.problem, owner.request)
    assert isinstance(prepared, PreparedExecution)
    return prepared.start(Start.entrance(owner.problem), owns=False)


def test_proxy_preserves_slotted_ownership_and_exact_start_arguments(scripted):
    clock, owner, timing = scripted
    assert not isinstance(owner, Backend)
    assert isinstance(timing.engine, ExecutionBackend)
    assert timing.engine.request is owner.request
    problem, request = owner.problem, owner.request
    start = Start.entrance(problem)
    prepared = timing.engine.prepare(problem, request)
    assert prepared.backend == owner.name
    assert prepared.memory_ledger() == {"prepared": 8}

    def listener(*_):
        pass

    def started(*_):
        pass

    stop = threading.Event()
    session = prepared.start(
        start, listener=listener, started=started, stop=stop, owns=False
    )
    assert isinstance(session, RunSession)
    assert isinstance(session, Session)
    with pytest.raises(SuiteError, match="target's class"):
        session.__class__ = type(session)
    assert owner.preparations == [(problem, request)]
    assert owner.factory.starts == [
        (start, {"listener": listener, "started": started, "stop": stop, "owns": False})
    ]
    assert timing.started_at == 50
    assert clock.now == 55  # First metadata read is inside the propagation window.
    extension = "march_refusal"
    assert getattr(session, extension) == owner.session.march_refusal
    owner.session.finished = True
    assert session.finished  # Property forwarding remains live.
    owner.session.record["source_evaluations"] = 11
    session.close()
    assert timing.evaluations() == 8
    assert timing.execution["source_evaluations"] == 11
    assert len(owner.session.records_read) == 2
    assert owner.session.closed
    assert not owner.closed and not owner.factory.closed
    session.close()  # Idempotent release does not reread a destroyed record.
    assert owner.session.close_calls == 2
    prepared.close()
    assert owner.factory.closed and not owner.closed
    timing.engine.close()
    assert owner.closed


@pytest.mark.parametrize(
    ("initial", "final", "key", "expected"),
    [
        (
            {"source_evaluations": 5},
            {"source_evaluations": 12},
            "source_evaluations",
            7,
        ),
        (
            {"attempted_source_evaluations": 5, "source_evaluations": 4},
            {"attempted_source_evaluations": 15, "source_evaluations": 12},
            "attempted_source_evaluations",
            10,
        ),
        (
            {"response_calls": {"startup": 1, "predictor": 2, "corrected": 0}},
            {"response_calls": {"startup": 1, "predictor": 8, "corrected": 0}},
            None,
            None,
        ),
        ({}, {"source_evaluations": 9}, None, None),
        (
            {"source_evaluations": 3},
            {"attempted_source_evaluations": 10},
            "source_evaluations",
            None,
        ),
    ],
)
def test_counter_reads_keep_the_same_actual_source(
    scripted, initial, final, key, expected
):
    _, owner, timing = scripted
    owner.session.record = initial
    session = _start(timing)
    owner.session.record = final
    session.close()
    assert timing.counter == key
    assert timing.evaluations() == expected


@pytest.mark.parametrize("value", [True, -1, 1.5, "3"])
def test_malformed_initial_counter_fails_but_releases_the_returned_session(
    scripted, value
):
    _, owner, timing = scripted
    owner.session.record["source_evaluations"] = value
    with pytest.raises(SuiteError, match="invalid session source counter"):
        _start(timing)
    assert owner.session.closed and owner.session.close_calls == 1
    assert not owner.closed


def test_counter_reset_cannot_be_reported_as_negative_work(scripted):
    _, owner, timing = scripted
    session = _start(timing)
    owner.session.record["source_evaluations"] = 2
    session.close()
    with pytest.raises(SuiteError, match="decreased"):
        timing.evaluations()


@pytest.mark.parametrize("failure", [RuntimeError("propagation"), KeyboardInterrupt()])
def test_original_error_survives_record_and_cleanup_errors(scripted, failure):
    _, owner, timing = scripted
    session = _start(timing)
    owner.session.record_error = ValueError("record unavailable")
    owner.session.close_error = LookupError("cleanup failed")
    with pytest.raises(type(failure)) as found:
        try:
            raise failure
        finally:
            session.close()
    assert found.value is failure
    assert any("record unavailable" in note for note in failure.__notes__)
    assert any("cleanup failed" in note for note in failure.__notes__)
    assert owner.session.close_calls == 1
    assert not owner.closed


def test_final_record_error_fails_a_successful_run_after_cleanup(scripted):
    _, owner, timing = scripted
    session = _start(timing)
    failure = ValueError("record unavailable")
    owner.session.record_error = failure
    with pytest.raises(ValueError) as found:
        session.close()
    assert found.value is failure
    assert owner.session.closed and owner.session.close_calls == 1


def test_reused_exception_identity_cannot_loop_while_annotating_cleanup(scripted):
    _, owner, timing = scripted
    session = _start(timing)
    failure = RuntimeError("shared failure")
    owner.session.record_error = failure
    owner.session.close_error = ValueError("release failed")
    with pytest.raises(RuntimeError) as found:
        try:
            raise failure
        finally:
            session.close()
    assert found.value is failure
    assert failure.__notes__ == [
        "session cleanup also failed: ValueError('release failed')"
    ]


def test_close_failure_and_start_failure_retain_original_identity(scripted):
    _, owner, timing = scripted
    failure = RuntimeError("start failed")
    owner.start_error = failure
    with pytest.raises(RuntimeError) as found:
        _start(timing)
    assert found.value is failure
    assert timing.started_at is None and owner.session.close_calls == 0
    owner.start_error = None
    session = _start(timing)
    owner.session.close_error = failure
    with pytest.raises(RuntimeError) as found:
        session.close()
    assert found.value is failure


def test_missing_or_multiple_start_and_missing_close_refuse(scripted):
    _, owner, timing = scripted
    with pytest.raises(SuiteError, match="never returned a session"):
        timing.evaluations()
    session = _start(timing)
    with pytest.raises(SuiteError, match="never closed"):
        timing.evaluations()
    with pytest.raises(SuiteError, match="exactly one session"):
        _start(timing)
    assert len(owner.factory.starts) == 1
    session.close()


def test_proxy_context_lifecycle_still_runs_the_original_hooks(scripted):
    _, owner, timing = scripted
    with _start(timing) as session:
        assert owner.session.entered
        assert not session.closed
    assert owner.session.exited and owner.session.closed
    assert timing.evaluations() == 0


def test_timing_proxy_does_not_create_an_owner_retaining_reference_cycle(scripted):
    _, owner, _ = scripted
    timing = worker._Started(owner)
    reference = weakref.ref(timing)
    proxy = timing.engine
    del timing
    assert reference() is not None  # The live proxy deliberately retains its timer.
    del proxy
    assert reference() is None  # No cyclic collection is needed to release the owner.


@pytest.fixture
def case():
    simulation = build(CASES["smoke"].configuration("default", 1, "numpy_ref"))
    return simulation, simulation.translate()


def test_timed_run_excludes_prepare_and_start_but_keeps_inclusive_wall(
    scripted, case, monkeypatch
):
    clock, owner, _ = scripted
    simulation, problem = case
    values = {
        "step_history/accepted": np.array([True, False]),
        "z": simulation._run_request().observation.save_planes[-1:],
        "step_stats/z": np.empty(0),
    }

    class Output:
        n_saved = 1

        def __getitem__(self, key):
            return values[key]

        def plane_field(self, *_):
            return np.ones(problem.shape, dtype=complex)

        def close(self):
            pass

    def run(problem, engine, _output, *, start, **_):
        session = engine.prepare(problem, owner.request).start(start, owns=False)
        clock.spend(80)
        owner.session.record["source_evaluations"] = 10
        session.close()

    import solprop.output  # noqa: PLC0415

    monkeypatch.setattr(solprop.output, "MemoryOutput", Output)
    monkeypatch.setattr(driver_run, "run_problem", run)
    record, field = worker.timed_run(simulation, problem, owner, FieldRetention("all"))
    # Prepare20 + start30 precede the boundary; two record reads5, work80 and
    # release40 follow it. Clock values are deterministic, not timing tolerances.
    assert record["wall_seconds"] == pytest.approx(180e-9)
    assert record["seconds"] == pytest.approx(130e-9)
    assert record["timing_boundary"] == worker.TIMING_BOUNDARY
    assert (
        record["source_evaluations_before"],
        record["source_evaluations_after"],
    ) == (3, 10)
    assert record["source_evaluations"] == 7
    assert record["accepted"] == record["rejected"] == 1
    assert field is not None and record["finite"]


def test_historical_build_without_common_start_refuses_before_running(
    case, monkeypatch
):
    simulation, problem = case
    owner = Owner(Clock(), problem, simulation._run_request())
    monkeypatch.setattr(worker, "_session_start", lambda: None)
    with pytest.raises(SuiteError, match="historical build"):
        worker.timed_run(simulation, problem, owner, FieldRetention("all"))
    assert not owner.preparations and not owner.session.closed


def test_prepared_runtime_reads_effective_late_metadata(scripted, monkeypatch):
    import threadpoolctl  # noqa: PLC0415

    _, owner, timing = scripted
    assert "setup_id" not in owner.metadata()
    session = _start(timing)
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 2}])
    facts = worker._prepared_runtime(owner)
    assert facts["execution"]["setup_id"] == "verified"
    assert facts["execution"]["provider"] == "chosen"
    assert facts["execution"]["julia_threads"] == 2
    assert facts["libraries"] == [{"num_threads": 2}]
    session.close()


def test_julia_worker_provenance_does_not_initialize_an_execution_runtime():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from benchmarks.worker import runtime; runtime('julia'); "
            "forbidden = ('juliacall', 'juliapkg', 'solprop_native'); "
            "assert not any(name.startswith(forbidden) "
            "for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_proxy_preserves_the_existing_reference_session_facts_path(case):
    from solprop.driver.service import _facts  # noqa: PLC0415

    simulation, problem = case
    request = simulation._run_request()
    engine = select("numpy_ref", problem, request, grid=simulation.grid)
    timing = worker._Started(engine)
    try:
        session = timing.engine.prepare(problem, request).start(
            Start.entrance(problem), owns=False
        )
        try:
            assert _facts(session) is engine
        finally:
            session.close()
    finally:
        engine.close()


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
def test_actual_legacy_and_session_only_owner_share_the_same_window(case, backend):
    if backend == "native" and importlib.util.find_spec("solprop_native") is None:
        pytest.skip("the native backend is not installed in this environment")
    simulation, problem = case
    request = simulation._run_request()

    class SessionOnly:
        """Expose only ownership methods of a real reference engine."""

        def __init__(self, engine):
            self.name, self.request = "session_only", engine.request
            self.prepare, self.close = engine.prepare, engine.close
            self.metadata, self.memory_ledger = engine.metadata, engine.memory_ledger
            self.march_export_refusal = engine.march_export_refusal
            self.edge_memory, self.timing_probe = (
                engine.edge_memory,
                engine.timing_probe,
            )
            self.available_device_memory = engine.available_device_memory
            self.start_group = engine.start_group
            self._engine = engine

        @property
        def closed(self):
            return self._engine.closed

    records, fields = [], []
    for session_only in (False, True):
        engine = select(backend, problem, request, grid=simulation.grid)
        owner = SessionOnly(engine) if session_only else engine
        assert isinstance(owner, ExecutionBackend)
        assert isinstance(owner, Backend) is not session_only
        try:
            record, field = worker.timed_run(
                simulation, problem, owner, FieldRetention("all")
            )
            records.append(record)
            fields.append(field)
            assert not owner.closed
        finally:
            owner.close()
    assert [item["timing_boundary"] for item in records] == [worker.TIMING_BOUNDARY] * 2
    assert records[0]["source_evaluations"] == records[1]["source_evaluations"] > 0
    assert all(item["wall_seconds"] >= item["seconds"] > 0 for item in records)
    assert all(field is not None and np.all(np.isfinite(field)) for field in fields)
    assert records[0]["accepted"] == records[1]["accepted"]
    assert records[0]["rejected"] == records[1]["rejected"]
