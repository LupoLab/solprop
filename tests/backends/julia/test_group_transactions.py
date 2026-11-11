# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise group publication and adoption with a scripted nonnumerical core.

The core supplies immutable event facts and already-computed attempt arrays. It
does not integrate fields or reproduce Julia's scheduling decisions. These tests
exercise the actual Python group adapter and its shared interrupt/receipt logic.
"""

import signal
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from weakref import WeakSet

import numpy as np
import pytest

from backends.julia.cases import cpu_case
from solprop import _interrupts
from solprop.backends.events import Eligibility, Event, Outcome, Progress, Terminal
from solprop.backends.julia import _calls, _group, _group_calls
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.julia._session import JuliaSession
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.protocol import BackendHistory, StepStats
from solprop.backends.session import SessionError, SessionLedger, Start, StopToken
from solprop.exceptions import BackendError


def _history() -> BackendHistory:
    return BackendHistory(
        np.array([0.125]),
        np.array([0.125]),
        np.array([0.25]),
        np.array([True]),
        np.array([0.001]),
    )


class _Member:
    """Translate scripted events without numerical stepping or accounting policy."""

    def __init__(self) -> None:
        self._busy = False
        self.deliveries: list[tuple[int, int]] = []

    def _group_event(
        self,
        report: _calls.EngineReport,
        chunk: int,
        *,
        error: BaseException | None = None,
    ) -> Event:
        self.deliveries.append((report.generation, chunk))
        if report.kind == 3:
            return Progress(report.z, _history(), StepStats.empty())
        return Terminal(
            {5: Outcome.COMPLETED, 6: Outcome.STOPPED, 7: Outcome.FAILED}[report.kind],
            report.z,
            Eligibility.ELIGIBLE,
            error,
            None,
            _history(),
            StepStats.empty(),
        )


class _Core:
    """Hold one numerical transport and retain its chunk on terminal replacement."""

    def __init__(self, *, terminal: bool) -> None:
        self.pending = False
        self.advance_calls = 0
        self.acknowledgments = 0
        self.failure_calls = 0
        self.after_ack: Callable[[], None] | None = None
        self.report = _group_calls.GroupReport(
            1, 3 if terminal else 0, 0, int(terminal), -1
        )
        self.member = _calls.EngineReport(
            5 if terminal else 3,
            0.125,
            0,
            1,
            1,
            0,
            1,
            0,
            0.25,
            0.25,
            0.001,
            True,
            False,
        )

    def advance(self, *_: object) -> _group_calls.GroupReport:
        if self.pending:
            raise BackendError("scripted core refuses advancing a pending transport")
        self.advance_calls += 1
        self.pending = True
        return self.report

    def read(self, *_: object) -> _group_calls.GroupReport:
        if not self.pending:
            raise BackendError(
                "scripted core refuses reading an acknowledged transport"
            )
        return self.report

    def members(self, *_: object) -> tuple[_group_calls.MemberReport, ...]:
        self.read()
        return (_group_calls.MemberReport(True, True, True, 1, self.member),)

    def fail(self, *_: object) -> _group_calls.GroupReport:
        self.failure_calls += 1
        if self.report.kind == 3:
            return self.read()
        generation = self.report.generation + int(not self.pending)
        self.pending = True
        self.report = replace(self.report, generation=generation, kind=3, outcome=3)
        self.member = replace(
            self.member, kind=7, generation=self.member.generation + 1
        )
        return self.report

    def acknowledge(self, *_: object) -> None:
        if not self.pending:
            raise BackendError("scripted core refuses repeated acknowledgment")
        self.pending = False
        self.acknowledgments += 1
        callback, self.after_ack = self.after_ack, None
        if callback is not None:
            callback()


@dataclass
class _Harness:
    group: _group.JuliaGroupSession
    core: _Core
    member: _Member


@pytest.fixture
def scripted_group(monkeypatch):
    monkeypatch.setattr(_interrupts, "_GUARD", _interrupts._Guard())

    def make(*, terminal=False):
        core, member = _Core(terminal=terminal), _Member()
        group = object.__new__(_group.JuliaGroupSession)
        for name, value in {
            "_owners": (),
            "_ledger": SessionLedger(),
            "_busy": False,
            "_stop": None,
            "_token": StopToken(),
            "_report": None,
            "_batch": None,
            "_tickets": {},
            "_chunks": [0],
            "_events": [0],
            "_outcomes": [None],
            "_errors": {},
            "_primary": None,
            "_members": (member,),
            "_runtime": object(),
            "_engine": object(),
            "_windows": None,
            "_recovering": False,
        }.items():
            monkeypatch.setattr(group, name, value, raising=False)
        for name, operation in (
            ("advance", core.advance),
            ("report", core.read),
            ("members", core.members),
            ("fail", core.fail),
        ):
            monkeypatch.setattr(_group_calls, name, operation)
        monkeypatch.setattr(_calls, "call_void", core.acknowledge)
        return _Harness(group, core, member)

    return make


def _assert_one_chunk(batch) -> None:
    (event,) = batch.members[0]
    assert isinstance(event, Terminal) and event.history is not None
    expected = _history()
    for name in ("z", "dz", "error_norm", "accepted", "wall_time"):
        # This is lossless transport of the very same precomputed input arrays.
        np.testing.assert_array_equal(
            getattr(event.history, name), getattr(expected, name)
        )


def test_deferred_interrupt_after_internal_ack_preserves_work_and_primary(
    scripted_group,
):
    harness = scripted_group()
    group, core = harness.group, harness.core
    core.after_ack = lambda: _interrupts._handle(signal.SIGINT, None)
    with pytest.raises(KeyboardInterrupt) as raised:
        group.advance()
    assert not group._busy and not harness.member._busy
    batch = group.advance()
    assert batch.outcome is Outcome.FAILED and batch.error is raised.value
    assert core.advance_calls == 1 and core.acknowledgments == 1
    _assert_one_chunk(batch)
    with pytest.raises(SessionError, match="acknowledge"):
        group.advance()
    group.acknowledge(batch.receipt)
    assert group.finished and not core.pending
    with pytest.raises(SessionError, match="ended"):
        group.advance()


@pytest.mark.parametrize("failures", [1, 2, 4])
def test_terminal_publication_retries_without_duplicate_work(
    scripted_group, monkeypatch, failures
):
    harness = scripted_group(terminal=True)
    group, core = harness.group, harness.core
    real_join = _group.join_attempts
    errors = [
        MemoryError(f"publication allocation {index}") for index in range(failures)
    ]
    calls = 0

    def fail_then_join(
        chunks: Iterable[tuple[BackendHistory | None, StepStats | None]],
    ) -> tuple[BackendHistory, StepStats]:
        nonlocal calls
        calls += 1
        if calls <= failures:
            raise errors[calls - 1]
        return real_join(chunks)

    monkeypatch.setattr(_group, "join_attempts", fail_then_join)
    for _ in range(failures // 2):
        with pytest.raises(MemoryError):
            group.advance()
        assert group._ledger.pending is None and not group.finished
        assert group._batch is None and group._windows is not None
        assert not group._busy and not harness.member._busy
    batch = group.advance()
    assert batch.outcome is Outcome.FAILED and batch.error is errors[0]
    assert core.advance_calls == 1
    assert harness.member.deliveries == [(1, 1)]
    _assert_one_chunk(batch)
    assert group._windows is None and not group._recovering
    group.acknowledge(batch.receipt)
    assert group.finished and not core.pending


def test_stopped_publication_retry_retains_one_shared_primary(
    scripted_group, monkeypatch
):
    harness = scripted_group(terminal=True)
    harness.core.report = replace(harness.core.report, outcome=2)
    harness.core.member = replace(harness.core.member, kind=6)
    real_join = _group.join_attempts
    calls = 0

    def fail_once(chunks):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MemoryError("first stopped publication")
        return real_join(chunks)

    monkeypatch.setattr(_group, "join_attempts", fail_once)
    batch = harness.group.advance()
    (event,) = batch.members[0]
    assert isinstance(event, Terminal) and event.outcome is Outcome.STOPPED
    assert batch.outcome is Outcome.STOPPED and event.error is batch.error
    _assert_one_chunk(batch)
    harness.group.acknowledge(batch.receipt)


class _Resource:
    def __init__(self, name: str, released: list[str]) -> None:
        self.name, self.released = name, released

    def close(self) -> None:
        self.released.append(self.name)


@pytest.mark.parametrize("interrupted", [0, 1])
def test_constructor_interrupt_restores_original_starts_and_releases_partial_group(
    monkeypatch, interrupted
):
    case = cpu_case()
    starts = [Start.entrance(case.problem), Start.entrance(case.problem)]
    fields = tuple(start.field for start in starts)
    owners = [object.__new__(JuliaBackend), object.__new__(JuliaBackend)]
    runtime = object.__new__(JuliaRuntime)
    released: list[str] = []
    for owner in owners:
        owner._groups = WeakSet()
        owner._closed = False
    engine = _Resource("engine", released)
    monkeypatch.setattr(runtime, "own_call", lambda *args, **kwargs: engine)

    class Lease(_Resource):
        def __init__(self, *_: object) -> None:
            super().__init__("stop lease", released)
            self.flag = object()

    class Prepared:
        def __init__(self, index: int) -> None:
            self.index = index

        def start(self, start: Start, *, owns: bool) -> JuliaSession:
            assert not owns and all(start is not original for original in starts)
            start.take_field()
            session = object.__new__(JuliaSession)
            session._runtime = runtime
            monkeypatch.setattr(session, "_engine", object(), raising=False)
            resource = _Resource(f"member {self.index}", released)
            monkeypatch.setattr(session, "close", resource.close)
            return session

    for index, owner in enumerate(owners):
        prepared = Prepared(index)
        monkeypatch.setattr(owner, "prepare", lambda *args, made=prepared: made)
    # Numerical/request admission is exercised by the real group tests. This
    # fixture starts at the already-admitted ownership transaction itself.
    monkeypatch.setattr(_group, "check_compatible", lambda *args: None)
    monkeypatch.setattr(_group, "StopLease", Lease)
    monkeypatch.setattr(_interrupts, "_GUARD", _interrupts._Guard())
    take = Start.take_field

    def interrupted_take(start: Start) -> np.ndarray:
        value = take(start)
        if start is starts[interrupted]:
            _interrupts._handle(signal.SIGINT, None)
        return value

    with monkeypatch.context() as patch:
        patch.setattr(Start, "take_field", interrupted_take)
        with pytest.raises(KeyboardInterrupt):
            _group.JuliaGroupSession(
                owners, [case.problem] * 2, [case.request] * 2, starts
            )
    assert all(
        start.field is value for start, value in zip(starts, fields, strict=True)
    )
    assert all(not owner.closed and not owner._groups for owner in owners)
    assert released == ["engine", "stop lease", "member 1", "member 0"]

    # The same starts and caller-owned owners remain usable after rollback.
    retried = _group.JuliaGroupSession(
        owners, [case.problem] * 2, [case.request] * 2, starts
    )
    assert all(start.field is None for start in starts)
    assert all(retried in owner._groups for owner in owners)
    for owner in owners:
        monkeypatch.setattr(
            owner, "close", lambda closing=owner: setattr(closing, "_closed", True)
        )
    monkeypatch.setattr(_calls, "call_void", lambda *args: None)
    retried.close()
    assert all(owner.closed and not owner._groups for owner in owners)
