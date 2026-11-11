# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dispatch sessions without requiring the legacy numerical operation surface."""

from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from sessions import executors
from sessions.executors import soliton, window
from sessions.test_group import available, constructed, drive, member
from solprop.backends.events import Outcome
from solprop.backends.protocol import Backend, ExecutionBackend
from solprop.backends.registry import select
from solprop.backends.session import SessionError, Start
from solprop.backends.sessions import start_group, start_session
from solprop.exceptions import BackendError
from solprop.stepper import FixedSchedule


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("device", [False, True])
def test_executor_repeatability_is_explicit_and_independent_of_its_name(
    monkeypatch, threads, device
):
    monkeypatch.setattr(executors, "_one_thread", lambda: threads == 1)
    monkeypatch.setattr(executors, "device_build", lambda: device)
    factory = executors.EXECUTORS["reference"].start
    for name in ("reference", "native", "julia-cpu", "renamed"):
        assert executors.Executor(name, factory, repeatability="exact").deterministic
        assert not executors.Executor(
            name, factory, repeatability="rounding"
        ).deterministic
        assert executors.Executor(
            name, factory, repeatability="native"
        ).deterministic is (threads == 1 and not device)


def _owner(driver, *, group=None) -> ExecutionBackend:
    owner = SimpleNamespace(
        name="session-only",
        request=driver.request,
        prepare=driver.prepare,
        start_group=driver.start_group if group is None else group,
        metadata=driver.metadata,
        memory_ledger=driver.memory_ledger,
        close=driver.close,
        closed=driver.closed,
        timing_probe=driver.timing_probe,
        edge_memory=driver.edge_memory,
        available_device_memory=driver.available_device_memory,
        march_export_refusal=driver.march_export_refusal,
    )
    assert isinstance(owner, ExecutionBackend)
    return owner


def test_one_session_needs_no_legacy_state_operations():
    problem, request = soliton()
    driver = select("numpy_ref", problem, request, grid=window())
    owner = _owner(driver)
    assert isinstance(owner, ExecutionBackend)
    assert not isinstance(owner, Backend)
    entrance = Start.entrance(problem)
    try:
        with start_session(owner, problem, request, entrance) as session:
            assert entrance.field is None
            batch = session.advance()
            assert batch.events
            session.acknowledge(batch.receipt)
    finally:
        owner.close()


def test_group_uses_its_owner_factory_without_registry_name_dispatch():
    simulations = [member(index, "numpy_ref") for index in range(2)]
    problems = [simulation.translate() for simulation in simulations]
    requests = [simulation._run_request() for simulation in simulations]
    starts = [Start.entrance(problem) for problem in problems]
    calls = []
    with ExitStack() as cleanup:
        drivers = [constructed(simulation, cleanup) for simulation in simulations]

        def construct(owners, supplied, policies, entrances, *, stop):
            assert all(not isinstance(owner, Backend) for owner in owners)
            assert supplied is problems and policies is requests and entrances is starts
            assert all(entrance.field is not None for entrance in entrances)
            calls.append(tuple(owner.name for owner in owners))
            return drivers[0].start_group(
                drivers, supplied, policies, entrances, stop=stop
            )

        owners = [_owner(driver, group=construct) for driver in drivers]
        group = start_group(owners, problems, requests, starts)
        cleanup.callback(group.close)
        assert calls == [("session-only", "session-only")]
        assert all(entrance.field is None for entrance in starts)
        assert drive(group)[-1].outcome is Outcome.COMPLETED


def test_mixed_owner_names_refuse_before_the_group_takes_any_field():
    simulations = [member(index, "numpy_ref") for index in range(2)]
    problems = [simulation.translate() for simulation in simulations]
    requests = [simulation._run_request() for simulation in simulations]
    starts = [Start.entrance(problem) for problem in problems]
    with ExitStack() as cleanup:
        owners = [
            _owner(constructed(simulation, cleanup)) for simulation in simulations
        ]
        owners[1].name = "other-session-owner"
        with pytest.raises(SessionError, match="mixed backends"):
            start_group(owners, problems, requests, starts)
        assert all(entrance.field is not None for entrance in starts)


@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
def test_owner_factory_refuses_foreign_members_before_consuming_fields(backend):
    available(backend)
    simulations = [member(index, backend) for index in range(2)]
    problems = [simulation.translate() for simulation in simulations]
    requests = [simulation._run_request() for simulation in simulations]
    starts = [Start.entrance(problem) for problem in problems]
    with ExitStack() as cleanup:
        drivers = [constructed(simulation, cleanup) for simulation in simulations]
        owners = [drivers[0], _owner(drivers[1])]
        with pytest.raises(BackendError, match="own member owners"):
            drivers[0].start_group(owners, problems, requests, starts)
        assert all(entrance.field is not None for entrance in starts)


@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
@pytest.mark.parametrize(
    "invalid", ["schedule", "request", "used", "same_start", "same_owner"]
)
def test_group_construction_failure_preserves_entrances_and_owners(backend, invalid):
    available(backend)
    simulations = [member(index, backend) for index in range(2)]
    if invalid == "schedule":
        simulations[1] = member(
            1,
            backend,
            schedule=FixedSchedule.uniform(
                0.0, 0.001, 0.0005, save_planes=[0.0005, 0.001]
            ),
        )
    problems = [simulation.translate() for simulation in simulations]
    requests = [simulation._run_request() for simulation in simulations]
    starts = [Start.entrance(problem) for problem in problems]
    if invalid == "used":
        starts[1].take_field()
    elif invalid == "same_start":
        starts[1] = starts[0]
    fields = [entrance.field for entrance in starts]
    with ExitStack() as cleanup:
        owners = [constructed(simulation, cleanup) for simulation in simulations]
        if invalid == "same_owner":
            owners[1] = owners[0]
        elif invalid == "request":
            requests[1] = requests[1].with_stepper(requests[1].stepper)
        with pytest.raises(BackendError):
            start_group(owners, problems, requests, starts)
        assert all(
            entrance.field is field
            for entrance, field in zip(starts, fields, strict=True)
        )
        assert all(not owner.closed for owner in owners)


def test_native_partial_group_construction_keeps_caller_ownership(monkeypatch):
    group_module = pytest.importorskip("solprop_native.group")
    simulations = [member(index, "native") for index in range(2)]
    problems = [simulation.translate() for simulation in simulations]
    requests = [simulation._run_request() for simulation in simulations]
    starts = [Start.entrance(problem) for problem in problems]
    fields = [entrance.field for entrance in starts]
    sessions = []
    real_session = group_module.NativeSession

    def second_fails(*args, **kwargs):
        if sessions:
            raise BackendError("second session allocation failed")
        session = real_session(*args, **kwargs)
        sessions.append(session)
        return session

    with ExitStack() as cleanup:
        owners = [constructed(simulation, cleanup) for simulation in simulations]
        with monkeypatch.context() as patch:
            patch.setattr(group_module, "NativeSession", second_fails)
            with pytest.raises(BackendError, match="second session allocation failed"):
                start_group(owners, problems, requests, starts)
        assert sessions[0].closed
        assert all(
            entrance.field is field
            for entrance, field in zip(starts, fields, strict=True)
        )
        assert all(not owner.closed for owner in owners)
        group = start_group(owners, problems, requests, starts)
        cleanup.callback(group.close)
        assert all(entrance.field is None for entrance in starts)
        assert drive(group)[-1].outcome is Outcome.COMPLETED
        group.close()
        assert all(owner.closed for owner in owners)
