# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify Julia-owned group windows, lifetime adoption and independent failures."""

import os
from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.bounds import adams_rounding_budget, rk_rounding_budget
from backends.julia.cases import adams_case, cpu_case, low_storage_case
from backends.julia.test_cpu_monitor import _case as monitored_case
from backends.julia.test_cpu_monitor import _start as start_single
from sessions.executors import IDENTITY, Started, drive, fibre, window
from solprop.backends.events import ObservationReady, Outcome, Terminal, join_attempts
from solprop.backends.julia import _calls
from solprop.backends.julia._group import JuliaGroupSession
from solprop.backends.julia.backend import JuliaBackend, JuliaPrepared
from solprop.backends.session import ExecutionOptions, SessionError, Start, StopToken
from solprop.backends.sessions import start_group
from solprop.exceptions import BackendError, BoundaryWarning, ResponseBudgetError
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.stats import StatsSpec
from solprop.stepper.multistep import Multistep
from solprop.stepper.tableau import DOPRI5, TSIT5
from solprop.translate.problem import Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _owners(cases, cleanup):
    owners = []
    for case in cases:
        owner = JuliaBackend(
            case.problem,
            case.request,
            grid=window(),
            modes=(fibre().mode,),
            options=ExecutionOptions(
                backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]}
            ),
        )
        cleanup.callback(owner.close)
        owners.append(owner)
    return owners


def _group(cases, cleanup, *, stop=None):
    owners = _owners(cases, cleanup)
    starts = [Start.entrance(case.problem) for case in cases]
    group = start_group(
        owners,
        [case.problem for case in cases],
        [case.request for case in cases],
        starts,
        stop=stop,
    )
    cleanup.callback(group.close)
    assert isinstance(group, JuliaGroupSession)
    assert all(start.field is None for start in starts)
    return group, owners


def _drive(group):
    batches = []
    fields = [[] for _ in range(group.n_members)]
    while not group.finished:
        batch = group.advance()
        for index, events in enumerate(batch.members):
            for event in events:
                if isinstance(event, ObservationReady):
                    fields[index].append(group.export(index, event.ticket, IDENTITY))
        batches.append(batch)
        group.acknowledge(batch.receipt)
    return batches, fields


@pytest.mark.parametrize("factory", [cpu_case, adams_case, low_storage_case])
def test_group_fields_follow_independent_julia_sessions_with_exact_work(factory):
    original = (
        factory(method=Multistep(order=5, variable_order=False))
        if factory is adams_case
        else factory()
    )
    cases = []
    for scale in (0.75, 1.0):
        problem = replace(
            original.problem,
            initial=replace(
                original.problem.initial, field0=scale * original.problem.initial.field0
            ),
        )
        cases.append(Translation(problem, original.request))
    bounds = [
        adams_rounding_budget(case, case.request.integration.method.order)
        if factory is adams_case
        else rk_rounding_budget(case)
        for case in cases
    ]
    with ExitStack() as cleanup:
        group, owners = _group(cases, cleanup)
        batches, fields = _drive(group)
        assert len(batches) == 4 and batches[-1].outcome is Outcome.COMPLETED
        for index, case in enumerate(cases):
            with start_single(case) as session:
                alone = drive(Started(session), export=IDENTITY, deadline=60.0)
            assert bounds[index] > 0
            for got, expected in zip(fields[index], alone.fields, strict=True):
                # Two matched independent computations share the operation-derived
                # propagation envelope, not a bitwise threaded-execution promise.
                assert np.linalg.norm((got - expected).ravel()) <= bounds[index]
            history, rows = join_attempts(
                (event.history, event.step_stats)
                for batch in batches
                for event in batch.members[index]
            )
            assert history.n_steps == 8 and rows.z.size == 0
            assert group.member_status(index).attempts == 8
            assert (
                group.execution_record(index)["batch_strategy"] == "julia_group_session"
            )
        assert any(name.startswith("group.") for name in owners[0].memory_ledger())
        assert not any(name.startswith("group.") for name in owners[1].memory_ledger())
        group.close()
        assert all(owner.closed for owner in owners)
        assert all(np.all(np.isfinite(value)) for member in fields for value in member)


def test_group_tickets_refuse_wrong_member_stale_generation_and_bad_destination():
    with ExitStack() as cleanup:
        group, _ = _group([cpu_case(), cpu_case()], cleanup)
        launch = group.advance()
        event = launch.members[0][0]
        assert isinstance(event, ObservationReady)
        expected = group.export(0, event.ticket, IDENTITY)
        out = np.full(expected.shape, 3.0 + 4.0j)
        with pytest.raises(SessionError, match="not member"):
            group.export(1, event.ticket, IDENTITY, out)
        assert np.all(out == 3.0 + 4.0j)
        with pytest.raises(SessionError, match="held member"):
            group.export(0, replace(event.ticket, generation=9999), IDENTITY, out)
        with pytest.raises(BackendError, match="complex128"):
            group.export(
                0, event.ticket, IDENTITY, np.empty(expected.shape, dtype=np.complex64)
            )
        assert group.export(0, event.ticket, IDENTITY, out) is out
        assert np.array_equal(out, expected)
        assert group.noise_phase(0, event.ticket) is None
        with pytest.raises(SessionError, match="acknowledge"):
            group.advance()
        group.acknowledge(launch.receipt)
        with pytest.raises(SessionError, match="expired"):
            group.export(0, event.ticket, IDENTITY)
        with pytest.raises(SessionError, match="stale or foreign"):
            group.acknowledge(launch.receipt)


@pytest.mark.parametrize(
    "invalid", ["schedule", "same_owner", "same_start", "foreign", "tableau"]
)
def test_group_refusal_preserves_all_entrances_and_owners(invalid):
    cases = [cpu_case(), cpu_case(steps=4 if invalid == "schedule" else 8)]
    if invalid == "tableau":
        first = cases[0].request.integration.tableau
        other = TSIT5 if first is DOPRI5 else DOPRI5
        cases[1] = Translation(
            cases[1].problem,
            replace(
                cases[1].request,
                integration=replace(cases[1].request.integration, tableau=other),
            ),
        )
    with ExitStack() as cleanup:
        owners = _owners(cases, cleanup)
        starts = [Start.entrance(case.problem) for case in cases]
        fields = [start.field for start in starts]
        supplied = owners.copy()
        if invalid == "same_owner":
            supplied[1] = supplied[0]
        elif invalid == "same_start":
            starts[1] = starts[0]
            fields[1] = fields[0]
        elif invalid == "foreign":
            supplied.reverse()
        with pytest.raises(BackendError):
            owners[0].start_group(
                supplied,
                [case.problem for case in cases],
                [case.request for case in cases],
                starts,
            )
        assert all(
            start.field is value for start, value in zip(starts, fields, strict=True)
        )
        assert all(not owner.closed for owner in owners)


def test_second_member_allocation_failure_keeps_original_starts_and_owners(monkeypatch):
    cases = [cpu_case(), cpu_case()]
    real_start = JuliaPrepared.start
    made = []

    def fail_second(self, *args, **kwargs):
        if made:
            raise BackendError("second member allocation")
        session = real_start(self, *args, **kwargs)
        made.append(session)
        return session

    with ExitStack() as cleanup:
        owners = _owners(cases, cleanup)
        starts = [Start.entrance(case.problem) for case in cases]
        fields = [start.field for start in starts]
        with monkeypatch.context() as patch:
            patch.setattr(JuliaPrepared, "start", fail_second)
            with pytest.raises(BackendError, match="second member allocation"):
                start_group(
                    owners,
                    [case.problem for case in cases],
                    [case.request for case in cases],
                    starts,
                )
        assert made[0].closed
        assert all(not owner.closed for owner in owners)
        assert all(
            start.field is value for start, value in zip(starts, fields, strict=True)
        )
        group = start_group(
            owners,
            [case.problem for case in cases],
            [case.request for case in cases],
            starts,
        )
        cleanup.callback(group.close)
        assert _drive(group)[0][-1].outcome is Outcome.COMPLETED


def test_group_stop_ends_both_members_at_one_accepted_edge():
    token = StopToken()
    with ExitStack() as cleanup:
        group, _ = _group([cpu_case(), cpu_case()], cleanup, stop=token)
        launch = group.advance()
        group.acknowledge(launch.receipt)
        token.set()
        stopped = group.advance()
        assert stopped.outcome is Outcome.STOPPED
        for events in stopped.members:
            (event,) = events
            assert isinstance(event, Terminal) and event.outcome is Outcome.STOPPED
            assert event.history is not None
            assert event.z == 0.0 and event.history.n_steps == 0
        group.acknowledge(stopped.receipt)


def test_group_warning_as_error_ends_only_its_member_and_preserves_monitor_rows():
    base = monitored_case(budget=1.0)
    strict = BoundaryMonitor.build(
        window(), None, BoundaryCheck(budget=1e-3, stop_factor=1e6)
    )
    cases = [Translation(base.problem, base.request.for_run(monitor=strict)), base]
    with ExitStack() as cleanup:
        group, _ = _group(cases, cleanup)
        batches, _ = _drive(group)
        assert batches[-1].outcome is Outcome.COMPLETED
        stopped = [
            event
            for batch in batches
            for event in batch.members[0]
            if isinstance(event, Terminal)
        ]
        assert len(stopped) == 1 and isinstance(stopped[0].error, BoundaryWarning)
        assert group.member_status(0).outcome is Outcome.FAILED
        assert group.member_status(1).outcome is Outcome.COMPLETED
        record = group.monitor_record(1)
        assert record is not None
        planes = record["planes"]
        assert isinstance(planes, dict) and planes["z"] == [0.0, 0.01, 0.02]
        group.close()
        assert group.monitor_record(1) == record


@pytest.mark.parametrize("factory", [cpu_case, adams_case])
def test_group_transport_failure_delivers_previously_copied_chunks_once(
    monkeypatch, factory
):
    case = monitored_case(factory)
    real_attempts = _calls.attempts
    copied = 0

    def fail_second(runtime, handle, generation):
        nonlocal copied
        if generation > 1:
            copied += 1
            if copied == 2:
                raise BackendError("consumer copy failed")
        return real_attempts(runtime, handle, generation)

    with ExitStack() as cleanup:
        group, _ = _group([case, case], cleanup)
        launch = group.advance()
        group.acknowledge(launch.receipt)
        monkeypatch.setattr(_calls, "attempts", fail_second)
        failed = group.advance()
        assert failed.outcome is Outcome.FAILED
        assert isinstance(failed.error, BackendError)
        for index, events in enumerate(failed.members):
            (event,) = events
            assert isinstance(event, Terminal) and event.error is failed.error
            assert event.history is not None
            assert event.history.n_steps == group.member_status(index).attempts
            record = group.monitor_record(index)
            assert record is not None
            planes = record["planes"]
            assert isinstance(planes, dict)
            positions = planes["z"]
            assert len(positions) == len(set(positions))
            if factory is adams_case:
                execution = group.execution_record(index)
                calls = execution["response_calls"]
                assert isinstance(calls, dict)
                assert calls["accepted_steps"] == event.history.n_steps
                assert (
                    sum(count for _, count in calls["orders"]) == event.history.n_steps
                )
        group.acknowledge(failed.receipt)


def test_group_numerical_failure_keeps_actual_member_positions_and_one_primary_error():
    case = adams_case(method=Multistep(max_calls=3))
    with ExitStack() as cleanup:
        group, owners = _group([case, case], cleanup)
        launch = group.advance()
        group.acknowledge(launch.receipt)
        failed = group.advance()
        assert failed.outcome is Outcome.FAILED
        assert isinstance(failed.error, ResponseBudgetError)
        assert group.member_status(0).z > group.member_status(1).z
        for index, events in enumerate(failed.members):
            (event,) = events
            assert isinstance(event, Terminal) and event.error is failed.error
            assert event.history is not None
            assert event.history.n_steps == group.member_status(index).attempts
            assert event.ticket is not None
            exported = group.export(index, event.ticket, IDENTITY)
            assert np.all(np.isfinite(exported))
            if index == 0:
                assert isinstance(failed.error.field, np.ndarray)
                assert np.array_equal(exported, failed.error.field)
        assert any("delivery" in name for name in owners[0].memory_ledger())
        group.acknowledge(failed.receipt)


def test_group_statistics_rows_cover_each_member_without_export():
    base = cpu_case()
    spec = StatsSpec(names=("energy",), periodic=("energy",), period=3)
    case = Translation(
        base.problem,
        replace(
            base.request, observation=replace(base.request.observation, stats=spec)
        ),
    )
    with ExitStack() as cleanup:
        group, _ = _group([case, case], cleanup)
        batches = []
        while not group.finished:
            batch = group.advance()
            batches.append(batch)
            group.acknowledge(batch.receipt)
        for index in range(2):
            history, rows = join_attempts(
                (event.history, event.step_stats)
                for batch in batches
                for event in batch.members[index]
            )
            assert history.n_steps == 8 and rows.step.tolist() == [2, 5]
            assert np.all(np.isfinite(rows.values["energy"]))
            assert group._members[index]._field_exports == 0
