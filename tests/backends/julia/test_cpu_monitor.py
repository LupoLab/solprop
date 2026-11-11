# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify Julia-owned monitor records and service barriers on short real marches."""

import os
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from backends.julia.cases import adams_case, cpu_case, low_storage_case
from sessions.executors import IDENTITY, Started, drive, fibre, window
from solprop.backends.events import (
    ObservationReady,
    Outcome,
    ServiceKind,
    ServiceRequired,
)
from solprop.backends.julia import _session
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.session import ExecutionOptions, SessionError, Start
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.translate.problem import Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _case(factory=cpu_case, *, budget=1.0):
    original = factory()
    monitor = BoundaryMonitor.build(window(), None, BoundaryCheck(budget=budget))
    return Translation(original.problem, original.request.for_run(monitor=monitor))


def _start(case, start=None):
    owner = JuliaBackend(
        case.problem,
        case.request,
        grid=window(),
        modes=(fibre().mode,),
        options=ExecutionOptions(
            backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]}
        ),
    )
    return owner.prepare(case.problem, case.request).start(
        Start.entrance(case.problem) if start is None else start
    )


def _record(session) -> dict[str, Any]:
    # The public JSON record is heterogeneous; assertions below check its used fields.
    record = session.monitor_record()
    assert record is not None
    return cast("dict[str, Any]", record)


@pytest.mark.parametrize("factory", [cpu_case, adams_case, low_storage_case])
def test_temporal_monitor_attaches_after_launch_and_keeps_owned_rows(factory):
    case = _case(factory)
    session = _start(case)
    with session:
        before = _record(session)
        assert before is not None and before["launch_energy"] is None
        batch = session.advance()
        assert isinstance(batch.events[0], ObservationReady)
        assert session.monitor_record() == before
        session.acknowledge(batch.receipt)
        result = drive(Started(session), deadline=60.0)
        assert result.status.outcome is Outcome.COMPLETED
        record = _record(session)
        assert record is not None
        assert record["steps"] + record["skipped"] == 8
        assert record["planes"]["z"] == [0.0, 0.01, 0.02]
        assert case.request.boundary.monitor is not None
        assert case.request.boundary.monitor.summary()["launch_energy"] is None
        assert any(name.startswith("monitor.") for name in session.memory_ledger())
    assert session.monitor_record() == record
    record["planes"]["z"].append(99.0)
    assert _record(session)["planes"]["z"] == [0.0, 0.01, 0.02]


@pytest.mark.parametrize("plane_pending", [False, True])
def test_resumed_monitor_keeps_first_continuation_and_saved_launch(plane_pending):
    case = _case()
    with _start(case) as initial:
        launch = initial.advance()
        initial.acknowledge(launch.receipt)
        batch = initial.advance(deadline=60.0)
        event = batch.events[0]
        assert isinstance(event, ObservationReady) and event.z == 0.01
        field = initial.export(event.ticket, IDENTITY)
        saved = _record(initial)
        assert saved is not None
    continuation = {"record": saved, "plane": True, "unwatched": [(0.002, 0.003)]}
    start = Start(
        field=field.copy(),
        z=0.01,
        plane_index=1,
        plane_pending=plane_pending,
        resume=True,
        monitor=continuation,
    )
    with _start(_case(), start) as resumed:
        # The already supplied Start continuation wins even over a malformed fallback.
        resumed.continue_monitor({"record": {"launch_energy": 1.0, "steps": np.inf}})
        result = drive(Started(resumed), deadline=60.0)
        assert result.status.outcome is Outcome.COMPLETED
        record = _record(resumed)
        assert record is not None
        assert record["launch_energy"] == saved["launch_energy"]
        assert record["steps"] + record["skipped"] == 8
        assert record["planes"]["z"] == [0.0, 0.01, 0.02]
        assert record["resumed_at"] == [0.01]
        assert record["unwatched"] == [[0.002, 0.003]]
        with pytest.raises(SessionError, match="before it watches"):
            resumed.continue_monitor({})


def test_failed_monitor_copy_can_retry_without_duplicate_rows_or_attempts(monkeypatch):
    with _start(_case()) as session:
        launch = session.advance()
        session.acknowledge(launch.receipt)
        original = _session.read_monitor
        fail = True

        def interrupted(*args):
            nonlocal fail
            payload = original(*args)
            if fail:
                fail = False
                raise MemoryError("copy interrupted after numerical rows were read")
            return payload

        monkeypatch.setattr(_session, "read_monitor", interrupted)
        with pytest.raises(MemoryError, match="copy interrupted"):
            session.advance(deadline=60.0)
        ticket = session.pending_ticket()
        assert ticket is not None and ticket.z == 0.01
        history, rows = session.undelivered(ticket)
        assert history.n_steps == 4 and rows.n_rows == 0
        assert session.undelivered(ticket)[0].n_steps == 0
        record = _record(session)
        assert record is not None and record["planes"]["z"] == [0.0, 0.01]
        assert session.monitor_record() == record
        session.acknowledge(ticket.receipt)
        drive(Started(session), deadline=60.0)
        assert _record(session)["planes"]["z"] == [0.0, 0.01, 0.02]


@pytest.mark.parametrize("when", ["initial", "pending", "acknowledged"])
def test_output_fallback_can_arrive_at_each_unattached_launch_boundary(when):
    with _start(_case()) as session:
        launch = None
        if when != "initial":
            launch = session.advance()
            if when == "acknowledged":
                session.acknowledge(launch.receipt)
        session.continue_monitor({"plane": True, "unwatched": [(0.001, 0.002)]})
        if when == "pending":
            assert launch is not None
            session.acknowledge(launch.receipt)
        result = drive(Started(session), deadline=60.0)
        assert result.status.outcome is Outcome.COMPLETED
        record = _record(session)
        assert record["resumed_at"] == [0.0]
        assert record["unwatched"] == [[0.001, 0.002]]
        assert record["planes"]["z"] == [0.0, 0.01, 0.02]


@pytest.mark.parametrize("stop", [False, True])
def test_julia_guard_decisions_cross_as_a_warning_or_boundary_error(stop):
    case = _case(budget=1e-8 if stop else 1e-3)
    if not stop:
        monitor = case.request.boundary.monitor
        assert monitor is not None
        monitor.check = replace(monitor.check, stop_factor=1e6)
    with _start(case) as session:
        launch = session.advance()
        first = launch.events[0]
        assert isinstance(first, ObservationReady)
        launch_field = session.export(first.ticket, IDENTITY)
        session.acknowledge(launch.receipt)
        batch = session.advance(deadline=60.0)
        if stop:
            terminal = batch.terminal
            assert terminal is not None and terminal.outcome is Outcome.FAILED
            assert isinstance(terminal.error, BoundaryError)
            assert terminal.ticket is not None
            assert terminal.history is not None and terminal.history.n_steps == 0
            # Launch admission already applied the exact active mask. A guard stop
            # preserves that committed launch, not the unfiltered input descriptor.
            np.testing.assert_array_equal(
                session.export(terminal.ticket, IDENTITY), launch_field
            )
        else:
            event = batch.events[0]
            assert (
                isinstance(event, ServiceRequired) and event.kind is ServiceKind.WARNING
            )
            assert isinstance(event.warning, BoundaryWarning)
            assert event.warning.parameter == "boundary_check"
            assert event.ticket is not None and event.z == 0.0
            assert event.history is not None and event.history.n_steps == 0
            assert _record(session)["planes"]["z"] == [0.0]
