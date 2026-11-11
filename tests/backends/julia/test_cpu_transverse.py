# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify short Julia spatial-monitor sessions through the real Python bridge."""

import math
import os
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from tests.bounds import gamma

from backends.julia.test_transverse_records import edge_case
from sessions.executors import IDENTITY, Started, drive
from solprop.backends.events import ObservationReady, Outcome
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.session import ExecutionOptions, Start
from solprop.translate.problem import Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _start(case, grid, start=None):
    owner = JuliaBackend(
        case.problem,
        case.request,
        grid=grid,
        options=ExecutionOptions(
            node_chunk=3,
            backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]},
        ),
    )
    try:
        return owner.prepare(case.problem, case.request).start(
            Start.entrance(case.problem) if start is None else start
        )
    except BaseException:
        owner.close()
        raise


def _record(session) -> dict[str, Any]:
    # Public heterogeneous JSON; each consumed numerical field is checked below.
    record = session.monitor_record()
    assert record is not None
    return cast("dict[str, Any]", record)


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
@pytest.mark.parametrize("method", ["rk", "adams", "low_storage"])
def test_spatial_edge_attaches_after_launch_with_owned_record(kind, geometry, method):
    case, grid, edge = edge_case(geometry, kind=kind, method=method)
    field = np.array(case.problem.initial.field0, copy=True)
    field[..., ~case.problem.grids.active_mask] = 0
    reading = edge.reduction.reading(field, full=True)
    assert reading.totals is not None and reading.faces is not None
    launch = float(reading.totals[0])
    baseline = reading.faces[: len(edge._faces)] / launch
    # Two independent positive reductions and coarse inverse transforms. The
    # declared CPU envelope includes FFT normalization and the launch division;
    # absolute shares use total launch magnitude, including tiny face readings.
    bound = 8 * gamma(64 * (field.shape[0] + field.shape[-1]))
    assert 0 < bound < 1e-8 and math.isfinite(launch) and launch > 0
    with _start(case, grid) as session:
        initial = _record(session)
        assert initial["transverse"]["launch_energy"] is None
        batch = session.advance()
        event = batch.events[0]
        assert isinstance(event, ObservationReady)
        np.testing.assert_array_equal(session.export(event.ticket, IDENTITY), field)
        assert _record(session) == initial
        session.acknowledge(batch.receipt)
        result = drive(Started(session), work_limit=2, deadline=0.0)
        assert result.status.outcome is Outcome.COMPLETED
        record = _record(session)
        actual = record["transverse"]
        assert abs(actual["launch_energy"] - launch) <= bound * launch
        np.testing.assert_allclose(
            actual["edge_baseline"], baseline, rtol=0, atol=bound
        )
        assert actual["faces"] == [face.name for face in edge._faces]
        assert actual["reads"] > 0
        assert actual["transverse_status"] == "coarse"
        assert record["steps"] + record["skipped"] == 8
        assert edge.summary()["launch_energy"] is None
        assert any("transverse" in name for name in session.memory_ledger())
    assert _record(session) == record
    actual["edge_baseline"][0] = 999.0
    assert _record(session)["transverse"]["edge_baseline"][0] != 999.0


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
def test_transverse_resume_restores_original_totals_once(geometry):
    case, grid, _ = edge_case(geometry)
    planes = np.array([4e-6, 8e-6])
    planes.setflags(write=False)
    case = Translation(
        case.problem,
        replace(
            case.request,
            observation=replace(case.request.observation, save_planes=planes),
        ),
    )
    with _start(case, grid) as initial:
        launch = initial.advance()
        initial.acknowledge(launch.receipt)
        batch = initial.advance(deadline=60.0)
        event = batch.events[0]
        # The fixed schedule's launch warning is an ordinary barrier to acknowledge.
        while not isinstance(event, ObservationReady):
            assert batch.terminal is None
            initial.acknowledge(batch.receipt)
            batch = initial.advance(deadline=60.0)
            event = batch.events[0]
        assert event.z == planes[0]
        field = initial.export(event.ticket, IDENTITY)
        saved = _record(initial)
    start = Start(
        field=field,
        z=float(planes[0]),
        plane_index=1,
        plane_pending=False,
        resume=True,
        monitor={"record": saved, "plane": False},
    )
    with _start(case, grid, start) as resumed:
        resumed.continue_monitor({"record": {"transverse": {"reads": math.inf}}})
        result = drive(Started(resumed), work_limit=2, deadline=0.0)
        assert result.status.outcome is Outcome.COMPLETED
        actual = _record(resumed)
        for name in ("launch_energy", "edge_baseline", "noise_allowance"):
            assert actual["transverse"][name] == saved["transverse"][name]
        assert actual["transverse"]["reads"] > saved["transverse"]["reads"]
        assert actual["resumed_at"] == [planes[0]]
        assert actual["steps"] + actual["skipped"] == 8
