# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check owned edge lowering and legacy/public translations against the shared edge."""

import math
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from backends.julia.spatial_cases import spatial_case
from solprop import transverse_monitor as shared
from solprop.backends.julia._monitor import (
    MonitorLog,
    MonitorNotice,
    MonitorPayload,
    monitor_error,
    parse_resume,
)
from solprop.backends.julia._observations import lower_observations
from solprop.backends.julia._transverse import lower_transverse, transverse_format
from solprop.backends.julia._transverse_record import (
    TransversePayload,
    parse_transverse_resume,
    transverse_error,
    transverse_warning,
)
from solprop.exceptions import BackendError
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.spatial_absorber import SpatialAbsorber, SpatialAbsorberDesign
from solprop.translate.problem import Translation


def edge_case(geometry, *, collar=False, kind="envelope", method="rk"):
    case, grid = spatial_case(kind, geometry, method=method)
    if geometry == "radial":
        space = RadialGrid(4e-4, 8, collar_fraction=0.0)
    elif geometry == "line":
        space = LineGrid(4e-4, 8, collar_fraction=0.0)
    else:
        space = FreeGrid(4e-4, 6, 5e-4, 4, collar_fraction=0.0)
    design = None
    if collar:
        design = SpatialAbsorberDesign(
            SpatialAbsorber(kappa=3e5),
            np.zeros(case.problem.shape[0]),
            1.0,
            (1e6,),
            collar_rate=np.zeros(case.problem.shape[0]),
            alpha0=1e3,
            width=(1e-4,),
            kappa_p=(3e5,),
        )
    edge = shared.TransverseEdge(
        space,
        design,
        0.1,
        10.0,
        grid.length,
        np.arange(grid.n_omega),
        2 * np.pi / 800e-9,
    )
    monitor = BoundaryMonitor.build(
        grid, None, BoundaryCheck(budget=1.0), columns=space.spectral_weights
    )
    monitor.transverse = edge
    return Translation(case.problem, case.request.for_run(monitor=monitor)), grid, edge


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
@pytest.mark.parametrize("collar", [False, True])
def test_lowered_edge_owns_geometry_and_preserves_shared_policy(geometry, collar):
    case, grid, edge = edge_case(geometry, collar=collar)
    edge._launch, edge._reads = 92.0, 45
    actual = lower_observations(case.problem, case.request, grid=grid, trace_chunk=3)
    value = actual.transverse
    assert value is not None and value.trace_chunk == 3
    assert value.policy.warn == edge.budget**2
    assert value.policy.stop == (edge.stop_factor * edge.budget) ** 2
    assert value.policy.backstop_start == edge.length / shared.READS
    assert value.policy.backstop_max == edge.length / shared.BACKSTOP_READS
    assert value.policy.guard_numerator == 4.0 * edge._guard
    assert value.policy.has_collar is collar
    np.testing.assert_array_equal(value.faces, edge.tables.faces.T)
    np.testing.assert_array_equal(value.clip, edge.tables.clip.T)
    np.testing.assert_array_equal(value.totals, edge.tables.totals.T)
    np.testing.assert_array_equal(value.order, edge._kappa_order + 1)
    names = tuple(face.name for face in edge._faces)
    assert value.face_ranks == tuple(sorted(names).index(name) + 1 for name in names)
    assert edge._launch == 92.0 and edge._reads == 45
    for source, line in zip(edge.reduction.lines, value.lines, strict=True):
        np.testing.assert_array_equal(line.strip, source.strip + 1)
        assert line.inward == (source.inward[0] + 1, source.inward[1])
        assert line.onset == (
            None if source.onset is None else tuple(one + 1 for one in source.onset)
        )
        if isinstance(edge.space, RadialGrid):
            scale = np.sqrt(edge.space.r)
        else:
            assert isinstance(edge.space, FreeGrid)
            scale = np.ones(edge.space.n_x if source.axis == "x" else edge.space.n_y)
        np.testing.assert_array_equal(line.sample_scale, scale)

    def check_owned(record):
        if isinstance(record, np.ndarray):
            assert record.flags.owndata and record.flags.f_contiguous
            assert not record.flags.writeable
        elif is_dataclass(record):
            for descriptor in fields(record):
                check_owned(getattr(record, descriptor.name))
        elif isinstance(record, tuple):
            for item in record:
                check_owned(item)

    check_owned(value)
    assert not np.shares_memory(value.faces, edge.tables.faces)
    assert not np.shares_memory(value.column_weights, edge._columns)
    with pytest.raises(BackendError, match="chunk"):
        lower_transverse(case.problem, edge, 0)


def _record(edge):
    original = edge.summary()
    original.update(
        launch_energy=4.0,
        edge_baseline=[0.125] * len(edge._faces),
        edge_max=[0.25] * len(edge._faces),
        edge_max_raw=[0.5] * len(edge._faces),
        raw_ramp_removed=0.25,
        noise_ramp_removed=0.0625,
        noise_share_in_ramp=0.03125,
        declared_share_in_ramp=0.015625,
        noise_allowance=0.125,
        spatial_removed=0.125,
        current=0.25,
        current_baseline=0.125,
        strip_baseline=0.25,
        strip_floor=0.125,
        strip_max=0.5,
        returned_max=0.25,
        reads=7,
        largest_gap=0.01,
        warned_at=0.005,
        transverse_status="warned",
    )
    return original


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
@pytest.mark.parametrize(
    "status", ["clear", "warned", "insensitive", "coarse", "stopped"]
)
def test_edge_legacy_resume_and_record_match_shared_values(geometry, status):
    _, _, edge = edge_case(geometry)
    record = {**_record(edge), "transverse_status": status}
    parsed = parse_transverse_resume(record, len(edge._faces))
    assert parsed is not None
    edge.resume(record)
    expected = edge.summary()
    context = transverse_format(edge)
    assert context is not None
    payload = TransversePayload(
        parsed,
        0,
        edge.ramp_removed,
        True,
        (parsed.status, max(parsed.peak), edge.ramp_removed, parsed.reads),
    )
    assert payload.summary(context) == expected
    record["edge_baseline"][0] = 99.0
    assert parsed.baseline[0] == 0.125
    resumed = parse_resume({"record": {"transverse": expected}}, faces=len(edge._faces))
    assert resumed.numerical is None and resumed.transverse == parsed
    rows = np.empty((0, 5))
    transport = MonitorPayload(
        1, {}, rows, True, False, False, 0.01, None, None, payload
    )
    log = MonitorLog(BoundaryCheck(), 0.01, transverse=context).continued(resumed)
    log = log.captured(transport)
    summary = log.summary()
    assert summary is not None and summary["transverse"] == expected
    spatial = summary["transverse"]
    assert isinstance(spatial, dict)
    spatial["edge_baseline"][0] = 999.0
    repeated = log.summary()
    assert repeated is not None and repeated["transverse"] == expected


@pytest.mark.parametrize("launch", [None, False, 0.0, -1.0, math.nan])
def test_unusable_edge_resume_does_not_parse_later_malformed_fields(launch):
    record = {
        "launch_energy": launch,
        "edge_baseline": [0.0],
        "reads": math.inf,
        "edge_max": [object()],
    }
    assert parse_transverse_resume(record, 1) is None
    assert parse_transverse_resume(None, 1) is None


@pytest.mark.parametrize("count", [math.nan, math.inf])
def test_usable_edge_resume_preserves_shared_count_refusal(count):
    _, _, edge = edge_case("line")
    record = {**_record(edge), "reads": count}
    with pytest.raises(ValueError if math.isnan(count) else OverflowError):
        parse_transverse_resume(record, 2)
    with pytest.raises(ValueError if math.isnan(count) else OverflowError):
        edge.resume(record)


@pytest.mark.parametrize("code", [4, 6, 7, 8, 9, 11, 14, 15])
def test_edge_warning_uses_shared_text_and_one_based_face(code):
    _, _, edge = edge_case("line")
    context = transverse_format(edge)
    assert context is not None
    z, value, auxiliary = 0.005, 0.125, 0.001
    face = context.faces[1]
    expected = {
        4: shared._face_message(face, z, value),
        6: shared._ramp_band_message(value, stop=False),
        7: shared._clip_message(value, face),
        8: shared._coarse_message(value, auxiliary),
        9: shared._return_message(value, z, None),
        11: shared._ramp_removal_message(value, None),
        14: shared._insensitive_message(value),
        15: shared._UNWATCHED_MESSAGE,
    }
    assert (
        str(transverse_warning(context, code, (z, value, 2, auxiliary)))
        == expected[code]
    )
    with pytest.raises(BackendError, match="unsupported"):
        transverse_warning(context, 5, (z, value, 2, auxiliary))


@pytest.mark.parametrize("code", [5, 10, 12, 13])
def test_edge_stop_uses_shared_text_and_warning_coordinate(code):
    _, _, edge = edge_case("radial")
    context = transverse_format(edge)
    assert context is not None
    z, value, auxiliary = 0.005, 0.125, 0.001
    expected = {
        5: shared._stop_message(
            edge.space, edge.length, context.stop, "r", value, z=z, warned_at=auxiliary
        ),
        10: shared._return_message(value, z, context.stop),
        12: shared._ramp_removal_message(value, context.stop),
        13: shared._ramp_band_message(value, stop=True),
    }
    assert (
        str(transverse_error(context, code, (z, value, 1, auxiliary))) == expected[code]
    )
    with pytest.raises(BackendError, match="invalid face"):
        transverse_error(context, 5, (z, value, 0, auxiliary))


@pytest.mark.parametrize("warned_at", [math.nan, 2e-6])
def test_face_stop_uses_recorded_first_warning_not_unused_notice_scalar(warned_at):
    _, _, edge = edge_case("radial")
    context = transverse_format(edge)
    assert context is not None
    record = {**_record(edge), "warned_at": warned_at}
    state = parse_transverse_resume(record, 1)
    assert state is not None
    spatial = TransversePayload(state, 0, 0.0, False, (2, 0.5, 0.0, 7))
    payload = MonitorPayload(
        1,
        {},
        np.empty((0, 5)),
        True,
        False,
        False,
        4e-6,
        MonitorNotice(5, 4e-6, 0.5, 1, 0.0),
        None,
        spatial,
    )
    expected = shared._stop_message(
        edge.space,
        edge.length,
        context.stop,
        "r",
        0.5,
        z=4e-6,
        warned_at=warned_at,
    )
    assert str(monitor_error(BoundaryCheck(), payload, context)) == expected
