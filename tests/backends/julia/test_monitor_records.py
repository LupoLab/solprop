# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify detached monitor records against the shared continuation semantics."""

import math
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from sessions.executors import spy_monitor
from solprop.backends.julia._monitor import (
    MonitorLog,
    MonitorNotice,
    MonitorPayload,
    monitor_error,
    monitor_warning,
    parse_resume,
)
from solprop.exceptions import BackendError, BoundaryError, BoundaryWarning
from solprop.monitor import BoundaryCheck


def _old_record():
    record = spy_monitor([]).summary()
    return {
        **record,
        "launch_energy": 10.0,
        "launch_guard_low": 0.1,
        "launch_guard_high": 0.2,
        "launch_collar": 0.3,
        "raw_removed_fraction": 0.25,
        "removed_fraction": 0.125,
        "removed_signed": 0.0625,
        "floor_removed_fraction": 0.1875,
        "quiet_checks": 3,
        "collar_quiet": True,
        "steps": 5,
        "skipped": 2,
        "collar_left_out_checks": 1,
        "planes": {
            "z": [0.0, 0.1],
            "collar_occupancy": [0.3, None],
            "guard_low": [0.1, 0.2],
            "guard_high": [0.2, 0.3],
            "removed_fraction": [0.0, 0.125],
        },
        "resumed_at": [0.05],
        "unwatched": [[0.08, 0.09]],
    }


@pytest.mark.parametrize("quiet", [0, 3.75, -0.75, None, math.nan, math.inf])
def test_resume_preserves_shared_quiet_count_presence(quiet):
    old = {**_old_record(), "quiet_checks": quiet}
    parsed = parse_resume({"record": old, "plane": True})
    oracle = spy_monitor([])
    oracle.resume(old, plane=True)
    expected = oracle._restored_quiet
    assert parsed.numerical is not None
    assert parsed.numerical.has_quiet_run is (expected is not None)
    if expected is not None:
        assert parsed.numerical.quiet_run == expected[0]
        assert parsed.numerical.quiet is expected[1]


@pytest.mark.parametrize("name", ["steps", "skipped", "collar_left_out_checks"])
@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_resume_count_refusals_match_the_shared_monitor(name, value):
    old = {**_old_record(), name: value}
    with pytest.raises(BoundaryError, match=name):
        parse_resume({"record": old})
    with pytest.raises(BoundaryError, match=name):
        spy_monitor([]).resume(old, plane=False)


@pytest.mark.parametrize(
    "remove",
    [(), ("floor_removed_fraction",), ("floor_removed_fraction", "removed_signed")],
)
def test_legacy_removal_fallback_and_owned_history_match_reference(remove):
    old = _old_record()
    for key in remove:
        del old[key]
    continuation = {
        "record": old,
        "plane": True,
        "unwatched": [(0.1, 0.2), (0.2, 0.2), (0.3, 0.2)],
    }
    parsed = parse_resume(continuation)
    oracle = spy_monitor([])
    oracle.resume(old, plane=True, unwatched=continuation["unwatched"])
    expected = oracle.summary()
    assert parsed.numerical is not None
    assert parsed.numerical.floor_removed == expected["floor_removed_fraction"]
    assert parsed.numerical.last_plane_z == 0.1
    assert [list(one) for one in parsed.unwatched] == expected["unwatched"]
    assert list(parsed.resumed_at) == expected["resumed_at"]
    assert parsed.planes.shape == (2, 5) and not parsed.planes.flags.writeable
    assert np.isnan(parsed.planes[1, 1])
    old["planes"]["z"][0] = 99.0
    old["resumed_at"].append(99.0)
    old["unwatched"][0][0] = 99.0
    assert parsed.planes[0, 0] == 0.0
    assert parsed.resumed_at == (0.05,)
    assert parsed.unwatched[0] == (0.08, 0.09)


@pytest.mark.parametrize("launch", [None, 0.0, -1.0, math.nan])
def test_unusable_launch_keeps_only_unwatched_metadata(launch):
    old = {**_old_record(), "launch_energy": launch, "steps": math.inf}
    parsed = parse_resume({"record": old, "plane": True})
    assert parsed.numerical is None and parsed.plane
    assert parsed.planes.shape == (0, 5) and parsed.resumed_at == ()
    assert parsed.unwatched == ((0.08, 0.09),)


def test_no_record_is_still_a_resumed_execution():
    parsed = parse_resume({})
    assert parsed.numerical is None and not parsed.plane
    log = MonitorLog(BoundaryCheck(), 0.15).continued(parsed)
    assert log.resume is parsed
    assert log.continued(parse_resume({"record": _old_record()})) is log


def test_malformed_plane_lengths_refuse_before_restore():
    old = _old_record()
    old["planes"]["z"].append(0.2)
    with pytest.raises(BackendError, match="unequal plane columns"):
        parse_resume({"record": old})


def _payload(generation, *, rows=(), resuming=False, committed=0.2, attached=True):
    packed = np.asarray(rows, dtype=float).reshape(-1, 5)
    packed.setflags(write=False)
    record = {
        key: math.nan if value is None else value
        for key, value in _old_record().items()
        if key not in ("planes", "resumed_at", "unwatched")
    }
    return MonitorPayload(
        generation, record, packed, attached, False, resuming, committed, None, None
    )


def test_publication_delivers_rows_and_resume_coordinate_exactly_once():
    parsed = parse_resume({"record": _old_record(), "plane": True})
    log = MonitorLog(BoundaryCheck(), 0.15).continued(parsed)
    initial = _payload(0, resuming=True, attached=False, committed=math.nan)
    log = log.captured(initial)
    assert not log.resumed and log.resumed_at == (0.05,)
    first = _payload(2, rows=((0.2, 0.0, 0.0, 0.0, 0.125),), committed=0.2)
    log = log.captured(first)
    assert log.captured(first) is log
    assert log.resumed_at == (0.05, 0.15)
    log = log.captured(_payload(3, committed=0.3))
    assert log.resumed_at == (0.05, 0.15)
    first_summary = log.summary()
    assert first_summary is not None
    planes = cast("dict[str, list[float | None]]", first_summary["planes"])
    assert planes["z"] == [0.0, 0.1, 0.2]
    planes["z"].append(99.0)
    cast("list[list[float]]", first_summary["unwatched"])[0][0] = 99.0
    second_summary = log.summary()
    assert second_summary is not None
    second_planes = cast("dict[str, list[float | None]]", second_summary["planes"])
    assert second_planes["z"] == [0.0, 0.1, 0.2]
    assert second_summary["unwatched"] == [[0.08, 0.09]]


def test_partial_launch_failure_preserves_consumed_resume_metadata():
    log = MonitorLog(BoundaryCheck(), 0.15).continued(parse_resume({}))
    log = log.captured(_payload(1, committed=0.15, attached=True))
    assert log.resumed_at == (0.15,)
    assert log.payload is not None
    assert not log.payload.finished


def test_unmeasured_fields_and_plane_cells_format_without_numeric_changes():
    payload = _payload(1, rows=((0.0, math.nan, 0.0, math.nan, 0.0),))
    payload = replace(payload, record={**payload.record, "launch_energy": math.nan})
    log = MonitorLog(BoundaryCheck(), 0.0)
    assert log.summary() is None
    record = log.captured(payload).summary()
    assert record is not None
    assert record["launch_energy"] is None
    planes = cast("dict[str, list[float | None]]", record["planes"])
    assert planes["collar_occupancy"] == [None]
    assert planes["guard_low"] == [0.0]


def test_notices_format_the_existing_warning_and_error_types():
    check = BoundaryCheck()
    warning = monitor_warning(check, MonitorNotice(1, math.nan, 0.002))
    assert isinstance(warning, BoundaryWarning)
    assert warning.parameter == "boundary_check"
    assert "launch" in str(warning)
    assert isinstance(
        monitor_warning(check, MonitorNotice(3, math.nan, 0.01)), BoundaryWarning
    )
    with pytest.raises(BackendError, match="unsupported code"):
        monitor_warning(check, MonitorNotice(2, 0.1, 0.1))
    payload = replace(_payload(1), notice=MonitorNotice(2, 0.1, 0.1))
    assert isinstance(monitor_error(check, payload), BoundaryError)
    crossing = replace(_payload(1), crossing=(0.1, 0.01, 0.02))
    assert isinstance(monitor_error(check, crossing), BoundaryError)
    with pytest.raises(BackendError, match="no temporal refusal"):
        monitor_error(check, _payload(1))
