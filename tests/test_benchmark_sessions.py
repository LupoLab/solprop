# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check that SEP 47's benchmark windows hold the events they measure.

Each case of the ``sep47`` set runs once, briefly, with the native adapter's structural
counters on (``benchmarks.worker.measure(count_events=True)``), and its counts must show
the events it was declared for: a statistics row at every accepted step, snapshots
serviced, a group's lockstep steps, the native or host march of a coupled flow.
"""

import warnings

import numpy as np
import pytest
from benchmarks.builders import build
from benchmarks.records import SuiteError
from benchmarks.registry import CASES, Case, select
from benchmarks.worker import measure

from solprop.exceptions import HostMarchWarning

backend = pytest.importorskip("solprop_native.backend")
_counters = pytest.importorskip("solprop_native._counters")
NativeBackend = backend.NativeBackend


FIBRES = (
    "stats_period1_adams",
    "stats_period1_rk",
    "snapshots_idle",
    "snapshots_sparse",
    "snapshots_every",
    "with_input_repeat",
    "scan_local",
    "group_batch",
)


def counted(name, tmp_path, fraction=0.125, repeats=0):
    """Run a case's counted repetition at `fraction`, after `repeats` timed ones."""
    configuration = CASES[name].configuration("default", 1, "native", fraction)
    result = measure(configuration, repeats, None, count_events=True, scratch=tmp_path)
    counted = result["counted"]
    assert counted["counters"] is not None
    timed = result["repetitions"][0] if repeats else None
    return timed, counted, counted["counters"]


def test_the_set_holds_the_matrix_and_keeps_older_digests():
    names = {case.name for case in select(sets=("sep47",))}
    assert set(FIBRES) <= names
    assert {"taper_flow_4", "taper_flow_17"} <= names
    for base in ("radial_filament_256", "cartesian_filament_128"):
        for kept in ("saved", "free"):
            for output in ("memory", "hdf5"):
                case = CASES[f"{base}_{kept}_{output}"]
                assert case.output == output
                assert case.fields == ("all" if kept == "saved" else "none")
    # A field added at its default leaves a definition's digest as it was.
    plain = Case("x", 1, (), "prop_gnlse", "gnlse", 1.0, (1, 2, 2), "test")
    assert (
        plain.digest
        == Case(
            "x", 1, (), "prop_gnlse", "gnlse", 1.0, (1, 2, 2), "test", workload="run"
        ).digest
    )
    assert (
        plain.digest
        != Case(
            "x", 1, (), "prop_gnlse", "gnlse", 1.0, (1, 2, 2), "test", output="hdf5"
        ).digest
    )


@pytest.mark.parametrize(
    "keywords",
    [
        {"workload": "absent"},
        {"output": "disk"},
        {"fields": "some"},
        {"workload": "group"},
        {"snapshots": 1},
    ],
)
def test_a_case_refuses_what_the_worker_cannot_run(keywords):
    with pytest.raises(SuiteError):
        Case("x", 1, (), "prop_gnlse", "gnlse", 1.0, (1, 2, 2), "test", **keywords)


def test_field_free_and_scanned_cases_build_their_retention():
    free = CASES["radial_filament_256_free_hdf5"].configuration("default", 1)
    assert free["fields"] == "none"
    # A scan's points keep the default; run_scan sets what the scan keeps.
    scanned = build(CASES["scan_local"].configuration("default", 1))
    assert scanned.settings.fields == "all"
    group = build(CASES["group_batch"].configuration("default", 1))
    assert group.settings.schedule is not None
    assert group.settings.schedule.n_steps == 10
    snapshots = build(CASES["snapshots_every"].configuration("default", 1))
    assert snapshots.settings.step_planes == 1


@pytest.mark.parametrize("name", ["stats_period1_adams", "stats_period1_rk"])
def test_period_one_statistics_take_a_row_at_every_accepted_step(name, tmp_path):
    timed, run, counts = counted(name, tmp_path, repeats=1)
    assert counts["statistics_rows"] == run["accepted"] == counts["accepted"] > 4
    assert counts["attempts"] == run["accepted"] + run["rejected"]
    march = "adams_march" if name.endswith("adams") else "rk_march"
    assert counts["march_calls"][march] > 4
    # Today every row is a return to Python.
    assert counts["native_returns"]["host_step:statistics"] > 4
    assert timed is not None
    assert (timed["accepted"], timed["rejected"]) == (run["accepted"], run["rejected"])


@pytest.mark.parametrize(
    ("name", "every"),
    [
        ("snapshots_idle", 0),
        ("snapshots_sparse", 8),
        # A copy of the growing file at every step.
        pytest.param("snapshots_every", 1, marks=pytest.mark.medium),
    ],
)
def test_snapshot_cases_service_what_they_ask_for(name, every, tmp_path):
    _, run, counts = counted(name, tmp_path)
    assert run["accepted"] > 4
    assert run["snapshots"] == (run["accepted"] // every if every else 0)
    # step_planes=1: a plane, and today a return, at every accepted step.
    assert run["planes"] == run["accepted"] + 1
    assert counts["native_returns"]["host_step:step_planes"] >= run["accepted"] - 2


@pytest.mark.medium
def test_repeated_inputs_scans_and_groups_run_their_members(tmp_path):
    _, run, counts = counted("with_input_repeat", tmp_path)
    assert run["runs"] == 3
    assert run["finite"] is True
    assert counts["march_calls"]["rk_march"] >= 3
    assert counts["field_crossings"]["plane"] == 3 * (
        CASES["with_input_repeat"].planes + 1
    )
    _, run, counts = counted("scan_local", tmp_path / "scan")
    assert run["runs"] == 4
    # Field-free: finiteness from the points' statistics; the watched runs' steps.
    assert run["finite"] is True
    assert run["accepted"] == counts["accepted"]
    assert counts["accepted"] > 4
    # A field-free scan's members never copy a field to the host.
    assert counts["field_bytes"] == 0
    _, run, counts = counted("group_batch", tmp_path / "group", fraction=0.5)
    schedule = build(CASES["group_batch"].configuration("default", 1, "native", 0.5))
    assert schedule.settings.schedule is not None
    steps = schedule.settings.schedule.n_steps
    assert counts["group_steps"] == steps > 1
    assert counts["native_returns"] == {"group_step": steps}
    assert counts["host_attempts"] == counts["accepted"] == 4 * steps
    # Batch watches no point's run: its steps are uncounted, not zero.
    assert (run["accepted"], run["rejected"], run["finite"]) == (None, None, True)


def test_counters_attribute_a_host_step_to_the_work_due_on_it():
    counters = _counters.Counters()
    rows = np.zeros((3, 6))
    rows[:, 3] = (1, 0, 1)
    due = {"statistics": 2, "step_planes": 5}
    counters.march("rk_march", "host_step", rows, due, None)
    due = {"statistics": 2, "step_planes": 2}
    counters.march("rk_march", "host_step", rows, due, None)
    counters.march("rk_march", "host_step", rows, {"statistics": 4}, 0.1)
    counters.march("adams_march", "reached", rows[:1], {}, None)
    record = counters.record()
    assert record["native_returns"] == {
        "host_step:statistics": 1,
        "host_step:statistics+step_planes": 1,
        "host_step:transverse_read": 1,
        "reached": 1,
    }
    assert record["march_calls"] == {"adams_march": 1, "rk_march": 3}
    assert (record["attempts"], record["accepted"]) == (10, 7)
    # The per-attempt hooks live on the backend only while counting.
    hooks = ("_record", "_record_step_stats")
    assert _counters.disable() is None
    assert not any(name in NativeBackend.__dict__ for name in hooks)
    assert _counters.enable() is _counters.active
    assert all(name in NativeBackend.__dict__ for name in hooks)
    assert isinstance(_counters.disable(), _counters.Counters)
    assert _counters.active is None
    assert not any(name in NativeBackend.__dict__ for name in hooks)
    # The reason names are the backend's, in march::Reason order.
    assert backend._REASONS[backend._HOST_STEP] == "host_step"
    assert backend._reason_name(99) == "reason_99"


def test_cases_that_set_their_integrator_refuse_an_override():
    for name in ("stats_period1_adams", "stats_period1_rk"):
        case = CASES[name]
        assert case.steppers == ("default",)
        with pytest.raises(SuiteError, match="admits only default"):
            case.configuration("rk", 1)
    from benchmarks.runner import plan_rows  # noqa: PLC0415

    _, skipped = plan_rows((CASES["stats_period1_rk"],), ("multistep",), "1", 1)
    assert skipped == [
        "stats_period1_rk multistep: the case runs its own integrator only"
    ]


def test_unchecked_work_and_fields_are_never_a_match():
    from benchmarks.ab import same_work  # noqa: PLC0415

    assert same_work({(3, 1)}, {(3, 1)}) is True
    assert same_work({(3, 1)}, {(3, 0)}) is False
    assert same_work({(None, None)}, {(None, None)}) is None


@pytest.mark.slow
def test_a_narrow_flow_marches_natively_and_a_wide_one_on_the_host(tmp_path):
    # Both keep the wall tutorials' walk-off absorber and its warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, _, narrow = counted("taper_flow_4", tmp_path, fraction=1.0)
        found = {type(item.message) for item in caught}
        assert HostMarchWarning not in found
        _, _, wide = counted("taper_flow_17", tmp_path / "wide", fraction=1.0)
    assert HostMarchWarning in {type(item.message) for item in caught}
    assert narrow["march_calls"].get("adams_march", 0) > 0
    assert narrow["host_attempts"] == 0
    assert wide["march_calls"] == {}
    assert wide["host_attempts"] == wide["native_returns"]["host_attempt"] > 0
    # The host engine copies every accepted state to the host.
    assert wide["field_crossings"]["step"] == wide["accepted"]
