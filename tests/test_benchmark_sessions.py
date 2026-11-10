# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check that SEP 47's benchmark windows hold the events they measure.

Each case of the ``sep47`` set runs once, briefly, with the native adapter's structural
counters on (``benchmarks.worker.measure(count_events=True)``), and its counts must show
the events it was declared for: a statistics row at every accepted step, snapshots
serviced, a group's common steps, the native or host march of a coupled flow. A run is
served by the native execution session wherever it admits the run, as in production;
the transitional bridge's counts are checked beside it where they document what the
session removed (SEP 47 section 7.3).
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


def _no_native_session(*args):
    """Refuse every run to the native session, so the bridge serves it."""
    del args
    return "the test's bridge"


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
    march = "session_adams" if name.endswith("adams") else "session_runge_kutta"
    assert counts["march_calls"] == {march: counts["march_calls"][march]}
    assert counts["march_calls"][march] > 4
    # The rows are taken natively: the session returns to Python at its launch, its
    # planes and its end alone, and exports each kept plane once.
    observations = CASES[name].planes + 1
    assert counts["native_returns"] == {
        "session_observation": observations,
        "session_terminal": 1,
    }
    assert counts["planes"] == observations - 1
    assert counts["field_crossings"] == {"export": observations}
    assert counts["statistics_evaluations"] == counts["statistics_rows"] + observations
    assert counts["host_attempts"] == counts["transverse_reads"] == 0
    assert timed is not None
    assert (timed["accepted"], timed["rejected"]) == (run["accepted"], run["rejected"])


@pytest.mark.parametrize("name", ["stats_period1_rk", "snapshots_sparse"])
def test_the_bridge_counts_the_same_work_with_a_return_per_step(
    name, tmp_path, monkeypatch
):
    # The transitional bridge still serves what the native session refuses. Measured
    # the same way, it makes the same attempts, rows, planes and copies, and returns
    # to Python at every step the session keeps native.
    _, run, session = counted(name, tmp_path)
    monkeypatch.setattr("solprop_native.session.session_refusal", _no_native_session)
    _, bridged, bridge = counted(name, tmp_path / "bridge")
    assert (bridged["accepted"], bridged["rejected"]) == (
        run["accepted"],
        run["rejected"],
    )
    same = (
        "attempts",
        "accepted",
        "planes",
        "statistics_rows",
        "statistics_evaluations",
        "field_bytes",
        "field_loads",
        "load_bytes",
    )
    assert {key: bridge[key] for key in same} == {key: session[key] for key in same}
    assert (
        sum(bridge["field_crossings"].values()) == session["field_crossings"]["export"]
    )
    assert set(bridge["field_crossings"]) == {"plane"}
    cause = "statistics" if name.startswith("stats") else "step_planes"
    assert bridge["native_returns"][f"host_step:{cause}"] >= run["accepted"] - 2
    assert all(not key.startswith("session_") for key in bridge["native_returns"])
    assert all(not key.startswith("host_step") for key in session["native_returns"])


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
    # step_planes=1: a plane at every accepted step, each a requested observation and
    # a return; each snapshot asked for is one more, serviced at its endpoint, and
    # snapshots enabled but not asked for cost none (SEP 47 section 7.3).
    assert run["planes"] == run["accepted"] + 1
    expected = {"session_observation": run["planes"], "session_terminal": 1}
    if run["snapshots"]:
        expected["session_snapshot"] = run["snapshots"]
    assert counts["native_returns"] == expected
    # Every plane's field is exported once; a Runge-Kutta march has no history to
    # copy for a snapshot.
    assert counts["field_crossings"] == {"export": run["planes"]}


@pytest.mark.medium
def test_repeated_inputs_scans_and_groups_run_their_members(tmp_path):
    _, run, counts = counted("with_input_repeat", tmp_path)
    assert run["runs"] == 3
    assert run["finite"] is True
    assert counts["march_calls"]["session_runge_kutta"] >= 3
    observations = CASES["with_input_repeat"].planes + 1
    # Each run is a native session of its own: loaded once, a return at each plane
    # and at its end, each kept plane exported once.
    assert counts["native_returns"] == {
        "session_observation": 3 * observations,
        "session_terminal": 3,
    }
    assert counts["field_crossings"] == {"export": 3 * observations}
    assert counts["field_loads"] == {"start": 3}
    _, run, counts = counted("scan_local", tmp_path / "scan")
    assert run["runs"] == 4
    # Field-free: finiteness from the points' statistics; the watched runs' steps.
    assert run["finite"] is True
    assert run["accepted"] == counts["accepted"]
    assert counts["accepted"] > 4
    observations = CASES["scan_local"].planes + 1
    assert counts["native_returns"] == {
        "session_observation": 4 * observations,
        "session_terminal": 4,
    }
    # A field-free scan's members never copy a field to the host.
    assert counts["field_crossings"] == {}
    assert counts["field_bytes"] == 0
    _, run, counts = counted("group_batch", tmp_path / "group", fraction=0.5)
    schedule = build(CASES["group_batch"].configuration("default", 1, "native", 0.5))
    assert schedule.settings.schedule is not None
    steps = schedule.settings.schedule.n_steps
    assert counts["group_steps"] == steps > 1
    # The native group session: each member's native session returns at its launch,
    # at each plane and at its end, its steps native; and each point's resource
    # probe, a native session of its own, at its launch.
    planes = len(schedule.settings.save_planes)
    assert counts["native_returns"] == {
        "session_observation": 4 * (planes + 1) + 4,
        "session_terminal": 4,
    }
    assert counts["field_loads"] == {"start": 8}
    assert counts["field_crossings"] == {}
    assert counts["attempts"] == counts["accepted"] == 4 * steps
    assert counts["host_attempts"] == 0
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
    # A native session's batch: one return by its event, its chunk's work.
    counters.session_batch(
        "progress:work_limit",
        4,
        3,
        rows=3,
        evaluations=3,
        march="session_adams",
        calls=2,
    )
    counters.session_batch(
        "observation", 0, 0, evaluations=1, plane=True, march="session_adams"
    )
    counters.crossing("export", 64)
    counters.load("start", 32)
    record = counters.record()
    assert record["native_returns"]["session_progress:work_limit"] == 1
    assert record["native_returns"]["session_observation"] == 1
    assert record["march_calls"]["session_adams"] == 2
    assert (record["attempts"], record["accepted"]) == (14, 10)
    assert (record["statistics_rows"], record["statistics_evaluations"]) == (3, 4)
    assert record["planes"] == 1
    assert (record["field_crossings"], record["field_bytes"]) == ({"export": 1}, 64)
    assert (record["field_loads"], record["load_bytes"]) == ({"start": 1}, 32)
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
def test_a_narrow_and_a_wide_flow_march_natively(tmp_path):
    # Both keep the wall tutorials' walk-off absorber and its warning. Either build
    # steps a block of any width on the native Adams march, in the native session
    # (SEP 47 phase 4).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, _, narrow = counted("taper_flow_4", tmp_path, fraction=1.0)
        _, _, wide = counted("taper_flow_17", tmp_path / "wide", fraction=1.0)
    assert HostMarchWarning not in {type(item.message) for item in caught}
    for counts in (narrow, wide):
        assert counts["march_calls"].get("session_adams", 0) > 0
        assert counts["host_attempts"] == 0
