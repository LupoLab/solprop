# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise standard-suite control and recovery without a benchmark campaign."""

import dataclasses
import json
import math
import os
import subprocess
import sys

import pytest
from benchmarks import __main__ as cli
from benchmarks import collection, system
from benchmarks.__main__ import parser_for, resume_request
from benchmarks.comparison import (
    aa_control,
    clean_pairs,
    compare,
    compare_configuration,
    completed,
    needs_more,
    repeatability,
    resource_flags,
)
from benchmarks.planning import Selection, matrix, widths
from benchmarks.records import (
    SuiteError,
    atomic_record,
    attempts,
    canonical,
    distribution,
    identity,
    metadata_value,
    read_record,
    run_lock,
)
from benchmarks.references import measurement_identity
from benchmarks.registry import ADAPTIVE, CASES, KERR, select


@pytest.fixture
def profile():
    return system.Profile("test-machine", 1, 2, (0, 1), "topology")


def samples(arm, *, count=5, seconds=1.0):
    return [
        {
            "sample_id": f"{arm}-{index}",
            "arm": arm,
            "round": index,
            "configuration_id": "case",
            "status": "ok",
            "seconds": seconds,
            "clean": True,
            "resolution_hash": "same",
            "accuracy": {
                "status": "passed",
                "qualified_for_adoption": True,
                "reference_id": "same-reference-and-policy",
            },
            "peak_rss_bytes": 2**30,
            "startup_seconds": 2.0,
        }
        for index in range(count)
    ]


def test_registry_selection_deduplicates_and_preserves_pending():
    assert len(select()) == 4
    assert select(("rdw",), ("core",)) == select()
    union = select(all_cases=True)
    assert len(union) == len({case.name for case in union})
    assert "smoke" not in {case.name for case in union}
    assert CASES["vector_wall"] in union
    with pytest.raises(SuiteError, match="unknown set"):
        select(sets=("all",))
    with pytest.raises(SuiteError, match="unknown case"):
        select(("absent",))


def test_metadata_import_and_dry_run_cannot_construct_simulations():
    script = """
import importlib.abc
import sys
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        roots = ('numpy', 'solprop', 'solprop_native', 'numba', 'pyfftw')
        if fullname.split('.')[0] in roots:
            raise AssertionError('scientific import: ' + fullname)
sys.meta_path.insert(0, Refuse())
from benchmarks.__main__ import main
assert main(['list', '--set', 'core']) == 0
assert main(['run', '--dry-run', '--case', 'rdw']) == 0
assert main(['--dry-run', '--case', 'rdw']) == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_matrix_core_seven_quick_and_twenty_standard(profile):
    quick = matrix(select(), profile, Selection())
    assert len(quick["rows"]) == 7
    assert all(row["configuration"]["threads"] == 2 for row in quick["rows"])
    standard = matrix(select(), profile, Selection(mode="standard"))
    assert len(standard["rows"]) == 20
    assert all(row["status"] == "ready" for row in standard["rows"])
    radial = CASES["radial_filament_256"].measurement("fixed-rk")
    assert len(radial["schedule"]) == 11
    assert radial["schedule"][0] == 0.0
    assert radial["stop"] == radial["schedule"][-1]
    assert 0 < radial["stop"] < CASES["radial_filament_256"].length
    refused = matrix(
        select(("kerr_beam_384",)), profile, Selection(tracks=(ADAPTIVE[0],))
    )
    assert refused["rows"][0]["status"] == "excluded"
    assert CASES["kerr_beam_384"].tracks == (KERR,)
    assert widths("1,production,1", profile, "quick") == (1, 2)
    with pytest.raises(SuiteError):
        widths("3", profile, "standard")


@pytest.mark.parametrize("reviewed", [False, True])
def test_planning_reports_receipt_compatibility_without_loading_arrays(
    tmp_path, profile, reviewed, capsys
):
    selection = Selection(receipts=tmp_path, tracks=(ADAPTIVE[1],))
    cases = select(("rdw",))
    configuration = matrix(cases, profile, selection)["rows"][0]["configuration"]
    receipt = {
        "revision": 1,
        "case_hash": configuration["case_hash"],
        "measurement_hash": measurement_identity(configuration),
        "qualified_for_adoption": reviewed,
        "fields_file": "not-present.npy",
    }
    path = tmp_path / "rdw-adaptive-rk.json"
    atomic_record(path, receipt)
    planned = matrix(cases, profile, selection)
    assert planned["rows"][0]["reference"].startswith(
        "reviewed receipt" if reviewed else "available receipt"
    )
    cli.describe(planned)
    assert "memory budget 8.00 GiB" in capsys.readouterr().out
    atomic_record(path, {**receipt, "measurement_hash": "changed-window"})
    assert matrix(cases, profile, selection)["rows"][0]["reference"] == (
        "incompatible receipt"
    )
    path.write_text("broken JSON", encoding="utf-8")
    assert matrix(cases, profile, selection)["rows"][0]["reference"] == (
        "invalid receipt"
    )


def test_default_command_runs_quick_and_prints_failures(monkeypatch, tmp_path, capsys):
    observed = []

    def collect_stub(plan, options):
        observed.append((plan, options))
        return {"status": "collected", "execution_ok": True}

    monkeypatch.setattr(cli, "collect", collect_stub)
    monkeypatch.setattr(cli, "attempts", lambda path: [])
    monkeypatch.setattr(
        cli, "new_collection", lambda args: collection.Collection(tmp_path)
    )
    assert cli.main([]) == 0
    plan = observed[0][0]
    assert plan["mode"] == "quick"
    assert len(plan["rows"]) == 7
    assert all(row["configuration"]["backend"] == "native" for row in plan["rows"])
    row = samples("candidate", count=1)[0]
    row.update(
        configuration_id=plan["rows"][0]["id"], status="failed", error="worker crashed"
    )
    cli.show_results(plan, [row])
    output = capsys.readouterr().out
    assert "worker crashed" in output
    assert "warm / s" in output


def test_child_environment_overrides_incoming_pools_before_import():
    incoming = {
        "PATH": "test",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OPENBLAS_NUM_THREADS": "72",
        "SOLPROP_THREADS": "72",
        "SOLPROP_OVERSUBSCRIBE": "1",
        "SOLPROP_FFTW_PLAN": "estimate",
    }
    result = system.thread_environment(1, "budget", incoming)
    assert result["OPENBLAS_NUM_THREADS"] == "1"
    assert result["SOLPROP_THREADS"] == "1"
    assert "SOLPROP_OVERSUBSCRIBE" not in result
    assert result["SOLPROP_FFTW_THREADS"] == "1"
    assert result["SOLPROP_FFTW_PLAN"] == "estimate"
    assert result["OMP_PROC_BIND"] == "false"
    assert "OMP_PLACES" not in result
    assert incoming["OMP_PROC_BIND"] == "close"


@pytest.mark.parametrize("policy", ["budget", "blas1", "all-pools-N", "as-installed"])
def test_child_environment_handles_native_pool_overrides(policy):
    pools = (
        "SOLPROP_NATIVE_KERNEL_THREADS",
        "SOLPROP_NATIVE_FFT_THREADS",
        "SOLPROP_NATIVE_RAMAN_FFT_THREADS",
        "SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS",
    )
    threshold = "SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES"
    incoming = dict.fromkeys(pools, "99") | {threshold: "12345"}
    result = system.thread_environment(4, policy, incoming)
    if policy == "as-installed":
        assert result == incoming
    else:
        assert threshold not in result
        for name in pools:
            if policy == "all-pools-N":
                assert result[name] == "4"
            else:
                assert name not in result
    assert incoming == dict.fromkeys(pools, "99") | {threshold: "12345"}


def test_environment_identity_accepts_runtime_widths_but_refuses_changed_builds():
    inventory = {
        "packages": {"solprop": "test"},
        "extension_hash": "original-binary",
        "build_info": {
            "compiler": "Clang",
            "runtime_threads": 18,
            "fftw_threads": 18,
            "blas_threads": 18,
            "explicit_thread_pools": 0,
            "blas_thread_policy": "the Kokkos thread count",
        },
    }
    sample = {
        **inventory,
        "build_info": {
            **inventory["build_info"],
            "runtime_threads": 1,
            "fftw_threads": 1,
            "blas_threads": 1,
            "explicit_thread_pools": 15,
            "blas_thread_policy": "OMP_NUM_THREADS at process start",
        },
    }
    assert collection.environment_identity(
        inventory
    ) == collection.environment_identity(sample)
    changed = {**sample, "extension_hash": "replacement-binary"}
    assert collection.environment_identity(
        inventory
    ) != collection.environment_identity(changed)
    changed = {**sample, "build_info": {**sample["build_info"], "compiler": "GCC"}}
    assert collection.environment_identity(
        inventory
    ) != collection.environment_identity(changed)
    assert sample["build_info"]["runtime_threads"] == 1


def test_calibration_and_balanced_adjacent_rounds():
    rows = [{"id": index} for index in range(8)]
    first = collection.ordered(rows, 30, 0, True)
    second = collection.ordered(rows, 30, 1, True)
    assert first == collection.ordered(rows, 30, 0, True)
    assert {row["id"] for row, _ in first} == set(range(8))
    assert all(arms == ("baseline", "candidate") for _, arms in first)
    assert all(arms == ("candidate", "baseline") for _, arms in second)
    assert collection.calibration([0.2, 0.3], 1e-9) == (1, False)
    assert collection.calibration([0.02, 0.04], 1e-9) == (5, False)
    assert collection.calibration([0.001, 0.01], 1e-9) == (10, True)


def test_contention_extends_whole_rounds_without_joining_gaps():
    left, right = samples("baseline", count=7), samples("candidate", count=7)
    right[2]["clean"] = False
    a, b = clean_pairs(left, right)
    assert [row["round"] for row in a] == [0, 1, 3, 4, 5, 6]
    assert len(b) == 6
    assert repeatability(a) == 0
    assert repeatability(a[:4]) is None
    assert needs_more(left[:5] + right[:5], ["case"], ("baseline", "candidate"))
    assert not needs_more(left + right, ["case"], ("baseline", "candidate"))
    for index in (0, 1):
        right[index]["clean"] = False
    assert needs_more(left + right, ["case"], ("baseline", "candidate"))


def test_comparison_requires_complete_qualified_production_set():
    rows = samples("baseline") + samples("candidate", seconds=0.9)
    plan = {
        "mode": "standard",
        "profile": {"production": 2},
        "rows": [
            {
                "id": "case",
                "configuration": {"case": "rdw", "threads": 2, "track": "adaptive-rk"},
            }
        ],
    }
    assert compare(plan, rows)["status"] == "passed"
    quick = compare({**plan, "mode": "quick"}, rows)
    assert quick["status"] == "exploratory"
    # Exact synthetic times: this ratio is descriptive even without adoption.
    assert quick["configurations"][0]["all_ratio"] == pytest.approx(0.9, abs=1e-15)
    assert (
        compare({**plan, "profile": {"production": 4}}, rows)["status"] == "unresolved"
    )
    assert compare(plan, rows[:5])["status"] == "incomplete"
    assert (
        compare_configuration(rows[:5], rows[5:], paired=False)["status"]
        == "unresolved"
    )
    rows[-1]["resolution_hash"] = "changed"
    assert compare(plan, rows)["status"] == "incompatible"
    rows[-1]["status"] = "invalid"
    assert compare(plan, rows)["status"] == "numerically-invalid"


def test_resource_flags_and_single_sample_dispersion_are_explicit():
    left, right = samples("baseline"), samples("candidate")
    right[0]["peak_rss_bytes"] += 70 * 2**20
    for row in right:
        row["startup_seconds"] = 2.3
    flags = resource_flags(left, right)
    assert flags["peak_rss_bytes"]["flag"]
    assert flags["startup_seconds"]["flag"]
    assert resource_flags(left, right[:4])["peak_rss_bytes"]["status"] == "unavailable"
    assert distribution([1.0]).mad is None
    with pytest.raises(SuiteError, match="finite"):
        distribution([math.inf])


def test_atomic_shards_ignore_interruption_and_preserve_retries(tmp_path, monkeypatch):
    shard = tmp_path / "attempts" / "000000.json"
    failed = {**samples("candidate")[0], "status": "failed"}
    atomic_record(shard, failed, immutable=True)
    with pytest.raises(SuiteError, match="replace"):
        atomic_record(shard, samples("candidate")[0], immutable=True)
    atomic_record(
        shard.with_name("000001.json"), samples("candidate")[0], immutable=True
    )
    shard.with_name(".pending-interrupted").write_text(
        '{"incomplete":', encoding="utf-8"
    )
    assert len(attempts(tmp_path)) == 2
    assert completed(attempts(tmp_path))[0]["status"] == "ok"

    def fail_sync(_descriptor):
        raise OSError("interrupted flush")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="interrupted"):
        atomic_record(shard.with_name("000002.json"), failed)
    assert not shard.with_name("000002.json").exists()
    assert len(attempts(tmp_path)) == 2


def test_lock_is_exclusive_and_released_after_error(tmp_path):
    with pytest.raises(RuntimeError), run_lock(tmp_path):
        with pytest.raises(SuiteError, match="locked"), run_lock(tmp_path):
            pytest.fail("second writer acquired a held lock")
        raise RuntimeError("interrupted controller")
    with run_lock(tmp_path):
        atomic_record(tmp_path / "recovered.json", {"ok": True})


def test_synthetic_excessive_adams_memory_is_refused_before_launch(monkeypatch):
    monkeypatch.setattr(system, "effective_memory", lambda: 64 * 2**30)
    assert not system.admission(200 * 2**30, None)["admitted"]
    assert not system.admission(None, None)["admitted"]
    assert system.admission(None, 16 * 2**30, allow_unestimated=True)["admitted"]
    assert not system.admission(None, None, allow_unestimated=True)["admitted"]


def test_counter_guard_does_not_treat_unknown_as_zero():
    before = {"time": 1.0, "busy": [1.0, 2.0], "swap_bytes": None}
    after = {"time": 2.0, "busy": [1.1, 2.1], "swap_bytes": None}
    assert system.interval(before, after, (0, 1))["confidence"] == "unknown"
    assert not system.interval(before, after, (0, 1))["quiet"]
    before["swap_bytes"], after["swap_bytes"] = 0, 0
    assert system.interval(before, after, (0, 1), own_cpu=0.1)["quiet"]


def test_strict_json_distinguishes_metadata_infinity_from_samples(tmp_path):
    with pytest.raises(ValueError):
        canonical({"seconds": math.inf})
    material = metadata_value({"validity": [0.0, math.inf]})
    assert json.loads(canonical(material))["validity"][1] == {
        "nonfinite_metadata": "inf"
    }
    path = tmp_path / "invalid.json"
    path.write_text('{"seconds": NaN}', encoding="utf-8")
    with pytest.raises(SuiteError, match="non-finite"):
        read_record(path)
    assert identity(dataclasses.asdict(distribution([1.0])))


@pytest.fixture
def fake_worker(tmp_path):
    package = tmp_path / "fake" / "benchmarks"
    package.mkdir(parents=True)
    (package / "worker.py").write_text(
        "import json, os, sys, time\n"
        "r = json.load(open(sys.argv[1], encoding='utf-8'))\n"
        "if r['scenario'] == 'crash': os._exit(7)\n"
        "if r['scenario'] == 'timeout': time.sleep(100)\n"
        "event = {'event': 'result', 'result': {'status': 'ok'}}\n"
        "print(json.dumps(event), flush=True)\n",
        encoding="utf-8",
    )
    return package.parent


@pytest.mark.parametrize(
    "scenario, status, limit",
    [
        ("crash", "failed", 60.0),
        ("timeout", "timeout", 0.3),
        ("missing", "failed", 60.0),
    ],
)
def test_worker_crash_timeout_and_missing_receipts_are_rows(
    tmp_path,
    fake_worker,
    scenario,
    status,
    limit,
):
    # Only the timeout case runs into its limit; the others get room for a slow
    # interpreter start (a loaded Windows runner can take longer than 0.3 s).
    result = collection.launch(
        sys.executable,
        {"scenario": scenario, "harness_root": str(fake_worker)},
        dict(os.environ),
        tmp_path / "attempt",
        (limit, None),
        system.eligible_cpus(),
    )
    assert result["status"] == status
    assert result["error"]
    assert result["controller_cpu_seconds"] >= 0
    assert (tmp_path / "attempt" / "request.json").exists()


def test_resume_preserves_failure_then_skips_completed_sample(
    tmp_path, monkeypatch, profile
):
    calls = []
    provenance = {"packages": {"solprop": "test"}}

    def launch(_python, request, _environment, _directory, _limits, _allocation):
        if request.get("inventory"):
            return {"status": "ok", "provenance": provenance}
        calls.append(request)
        if len(calls) == 1:
            return {"status": "failed", "error": "synthetic crash"}
        return {
            "status": "ok",
            "provenance": provenance,
            "seconds": 1.0,
            "peak_rss_bytes": 1024,
            "during_guard": {"quiet": True},
            "accuracy": {"status": "not-qualified"},
            "timer_resolution": 1e-9,
        }

    monkeypatch.setattr(collection, "launch", launch)
    monkeypatch.setattr(collection, "guard", lambda *a, **k: {"quiet": True})
    monkeypatch.setattr(collection, "git_receipt", lambda _root: {"head": "same"})
    monkeypatch.setattr(system, "effective_memory", lambda: 64 * 2**30)
    plan = matrix(select(("smoke",)), profile, Selection(tracks=(ADAPTIVE[1],)))
    options = collection.Collection(tmp_path / "run")
    assert not collection.collect(plan, options)["execution_ok"]
    resume = dataclasses.replace(options, resume=True)
    assert collection.collect(plan, resume)["execution_ok"]
    assert collection.collect(plan, resume)["execution_ok"]
    assert len(calls) == 2
    assert [row["status"] for row in attempts(options.output)] == ["failed", "ok"]
    with pytest.raises(SuiteError, match="identity differs"):
        collection.collect({**plan, "suite_version": 2}, resume)


def test_resume_preserves_interrupted_worker_files_and_uses_fresh_attempt_ids(
    tmp_path, monkeypatch, profile
):
    provenance = {"packages": {"solprop": "test"}}
    directories = []

    def launch(_python, request, _environment, directory, _limits, _allocation):
        if request.get("inventory"):
            return {"status": "ok", "provenance": provenance}
        atomic_record(directory / "request.json", request, immutable=True)
        directories.append(directory.name)
        if len(directories) == 2:
            raise KeyboardInterrupt
        return {
            "status": "ok",
            "provenance": provenance,
            "seconds": 1.0,
            "during_guard": {"quiet": True},
            "accuracy": {"status": "not-qualified"},
        }

    monkeypatch.setattr(collection, "launch", launch)
    monkeypatch.setattr(collection, "guard", lambda *a, **k: {"quiet": True})
    monkeypatch.setattr(collection, "git_receipt", lambda _root: {"head": "same"})
    monkeypatch.setattr(system, "effective_memory", lambda: 64 * 2**30)
    plan = matrix(select(("smoke",)), profile, Selection(tracks=(ADAPTIVE[1],)))
    plan.update(rounds=3, maximum_rounds=3)
    options = collection.Collection(tmp_path / "run")
    with pytest.raises(KeyboardInterrupt):
        collection.collect(plan, options)
    interrupted = options.output / "raw" / "000001" / "request.json"
    original = interrupted.read_bytes()
    resume = dataclasses.replace(options, resume=True)
    assert collection.collect(plan, resume)["execution_ok"]
    assert collection.collect(plan, resume)["execution_ok"]
    assert interrupted.read_bytes() == original
    assert directories == ["000000", "000001", "000002", "000003"]
    assert [row["attempt_id"] for row in attempts(options.output)] == [0, 2, 3]


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu"])
@pytest.mark.parametrize("track", ADAPTIVE)
def test_small_worker_reset_and_history_conventions(tmp_path, profile, backend, track):
    configuration = matrix(
        select(("smoke",)),
        profile,
        Selection(backend=backend, threads="1", tracks=(track,)),
    )["rows"][0]["configuration"]
    result = collection.launch(
        sys.executable,
        {"configuration": configuration, "repetitions": 2, "verify_reset": True},
        system.thread_environment(1, "budget", dict(os.environ)),
        tmp_path / "worker",
        (180.0, 2 * 2**30),
        system.eligible_cpus(),
    )
    assert result["status"] == "ok", result
    reset = result["reset_evidence"]
    # Identical discrete reference runs: 64 rounding operations per stage and
    # seven stages per accepted step give a conservative accumulated L2 budget.
    bound = 64 * 7 * result["repetitions"][0]["accepted"] * sys.float_info.epsilon
    assert max(reset["plane_errors"]) <= bound
    assert all(counts == reset["fresh_counts"] for counts in reset["reset_counts"])
    history = reset["history_statistics_noise"]
    assert history["same_columns"]
    assert history["step_history/accepted"]["same_decisions"]
    for name, item in history.items():
        if name in ("same_columns", "step_history/accepted"):
            continue
        assert item["same_shape"]
        # Repeated histories use the same operations and the field budget above.
        assert (item["relative_l2"] or item["absolute_l2"]) <= bound
    assert result["pilot_edges"][0] == 0
    assert all(
        right > left
        for left, right in zip(
            result["pilot_edges"], result["pilot_edges"][1:], strict=False
        )
    )
    assert result["peak_rss_bytes"] == result["measurement_peak_bytes"]
    assert result["startup_seconds"] > 0


def test_harness_snapshot_is_frozen_before_workers(tmp_path, monkeypatch):
    root = tmp_path / "source"
    source = root / "benchmarks" / "worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("original", encoding="utf-8")
    monkeypatch.setattr(collection, "ROOT", root)
    snapshot = tmp_path / "snapshot"
    first = collection.freeze_harness(snapshot)
    assert first == collection.freeze_harness(snapshot)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(SuiteError, match="stored harness differs"):
        collection.freeze_harness(snapshot)


def test_aa_bands_supply_unpaired_repeatability_only_with_clean_counts():
    rows = samples("baseline") + samples("candidate", seconds=1.01)
    plan = {
        "mode": "standard",
        "profile": {"production": 2},
        "environments": {"baseline": {"build": "same"}},
        "rows": [
            {
                "id": "case",
                "configuration": {"case": "rdw", "threads": 2, "track": "adaptive-rk"},
            }
        ],
    }
    report = aa_control(plan, rows)
    band = report["configurations"]["case"]["band"]
    assert band == pytest.approx(2 * math.log(1.01), rel=4 * sys.float_info.epsilon)
    faster = samples("baseline") + samples("candidate", seconds=0.9)
    assert (
        compare(plan, faster, paired=False, aa_bands=report["configurations"])["status"]
        == "passed"
    )
    rows[5]["clean"] = False
    assert aa_control(plan, rows)["configurations"]["case"]["band"] is None


def test_asymmetric_memory_refusal_is_a_resource_regression():
    left, right = samples("baseline"), samples("candidate")
    right[0]["status"] = "refused"
    report = compare_configuration(left, right)
    assert report["status"] == "regression"
    assert "memory admission" in report["resource_regression"]
    left[0]["status"] = "refused"
    assert compare_configuration(left, right)["status"] == "incomplete"


def test_failed_calibration_requires_retry_and_reuses_successful_arm(
    tmp_path, monkeypatch, profile
):
    calls = []

    def attempt(row, sample, options, plan, receipt):
        calls.append(sample.arm)
        return {
            "attempt_id": len(attempts(options.output)),
            "sample_id": sample.arm,
            "configuration_id": row["id"],
            "arm": sample.arm,
            "phase": "pilot",
            "status": "failed" if len(calls) == 2 else "ok",
            "seconds": 0.02,
            "timer_resolution": 1e-9,
        }

    monkeypatch.setattr(collection, "_attempt", attempt)
    monkeypatch.setattr(collection, "guard", lambda *a, **k: {"quiet": True})
    plan = matrix(
        select(("smoke",)),
        profile,
        Selection(mode="standard", tracks=(ADAPTIVE[1],), threads="1"),
    )
    options = collection.Collection(tmp_path, baseline_python=sys.executable)
    with pytest.raises(SuiteError, match="calibration failed"):
        collection.calibrate(plan, options)
    assert not (tmp_path / "repetitions.json").exists()
    counts = collection.calibrate(plan, options)
    assert len(calls) == 3
    assert list(counts.values()) == [{"count": 5, "capped": False}]
    assert len(attempts(tmp_path)) == 3


def test_resume_cli_restores_saved_options_and_rejects_changes(tmp_path):
    options = dataclasses.asdict(collection.Collection(tmp_path, seed=123, timeout=9.0))
    options.pop("resume")
    options["output"] = str(tmp_path)
    options["references"] = str(options["references"].resolve())
    atomic_record(tmp_path / "manifest.json", {"plan": {"options": options}})
    tokens = ["run", "--resume", str(tmp_path)]
    _, restored = resume_request(parser_for().parse_args(tokens), tokens)
    assert restored.resume
    assert restored.seed == 123
    assert restored.timeout == 9.0
    changed = [*tokens, "--seed", "1"]
    with pytest.raises(SuiteError, match="resume seed differs"):
        resume_request(parser_for().parse_args(changed), changed)
    changed = [*tokens, "--case", "rdw"]
    with pytest.raises(SuiteError, match="restores the saved selection"):
        resume_request(parser_for().parse_args(changed), changed)


def test_comparison_requires_the_same_reference_and_policy_revision():
    left, right = samples("baseline"), samples("candidate", seconds=0.9)
    right[0]["accuracy"]["reference_id"] = "corrected-reference"
    result = compare_configuration(left, right)
    assert result["status"] == "incompatible"
    assert "revalidate both arms" in result["reason"]


@pytest.mark.parametrize("crash", [False, True])
def test_accuracy_override_continues_rounds_but_does_not_hide_crashes(
    tmp_path, monkeypatch, profile, crash
):
    calls = []
    provenance = {"packages": {"solprop": "test"}}

    def launch(_python, request, _environment, _directory, _limits, _allocation):
        if request.get("inventory"):
            return {"status": "ok", "provenance": provenance}
        calls.append(request)
        assert request["allow_accuracy_failure"]
        return {
            "status": "failed" if crash else "ok",
            "provenance": provenance,
            "accuracy": {"status": "invalid", "reason": "synthetic accuracy failure"},
            "accuracy_overridden": not crash,
            "seconds": 1.0,
            "timer_resolution": 1e-9,
            "during_guard": {"quiet": True},
        }

    monkeypatch.setattr(collection, "launch", launch)
    monkeypatch.setattr(collection, "guard", lambda *a, **k: {"quiet": True})
    monkeypatch.setattr(collection, "git_receipt", lambda _root: {"head": "same"})
    monkeypatch.setattr(system, "effective_memory", lambda: 64 * 2**30)
    plan = matrix(select(("smoke",)), profile, Selection(tracks=(ADAPTIVE[1],)))
    plan.update(rounds=5, maximum_rounds=5)
    options = collection.Collection(tmp_path, allow_accuracy_failure=True)
    result = collection.collect(plan, options)
    assert result["execution_ok"] is not crash
    assert len(calls) == (1 if crash else 5)
    assert result["status"] == "exploratory"
    assert result["accuracy_failures"] == len(calls)
    saved = read_record(tmp_path / "manifest.json")["plan"]
    assert saved["exploratory"]
    assert saved["options"]["allow_accuracy_failure"]
    assert len(attempts(tmp_path)) == len(calls)
