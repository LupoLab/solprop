# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the tracking command's plan, hardware identity and history (SEP 38)."""

import argparse
import csv
import importlib.util
import socket
import sys

import pytest
from benchmarks import __main__ as cli
from benchmarks import history, runner, system, telemetry, track
from benchmarks.records import SuiteError, read_record
from benchmarks.registry import CASES, select


@pytest.fixture
def sixteen_cpus(monkeypatch):
    """Plan for a 16-CPU allocation, whatever this runner has."""
    monkeypatch.setattr(system, "eligible_cpus", lambda: tuple(range(16)))


@pytest.mark.usefixtures("sixteen_cpus")
def test_the_default_plan_runs_each_case_at_its_declared_widths():
    rows, skipped = runner.plan_rows(select(), ("default",), None, 8)
    assert not skipped
    widths = {}
    for row in rows:
        widths.setdefault(row["configuration"]["case"], []).append(
            row["configuration"]["threads"]
        )
    assert widths["rdw"] == [1, 8]
    assert widths["ctc"] == [8]
    assert widths["kerr_beam_192"] == [1]
    # One-thread rows first, so a run's widths are not interleaved.
    threads = [row["configuration"]["threads"] for row in rows]
    assert threads == sorted(threads)
    assert runner.estimate(rows, 3) is not None


@pytest.mark.usefixtures("sixteen_cpus")
def test_named_widths_and_steppers_apply_to_every_case_and_explain_skips():
    cases = (CASES["rdw"], CASES["kerr_beam_192"])
    rows, skipped = runner.plan_rows(cases, ("default", "rk"), "2", 4)
    assert [row["id"] for row in rows] == [
        "rdw.default.t2",
        "rdw.rk.t2",
        "kerr_beam_192.default.t2",
    ]
    assert skipped == ["kerr_beam_192 rk: a fixed-step case runs its own schedule only"]
    assert runner.estimate(rows, 1) is None
    assert track.steppers("rk, default,rk") == ("rk", "default")
    with pytest.raises(SuiteError, match="unknown stepper"):
        track.steppers("adams")
    with pytest.raises(SuiteError, match="must fit"):
        system.widths("1", 10**6)
    with pytest.raises(SuiteError, match="positive"):
        system.widths("0", 1)


@pytest.mark.parametrize(
    ("model", "cores", "expected"),
    [
        ("Apple M5 Max", 18, "apple-m5-max-18c"),
        ("AMD EPYC 7763 64-Core Processor", 16, "amd-epyc-7763-16c"),
        ("Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz", 36, "intel-xeon-gold-6240-36c"),
        ("", 1, "unknown-cpu-1c"),
    ],
)
def test_the_hardware_slug_names_the_cpu_model_and_cores(model, cores, expected):
    assert system.slug(model, cores) == expected


def test_the_hardware_record_names_no_host():
    record = system.hardware(1, "aws m8g.4xlarge")
    assert record["instance_type"] == "aws m8g.4xlarge"
    # The slug counts the cores the run uses, then names the instance family.
    assert record["slug"].endswith("-1c-aws-m8g")
    text = repr(record).lower()
    assert socket.gethostname().split(".")[0].lower() not in text


def result_row(case="rdw", threads=1, median=2.0):
    return {
        "case": case,
        "revision": CASES[case].revision,
        "stepper": "default",
        "threads": threads,
        "threading": "passive",
        "status": "ok",
        "busy": False,
        "median_s": median,
        "min_s": median * 0.99,
        "max_s": median * 1.01,
        "repeats": 3,
        "accepted": 100,
        "rejected": 2,
        "same_steps": True,
        "repeat_difference": 0.0,
        "native_threads": threads,
        "consistent": True,
        "solprop_path": str(runner.ROOT / "src" / "solprop" / "__init__.py"),
        "source_evaluations": 104,
        "peak_rss_gib": 0.5,
        "construction_s": 3.0,
        "warmup_s": 0.1,
        "integrator": "Multistep(exponential)",
        "rtol": 1e-6,
        "atol": 1e-10,
        "noise": "input",
        "finite": True,
    }


def test_history_rows_append_read_and_summarise(tmp_path):
    machine = {
        "slug": "test-cpu-2c",
        "cpu_model": "Test CPU",
        "instance_type": None,
        "system": "Linux",
        "machine": "x86_64",
        "numa_domains": 1,
    }
    provenance = {
        "packages": {"solprop": "1.0"},
        "build_info": {"package_version": "1.0", "build_profile": "local"},
    }
    path = tmp_path / "test-cpu-2c.csv"
    first = track.history_rows(
        [result_row(), result_row("ctc", 2, 10.0)],
        machine,
        {"revision": "a" * 40},
        provenance,
    )
    assert all(tuple(row) == history.COLUMNS for row in first)
    history.append(path, first)
    later = [
        {**row, "date": "2099-01-01T00:00Z", "commit": "b" * 12, "median_s": value}
        for row, value in zip(first, ("1.0", "5.0"), strict=True)
    ]
    history.append(path, later)
    rows = history.read(path)
    assert len(rows) == 4
    assert rows[0]["native_build"] == "local"
    assert rows[0]["commit"] == "a" * 12
    summary = history.summary(rows)
    assert [
        (row["threads"], row["ratio_first"], row["ratio_previous"]) for row in summary
    ] == [("1", 1.0, 1.0), ("2", 1.0, 1.0), ("1", 0.5, 0.5), ("2", 0.5, 0.5)]
    # A changed tolerance is a new configuration, not a speed-up.
    changed = [{**row, "date": "2100-01-01T00:00Z", "rtol": "1e-05"} for row in later]
    history.append(path, changed)
    assert history.summary(history.read(path))[-1]["shared_previous"] == 0
    with pytest.raises(SuiteError, match="exactly the declared columns"):
        history.append(tmp_path / "new.csv", [{"date": "x"}])


def test_older_history_files_gain_new_columns_and_newer_ones_are_refused(tmp_path):
    path = tmp_path / "old.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "case", "median_s"])
        writer.writerow(["2026-01-01T00:00Z", "rdw", "2.0"])
    row: dict[str, object] = dict.fromkeys(history.COLUMNS, "x")
    history.append(path, [row])
    rows = history.read(path)
    assert tuple(rows[0]) == history.COLUMNS
    assert (rows[0]["case"], rows[0]["commit"]) == ("rdw", "")
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow([*history.COLUMNS, "future"])
    with pytest.raises(SuiteError, match="later version"):
        history.append(path, [row])


def namespace(**values):
    defaults = {
        "backend": "native",
        "allow_dirty": False,
        "repeats": 3,
        "device": False,
    }
    return argparse.Namespace(**(defaults | values))


@pytest.mark.usefixtures("sixteen_cpus")
def test_the_history_takes_only_a_complete_default_run_of_committed_code():
    rows, _ = runner.plan_rows(select(), ("default",), None, 16)
    clean = {"revision": "a" * 40, "dirty": False}
    track.refuse_partial(namespace(), rows, 16, clean)
    for args, source, message in (
        (namespace(backend="numpy_ref"), clean, "native backend only"),
        (namespace(), {"revision": None, "dirty": None}, "source state is unknown"),
        (namespace(), {**clean, "dirty": True}, "uncommitted changes"),
        (namespace(repeats=2), clean, "three repeats"),
    ):
        with pytest.raises(SuiteError, match=message):
            track.refuse_partial(args, rows, 16, source)
    track.refuse_partial(
        namespace(allow_dirty=True), rows, 16, {**clean, "dirty": True}
    )
    with pytest.raises(SuiteError, match="whole default plan"):
        track.refuse_partial(namespace(), rows[:-1], 16, clean)


def test_inconsistent_or_foreign_runs_are_not_recorded():
    good = [result_row()]
    assert track.admissible(good) is None
    assert "failed" in str(track.admissible([{**good[0], "status": "failed"}]))
    assert "repeats" in str(track.admissible([{**good[0], "consistent": False}]))
    source = {"revision": "a" * 40}
    native = {"build_info": {"git_commit": "a" * 40}}
    assert track.imported_elsewhere(good, native, source) is None
    stale = {"build_info": {"git_commit": "b" * 40}}
    assert "rebuild" in str(track.imported_elsewhere(good, stale, source))
    elsewhere = [{**good[0], "solprop_path": "/elsewhere/solprop/__init__.py"}]
    assert "outside" in str(track.imported_elsewhere(elsewhere, native, source))
    row = {"id": "x", "configuration": CASES["rdw"].configuration("default", 4)}
    receipt = {
        "status": "ok",
        "repetitions": [{"seconds": 1.0, "accepted": 5, "rejected": 0,
                         "source_evaluations": 6, "z": 1.0}],
        "repeat_difference": 0.0, "peak_rss_bytes": 1, "construction_seconds": 1.0,
        "backend_seconds": 0.0, "warmup_seconds": 0.0, "integrator": "RungeKutta",
        "rtol": 1e-6, "atol": 0.0, "noise": None, "finite": True,
        "provenance": {"build_info": {"runtime_threads": 2}},
    }  # fmt: skip
    # The thread budget gave the native backend fewer threads than asked for.
    assert not track.summarise(row, receipt, busy=False)["consistent"]


def test_a_machine_without_swap_counters_is_judged_by_cpu_alone():
    before = {"time": 0.0, "busy": [0.0], "swap_bytes": None}
    after = {"time": 1.0, "busy": [0.05], "swap_bytes": None}
    assert system.interval(before, after, (0,))["quiet"]


def test_a_failed_worker_is_summarised_without_statistics():
    row = {"id": "x", "configuration": CASES["rdw"].configuration("default", 1)}
    result = track.summarise(row, {"status": "failed", "error": "boom"}, busy=True)
    assert result["status"] == "failed"
    assert result["error"] == "boom"
    assert "median_s" not in result


def test_worker_timeout_and_missing_receipts_are_failures(tmp_path):
    events, _ = runner.launch_worker(
        sys.executable, {"repeats": 1}, {"PATH": ""}, tmp_path / "slow", 1e-3
    )
    assert runner.receipt(events)["error"] == "worker timeout"
    assert runner.receipt([])["error"] == "no receipt from the worker"


def test_list_and_dry_run_plan_without_running(capsys):
    assert cli.main([]) == 0
    assert cli.main(["track", "--dry-run", "--case", "rdw", "--threads", "1"]) == 0
    text = capsys.readouterr().out
    assert "configurations" in text
    assert "estimated" in text
    assert cli.main(["track", "--case", "absent", "--dry-run"]) == 2
    assert "unknown case" in capsys.readouterr().err


@pytest.mark.medium
def test_a_tracking_run_records_complete_runs_and_appends_history(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(history, "HISTORY", tmp_path / "history")
    monkeypatch.setattr(track, "guard", lambda *_args: {"quiet": True})
    out = tmp_path / "run"
    status = cli.main(
        [
            "track",
            "--case",
            "smoke",
            "--stepper",
            "default,multistep",
            "--backend",
            "numpy_ref",
            "--threads",
            "1",
            "--repeats",
            "2",
            "--out",
            str(out),
        ]
    )
    text = capsys.readouterr().out
    assert status == 0, text
    results = read_record(out / "results.json")
    assert [row["integrator"] for row in results["rows"]] == [
        "RungeKutta",
        "Multistep(exponential)",
    ]
    for row in results["rows"]:
        assert row["repeats"] == 2
        assert row["same_steps"]
        assert row["finite"]
    assert read_record(out / "plan.json")["hardware"]["slug"]
    # The history takes the native backend only.
    assert (
        cli.main(
            [
                "track",
                "--case",
                "smoke",
                "--backend",
                "numpy_ref",
                "--append-history",
                "--out",
                str(tmp_path / "refused"),
            ]
        )
        == 2
    )
    assert "native backend only" in capsys.readouterr().err


def test_threading_policies_replace_the_callers_openmp_settings():
    incoming = {"OMP_WAIT_POLICY": "ACTIVE", "OMP_PLACES": "threads", "PATH": "/bin"}
    passive = system.thread_environment(4, incoming)
    assert passive["OMP_WAIT_POLICY"] == "PASSIVE"
    assert passive["SOLPROP_NATIVE_THREADS"] == "4"
    assert "OMP_PLACES" not in passive
    assert passive["PATH"] == "/bin"
    bound = system.thread_environment(4, incoming, "bound")
    assert (bound["OMP_PROC_BIND"], bound["OMP_PLACES"]) == ("close", "cores")
    assert "OMP_WAIT_POLICY" not in bound
    assert bound["GOMP_SPINCOUNT"] == "300000"
    for policy in ("passive-spread", "spread", "spread-active"):
        spread = system.thread_environment(4, incoming, policy)
        assert (spread["OMP_PROC_BIND"], spread["OMP_PLACES"]) == ("spread", "cores")
    assert (
        system.thread_environment(4, incoming, "passive-spread")["GOMP_SPINCOUNT"]
        == "0"
    )
    assert (
        system.thread_environment(4, incoming, "spread-active")["OMP_WAIT_POLICY"]
        == "ACTIVE"
    )
    # Every policy names its waiting, so the native backend's own default never applies.
    waiting = {"OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME"}
    for policy in system.THREADING:
        assert waiting & set(system.thread_environment(4, {}, policy)), policy
    with pytest.raises(SuiteError, match="unknown threading"):
        system.thread_environment(4, incoming, "spin")


@pytest.mark.medium
@pytest.mark.skipif(
    importlib.util.find_spec("solprop_native") is None,
    reason="the native backend is not installed",
)
def test_a_native_tracking_run_records_its_build(tmp_path, monkeypatch):
    monkeypatch.setattr(track, "guard", lambda *_args: {"quiet": True})
    out = tmp_path / "native"
    arguments = ["track", "--case", "smoke", "--threads", "1", "--repeats", "1"]
    assert cli.main([*arguments, "--out", str(out)]) == 0
    results = read_record(out / "results.json")
    assert results["provenance"]["build_info"]["package_version"]
    assert results["rows"][0]["native_threads"] == 1
    assert (out / "raw" / "smoke.default.t1" / "environment.json").exists()


def test_memory_domains_count_from_the_topology_and_separate_series():
    assert system.numa_domains(None) == 1
    rows = [{"numa_nodes": [0]}, {"numa_nodes": [0]}, {"numa_nodes": [3]}]
    assert system.numa_domains(rows) == 2
    row: dict[str, object] = dict.fromkeys(history.COLUMNS, "1")
    assert history.comparable(row) != history.comparable({**row, "numa_domains": "8"})


def test_a_native_build_counts_as_current_unless_native_changed():
    assert not system.native_changed(runner.ROOT, "HEAD", "HEAD")
    assert system.native_changed(runner.ROOT, "0" * 40, "HEAD")


@pytest.mark.usefixtures("sixteen_cpus")
def test_a_device_plan_runs_each_case_once_and_its_rows_stand_apart():
    rows, _ = runner.plan_rows(select(), ("default",), None, 8, device=True)
    assert len(rows) == len(select())
    assert all(row["id"].endswith(".device") for row in rows)
    assert {row["configuration"]["threads"] for row in rows} == {8}
    assert all(row["configuration"]["device"] for row in rows)
    assert runner.estimate(rows, 3) is None
    clean = {"revision": "a" * 40, "dirty": False}
    track.refuse_partial(namespace(device=True), rows, 8, clean)
    with pytest.raises(SuiteError, match="whole default plan"):
        track.refuse_partial(namespace(), rows, 8, clean)
    receipt = {
        "status": "ok",
        "repetitions": [{"seconds": 1.0, "accepted": 5, "rejected": 0,
                         "source_evaluations": 6, "z": 1.0}],
        "repeat_difference": 0.0, "peak_rss_bytes": 1, "device_bytes": 2**31,
        "construction_seconds": 1.0, "backend_seconds": 0.0, "warmup_seconds": 0.0,
        "integrator": "RungeKutta", "rtol": 1e-6, "atol": 0.0, "noise": None,
        "finite": True, "provenance": {"build_info": {"runtime_threads": 1}},
    }  # fmt: skip
    # The host threads of a device build do not set its pace, so are not checked.
    summary = track.summarise(rows[0], receipt, busy=False)
    assert summary["consistent"]
    assert summary["threads"] == "device"
    assert summary["device_gib"] == 2.0


@pytest.mark.parametrize(
    ("nvidia", "amd", "model"),
    [
        ("NVIDIA A40\nNVIDIA A40\n", None, "NVIDIA A40"),
        (None, "device,Card Series,Card Model\ncard0,MI325X,0x74a5\n", "AMD MI325X"),
        (
            None,
            "device,Card Series\ncard0,AMD Instinct MI300X\n",
            "AMD Instinct MI300X",
        ),
        (None, None, None),
        ("", "device,Card Model\ncard0,0x74a5\n", None),
    ],
)
def test_the_gpu_is_named_from_the_vendor_query(nvidia, amd, model):
    assert telemetry.gpu_model(nvidia, amd) == model


def test_a_device_slug_names_the_gpu_and_any_cloud_family():
    assert system.device_slug("NVIDIA H200") == "nvidia-h200"
    assert system.device_slug("NVIDIA A40", "aws g5.xlarge") == "nvidia-a40-aws-g5"
