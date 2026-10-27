# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the tracking command's plan, hardware identity and history (SEP 38)."""

import csv
import socket
import sys

import pytest
from benchmarks import __main__ as cli
from benchmarks import history, runner, system, track
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
    record = system.hardware(1, "aws c8a.4xlarge")
    assert record["instance_type"] == "aws c8a.4xlarge"
    assert record["slug"].endswith(f"-{system.physical_cores()}c")
    text = repr(record).lower()
    assert socket.gethostname().split(".")[0].lower() not in text


def result_row(case="rdw", threads=1, median=2.0):
    return {
        "case": case,
        "revision": CASES[case].revision,
        "stepper": "default",
        "threads": threads,
        "status": "ok",
        "busy": False,
        "median_s": median,
        "min_s": median * 0.99,
        "max_s": median * 1.01,
        "repeats": 3,
        "accepted": 100,
        "rejected": 2,
        "same_steps": True,
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
    assert [(row["threads"], row["ratio"]) for row in summary] == [
        ("1", 1.0),
        ("2", 1.0),
        ("1", 0.5),
        ("2", 0.5),
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(["date"])
    with pytest.raises(SuiteError, match="columns differ"):
        history.append(path, first)
    with pytest.raises(SuiteError, match="exactly the declared columns"):
        history.append(tmp_path / "new.csv", [{"date": "x"}])


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
