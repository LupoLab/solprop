# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the exploratory A/B mode on synthetic receipts and a small case."""

import json
import math
import sys

import numpy as np
import pytest
from benchmarks import __main__ as cli
from benchmarks import ab
from benchmarks.records import read_record


def receipt(identifier, arm, round_index, seconds, *, work=(10, 1)):
    return {
        "id": identifier,
        "arm": arm,
        "round": round_index,
        "status": "ok",
        "repetitions": [
            {"seconds": value, "accepted": work[0], "rejected": work[1]}
            for value in seconds
        ],
    }


def test_build_order_alternates_between_rounds():
    assert ab.arm_order(0) == ("baseline", "candidate")
    assert ab.arm_order(1) == ("candidate", "baseline")
    assert ab.arm_order(2) == ab.arm_order(0)


def test_summary_pairs_rounds_and_reports_spread_and_fields(tmp_path):
    rows = [
        {"id": "a", "configuration": {"case": "x", "stepper": "rk", "threads": 1}},
        {"id": "b", "configuration": {"case": "y", "stepper": "rk", "threads": 1}},
    ]
    samples = [
        receipt("a", "baseline", 0, [1.0, 1.2, 1.1]),
        receipt("a", "candidate", 0, [0.9, 0.95, 1.0]),
        receipt("a", "baseline", 1, [1.0]),
        receipt("a", "candidate", 1, [1.05]),
        receipt("a", "baseline", 2, [2.0]),
        receipt("a", "candidate", 2, [1.6]),
        receipt("b", "baseline", 0, [1.0]),
        receipt("b", "candidate", 0, [1.0], work=(11, 1)),
        {"id": "b", "arm": "candidate", "round": 1, "status": "failed", "error": "e"},
        receipt("b", "baseline", 1, [1.0]),
    ]
    fields = tmp_path / "fields"
    for arm, scale in (("baseline", 1.0), ("candidate", 1.0 + 1e-12)):
        (fields / arm).mkdir(parents=True)
        np.save(fields / arm / "a.npy", scale * np.ones((2, 3), complex))
    np.save(fields / "baseline" / "b.npy", np.ones(3))
    np.save(fields / "candidate" / "b.npy", np.ones(4))
    first, second = ab.summarise(rows, samples, fields)
    assert first["rounds"] == 3
    assert first["ratios"] == pytest.approx([0.95 / 1.1, 1.05, 0.8])
    assert first["ratio"] == pytest.approx(0.95 / 1.1)
    assert (first["ratio_min"], first["ratio_max"]) == pytest.approx((0.8, 1.05))
    assert first["baseline_seconds"] == pytest.approx(1.1)
    assert first["same_work"]
    assert first["field_difference"] == pytest.approx(1e-12, rel=1e-3)
    assert second["rounds"] == 1
    assert not second["same_work"]
    assert second["failures"] == ["e"]
    # Fields on different grids have no difference, and the record stays valid JSON.
    assert second["field_difference"] is None
    json.dumps(second, allow_nan=False)


def test_field_difference_without_fields_or_with_a_zero_reference(tmp_path):
    assert ab.field_difference(tmp_path / "a.npy", tmp_path / "b.npy") is None
    np.save(tmp_path / "a.npy", np.zeros(3))
    np.save(tmp_path / "b.npy", np.full(3, 2.0))
    assert ab.field_difference(tmp_path / "a.npy", tmp_path / "b.npy") == (
        pytest.approx(2 * math.sqrt(3))
    )


def test_summary_output_states_ratio_direction_without_a_verdict(capsys):
    ab.show_summary(
        [
            {
                "case": "x",
                "stepper": "rk",
                "threads": 1,
                "baseline_seconds": 1.0,
                "candidate_seconds": None,
                "ratio": None,
                "field_difference": None,
                "same_work": False,
                "failures": ["boom"],
            },
            {
                "case": "y",
                "stepper": "rk",
                "threads": 1,
                "baseline_seconds": 2.0,
                "candidate_seconds": 1.0,
                "ratio": 0.5,
                "ratio_min": 0.4,
                "ratio_max": 0.6,
                "field_difference": 0.0,
                "same_work": True,
                "failures": [],
            },
        ]
    )
    text = capsys.readouterr().out
    assert "failed: boom" in text
    assert "0.400-0.600" in text
    assert "Geometric mean candidate/baseline: 0.5000" in text
    assert "no adoption verdict" in text


def test_invalid_options_and_existing_output_are_refused(tmp_path, capsys):
    base = ["ab", "--baseline-python", sys.executable, "--threads", "1"]
    assert cli.main([*base, "--case", "smoke", "--rounds", "0"]) == 2
    assert (
        cli.main([*base, "--case", "kerr_beam_192", "--stepper", "rk", "--out", "x"])
        == 2
    )
    (tmp_path / "kept").write_text("x")
    assert cli.main([*base, "--case", "smoke", "--out", str(tmp_path)]) == 2
    errors = capsys.readouterr().err
    assert "must be positive" in errors
    assert "no configuration in the selection" in errors
    assert "output already exists" in errors


def test_missing_receipts_are_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "launch_worker", lambda *_args: ([], 0.0))
    monkeypatch.setattr(ab, "guard", lambda *_args: {"quiet": False})
    out = tmp_path / "run"
    status = cli.main(
        [
            "ab",
            "--case",
            "smoke",
            "--threads",
            "1",
            "--rounds",
            "1",
            "--baseline-python",
            sys.executable,
            "--out",
            str(out),
            "--json",
        ]
    )
    assert status == 1
    results = read_record(out / "results.json")
    assert results["busy_samples"] == 2
    assert results["summary"][0]["failures"] == ["no receipt from the worker"] * 2


@pytest.mark.child_runs_auto
@pytest.mark.medium
def test_small_case_runs_both_builds_in_alternating_order(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(ab, "guard", lambda *_args: {"quiet": True})
    out = tmp_path / "run"
    status = cli.main(
        [
            "ab",
            "--case",
            "smoke",
            "--backend",
            "numpy_ref",
            "--threads",
            "1",
            "--length",
            "0.5",
            "--rounds",
            "2",
            "--repeats",
            "1",
            "--baseline-python",
            sys.executable,
            "--out",
            str(out),
        ]
    )
    text = capsys.readouterr().out
    assert status == 0, text
    assert "A/A control" in text
    lines = [line for line in text.splitlines() if line.startswith("round")]
    assert [line.split(":")[0].split()[-1] for line in lines] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    results = read_record(out / "results.json")
    assert len(results["samples"]) == 4
    assert all(len(item["repetitions"]) == 1 for item in results["samples"])
    assert all(
        rep["z"] == pytest.approx(0.5e-4)
        for item in results["samples"]
        for rep in item["repetitions"]
    )
    (row,) = results["summary"]
    assert row["rounds"] == 2
    assert row["same_work"]
    # The same interpreter and build repeat the same operations exactly.
    assert row["field_difference"] == 0.0
    assert {item["python_executable"] for item in results["provenance"]["baseline"]}
    assert read_record(out / "plan.json")["harness_files"]


def test_each_arm_runs_its_own_backend():
    values = {"case": "x", "backend": "native"}
    assert ab.arm_configuration(values, "numpy_ref") == {
        "case": "x",
        "backend": "numpy_ref",
    }
    assert ab.arm_configuration(values, "native")["backend"] == "native"


def test_each_arm_takes_its_own_environment(tmp_path, monkeypatch):
    """A calibrated arm against the built-in table (SEP 46): only the arm's own
    variables differ, over the threading preset, and the plan records them."""
    seen = []

    def launch(python, request, environment, *_args):
        seen.append(environment.get("SOLPROP_CALIBRATION"))
        return [], 0.0

    monkeypatch.setattr(ab, "launch_worker", launch)
    monkeypatch.setattr(ab, "guard", lambda *_args: {"quiet": True})
    out = tmp_path / "run"
    cli.main(
        [
            "ab",
            "--case",
            "smoke",
            "--threads",
            "1",
            "--rounds",
            "1",
            "--baseline-python",
            sys.executable,
            "--baseline-env",
            "SOLPROP_CALIBRATION=off",
            "--candidate-env",
            "SOLPROP_CALIBRATION=use",
            "--calibrate",
            "none",
            "--out",
            str(out),
        ]
    )
    assert sorted(seen) == ["off", "use"]
    plan = read_record(out / "plan.json")
    assert plan["environments"] == {
        "baseline": {"SOLPROP_CALIBRATION": "off"},
        "candidate": {"SOLPROP_CALIBRATION": "use"},
    }


def test_an_arms_environment_takes_name_value_pairs():
    assert ab.arm_environment(["A=1", "B="]) == {"A": "1", "B": ""}
    with pytest.raises(ab.SuiteError, match="NAME=VALUE"):
        ab.arm_environment(["broken"])


def test_each_build_is_calibrated_once_unless_off(tmp_path, monkeypatch):
    """SEP 46 section 6.13: a series calibrates each build first, once per
    interpreter, with its arm's variables; an arm held off calibrates nothing."""
    from benchmarks import runner  # noqa: PLC0415

    ran = []

    class Completed:
        returncode = 0
        stdout = "calibration (quick): published in 1.0 s\n"
        stderr = ""

    def run(command, **kwargs):
        ran.append((command, kwargs["env"].get("A")))
        return Completed()

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.delenv("SOLPROP_CALIBRATION", raising=False)
    found = runner.calibrate_builds(
        {"baseline": "/py/a", "candidate": "/py/a"},
        "quick",
        tmp_path,
        {"baseline": {"A": "1"}},
    )
    assert [command for command, _ in ran] == [
        ["/py/a", "-m", "solprop", "calibrate", "--quick"]
    ]
    assert found["candidate"] == found["baseline"]
    assert (tmp_path / "calibration" / "baseline.txt").exists()
    held = runner.calibrate_builds(
        {"baseline": "/py/b"},
        "quick",
        tmp_path,
        {"baseline": {"SOLPROP_CALIBRATION": "off"}},
    )
    assert held == {"baseline": {"level": "off"}}
    assert runner.calibrate_builds({"baseline": "/py/c"}, "none", tmp_path) == {}
    assert len(ran) == 1
    # The command's default level takes no flag: it has no --standard.
    runner.calibrate_builds({"baseline": "/py/d"}, "standard", tmp_path)
    assert ran[-1][0] == ["/py/d", "-m", "solprop", "calibrate"]
    assert runner.worker_calibration({"SOLPROP_CALIBRATION": "OFF"}) == {
        "SOLPROP_CALIBRATION": "off"
    }
    assert runner.worker_calibration({}) == {"SOLPROP_CALIBRATION": "use"}
