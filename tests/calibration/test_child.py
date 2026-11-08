# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A calibration child's protocol, and one real calibration (SEP 46 §6.2)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from solprop.calibration import child, problems
from solprop.calibration.controller import calibrate


def _two_cpus() -> list[int] | None:
    """Return two CPUs of this process's mask, or ``None`` without masks."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    return sorted(os.sched_getaffinity(0))[:2]


def _serve(*requests: dict[str, object]) -> list[dict[str, object]]:
    replies = io.StringIO()
    lines = "\n".join(json.dumps(request) for request in requests) + "\n\n"
    child.serve(io.StringIO(lines), replies)
    return [json.loads(line) for line in replies.getvalue().splitlines()]


def test_a_child_prepares_times_and_probes_regions() -> None:
    record: dict[str, object] = {"op": "time", "problem": "gnlse", "repeats": 2}
    replies = _serve(
        {"op": "prepare", "problems": {"gnlse": 0.5}},
        record | {"record": None},
        {"op": "regions", "elements": 100, "repeats": 5},
        {"op": "sizing", "record": None},
        {"op": "colour"},
        {"op": "time", "problem": "missing", "repeats": 1},
        {"op": "exit"},
        {"op": "prepare", "problems": {}},
    )
    prepared = replies[0]["ok"]
    assert isinstance(prepared, dict)
    assert prepared["problems"]["gnlse"]["shape"][-1] > 0  # pyright: ignore[reportIndexIssue, reportUnknownMemberType]
    assert set(prepared) >= {"process", "execution", "builtin", "cpus"}
    timed = replies[1]["ok"]
    assert isinstance(timed, float)
    assert timed > 0
    regions = replies[2]["ok"]
    assert isinstance(regions, dict)
    assert set(regions) == {"copy", "empty"}
    sizing = replies[3]["ok"]
    assert isinstance(sizing, dict)
    assert any(key.startswith("kernels.stage.") for key in sizing["gnlse"])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert replies[4] == {"error": "ValueError: unknown request 'colour'"}
    assert "KeyError" in str(replies[5]["error"])
    # Nothing is answered after exit.
    assert replies[6] == {"ok": True}
    assert len(replies) == 7


def test_a_child_times_under_a_given_record() -> None:
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        BUILTIN_EXECUTION,
    )

    serial = BUILTIN_EXECUTION.updated(
        {"stage_threshold": 1 << 40, "stage_full": 1 << 40}
    )
    replies = _serve(
        {"op": "prepare", "problems": {"gnlse": 0.5}},
        {"op": "time", "problem": "gnlse", "repeats": 1, "record": serial.native()},
    )
    assert isinstance(replies[1]["ok"], float)


def test_every_calibration_problem_builds() -> None:
    for name in problems.PROBLEMS:
        simulation = problems.build(name)
        assert simulation.settings.backend == "native"
    with pytest.raises(KeyError):
        problems.build("unknown")


def test_pinning_sets_the_affinity_mask() -> None:
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("no affinity masks on this platform")
    before = os.sched_getaffinity(0)
    try:
        first = min(before)
        child.pin([first])
        assert os.sched_getaffinity(0) == {first}
        child.pin([])
        assert os.sched_getaffinity(0) == {first}
    finally:
        os.sched_setaffinity(0, before)


def test_the_child_module_serves_on_its_standard_streams(tmp_path: Path) -> None:
    cpus = (
        sorted(os.sched_getaffinity(0))[:2] if hasattr(os, "sched_getaffinity") else []
    )
    completed = subprocess.run(
        [sys.executable, "-m", "solprop.calibration.child", ",".join(map(str, cpus))],
        input=json.dumps({"op": "exit"}) + "\n",
        capture_output=True,
        text=True,
        check=True,
        env=os.environ | {"SOLPROP_CALIBRATION": "off"},
    )
    assert json.loads(completed.stdout) == {"ok": True}
    child_main = child.main  # the in-process entry point takes the same argument
    assert callable(child_main)


@pytest.mark.medium
def test_a_quick_calibration_runs_its_children_and_publishes_or_declines(
    tmp_path: Path,
) -> None:
    """A real quick calibration of two CPUs: published, or inconclusive on a busy
    runner; either way within its deadline plus the children's shutdown."""
    cpus = (
        sorted(os.sched_getaffinity(0))[:2]
        if hasattr(os, "sched_getaffinity")
        else None
    )
    environ = os.environ | {"SOLPROP_CACHE_DIR": str(tmp_path)}
    report = calibrate("quick", 60.0, environ=environ, cpus=cpus)
    assert report.status in {"published", "inconclusive"}
    assert report.seconds < 60.0 + 10.0
    if report.status == "published":
        assert report.path is not None and report.path.exists()
        assert {"threads", "wait"} <= set(report.chosen)


def test_the_childs_entry_point_keeps_its_protocol_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = io.StringIO()
    pinned: list[list[int]] = []
    monkeypatch.setattr(child, "pin", pinned.append)
    monkeypatch.setattr(child.os, "dup", lambda descriptor: 99)
    monkeypatch.setattr(child.os, "dup2", lambda source, target: None)
    monkeypatch.setattr(child.os, "fdopen", lambda *args, **kwargs: replies)
    monkeypatch.setattr(child.sys, "stdin", io.StringIO('{"op": "exit"}\n'))
    monkeypatch.setattr(child.sys, "stdout", child.sys.stdout)
    monkeypatch.setattr(child.warnings, "simplefilter", lambda action: None)
    assert child.main(["3,4"]) == 0
    assert pinned == [[3, 4]]
    assert json.loads(replies.getvalue()) == {"ok": True}
    monkeypatch.setattr(child.sys, "argv", ["child"])
    monkeypatch.setattr(child.sys, "stdin", io.StringIO(""))
    assert child.main() == 0
    assert pinned[-1] == []


def test_a_child_captures_each_plans_wisdom_and_times_under_a_planning() -> None:
    replies = _serve(
        {"op": "prepare", "problems": {"gnlse": 1.0}},
        {"op": "capture", "effort": "measure", "limit": 0.5, "record": None},
    )
    captured = replies[1]["ok"]
    assert isinstance(captured, dict)
    wisdom = cast(dict[str, str], captured["wisdom"])
    groups = cast(dict[str, str], captured["groups"])
    assert wisdom
    assert set(wisdom) <= set(groups)
    assert all(text.startswith("(fftw-3") for text in wisdom.values())
    assert set(cast(dict[str, list[str]], captured["problems"])) == {"gnlse"}
    planning = {
        "mode": "profile",
        "efforts": {"rows": "measure", "threaded_rows": "measure", "planes": "measure"},
        "wisdom": wisdom,
    }
    timed = _serve(
        {"op": "prepare", "problems": {"gnlse": 1.0}},
        {
            "op": "time",
            "problem": "gnlse",
            "repeats": 1,
            "record": None,
            "planning": planning,
        },
    )
    assert isinstance(timed[1]["ok"], float)
