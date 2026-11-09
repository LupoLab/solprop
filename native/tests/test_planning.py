# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A workspace's FFT planning: the policy, a capture and a profile (solprop SEP 46
section 6.7 and section 7 item 9)."""

from __future__ import annotations

from typing import cast

import pytest
from conftest import PhaseACase

import solprop_native as native
from solprop_native import _core
from solprop_native.backend import NativeBackend

if native.build_info().device_backend != "none":  # pragma: no cover - device builds
    pytest.skip("device transforms keep no FFTW wisdom", allow_module_level=True)

POLICY = {"rows": "measure", "threaded_rows": "estimate", "planes": "measure"}


def _plan(case: PhaseACase, planning: dict[str, object] | None) -> dict[str, object]:
    """Build the case's workspace under `planning`; return what its planning found."""
    problem, grid = case
    backend = NativeBackend(problem, grid=grid, fft_planning=planning)
    try:
        found = dict(backend._native_state.fft_planning)  # pyright: ignore[reportPrivateUsage]
        found["metadata"] = cast(dict[str, object], backend.metadata()["calibration"])[
            "fft_planning"
        ]
        return found
    finally:
        backend.close()


def _entries(wisdom: str | None) -> int:
    """Return the plans a wisdom text holds, past its preamble."""
    return 0 if wisdom is None else max(0, len(wisdom.strip().splitlines()) - 2)


def _capture(case: PhaseACase) -> dict[str, str]:
    found = _plan(
        case, {"mode": "capture", "capture_effort": "measure", "time_limit": 1.0}
    )
    return cast(dict[str, str], found["captured"])


def test_without_a_planning_the_policy_plans(phase_a_case: PhaseACase) -> None:
    found = _plan(phase_a_case, None)
    assert found["groups"] == {}
    assert found["metadata"] == "policy"


def test_a_capture_keeps_each_plans_wisdom_alone(phase_a_case: PhaseACase) -> None:
    found = _plan(
        phase_a_case,
        {"mode": "capture", "capture_effort": "measure", "time_limit": 1.0},
    )
    captured = cast(dict[str, str], found["captured"])
    groups = cast(dict[str, str], found["groups"])
    assert captured
    assert set(captured) <= set(groups)
    assert set(groups.values()) <= {"rows", "threaded_rows", "planes"}
    for key, text in captured.items():
        assert text.startswith("(fftw-3")
        assert key.split(":")[-1].startswith("t")
    # Nothing a capture learnt outlives its workspace.
    assert _entries(_core.export_fftw_wisdom()) == 0
    # A capture at estimate keeps nothing.
    estimated = _plan(phase_a_case, {"mode": "capture", "capture_effort": "estimate"})
    assert estimated["captured"] == {}


def test_a_profile_builds_its_plans_from_their_wisdom_and_forgets(
    phase_a_case: PhaseACase,
) -> None:
    wisdom = _capture(phase_a_case)
    efforts = {"rows": "measure", "threaded_rows": "measure", "planes": "measure"}
    found = _plan(
        phase_a_case, {"mode": "profile", "efforts": efforts, "wisdom": wisdom}
    )
    assert found["missing"] == []
    assert found["metadata"] == {"plans": len(found["groups"]), "missing": []}  # type: ignore[arg-type]
    assert _entries(_core.export_fftw_wisdom()) == 0


def test_stale_wisdom_never_reaches_a_profiled_workspace(
    phase_a_case: PhaseACase,
) -> None:
    # Wisdom an earlier run or a file left in the process (SEP 46 section 7 item 9):
    # a profile forgets it before planning, and its estimate plans are FFTW's own.
    wisdom = _capture(phase_a_case)
    for text in wisdom.values():
        assert _core.import_fftw_wisdom(text)
    assert _entries(_core.export_fftw_wisdom()) > 0
    efforts = {"rows": "estimate", "threaded_rows": "estimate", "planes": "estimate"}
    found = _plan(phase_a_case, {"mode": "profile", "efforts": efforts, "wisdom": {}})
    assert found["missing"] == []
    assert _entries(_core.export_fftw_wisdom()) == 0


def test_two_workspaces_plan_the_same_in_either_order(phase_a_case: PhaseACase) -> None:
    first = _plan(phase_a_case, {"mode": "capture", "capture_effort": "measure"})
    _plan(phase_a_case, {"mode": "profile", "efforts": POLICY, "wisdom": {}})
    second = _plan(phase_a_case, {"mode": "capture", "capture_effort": "measure"})
    assert first["groups"] == second["groups"]
    assert set(cast(dict[str, str], first["captured"])) == set(
        cast(dict[str, str], second["captured"])
    )


def test_wisdom_that_no_longer_builds_a_plan_falls_back_and_is_recorded(
    phase_a_case: PhaseACase,
) -> None:
    wisdom = _capture(phase_a_case)
    broken = dict.fromkeys(wisdom, "(not wisdom)")
    found = _plan(
        phase_a_case,
        {
            "mode": "profile",
            "efforts": dict.fromkeys(POLICY, "measure"),
            "wisdom": broken,
        },
    )
    assert sorted(cast(list[str], found["missing"])) == sorted(broken)
    # Another FFTW's or another machine's wisdom imports but builds no plan here.
    foreign = {
        key: "(fftw-3.3.11 fftw_wisdom #x3c273403 #x192df114 #x4d08727c #xe98e9b9d\n)\n"
        for key in wisdom
    }
    found = _plan(
        phase_a_case,
        {
            "mode": "profile",
            "efforts": dict.fromkeys(POLICY, "measure"),
            "wisdom": foreign,
        },
    )
    assert sorted(cast(list[str], found["missing"])) == sorted(foreign)


@pytest.mark.parametrize(
    ("planning", "message"),
    [
        ({"mode": "sometimes"}, "legacy, profile or capture"),
        (
            {"mode": "profile", "efforts": {"rows": "quickly"}},
            "estimate, measure or patient",
        ),
        ({"mode": "profile", "efforts": {"columns": "measure"}}, "rows, threaded_rows"),
        (
            {"mode": "capture", "capture_effort": "exhaustive"},
            "estimate, measure or patient",
        ),
        ({"mode": 3}, "wrong types"),
    ],
)
def test_a_malformed_planning_is_refused(
    phase_a_case: PhaseACase, planning: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _plan(phase_a_case, planning)
