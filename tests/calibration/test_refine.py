# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stage 2: refining one simulation and reusing it by descriptor (SEP 46 §7.12)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import solprop_native  # pyright: ignore[reportMissingImports]
import solprop_native.backend  # pyright: ignore[reportMissingImports]
from solprop_native._profile import (  # pyright: ignore[reportMissingImports]
    choose,
)

from solprop.calibration import _native, grain, problems, refine
from solprop.calibration.refine import RefinementReport, _Windows
from solprop.exceptions import CalibrationError


@pytest.fixture
def profiled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A process that found this machine's (empty) store, as SOLPROP_CALIBRATION=use."""
    store = tmp_path / "store"
    monkeypatch.setenv("SOLPROP_CACHE_DIR", str(store))
    found = choose(
        {"SOLPROP_CALIBRATION": "use", "SOLPROP_CACHE_DIR": str(store)},
        packages=_native.packages(),
    )
    monkeypatch.setattr(solprop_native, "PROFILE", found)
    return store


def _script(
    monkeypatch: pytest.MonkeyPatch,
    speed: Callable[[str], float],
    drift: Callable[[str], float] = lambda name: 0.0,
) -> None:
    """Time each window by its record's medium width; perturb its fields by `drift`."""

    def name(window: _Windows) -> str:
        record = window.backend.metadata()["calibration"]
        assert isinstance(record, dict)
        medium = record["workspace"]["medium_threads"]["applied"]  # type: ignore[index]
        return f"medium {medium}"

    def time(window: _Windows, repeats: int) -> float:
        return 1e-3 * speed(name(window))

    def fields(window: _Windows) -> list[np.ndarray]:
        return [np.ones(4) * (1.0 + drift(name(window)))]

    monkeypatch.setattr(_Windows, "time", time)
    monkeypatch.setattr(_Windows, "fields", fields)
    monkeypatch.setattr(
        refine,
        "_candidates",
        lambda problem, sizing, base, threads: [
            grain.Candidate("medium", "medium 2", {"medium_threads": 2}),
            grain.Candidate("medium", "medium 8", {"medium_threads": 8}),
        ],
    )


def test_a_refinement_is_stored_and_a_later_run_applies_it(
    profiled: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script(monkeypatch, lambda name: 0.8 if name == "medium 2" else 1.0)
    simulation = problems.build("gnlse")
    lines: list[str] = []
    report = refine.refine(simulation, 30.0, progress=lines.append)
    assert report.status == "published"
    assert report.changes == {"medium_threads": 2}
    assert report.path is not None and report.path.exists()
    assert report.seconds > 0
    assert "refinement: published" in report.describe()
    assert any("validating" in line for line in lines)
    # A later backend of the same execution finds and applies it.
    problem = simulation.translate()
    backend = solprop_native.backend.NativeBackend(
        problem, grid=simulation.grid, modes=simulation.modes
    )
    try:
        record = backend.metadata()["calibration"]
        assert isinstance(record, dict)
        assert record["workspace_source"] == f"refinement {report.descriptor}"
        assert record["workspace"]["medium_threads"]["applied"] == 2  # type: ignore[index]
    finally:
        backend.close()
    # Another execution (another grid) has another descriptor and applies none.
    other = problems.build("gnlse", 2.0)
    backend = solprop_native.backend.NativeBackend(
        other.translate(), grid=other.grid, modes=other.modes
    )
    try:
        record = backend.metadata()["calibration"]
        assert isinstance(record, dict)
        assert not str(record["workspace_source"]).startswith("refinement")
    finally:
        backend.close()


def test_a_refinement_of_another_machine_record_is_not_applied(
    profiled: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script(monkeypatch, lambda name: 0.8 if name == "medium 2" else 1.0)
    simulation = problems.build("gnlse")
    report = refine.refine(simulation, 30.0)
    assert report.path is not None
    # The machine profile changed since: the refinement's base no longer holds.
    changed = solprop_native.EXECUTION.updated({"compact_threads": 3})
    monkeypatch.setattr(solprop_native, "EXECUTION", changed)
    backend = solprop_native.backend.NativeBackend(
        simulation.translate(), grid=simulation.grid, modes=simulation.modes
    )
    try:
        record = backend.metadata()["calibration"]
        assert isinstance(record, dict)
        assert not str(record["workspace_source"]).startswith("refinement")
    finally:
        backend.close()


def test_an_inaccurate_candidate_is_never_admitted(
    profiled: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script(
        monkeypatch,
        lambda name: 0.5 if name == "medium 2" else 1.0,
        drift=lambda name: 1.0 if name == "medium 2" else 0.0,
    )
    report = refine.refine(problems.build("gnlse"), 30.0)
    assert report.status == "kept"
    assert report.reason == "no accurate candidate won"
    assert report.evidence["accurate"] == {"medium 2": False, "medium 8": True}


def test_a_combination_that_fails_its_validation_is_kept_back(
    profiled: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def speed(name: str) -> float:
        calls.append(name)
        # Faster in its rounds, slower once validated together.
        return (
            (0.8 if len([c for c in calls if c == name]) <= 5 else 1.3)
            if (name == "medium 2")
            else 1.0
        )

    _script(monkeypatch, speed)
    report = refine.refine(problems.build("gnlse"), 30.0)
    assert report.status == "kept"
    assert report.reason == "the combination's validation failed"


def test_an_inaccurate_combination_is_kept_back(
    profiled: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[int] = []
    _script(monkeypatch, lambda name: 0.8 if name == "medium 2" else 1.0)
    original = _Windows.fields

    def fields(window: _Windows) -> list[np.ndarray]:
        built.append(1)
        # The candidates' windows agree; the combination's, built last, does not.
        return [np.ones(4) * (3.0 if len(built) > 3 else 1.0)]

    monkeypatch.setattr(_Windows, "fields", fields)
    report = refine.refine(problems.build("gnlse"), 30.0)
    assert report.status == "kept"
    assert report.reason == "the combination is inaccurate"
    assert original is not fields


def test_noise_or_no_candidate_or_no_profile_decides_nothing(
    profiled: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def noisy(name: str) -> float:
        noisy.calls += 1  # type: ignore[attr-defined]
        return 1.5 if noisy.calls % 3 == 0 else 1.0  # type: ignore[attr-defined]

    noisy.calls = 0  # type: ignore[attr-defined]
    _script(monkeypatch, noisy)
    report = refine.refine(problems.build("gnlse"), 30.0)
    assert report.status == "inconclusive"
    assert report.reason == "too noisy or too little time to decide"
    monkeypatch.setattr(refine, "_candidates", lambda *args: [])
    empty = refine.refine(problems.build("gnlse"), 30.0)
    assert (empty.status, empty.reason) == ("kept", "no candidate changes a width")
    off = choose({"SOLPROP_CALIBRATION": "off"}, packages=_native.packages())
    monkeypatch.setattr(solprop_native, "PROFILE", off)
    none = refine.refine(problems.build("gnlse"), 30.0)
    assert none.status == "inconclusive"
    assert "no profile to refine" in none.reason


def test_a_refinement_refuses_a_bad_budget_or_backend() -> None:
    simulation = problems.build("gnlse")
    with pytest.raises(CalibrationError, match="must be positive"):
        refine.refine(simulation, 0.0)
    simulation.settings = dataclasses.replace(simulation.settings, backend="numpy_ref")
    with pytest.raises(CalibrationError, match="fits the native backend"):
        refine.refine(simulation, 1.0)


def test_the_real_windows_time_and_compare_trial_steps(profiled: Path) -> None:
    simulation = problems.build("gnlse")
    base = _Windows(simulation, None)
    serial = _Windows(
        simulation,
        solprop_native.EXECUTION.updated(
            {"stage_threshold": 1 << 40, "stage_full": 1 << 40}
        ).native(),
    )
    try:
        assert len(base.positions) == refine.POSITIONS
        assert base.time(1) > 0
        reference = base.fields()
        # Another width computes the same steps to within the run's tolerance.
        assert refine._accurate(serial.fields(), reference, 1e-6)  # pyright: ignore[reportPrivateUsage]
        assert not refine._accurate([np.full(4, np.nan)], [np.ones(4)], 1e-6)  # pyright: ignore[reportPrivateUsage]
        assert not refine._accurate([np.ones(4) * 2], [np.ones(4)], 1e-6)  # pyright: ignore[reportPrivateUsage]
    finally:
        base.close()
        serial.close()


def test_the_windows_time_the_run_s_own_node_chunk(profiled: Path) -> None:
    """A user's node budget reaches the timed workspaces (SEP 47 section 6.5)."""
    from solprop.backends.node_chunk import plan_node_chunk  # noqa: PLC0415

    simulation = problems.build("line")
    problem = simulation.translate()
    automatic = plan_node_chunk(problem).chunk
    budget = 20 * 16 * problem.shape[1] * problem.grids.n_to * 3
    simulation.settings = dataclasses.replace(simulation.settings, node_budget=budget)
    expected = plan_node_chunk(problem, simulation.settings.execution_options()).chunk
    assert expected < automatic
    window = _Windows(simulation, None)
    try:
        assert window.backend.node_chunk == expected
    finally:
        window.close()


def test_the_report_describes_a_kept_refinement() -> None:
    text = RefinementReport("kept", "abc", reason="none won").describe()
    assert text.startswith("refinement: kept in 0.0 s (abc)")
    assert "refinement:" in text


def test_a_low_storage_simulation_times_its_own_march(profiled: Path) -> None:
    from solprop.stepper import FixedSchedule, LowStorageRungeKutta  # noqa: PLC0415

    simulation = problems.build("gnlse")
    planes = list(simulation.settings.save_planes)
    length = float(max(planes))
    simulation.settings = dataclasses.replace(
        simulation.settings,
        integrator=LowStorageRungeKutta(scheme="rk4ip"),
        schedule=FixedSchedule.uniform(0.0, length, length / 10, save_planes=planes),
        propagator_mode="tabulate",
    )
    window = _Windows(simulation, None)
    try:
        assert window.positions == [0.0]
        assert window.low_storage
        assert window.time(1) > 0
        # Its march advances its state: no fields to compare, so nothing is refused.
        assert window.fields() == []
    finally:
        window.close()


def test_candidates_are_the_grain_moves_that_change_the_simulation() -> None:
    table = _native.execution.builtin("Linux").native()

    class Problem:
        shape = (1, 1, 1, 4096)

    sizing = {"kernels.response.98304.samples.4096": {}}
    found = refine._candidates(Problem(), sizing, table, 8)  # pyright: ignore[reportPrivateUsage]
    names = [candidate.name for candidate in found]
    assert "medium 2" in names
    assert "compact 4" in names
    assert all(not name.startswith("stage") for name in names)


def test_a_budget_too_short_for_a_round_is_inconclusive(
    profiled: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script(monkeypatch, lambda name: 1.0)
    report = refine.refine(problems.build("gnlse"), 1e-3)
    assert report.status == "inconclusive"
    assert report.evidence["noise"] is None


def test_a_simulation_calibrates_through_stage_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[float] = []

    def fake(simulation: object, budget: float) -> RefinementReport:
        asked.append(budget)
        return RefinementReport("kept", "abc")

    monkeypatch.setattr(refine, "refine", fake)
    assert problems.build("gnlse").calibrate(12.0).status == "kept"
    assert asked == [12.0]
