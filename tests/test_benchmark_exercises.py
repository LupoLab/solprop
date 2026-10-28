# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check that the backend exercises build, and that a worker's repeats agree."""

import contextlib

import numpy as np
import pytest
from benchmarks.builders import build
from benchmarks.exercises import exercise, plasma
from benchmarks.records import SuiteError
from benchmarks.registry import CASES, select
from benchmarks.worker import measure

from solprop.driver import run as driver_run
from solprop.exceptions import MaterialRangeWarning


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", [case.name for case in select(sets=("backend-exercise",))]
)
def test_every_exercise_builds_and_translates(name):
    case = CASES[name]
    expected = (
        # optmat's H2 Raman response, used below its validated density.
        pytest.warns(MaterialRangeWarning, match="validated density range")
        if name == "vector_raman"
        else contextlib.nullcontext()
    )
    with expected:
        simulation = build(case.configuration("default", 1, "numpy_ref"))
        problem = simulation.translate()
    planes = np.asarray(problem.stepper.save_planes)
    assert planes[-1] == pytest.approx(case.length)
    assert len(planes) == case.planes


def test_exercise_builders_refuse_unknown_names():
    for builder in (exercise, plasma):
        with pytest.raises(SuiteError, match="not implemented"):
            builder("absent")


@pytest.mark.medium
def test_repeats_restart_from_the_launch_and_take_the_same_steps(tmp_path):
    configuration = CASES["noise"].configuration("default", 1, "numpy_ref")
    result = measure(configuration, 2, tmp_path / "field.npy")
    first, second = result["repetitions"]
    assert (first["accepted"], first["rejected"]) == (
        second["accepted"],
        second["rejected"],
    )
    assert first["z"] == second["z"] == pytest.approx(CASES["noise"].length)
    assert result["finite"]
    assert result["noise"] == "modified"
    assert np.isfinite(np.load(tmp_path / "field.npy")).all()


@pytest.mark.medium
def test_repeats_of_an_input_noise_run_reach_the_same_field():
    # The tracked default noise: each repeat restarts from the same seeded launch.
    configuration = CASES["gnlse_raman_shock"].configuration(
        "default", 1, "numpy_ref", 0.002
    )
    result = measure(configuration, 2, None)
    first, second = result["repetitions"]
    assert result["noise"] == "input"
    assert (first["accepted"], first["rejected"]) == (
        second["accepted"],
        second["rejected"],
    )
    assert result["repeat_difference"] <= 1e-12


@pytest.mark.medium
def test_each_repeat_checks_the_boundaries_with_a_monitor_of_its_own(monkeypatch):
    # What a user runs: Simulation.run attaches a fresh boundary monitor to every run.
    monitors = []
    run_problem = driver_run.run_problem

    def recording(*args, **kwargs):
        monitors.append(kwargs.get("boundary_monitor"))
        return run_problem(*args, **kwargs)

    monkeypatch.setattr(driver_run, "run_problem", recording)
    measure(CASES["smoke"].configuration("default", 1, "numpy_ref"), 2, None)
    assert len(monitors) == 2
    assert all(monitor is not None for monitor in monitors)
    assert monitors[0] is not monitors[1]
