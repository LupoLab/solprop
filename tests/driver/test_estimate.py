# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run's cost before it runs (SEP 41, S11)."""

import dataclasses
import warnings

import pytest

from solprop.backends.registry import available
from solprop.driver.memory import admit_memory
from solprop.exceptions import MaterialRangeWarning, SolpropWarning
from solprop.recipe import Recipe
from solprop.stepper import FixedSchedule


def _built(name, backend):
    recipe = Recipe.example(name)
    recipe.set("backend", backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return recipe.build()


@pytest.mark.parametrize(
    "backend",
    [
        "numpy_ref",
        "numba_cpu",
        pytest.param(
            "native",
            marks=pytest.mark.skipif(
                "native" not in available(), reason="native is not installed"
            ),
        ),
    ],
)
def test_the_estimate_reports_the_memory_admission_would_admit(backend):
    simulation = _built("soliton", backend)
    estimate = simulation.estimate()
    problem = simulation.translate()
    admitted = admit_memory(problem, backend)
    assert estimate.admitted and estimate.refusal is None
    assert estimate.memory == admitted.estimate.total
    assert estimate.shape == tuple(problem.shape)
    assert estimate.rhs is not None and estimate.rhs > 0
    assert estimate.setup is not None and estimate.setup > 0
    assert estimate.step == estimate.rhs * estimate.calls_per_step
    assert estimate.threads is not None
    assert estimate.calls_per_step == 6  # the GNLSE route's Runge-Kutta march
    assert estimate.total_calls is None


@pytest.mark.parametrize(
    "backend",
    [
        "numpy_ref",
        "numba_cpu",
        pytest.param(
            "native",
            marks=pytest.mark.skipif(
                "native" not in available(), reason="native is not installed"
            ),
        ),
    ],
)
def test_a_budget_below_the_memory_is_reported_without_allocating(backend, monkeypatch):
    simulation = _built("soliton", backend)
    built = []
    monkeypatch.setattr(
        "solprop.driver.estimate.select", lambda *args, **kwargs: built.append(1)
    )
    estimate = simulation.estimate(memory_budget=1000)
    assert not estimate.admitted
    assert estimate.refusal is not None and "over the budget" in estimate.refusal
    assert estimate.rhs is None and estimate.setup is None
    assert built == []
    assert estimate.memory > 1000


def test_the_output_size_counts_only_the_retained_fields():
    simulation = _built("soliton", "numpy_ref")
    every = simulation.estimate().output_bytes
    simulation.settings = dataclasses.replace(simulation.settings, fields="none")
    assert simulation.estimate().output_bytes < every


def test_an_adams_run_takes_one_call_and_a_schedule_counts_its_total():
    assert _built("rdw", "numpy_ref").estimate().calls_per_step == 1
    simulation = _built("rdw", "numpy_ref")
    planes = list(simulation.settings.save_planes)
    # Steps short enough for the Adams method's collar (h alpha / 2 below 0.0877).
    end = planes[-1]
    schedule = FixedSchedule.uniform(0.0, end, end / 2000, save_planes=planes)
    # The recipe does not describe the schedule, so it is dropped, with a warning.
    with pytest.warns(SolpropWarning, match="schedule changed"):
        simulation.settings = dataclasses.replace(
            simulation.settings, schedule=schedule
        )
    estimate = simulation.estimate()
    assert estimate.total_calls == schedule.steps.size * estimate.calls_per_step
