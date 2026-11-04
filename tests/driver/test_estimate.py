# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run's cost before it runs (SEP 41, S11)."""

import dataclasses
import math
import warnings

import pytest

from solprop.backends.registry import available
from solprop.exceptions import MaterialRangeWarning, SolpropWarning
from solprop.recipe import Recipe
from solprop.stepper import FixedSchedule, LowStorageRungeKutta


def _built(name, backend):
    recipe = Recipe.example(name)
    recipe.set("backend", backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return recipe.build()


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
