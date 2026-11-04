# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run's cost before it runs (SEP 41, S11)."""

import dataclasses
import math
import warnings

import pytest

from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.driver import estimate as estimate_module
from solprop.exceptions import MaterialRangeWarning, SolpropWarning
from solprop.recipe import Recipe
from solprop.stepper import FixedSchedule, LowStorageRungeKutta


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
        pytest.param(
            "native",
            marks=pytest.mark.skipif(
                "native" not in available(), reason="native is not installed"
            ),
        ),
    ],
)
def test_the_estimate_reports_what_the_backend_holds_and_costs(backend):
    simulation = _built("soliton", backend)
    estimate = simulation.estimate()
    problem = simulation.translate()
    assert estimate.memory >= 16 * math.prod(problem.shape)
    assert estimate.shape == tuple(problem.shape)
    assert estimate.rhs > 0
    assert estimate.setup > 0
    assert estimate.step == estimate.rhs * estimate.calls_per_step
    assert estimate.threads is not None
    assert estimate.calls_per_step == 6  # the GNLSE route's Runge-Kutta march
    assert estimate.total_calls is None


def test_the_output_size_counts_only_the_retained_fields():
    simulation = _built("soliton", "numpy_ref")
    every = simulation.estimate().output_bytes
    simulation.settings = dataclasses.replace(simulation.settings, fields="none")
    assert simulation.estimate().output_bytes < every


def test_a_low_storage_run_takes_four_calls_a_step():
    simulation = _built("soliton", "numpy_ref")
    planes = list(simulation.settings.save_planes)
    schedule = FixedSchedule.uniform(
        0.0, planes[-1], planes[-1] / 200, save_planes=planes
    )
    # The recipe does not describe the schedule, so it is dropped, with a warning.
    with pytest.warns(SolpropWarning, match="schedule, integrator changed"):
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=LowStorageRungeKutta(), schedule=schedule
        )
    estimate = simulation.estimate()
    assert estimate.calls_per_step == 4
    assert estimate.total_calls == schedule.steps.size * 4


@pytest.mark.parametrize(
    "backend",
    [
        "numpy_ref",
        pytest.param(
            "native",
            marks=pytest.mark.skipif(
                "native" not in available(), reason="native is not installed"
            ),
        ),
    ],
)
def test_a_low_storage_estimate_times_the_march_s_own_step(backend, monkeypatch):
    # #492: the Runge-Kutta trial step isn't this march's; on native it has no views.
    length = 0.02
    schedule = FixedSchedule.uniform(0.0, length, length / 8, save_planes=[length / 2])
    simulation = Simulation.from_gnlse(
        0.11,
        length,
        [0.0, 0.0, -11.83e-27],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        save_planes=[length / 2, length],
        schedule=schedule,
        integrator=LowStorageRungeKutta(),
        backend=backend,
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        absorber=False,
        noise=None,
    )
    taken = []
    build = estimate_module._timed_step

    def spy(constructed, problem):
        step, evaluations = build(constructed, problem)
        monkeypatch.setattr(
            constructed,
            "_trial_step",
            lambda *_: pytest.fail("a low-storage estimate took a trial step"),
        )

        def counted():
            taken.append(1)
            return step()

        return counted, evaluations

    monkeypatch.setattr(estimate_module, "_timed_step", spy)
    estimate = simulation.estimate()
    assert len(taken) == 4  # the untimed first step and three timed
    assert estimate.calls_per_step == 4
    assert estimate.rhs > 0
    assert estimate.step == estimate.rhs * 4


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
