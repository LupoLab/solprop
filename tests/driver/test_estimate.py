# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run's cost before it runs (SEP 41, S11)."""

import dataclasses
import math
import warnings
from typing import Any, cast

import numpy as np
import pytest

from capability_routes import ROUTES
from solprop.backends.registry import available, select
from solprop.backends.session import SessionError, Start
from solprop.driver import Simulation
from solprop.driver import estimate as estimate_module
from solprop.exceptions import BackendError, MaterialRangeWarning, SolpropWarning
from solprop.monitor import BoundaryCheck
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


def test_an_estimate_adds_nothing_for_an_edge_read_on_the_host():
    # `numpy_ref` reads a copy it has anyway, so the check holds no more memory.
    simulation = ROUTES["line"](None)
    simulation.settings = dataclasses.replace(
        simulation.settings,
        backend="numpy_ref",
        boundary_check=BoundaryCheck(budget=1.0, stop_factor=1.0),
    )
    checked = simulation.estimate().memory
    simulation.settings = dataclasses.replace(simulation.settings, boundary_check=None)
    assert simulation.estimate().memory == checked


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


def test_the_output_size_needs_no_backend(monkeypatch):
    simulation = Recipe.example("soliton").build()
    expected = simulation.estimate().output_bytes

    def refuse(*_args, **_kwargs):
        raise AssertionError("a backend was constructed")

    monkeypatch.setattr(estimate_module, "select", refuse)
    assert simulation.output_bytes() == expected > 0


BACKENDS = [
    "numpy_ref",
    pytest.param(
        "native",
        marks=pytest.mark.skipif(
            "native" not in available(), reason="native is not installed"
        ),
    ),
]
"""The backends a timing probe is checked on: the reference and the native bridge."""


def _backend(simulation):
    problem = simulation.translate()
    return problem, select(
        simulation.settings.backend,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_timing_probe_commits_nothing_and_reads_back_its_trial(backend):
    """SEP 47 section 6.6: one trial from the loaded state, with a field readback."""
    simulation = _built("soliton", backend)
    problem, probed = _backend(simulation)
    _, fresh = _backend(simulation)
    try:
        start = Start.entrance(problem)
        probe = probed.timing_probe(start)
        # The start is handed over once.
        assert start.field is None
        with pytest.raises(SessionError, match="already used"):
            probed.timing_probe(start)
        with pytest.raises(BackendError, match="take a trial"):
            probe.field()
        dz = float(problem.stepper.dz_init)
        probe.trial()
        probe.trial(float(problem.initial.z), 0.5 * dz)
        half = probe.field()
        probe.trial()
        whole = probe.field()
        assert half.shape == whole.shape == problem.shape
        # The two lengths end in different states: the readback is the last trial's.
        assert not np.array_equal(half, whole)
        # Nothing was committed: no attempt, the launch position, the loaded state.
        fresh.set_initial(problem.initial.field0, None, z=problem.initial.z)
        expected = fresh.observe()
        observed = probed.observe()
        assert observed.n_steps == 0
        assert observed.z == problem.initial.z
        assert observed.field is not None and expected.field is not None
        assert np.array_equal(observed.field, expected.field)
        with pytest.raises(BackendError, match="take a trial"):
            probe.field()
        # The readback is the trial's own state as a kept step finishes it: the same
        # arithmetic again, on one thread layout in one process, so agreement to a few
        # roundings of the field's largest magnitude bounds it (SEP 25).
        stepping = cast("Any", fresh)  # the driver's own trial, as a kernel test does
        _, state = stepping._trial_step(float(problem.initial.z), dz)
        trial = np.asarray(stepping._observed_field(stepping._finish(state, dz)))[0]
        scale = float(np.max(np.abs(trial)))
        np.testing.assert_allclose(whole, trial, rtol=0.0, atol=64e-16 * scale)
        # A copy the probe owns: a later trial does not write to it.
        before = whole.copy()
        probe.trial(float(problem.initial.z), 0.25 * dz)
        assert np.array_equal(whole, before)
    finally:
        probed.close()
        fresh.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_low_storage_probe_has_no_trial_field(backend):
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
    problem, probed = _backend(simulation)
    try:
        probe = probed.timing_probe(Start.entrance(problem))
        probe.trial()
        with pytest.raises(BackendError, match="no trial field"):
            probe.field()
    finally:
        probed.close()
