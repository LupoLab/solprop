# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The transverse collar's rate on every backend (SEP 22 sections 4.2, 4.3 and 7).

A zero rate is today's element exactly; the rate is exact under the cumulative flow
(each column falls by ``exp(-alpha_perp L / 2)`` to a rounding bound) and never
reaches the noise phase; the three backends agree on a ten-step run carrying it; and
on an aliased product the collar weakens the folded wave as the weak-conversion
integral predicts.
"""

from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.test_free_ladders import amplitude_bound
from bounds import EPS, gamma
from solprop.backends.bodies import PLAIN
from solprop.backends.driver import StepDriver
from solprop.backends.numba_cpu.kernels import HOST
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import TransverseCollar, transverse_collar_rate
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.output.schema import PlaneStore
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController, RungeKutta


@pytest.mark.parametrize("propagating", [True, False])
@pytest.mark.parametrize("active", [True, False])
def test_a_zero_rate_is_todays_element_exactly(propagating, active):
    square = (4 + 1e-3j) if propagating else (-2 + 1e-3j)
    operands = (square, 0.5, 1e-17 + 0j, 1e-17, 0.125, 0.0625, -2j, -0.5j, active, 0.75)
    # Today's element, written out: the attenuation, the frame and the spectral rate.
    longitudinal = (square - 0.5) + (1e-17 - 1e-17)
    root = np.sqrt(longitudinal)
    attenuation = root.imag if propagating else min(abs(root.imag), 0.75)
    today = (
        complex(-attenuation, -(root.real - 0.125)) if active else 0j
    ) - 0.5 * 0.0625
    for bodies in (PLAIN, HOST):
        rate, _ = bodies.spatial_element(*operands, 0.0)
        assert rate == today  # s + 0 = s: an exact no-op
        # The rate joins the attenuation on the active band alone (Q6).
        collared, _ = bodies.spatial_element(*operands, 3.0)
        assert collared == (today - 1.5 if active else today)


def column_run(backend, *, collar, schedule, rtol=1e-6):
    """A linear vacuum Cartesian run, its launch and its end, with `collar`."""
    length = 1e-3
    grid = EnvelopeGrid.from_limits(length, 800e-9, (700e-9, 950e-9), 150e-15)
    # A coarse grid: every column propagates, so vacuum attenuates nothing else.
    space = FreeGrid(4e-4, 16, 3e-4, 12, window_factor=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9), 1e-4
    )
    settings = SimulationSettings(
        controller=PIController(rtol=rtol, atol=1e-14),
        save_planes=[length],
        dz_max=length / 4 if schedule is None else schedule,
        dz_init=length / 8 if schedule is None else schedule,
        schedule=None
        if schedule is None
        else FixedSchedule.uniform(0.0, length, schedule, save_planes=[length]),
        integrator=RungeKutta(),
        backend=backend,
        transverse_absorber=collar,
    )
    simulation = Simulation.from_free(
        grid, space, SpatialMedium(), [], pulse, settings=settings, noise=None
    )
    output = simulation.run(progress="none")
    assert isinstance(output, PlaneStore)
    return simulation, output


@pytest.mark.parametrize("schedule", [None, 1e-3 / 7, 1e-3 / 16])
def test_the_rate_attenuates_each_column_exactly_over_the_run(schedule):
    collar = TransverseCollar(4e3, fraction=0.5)
    simulation, output = column_run("numpy_ref", collar=collar, schedule=schedule)
    launch, end = output.plane_field(0), output.plane_field(1)
    space = simulation.spatial_grid
    assert isinstance(space, FreeGrid)
    rate = transverse_collar_rate(space, collar)
    length = float(output["z"][-1])
    steps = int(np.sum(np.asarray(output["step_history/accepted"])))
    expected = np.exp(-rate * length / 2)[:, None, None] * np.abs(launch)
    live = np.abs(launch) > 1e-6 * np.max(np.abs(launch))
    # One exponential, one product and one rounded exponent a step (section 7).
    bound = gamma(3 * steps) + 0.5 * float(np.max(rate)) * length * EPS
    assert np.any(rate > 0) and np.all(np.isfinite(end))
    assert np.max(np.abs(np.abs(end)[live] / expected[live] - 1)) <= bound


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu"])
def test_the_rate_never_reaches_the_noise_phase(backend):
    from solprop.backends.registry import select  # noqa: PLC0415

    phases = []
    for collar in (None, TransverseCollar(4e3, fraction=0.5)):
        grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (700e-9, 950e-9), 150e-15)
        space = FreeGrid(4e-4, 16, 3e-4, 12, window_factor=0)
        pulse = SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9), 1e-4
        )
        simulation = Simulation.from_free(
            grid,
            space,
            SpatialMedium(),
            [],
            pulse,
            settings=SimulationSettings(
                controller=PIController(rtol=1e-6, atol=1e-14),
                save_planes=[1e-3],
                dz_max=1e-4,
                dz_init=1e-4,
                integrator=RungeKutta(),
                backend=backend,
                transverse_absorber=collar,
            ),
            noise="modified",
            seed=3,
        )
        problem = simulation.translate()
        selected = select(backend, problem, grid=grid)
        owner = cast("StepDriver", selected)
        try:
            owner.set_initial(problem.initial.field0, None)
            phases.append(owner._unitary_phase(owner._stage_scalars(0.0, 7e-4)).copy())
        finally:
            selected.close()
    # The rate is real: Im Phi is the same arithmetic, an exact structural identity.
    assert np.array_equal(phases[0], phases[1])


def collared_problem(steps):
    """A ten-step Cartesian Kerr problem with a transverse collar and its grid."""
    length = 1e-4
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0.1)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-6), 6e-5
    )
    simulation = Simulation.from_free(
        grid,
        space,
        SpatialMedium(),
        [KerrResponse(1e-22, thg=False)],
        pulse,
        settings=SimulationSettings(
            controller=PIController(rtol=1e-6, atol=1e-14),
            save_planes=[length],
            dz_max=length / steps,
            dz_init=length / steps,
            schedule=FixedSchedule.uniform(
                0.0, length, length / steps, save_planes=[length]
            ),
            integrator=RungeKutta(),
            backend="numpy_ref",
            transverse_absorber=TransverseCollar(2e4, fraction=0.3),
            evanescent_rate=5e3,
        ),
        noise=None,
    )
    return grid, space, simulation.translate(), length


def test_the_backends_agree_on_a_run_carrying_the_rate(spatial_backend_class, request):
    steps = 10
    grid, space, problem, length = collared_problem(steps)
    operator = problem.linear.separable
    assert operator is not None and operator.kperp_rate is not None
    assert np.any(operator.kperp_rate > 0)
    bound = amplitude_bound("envelope", grid, space, spatial_backend_class.name, steps)
    assert bound > 0
    values = []
    for implementation in (NumpyRefBackend, spatial_backend_class):
        backend = implementation(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(length)
            values.append(backend.observe().require_field().copy())
        finally:
            backend.close()
    reference, actual = values
    assert_allclose(actual, reference, rtol=0, atol=bound * np.max(np.abs(reference)))


# --- the aliased product (section 7) ---------------------------------------------

ALIAS_LENGTH = 20e-6
ALIAS_STEPS = 40
WEAK, TARGET = 19, 38
"""Columns of the weak beam (k_x = 0.95 k_max) and the fold of 2 k_weak (-0.1 k_max)."""


def aliased_run(collar):
    """Plane waves at k_x = 0 and 0.95 k_max on a 40-column line, envelope Kerr."""
    grid = EnvelopeGrid.from_limits(
        ALIAS_LENGTH, 800e-9, (780e-9, 820e-9), 4e-12, thg=False
    )
    space = LineGrid(4e-5, 40, window_factor=0.0)
    spectrum = GaussianPulse(wavelength=800e-9, fwhm=1e-12, energy=1.0).build(grid)[
        0, 0
    ]
    spectrum = spectrum / np.max(np.abs(spectrum))
    field = np.zeros((space.n_columns, 1, grid.n_omega), complex)
    field[0, 0] = 1e8 * spectrum
    field[WEAK, 0] = 1e8 * spectrum
    step = ALIAS_LENGTH / ALIAS_STEPS
    simulation = Simulation.from_free(
        grid,
        space,
        SpatialMedium(),
        [KerrResponse(1e-13, thg=False)],
        field,
        settings=SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-20),
            save_planes=[ALIAS_LENGTH],
            dz_max=step,
            dz_init=step,
            schedule=FixedSchedule.uniform(
                0.0, ALIAS_LENGTH, step, save_planes=[ALIAS_LENGTH]
            ),
            integrator=RungeKutta(),
            backend="numpy_ref",
            transverse_absorber=collar,
        ),
        noise=None,
    )
    output = simulation.run(progress="none")
    assert isinstance(output, PlaneStore)
    end = np.asarray(output.plane_field(1))
    launched = float(np.sum(np.abs(field[WEAK]) ** 2))
    return simulation, float(np.sum(np.abs(end[TARGET]) ** 2)), launched


@pytest.mark.medium
def test_the_collar_weakens_an_aliased_product_as_predicted():
    strength = 2 / ALIAS_LENGTH
    plain, without, launched = aliased_run(None)
    _, with_collar, _ = aliased_run(TransverseCollar(strength))
    space = plain.spatial_grid
    assert isinstance(space, LineGrid)
    rho = np.abs(space.k_x) / np.max(np.abs(space.k_x))
    assert rho[WEAK] == pytest.approx(0.95) and rho[TARGET] == pytest.approx(0.1)
    # Weak conversion, so that depletion and cascades stay below the 10 % tolerance.
    assert without / launched < 1e-3
    # The product's source decays at kappa = S P(0.95) = S / 2; the phase mismatch is
    # the operator's own, at the carrier.
    operator = plain.translate().linear.separable
    assert operator is not None
    carrier = int(np.argmin(np.abs(plain.grid.omega - plain.grid.omega0)))
    root = np.sqrt(
        (operator.k_squared[carrier] - operator.kperp_sq)
        + (operator.k_squared_low[carrier] - operator.kperp_sq_low)
    )
    mismatch = 2 * root[WEAK].real - root[0].real - root[TARGET].real
    kappa = strength / 2
    damped = (np.exp((-kappa + 1j * mismatch) * ALIAS_LENGTH) - 1) / (
        -kappa + 1j * mismatch
    )
    undamped = (np.exp(1j * mismatch * ALIAS_LENGTH) - 1) / (1j * mismatch)
    predicted = abs(damped) ** 2 / abs(undamped) ** 2
    assert with_collar / without == pytest.approx(predicted, rel=0.1)
