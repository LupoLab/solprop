# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The varying-area GNLSE through the existing route, against its analytic phase."""

from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from march_bounds import FIXED_ORDER, assert_matched, simulation_bound
from solprop import fourier
from solprop.constants import C
from solprop.driver import Simulation, prop_gnlse
from solprop.driver.scaffold import propagate_fixed_rk4
from solprop.exceptions import ModeError, StepperError
from solprop.grid import EnvelopeGrid
from solprop.nonlinear.gnlse import SpecifiedMode, problem_from
from solprop.output.memory import MemoryOutput
from solprop.profiles import Constant, FromCallable, Linear, PiecewiseLinear
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, Multistep
from solprop.translate.modal import _radius_kinks, mode_averaged_transform

LENGTH = 1e-3
"""A millimetre of fibre: the qualification scope's short window (AGENTS.md)."""

ENTRANCE_AREA = 1e-10
GROWTH = 500.0
"""``A_eff(z) = A_eff(0) (1 + GROWTH z)``: half again over the millimetre."""

GAMMA = 0.1
OMEGA0 = 2 * np.pi * C / 835e-9

GRID = {
    "wavelength": 835e-9,
    "wavelength_limits": (600e-9, 1300e-9),
    "trange": 2e-12,
}
PULSE = {"fwhm": 200e-15, "energy": 1e-12, "shape": "gauss"}


def fibre(area):
    """Return a dispersion-free fibre of the declared effective area."""
    return SpecifiedMode.from_taylor([0.0], omega_ref=OMEGA0, aeff=area)


def run(area, gamma=GAMMA, **keywords):
    """Propagate the millimetre with no dispersion and no loss, on the Adams march.

    The landing and accuracy figures below were qualified on ``Multistep()``, the
    GNLSE default before SEP 37 made it the Runge-Kutta march.
    """
    keywords.setdefault("integrator", Multistep())
    return prop_gnlse(
        gamma,
        LENGTH,
        fibre=fibre(area),
        shock=False,
        n_save=2,
        progress="none",
        noise=None,
        **GRID,
        **PULSE,
        **keywords,
    )


def envelopes(output):
    """Return the entrance and exit envelopes in time, internal convention."""
    spectra = output.fields("fft")[:, 0, 0]
    return [fourier.inverse(one, kind="envelope") for one in (spectra[0], spectra[-1])]


@pytest.mark.medium
def test_varying_area_spm():
    """The analytic phase ``P integral gamma(z) dz`` of a dispersion-free fibre.

    With no dispersion and no loss every temporal slice keeps its power and turns
    by ``-P(t) integral gamma(z) dz`` in the internal convention, and for the
    entrance coefficient rule ``gamma(z) = gamma A_eff(0) / A_eff(z)`` a linear
    area gives ``gamma ln(1 + kappa L) / kappa`` exactly.

    Predeclared bounds, written before the run: 1e-9 relative on that phase over
    the half of the pulse above a fifth of its peak power, and 1e-12 relative on
    the conserved energy. The phase is 4.7e-4 rad at the peak, so the first is
    5e-13 rad, which the default integrator reaches at ``rtol=1e-12``: measured
    9.6e-11 and 9e-16.
    """
    phase_rtol, energy_rtol = 1e-9, 1e-12
    area = Linear(ENTRANCE_AREA, ENTRANCE_AREA * (1 + GROWTH * LENGTH), LENGTH)
    entrance, exit_field = envelopes(run(area, rtol=1e-12))
    power = abs(entrance) ** 2
    accumulated = GAMMA * np.log1p(GROWTH * LENGTH) / GROWTH
    inside = power > 0.2 * power.max()
    assert_allclose(
        np.angle(exit_field[inside] / entrance[inside]),
        -accumulated * power[inside],
        rtol=phase_rtol,
        atol=0,
    )
    assert_allclose(
        np.sum(abs(exit_field) ** 2), np.sum(power), rtol=energy_rtol, atol=0
    )
    # Only the shape enters: doubling the declared area everywhere leaves the
    # entrance coefficient, and so the phase, where it was.
    doubled = Linear(
        2 * ENTRANCE_AREA, 2 * ENTRANCE_AREA * (1 + GROWTH * LENGTH), LENGTH
    )
    scaled = envelopes(run(doubled, rtol=1e-12))[1]
    assert_allclose(
        np.angle(scaled[inside] / entrance[inside]),
        -accumulated * power[inside],
        rtol=phase_rtol,
        atol=0,
    )
    # A constant area of the same value is the ordinary GNLSE, and its phase is the
    # larger one the entrance coefficient alone would give.
    constant = envelopes(run(ENTRANCE_AREA, rtol=1e-12))[1]
    assert_allclose(
        np.angle(constant[inside] / entrance[inside]),
        -GAMMA * LENGTH * power[inside],
        rtol=phase_rtol,
        atol=0,
    )


def test_the_area_path_is_the_one_kink_mechanism():
    corner = 0.4 * LENGTH
    area = PiecewiseLinear(
        np.array([0.0, corner, LENGTH]),
        np.array([ENTRANCE_AREA, 2 * ENTRANCE_AREA, ENTRANCE_AREA]),
    )
    mode = fibre(area)
    # The declared area path is a cross-section path like a bore radius: its
    # breakpoints are the mode's kinks, and they reach the schedule, which lands on
    # them exactly rather than stepping across.
    assert corner in _radius_kinks(mode)
    assert sorted(mode.parameter_paths) == ["effective_area"]
    assert not mode.z_constant
    output = cast("MemoryOutput", run(area, rtol=1e-12))
    history = output.planes()[-1].step_history
    assert corner in set(history.z.tolist())
    # And the phase across the kink is analytic too. With the entrance-coefficient
    # rule gamma(z) = gamma A(0)/A(z), a segment over which the area goes from A to
    # B linearly contributes dz ln(B/A)/(B - A) times gamma A(0); here that is
    # 0.4 L ln 2 + 0.6 L ln 2 = L ln 2 exactly, whatever the corner is, because the
    # area doubles and halves again. Over the half of the pulse above a fifth of its
    # peak power the phase is within 2e-9 relative. The phase is 4.7e-4 rad at the
    # peak, so that is 1e-12 rad, which the default integrator reaches at rtol=1e-12:
    # 3.9e-10 in 97 steps. The two segments and the schedule landing on the corner are
    # what is being tested.
    entrance, exit_field = envelopes(output)
    power = abs(entrance) ** 2
    inside = power > 0.2 * power.max()
    assert_allclose(
        np.angle(exit_field[inside] / entrance[inside]),
        -GAMMA * LENGTH * np.log(2.0) * power[inside],
        rtol=2e-9,
        atol=0,
    )
    # A wrapper keeps the geometry of the mode it was built from.
    assert corner in _radius_kinks(SpecifiedMode.from_mode(mode))


def test_scalar_reduction():
    # A constant area declared as a path is the existing route: the same fast
    # paths, the same arithmetic. On a prescribed schedule at a fixed order the two
    # runs are one march, so they agree within its rounding budget (SEP 25). Without
    # a collar, which refuses ten fixed steps and is not what this is about.
    prescribed = {
        "schedule": FixedSchedule.uniform(0.0, LENGTH, LENGTH / 10, save_planes=[]),
        "integrator": FIXED_ORDER,
        "absorber": False,
    }
    number = run(ENTRANCE_AREA, **prescribed).fields("fft")
    declared = run(Constant(ENTRANCE_AREA), **prescribed).fields("fft")
    simulation = Simulation.from_gnlse(
        GAMMA,
        LENGTH,
        fibre=fibre(ENTRANCE_AREA),
        shock=False,
        n_save=2,
        noise=None,
        **GRID,
        **PULSE,
        **prescribed,
    )
    assert_matched(declared, number, simulation_bound(simulation))
    assert fibre(Constant(ENTRANCE_AREA)).z_constant


def test_a_constant_area_is_still_frozen_where_it_always_was():
    mode = fibre(ENTRANCE_AREA)
    assert mode.aeff() == mode.aeff(0.5) == ENTRANCE_AREA
    assert mode.radius(1.0) == mode.radius()
    assert mode.dimlimits(1.0) == mode.dimlimits()
    assert _radius_kinks(mode) == ()
    # And an area that follows a path is evaluated at the position asked for.
    varying = fibre(Linear(ENTRANCE_AREA, 2 * ENTRANCE_AREA, LENGTH))
    assert varying.aeff(LENGTH) == pytest.approx(2 * ENTRANCE_AREA)
    assert varying.radius(LENGTH) > varying.radius(0.0)
    assert varying.dimlimits(LENGTH)[2][0] == varying.radius(LENGTH)
    assert varying.at_parameters({"effective_area": 3e-10}).aeff(LENGTH) == 3e-10
    with pytest.raises(ModeError, match="unknown SpecifiedMode"):
        varying.at_parameters({"radius": 1.0})


def test_a_path_that_is_not_an_area_is_refused_where_it_is_declared():
    with pytest.raises(ModeError, match="effective_area must be positive"):
        fibre(Linear(ENTRANCE_AREA, -ENTRANCE_AREA, LENGTH))
    with pytest.raises(ModeError, match="effective_area must be positive"):
        fibre(0.0)


def test_an_area_that_dips_between_the_declared_points_is_refused_by_name():
    # A cosine is positive at both ends and at every breakpoint it declares -- it
    # has none -- and negative in between, so the constructor's check passes it and
    # the transform is where it has to be caught. Before, the sqrt of a negative
    # area raised a bare ValueError from the maths library.
    dipping = FromCallable(
        lambda z: ENTRANCE_AREA * np.cos(4 * np.pi * z / LENGTH), (0.0, LENGTH)
    )
    mode = fibre(dipping)
    assert mode.aeff(0.0) > 0 and mode.aeff(LENGTH / 4) < 0
    grid = EnvelopeGrid.from_limits(LENGTH, 835e-9, (600e-9, 1300e-9), 2e-12)
    with pytest.raises(ModeError, match=r"effective area of .* at z ="):
        mode_averaged_transform(mode, grid, z_max=LENGTH)


def test_the_fixed_step_scaffold_refuses_a_fibre_whose_area_moves():
    grid = EnvelopeGrid.from_limits(LENGTH, 835e-9, (600e-9, 1300e-9), 2e-12)
    problem = problem_from([0.0], omega_ref=grid.omega0, gamma=GAMMA)
    start = GaussianPulse(wavelength=835e-9, fwhm=200e-15, energy=1e-12).build(grid)
    varying = SpecifiedMode.from_taylor(
        [0.0], omega_ref=grid.omega0, aeff=Linear(ENTRANCE_AREA, 2e-10, LENGTH)
    )
    with pytest.raises(StepperError, match="z-constant"):
        propagate_fixed_rk4(
            problem.__class__(**{**vars(problem), "mode": varying}),
            grid,
            start[0, 0],
            LENGTH,
            4,
        )
