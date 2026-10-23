# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from solprop.constants import EPS0, C
from solprop.exceptions import ModeError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.nonlinear.gnlse import SpecifiedMode, problem_from
from solprop.nonlinear.normalisation import (
    mode_average_prefactor,
    sqrtw_to_vm,
    uppe_prefactor,
)

CENTRE = 1030e-9
OMEGA_REF = 2.0 * np.pi * C / CENTRE
BETAS = [1.45 * OMEGA_REF / C, 1.47 / C, 2.0e-26]
AEFF = 8e-11

GRID = {
    "length": 0.1,
    "reference_wavelength": CENTRE,
    "wavelength_limits": (700e-9, 2.0e-6),
    "trange": 2e-12,
}


@pytest.fixture
def mode():
    return SpecifiedMode.from_taylor(BETAS, OMEGA_REF, aeff=AEFF)


@pytest.fixture
def field_grid():
    return FieldGrid.from_limits(**GRID)


@pytest.fixture
def envelope_grid():
    return EnvelopeGrid.from_limits(**GRID)


# ------------------------------------------------------------ square watts to volts


def test_the_conversion_makes_the_intensity_the_power_over_the_area():
    # conventions.md section 4: I = eps0 c |E_a|^2 / 2 in V/m must equal P / A_eff
    # when |E_a|^2 in sqrt(W) is the power.
    power = 3.7e5  # W
    field = np.sqrt(power) * sqrtw_to_vm(AEFF)
    intensity = 0.5 * EPS0 * C * field**2
    assert intensity == pytest.approx(power / AEFF, rel=1e-15)


def test_the_conversion_is_invertible():
    amplitude = 1.234e3
    assert amplitude * sqrtw_to_vm(AEFF) / sqrtw_to_vm(AEFF) == amplitude


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_a_bad_effective_area_is_rejected(bad):
    with pytest.raises(ModeError, match="aeff must be positive and finite"):
        sqrtw_to_vm(bad)


# ---------------------------------------------------------------- the UPPE prefactor


def test_the_prefactor_is_the_uppe_expression(mode, envelope_grid):
    prefactor = uppe_prefactor(mode, envelope_grid)
    active = envelope_grid.active_mask
    omega = envelope_grid.omega[active]
    expected = -1j * omega / (2.0 * EPS0 * C * np.real(mode.neff(omega)))
    assert np.allclose(prefactor[active], expected, rtol=1e-14, atol=0.0)


def test_the_prefactor_ignores_the_loss_and_stays_purely_imaginary(envelope_grid):
    # The loss lives in the linear operator. Keeping it here as well would give the
    # nonlinear term a real prefactor -- a gain that no derivation asks for.
    lossy = SpecifiedMode.from_taylor(BETAS, OMEGA_REF, loss=50.0, aeff=AEFF)
    lossless = SpecifiedMode.from_taylor(BETAS, OMEGA_REF, aeff=AEFF)
    # The same real index by two dispersions: a few roundings at most.
    np.testing.assert_allclose(
        uppe_prefactor(lossy, envelope_grid),
        uppe_prefactor(lossless, envelope_grid),
        rtol=4 * np.finfo(float).eps,
        atol=0,
    )
    # Structural: a purely imaginary product has an exactly zero real part.
    assert np.array_equal(
        np.real(uppe_prefactor(lossy, envelope_grid)),
        np.zeros(envelope_grid.n_omega),
    )


def test_the_prefactor_is_zero_outside_the_active_band(mode, field_grid):
    prefactor = uppe_prefactor(mode, field_grid)
    outside = ~field_grid.active_mask
    # Zero, not one: raw, unnormalised polarisation must not be left out of band.
    assert np.array_equal(prefactor[outside], np.zeros(int(np.sum(outside)), complex))


def test_turning_the_shock_off_freezes_only_the_numerator(mode, envelope_grid):
    active = envelope_grid.active_mask
    with_shock = uppe_prefactor(mode, envelope_grid)[active]
    without = uppe_prefactor(mode, envelope_grid, shock=False)[active]
    omega = envelope_grid.omega[active]
    # The ratio is exactly omega / omega0: the self-steepening factor and nothing
    # else. The index stays frequency dependent.
    assert np.allclose(
        with_shock / without, omega / envelope_grid.omega0, rtol=1e-14, atol=0.0
    )


def test_a_constant_index_replaces_the_mode_s(mode, envelope_grid):
    active = envelope_grid.active_mask
    fixed = uppe_prefactor(mode, envelope_grid, index=1.0)[active]
    omega = envelope_grid.omega[active]
    assert np.allclose(fixed, -1j * omega / (2.0 * EPS0 * C), rtol=1e-15, atol=0.0)


def test_the_prefactor_is_a_function_of_the_absolute_frequency_alone(
    mode, field_grid, envelope_grid
):
    # The two grids sample different frequencies, so compare the expression rather
    # than the arrays: on each grid the prefactor must be exactly the same function of
    # the absolute omega the grid stores.
    for grid in (field_grid, envelope_grid):
        active = grid.active_mask
        omega = grid.omega[active]
        expected = -1j * omega / (2.0 * EPS0 * C * mode.neff(omega))
        assert np.allclose(
            uppe_prefactor(mode, grid)[active], expected, rtol=1e-14, atol=0.0
        )


def test_a_vanishing_index_is_refused(envelope_grid):
    # beta_0 = 0 makes c beta / omega pass through zero, which the prefactor cannot
    # divide by, so it is refused rather than clamped.
    flat = SpecifiedMode.from_taylor([0.0], omega_ref=OMEGA_REF)
    with pytest.raises(ModeError, match="zero or not finite"):
        uppe_prefactor(flat, envelope_grid)


# ------------------------------------------------------- the mode-averaged prefactor


def test_the_mode_average_prefactor_is_the_composition(mode, envelope_grid):
    composed = mode_average_prefactor(mode, envelope_grid)
    plain = uppe_prefactor(mode, envelope_grid) / sqrtw_to_vm(AEFF)
    # One composition by two groupings: a couple of roundings.
    np.testing.assert_allclose(composed, plain, rtol=4 * np.finfo(float).eps, atol=0)


def test_the_mode_average_prefactor_matches_luna_s_algebra(mode, envelope_grid):
    active = envelope_grid.active_mask
    omega = envelope_grid.omega[active]
    # Luna: nl *= pre / beta * sqrt(Aeff) with pre = -i omega^2 / 4 / nlscale / c,
    # applied after Eto has been divided by nlscale * sqrt(Aeff)
    # (NonlinearRHS.jl:852-888). Written out, that is this.
    nlscale = np.sqrt(0.5 * EPS0 * C)
    beta = omega * np.real(mode.neff(omega)) / C
    luna = -1j * omega**2 / 4.0 / nlscale / C / beta * np.sqrt(AEFF)
    assert np.allclose(
        mode_average_prefactor(mode, envelope_grid)[active],
        luna,
        rtol=1e-14,
        atol=0.0,
    )


GAMMA = 0.011


def gnlse_coefficient(grid, problem, *, shock):
    """The coefficient of |A|^2 A that the whole mapping produces on `grid`.

    For a continuous wave the spectrum is one bin, so no transform is involved: the
    rate of change of the amplitude is the prefactor times the polarisation, and the
    transform normalisation cancels between the two.
    """
    centre = int(np.argmin(abs(grid.omega - grid.omega0)))
    prefactor = mode_average_prefactor(
        problem.mode, grid, shock=shock, index=problem.prefactor_index
    )[centre]
    amplitude = np.array([[2000.0 + 500.0j]])
    field = amplitude * sqrtw_to_vm(problem.aeff)
    polarisation = problem.kerr.evaluate(field, 1.0, kind="envelope")
    return (prefactor * polarisation / (np.abs(amplitude) ** 2 * amplitude))[0, 0]


def test_the_gnlse_chain_reproduces_the_textbook_coefficient(envelope_grid):
    # The whole mapping end to end: gamma -> n2 -> chi3 -> Kerr, and back through the
    # prefactor and the sqrt(W) conversion. It must come out as -i gamma |A|^2 A in
    # the internal convention, which is the GNLSE with a positive gamma
    # (docs/guide/waveguides/gnlse.md). With the shock off the prefactor is frozen at
    # omega0 and
    # the identity is exact rather than approximate.
    problem = problem_from(
        BETAS, omega_ref=envelope_grid.omega0, gamma=GAMMA, aeff=AEFF
    )
    coefficient = gnlse_coefficient(envelope_grid, problem, shock=False)
    # Eight multiplications and divisions on numbers spanning 10^-30 to 10^7; Higham's
    # n*eps for n = 8 is 2e-15, and the measured deviation is 4e-16.
    assert coefficient == pytest.approx(-1j * GAMMA, rel=2e-15)


def test_the_two_grids_give_the_same_nonlinear_coefficient(field_grid, envelope_grid):
    # Kerr prefactor consistency across grids. With the shock frozen at omega0 and
    # the GNLSE's unit index, the prefactor has no frequency dependence left, so the
    # two grids must agree to the few roundings of one expression.
    problem = problem_from(
        BETAS, omega_ref=envelope_grid.omega0, gamma=GAMMA, aeff=AEFF
    )
    on_field = gnlse_coefficient(field_grid, problem, shock=False)
    on_envelope = gnlse_coefficient(envelope_grid, problem, shock=False)
    assert on_field == pytest.approx(on_envelope, rel=4 * np.finfo(float).eps)
    assert on_field != 0.0


def test_the_gnlse_shock_factor_is_the_frequency_ratio(envelope_grid):
    problem = problem_from(BETAS, omega_ref=envelope_grid.omega0, gamma=GAMMA)
    active = envelope_grid.active_mask
    omega = envelope_grid.omega[active]
    with_shock = mode_average_prefactor(
        problem.mode, envelope_grid, index=problem.prefactor_index
    )[active]
    without = mode_average_prefactor(
        problem.mode, envelope_grid, shock=False, index=problem.prefactor_index
    )[active]
    assert np.allclose(
        with_shock / without, omega / envelope_grid.omega0, rtol=1e-14, atol=0.0
    )
