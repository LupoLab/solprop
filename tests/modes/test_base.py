# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math
import warnings

import numpy as np
import pytest
from witness_materials import borzsonyi

from solprop.constants import C
from solprop.exceptions import MaterialRangeWarning, ModeError
from solprop.modes import MarcatiliMode
from solprop.modes.base import (
    BETA2_RELATIVE_STEP,
    aeff,
    alpha,
    beta,
    beta1,
    beta2,
    db_per_m,
    loss_length,
    transmission,
    zdw,
)
from solprop.physdata.glasses import fused_silica
from solprop.profiles import Linear, PressureGradient


def omega_of(wavelength):
    return 2.0 * math.pi * C / np.asarray(wavelength)


def band(count=16):
    """A frequency sweep inside every material model's validity range."""
    return omega_of(np.linspace(450e-9, 950e-9, count))


@pytest.fixture
def mode():
    return MarcatiliMode(125e-6, "He", 1.0)


@pytest.fixture
def analytic_mode():
    """A constant cladding, so the only dispersion is the gas and the waveguide."""
    return MarcatiliMode(
        50e-6, "Ar", 2.0, susceptibility=borzsonyi("Ar"), cladding=1.45
    )


# ------------------------------------------------------------- beta and alpha


def test_beta_and_alpha_are_the_conventions(mode):
    omega = band()
    index = mode.neff(omega)
    np.testing.assert_allclose(beta(mode, omega), omega * np.real(index) / C, rtol=0.0)
    np.testing.assert_allclose(
        alpha(mode, omega), 2.0 * omega * np.imag(index) / C, rtol=0.0
    )


def test_alpha_is_positive_for_a_lossy_mode(mode):
    assert np.all(alpha(mode, band()) > 0.0)


def test_alpha_vanishes_for_a_lossless_mode():
    lossless = MarcatiliMode(125e-6, "He", 1.0, loss=False)
    np.testing.assert_array_equal(alpha(lossless, band()), 0.0)


def test_a_non_finite_frequency_is_refused(mode):
    for function in (beta, alpha, beta1, beta2, loss_length, db_per_m):
        with pytest.raises(ModeError, match="must be finite"):
            function(mode, np.array([np.nan]))


# --------------------------------------------------------------------- beta1


def test_beta1_is_the_analytic_derivative_of_beta(analytic_mode):
    """The identity beta1 = (n + omega dn/domega)/c, not a difference of beta."""
    omega = band(32)
    index = np.real(analytic_mode.neff(omega))
    derivative = np.real(analytic_mode.dneff_domega(omega))
    np.testing.assert_allclose(
        beta1(analytic_mode, omega), (index + omega * derivative) / C, rtol=0.0
    )


def test_beta1_matches_a_richardson_difference_of_beta(analytic_mode):
    """The reference is the difference, so the gate is the difference's own floor.

    beta is of order 1e7 and beta1 of order 3e-9, so a central difference of beta
    carries a rounding error of eps * beta / h; at a relative step of 1e-5 that is
    about 1e-14 absolute, i.e. 4e-6 relative to beta1. Richardson extrapolation
    removes the h^2 truncation but not that, so 1e-8 is the tightest honest gate, and
    it is the *analytic* beta1 that is exact -- the contour test in
    test_capillary.py pins it to 1e-12.
    """
    omega = band(16)
    step = 1e-5 * omega
    wide = (beta(analytic_mode, omega + step) - beta(analytic_mode, omega - step)) / (
        2.0 * step
    )
    half = step / 2.0
    narrow = (beta(analytic_mode, omega + half) - beta(analytic_mode, omega - half)) / (
        2.0 * half
    )
    richardson = (4.0 * narrow - wide) / 3.0
    analytic = beta1(analytic_mode, omega)
    assert np.max(np.abs(richardson / analytic - 1.0)) < 1e-8


def test_the_group_index_of_a_hollow_capillary_is_near_unity(mode):
    """A sanity bound on the frame velocity the whole stepper rides on."""
    group_index = beta1(mode, band()) * C
    assert np.all(np.abs(group_index - 1.0) < 1e-3)


# --------------------------------------------------------------------- beta2


def test_beta2_matches_the_second_derivative_of_a_mode_with_a_known_expansion():
    """Against the closed form of the lossless reduced model with no gas.

    An evacuated capillary with a constant cladding has, exactly,
    n_eff = 1 - (u c)^2 / (2 a^2) omega^-2, so beta = omega/c - (u^2 c / (2 a^2))
    omega^-1 and beta_2 = -2 (u^2 c / (2 a^2)) omega^-3. That is truth, not another
    numerical estimate, and it isolates the differencing in beta2 from everything
    else. The agreement is 5e-9 relative at the chosen step, which is the floor quoted
    on BETA2_RELATIVE_STEP; 1e-7 leaves room for a mode with more structure.
    """
    radius = 50e-6
    evacuated = MarcatiliMode(radius, model="reduced", cladding=1.45, loss=False)
    omega = band(16)
    coefficient = evacuated.u_nm**2 * C / (2.0 * radius**2)
    expected = -2.0 * coefficient / omega**3
    assert np.max(np.abs(beta2(evacuated, omega) / expected - 1.0)) < 1e-7


def test_beta2_is_negative_above_the_zero_dispersion_wavelength(mode):
    """Anomalous dispersion above the ZDW: the regime every RDW experiment uses.

    At 1 bar in a 125 um capillary the balance point sits just below 460 nm, so the
    sweep starts at 500 nm; below it the waveguide term wins and beta_2 turns positive,
    which the next assertion checks.
    """
    assert np.all(beta2(mode, omega_of(np.linspace(500e-9, 950e-9, 16))) < 0.0)
    assert beta2(mode, omega_of(np.array([440e-9])))[0] > 0.0


def test_beta2_refuses_a_non_positive_frequency(mode):
    with pytest.raises(ModeError, match="must be positive"):
        beta2(mode, np.array([0.0]))


def test_the_beta2_step_is_documented_and_small():
    assert 1e-6 < BETA2_RELATIVE_STEP < 1e-2


# ----------------------------------------------------------------------- zdw


def test_zdw_finds_the_root_of_beta2():
    mode = MarcatiliMode(125e-6, "He", 0.4, model="reduced", cladding=1.45)
    wavelength = zdw(mode, wavelength_limits=(150e-9, 900e-9))
    assert 150e-9 < wavelength < 900e-9
    residual = float(beta2(mode, omega_of(np.array([wavelength])))[0])
    scale = float(np.max(np.abs(beta2(mode, band()))))
    # brentq's default tolerance leaves beta2 six orders below its scale in band.
    assert abs(residual) / scale < 1e-6


def test_zdw_moves_the_right_way_with_pressure():
    """More gas pushes the balance point to longer wavelengths."""
    low = zdw(
        MarcatiliMode(125e-6, "He", 0.4, model="reduced", cladding=1.45),
        wavelength_limits=(150e-9, 900e-9),
    )
    high = zdw(
        MarcatiliMode(125e-6, "He", 1.5, model="reduced", cladding=1.45),
        wavelength_limits=(150e-9, 900e-9),
    )
    assert high > low


def test_zdw_raises_when_there_is_no_root_in_the_window():
    """Rather than return a missing value that propagates silently, this raises."""
    evacuated = MarcatiliMode(125e-6, cladding=1.45)
    with pytest.raises(ModeError, match="does not change sign"):
        zdw(evacuated, wavelength_limits=(200e-9, 900e-9))


@pytest.mark.parametrize(
    "wavelength_range", [(0.0, 1e-6), (1e-6, 1e-6), (2e-6, 1e-6), (-1e-6, 1e-6)]
)
def test_zdw_refuses_a_bad_window(mode, wavelength_range):
    with pytest.raises(ModeError, match="positive and ordered"):
        zdw(mode, wavelength_limits=wavelength_range)


def test_the_default_window_is_lunas():
    """Kept for comparability; it does make the material models warn, honestly."""
    # Luna's mode, with every input stated (``docs/dev/testing.md``): the cladding and
    # the temperature are solprop's own, deliberately -- optmat's silica and 293.15 K
    # against Luna's table and 294 K -- and the 1 nm gate absorbs both.
    mode = MarcatiliMode(
        125e-6,
        "He",
        0.4,
        model="reduced",
        loss=True,
        susceptibility=borzsonyi("He"),
        cladding=fused_silica(model="fit/SiO2"),
        temperature=293.15,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        # Deliberately the default window: that it is Luna's is what is tested.
        wavelength = zdw(mode)
    # Luna test_capillary.jl asserts |zdw(MarcatiliMode(125e-6, :HeB, 0.4)) - 379 nm|
    # < 1 nm for the same model and the same helium source, Börzsönyi's (:HeB),
    # supplied explicitly (``witness_materials``). The remaining difference
    # is the cladding data, which moves it by well under a nanometre.
    assert abs(wavelength * 1e9 - 379.0) < 1.0


# ------------------------------------------------------- loss-derived quantities


def test_loss_length_and_db_per_m_are_the_conversions(mode):
    omega = band()
    values = alpha(mode, omega)
    np.testing.assert_allclose(loss_length(mode, omega), 1.0 / values, rtol=0.0)
    # Not exact: the module folds 10 / ln(10) into a constant, so the two orderings of
    # the multiplication differ in the last bit.
    np.testing.assert_allclose(
        db_per_m(mode, omega), 10.0 * values / math.log(10.0), rtol=1e-15
    )


def test_loss_length_is_infinite_for_a_lossless_mode():
    lossless = MarcatiliMode(125e-6, "He", 1.0, loss=False)
    assert np.all(np.isinf(loss_length(lossless, band())))


def test_the_stored_luna_loss_agrees_to_the_cladding_data_difference():
    """Luna: losslength 7.0593 m and 0.61521 dB/m at 800 nm, 125 um, reduced.

    solprop gives 7.0604 m and 0.61512 dB/m -- 0.015 % apart. The difference is the
    cladding: Luna's v_n comes from an unreferenced 400-row table and solprop's from
    optmat's ``fit/SiO2``, given here deliberately, with solprop's own helium and
    temperature, since the difference is what is measured. The gate is 1e-3
    relative, an order above the observed shift and far below anything that would
    signal an error in the model.
    """
    mode = MarcatiliMode(
        125e-6,
        "He",
        1.0,
        model="reduced",
        loss=True,
        cladding=fused_silica(model="fit/SiO2"),
        temperature=293.15,
    )
    omega = omega_of(np.array([800e-9]))
    assert float(loss_length(mode, omega)[0]) == pytest.approx(
        7.0593180702769, rel=1e-3
    )
    assert float(db_per_m(mode, omega)[0]) == pytest.approx(
        0.6152074146252722, rel=1e-3
    )


# --------------------------------------------------------------- transmission


def test_transmission_is_the_closed_form_for_a_uniform_waveguide(mode):
    omega = band()
    for length in (0.5, 2.0):
        np.testing.assert_allclose(
            transmission(mode, omega, length),
            np.exp(-alpha(mode, omega) * length),
            rtol=1e-9,
        )


def test_transmission_of_no_length_is_one(mode):
    np.testing.assert_array_equal(transmission(mode, band(), 0.0), 1.0)


def test_transmission_integrates_a_taper():
    """Loss goes as a^-3, so the integral is not exp(-alpha L) at either end."""
    tapered = MarcatiliMode(Linear(125e-6, 75e-6, 1.0), "He", 1.0)
    omega = omega_of(np.array([800e-9]))
    value = float(transmission(tapered, omega, 1.0)[0])
    at_start = float(np.exp(-alpha(tapered, omega, 0.0)[0]))
    at_end = float(np.exp(-alpha(tapered, omega, 1.0)[0]))
    assert at_end < value < at_start


def test_transmission_composes_over_adjacent_sections():
    """exp of an integral, so two halves must multiply to the whole."""
    fill = PressureGradient.two_point(1.0, 2.0, 0.5)
    tapered = MarcatiliMode(Linear(125e-6, 75e-6, 1.0), "He", fill)
    omega = omega_of(np.array([800e-9]))
    whole = transmission(tapered, omega, 1.0)
    halves = transmission(tapered, omega, 0.5) * transmission(tapered, omega, 0.5, 0.5)
    # quad_vec's own tolerance on two smooth integrands.
    np.testing.assert_allclose(whole, halves, rtol=1e-9)


def test_transmission_refuses_a_negative_length(mode):
    with pytest.raises(ModeError, match="must not be negative"):
        transmission(mode, band(), -1.0)


# ----------------------------------------------------------------------- aeff


def test_aeff_delegates_to_the_mode():
    tapered = MarcatiliMode(Linear(125e-6, 75e-6, 1.0), "He", 1.0)
    for z in (0.0, 0.5, 1.0):
        assert aeff(tapered, z) == tapered.aeff(z)
