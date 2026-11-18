# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math
import warnings

import numpy as np
import pytest
from witness_materials import borzsonyi

from solprop.constants import C
from solprop.exceptions import MaterialRangeWarning, ModeError, NoZeroDispersionError
from solprop.modes import MarcatiliMode, StepIndexMode
from solprop.modes.base import (
    BETA2_RELATIVE_STEP,
    aeff,
    alpha,
    beta,
    beta1,
    beta2,
    beta3,
    db_per_m,
    loss_length,
    transmission,
    zdw,
)
from solprop.nonlinear.gnlse import SpecifiedMode
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
    lossless = MarcatiliMode(125e-6, "He", 1.0, loss_scale=0.0)
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
    evacuated = MarcatiliMode(radius, model="reduced", cladding=1.45, loss_scale=0.0)
    omega = band(16)
    coefficient = evacuated.u_nm**2 * C / (2.0 * radius**2)
    expected = -2.0 * coefficient / omega**3
    assert np.max(np.abs(beta2(evacuated, omega) / expected - 1.0)) < 1e-7


def test_beta3_matches_the_third_derivative_of_a_mode_with_a_known_expansion():
    """Against the same closed form: beta_3 = 6 (u^2 c / (2 a^2)) omega^-4.

    1.5e-6 relative at the chosen step (BETA3_RELATIVE_STEP); 5e-6 leaves room.
    """
    radius = 50e-6
    evacuated = MarcatiliMode(radius, model="reduced", cladding=1.45, loss_scale=0.0)
    omega = band(16)
    coefficient = evacuated.u_nm**2 * C / (2.0 * radius**2)
    expected = 6.0 * coefficient / omega**4
    assert np.max(np.abs(beta3(evacuated, omega) / expected - 1.0)) < 5e-6


def test_beta2_is_negative_above_the_zero_dispersion_wavelength(mode):
    """Anomalous dispersion above the ZDW: the regime every RDW experiment uses.

    At 1 bar in a 125 um capillary the balance point sits just below 460 nm, so the
    sweep starts at 500 nm; below it the waveguide term wins and beta_2 turns positive,
    which the next assertion checks.
    """
    assert np.all(beta2(mode, omega_of(np.linspace(500e-9, 950e-9, 16))) < 0.0)
    assert beta2(mode, omega_of(np.array([440e-9])))[0] > 0.0


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


@pytest.mark.parametrize("near", [0.0, -800e-9, math.inf, math.nan])
def test_zdw_refuses_a_bad_reference_wavelength(mode, near):
    with pytest.raises(ModeError, match="near must be a positive"):
        zdw(mode, wavelength_limits=(150e-9, 900e-9), near=near)


def two_zero_mode():
    """``beta2 = -1e-26 + 1e-54 (omega - 2e15)^2`` s^2/m: zeros at 1.9e15 and 2.1e15."""
    omega0 = 2e15
    return SpecifiedMode.from_taylor(
        [omega0 / C, 1.0 / C, -1e-26, 0.0, 2e-54], omega0, aeff=1e-10
    )


TWO_ZERO_WINDOW = (2.0 * math.pi * C / 2.2e15, 2.0 * math.pi * C / 1.8e15)


@pytest.mark.parametrize("knot", [100, 255, 256, 300])
def test_zdw_finds_a_zero_that_falls_exactly_on_a_sample(knot):
    """``beta2 = 1e-41 (omega - 2^51)`` s^2/m, with its zero on a sample of the search.

    Powers of two make the sample and the differenced ``beta2`` there exact, so the
    products of neighbouring signs are zero on both sides and not negative.
    """
    omega0, spacing, samples = 2.0**51, 2.0**40, 512
    mode = SpecifiedMode.from_taylor(
        [omega0 / C, 1.0 / C, 0.0, 1e-41], omega0, aeff=1e-10
    )
    lower = omega0 - knot * spacing
    upper = omega0 + (samples - 1 - knot) * spacing
    limits = (2.0 * math.pi * C / upper, 2.0 * math.pi * C / lower)
    # The search's own grid, from the limits as it converts them.
    grid = np.linspace(
        2.0 * math.pi * C / limits[1], 2.0 * math.pi * C / limits[0], samples
    )
    assert grid[knot] == omega0
    assert beta2(mode, grid)[knot] == 0.0
    found = zdw(mode, wavelength_limits=limits)
    assert found == pytest.approx(2.0 * math.pi * C / omega0, rel=1e-12)


def test_zdw_does_not_take_a_touch_of_zero_for_a_sign_change():
    """``beta2 = 1e-54 (omega - 2e15)^2`` touches zero and keeps its sign."""
    omega0 = 2e15
    mode = SpecifiedMode.from_taylor(
        [omega0 / C, 1.0 / C, 0.0, 0.0, 2e-54], omega0, aeff=1e-10
    )
    with pytest.raises(NoZeroDispersionError, match="does not change sign"):
        zdw(mode, wavelength_limits=TWO_ZERO_WINDOW)


def test_zdw_names_every_zero_when_the_window_holds_several():
    # The ends of the window have one sign, and both zeros are real.
    with pytest.raises(ModeError, match=r"2 zeros .* 897\.0, 991\.4 nm"):
        zdw(two_zero_mode(), wavelength_limits=TWO_ZERO_WINDOW)


@pytest.mark.parametrize(("near", "omega"), [(900e-9, 2.1e15), (1000e-9, 1.9e15)])
def test_zdw_returns_the_zero_nearest_a_reference(near, omega):
    found = zdw(two_zero_mode(), wavelength_limits=TWO_ZERO_WINDOW, near=near)
    # beta1 is a cubic, so the Richardson difference is exact but for rounding (1e-37
    # s^2/m against a slope of 2e-40 s^3/m per rad/s): the root to 1e-12, read at 1e-9.
    assert found == pytest.approx(2.0 * math.pi * C / omega, rel=1e-9)


def krypton():
    """A 14 um capillary at 15 bar of krypton: its resonances near 120 nm flip the
    sign of the differenced ``beta2`` sixteen times between 100 and 124 nm."""
    return MarcatiliMode(14e-6, "Kr", 15.0)


GAS_WINDOW = (100e-9, 3e-6)
"""A window reaching below the gas lines, the default before the noble gases' lines
were left out of it."""


@pytest.mark.parametrize("near", [800e-9, 200e-9])
def test_zdw_finds_a_zero_past_the_sign_changes_of_a_gas_resonance(near):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        found = zdw(krypton(), wavelength_limits=GAS_WINDOW, near=near)
    # The single zero of the qualified band (125 nm up), searched there alone.
    reference = zdw(krypton(), wavelength_limits=(125e-9, 3e-6))
    # Other samples, so other brackets: both stop on the noise of the differenced
    # beta2 (5e-9, BETA2_RELATIVE_STEP).
    assert found == pytest.approx(reference, rel=1e-8)
    assert reference == pytest.approx(770.5167e-9, rel=1e-6)


def test_zdw_does_not_cross_a_gas_line_to_a_nearer_zero():
    # From 150 nm, krypton's own zero at 108.9 nm (between its 104 and 116.5 nm
    # lines) is nearer in frequency than 770.5 nm, but the 116.5 and 123.6 nm lines
    # lie between: their group delay diverges or drops to the evanescent floor.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        found = zdw(krypton(), wavelength_limits=GAS_WINDOW, near=150e-9)
    # The noise of the differenced beta2, as above.
    assert found == pytest.approx(770.5167e-9, rel=1e-6)


@pytest.mark.parametrize(
    ("near", "expected"),
    [(123.47e-9, 119.8786e-9), (124e-9, 770.5167e-9)],
    ids=["on", "beside"],
)
def test_zdw_from_a_reference_on_a_gas_line_is_blocked_on_neither_side(near, expected):
    # Krypton's 123.6 nm line changes sign between the samples at 123.33 and 123.62 nm
    # (512 over GAS_WINDOW). A reference between them is on neither side of the line,
    # so it blocks neither: the zero at 119.9 nm, between the 116.5 and 123.6 nm lines,
    # is the nearest. From 124 nm the line lies between, and the answer is 770.5 nm.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        found = zdw(krypton(), wavelength_limits=GAS_WINDOW, near=near)
    # The noise of the differenced beta2, as above.
    assert found == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("radius", "pressure", "near", "expected"),
    [
        # The zero at 105.9 nm lies past argon's 106.7 nm line from 120 nm.
        (14e-6, 15.0, 120e-9, 644.0787e-9),
        # Without a reference: the 105.9 nm zero beside the lines is clustered with
        # their sign changes, so it is no zero-dispersion point.
        (125e-6, 1.0, None, 967.8074e-9),
    ],
)
def test_zdw_of_an_argon_capillary_is_its_zero_above_the_lines(
    radius, pressure, near, expected
):
    mode = MarcatiliMode(radius, "Ar", pressure)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        found = zdw(mode, wavelength_limits=GAS_WINDOW, near=near)
    # The noise of the differenced beta2 (5e-9) and the 7 digits pinned.
    assert found == pytest.approx(expected, rel=1e-6)


def test_zdw_does_not_report_a_resonance_artefact_as_a_zero():
    # 121-125 nm holds the floored plateau above krypton's 123.6 nm line, where the
    # differenced beta2 changes sign while the group delay is constant.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        with pytest.raises(NoZeroDispersionError, match="none is a zero"):
            zdw(krypton(), wavelength_limits=(121e-9, 125e-9))


def silica_fibre():
    """A standard single-mode fibre: 4.1 um core radius, NA 0.12, on optmat's silica,
    whose OH absorption bands flip the sign of beta2 four times between 1.36 and
    1.40 um and twice near 2.7 um."""
    return StepIndexMode.from_numerical_aperture(
        4.1e-6, 0.12, profile_wavelength=1.3e-6
    )


FIBRE_ZDW = 1295.83467e-9
"""The fibre's zero-dispersion wavelength, m."""


@pytest.mark.medium
@pytest.mark.parametrize(
    ("window", "near"),
    [
        ((100e-9, 3e-6), 1550e-9),
        ((700e-9, 2.5e-6), 1550e-9),
        ((1e-6, 2e-6), 1550e-9),
        (None, 1030e-9),
        (None, 800e-9),
        ((150e-9, 2.5e-6), None),
    ],
)
def test_zdw_of_a_silica_fibre_does_not_depend_on_the_window(window, near):
    """The OH band at 1.38 um lies between a 1550 nm pump and the 1.3 um zero: its
    clustered zeros are passed over, and it does not block the search like a line."""
    limits = {} if window is None else {"wavelength_limits": window}
    found = zdw(silica_fibre(), near=near, **limits)
    # Other windows, so other brackets: each stops on the noise of the differenced
    # beta2 (5e-9); they agree to 2e-10, and with the 9 digits pinned to 1e-9.
    assert found == pytest.approx(FIBRE_ZDW, rel=1e-8)


@pytest.mark.medium
def test_a_plain_zdw_of_a_silica_fibre_names_its_zeros_above_the_ultraviolet():
    """Not silica's zeros in the vacuum ultraviolet, where a lossless solve takes it
    as transparent: the default window starts above them. Past the OH band at 2.7 um,
    beta2 is normal from 2.85 to 2.91 um, a regime 2 % wide."""
    with pytest.raises(
        ModeError, match=r"3 zeros over 150-3000 nm, at 1295\.8, 2846\.1, 2906\.9 nm"
    ):
        zdw(silica_fibre())


@pytest.mark.medium
def test_zdw_passes_over_the_clustered_zeros_of_an_absorption_band():
    # beta2 changes sign four times here, through the OH overtone of the silica.
    with pytest.raises(NoZeroDispersionError, match=r"4 times .* none is a zero"):
        zdw(silica_fibre(), wavelength_limits=(1.36e-6, 1.40e-6))


def test_the_default_window_finds_lunas_helium_zero():
    """The default window starts at 150 nm, not Luna's 100 nm; the answer is Luna's."""
    # Luna's mode, with every input stated (``docs/dev/testing.md``): the cladding and
    # the temperature are solprop's own, deliberately -- optmat's silica and 293.15 K
    # against Luna's table and 294 K -- and the 1 nm gate absorbs both.
    mode = MarcatiliMode(
        125e-6,
        "He",
        0.4,
        model="reduced",
        loss_scale=1.0,
        susceptibility=borzsonyi("He"),
        cladding=fused_silica(model="fit/SiO2"),
        temperature=293.15,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        # Deliberately the default window.
        wavelength = zdw(mode)
    # Luna test_capillary.jl asserts |zdw(MarcatiliMode(125e-6, :HeB, 0.4)) - 379 nm|
    # < 1 nm for the same model and the same helium source, Börzsönyi's (:HeB),
    # supplied explicitly (``witness_materials``). The remaining difference
    # is the cladding data, which moves it by well under a nanometre.
    assert abs(wavelength * 1e9 - 379.0) < 1.0


def test_the_default_window_leaves_out_the_gas_lines():
    """A plain call on krypton is not refused for the gas's own zeros near 110 nm."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        found = zdw(krypton())
    # The noise of the differenced beta2, as above.
    assert found == pytest.approx(770.5167e-9, rel=1e-6)


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
    lossless = MarcatiliMode(125e-6, "He", 1.0, loss_scale=0.0)
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
        loss_scale=1.0,
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


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda mode: beta3(mode, np.array([0.0])), "beta3 differences beta1"),
        (lambda mode: beta2(mode, np.array([0.0])), "must be positive"),
        (lambda mode: transmission(mode, band(), -1.0), "must not be negative"),
    ],
)
def test_a_dispersion_or_loss_query_outside_its_domain_is_refused(mode, call, message):
    with pytest.raises(ModeError, match=message):
        call(mode)


# ----------------------------------------------------------------------- aeff


def test_aeff_delegates_to_the_mode():
    tapered = MarcatiliMode(Linear(125e-6, 75e-6, 1.0), "He", 1.0)
    for z in (0.0, 0.5, 1.0):
        assert aeff(tapered, z) == tapered.aeff(z)
