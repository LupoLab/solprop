# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Raman response models and the reference evaluation of the Raman polarisation.

The molecular models are exercised through the supplied route with solprop's former
Raman constants (``former_raman``, from its registry at dc4ca0d), on which this
module's expected values were recorded; fused silica's ``response_for`` is optmat's
bundle shape (SEP 3), and the intermediate-broadening model is driven by the former
Hollenbeck--Cantrell line set, which is also Luna's.
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import roots_legendre

from bounds import EPS, higham_gamma
from former_raman import (
    FORMER_CH4,
    FORMER_D2,
    FORMER_H2,
    FORMER_N2,
    FORMER_O2,
    FORMER_SF6,
    FORMER_SILICA_LINES,
    former_response,
    supplied_species,
)
from solprop import fourier
from solprop.constants import AMAGAT, EPS0, K_B, C
from solprop.exceptions import GridError, MaterialError
from solprop.nonlinear.raman import (
    DampedOscillator,
    Dephasing,
    IntermediateBroadening,
    MolecularOptions,
    RamanModel,
    RamanResponse,
    RotationalLines,
    SampledShape,
    _check_evaluate,
    normalised_damped_oscillator,
    raman_drive,
    response_for,
    rotational_nonrigid,
    undamped_dephasing,
    vibrational,
)
from solprop.nonlinear.responses import ResponseTraits
from solprop.physdata import (
    Gas,
    IntermediateBroadeningRaman,
    Linewidth,
    MolecularRaman,
    RotationalRaman,
)

# --------------------------------------------------------------------- dephasing

# The three zero-density limits, one gas each: H2 rotation has the Dicke term, N2
# rotation has only collisional broadening, and methane has a constant term.
DICKE = Linewidth(inverse_density=6.15e6, density=114e6)
COLLISIONAL = Linewidth(density=3.3e9)
CONSTANT = Linewidth(constant=8220e6)


def test_the_dephasing_time_matches_the_linewidth_at_positive_density():
    linewidth = Linewidth(constant=1e9, inverse_density=2e9, density=3e9)
    dephasing = Dephasing(linewidth)
    # Same quantity by two routes; the wrapper must not introduce its own arithmetic.
    assert dephasing.dephasing_time(2.0 * AMAGAT) == pytest.approx(
        linewidth.dephasing_time(2.0), rel=1e-15
    )


def test_the_rate_is_the_reciprocal_of_the_time():
    dephasing = Dephasing(DICKE)
    rate = dephasing.decay_rate(AMAGAT)
    assert rate * dephasing.dephasing_time(AMAGAT) == pytest.approx(1.0, rel=1e-15)


def test_a_dicke_narrowed_coherence_collapses_at_zero_density():
    # 1/tau2 = pi (A/n + B n) diverges as n -> 0, so the coherence has no lifetime.
    # A literal exp(-0/0) here would fill the polarisation with NaN.
    dephasing = Dephasing(DICKE)
    assert dephasing.decay_rate(0.0) == math.inf
    assert dephasing.dephasing_time(0.0) == 0.0
    damping = dephasing.damping(np.array([0.0, 1e-13, 1e-9]), 0.0)
    assert np.all(damping == 0.0)
    assert np.all(np.isfinite(damping))


def test_a_collisionally_broadened_coherence_is_undamped_at_zero_density():
    dephasing = Dephasing(COLLISIONAL)
    assert dephasing.decay_rate(0.0) == 0.0
    assert dephasing.dephasing_time(0.0) == math.inf
    assert np.all(dephasing.damping(np.array([0.0, 1e-9]), 0.0) == 1.0)


def test_a_constant_linewidth_keeps_its_time_at_zero_density():
    dephasing = Dephasing(CONSTANT)
    assert dephasing.dephasing_time(0.0) == pytest.approx(
        1.0 / (math.pi * 8220e6), rel=1e-15
    )


@pytest.mark.parametrize("linewidth", [DICKE, COLLISIONAL, CONSTANT])
@pytest.mark.parametrize("density", [0.0, 1e-6 * AMAGAT, AMAGAT, 100.0 * AMAGAT])
def test_the_damping_factor_is_always_a_finite_probability(linewidth, density):
    damping = Dephasing(linewidth).damping(np.linspace(0.0, 1e-9, 17), density)
    assert np.all(np.isfinite(damping))
    assert np.all((damping >= 0.0) & (damping <= 1.0))


def test_the_damping_factor_is_the_exponential_it_claims_to_be():
    dephasing = Dephasing(DICKE)
    t = np.linspace(0.0, 1e-10, 11)
    expected = np.exp(-t / dephasing.dephasing_time(AMAGAT))
    # Both sides are one exp of the same argument up to one division; two ulp.
    assert np.allclose(dephasing.damping(t, AMAGAT), expected, rtol=4e-16, atol=0.0)


@pytest.mark.parametrize("bad", [-1e-30, -1.0, -AMAGAT])
def test_a_negative_density_is_refused(bad):
    with pytest.raises(MaterialError, match="non-negative number density"):
        Dephasing(DICKE).decay_rate(bad)


def test_an_undamped_dephasing_never_decays():
    dephasing = undamped_dephasing()
    for density in (0.0, AMAGAT, 1e3 * AMAGAT):
        assert dephasing.decay_rate(density) == 0.0
    assert np.all(dephasing.damping(np.linspace(0.0, 1.0, 5), AMAGAT) == 1.0)


# -------------------------------------------------------------- damped oscillator


def test_the_damped_oscillator_is_a_raman_model():
    assert isinstance(DampedOscillator(1.0, 1e14, undamped_dephasing()), RamanModel)


def test_the_undamped_response_is_zero_at_and_before_the_origin():
    # Causality: nothing responds before, or at, the instant it is driven.
    line = DampedOscillator(3.0, 1e14, undamped_dephasing())
    t = np.array([-1e-12, -1e-15, 0.0])
    assert np.all(line.undamped(t) == 0.0)


def test_the_undamped_response_is_the_sine_it_claims_to_be():
    line = DampedOscillator(3.0, 1e14, undamped_dephasing())
    t = np.linspace(1e-16, 1e-13, 64)
    assert np.allclose(line.undamped(t), 3.0 * np.sin(1e14 * t), rtol=2e-16, atol=0.0)


def test_a_negative_amplitude_is_allowed():
    # Rotational line weights change sign with the population difference behind them.
    line = DampedOscillator(-1.0, 1.0, undamped_dephasing())
    assert float(line.undamped(np.array([math.pi / 2.0]))[0]) == -1.0


@pytest.mark.parametrize("bad", [0.0, -1e14, math.inf, math.nan])
def test_a_line_without_a_usable_frequency_is_refused(bad):
    with pytest.raises(MaterialError, match="positive, finite angular frequency"):
        DampedOscillator(1.0, bad, undamped_dephasing())


# ------------------------------------------------------- the normalised oscillator


def damped(line, t, density=1.0):
    """The full response of one line, undamped shape times its decay."""
    times = np.atleast_1d(np.asarray(t, dtype=float))
    return line.undamped(times) * line.dephasing.damping(times, density)


@pytest.mark.parametrize(
    ("amplitude", "frequency", "tau2"),
    [
        (1.0, 2.0 * math.pi * 100e12, 100e-15),
        (0.5, 2.0 * math.pi * 13e12, 32e-15),
        (-2.5, 2.0 * math.pi * 40e12, 3e-12),
    ],
)
def test_the_normalised_oscillator_integrates_to_its_amplitude(
    amplitude, frequency, tau2
):
    line = normalised_damped_oscillator(amplitude, frequency, tau2)
    # The integral is analytic: K Omega / (Omega^2 + 1/tau2^2). Exact in real
    # arithmetic, so only the round-off of the three products and one division.
    analytic = line.amplitude * frequency / (frequency**2 + 1.0 / tau2**2)
    assert analytic == pytest.approx(amplitude, rel=4e-16)


def test_the_normalised_oscillator_integrates_to_its_amplitude_by_quadrature():
    tau2 = 100e-15
    line = normalised_damped_oscillator(1.0, 2.0 * math.pi * 100e12, tau2)
    # Forty coherence times, so the truncated tail is exp(-40) = 4e-18 and the residual
    # is quadrature alone. quad's own reported error on this smooth integrand is then
    # about 1e-12, which is what the assertion below allows for.
    integral, error = quad(
        lambda t: float(damped(line, t)[0]),
        0.0,
        40.0 * tau2,
        limit=2000,
        epsabs=1e-14,
        epsrel=1e-12,
    )
    assert error < 1e-11
    assert integral == pytest.approx(1.0, rel=1e-10)


def test_the_normalised_oscillator_carries_the_dephasing_time_it_was_given():
    line = normalised_damped_oscillator(1.0, 1e14, 250e-15)
    for density in (0.0, AMAGAT, 1e3 * AMAGAT):
        assert line.dephasing.dephasing_time(density) == pytest.approx(
            250e-15, rel=1e-15
        )


@pytest.mark.parametrize("bad", [0.0, -1e-15])
def test_a_normalised_oscillator_needs_a_positive_dephasing_time(bad):
    with pytest.raises(MaterialError, match="positive dephasing time"):
        normalised_damped_oscillator(1.0, 1e14, bad)


@pytest.mark.parametrize("bad", [0.0, -1e14, math.nan])
def test_a_normalised_oscillator_needs_a_positive_frequency(bad):
    with pytest.raises(MaterialError, match="positive, finite angular frequency"):
        normalised_damped_oscillator(1.0, bad, 1e-13)


def gas_id(parameters: MolecularRaman) -> str:
    """Name a parametrised set of molecular constants after its gas."""
    return str(getattr(parameters.gas, "value", parameters.gas))


# solprop's former molecular constants (dc4ca0d), each a different linewidth shape.
FORMER_MOLECULES = [FORMER_N2, FORMER_O2, FORMER_H2, FORMER_D2, FORMER_CH4, FORMER_SF6]


# Real dephasing data of every shape, as a guard that the wrapper accepts them all.
@pytest.mark.parametrize("parameters", FORMER_MOLECULES, ids=gas_id)
def test_every_tabulated_linewidth_gives_a_finite_damping_at_every_density(
    parameters,
):
    both = (parameters.rotational, parameters.vibrational)
    parts = [part for part in both if part is not None]
    assert parts
    t = np.linspace(0.0, 1e-9, 33)
    for part in parts:
        for density in (0.0, 0.1 * AMAGAT, AMAGAT, 50.0 * AMAGAT):
            damping = Dephasing(part.linewidth).damping(t, density)
            assert np.all(np.isfinite(damping))


# ------------------------------------------------------------- the vibrational mode

VIBRATING = FORMER_MOLECULES


@pytest.mark.parametrize("parameters", VIBRATING, ids=gas_id)
def test_the_vibrational_line_is_the_modes_own_frequency(parameters):
    mode = parameters.vibrational
    assert mode is not None
    assert vibrational(mode).frequency == mode.frequency


@pytest.mark.parametrize("parameters", VIBRATING, ids=gas_id)
def test_the_vibrational_amplitude_is_the_closed_form(parameters):
    mode = parameters.vibrational
    assert mode is not None
    # K = (4 pi eps0 dalpha/dQ)^2 / (2 mu Omega eps0). Written here as the same
    # products in a different order, so the two differ by at most a few ulp.
    expected = (4.0 * math.pi * EPS0 * mode.polarisability_derivative) ** 2 / (
        2.0 * mode.reduced_mass * mode.frequency * EPS0
    )
    assert vibrational(mode).amplitude == pytest.approx(expected, rel=4e-16, abs=0.0)


@pytest.mark.parametrize("parameters", VIBRATING, ids=gas_id)
def test_the_vibrational_line_is_positive(parameters):
    # A vibration always adds gain on the Stokes side, so its weight cannot be signed.
    mode = parameters.vibrational
    assert mode is not None
    assert vibrational(mode).amplitude > 0.0


def test_the_vibrational_line_carries_the_modes_dephasing():
    mode = FORMER_H2.vibrational
    assert mode is not None
    assert vibrational(mode).dephasing.linewidth is mode.linewidth


# --------------------------------------------------------------- the rotational ladder

ROTATING = [FORMER_N2, FORMER_O2, FORMER_H2, FORMER_D2]


@pytest.mark.parametrize("gas", ROTATING, ids=gas_id)
def test_the_rotational_ladder_is_a_raman_model(gas):
    assert isinstance(rotational_of(gas), RamanModel)


def rotational_parameters(gas: MolecularRaman) -> RotationalRaman:
    """The rotational constants of a gas that has them."""
    part = gas.rotational
    assert part is not None
    return part


def rotational_of(gas: MolecularRaman, temperature=294.0):
    """The rotational ladder of a gas at a stated temperature."""
    return rotational_nonrigid(rotational_parameters(gas), temperature)


@pytest.mark.parametrize("gas", ROTATING, ids=gas_id)
def test_rotational_populations_sum_to_one(gas):
    total = float(rotational_of(gas).populations.sum())
    # A sum of up to 51 normalised terms: Higham's n*eps is 1.2e-14, and the divide
    # itself contributes one more rounding.
    assert total == pytest.approx(1.0, rel=1.2e-14)


@pytest.mark.parametrize("gas", ROTATING, ids=gas_id)
def test_rotational_populations_are_a_boltzmann_distribution(gas):
    parameters = rotational_parameters(gas)
    lines = rotational_nonrigid(parameters, 294.0)
    j = lines.quantum_numbers
    degeneracy = np.where(
        j % 2 == 1, parameters.degeneracy_odd, parameters.degeneracy_even
    )
    weights = degeneracy * (2 * j + 1) * np.exp(-lines.energies / (K_B * 294.0))
    expected = weights / weights.sum()
    # The same expression evaluated twice; only the order of the normalisation differs.
    assert np.allclose(lines.populations, expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("gas", ROTATING, ids=gas_id)
def test_rotational_line_frequencies_match_the_closed_form(gas):
    parameters = rotational_parameters(gas)
    lines = rotational_nonrigid(parameters, 294.0)
    j = lines.quantum_numbers[:-2].astype(float)
    # Omega_J = (E_{J+2} - E_J)/hbar with E_J = 2 pi hbar c (B J(J+1) - D (J(J+1))^2):
    # hbar cancels exactly, leaving 2 pi c times a polynomial in J.
    upper = (j + 2.0) * (j + 3.0)
    lower = j * (j + 1.0)
    expected = (
        2.0
        * math.pi
        * C
        * (parameters.b * (upper - lower) - parameters.d * (upper**2 - lower**2))
    )
    # A difference of two nearly equal energies: the cancellation costs a few digits
    # for the lowest lines, where upper - lower is 6 against energies of order 1e2.
    assert np.allclose(lines.frequencies, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("gas", ROTATING, ids=gas_id)
def test_rotational_line_amplitudes_are_positive(gas):
    # K is negative and the population difference behind each line is too, so every
    # line comes out positive: the sign that amplifies the Stokes side.
    lines = rotational_of(gas)
    allowed = lines.populations[:-2] > 0.0
    assert np.all(lines.amplitudes[allowed] > 0.0)
    np.testing.assert_array_equal(lines.amplitudes[~allowed], 0.0)


def test_the_energy_ladder_is_truncated_where_it_stops_increasing():
    # H2 has a large centrifugal distortion: B J(J+1) - D (J(J+1))^2 turns over at
    # J(J+1) = B/(2D) = 589, that is J = 24, and levels beyond it are an artefact of
    # the expansion rather than states of the molecule.
    lines = rotational_of(FORMER_H2)
    assert int(lines.quantum_numbers[-1]) == 24
    assert np.all(np.diff(lines.energies) > 0.0)


def test_a_ladder_without_distortion_keeps_every_level_asked_for():
    parameters = RotationalRaman(
        b=199.0,
        d=0.0,
        degeneracy_odd=1,
        degeneracy_even=2,
        polarisability_anisotropy=6.7e-31,
        linewidth=Linewidth(density=3.3e9),
        citation="test",
    )
    lines = rotational_nonrigid(parameters, 294.0, min_j=0, max_j=30)
    assert lines.quantum_numbers.size == 31
    assert lines.n_lines == 29


def test_the_ladder_can_start_above_zero():
    lines = rotational_nonrigid(
        rotational_parameters(FORMER_N2), 294.0, min_j=2, max_j=20
    )
    assert int(lines.quantum_numbers[0]) == 2
    assert lines.quantum_numbers.size == 19


def test_a_hotter_gas_populates_higher_levels():
    cold = rotational_of(FORMER_N2, 100.0)
    hot = rotational_of(FORMER_N2, 1000.0)
    # The mean J rises with temperature; nothing subtler is needed to catch a sign or
    # a missing k_B in the exponent.
    mean_cold = float(cold.populations @ cold.quantum_numbers)
    mean_hot = float(hot.populations @ hot.quantum_numbers)
    assert mean_hot > 3.0 * mean_cold


@pytest.mark.parametrize("bad", [0.0, -294.0, math.inf, math.nan])
def test_a_ladder_needs_a_usable_temperature(bad):
    with pytest.raises(MaterialError, match="positive, finite temperature"):
        rotational_nonrigid(rotational_parameters(FORMER_N2), bad)


@pytest.mark.parametrize(("min_j", "max_j"), [(-1, 10), (5, 4), (0, -1)])
def test_a_ladder_needs_an_ordered_non_negative_range(min_j, max_j):
    with pytest.raises(MaterialError, match="0 <= min_j <= max_j"):
        rotational_nonrigid(
            rotational_parameters(FORMER_N2), 294.0, min_j=min_j, max_j=max_j
        )


def test_a_ladder_that_turns_over_too_early_is_refused():
    # A distortion constant comparable with B makes the energy fall after two levels,
    # so no J, J+2 pair survives. Luna's guard compares a J value with an array index
    # and gets this wrong for any min_j other than zero.
    parameters = RotationalRaman(
        b=199.0,
        d=60.0,
        degeneracy_odd=1,
        degeneracy_even=2,
        polarisability_anisotropy=6.7e-31,
        linewidth=Linewidth(density=3.3e9),
        citation="test",
    )
    with pytest.raises(MaterialError, match="stops increasing"):
        rotational_nonrigid(parameters, 294.0)


def test_a_ladder_with_too_short_a_range_is_refused():
    with pytest.raises(MaterialError, match="at least 3 levels"):
        rotational_nonrigid(rotational_parameters(FORMER_N2), 294.0, min_j=0, max_j=1)


def test_the_ladder_arrays_are_read_only():
    lines = rotational_of(FORMER_N2)
    with pytest.raises(ValueError, match="read-only"):
        lines.populations[0] = 0.0


def test_the_ladder_response_is_the_sum_of_its_lines():
    lines = rotational_of(FORMER_H2)
    t = np.linspace(1e-15, 5e-13, 129)
    expected = np.zeros_like(t)
    magnitude = np.zeros_like(t)
    for amplitude, frequency in zip(lines.amplitudes, lines.frequencies, strict=True):
        expected += amplitude * np.sin(frequency * t)
        magnitude += np.abs(amplitude * np.sin(frequency * t))
    # The same 23-term sum of signed terms: each side within gamma(23) of the sum of
    # their magnitudes, whatever order it adds them in, and a sine's ulp apiece.
    count = lines.amplitudes.size
    limit = 2 * (higham_gamma(count) + 2 * EPS) * magnitude
    assert np.all(np.abs(lines.undamped(t) - expected) <= limit)
    assert np.any(expected)


def test_the_ladder_response_is_causal():
    lines = rotational_of(FORMER_H2)
    assert np.all(lines.undamped(np.array([-1e-12, -1e-15, 0.0])) == 0.0)


# ------------------------------------------------------- the intermediate broadening


def test_silica_is_a_raman_model():
    assert isinstance(IntermediateBroadening(FORMER_SILICA_LINES), RamanModel)


def test_the_silica_response_does_not_depend_on_density():
    model = IntermediateBroadening(FORMER_SILICA_LINES)
    t = np.linspace(0.0, 1e-12, 9)
    assert np.all(model.dephasing.damping(t, 0.0) == 1.0)
    assert np.all(model.dephasing.damping(t, 1e28) == 1.0)


def gauss_legendre_integral(model, upper, panels=100_000, order=16):
    """A composite Gauss-Legendre integral of a model's undamped response.

    Adaptive quadrature is the wrong tool here: the integrand crosses zero thousands
    of times over its support and `scipy.integrate.quad` misses the oscillations
    entirely if the interval is much longer than the decay.
    """
    nodes, weights = roots_legendre(order)
    edges = np.linspace(0.0, upper, panels + 1)
    half = 0.5 * np.diff(edges)
    middle = 0.5 * (edges[:-1] + edges[1:])
    points = (middle[:, None] + half[:, None] * nodes[None, :]).ravel()
    scaled = (half[:, None] * weights[None, :]).ravel()
    return float(np.dot(model.undamped(points), scaled))


@pytest.mark.parametrize("amplitude", [1.0, 2.5e-30])
def test_the_analytic_normalisation_agrees_with_quadrature(amplitude):
    model = IntermediateBroadening(FORMER_SILICA_LINES, amplitude=amplitude)
    # 50 ps is forty times the slowest Lorentzian decay (1/gamma_min = 1.3 ps), so the
    # truncated tail is below 1e-17 of the peak and the residual is the rule's own.
    # The bound is 1e-8; the measured agreement is 7e-16.
    quadrature = gauss_legendre_integral(model, 5e-11)
    assert quadrature == pytest.approx(amplitude, rel=1e-8)


def test_the_silica_response_integrates_to_its_amplitude():
    model = IntermediateBroadening(FORMER_SILICA_LINES, amplitude=2.5e-30)
    assert model.scale * model.raw_integral == pytest.approx(2.5e-30, rel=4e-16)


def test_the_silica_scale_reproduces_lunas_own_normalisation():
    # A historical witness on supplied data: the Hollenbeck-Cantrell line set Luna
    # carries, as solprop's former registry held it (dc4ca0d; FORMER_SILICA_LINES).
    # Luna fixes the same factor by adaptive quadrature to 1 ns (Raman.jl:91). Its
    # answer is right; only the method is fragile. Recorded here as a plain
    # number rather than through the witness file because it is a pure shape factor,
    # free of any physical constant and so of any CODATA difference.
    assert IntermediateBroadening(FORMER_SILICA_LINES).scale == pytest.approx(
        2.664828467563587e11, rel=1e-10
    )


def test_the_silica_response_is_causal():
    model = IntermediateBroadening(FORMER_SILICA_LINES)
    assert np.all(model.undamped(np.array([-1e-12, -1e-15, 0.0])) == 0.0)


def test_the_silica_response_decays_within_a_picosecond():
    model = IntermediateBroadening(FORMER_SILICA_LINES)
    t = np.linspace(0.0, 5e-12, 4001)
    response = np.abs(model.undamped(t))
    assert response.max() > 0.0
    # Measured: 2.9e-3 of the peak survives past 1 ps and 1.7e-8 past 3 ps. Silica's
    # coherence is picoseconds where a gas rotation lives for nanoseconds, which is
    # why the doubled convolution grid matters for gases and not for glass.
    assert response[t > 1e-12].max() < 1e-2 * response.max()
    assert response[t > 3e-12].max() < 1e-6 * response.max()


def test_a_line_set_that_integrates_to_zero_cannot_be_normalised():
    # Two identical lines of opposite sign: a legal line set with no scale.
    lines = IntermediateBroadeningRaman(
        frequencies=(1e14, 1e14),
        amplitudes=(1.0, -1.0),
        gaussian_widths=(1e13, 1e13),
        lorentzian_widths=(1e12, 1e12),
        citation="test",
    )
    with pytest.raises(MaterialError, match="integrates to zero"):
        _ = IntermediateBroadening(lines).scale


# ---------------------------------------------------------------------- the drive


def test_the_field_drive_with_the_third_harmonic_is_the_plain_square():
    field = np.array([[0.0, 1.0, -2.0, 3.0]])
    assert np.array_equal(
        raman_drive(field, kind="field", thg=True), np.array([0.0, 1.0, 4.0, 9.0])
    )


def test_the_envelope_drive_is_half_the_modulus_squared():
    envelope = np.array([[3.0 + 4.0j, 1.0 - 1.0j]])
    # Exact: the drive sums the squares of the parts rather than squaring a modulus,
    # so no square root is taken and these integer-valued cases round to themselves.
    assert np.array_equal(
        raman_drive(envelope, kind="envelope", thg=True), np.array([12.5, 1.0])
    )


def test_the_envelope_drive_ignores_the_third_harmonic_flag():
    envelope = np.array([[3.0 + 4.0j, 1.0 - 1.0j]])
    assert np.array_equal(
        raman_drive(envelope, kind="envelope", thg=True),
        raman_drive(envelope, kind="envelope", thg=False),
    )


def test_the_field_drive_without_the_third_harmonic_has_no_beat():
    # E^2 of a carrier has a 2 omega0 beat; |E_a|^2 / 2 does not, and that is the
    # whole of the difference between the two field-grid forms.
    t = (np.arange(2048) - 1024) * 2e-16
    omega0 = 2.0 * np.pi * C / 1030e-9
    # 30 fs in a 410 fs window: the Gaussian is 5e-21 at the edge, so the analytic
    # signal is limited only by the overlap of the two frequency lobes, which for
    # this carrier is far below double precision.
    envelope = np.exp(-((t / 3e-14) ** 2))
    field = (envelope * np.cos(omega0 * t))[None, :]
    with_beat = raman_drive(field, kind="field", thg=True)
    without = raman_drive(field, kind="field", thg=False)
    peak = float(np.max(without))
    assert np.allclose(without, 0.5 * envelope**2, rtol=0.0, atol=1e-13 * peak)
    assert np.max(np.abs(with_beat - without)) > 0.2 * peak


# ------------------------------------------------------------------------ traits


def test_traits_are_declared_as_the_design_requires():
    assert former_response(FORMER_N2).traits == ResponseTraits(
        domain="convolution",
        grids=("field", "envelope"),
        polarisation="tensor",
        z_profiles=("density",),
    )


def test_traits_distinguish_tensor_capability_independently_of_thg():
    assert (
        former_response(FORMER_H2).traits
        == former_response(FORMER_H2, thg=False).traits
    )
    assert response_for("SiO2", thg=False).traits.polarisation == "scalar"


# --------------------------------------------------------------------- the factory


def test_a_diatomic_gas_gets_a_rotational_and_a_vibrational_term():
    models = former_response(FORMER_H2).models
    assert len(models) == 2
    assert isinstance(models[0], RotationalLines)
    assert isinstance(models[1], DampedOscillator)


def test_a_spherical_top_gets_only_a_vibrational_term():
    models = former_response(FORMER_CH4).models
    assert len(models) == 1
    assert isinstance(models[0], DampedOscillator)


def test_the_parts_can_be_switched_off_individually():
    only_rotation = former_response(FORMER_N2, MolecularOptions(vibration=False))
    only_vibration = former_response(FORMER_N2, MolecularOptions(rotation=False))
    assert len(only_rotation.models) == len(only_vibration.models) == 1
    assert isinstance(only_rotation.models[0], RotationalLines)
    assert isinstance(only_vibration.models[0], DampedOscillator)


def test_switching_everything_off_is_refused():
    with pytest.raises(MaterialError, match="has no Raman response"):
        former_response(FORMER_N2, MolecularOptions(rotation=False, vibration=False))


def test_a_spherical_top_without_its_vibration_is_refused():
    # Rotation left at its default takes the ladder only when the species has one;
    # methane has none, so nothing remains once the vibration is switched off.
    with pytest.raises(MaterialError, match="no rotational spectrum"):
        former_response(FORMER_CH4, MolecularOptions(vibration=False))
    with pytest.raises(MaterialError, match="selected rotational term"):
        former_response(FORMER_CH4, MolecularOptions(rotation=True, vibration=False))


@pytest.mark.parametrize("name", ["SiO2", "silica", "FUSED_SILICA", " Silica "])
def test_silica_is_recognised_by_several_spellings(name):
    # optmat's silica bundle shape since SEP 3; the former intermediate-broadening
    # line set is a caller's data now (IntermediateBroadening(FORMER_SILICA_LINES)).
    assert isinstance(response_for(name).models[0], SampledShape)


def test_the_silica_amplitude_reaches_the_model():
    response = response_for("SiO2", amplitude=3e-30)
    model = response.models[0]
    assert isinstance(model, SampledShape)
    assert model.amplitude == 3e-30


def test_a_fraction_is_refused_for_a_molecular_gas():
    # The molecular models are absolute, so a fraction would scale them twice: a
    # caller's constants and optmat's built-in response alike.
    with pytest.raises(MaterialError, match="already absolute"):
        response_for(supplied_species(FORMER_H2), fraction=0.18)
    with pytest.raises(MaterialError, match="is absolute; a fraction would scale"):
        response_for(Gas.H2, fraction=0.18)


def test_an_unknown_material_is_refused():
    with pytest.raises(MaterialError):
        response_for("unobtainium")


def test_a_monatomic_gas_has_no_raman_response():
    with pytest.raises(
        MaterialError, match="no molecular Raman response: it is an atom"
    ):
        response_for(Gas.Ar)


def test_a_response_needs_at_least_one_model():
    with pytest.raises(MaterialError, match="at least one model"):
        RamanResponse(models=())


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_a_fraction_outside_the_unit_interval_is_refused(bad):
    with pytest.raises(MaterialError, match=r"must lie in \(0, 1\]"):
        RamanResponse(models=response_for("SiO2").models, fraction=bad)


def test_the_fraction_scales_the_response():
    t = np.linspace(1e-15, 1e-12, 257)
    whole = response_for("SiO2").impulse(t, 1.0)
    part = response_for("SiO2", fraction=0.18).impulse(t, 1.0)
    assert np.allclose(part, 0.18 * whole, rtol=2e-16, atol=0.0)


# ------------------------------------------------------- the reference convolution


def gaussian_field(n_to=64, dt=2e-15, width=8e-15, amplitude=1e9):
    """A real Gaussian pulse envelope on a short grid, in V/m."""
    t = (np.arange(n_to) - n_to // 2) * dt
    return (amplitude * np.exp(-((t / width) ** 2)))[None, :]


def direct_convolution(
    response, field, density, dt, *, kind: fourier.GridKind = "field"
):
    """The convolution of the reference evaluate, as an explicit O(n^2) sum."""
    n_to = field.shape[1]
    kernel = response.impulse(np.arange(n_to) * dt, density)
    drive = raman_drive(field, kind=kind, thg=response.thg)
    return np.array(
        [float(np.dot(kernel[: i + 1], drive[i::-1])) * dt for i in range(n_to)]
    )


def test_the_reference_convolution_is_the_direct_sum():
    response = former_response(FORMER_N2)
    field = gaussian_field()
    density = 2.5e25
    out = response.evaluate(field, density, kind="field", dt=2e-15)
    expected = (
        EPS0 * density * field[0] * direct_convolution(response, field, density, 2e-15)
    )
    # A 64-point transform pair against a 64-term dot product: the transform's error
    # is a few times log2(128) * eps relative to the largest partial sum, and the
    # response oscillates through zero, so the bound is relative to the peak.
    assert np.max(np.abs(out[0] - expected)) < 1e-12 * np.max(np.abs(expected))
    assert np.max(np.abs(expected)) > 0.0


def test_the_reference_convolution_is_the_direct_sum_on_an_envelope():
    response = former_response(FORMER_H2)
    field = gaussian_field().astype(complex)
    density = 1.0e25
    out = response.evaluate(field, density, kind="envelope", dt=2e-15)
    expected = (
        EPS0
        * density
        * field[0]
        * direct_convolution(response, field, density, 2e-15, kind="envelope")
    )
    assert np.max(np.abs(out[0] - expected)) < 1e-12 * np.max(np.abs(expected))


def test_the_polarisation_is_causal():
    # Two fields that agree up to a sample and differ after it must give the same
    # polarisation up to that sample: the response cannot reach backwards in time.
    response = former_response(FORMER_N2)
    field = gaussian_field(n_to=128, dt=2e-15, width=3e-14)
    disturbed = field.copy()
    cut = 80
    disturbed[0, cut:] *= 3.0
    first = response.evaluate(field, 2.5e25, kind="field", dt=2e-15)
    second = response.evaluate(disturbed, 2.5e25, kind="field", dt=2e-15)
    scale = np.max(np.abs(first))
    assert scale > 0.0
    # The doubled grid makes the transform product an exact linear convolution, so
    # what is left before the cut is the transform pair's own round-off.
    assert np.max(np.abs(first[0, :cut] - second[0, :cut])) < 1e-13 * scale
    assert np.max(np.abs(first[0, cut:] - second[0, cut:])) > 0.1 * scale


def test_the_polarisation_is_linear_in_the_density_at_fixed_dephasing():
    # rho appears twice: outside as a factor and inside the dephasing. A model with no
    # density-dependent linewidth isolates the first.
    response = response_for("SiO2")
    field = gaussian_field()
    one = response.evaluate(field, 1e25, kind="field", dt=2e-15)
    two = response.evaluate(field, 2e25, kind="field", dt=2e-15)
    assert np.allclose(two, 2.0 * one, rtol=1e-15, atol=0.0)


def test_the_polarisation_vanishes_at_zero_density():
    response = former_response(FORMER_H2)
    out = response.evaluate(gaussian_field(), 0.0, kind="field", dt=2e-15)
    assert np.all(out == 0.0)


# ------------------------------------------------------------------- error paths


def test_a_scalar_only_vector_field_is_refused():
    response = RamanResponse(models=former_response(FORMER_N2).models)
    with pytest.raises(GridError, match="vector Raman requires tensor terms"):
        response.evaluate(np.zeros((2, 16)), 1e25, kind="field", dt=1e-15)


def test_an_unknown_grid_is_refused():
    # Through the validator directly: the test session's beartype hook would reject a
    # bad literal first, so the run-time guard would never be reached.
    with pytest.raises(GridError, match="kind must be"):
        _check_evaluate(np.zeros((1, 16)), "spectral", 1e-15)


def test_a_complex_field_grid_is_refused():
    with pytest.raises(GridError, match="carries the real field"):
        former_response(FORMER_N2).evaluate(
            np.zeros((1, 16), dtype=complex), 1e25, kind="field", dt=1e-15
        )


def test_a_real_envelope_grid_is_refused():
    with pytest.raises(GridError, match="carries a complex envelope"):
        former_response(FORMER_N2).evaluate(
            np.zeros((1, 16)), 1e25, kind="envelope", dt=1e-15
        )


@pytest.mark.parametrize("shape", [(16,), (1, 2, 16)])
def test_a_field_that_is_not_two_dimensional_is_refused(shape):
    with pytest.raises(GridError, match=r"shape \(n_pol, n_to\)"):
        _check_evaluate(np.zeros(shape), "field", 1e-15)


@pytest.mark.parametrize("bad", [0.0, -1e-15, math.inf, math.nan])
def test_a_bad_sample_spacing_is_refused(bad):
    with pytest.raises(GridError, match="positive, finite sample spacing"):
        former_response(FORMER_N2).evaluate(
            np.zeros((1, 16)), 1e25, kind="field", dt=bad
        )


def test_a_negative_density_is_refused_by_evaluate():
    with pytest.raises(MaterialError, match="non-negative number density"):
        former_response(FORMER_N2).evaluate(
            np.zeros((1, 16)), -1.0, kind="field", dt=1e-15
        )


def test_a_grid_too_short_to_convolve_is_refused():
    with pytest.raises(GridError, match="at least two samples"):
        former_response(FORMER_N2).evaluate(
            np.zeros((1, 1)), 1e25, kind="field", dt=1e-15
        )


# ------------------------------------------------------------ the sign of the gain

# A window long enough to resolve a Raman line: 65.5 ps doubled to 131 ps gives a bin
# spacing of 4.79e10 rad/s, which is 1.1e-4 of the N2 vibration at 4.39e14 rad/s.
GAIN_N_TO = 65536
GAIN_DT = 1e-15
GAIN_RTOL = 3e-4


def kernel_spectrum_of(response, density, n_to=GAIN_N_TO, dt=GAIN_DT):
    """The kernel spectrum on the doubled grid, and its angular frequency axis."""
    kernel = np.zeros(2 * n_to)
    kernel[:n_to] = response.impulse(np.arange(n_to) * dt, density)
    return np.fft.rfftfreq(2 * n_to, dt) * 2.0 * np.pi, np.fft.rfft(kernel) * dt


def test_the_kernel_spectrum_peaks_at_the_vibrational_line():
    # Luna's own test locates the resonance the same way, at the *minimum* of the
    # imaginary part (test_raman.jl:31), which is the sign that amplifies the Stokes
    # side in the internal FFT convention: the Stokes growth rate is -Im H(Omega).
    mode = FORMER_N2.vibrational
    assert mode is not None
    response = former_response(FORMER_N2, MolecularOptions(rotation=False))
    omega, spectrum = kernel_spectrum_of(response, 2.5e25)
    peak = omega[int(np.argmin(np.imag(spectrum)))]
    assert peak == pytest.approx(mode.frequency, rel=GAIN_RTOL)


def test_the_kernel_spectrum_is_negative_imaginary_at_the_vibrational_line():
    mode = FORMER_N2.vibrational
    assert mode is not None
    response = former_response(FORMER_N2, MolecularOptions(rotation=False))
    omega, spectrum = kernel_spectrum_of(response, 2.5e25)
    at_line = int(np.argmin(np.abs(omega - mode.frequency)))
    # Im H < 0 is gain on the Stokes side and loss on the anti-Stokes side. A positive
    # value here would mean the model absorbed where it should amplify.
    assert np.imag(spectrum[at_line]) < 0.0
    assert np.imag(spectrum[at_line]) == pytest.approx(
        float(np.imag(spectrum).min()), rel=1e-3
    )


def test_the_kernel_spectrum_peaks_at_the_strongest_rotational_line():
    lines = rotational_of(FORMER_H2)
    strongest = float(lines.frequencies[int(np.argmax(lines.amplitudes))])
    response = former_response(FORMER_H2, MolecularOptions(vibration=False))
    omega, spectrum = kernel_spectrum_of(response, 2.5e25)
    peak = omega[int(np.argmin(np.imag(spectrum)))]
    assert peak == pytest.approx(strongest, rel=GAIN_RTOL)
    # Luna asserts 2 pi * 17.45 THz for the same line (test_raman.jl:67).
    assert peak == pytest.approx(2.0 * np.pi * 17.45e12, rel=1e-3)


def test_every_rotational_line_shows_as_a_dip_in_the_imaginary_part():
    lines = rotational_of(FORMER_H2)
    response = former_response(FORMER_H2, MolecularOptions(vibration=False))
    omega, spectrum = kernel_spectrum_of(response, 2.5e25)
    imaginary = np.imag(spectrum)
    floor = float(np.median(imaginary))
    # The four strongest lines, each at its own frequency and each a dip.
    order = np.argsort(lines.amplitudes)[::-1][:4]
    for index in order:
        at_line = int(np.argmin(np.abs(omega - lines.frequencies[index])))
        assert imaginary[at_line] < floor


def test_the_kernel_spectrum_is_the_conjugate_symmetric_one_of_a_real_kernel():
    # H(-Omega) = conj(H(Omega)) is what makes the drive at the Stokes offset pick up
    # conj(H) and the anti-Stokes one pick up H, so gain on one side is loss on the
    # other. A one-sided transform states it; this checks the kernel really is real.
    response = response_for("SiO2")
    kernel = np.zeros(2048)
    kernel[:1024] = response.impulse(np.arange(1024) * 2e-15, 1.0)
    assert not np.iscomplexobj(kernel)
    full = np.fft.fft(kernel)
    # Exact in exact arithmetic; what is left is the transform's own round-off,
    # bounded by a few times log2(2048) * eps of the largest bin.
    scale = float(np.max(np.abs(full)))
    assert np.max(np.abs(full[1:1024] - np.conj(full[-1:-1024:-1]))) < 1e-14 * scale


# ------------------------------------------------------- field and envelope parity

PARITY_N_TO = 8192
PARITY_DT = 2e-16
PARITY_DURATION = 100e-15
# A 100 fs pulse in a 1.64 ps window: the Gaussian is 8e-30 at the edge and the
# overlap between the positive and negative frequency lobes at 1030 nm is far below
# double precision, so the analytic signal recovers the envelope exactly and the
# parity is an algebraic identity. The bound is 1e-8; measured worst is 4e-15.
PARITY_ATOL_FRACTION = 1e-10


def parity_signals(amplitude=1e9):
    """A real carrier-resolved field and the envelope it is the real part of."""
    t = (np.arange(PARITY_N_TO) - PARITY_N_TO // 2) * PARITY_DT
    omega0 = 2.0 * np.pi * C / 1030e-9
    envelope = amplitude * np.exp(-((t / PARITY_DURATION) ** 2))
    field = envelope * np.cos(omega0 * t)
    return t, omega0, field[None, :], envelope.astype(complex)[None, :]


@pytest.mark.parametrize(
    "response",
    [
        former_response(FORMER_N2, thg=False),
        former_response(FORMER_O2, thg=False),
        former_response(FORMER_H2, thg=False),
        response_for("SiO2", thg=False),
    ],
    ids=["N2", "O2", "H2", "SiO2"],
)
def test_the_two_grids_give_the_same_polarisation(response):
    t, omega0, field, envelope = parity_signals()
    density = 2.5e25
    on_field = response.evaluate(field, density, kind="field", dt=PARITY_DT)
    on_envelope = response.evaluate(envelope, density, kind="envelope", dt=PARITY_DT)
    rebuilt = np.real(on_envelope * np.exp(1j * omega0 * t))
    scale = float(np.max(np.abs(on_field)))
    assert scale > 0.0
    assert np.max(np.abs(on_field - rebuilt)) < PARITY_ATOL_FRACTION * scale


def test_the_response_cannot_follow_the_third_harmonic_beat():
    # The two field-grid drives differ by the 2 omega0 beat, which for a 1030 nm
    # carrier is at 3.7e15 rad/s -- ten times the highest Raman line. The response is
    # a resonant low pass, so it filters the beat out almost entirely and the two
    # forms give nearly the same polarisation. Measured: 2.4e-4 of the peak. This is
    # why the thg switch matters much less for Raman than for Kerr, and why the
    # difference must nevertheless not be exactly zero.
    _, _, field, _ = parity_signals()
    density = 2.5e25
    first = former_response(FORMER_N2, thg=True).evaluate(
        field, density, kind="field", dt=PARITY_DT
    )
    second = former_response(FORMER_N2, thg=False).evaluate(
        field, density, kind="field", dt=PARITY_DT
    )
    relative = float(np.max(np.abs(first - second)) / np.max(np.abs(first)))
    assert 1e-8 < relative < 1e-3


# ------------------------------------------ the ladder's own shape validation


# Built by hand rather than through rotational_nonrigid: the axis names in the
# annotations are independent, exactly so that this validation stays reachable in
# production, where no runtime type checker is installed (AGENTS.md, "Types and
# arrays"). The runtime checker in this session therefore lets these through to it.
def hand_built_ladder(
    quantum_numbers=None,
    energies=None,
    populations=None,
    frequencies=None,
    amplitudes=None,
) -> RotationalLines:
    """A four-level ladder with two lines, with any array replaced."""
    return RotationalLines(
        quantum_numbers=np.arange(4) if quantum_numbers is None else quantum_numbers,
        energies=np.arange(4, dtype=float) if energies is None else energies,
        populations=np.full(4, 0.25) if populations is None else populations,
        frequencies=np.array([1e14, 2e14]) if frequencies is None else frequencies,
        amplitudes=np.array([1.0, 2.0]) if amplitudes is None else amplitudes,
        dephasing=undamped_dephasing(),
        temperature=294.0,
    )


def test_a_hand_built_ladder_with_matching_shapes_is_accepted():
    assert hand_built_ladder().n_lines == 2


def test_a_ladder_with_fewer_than_three_levels_is_refused():
    with pytest.raises(MaterialError, match="at least 3 levels"):
        hand_built_ladder(
            quantum_numbers=np.arange(2),
            energies=np.arange(2, dtype=float),
            populations=np.full(2, 0.5),
            frequencies=np.array([]),
            amplitudes=np.array([]),
        )


def test_a_ladder_whose_level_arrays_disagree_is_refused():
    with pytest.raises(MaterialError, match="level arrays must all have"):
        hand_built_ladder(populations=np.full(3, 1.0 / 3.0))


def test_a_ladder_whose_line_arrays_disagree_is_refused():
    with pytest.raises(MaterialError, match="has 2 lines, got"):
        hand_built_ladder(amplitudes=np.array([1.0, 2.0, 3.0]))
