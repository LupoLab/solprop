# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.signal import hilbert

from solprop import fourier
from solprop.constants import HBAR
from solprop.exceptions import GridError

# Machine epsilon for float64. Tolerances below are quoted as a multiple of this times
# the length of the reduction, which is Higham's bound for a floating-point sum.
EPS = np.finfo(np.float64).eps

KINDS = ("field", "envelope")


def make_spectrum(kind, n_t, seed=0, windowed=True):
    """Random spectrum of the given layout, optionally with its end bins zeroed."""
    rng = np.random.default_rng(seed)
    n_omega = fourier.n_omega_for(n_t, kind=kind)
    spectrum = rng.standard_normal(n_omega) + 1j * rng.standard_normal(n_omega)
    if windowed and kind == "field":
        # A physical field grid apodises the zero-frequency and Nyquist bins to zero;
        # several identities below hold exactly only there (see spectral_density).
        spectrum[0] = 0.0
        spectrum[-1] = 0.0
    return spectrum


# --- the sign convention -------------------------------------------------------------


@pytest.mark.parametrize(
    "array",
    [
        np.array([1 + 2j, 3 - 1j, -0.5 + 0j]),  # a real field's one-sided spectrum
        np.array([1 + 1j, 2 - 3j, 0j, -1 + 0.5j]),  # an envelope spectrum, FFT order
        np.array([0.5 - 2j, 1 + 0j, -3 + 1j, 2j]),  # the same envelope in time
    ],
    ids=["field-spectrum", "envelope-spectrum", "envelope-time"],
)
def test_to_optics_equals_conjugation_and_is_involutive(array):
    # Conjugation flips a sign bit: exact, a structural contract.
    assert np.array_equal(fourier.to_optics(array), np.conj(array))
    assert np.array_equal(fourier.to_internal(array), np.conj(array))
    assert np.array_equal(fourier.to_internal(fourier.to_optics(array)), array)
    assert np.array_equal(fourier.to_optics(fourier.to_internal(array)), array)


def test_to_optics_leaves_a_real_field_alone():
    field = np.array([1.0, -2.0, 3.0, 0.5])
    assert np.array_equal(fourier.to_optics(field), field)


# --- the spectrum phase origin --------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n_t", [4, 8, 16])
def test_spectrum_helpers_are_involutive(kind, n_t):
    spectrum = make_spectrum(kind, n_t, seed=n_t)
    to_optics = fourier.spectrum_to_optics(spectrum, kind=kind)
    to_internal = fourier.spectrum_to_internal(spectrum, kind=kind)
    # Both steps are their own inverse: conjugation, and a multiplication by +-1 that
    # is exact in binary floating point. Exact equality is the contract.
    assert np.array_equal(to_optics, to_internal)
    assert np.array_equal(fourier.spectrum_to_internal(to_optics, kind=kind), spectrum)
    assert np.array_equal(fourier.spectrum_to_optics(to_internal, kind=kind), spectrum)


@pytest.mark.parametrize("kind", KINDS)
def test_the_spectrum_helpers_add_the_alternating_sign_to_the_conjugation(kind):
    n_t = 8
    spectrum = make_spectrum(kind, n_t, seed=3)
    sign = np.array([(-1.0) ** k for k in range(spectrum.size)])
    # Exact: sign is exactly +-1, so the product reassociates without error.
    assert np.array_equal(
        fourier.spectrum_to_optics(spectrum, kind=kind),
        sign * fourier.to_optics(spectrum),
    )


@pytest.mark.parametrize("kind", KINDS)
def test_a_flat_optics_spectrum_is_a_pulse_at_the_window_centre(kind):
    # The whole point of the origin shift: a spectrum that is flat and real in the
    # optics convention is a transform-limited pulse sitting at t = 0, which is sample
    # n_t/2 of the stored array (conventions.md section 2).
    n_t = 16
    n_omega = fourier.n_omega_for(n_t, kind=kind)
    flat = np.ones(n_omega, dtype=complex)
    samples = fourier.inverse(fourier.spectrum_to_internal(flat, kind=kind), kind=kind)
    assert int(np.argmax(np.abs(samples))) == n_t // 2


@pytest.mark.parametrize("kind", KINDS)
def test_the_origin_shift_commutes_with_a_delay_and_a_taylor_phase(kind):
    # A pure delay and a Taylor phase about a reference frequency may be applied to a
    # stored spectrum directly, precisely because they commute with the origin shift.
    # Without that, every builder would need the round trip through the helpers.
    n_t = 32
    spectrum = make_spectrum(kind, n_t, seed=11)
    d_omega = 2.0 * np.pi / n_t
    omega = (
        np.arange(n_t // 2 + 1) * d_omega
        if kind == "field"
        else (np.fft.fftfreq(n_t) * 2.0 * np.pi + 40.0 * d_omega)
    )
    phase = 0.37 * omega + 0.11 * (omega - 40.0 * d_omega) ** 2
    shifted_then_phased = np.exp(1j * phase) * fourier.spectrum_to_optics(
        spectrum, kind=kind
    )
    phased_then_shifted = fourier.spectrum_to_optics(
        np.exp(-1j * phase) * spectrum, kind=kind
    )
    # Two orderings of the same three exact-magnitude factors; only the complex
    # multiplications round, and there are three of them per bin.
    assert_allclose(shifted_then_phased, phased_then_shifted, rtol=8 * EPS, atol=0.0)


@pytest.mark.parametrize(("kind", "n_omega"), [("field", 2), ("envelope", 7)])
def test_the_spectrum_helpers_reject_a_bad_layout(kind, n_omega):
    for helper in (fourier.spectrum_to_optics, fourier.spectrum_to_internal):
        with pytest.raises(GridError, match="which is not an even integer"):
            helper(np.ones(n_omega, dtype=complex), kind=kind)


# --- transforms -----------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n_t", [4, 16, 64])
def test_forward_inverse_round_trip(kind, n_t):
    rng = np.random.default_rng(3)
    samples = rng.standard_normal(n_t)
    if kind == "envelope":
        samples = samples + 1j * rng.standard_normal(n_t)
    recovered = fourier.inverse(fourier.forward(samples, kind=kind), kind=kind)
    # A DFT pair loses at most a few epsilon per term of the transform's inner sum.
    assert_allclose(recovered, samples, rtol=0.0, atol=8 * n_t * EPS)


@pytest.mark.parametrize(
    ("kind", "n_t", "n_omega"), [("field", 8, 5), ("envelope", 8, 8)]
)
def test_forward_produces_the_layout_of_its_kind(kind, n_t, n_omega):
    samples = np.ones(n_t, dtype=complex if kind == "envelope" else float)
    assert fourier.forward(samples, kind=kind).shape == (n_omega,)
    assert fourier.n_omega_for(n_t, kind=kind) == n_omega


def test_forward_transforms_the_trailing_axis_of_a_stacked_array():
    rng = np.random.default_rng(5)
    stack = rng.standard_normal((2, 3, 16))
    spectrum = fourier.forward(stack, kind="field")
    assert spectrum.shape == (2, 3, 9)
    assert_allclose(spectrum[1, 2], fourier.forward(stack[1, 2], kind="field"))


def test_forward_refuses_a_complex_array_on_a_field_grid():
    with pytest.raises(GridError, match="real electric field"):
        fourier.forward(np.ones(8, dtype=complex), kind="field")


def test_forward_refuses_a_real_array_on_an_envelope_grid():
    with pytest.raises(GridError, match="complex analytic signal"):
        fourier.forward(np.ones(8), kind="envelope")


def test_an_unknown_kind_is_rejected():
    # Checked on the private guard rather than through a public function: GridKind is a
    # Literal, so in the test session beartype rejects the bad value at the public
    # boundary before this check runs. Outside the test session it is the only guard.
    with pytest.raises(GridError, match="kind must be"):
        fourier._check_kind("spectral")


@pytest.mark.parametrize(("kind", "n_omega"), [("field", 2), ("envelope", 7)])
def test_inverse_rejects_a_spectrum_of_the_wrong_length(kind, n_omega):
    with pytest.raises(GridError, match="n_t"):
        fourier.inverse(np.ones(n_omega, dtype=complex), kind=kind)


@pytest.mark.parametrize("n_t", [0, 2, 3])
def test_layout_rejects_a_grid_that_is_too_small_or_odd(n_t):
    with pytest.raises(GridError, match="even integer"):
        fourier.n_omega_for(n_t, kind="field")


# --- the analytic signal --------------------------------------------------------------


def test_analytic_signal_of_a_cosine_is_the_complex_exponential():
    n_t = 32
    t = np.arange(n_t) / n_t
    envelope = fourier.analytic_signal(np.cos(2 * np.pi * 3 * t))
    # Exact up to the round-off of one forward and one inverse transform.
    assert_allclose(envelope, np.exp(2j * np.pi * 3 * t), rtol=0.0, atol=8 * n_t * EPS)


def test_analytic_signal_reproduces_the_field_in_its_real_part():
    rng = np.random.default_rng(7)
    field = rng.standard_normal(64)
    assert_allclose(
        np.real(fourier.analytic_signal(field)), field, rtol=0.0, atol=8 * 64 * EPS
    )


def test_analytic_signal_keeps_the_nyquist_bin():
    # The documented divergence from Luna's plan_hilbert, which zeroes it. A field that
    # is nothing but the Nyquist component must survive; under Luna's convention the
    # analytic signal of this array is identically zero.
    n_t = 8
    field = np.array([1.0, -1.0] * (n_t // 2))
    envelope = fourier.analytic_signal(field)
    assert_allclose(np.real(envelope), field, rtol=0.0, atol=8 * n_t * EPS)
    assert np.abs(envelope).max() > 0.5


def test_analytic_signal_refuses_a_complex_array():
    with pytest.raises(GridError, match="real electric field"):
        fourier.analytic_signal(np.ones(8, dtype=complex))


def test_analytic_signal_matches_scipy_hilbert():
    rng = np.random.default_rng(11)
    field = rng.standard_normal(32)
    # The helper is SciPy's transform, called as it is: the same computation, exactly.
    assert np.array_equal(fourier.analytic_signal(field), hilbert(field))


# --- energy and spectral density ------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n_t", [8, 64, 512])
def test_parseval_holds_between_the_two_energy_functionals(kind, n_t):
    spectrum = make_spectrum(kind, n_t, seed=n_t)
    samples = fourier.inverse(spectrum, kind=kind)
    dt = 1e-15
    energy_time = fourier.energy_t(samples, dt, kind=kind)
    energy_freq = fourier.energy_omega(
        fourier.forward(samples, kind=kind), dt, kind=kind
    )
    # The target is 1e-14 * n_t. Both sides are the same sum of n_t non-negative
    # terms evaluated two ways, so Higham's n*eps bound applies and the realised error
    # is
    # far below that figure; assert the tighter bound so a regression shows up.
    assert_allclose(energy_time, energy_freq, rtol=n_t * EPS, atol=0.0)
    assert energy_time > 0.0


def test_gaussian_pulse_energy_agrees_between_the_domains():
    # A band-limited Gaussian: built in the frequency domain so that the zero-frequency
    # and Nyquist bins are exactly zero, as a physical apodised grid guarantees.
    n_t, dt = 1024, 1e-16
    d_omega = 2 * np.pi / (n_t * dt)
    omega = np.arange(n_t // 2 + 1) * d_omega
    centre = omega[n_t // 8]
    spectrum = np.exp(-(((omega - centre) / (0.15 * centre)) ** 2)).astype(complex)
    spectrum[0] = 0.0
    spectrum[-1] = 0.0
    field = fourier.inverse(spectrum, kind="field")
    energy_time = fourier.energy_t(field, dt, kind="field")
    energy_freq = fourier.energy_omega(spectrum, dt, kind="field")
    # The target is 1e-12; the exact discrete form reaches the summation bound.
    assert_allclose(energy_time, energy_freq, rtol=n_t * EPS, atol=0.0)


@pytest.mark.parametrize("kind", KINDS)
def test_energy_scales_as_the_square_of_the_amplitude(kind):
    spectrum = make_spectrum(kind, 32, seed=2)
    single = fourier.energy_omega(spectrum, 1e-15, kind=kind)
    doubled = fourier.energy_omega(2.0 * spectrum, 1e-15, kind=kind)
    assert_allclose(doubled, 4.0 * single, rtol=8 * EPS, atol=0.0)


@pytest.mark.parametrize("kind", KINDS)
def test_spectral_density_integrates_to_the_energy(kind):
    n_t, dt = 64, 1e-15
    spectrum = make_spectrum(kind, n_t, seed=4)
    d_omega = 2 * np.pi / (n_t * dt)
    density = fourier.spectral_density(spectrum, dt, kind=kind)
    energy = fourier.energy_omega(spectrum, dt, kind=kind)
    # Exact only because make_spectrum zeroes the field grid's end bins: those two are
    # the documented exception, since the analytic signal does not double them.
    assert_allclose(density.sum() * d_omega, energy, rtol=n_t * EPS, atol=0.0)


@pytest.mark.parametrize("kind", KINDS)
def test_one_photon_per_bin_carries_hbar_omega(kind):
    # conventions.md section 4 fixes the shot-noise amplitudes and the spectral density
    # prefactors as a matched pair: one photon in a bin must give S(omega) d_omega =
    # hbar*omega. This ties the whole normalisation chain together in one assertion.
    n_t, dt, bin_index = 64, 1e-16, 7
    d_omega = 2 * np.pi / (n_t * dt)
    omega = bin_index * d_omega
    amplitude = np.sqrt(HBAR * omega / d_omega) * np.sqrt(2 * np.pi)
    amplitude /= 2 * dt if kind == "field" else dt
    spectrum = np.zeros(fourier.n_omega_for(n_t, kind=kind), dtype=complex)
    spectrum[bin_index] = amplitude
    density = fourier.spectral_density(spectrum, dt, kind=kind)
    assert_allclose(density[bin_index] * d_omega, HBAR * omega, rtol=8 * EPS, atol=0.0)


def test_energy_t_refuses_a_mismatched_dtype():
    with pytest.raises(GridError, match="complex analytic signal"):
        fourier.energy_t(np.ones(8), 1.0, kind="envelope")


# --- oversampling ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("n_t", "n_to"), [(8, 8), (8, 16), (16, 48)])
def test_oversample_then_crop_is_exact(kind, n_t, n_to):
    spectrum = make_spectrum(kind, n_t, seed=n_t + n_to)
    fine = fourier.oversample(spectrum, n_to, kind=kind)
    assert fine.shape == (fourier.n_omega_for(n_to, kind=kind),)
    recovered = fourier.crop(fine, n_t, kind=kind)
    # Pure copies and a multiply by n_to/n_t followed by its exact reciprocal.
    assert_allclose(recovered, spectrum, rtol=8 * EPS, atol=0.0)


@pytest.mark.parametrize("kind", KINDS)
def test_oversample_returns_a_new_array(kind):
    spectrum = make_spectrum(kind, 8, seed=1)
    fine = fourier.oversample(spectrum, 16, kind=kind)
    assert fine.base is None
    fine[...] = 0.0
    assert np.abs(spectrum).max() > 0.0


def test_pad_and_crop_scales_are_reciprocal():
    assert fourier.pad_scale(4096, 8192) == 2.0
    assert fourier.crop_scale(4096, 8192) == 0.5
    assert fourier.pad_scale(64, 64) == fourier.crop_scale(64, 64) == 1.0


@pytest.mark.parametrize("kind", KINDS)
def test_oversampling_preserves_the_signal_at_coincident_samples(kind):
    # This is what the scale factor is for: the fine time samples must reproduce the
    # coarse ones wherever the two grids share a sample.
    n_t, n_to = 16, 64
    spectrum = make_spectrum(kind, n_t, seed=9)
    coarse = fourier.inverse(spectrum, kind=kind)
    fine = fourier.inverse(fourier.oversample(spectrum, n_to, kind=kind), kind=kind)
    assert_allclose(fine[:: n_to // n_t], coarse, rtol=0.0, atol=8 * n_to * EPS)


@pytest.mark.parametrize("kind", KINDS)
def test_oversampling_preserves_the_energy(kind):
    n_t, n_to, dt = 16, 64, 1e-15
    spectrum = make_spectrum(kind, n_t, seed=13)
    fine = fourier.oversample(spectrum, n_to, kind=kind)
    assert_allclose(
        fourier.energy_omega(fine, dt * n_t / n_to, kind=kind),
        fourier.energy_omega(spectrum, dt, kind=kind),
        rtol=n_to * EPS,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("n_t", "n_to", "message"),
    [
        (8, 4, "at least n_t"),
        (8, 12, "integer multiple"),
        (3, 12, "even integer"),
        (8, 15, "even integer"),
    ],
)
def test_size_pairs_are_validated(n_t, n_to, message):
    with pytest.raises(GridError, match=message):
        fourier.crop_index(n_t, n_to, kind="field")


def test_crop_index_places_the_envelope_halves_at_both_ends():
    index = fourier.crop_index(4, 16, kind="envelope")
    assert index.tolist() == [0, 1, 14, 15]
    assert fourier.crop_index(4, 16, kind="field").tolist() == [0, 1, 2]


# --- properties -----------------------------------------------------------------------


@given(
    n_half=st.integers(min_value=2, max_value=64),
    kind=st.sampled_from(KINDS),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_parseval_holds_for_any_size(n_half, kind, seed):
    n_t = 2 * n_half
    spectrum = make_spectrum(kind, n_t, seed=seed)
    samples = fourier.inverse(spectrum, kind=kind)
    energy_time = fourier.energy_t(samples, 1.0, kind=kind)
    energy_freq = fourier.energy_omega(
        fourier.forward(samples, kind=kind), 1.0, kind=kind
    )
    assert_allclose(energy_time, energy_freq, rtol=n_t * EPS, atol=0.0)
