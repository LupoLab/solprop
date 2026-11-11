# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the monitor ratio bound with independent transform perturbations."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from backends.julia.noise_bounds import temporal_noise_floor_bound


def _fixture():
    noise = np.array([1 + 2j, -0.5j, 0.25 - 1j, -0.125 + 0.5j])
    launch = np.array([4 + 1j, -2 + 0.5j, 1 - 0.25j, 0.5 + 0.125j])
    return noise[None, None], launch[None, None]


def _perturbed_energy(spectrum, direction):
    # Four-point inverse transforms of these small dyadic inputs are exact:
    # roots are 1,-1,+i,-i, and normalization is division by four. Independently
    # perturb both sample components up to the declared componentwise envelope.
    samples = np.fft.ifft(spectrum[0, 0])
    norm = sum(
        (
            abs(Fraction(float(value.real))) + abs(Fraction(float(value.imag)))
            for value in spectrum[0, 0]
        ),
        start=Fraction(),
    )
    delta = Fraction(32 * samples.size, 2**53 - 32 * samples.size) * norm / samples.size
    total = Fraction()
    for value in samples:
        real, imag = Fraction(float(value.real)), Fraction(float(value.imag))
        real += direction * delta * (1 if real >= 0 else -1)
        imag += direction * delta * (1 if imag >= 0 else -1)
        total += real**2 + imag**2
    # The actual scalar route may incur roundings after the transform. Exercise
    # both signs of a legal relative perturbation and the final rounded division.
    return total * (1 + direction * Fraction(samples.size + 8, 2**53))


def test_ratio_bound_encloses_independent_fft_and_scalar_extremes():
    noise, launch = _fixture()
    bound = temporal_noise_floor_bound(noise, launch)
    assert math.isfinite(bound) and bound > 0
    candidates = [
        float(_perturbed_energy(noise, first) / _perturbed_energy(launch, second))
        for first, second in itertools.product((-1, 1), repeat=2)
    ]
    spread = max(candidates) - min(candidates)
    assert 0 < spread <= bound


def test_ratio_bound_allows_temporal_cancellation_without_a_relative_fft_assumption():
    noise = np.array([1, -1, 1, -1], dtype=np.complex128)[None, None]
    launch = np.array([1, 1, 1, 1], dtype=np.complex128)[None, None]
    assert np.count_nonzero(np.fft.ifft(noise[0, 0])) == 1
    bound = temporal_noise_floor_bound(noise, launch)
    ratios = [
        float(_perturbed_energy(noise, sign) / _perturbed_energy(launch, -sign))
        for sign in (-1, 1)
    ]
    assert 0 < max(ratios) - min(ratios) <= bound


@pytest.mark.parametrize("factor", [0.125, 8.0])
def test_common_exact_rescaling_preserves_the_ratio_budget(factor):
    noise, launch = _fixture()
    original = temporal_noise_floor_bound(noise, launch)
    scaled = temporal_noise_floor_bound(factor * noise, factor * launch)
    # All relative terms are scale invariant. The bounded subnormal allowance is
    # far below one ulp for this normal-scale fixture; permit final outward rounding.
    assert abs(original - scaled) <= math.ulp(original)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("value", [0.0, 1e-160, 1e155, math.inf, math.nan])
def test_ratio_bound_refuses_uncertified_ranges(which, value):
    spectra = list(_fixture())
    spectra[which] = np.full_like(spectra[which], value)
    with pytest.raises(AssertionError):
        temporal_noise_floor_bound(*spectra)


@pytest.mark.parametrize("shape", [(1, 1, 0), (2, 1, 4), (1, 2, 4)])
def test_ratio_bound_refuses_empty_or_nonscalar_fields(shape):
    spectrum = np.ones(shape, dtype=np.complex128)
    with pytest.raises(AssertionError):
        temporal_noise_floor_bound(spectrum, spectrum)


def test_ratio_bound_refuses_mismatched_shapes_and_binary32():
    noise, launch = _fixture()
    with pytest.raises(AssertionError):
        temporal_noise_floor_bound(noise, launch[..., :2])
    with pytest.raises(AssertionError):
        temporal_noise_floor_bound(noise.astype(np.complex64), launch)


def test_ratio_bound_refuses_overflow_even_when_both_energy_inputs_are_safe():
    noise = np.full((1, 1, 4), 1e150, dtype=np.complex128)
    launch = np.full((1, 1, 4), 1e-5, dtype=np.complex128)
    with pytest.raises(AssertionError, match="ratio may overflow"):
        temporal_noise_floor_bound(noise, launch)
