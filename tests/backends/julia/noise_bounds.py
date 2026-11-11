# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bound a scalar envelope monitor's noise-floor ratio on identical input fields.

The monitor forms unweighted temporal sums of squared amplitudes, then divides
the noise energy by the original launch energy. Both implementations start from
the same binary64 spectra. This helper does not bound propagated-field error.

Each inverse transform has the qualification hypothesis already stated in
``observation_bounds``: each real/imaginary component differs from the exact DFT
by at most ``gamma(32*n)*sum(abs(spectrum))/n``. This is an explicit CPU FFT
qualification hypothesis, not a universal FFT-library theorem. The factor two
below converts that envelope conservatively to complex modulus. No measured
backend result enters the bound.

For exact temporal energy E, spectral component L1 bound L, and complex sample
error d, Parseval gives E exactly from the supplied spectrum and the triangle
inequality gives ``sum(abs(samples)) <= L``. Thus the transform perturbs the
energy by at most ``2*L*d + n*d**2``. Eight scalar roundings per density and n
nonnegative reduction roundings give the subsequent gamma bound. An absolute
allowance covers gradual underflow in those scalar operations; the transform
itself remains subject to the stated qualification hypothesis. We refuse
nonfinite, overflow-prone or subnormal-energy fixtures.

For noise/launch energies N,D with errors eN,eD and D>eD, each exact quotient
deviates from N/D by at most ``(eN*D + N*eD)/(D*(D-eD))``. Add the final division
rounding, then double for two independent computations. Exact rationals keep all
intermediate bounds outward; the final result rounds upward.

The reduction model is Higham's ``gamma_k = k*u/(1-k*u)``, with binary64
``u=2**-53`` (Accuracy and Stability of Numerical Algorithms, 2nd ed., sections
3.1 and 24.1, https://doi.org/10.1137/1.9780898718027).
"""

from fractions import Fraction

import numpy as np
from jaxtyping import Complex

from backends.julia.observation_bounds import _gamma, _upper

__all__ = ["temporal_noise_floor_bound"]

_HALF_SUBNORMAL = Fraction(1, 2**1075)


def _energy_envelope(spectrum: Complex[np.ndarray, " n_omega"]):
    """Return exact temporal energy and one implementation's absolute error bound.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Finite nonempty complex128 scalar-envelope spectrum, shape ``(n_omega,)``.

    Returns
    -------
    tuple of fractions.Fraction
        Exact Parseval energy and its outward transform/power/reduction allowance.
    """
    values = [
        (Fraction(float(value.real)), Fraction(float(value.imag))) for value in spectrum
    ]
    count = len(values)
    total = sum((real**2 + imag**2 for real, imag in values), start=Fraction())
    norm = sum((abs(real) + abs(imag) for real, imag in values), start=Fraction())
    energy = total / count
    largest = Fraction(np.finfo(float).max)
    assert energy > Fraction(np.finfo(float).tiny), "subnormal or zero input energy"
    assert norm**2 < largest / 16, "the FFT/power fixture may overflow"
    error = 2 * _gamma(32 * count) * norm / count
    transformed = 2 * norm * error + count * error**2
    scalar = _gamma(count + 8)
    absolute = (8 * count + 2) * _HALF_SUBNORMAL * (1 + scalar)
    bound = transformed + scalar * (energy + transformed) + absolute
    assert energy + bound < largest
    return energy, bound


def temporal_noise_floor_bound(
    noise_spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"],
    launch_spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> float:
    """Bound two scalar-envelope noise-floor ratios from the same inputs.

    Parameters
    ----------
    noise_spectrum : numpy.ndarray
        Full supplied complex128 noise draw, shape ``(1,1,n_omega)``, in internal
        FFT units. It is the monitor's draw, before any launch masking.
    launch_spectrum : numpy.ndarray
        Actual masked complex128 launch field, the same shape and units. Include
        the draw if this fresh launch added it. Monitor columns have unit weight.

    Returns
    -------
    float
        Positive absolute bound for the difference of two dimensionless floor
        ratios. All inputs and the qualification hypothesis precede comparison.

    Raises
    ------
    AssertionError
        If the fixture lacks the stated shape, binary64 dtype, finite normal
        energy, safe range or a certified positive denominator.
    """
    assert noise_spectrum.shape == launch_spectrum.shape
    assert noise_spectrum.ndim == 3 and noise_spectrum.shape[:2] == (1, 1)
    assert noise_spectrum.shape[-1] > 0
    assert noise_spectrum.dtype == launch_spectrum.dtype == np.dtype(np.complex128)
    assert np.all(np.isfinite(noise_spectrum)) and np.all(np.isfinite(launch_spectrum))
    numerator, numerator_error = _energy_envelope(noise_spectrum[0, 0])
    denominator, denominator_error = _energy_envelope(launch_spectrum[0, 0])
    assert denominator > denominator_error, "uncertified launch-energy denominator"
    lower = denominator - denominator_error
    variation = (numerator_error * denominator + numerator * denominator_error) / (
        denominator * lower
    )
    upper_ratio = (numerator + numerator_error) / lower
    assert upper_ratio < Fraction(np.finfo(float).max), "the floor ratio may overflow"
    return _upper(2 * (variation + _gamma(1) * upper_ratio + _HALF_SUBNORMAL))
