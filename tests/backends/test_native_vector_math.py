# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native backend's host vector maths against a 50-digit reference (ADR-059).

The CPU kernels form their transcendentals in passes over blocks through one layer
(``native/include/solprop_native/vmath.hpp``): SLEEF's u10 functions on x86-64 builds,
the system libm on Arm. Either way each of ``exp``, ``log``, ``sin`` and ``cos`` is
within 1.0 ulp of the correctly rounded value (SLEEF documents 1.0 ulp over each
function's whole domain; glibc's and Apple's are within it), measured here against
mpmath at 50 significant digits over the argument ranges the kernels form:

- the propagators' and phi-functions' exponents and the magnitudes ``e^{+-a}`` of the
  stage entry, and the ionisation rate's ``exp`` of its log-rate cubic (from about
  ``-745``, where ``exp`` leaves the subnormals, to the largest finite results);
- the rate's ``log`` of a field magnitude, at field strengths and over the whole
  positive range, subnormals included;
- the stage, step and noise phases up to ``1e4`` rad, beyond to ``1e7``, at the
  multiples of ``pi / 2`` where a reduction loses the most, and at ``1e15``.

The complex exponential ``e^a (cos b, sin b)`` is the exponential and the cosine
(sine) each within one ulp and their product rounded once: each component is within
``2 u + 2 u + u = 5 u`` of ``|e^a|`` (``u = eps / 2``), the bound below.
"""

import mpmath as mp
import numpy as np
import pytest

native = pytest.importorskip("solprop_native")
core = native._core  # pyright: ignore[reportPrivateUsage] - the layer's test bindings

EPS = np.finfo(np.float64).eps
TINY = np.finfo(np.float64).smallest_subnormal
DIGITS = 50
# One ulp of the result: SLEEF's u10 bound, and the system libm's.
ULPS = 1.0


def _ulps(computed, exact):
    """Each element's error in ulps of its correctly rounded value, from mpf values."""
    error = np.empty(len(exact))
    for i, (value, reference) in enumerate(zip(computed, exact, strict=True)):
        rounded = float(reference)
        spacing = np.spacing(np.abs(rounded)) if rounded != 0.0 else TINY
        error[i] = float(abs(mp.mpf(float(value)) - reference)) / max(spacing, TINY)
    return error


def _exponents():
    rng = np.random.default_rng(58)
    return np.concatenate(
        [
            rng.uniform(-60.0, 60.0, 3000),
            rng.uniform(-745.0, 709.7, 3000),
            rng.uniform(-1e-6, 1e-6, 500),
            [-745.1, -740.0, -708.4, 0.0, -0.0, 709.78],
        ]
    )


def _magnitudes():
    rng = np.random.default_rng(59)
    return np.concatenate(
        [
            10.0 ** rng.uniform(3.0, 14.0, 3000),
            10.0 ** rng.uniform(-307.0, 308.0, 2000),
            rng.uniform(0.5, 2.0, 1000),
            [TINY, 1e-310, 1.0, np.nextafter(1.0, 2.0), np.finfo(np.float64).max],
        ]
    )


def _phases():
    rng = np.random.default_rng(60)
    quarter = np.arange(-4000, 4001, 37) * (np.pi / 2.0)
    return np.concatenate(
        [
            rng.uniform(-1e4, 1e4, 3000),
            rng.uniform(-10.0, 10.0, 1000),
            rng.uniform(-1e7, 1e7, 1000),
            rng.uniform(-1e15, 1e15, 300),
            quarter,
            np.nextafter(quarter, np.inf),
            [0.0, -0.0, 1e-300, -1e-300],
        ]
    )


def test_exp_is_within_one_ulp():
    x = _exponents()
    out = np.empty_like(x)
    core.vector_exp(x, out)
    with mp.workdps(DIGITS):
        exact = [mp.exp(mp.mpf(float(value))) for value in x]
        assert _ulps(out, exact).max() <= ULPS


def test_log_is_within_one_ulp():
    x = _magnitudes()
    out = np.empty_like(x)
    core.vector_log(x, out)
    with mp.workdps(DIGITS):
        exact = [mp.log(mp.mpf(float(value))) for value in x]
        assert _ulps(out, exact).max() <= ULPS


def test_sincos_is_within_one_ulp():
    x = _phases()
    sine = np.empty_like(x)
    cosine = np.empty_like(x)
    core.vector_sincos(x, sine, cosine)
    with mp.workdps(DIGITS + 20):
        # mpmath reduces exactly at this precision: 1e15 needs about 16 more digits.
        exact_sine = [mp.sin(mp.mpf(float(value))) for value in x]
        exact_cosine = [mp.cos(mp.mpf(float(value))) for value in x]
        assert _ulps(sine, exact_sine).max() <= ULPS
        assert _ulps(cosine, exact_cosine).max() <= ULPS


def test_complex_exponential_is_within_its_product_bound():
    rng = np.random.default_rng(61)
    a = np.concatenate([rng.uniform(-60.0, 5.0, 3000), [0.0, -0.0]])
    b = np.concatenate([rng.uniform(-1e4, 1e4, 3000), [0.5, -0.5]])
    real = np.empty_like(a)
    imag = np.empty_like(a)
    core.vector_exp_complex(a, b, real, imag)
    with mp.workdps(DIGITS):
        for i in range(len(a)):
            magnitude = mp.exp(mp.mpf(float(a[i])))
            angle = mp.mpf(float(b[i]))
            bound = 2.5 * EPS * float(magnitude)
            assert (
                float(abs(mp.mpf(float(real[i])) - magnitude * mp.cos(angle))) <= bound
            )
            assert (
                float(abs(mp.mpf(float(imag[i])) - magnitude * mp.sin(angle))) <= bound
            )


def test_lengths_must_match():
    x = np.zeros(4)
    with pytest.raises(ValueError, match="vector_exp: the arrays have lengths 4 and 3"):
        core.vector_exp(x, np.zeros(3))
    with pytest.raises(TypeError):
        core.vector_exp(x.astype(np.float32), np.zeros(4))
