# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The precision record, its admission and the range helpers of SEP 33."""

import math

import numpy as np
import pytest
from jaxtyping import TypeCheckError
from scipy import integrate

from bounds import UNIT_ROUNDOFF_32, higham_gamma, prefix_bound
from solprop.exceptions import TranslationError
from solprop.translate.precision import (
    DOUBLE,
    SINGLE,
    SINGLE_RTOL_FLOOR,
    Precision,
    admit_precision,
    block_exponent,
    combine,
    launch_peak,
    narrow,
    pairwise_sum,
    precision_named,
    prefix_sum,
    scaled_square_sum,
    state_scale,
    study_admission,
)

TINY32 = float(np.finfo(np.float32).tiny)


def test_the_public_values_name_equal_fields_and_a_mixed_record_is_refused():
    assert precision_named("double") is DOUBLE
    assert precision_named("single") is SINGLE
    assert SINGLE.complex_dtype == np.complex64 and DOUBLE.real_dtype == np.float64
    with pytest.raises(TranslationError, match="mixed precision"):
        Precision("single", "double")
    # The annotation refuses the value first wherever annotations are checked.
    with pytest.raises((TranslationError, TypeCheckError)):
        precision_named("half")  # pyright: ignore[reportArgumentType]


def test_admission_refuses_by_name_and_the_study_reaches_adams():
    admit = {"backend": "numpy_ref", "adams": False, "rtol": 1e-6, "schedule": False}
    admit_precision(SINGLE, **admit)
    admit_precision(DOUBLE, **(admit | {"backend": "native", "rtol": 1e-12}))
    with pytest.raises(TranslationError, match=r"numba_cpu.*binary64 only"):
        admit_precision(SINGLE, **(admit | {"backend": "numba_cpu"}))
    with pytest.raises(TranslationError, match="floor 1e-06"):
        admit_precision(SINGLE, **(admit | {"rtol": 1e-7}))
    # A replayed schedule reads no tolerance.
    admit_precision(SINGLE, **(admit | {"rtol": 1e-12, "schedule": True}))
    with pytest.raises(TranslationError, match="Adams integrator"):
        admit_precision(SINGLE, **(admit | {"adams": True}))
    with study_admission():
        admit_precision(SINGLE, **(admit | {"adams": True}))
    assert SINGLE_RTOL_FLOOR == 1e-6


def test_a_molecular_kernel_keeps_every_bin_under_its_block_exponent():
    """The review's counterexample: a spectrum near 5e-50 is all zero in binary32."""
    rng = np.random.default_rng(33)
    kernel = 5.2e-50 * (rng.random(64) + 1j * rng.random(64))
    assert not np.any(kernel.astype(np.complex64))
    narrowed = narrow(kernel)
    assert narrowed.values.dtype == np.complex64
    assert 1.0 <= float(np.max(np.abs(narrowed.values))) < 2.0
    # One rounding per entry: the widened values are the kernel to a unit roundoff.
    relative = np.abs(narrowed.widened() - kernel) / np.abs(kernel)
    assert float(np.max(relative)) <= 2 * UNIT_ROUNDOFF_32
    assert narrow(np.zeros(3)).exponent == 0


def test_block_exponents_and_the_state_scale_are_exact_powers_of_two():
    for value in (1.0, 1.5, 2.0, 3.0e4, 7.3e-12):
        exponent = block_exponent(value)
        assert 1.0 <= value * 2.0**-exponent < 2.0
        assert state_scale(value) == 2.0**exponent
    assert state_scale(0.0) == 1.0
    with pytest.raises(TranslationError, match="finite"):
        block_exponent(np.array([1.0, np.inf]))


def test_the_launch_peak_is_the_largest_temporal_magnitude():
    spectrum = np.zeros((1, 1, 1, 8), dtype=complex)
    spectrum[..., 0] = 8.0
    assert launch_peak(spectrum, "envelope") == pytest.approx(1.0)
    assert launch_peak(np.zeros((1, 1, 1, 8), dtype=complex), "envelope") == 0.0


def test_a_combination_never_leaves_binary64_and_checks_binary32():
    density, eps0 = 2.5e25, 8.8541878128e-12
    coefficient = combine(density, eps0, exponent=-164)
    assert math.isclose(
        coefficient.value, density * eps0 * 2.0**-164, rel_tol=4 * 2.0**-53
    )
    # The exponents add as integers: no intermediate 2**-1100 is formed.
    tiny = combine(1e-300, 1e-300, exponent=1000)
    assert math.isclose(tiny.value, (1e-300 * 2.0**500) ** 2, rel_tol=1e-15)
    assert combine(0.0, 1.0).value == 0.0
    with pytest.raises(TranslationError, match="normal range"):
        combine(1e-30, 1e-30).binary32()
    assert combine(3.0, 0.5).binary32() == 1.5


def test_the_pairwise_order_is_within_its_tree_bound():
    rng = np.random.default_rng(1)
    values = rng.standard_normal(10_000).astype(np.float32)
    exact = math.fsum(values.astype(np.float64))
    depth = math.ceil(math.log2(values.size))
    bound = higham_gamma(depth, UNIT_ROUNDOFF_32) * float(np.sum(np.abs(values)))
    assert abs(float(pairwise_sum(values)) - exact) <= bound
    assert pairwise_sum(np.zeros(0, np.float32)).shape == ()


def test_the_prefix_order_holds_its_depth_where_a_sequential_sum_would_not():
    """Every prefix of a doubling scan is a sum of depth at most ceil(log2 n)."""
    rng = np.random.default_rng(2)
    increments = rng.random(1 << 14).astype(np.float32)
    exact = np.cumsum(increments.astype(np.float64))
    depth = math.ceil(math.log2(increments.size))
    bound = prefix_bound(increments.astype(np.float64), depth, u=UNIT_ROUNDOFF_32)
    error = np.abs(prefix_sum(increments).astype(np.float64) - exact)
    assert np.all(error <= bound)
    assert prefix_sum(increments).dtype == np.float32
    # The trapezoid's increments through the same scan reproduce SciPy's integral.
    values = rng.random(4096)
    trapezoid = np.concatenate(
        ([0.0], prefix_sum(0.5 * (values[:-1] + values[1:]) * 1e-15))
    )
    reference = integrate.cumulative_trapezoid(values, dx=1e-15, initial=0)
    assert np.max(np.abs(trapezoid - reference)) < 1e-13 * reference[-1]


def test_a_sum_of_squares_that_overflows_binary32_is_scaled_first():
    """The Cartesian case of SEP 33 section 4.3: squares near 1e37, a sum near 1e44."""
    values = np.full(4096, 4.0e18 + 3.0e18j, dtype=np.complex64)
    with np.errstate(over="ignore"):
        naive = np.sum(np.abs(values) ** 2)
    assert not np.isfinite(naive)
    total = scaled_square_sum(values)
    assert math.isclose(total, 4096 * 25e36, rel_tol=16 * UNIT_ROUNDOFF_32)
    weights = np.full(4096, 0.5, dtype=np.float32)
    assert math.isclose(
        scaled_square_sum(values, weights), 0.5 * total, rel_tol=4 * 2.0**-24
    )
    subnormal = np.full(8, 1e-40, dtype=np.float32)
    assert scaled_square_sum(subnormal) > 0.0
    assert TINY32 > 1e-40
