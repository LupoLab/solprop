# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The comparison helpers every rounding-bound and accuracy test is written with."""

import math

import numpy as np
import pytest

from bounds import (
    EPS,
    UNIT_ROUNDOFF,
    AccuracyEnvelope,
    assert_within,
    assert_work_within,
    higham_gamma,
    prefix_bound,
    propagation_growth,
    stencil_bound,
)


def test_unit_roundoff_is_half_of_numpys_epsilon():
    """``u`` and ``eps`` are not interchangeable: ``eps = 2 u``."""
    assert UNIT_ROUNDOFF == EPS / 2


def test_higham_gamma_and_its_range():
    """``gamma_k = k u / (1 - k u)``, refused where ``k u >= 1``."""
    assert higham_gamma(4) == pytest.approx(4 * UNIT_ROUNDOFF, rel=1e-15)
    with pytest.raises(ValueError, match="beyond"):
        higham_gamma(2**53)


def test_a_reordered_sum_lies_inside_its_prefix_bound():
    """Two summation orders of the same signed increments differ within the bound."""
    rng = np.random.default_rng(25)
    increments = rng.standard_normal(4096) * np.exp(rng.uniform(-20, 20, 4096))
    serial = np.cumsum(increments)
    pairwise = np.array([math.fsum(increments[: j + 1]) for j in range(4096)])
    bound = prefix_bound(increments, depth=4095)
    assert np.all(np.abs(serial - pairwise) <= bound)


def test_a_stencil_bound_covers_a_reassociated_stencil():
    """The paired and the left-to-right evaluations of a symmetric stencil agree."""
    rng = np.random.default_rng(7)
    taps = np.array([0.5, 0.3, -0.07, 0.01])
    x = rng.standard_normal(256)
    padded = np.pad(x, 3)
    paired = taps[0] * x + sum(
        taps[k] * (padded[3 - k : 3 - k + 256] + padded[3 + k : 3 + k + 256])
        for k in range(1, 4)
    )
    straight = sum(taps[abs(k)] * padded[3 + k : 3 + k + 256] for k in range(-3, 4))
    assert np.all(np.abs(paired - straight) <= 2 * stencil_bound(taps, x))


def test_assert_within_accepts_rounding_and_catches_a_dropped_term():
    """A perturbation the size of a dropped term fails; rounding passes."""
    rng = np.random.default_rng(3)
    expected = rng.standard_normal(1000)
    margin = assert_within(expected * (1 + EPS), expected, 16 * EPS)
    assert 0.0 < margin < 1.0
    with pytest.raises(AssertionError, match="exceeds"):
        assert_within(expected + 1e-6 * expected[::-1], expected, 16 * EPS)


def test_assert_within_refuses_trivial_and_non_finite_references():
    """A zero reference makes a relative test vacuous; NaN is never within a bound."""
    with pytest.raises(AssertionError, match="identically zero"):
        assert_within(np.zeros(3), np.zeros(3), 1e-12)
    with pytest.raises(AssertionError, match="non-finite"):
        assert_within(np.array([np.nan]), np.array([1.0]), 1e-12)
    with pytest.raises(AssertionError, match="shapes"):
        assert_within(np.ones(2), np.ones(3), 1e-12)


def test_elementwise_bounds_and_absolute_scales():
    """Per-element absolute bounds, and an absolute normwise bound with ``scale=1``."""
    expected = np.array([1.0, -2.0, 3.0])
    assert_within(expected + 1e-17, expected, np.full(3, 1e-16), elementwise=True)
    with pytest.raises(AssertionError, match="outside"):
        assert_within(expected + 1e-15, expected, np.full(3, 1e-16), elementwise=True)
    assert_within(np.zeros(3) + 1e-20, np.zeros(3), 1e-18, scale=1.0)


def test_accuracy_envelopes_and_work_bands():
    """An envelope scales the reference error; a work band brackets a baseline."""
    envelope = AccuracyEnvelope("rtol 1e-9 run", error=1e-7, factor=4.0)
    assert envelope.bound == pytest.approx(4e-7)
    assert_work_within(110, 100, factor=1.25)
    with pytest.raises(AssertionError, match="outside"):
        assert_work_within(200, 100, factor=1.25)
    assert propagation_growth(2.0, 0.5) == pytest.approx(math.e)
