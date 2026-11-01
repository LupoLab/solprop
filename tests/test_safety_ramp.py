# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The k-perp safety ramp, the attenuation search and the physical evanescent rate.

SEP 22 sections 4.3 to 4.5. The profile is checked point by point (exact zeros, its
strength at the grid's edge, its slope and its column order); the sorted search and
the rate bounds against a brute-force maximum over every column.
"""

import math

import numpy as np
import pytest

from bounds import gamma
from solprop.boundaries import TransverseCollar, transverse_collar_rate
from solprop.exceptions import GridError, TranslationError
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.spatial import (
    EVANESCENT_RATE,
    SpatialMedium,
    attenuation_extremes,
    physical_evanescent_rate,
)
from solprop.translate.spatial import SeparableOperator, rate_bounds, separable_values


def ramp(rho: np.ndarray, fraction: float) -> np.ndarray:
    """Section 4.3's P(rho), written out independently of the module."""
    out = np.zeros_like(rho)
    inside = rho > 1 - fraction
    out[inside] = 0.5 * (1 - np.cos(np.pi * (rho[inside] - 1 + fraction) / fraction))
    return out


def test_the_ramp_is_zero_inside_and_its_strength_at_the_edge():
    space = FreeGrid(1e-4, 40, 2e-4, 24)
    collar = TransverseCollar(strength=3e4, fraction=0.25)
    rate = transverse_collar_rate(space, collar)
    rho_x = np.abs(space.k_x) / np.max(np.abs(space.k_x))
    rho_y = np.abs(space.k_y) / np.max(np.abs(space.k_y))
    # Column ix + n_x*iy, the max-norm of the two axes.
    rho = np.maximum(rho_x[None, :], rho_y[:, None]).reshape(-1)
    # One cosine and a few products: a handful of roundings of the strength.
    np.testing.assert_allclose(
        rate, collar.strength * ramp(rho, 0.25), rtol=0, atol=gamma(8) * 3e4
    )
    assert np.all(rate[rho <= 0.75] == 0.0)  # exactly zero inside: a structural zero
    edge = np.flatnonzero(rho == 1.0)
    assert edge.size and np.all(rate[edge] == collar.strength)
    assert rate.min() >= 0.0 and rate.max() == collar.strength


def test_the_ramp_rises_with_a_continuous_slope_through_its_midpoint():
    fraction = 0.1
    rho = np.linspace(0.85, 1.0, 3001)
    profile = ramp(rho, fraction)
    # The value one half at the ramp's centre, and a zero slope at both of its ends.
    centre = int(np.argmin(np.abs(rho - (1 - fraction / 2))))
    assert profile[centre] == pytest.approx(0.5, abs=1e-3)
    slope = np.gradient(profile, rho)
    start = int(np.argmin(np.abs(rho - (1 - fraction))))
    assert abs(slope[start]) < 1e-2 and abs(slope[-1]) < 1e-1
    assert np.max(slope) == pytest.approx(np.pi / (2 * fraction), rel=1e-3)
    # The module's profile is the same function.
    space = LineGrid(1e-4, 400)
    rate = transverse_collar_rate(space, TransverseCollar(1.0, fraction))
    expected = ramp(np.abs(space.k_x) / np.max(np.abs(space.k_x)), fraction)
    np.testing.assert_allclose(rate, expected, rtol=0, atol=gamma(8))


def test_a_line_takes_its_collar_from_k_x_alone():
    space = LineGrid(1e-4, 40)
    rate = transverse_collar_rate(space, TransverseCollar(2.0))
    assert rate.shape == (40,)
    # m = 19 of 20 is rho = 0.95, the ramp's midpoint for the default fraction.
    index = int(np.argmin(np.abs(space.k_x - 19 * space.k_x[1])))
    assert rate[index] == pytest.approx(1.0, rel=gamma(8))


@pytest.mark.parametrize(
    ("strength", "fraction"),
    [(-1.0, 0.1), (math.inf, 0.1), (1.0, 0.0), (1.0, 1.0), (1.0, math.nan)],
)
def test_an_invalid_collar_is_refused(strength, fraction):
    with pytest.raises(GridError, match="TransverseCollar"):
        TransverseCollar(strength, fraction)


def test_a_collar_is_refused_off_the_cartesian_grids():
    with pytest.raises(GridError, match="radial kind is not periodic"):
        transverse_collar_rate(RadialGrid(1e-4, 16), TransverseCollar(1.0))


def brute_extremes(square, transverse):
    """Every column's Im k_z, propagating and evanescent, by brute force."""
    q = square[:, None] - transverse[None, :]
    imag = np.imag(np.sqrt(q))
    propagating = np.real(q) > 0
    largest = np.max(np.where(propagating, imag, -np.inf), axis=1)
    smallest = np.min(np.where(propagating, imag, np.inf), axis=1)
    evanescent = np.max(np.where(propagating, 0.0, np.abs(imag)), axis=1)
    return largest, smallest, evanescent


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_sorted_search_is_the_brute_force_maximum(sign):
    rng = np.random.default_rng(31)
    square = rng.uniform(0.5, 4.0, 64) * (1 + sign * 1e-3j)
    transverse = rng.permutation(np.concatenate((rng.uniform(0, 6, 200), [0.0])))
    expected = brute_extremes(square, transverse)
    actual = attenuation_extremes(square, transverse)
    # The search evaluates the same columns' roots: equal to the last bit.
    for one, other in zip(actual, expected, strict=True):
        assert np.array_equal(one, other)


@pytest.mark.parametrize("index", [1.45 + 1e-4j, 1.45 - 1e-4j])
@pytest.mark.parametrize(
    "space", [FreeGrid(1.5e-6, 16, 1.2e-6, 12), RadialGrid(1.5e-6, 24)]
)
def test_the_physical_evanescent_rate_is_the_largest_over_every_pair(index, space):
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (700e-9, 950e-9), 200e-15)
    medium = SpatialMedium(index)
    active = grid.active_mask
    square = (
        grid.omega[active] * medium.index(grid.omega[active], 0.0) / 299792458.0
    ) ** 2
    transverse = space.kperp_sq if isinstance(space, FreeGrid) else space.k_r**2
    _, _, evanescent = brute_extremes(square, transverse)
    assert np.max(evanescent) > 0
    assert physical_evanescent_rate(space, medium, grid) == float(np.max(evanescent))


def test_a_grid_without_evanescent_pairs_keeps_the_default_rate():
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (700e-9, 950e-9), 200e-15)
    space = FreeGrid(1e-3, 8, 1e-3, 8)
    assert physical_evanescent_rate(space, SpatialMedium(), grid) == EVANESCENT_RATE
    planar = PlanarGrid([SlabMode(50e-6, "Ar", 1.0)], 1e-4, 8)
    with pytest.raises(GridError, match="FreeGrid, LineGrid or RadialGrid"):
        physical_evanescent_rate(planar, SpatialMedium(), grid)


def separable(**overrides) -> SeparableOperator:
    """A small operator with loss, gain, evanescence and an inactive bin."""
    values = {
        "k_squared": np.array([4 + 0.2j, 9 - 0.1j, 16 + 0j, 0j]),
        "kperp_sq": np.array([0.0, 3.0, 8.0, 12.0, 20.0]),
        "k_squared_low": np.zeros(4, complex),
        "kperp_sq_low": np.zeros(5),
        "frame": np.zeros(4),
        "spectral_rate": np.array([0.5, 0.0, 1.0, 40.0]),
        "prefactor_numerator": np.zeros(4, complex),
        "prefactor_clamped": np.zeros(4, complex),
        "active": np.array([True, True, True, False]),
        "evanescent_rate": 3.0,
        "n_pol": 1,
        "z_span": (0.0, 1.0),
        "kperp_rate": np.array([0.0, 0.0, 1.0, 2.0, 6.0]),
    }
    values.update(overrides)
    return SeparableOperator(**values)


@pytest.mark.parametrize("evanescent_rate", [3.0, 1e6])
def test_the_rate_bounds_bound_the_operator_it_forms(evanescent_rate):
    operator = separable(evanescent_rate=evanescent_rate)
    rate, _ = separable_values(operator)
    bounds = rate_bounds(operator)
    real = np.real(rate[0, :, 0, :])
    assert bounds.loss >= float(np.max(-real))
    assert bounds.gain >= float(np.max(real))
    assert bounds.gain > 0  # the second bin's medium has gain
    # The inactive bin's spectral rate alone is the binding term here.
    assert bounds.loss == pytest.approx(max(20.0, 0.5 * np.max(-2 * real)))


def test_the_transverse_rate_is_refused_where_it_is_not_a_rate():
    with pytest.raises(TranslationError, match="one nonnegative rate"):
        separable(kperp_rate=np.array([0.0, -1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(TranslationError, match="one nonnegative rate"):
        separable(kperp_rate=np.zeros(3))
    with pytest.raises(TranslationError, match="hybrid planar kind"):
        separable(kappa_sq=np.zeros((1, 4), complex), kperp_rate=np.ones(5))
    # Absent, it is held as exact zeros.
    absent = separable(kperp_rate=None).kperp_rate
    assert absent is not None and np.array_equal(absent, np.zeros(5))
