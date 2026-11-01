"""The k-perp safety ramp, the attenuation search and the physical evanescent rate.

SEP 22 sections 4.3 to 4.5 and SEP 40 section 4.4. The safety ramp's profile, as
translation resolves it, is checked point by point (exact zeros, its strength at the
grid's edge, its slope and its column order); the sorted search and the rate bounds
against a brute-force maximum over every column.
"""

import numpy as np
import pytest

from bounds import gamma
from solprop.boundaries import SpatialAbsorber
from solprop.exceptions import GridError, TranslationError
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid, RadialGrid
from solprop.linear import Frame
from solprop.modes import PlanarGrid, SlabMode
from solprop.spatial import (
    EVANESCENT_RATE,
    SpatialDispersion,
    SpatialMedium,
    attenuation_extremes,
    physical_evanescent_rate,
)
from solprop.spatial_absorber import SpatialAbsorberDesign
from solprop.translate.spatial import SeparableOperator, rate_bounds, separable_values
from solprop.translate.transverse import resolve_transverse


def ramp(rho: np.ndarray, fraction: float) -> np.ndarray:
    """Section 4.3's P(rho), written out independently of the module."""
    out = np.zeros_like(rho)
    inside = rho > 1 - fraction
    out[inside] = 0.5 * (1 - np.cos(np.pi * (rho[inside] - 1 + fraction) / fraction))
    return out


def resolved(space: FreeGrid, fraction: float = 0.1) -> SpatialAbsorberDesign:
    """The safety ramp a vacuum run on `space` resolves, ramp alone, no launch."""
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (700e-9, 950e-9), 200e-15)
    dispersion = SpatialDispersion(SpatialMedium(), grid, Frame.comoving(grid))
    launch = np.zeros((1, space.n_columns, 1, grid.n_omega), complex)
    specification = SpatialAbsorber(real_space=False, safety_fraction=fraction)
    return resolve_transverse(specification, space, dispersion, launch, grid.length)


def test_the_ramp_is_zero_inside_and_its_strength_at_the_edge():
    space = FreeGrid(1e-4, 40, 2e-4, 24)
    design = resolved(space, 0.25)
    rate, strength = design.ramp_rate, design.ramp_strength
    rho_x = np.abs(space.k_x) / np.max(np.abs(space.k_x))
    rho_y = np.abs(space.k_y) / np.max(np.abs(space.k_y))
    # Column ix + n_x*iy, the max-norm of the two axes.
    rho = np.maximum(rho_x[None, :], rho_y[:, None]).reshape(-1)
    # One cosine and a few products: a handful of roundings of the strength.
    np.testing.assert_allclose(
        rate, strength * ramp(rho, 0.25), rtol=0, atol=gamma(8) * strength
    )
    assert np.all(rate[rho <= 0.75] == 0.0)  # exactly zero inside: a structural zero
    edge = np.flatnonzero(rho == 1.0)
    assert edge.size and np.all(rate[edge] == strength)
    assert rate.min() >= 0.0 and rate.max() == strength
    # The strength is the larger of the two axes' designs.
    assert strength == max(design.notes["ramp_strengths"])  # pyright: ignore[reportArgumentType]


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
    design = resolved(space, fraction)
    expected = ramp(np.abs(space.k_x) / np.max(np.abs(space.k_x)), fraction)
    np.testing.assert_allclose(
        design.ramp_rate / design.ramp_strength, expected, rtol=0, atol=gamma(8)
    )


def test_a_line_takes_its_ramp_from_k_x_alone():
    space = LineGrid(1e-4, 40)
    design = resolved(space)
    assert design.ramp_rate.shape == (40,)
    assert len(design.kappa_s) == 1
    # m = 19 of 20 is rho = 0.95, the ramp's midpoint for the default fraction. Its
    # argument (|k| - k_s) / (k_max - k_s) subtracts, losing k_max / (k_max - k_s) =
    # 1 / 0.1 of the roundings of |k|.
    index = int(np.argmin(np.abs(space.k_x - 19 * space.k_x[1])))
    assert design.ramp_rate[index] == pytest.approx(
        0.5 * design.ramp_strength, rel=gamma(8) / 0.1
    )


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
    # Absent, it is held as exact zeros.
    absent = separable(kperp_rate=None).kperp_rate
    assert absent is not None and np.array_equal(absent, np.zeros(5))
