# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's far-field radius to the bit, and its pass at capped bands (SEP 43, 36, 37).

**The radius.** Every hard-mask decision rests on native forming the mask-plane radius
exactly as the reference does: products, sums, two squares, a sum and a correctly
rounded root, never contracted. The retired test compared 40 x 7 random inputs for exact
equality. The surviving C++ test uses exactly representable values, so a contraction
would pass it. Native's radius cannot be read, but it can be located: a hard hole of
exactly the reference's radius holds the column, and one a unit in the last place
smaller does not, so native's radius is the reference's, to the bit, if both decisions
come out so. The test first checks that these inputs could tell: a fused product would
differ on some of them, in the same arithmetic.

**The band cap.** A pass holds at most ``FAR_FIELD_BAND_BUDGET`` bytes of band partials.
The retired test lowered the budget to two bands. Native's is compiled in, so this makes
one real problem reach it, with 16 holes on a grid of 64 rows and about 2000 bins, and
holds its values at the same bound.

**Weaker than the original.** The scale is the grid's ``z c / omega`` and the axes the
grid's own, not free random numbers. The cap is what such a problem reaches, not two.
"""

import math

import numpy as np
import pytest

from backends.far_field_cases import (
    DISTANCE,
    assert_matches,
    magnitudes,
    reference,
    spec_of,
    time_grid,
)
from backends.test_native_far_field import native_stats, problem_of
from solprop.backends.spatial import FAR_FIELD_BAND_BUDGET, far_field_partials
from solprop.grid import FieldGrid, FreeGrid
from solprop.spatial import (
    FAR_FIELD_STATISTICS,
    aperture_outline,
    aperture_scale,
    mask_plane_offsets,
    mask_plane_radius,
)
from solprop.stats import ApertureSet, StatsSpec
from solprop.translate.spatial import aperture_tables

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend

CENTRE = (3.1e-4, -7.3e-4)
"""The original's centre: neither coordinate is a short binary fraction."""

N_BINS = 40
"""The original's scales, one per active bin."""

N_COLUMNS = 7
"""The original's pairs of axes, one per column."""

HOLES_PER_SET = 16
"""The most an ``ApertureSet`` holds."""


def edge_inputs(grid, space):
    """Return the columns, the bins and the scale row of the original's random draw."""
    rng = np.random.default_rng(3)
    columns = rng.choice(space.n_columns, N_COLUMNS, replace=False)
    active = np.flatnonzero(np.asarray(grid.active_mask))
    bins = np.sort(rng.choice(active, N_BINS, replace=False))
    return columns, bins


def radii(tables, space, column, bins):
    """Return the reference's radius of `column` at `bins`, ``(n_bins,)``."""
    k_x = tables.k_x[column % space.n_x : column % space.n_x + 1]
    k_y = tables.k_y[column // space.n_x : column // space.n_x + 1]
    return mask_plane_radius(tables.scale[0, bins], k_x, k_y, CENTRE)[0]


def test_the_radius_is_the_reference_radius_to_the_bit():
    """Each hole decides each column as the reference's radius says, at the last bit."""
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 12, 3e-4, 10, collar_fraction=0)
    columns, all_bins = edge_inputs(grid, space)
    tables = aperture_tables(
        spec_of(_holes(np.array([1e-3])), ("aperture_spectrum",)), grid, space
    )
    assert tables is not None
    fused = 0
    for column in columns:
        radius = radii(tables, space, int(column), all_bins)
        # The test can tell: a contracted radius differs from this on some inputs.
        fused += _count_fused(tables, space, int(column), all_bins, radius)
        # A set holds 16 holes: a bin's radius, and a unit in the last place inside it.
        for start in range(0, N_BINS, HOLES_PER_SET // 2):
            bins = all_bins[start : start + HOLES_PER_SET // 2]
            at = 2 * radius[start : start + HOLES_PER_SET // 2]
            diameters = np.concatenate((at, np.nextafter(at, 0)))
            spec = StatsSpec(names=("aperture_spectrum",), apertures=_holes(diameters))
            field = np.zeros((space.n_columns, 1, grid.n_omega), complex)
            field[column, 0, bins] = 1e3
            problem, run_request = problem_of(grid, space, field, spec)
            values, held = native_stats(problem, run_request, grid, field, spec.names)
            here = aperture_tables(spec, problem.grids, problem.transform.geometry)
            assert here is not None
            expected = reference(spec.names, grid, problem, held, here)
            spectrum = values["aperture_spectrum"]
            oracle = expected["aperture_spectrum"]
            inner = bins.size
            for index, bin_ in enumerate(bins):
                # Structural: the membership is exact: the term is in both or neither.
                assert spectrum[index, bin_] > 0
                assert oracle[index, bin_] > 0
                assert spectrum[inner + index, bin_] == 0
                assert oracle[inner + index, bin_] == 0
    assert fused > 0


def test_a_half_plane_and_a_rectangle_decide_as_the_reference_to_the_bit():
    """Each edge on a column's image, and a unit in the last place either side of it.

    Native's offset can be located as the radius was: a half-plane's edge exactly at the
    reference's image holds the column, and one a unit further out does not, whichever
    side it keeps. A fused offset would differ on some of these inputs.
    """
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 12, 3e-4, 10, collar_fraction=0)
    columns, all_bins = edge_inputs(grid, space)
    bins = all_bins[: N_BINS // 8]
    scale = aperture_scale(np.array([DISTANCE]), grid.omega)[0, bins]
    fused = 0
    for column in columns:
        k_x = float(space.k_x[column % space.n_x])
        if k_x == 0.0:
            continue
        # `across` is exactly zero at X = -(s k), since the product is the one rounded.
        at = np.array([-(float(one) * k_x) for one in scale])
        fused += sum(
            math.fma(float(one), k_x, float(edge)) != 0.0
            for one, edge in zip(scale, at, strict=True)
        )
        further = np.nextafter(at, np.where(at > 0, np.inf, -np.inf))
        nearer = np.nextafter(at, 0.0)
        rows = 3 * bins.size
        # Stored k > 0 means physical kappa < 0: nearer the origin is above the image.
        field = np.zeros((space.n_columns, 1, grid.n_omega), complex)
        field[column, 0, bins] = 1e3
        centres = np.column_stack(
            (np.concatenate((at, further, nearer)), np.zeros(rows))
        )
        for shape in ("half_plane_x_ge", "half_plane_x_lt"):
            apertures = ApertureSet(
                centres=centres[:HOLES_PER_SET],
                diameters=np.zeros(min(rows, HOLES_PER_SET)),
                distances=np.full(min(rows, HOLES_PER_SET), DISTANCE),
                shapes=(shape,) * min(rows, HOLES_PER_SET),
            )
            spec = StatsSpec(names=("aperture_spectrum",), apertures=apertures)
            problem, run_request = problem_of(grid, space, field, spec)
            values, held = native_stats(problem, run_request, grid, field, spec.names)
            here = aperture_tables(spec, problem.grids, problem.transform.geometry)
            assert here is not None
            expected = reference(spec.names, grid, problem, held, here)
            spectrum = values["aperture_spectrum"]
            oracle = expected["aperture_spectrum"]
            for row in range(min(rows, HOLES_PER_SET)):
                bin_ = int(bins[row % bins.size])
                # Structural: the membership is exact: the term is in both or neither.
                assert (spectrum[row, bin_] > 0) == (oracle[row, bin_] > 0)
            # The edge itself belongs to the side kept by x >= X, and only to it.
            for row, bin_ in enumerate(bins):
                assert (oracle[row, bin_] > 0) == (shape == "half_plane_x_ge")
    assert fused > 0


def test_a_rectangle_and_an_ellipse_decide_as_the_reference_at_their_edges():
    """Each edge on a column's image, a rectangle's directly, an ellipse's by bisection.

    The rectangle's half-width is the offset itself; the ellipse's stretches move with
    its size, so the last size that holds the column is found on the reference's outline
    and native is asked of that size and the next one down.
    """
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 12, 3e-4, 10, collar_fraction=0)
    columns, all_bins = edge_inputs(grid, space)
    bin_ = int(all_bins[N_BINS // 2])
    scale = aperture_scale(np.array([DISTANCE]), grid.omega)[0, bin_ : bin_ + 1]
    for column in columns:
        k_x = space.k_x[column % space.n_x : column % space.n_x + 1]
        k_y = space.k_y[column // space.n_x : column // space.n_x + 1]
        across, along = mask_plane_offsets(scale, k_x, k_y, CENTRE)
        across, along = float(across[0, 0]), float(along[0, 0])
        width = 2 * abs(across)
        hold = _ellipse_threshold(across, along)
        below = np.nextafter(hold, 0.0)
        rows = [
            ("rectangle", (width, 4 * abs(along) + 1e-3), True),
            ("rectangle", (np.nextafter(width, 0.0), 4 * abs(along) + 1e-3), False),
            ("ellipse", (hold, hold / 2), True),
            ("ellipse", (below, below / 2), False),
        ]
        spec = StatsSpec(
            names=("aperture_spectrum",),
            apertures=ApertureSet(
                centres=[list(CENTRE)] * len(rows),
                diameters=np.zeros(len(rows)),
                distances=np.full(len(rows), DISTANCE),
                shapes=tuple(row[0] for row in rows),
                extents=[list(row[1]) for row in rows],
            ),
        )
        field = np.zeros((space.n_columns, 1, grid.n_omega), complex)
        field[column, 0, bin_] = 1e3
        problem, run_request = problem_of(grid, space, field, spec)
        values, held = native_stats(problem, run_request, grid, field, spec.names)
        here = aperture_tables(spec, problem.grids, problem.transform.geometry)
        assert here is not None
        oracle = reference(spec.names, grid, problem, held, here)["aperture_spectrum"]
        for row, (_, _, inside) in enumerate(rows):
            # Structural: the membership is exact: the term is in both or neither.
            assert (oracle[row, bin_] > 0) == inside
            assert (values["aperture_spectrum"][row, bin_] > 0) == inside


def _ellipse_threshold(across, along):
    """Return the smallest x width of a 2:1 ellipse whose outline holds the offsets."""
    holds = lambda width: (  # noqa: E731 - one use, the bisection's predicate
        float(
            aperture_outline("ellipse", 0.0, (width, width / 2), 0.0, "hard").weight(
                np.array([across]), np.array([along])
            )[0]
        )
        == 1.0
    )
    low, high = 0.0, 8 * math.hypot(across, 2 * along) + 1e-12
    assert holds(high)
    while True:
        middle = 0.5 * (low + high)
        if middle in (low, high):
            return high
        if holds(middle):
            high = middle
        else:
            low = middle


def _holes(diameters):
    """Return hard holes of `diameters` at the one centre, at the one distance."""
    return ApertureSet(
        centres=np.tile(CENTRE, (diameters.size, 1)),
        diameters=diameters,
        distances=[DISTANCE] * diameters.size,
    )


def _count_fused(tables, space, column, bins, radius):
    """Return how many of the radii a fused multiply-add would change."""
    k_x = float(tables.k_x[column % space.n_x])
    k_y = float(tables.k_y[column // space.n_x])
    count = 0
    for value, scale in zip(radius, tables.scale[0, bins], strict=True):
        across = math.fma(float(scale), k_x, CENTRE[0])
        along = math.fma(float(scale), k_y, CENTRE[1])
        count += math.sqrt(across * across + along * along) != value
    return count


def test_a_problem_at_the_band_cap_is_the_reference_s():
    """The budget's value witness: a real problem that the cap holds below its rows.

    At 16 holes on 64 rows and about 2000 bins the budget admits fewer bands than rows,
    so a band takes more than one, in ascending order just the same. Same bound as every
    far-field parity case (``far_field_cases``).
    """
    grid = FieldGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 3e-12)
    space = FreeGrid(3e-4, 4, 2.5e-4, 64, collar_fraction=0)
    rng = np.random.default_rng(5)
    apertures = ApertureSet(
        centres=rng.uniform(-1e-3, 1e-3, (HOLES_PER_SET, 2)),
        diameters=rng.uniform(1e-3, 2e-3, HOLES_PER_SET),
        distances=[DISTANCE] * HOLES_PER_SET,
        edges=np.full(HOLES_PER_SET, 1e-4),
        profile="tanh",
        weighting="amplitude",
    )
    spec = spec_of(apertures, FAR_FIELD_STATISTICS)
    shape = (space.n_columns, 1, grid.n_omega)
    field = 1e3 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    field[..., 0] = field[..., -1] = 0.0
    problem, run_request = problem_of(grid, space, field, spec)
    n_slots, n_bands = far_field_partials(problem, run_request.observation.stats)
    # The cap is what decides, not the rows.
    assert n_bands == FAR_FIELD_BAND_BUDGET // (8 * n_slots * grid.n_omega)
    assert 1 < n_bands < space.n_y
    actual, held = native_stats(problem, run_request, grid, field, FAR_FIELD_STATISTICS)
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    assert tables is not None
    expected = reference(FAR_FIELD_STATISTICS, grid, problem, held, tables)
    assert_matches(
        actual,
        expected,
        magnitudes(FAR_FIELD_STATISTICS, grid, problem, held, tables),
        problem,
        names=FAR_FIELD_STATISTICS,
    )
