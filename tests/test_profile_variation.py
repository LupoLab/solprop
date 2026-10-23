# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the variation enclosures and declared jumps of every profile."""

import math

import numpy as np
import pytest
from numpy.polynomial import Chebyshev

from bounds import EPS
from solprop.exceptions import TranslationError
from solprop.physdata import Gas
from solprop.profiles import (
    Constant,
    DensityPath,
    FromCallable,
    Linear,
    PathVariation,
    PiecewiseLinear,
    PressureGradient,
    Scaled,
    SeriesOfPath,
    Sum,
)

FILL = PressureGradient.from_stages([0.0, 0.6, 1.5], [1.0, 5.0, 0.2])
BORE = PiecewiseLinear.from_stages([0.0, 0.3, 1.0, 1.5], [125e-6, 100e-6, 60e-6, 90e-6])
SAMPLES = 20001


def cases():
    """Profiles, whether their pieces are monotone, and whether the record is exact.

    A record is exact where every piece is monotone, or turns only at points the
    enclosure evaluates (a series' critical points).
    """
    falling = Linear(3.0, 1.0, 1.5)
    return {
        "Constant": (Constant(2.0), True, True),
        "Linear": (Linear(125e-6, 75e-6, 1.5), True, True),
        "PiecewiseLinear": (
            PiecewiseLinear.from_stages([0.0, 0.5, 1.0, 1.5], [1.0, 3.0, 2.0, 4.0]),
            True,
            True,
        ),
        "PressureGradient": (FILL, True, True),
        "tapered PressureGradient": (
            PressureGradient.from_stages([0.0, 0.6, 1.5], [1.0, 5.0, 0.2], radius=BORE),
            True,
            True,
        ),
        "DensityPath": (DensityPath(Gas.Ar, FILL), True, True),
        "Scaled": (Scaled(FILL, -2.0), True, True),
        "co-monotone Sum": (Sum((FILL, Scaled(FILL, 0.5))), True, True),
        "opposed Sum": (Sum((Linear(1.0, 3.0, 1.5), falling)), False, False),
        "SeriesOfPath": (
            SeriesOfPath(
                Linear(-1.0, 1.0, 1.5),
                Chebyshev([0.0, 0.0, 1.0], domain=[-1.0, 1.0]),
            ),
            False,
            True,
        ),
        "FromCallable": (
            FromCallable(np.cos, (0.0, 4.0), turning=np.array([math.pi])),
            True,
            True,
        ),
    }


CASES = cases()


def sampled(path, start, end, knots):
    """Return the sampled total variation, a lower bound, and the sampled range."""
    z = np.unique(np.concatenate([np.linspace(start, end, SAMPLES), knots]))
    values = path(z)
    return float(np.sum(np.abs(np.diff(values)))), values


@pytest.mark.parametrize("name", list(CASES))
def test_every_enclosure_holds_and_monotone_pieces_are_exact(name):
    path, monotone, exact = CASES[name]
    start, end = 0.1, 1.4
    record = path.variation(start, end)
    assert isinstance(record, PathVariation)
    assert record.knots[0] == start
    assert record.knots[-1] == end
    total, values = sampled(path, start, end, record.knots)
    # Each sample rounds within a few ulps of the path's scale, so ulp-level
    # reversals of a monotone path add at most 4 eps max|f| per sample to the
    # sampled variation; sampling can otherwise only underestimate.
    scale = float(np.max(np.abs(values)))
    noise = 4 * EPS * scale * SAMPLES
    assert total <= record.total_variation + noise
    assert np.min(values) >= np.min(record.low) - noise
    assert np.max(values) <= np.max(record.high) + noise
    assert bool(np.all(record.monotone)) == monotone
    if exact:
        # The samples include every knot, so on monotone pieces they see the
        # whole variation, and a turning series' extreme to its sampling.
        assert record.total_variation == pytest.approx(total, abs=noise)


def test_piecewise_linear_records_every_stage_exactly():
    path = PiecewiseLinear.from_stages([0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 1.0, 4.0])
    record = path.variation(0.5, 2.5)
    # Exact: the knots are the interval ends and the interior stage positions,
    # and the limits are the interpolant's values there.
    assert record.knots.tolist() == [0.5, 1.0, 2.0, 2.5]
    assert record.first.tolist() == [1.0, 2.0, 1.0]
    assert record.last.tolist() == [2.0, 1.0, 2.5]
    assert record.total_variation == 3.5
    assert record.jumps.size == 0
    # Outside the declared range the path is held: flat pieces.
    assert path.variation(3.0, 5.0).total_variation == 0.0


def test_a_fill_varies_by_its_stage_pressures():
    fill = PressureGradient.from_stages([0.0, 0.6, 1.5], [1.0, 5.0, 0.2])
    record = fill.variation(0.0, 1.5)
    # The stage ends are evaluated through the convex combination, which returns
    # a stage pressure to within one rounding of the square root.
    assert record.total_variation == pytest.approx(4.0 + 4.8, rel=4 * EPS)


def test_a_density_through_a_peaked_fill_turns_once():
    path = DensityPath(Gas.Ar, PressureGradient.from_stages([0, 0.6, 1.5], [1, 5, 1]))
    record = path.variation(0.0, 1.5)
    rho = path(np.array([0.0, 0.6, 1.5]))
    # Up and down, from the spline's own values at the stage ends.
    expected = (rho[1] - rho[0]) + (rho[1] - rho[2])
    assert record.total_variation == pytest.approx(expected, rel=8 * EPS)
    assert record.monotone.tolist() == [True, True]


def test_an_opposed_sum_is_bounded_by_the_triangle_inequality():
    rising, falling = Linear(1.0, 3.0, 1.5), Linear(3.0, 1.0, 1.5)
    record = Sum((rising, falling)).variation(0.0, 1.5)
    # The sum is constant, but the enclosure cannot know the slopes cancel.
    assert record.total_variation == 4.0
    assert not record.monotone[0]


def test_a_callable_declares_its_jumps_and_turning_points():
    path = FromCallable(
        lambda z: np.where(z < 0.5, 1.0 + z, 3.0 - z),
        (0.0, 1.0),
        jumps=np.array([0.5]),
        turning=np.array([]),
    )
    assert path.jumps.tolist() == [0.5]
    assert path.breakpoints.tolist() == [0.5]
    record = path.variation(0.0, 1.0)
    assert record.jumps.tolist() == [0.5]
    # 1 -> 1.5 rising, a jump up to 2.5, then 2.5 -> 2 falling.
    assert record.total_variation == pytest.approx(0.5 + 1.0 + 0.5, rel=4 * EPS)


def test_a_callable_without_declared_variation_refuses():
    path = FromCallable(np.cos, (0.0, 4.0))
    with pytest.raises(TranslationError, match="no declared variation"):
        path.variation(0.0, 1.0)
    declared = FromCallable(np.cos, (0.0, 4.0), turning=np.array([math.pi]))
    with pytest.raises(TranslationError, match="leaves the declared span"):
        declared.variation(0.0, 5.0)


@pytest.mark.parametrize("jumps", [[0.0], [1.0], [0.6, 0.5]])
def test_a_callable_rejects_a_bad_jump_declaration(jumps):
    with pytest.raises(TranslationError, match="jumps"):
        FromCallable(np.cos, (0.0, 1.0), jumps=np.array(jumps))


def test_a_density_under_a_temperature_path_refuses_an_enclosure():
    path = DensityPath(Gas.He, Constant(1.0, (0.0, 1.0)), Linear(293.15, 350.0, 1.0))
    with pytest.raises(TranslationError, match="temperature varies"):
        path.variation(0.0, 1.0)


def test_a_record_refuses_bounds_that_miss_its_limits():
    with pytest.raises(TranslationError, match="enclose"):
        PathVariation(
            np.array([0.0, 1.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.5]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([True]),
        )
