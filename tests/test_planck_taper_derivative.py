# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the analytic Planck-taper derivative of SEP 26 section 4.2 (test T1).

Every test is fast: a few thousand evaluations of closed forms. The derivative is
compared with an independent closed form built from ``planck_taper`` itself, with a
fourth-order finite difference whose error must fall at its order, and with the
sampled edge weight; its evaluation is checked to be finite at every breakpoint and
at widths from 1e-30 s to 1 s, with every floating-point exception raised.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from solprop.exceptions import GridError
from solprop.grid import (
    Collar,
    planck_taper,
    planck_taper_derivative,
    planck_taper_derivative_error,
)
from solprop.translate.tdse import AtomicWindowRecord, lattice_times

EPS = np.finfo(float).eps


def _edges(width):
    """Return dyadic breakpoints ``(0, w, 3w, 4w)`` and each edge's outer/inner."""
    breakpoints = (0.0, width, 3 * width, 4 * width)
    return breakpoints, ((0.0, width), (4 * width, 3 * width))


@pytest.mark.parametrize("width", [2.0**-50, 2.0**-47, 1.0], ids=["0.9fs", "7fs", "1s"])
def test_derivative_matches_the_independent_closed_form_inside_each_edge(width):
    breakpoints, edges = _edges(width)
    steps = np.arange(1, 4096) * (width / 4096)
    for outer, inner in edges:
        # Dyadic points and widths: x - a, b - x and the mirror a + b - x are exact,
        # so the comparison sees only the two evaluations' own rounding.
        x = min(outer, inner) + steps
        gate = planck_taper(x, *breakpoints)
        # 1 - g(x) = g(a + b - x) on the same edge (zeta changes sign), formed by a
        # second gate evaluation rather than by the cancelling subtraction 1 - g.
        complement = planck_taper(outer + inner - x, *breakpoints)
        delta = inner - outer
        slope = -delta / (x - outer) ** 2 - delta / (x - inner) ** 2
        oracle = -gate * complement * slope
        zeta = np.abs(delta / (x - outer) + delta / (x - inner))
        derivative = planck_taper_derivative(x, *breakpoints)
        kept = (gate > 0.01) & (gate < 0.99)
        # Tolerance per sample, derived before evaluation: the derivative's own bound
        # (18 + 3|zeta|) eps |g'| (planck_taper_derivative_error), plus the oracle's.
        # Each gate value carries (8 + 2|zeta|) eps (the same count as the derivative:
        # its exponent's terms 3u each, the sum 4u|zeta| + 12u, exp, 1 + e and the
        # division), d zeta/dx 3 eps and the two products eps: (20 + 4|zeta|) eps.
        # Measured: at most 2.0 eps against a bound of at least 38 eps.
        bound = planck_taper_derivative_error(x, *breakpoints) + (
            20 + 4 * zeta
        ) * EPS * np.abs(oracle)
        assert np.count_nonzero(kept) > 1000
        assert np.all(np.abs(derivative - oracle)[kept] <= bound[kept])
        assert np.all(np.sign(derivative[kept]) == np.sign(delta))


def test_fourth_order_difference_converges_at_its_order_to_the_derivative():
    breakpoints = (0.0, 1.0, 3.0, 4.0)
    points = np.linspace(0.2, 0.8, 61)
    exact = planck_taper_derivative(points, *breakpoints)

    def difference(spacing):
        gate = lambda offset: planck_taper(points + offset, *breakpoints)  # noqa: E731
        return (
            gate(-2 * spacing)
            - 8 * gate(-spacing)
            + 8 * gate(spacing)
            - gate(2 * spacing)
        ) / (12 * spacing)

    coarse, fine = (
        float(np.max(np.abs(difference(spacing) - exact)))
        for spacing in (2.0**-7, 2.0**-8)
    )
    # Truncation h^4 g^(5)/30 dominates rounding (about eps/h = 6e-14) by seven
    # orders, so halving h divides the error by 2^4 = 16 within the SEP's +-2.
    # Measured: 2.46e-6 and 1.54e-7, a ratio of 15.96.
    assert 14 <= coarse / fine <= 18


def _edge_weights(samples):
    """Return ``dt sum |g'|`` of each edge of a taper ``samples`` samples wide."""
    spacing = 1e-17
    # An off-grid lattice: breakpoints are never moved to samples.
    times = (np.arange(-50, 4 * samples + 50) + 0.37) * spacing
    width = samples * spacing
    derivative = planck_taper_derivative(times, 0.0, width, 3 * width, 4 * width)
    return (
        spacing * float(np.sum(derivative[derivative > 0])),
        -spacing * float(np.sum(derivative[derivative < 0])),
    )


def test_sampled_edge_weight_is_reported_against_one():
    weights = {samples: _edge_weights(samples) for samples in (64, 128, 800)}
    # Reported against 1, with no convergence-rate claim. Measured: 64 samples
    # 1 + 2.9e-11 on each edge; 128 and 800 samples 1 to rounding. The 128- and
    # 800-sample weights agree within the certificate threshold 1e-10 (SEP 26
    # section 4.4), which is the claim.
    for edge in range(2):
        assert abs(weights[128][edge] - weights[800][edge]) <= 1e-10
        assert abs(weights[64][edge] - 1) <= 1e-9
        assert abs(weights[800][edge] - 1) <= 1e-12


@pytest.mark.parametrize("width", [1e-30, 1e-21, 2.0**-50, 5e-15, 1e-12, 1.0], ids=str)
def test_evaluation_is_finite_at_every_breakpoint_and_width(width):
    breakpoints = (-2 * width, -width, width, 2 * width)
    special = []
    for point in breakpoints:
        special += [point, np.nextafter(point, -np.inf), np.nextafter(point, np.inf)]
    for outer, inner in (
        (breakpoints[0], breakpoints[1]),
        (breakpoints[3], breakpoints[2]),
    ):
        cut = outer + (inner - outer) / 740
        far = inner - (inner - outer) / 740
        for point in (cut, far):
            special += [point, np.nextafter(point, outer), np.nextafter(point, inner)]
    special += list(np.linspace(-3 * width, 3 * width, 601))
    points = np.asarray(special)
    with np.errstate(all="raise", under="ignore"):
        derivative = planck_taper_derivative(points, *breakpoints)
        bound = planck_taper_derivative_error(points, *breakpoints)
    assert np.all(np.isfinite(derivative)) and np.all(np.isfinite(bound))
    assert np.all(bound >= 0)
    # h = g' * width is non-negative on both edges: g' >= 0 rising, <= 0 falling.
    rising = (points > breakpoints[0]) & (points < breakpoints[1])
    falling = (points > breakpoints[2]) & (points < breakpoints[3])
    assert np.all(derivative[rising] >= 0) and np.all(derivative[falling] <= 0)
    assert np.max(derivative[rising]) * width == pytest.approx(2.0, rel=1e-3)
    # Exactly zero at and beyond the breakpoints, where the analytic g' is zero.
    outside = ~(rising | falling)
    assert_array_equal(derivative[outside], np.zeros(np.count_nonzero(outside)))
    assert_array_equal(bound[outside], np.zeros(np.count_nonzero(outside)))


def test_cut_off_region_returns_zero_and_bounds_what_it_discards():
    breakpoints = (0.0, 1.0, 3.0, 4.0)
    inside = np.array([1 / 740, 1 / 1000, 1e-300, 1 - 1 / 1000, 4 - 1e-6])
    derivative = planck_taper_derivative(inside, *breakpoints)
    assert_array_equal(derivative, np.zeros(inside.size))
    # The analytic value there is below 1e-300 / |width| (SEP 26 section 4.2).
    assert_array_equal(planck_taper_derivative_error(inside, *breakpoints), 1e-300)
    beyond = np.nextafter(1 / 740, 1.0)
    assert 0.0 < planck_taper_derivative(np.array([beyond]), *breakpoints)[0] < 1e-300


@pytest.mark.parametrize(
    "breakpoints",
    [(0.0, 0.0, 1.0, 2.0), (0.0, 2.0, 1.0, 3.0), (0.0, 1.0, 2.0, 2.0)],
)
def test_unordered_breakpoints_are_refused(breakpoints):
    with pytest.raises(GridError, match="breakpoints"):
        planck_taper_derivative(np.zeros(1), *breakpoints)
    with pytest.raises(GridError, match="breakpoints"):
        planck_taper_derivative_error(np.zeros(1), *breakpoints)


def test_an_edge_whose_slope_would_overflow_is_refused():
    width = 1e-309
    with pytest.raises(GridError, match="unrepresentable slope"):
        planck_taper_derivative(np.zeros(1), 0.0, width, 1.0, 2.0)


def test_collar_and_window_expose_the_same_derivative():
    collar = Collar(-4e-15, -3e-15, 3e-15, 4.5e-15)
    times = lattice_times(512, 2e-17)
    assert_array_equal(
        collar.window_derivative(times),
        planck_taper_derivative(times, -4e-15, -3e-15, 3e-15, 4.5e-15),
    )
    window = AtomicWindowRecord(
        6e-15,
        1e-15,
        -3e-15,
        3e-15,
        106,
        406,
        -3e-15,
        3e-15,
        (-3e-15, -2e-15, 2e-15, 3e-15),
    )
    assert_array_equal(
        window.gate_derivative(times),
        planck_taper_derivative(times, *window.gate),
    )
