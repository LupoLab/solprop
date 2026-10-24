# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import math
import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.polynomial import chebyshev as chebyshev_basis
from scipy.integrate import quad_vec

from bounds import EPS, assert_within
from convergence_records import Column, Study, record
from solprop.exceptions import TranslationError
from solprop.translate import chebyshev
from solprop.translate.chebyshev import (
    PiecewiseChebyshev,
    build,
    build_from_paths,
)

# The mirror-extension FFT of length 2n loses O(log2(2n) * eps) and the Clenshaw sum
# over n_deg terms loses Higham's n * eps; at the builder's highest degree, 64, that
# is 64 * 2.2e-16 = 1.4e-14. Round up once and reuse.
ROUND_OFF_ATOL = 5e-14


def clenshaw_bound(coefficients, axis=0):
    """Return how far two Clenshaw sums of `coefficients` on ``[-1, 1]`` may differ.

    A Clenshaw pass of ``n`` terms is within ``O(n u) sum |c_k|`` of the exact sum
    (Higham, *Accuracy and Stability*, section 5.7); ``4 (n + 1) eps sum |c_k|`` takes
    that with a factor-of-four margin, per element of the trailing shape, and two
    passes that order or contract their operations differently are within twice it,
    which is what is returned.
    """
    magnitude = np.abs(np.asarray(coefficients))
    terms = magnitude.shape[axis]
    return 2 * 4 * (terms + 1) * EPS * np.sum(magnitude, axis=axis)


# The relative agreement required between the built integral and quad_vec.
# quad_vec is asked for 1e-13 and its own reported error estimate is used as the
# comparison tolerance as well, so the reference's accuracy is verified, not assumed.
ORACLE_RTOL = 1e-13


def smooth(z):
    return np.exp(np.sin(3.0 * z)) + 0.3 * z**2


def stepped(z):
    """A ramp with a jump of 0.8 at z = 0.4, written the way a user would write it."""
    return 1.0 + 0.6 * z + np.where(z >= 0.4, 0.8, 0.0)


def two_stage_pressure(z):
    """Steady flow: p**2 is linear in z on each stage, so p ~ sqrt(z) at a low start."""
    z = np.asarray(z, dtype=float)
    first = 0.2**2 + (z / 0.6) * (5.0**2 - 0.2**2)
    second = 5.0**2 + ((z - 0.6) / 0.4) * (1.0**2 - 5.0**2)
    return np.sqrt(np.where(z < 0.6, first, second))


# --------------------------------------------------------------------- exactness


@pytest.mark.parametrize(
    "coefficients",
    [[1.0], [0.5, -1.2], [1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, -2.0]],
)
def test_build_is_exact_for_polynomials_up_to_the_degree(coefficients):
    reference = np.asarray(coefficients)
    degree = reference.size - 1

    def polynomial(z):
        return chebyshev_basis.chebval(2.0 * z - 1.0, reference)

    representation = build(polynomial, 0.0, 1.0)

    assert representation.degrees.size == 1
    assert int(representation.degrees[0]) == degree
    probe = np.linspace(0.0, 1.0, 97)
    values = np.array([float(representation(float(z))) for z in probe])
    np.testing.assert_allclose(values, polynomial(probe), rtol=0.0, atol=ROUND_OFF_ATOL)


def test_build_collapses_a_constant_to_one_coefficient():
    representation = build(lambda z: np.full_like(z, 2.5), 0.0, 3.0)

    # ADR-005's fast path: one interval, and its integral is degree 1.
    assert representation.coefficients.shape == (1, 1)
    assert int(representation.degrees[0]) == 0
    assert float(representation(1.7)) == 2.5
    phi = representation.integral()
    assert int(phi.degrees[0]) == 1
    assert float(phi(3.0)) == pytest.approx(7.5, abs=ROUND_OFF_ATOL)


def test_build_collapses_a_piecewise_constant_to_one_interval_per_stage():
    def steps(z):
        return np.where(z < 0.25, 1.0, np.where(z < 0.75, 4.0, 2.0))

    representation = build(steps, 0.0, 1.0, mandatory_breakpoints=(0.25, 0.75))

    assert representation.degrees.tolist() == [0, 0, 0]
    np.testing.assert_array_equal(representation.breakpoints, [0.0, 0.25, 0.75, 1.0])
    assert float(representation.integral()(1.0)) == pytest.approx(
        0.25 + 2.0 + 0.5, abs=ROUND_OFF_ATOL
    )


# --------------------------------------------------------------------- integration


def test_integral_matches_quadrature_on_a_smooth_function():
    representation = build(smooth, 0.0, 2.0)
    phi = representation.integral()

    for z in (0.13, 0.5, 1.0, 1.7, 2.0):
        reference, error, info = quad_vec(
            lambda t: np.atleast_1d(smooth(t)),
            0.0,
            z,
            epsabs=1e-13,
            epsrel=1e-13,
            full_output=True,
        )
        # The oracle must be converged before it can be believed.
        assert info.success
        oracle = float(np.asarray(reference).item())
        deviation = abs(float(phi(z)) - oracle)
        # What the comparison actually rests on: the oracle's own reported accuracy
        # plus one Clenshaw pass of round-off on the scale of Phi.
        assert deviation <= error + ROUND_OFF_ATOL * abs(oracle)
        # And the target, stated directly.
        assert deviation <= ORACLE_RTOL * abs(oracle)


def test_integral_is_zero_at_the_start_and_continuous_across_every_breakpoint():
    phi = build(two_stage_pressure, 0.0, 1.0, mandatory_breakpoints=(0.6,)).integral()

    assert float(phi(0.0)) == 0.0
    scale = abs(float(phi(1.0)))
    assert phi.breakpoints.size > 2
    for index in range(1, phi.breakpoints.size - 1):
        # The two one-sided limits at an interior edge, taken from the polynomials
        # themselves rather than from a finite offset in z, which would add the value of
        # the integrand times the offset and hide the quantity under test.
        left = float(chebyshev_basis.chebval(1.0, phi.coefficients[index - 1]))
        right = float(chebyshev_basis.chebval(-1.0, phi.coefficients[index]))
        # Continuity is fixed exactly in the T_0 coefficient, so the two sides differ
        # only by the round-off of one Clenshaw pass over the scale of Phi.
        assert abs(left - right) < ROUND_OFF_ATOL * max(scale, 1.0)


def test_integral_telescopes():
    phi = build(smooth, 0.0, 2.0).integral()
    a, b, c = 0.3, 1.1, 1.9

    whole = float(phi(c)) - float(phi(a))
    parts = (float(phi(b)) - float(phi(a))) + (float(phi(c)) - float(phi(b)))
    # Exactly the same two Clenshaw evaluations enter both sides bar phi(b), which
    # cancels in exact arithmetic; the residue is one subtraction of order eps * scale.
    assert whole == pytest.approx(parts, abs=ROUND_OFF_ATOL * abs(whole))


def test_derivative_inverts_the_integral():
    representation = build(smooth, 0.0, 2.0, mandatory_breakpoints=(0.7,))
    recovered = representation.integral().derivative()

    probe = np.linspace(0.0, 2.0, 61)
    original = np.array([float(representation(float(z))) for z in probe])
    again = np.array([float(recovered(float(z))) for z in probe])
    np.testing.assert_allclose(again, original, rtol=0.0, atol=ROUND_OFF_ATOL)
    assert recovered.degrees.tolist() == representation.degrees.tolist()


def test_derivative_of_a_constant_interval_stays_a_single_zero_coefficient():
    derivative = build(lambda z: np.full_like(z, 2.5), 0.0, 1.0).derivative()

    assert derivative.coefficients.shape == (1, 1)
    assert derivative.degrees.tolist() == [0]
    assert float(derivative(0.5)) == 0.0


def test_derivative_matches_an_analytic_derivative():
    representation = build(smooth, 0.0, 2.0).derivative()

    probe = np.linspace(0.05, 1.95, 41)
    analytic = 3.0 * np.cos(3.0 * probe) * np.exp(np.sin(3.0 * probe)) + 0.6 * probe
    values = np.array([float(representation(float(z))) for z in probe])
    # Differentiating amplifies the interpolant's round-off by the degree squared over
    # the interval width; at degree ~30 on a span of 2 m that is a few times 1e-13.
    np.testing.assert_allclose(values, analytic, rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------- convergence


# The record of the ladder below, the "Chebyshev builds" row of
# `docs/dev/accuracy.md` section 1. Nothing here is near a round-off floor -- the
# finest rung is 5.2e-10 on a function of order unity -- so the tolerance is tight and
# `atol` only keeps a hypothetical zero from being compared relatively.
CHEBYSHEV_WIDTH = Study(
    name="chebyshev_width",
    title="Order of a fixed-degree Chebyshev interpolant in the interval width",
    case={
        "function": "the smooth test function of tests/translate/test_chebyshev.py",
        "degree": 8,
        "probe": "401 points, uniform on the interval",
    },
    axis={"name": "width", "unit": "1", "label": "interval width"},
    columns=(
        Column(
            name="width",
            unit="1",
            why="the ladder's input rather than a measurement, so it is compared "
            "exactly; it is here to give the figure a numeric axis",
            rtol=0.0,
        ),
        Column(
            name="error",
            unit="1 (absolute, on a function of order unity)",
            why="pure interpolation arithmetic three orders above any round-off "
            "floor, so the tolerance is the one a platform difference cannot reach",
            rtol=1e-9,
            atol=1e-15,
        ),
        Column(
            name="order",
            unit="1",
            why="a log2 of two errors each known to 1e-9, so the order is known far "
            "better than this; null on the coarsest rung, which has no pair",
            rtol=0.0,
            atol=1e-6,
        ),
    ),
)


def test_the_interpolant_converges_at_the_expected_order_in_the_interval_width(request):
    """The ladder of `docs/dev/accuracy.md` section 1, and the record it is drawn from.

    Measured: 2.225e-4, 3.227e-7 and 5.210e-10 at widths 0.8, 0.4 and 0.2, so 9.43 and
    9.27 against the degree-plus-one order of 9.
    """
    degree = 8
    widths = [0.8, 0.4, 0.2]
    errors = []
    for width in widths:
        # min_width equal to the span accepts the single interval at the initial degree,
        # so the only thing that changes between the three levels is the width.
        representation = build(
            smooth, 0.0, width, tol=1e-30, n_max=degree, min_width=width
        )
        assert representation.degrees.tolist() == [degree]
        probe = np.linspace(0.0, width, 401)
        values = np.array([float(representation(float(z))) for z in probe])
        errors.append(float(np.max(np.abs(values - smooth(probe)))))

    orders = [float(np.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)]
    expected = degree + 1
    assert all(order > expected - 1.0 for order in orders), orders
    # The order gate of docs/dev/testing.md section 1: an observed order more than three
    # above the expected one means the reference, not the method, is what is being
    # measured.
    assert all(order < expected + 3.0 for order in orders), orders
    record(
        CHEBYSHEV_WIDTH,
        request=request,
        rows=[
            {
                "series": f"degree {degree}",
                "axis": f"{width:g}",
                "width": width,
                "error": error,
                "order": None if index == 0 else orders[index - 1],
            }
            for index, (width, error) in enumerate(zip(widths, errors, strict=True))
        ],
    )


# --------------------------------------------------------------------- splitting


def test_a_declared_breakpoint_keeps_each_side_exact_and_is_reported_as_a_kink():
    representation = build(stepped, 0.0, 1.0, mandatory_breakpoints=(0.4,), tol=1e-14)

    # A ramp on each side needs exactly one interval of degree one; without the one-ulp
    # endpoint inset the left interval would sample the post-jump value at z = 0.4 and
    # bisect toward it until it hit the width floor.
    np.testing.assert_array_equal(representation.breakpoints, [0.0, 0.4, 1.0])
    assert representation.degrees.tolist() == [1, 1]
    assert representation.next_hard_kink(0.0) == 0.4
    assert float(representation(0.4 - 1e-12)) == pytest.approx(1.24, abs=1e-11)
    assert float(representation(0.4)) == pytest.approx(2.04, abs=ROUND_OFF_ATOL)
    assert float(representation.integral()(1.0)) == pytest.approx(
        1.0 + 0.3 + 0.8 * 0.6, abs=ROUND_OFF_ATOL
    )


def test_an_undeclared_jump_is_located_by_the_bisection():
    min_width = 1e-6
    representation = build(stepped, 0.0, 1.0, min_width=min_width)

    distances = np.abs(representation.breakpoints - 0.4)
    assert float(np.min(distances)) <= min_width
    assert representation.hard_kinks.size == 0
    # Only the neighbourhood of the jump is refined.
    widths = np.diff(representation.breakpoints)
    assert float(np.max(widths)) > 0.1


def test_a_square_root_cusp_splits_near_zero_and_nowhere_else():
    representation = build(np.sqrt, 0.0, 1.0, min_width=1e-6)

    edges = representation.breakpoints
    widths = np.diff(edges)
    # Geometric refinement into the branch point, not uniform subdivision.
    assert widths[0] < widths[-1] / 100.0
    # The two innermost intervals are the two halves of the last bisection, both at the
    # width floor; from there out every interval is twice its inner neighbour.
    assert widths[1] == widths[0]
    np.testing.assert_allclose(widths[2:7] / widths[1:6], 2.0, rtol=0.0, atol=1e-12)
    # And nowhere else: the whole right half is one interval.
    assert int(np.sum(edges[:-1] >= 0.5)) == 1
    # The endpoint inset is what keeps every sample inside the domain of sqrt.
    assert float(representation(0.75)) == pytest.approx(np.sqrt(0.75), abs=1e-13)


def test_the_narrowest_interval_is_local_and_not_a_global_step_cap():
    representation = build(np.sqrt, 0.0, 1.0, min_width=1e-6)

    narrowest = float(np.min(np.diff(representation.breakpoints)))
    assert representation.interval_width(1e-9) == pytest.approx(narrowest)
    # Capping every step by the narrowest interval anywhere costs orders of magnitude
    # in step count, so the accessor must report the local width.
    assert representation.interval_width(0.75) / narrowest > 1e4


def test_mandatory_breakpoints_may_be_given_unsorted_and_with_duplicates():
    representation = build(smooth, 0.0, 1.0, mandatory_breakpoints=[0.75, 0.25, 0.75])

    np.testing.assert_array_equal(representation.breakpoints, [0.0, 0.25, 0.75, 1.0])
    np.testing.assert_array_equal(representation.hard_kinks, [0.25, 0.75])


def test_the_domain_ends_are_not_hard_kinks():
    representation = build(smooth, 0.0, 1.0, mandatory_breakpoints=(0.0, 1.0))

    assert representation.hard_kinks.size == 0
    assert representation.next_hard_kink(0.0) is None


def test_the_default_width_floor_is_a_fraction_of_the_span():
    representation = build(np.sqrt, 0.0, 1.0)

    floor = chebyshev.MIN_WIDTH_FRACTION * 1.0
    assert float(np.min(np.diff(representation.breakpoints))) <= floor


# --------------------------------------------------------------------- array-valued


def _column(representation, index):
    """A scalar-valued view of one column of an array-valued representation."""
    return PiecewiseChebyshev(
        np.ascontiguousarray(
            representation.coefficients[(slice(None), slice(None), *index)]
        ),
        representation.breakpoints,
        representation.degrees,
        representation.hard_kinks,
        representation.budget_exceeded,
    )


def test_an_array_valued_build_agrees_with_per_column_scalar_builds():
    omega = np.linspace(1.0e15, 5.0e15, 2049)
    scale = np.array([1.0, 0.5, 2.0, 0.25])

    def operator(z):
        return (
            -1j
            * scale[None, :, None]
            * np.sqrt(1.0 + 0.4 * z)[:, None, None]
            * (omega[None, None, :] * 1e-15) ** 2
        )

    representation = build(operator, 0.0, 1.0)
    assert representation.value_shape == (4, 2049)
    assert representation.coefficients.dtype == np.complex128

    for index in ((0, 0), (2, 1024), (3, 2048)):
        column = _column(representation, index)
        # Clenshaw is a chain of elementwise operations, so the vectorised pass and a
        # scalar pass do the same operations on every element; a compiled loop may
        # still contract or order them differently, so they agree to the bound of two
        # Clenshaw passes over the column's coefficients.
        bound = np.max(clenshaw_bound(column.coefficients, axis=1))
        assert abs(representation(0.37)[index] - column(0.37)) <= bound

    # Every column is the same function up to a real scale factor, so each resolves at
    # the same degree on the same interval as the joint build (the resolution test is
    # far from its threshold here, so the breakpoints are exact); the transform runs
    # along the leading axis independently of the trailing ones, and a batched
    # transform may round differently from a single one: one transform of the samples,
    # 8 eps ceil(log2 n) per transform (the CPU constant of the dispersive bound),
    # normwise.
    for index in ((0, 0), (2, 1024)):
        alone = build(lambda z, i=index: operator(z)[:, i[0], i[1]], 0.0, 1.0)
        np.testing.assert_array_equal(alone.breakpoints, representation.breakpoints)
        joint = representation.coefficients[(0, slice(None), *index)]
        transform = 8 * EPS * math.ceil(math.log2(joint.size + 1))
        assert_within(alone.coefficients[0], joint, transform, what=str(index))


def test_a_complex_valued_build_integrates_exactly():
    def oscillation(z):
        return np.exp(3.0j * z)[:, None] * np.array([1.0, 2.0j])[None, :]

    phi = build(oscillation, 0.4, 2.1).integral()

    assert phi.coefficients.dtype == np.complex128
    for z in (0.6, 1.3, 2.1):
        analytic = (np.exp(3.0j * z) - np.exp(3.0j * 0.4)) / 3.0j
        np.testing.assert_allclose(
            phi(z),
            analytic * np.array([1.0, 2.0j]),
            rtol=0.0,
            atol=ROUND_OFF_ATOL,
        )


def test_the_tail_is_measured_per_element_so_a_small_column_is_not_swamped():
    def mixed(z):
        big = 1e6 * np.cos(0.5 * z)
        small = 1e-6 * np.cos(40.0 * z)
        return np.stack([big, small], axis=-1)

    representation = build(mixed, 0.0, 1.0)

    probe = np.linspace(0.0, 1.0, 51)
    values = np.array([representation(float(z))[1] for z in probe])
    # Without a per-element comparison scale the small column would be swallowed by the
    # large one and never resolved at all.
    np.testing.assert_allclose(
        values, 1e-6 * np.cos(40.0 * probe), rtol=0.0, atol=1e-6 * 1e-12
    )


def test_an_identically_zero_function_resolves_immediately():
    representation = build(np.zeros_like, 0.0, 1.0)

    assert representation.coefficients.shape == (1, 1)
    assert float(representation(0.5)) == 0.0


def test_intervals_of_different_degree_are_zero_padded_to_one_rectangle():
    def mixed(z):
        return np.where(z < 0.5, 1.0 + z, np.cos(30.0 * z))

    representation = build(mixed, 0.0, 1.0, mandatory_breakpoints=(0.5,))

    assert representation.degrees[0] < representation.degrees[1]
    assert representation.coefficients.shape[1] == int(representation.degrees.max()) + 1
    padded = representation.coefficients[0, int(representation.degrees[0]) + 1 :]
    np.testing.assert_array_equal(padded, 0.0)
    assert float(representation(0.25)) == pytest.approx(1.25, abs=ROUND_OFF_ATOL)


# --------------------------------------------------------------------- accessors


@pytest.fixture
def kinked():
    return build(smooth, 0.0, 2.0, mandatory_breakpoints=(0.7, 1.3))


def test_interval_containing_puts_a_breakpoint_in_the_interval_starting_there(kinked):
    assert kinked.interval_containing(0.7) == 1
    assert kinked.interval_containing(0.0) == 0
    assert kinked.interval_containing(2.0) == kinked.degrees.size - 1


def test_interval_containing_tolerates_a_few_ulp_outside_the_span(kinked):
    assert kinked.interval_containing(2.0 + 1e-12) == kinked.degrees.size - 1
    assert kinked.interval_containing(-1e-12) == 0


@pytest.mark.parametrize("z", [2.5, -0.5])
def test_interval_containing_rejects_a_position_outside_the_span(kinked, z):
    with pytest.raises(TranslationError, match="outside the represented span"):
        kinked.interval_containing(z)


def test_evaluating_outside_the_span_raises(kinked):
    with pytest.raises(TranslationError, match="outside the represented span"):
        kinked(3.0)


def test_hard_kinks_are_kept_apart_from_the_splits_own_edges(kinked):
    np.testing.assert_array_equal(kinked.hard_kinks, [0.7, 1.3])
    assert kinked.breakpoints.size - 1 > kinked.hard_kinks.size


def test_next_hard_kink_clears_the_kink_it_starts_on(kinked):
    assert kinked.next_hard_kink(0.0) == 0.7
    assert kinked.next_hard_kink(0.7) == 1.3
    assert kinked.next_hard_kink(1.4) is None


def test_value_shape_and_span(kinked):
    assert kinked.value_shape == ()
    assert kinked.z_span == (0.0, 2.0)


# --------------------------------------------------------------------- from paths


def test_build_from_paths_splits_on_the_paths_and_resolves_the_operator():
    omega = np.linspace(-3.0, 3.0, 129)

    def operator(_z, density):
        return (
            -1.0j * (1.0 + 0.4 * density[:, None]) ** 1.5 * omega[None, :] ** 2
            - 0.05 * density[:, None]
        )

    representation = build_from_paths(
        [two_stage_pressure], operator, 0.0, 1.0, mandatory_breakpoints=(0.6,)
    )

    np.testing.assert_array_equal(representation.hard_kinks, [0.6])
    assert 0.6 in representation.breakpoints.tolist()
    assert representation.value_shape == (129,)
    for z in (0.05, 0.4, 0.85):
        np.testing.assert_allclose(
            representation(z),
            operator(np.asarray([z]), np.asarray([two_stage_pressure(z)]))[0],
            rtol=0.0,
            atol=1e-11,
        )


def test_build_from_paths_with_no_paths_is_a_plain_build():
    representation = build_from_paths([], np.cos, 0.0, 1.0)

    plain = build(np.cos, 0.0, 1.0)
    np.testing.assert_array_equal(representation.coefficients, plain.coefficients)
    np.testing.assert_array_equal(representation.breakpoints, plain.breakpoints)


def test_build_from_paths_resolves_a_path_kink_the_operator_hides():
    # The operator is a fourth power of the path, so resolving the path is a proxy for
    # resolving the operator rather than a bound on it.
    def path(z):
        return np.abs(z - 0.375)

    representation = build_from_paths([path], lambda _z, p: p**4, 0.0, 1.0)

    distances = np.abs(representation.breakpoints - 0.375)
    assert float(np.min(distances)) < 1e-4
    probe = np.linspace(0.0, 1.0, 101)
    values = np.array([float(representation(float(z))) for z in probe])
    np.testing.assert_allclose(values, path(probe) ** 4, rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------- error paths


def test_build_rejects_an_empty_or_reversed_domain():
    with pytest.raises(TranslationError, match="need z1 > z0"):
        build(smooth, 1.0, 1.0)


def test_build_rejects_a_non_positive_tolerance():
    with pytest.raises(TranslationError, match="tol must be positive"):
        build(smooth, 0.0, 1.0, tol=0.0)


def test_build_rejects_a_degree_too_low_for_the_tail_test():
    with pytest.raises(TranslationError, match="n_max must be at least"):
        build(smooth, 0.0, 1.0, n_max=2)


def test_build_rejects_an_empty_interval_budget():
    with pytest.raises(TranslationError, match="max_intervals must be at least 1"):
        build(smooth, 0.0, 1.0, max_intervals=0)


def test_build_rejects_a_width_floor_below_the_representable_one():
    with pytest.raises(TranslationError, match="min_width must be at least"):
        build(smooth, 0.0, 1.0, min_width=1e-300)


def test_build_rejects_a_mandatory_breakpoint_outside_the_domain():
    with pytest.raises(TranslationError, match="must lie inside"):
        build(smooth, 0.0, 1.0, mandatory_breakpoints=(1.5,))


def test_build_rejects_a_non_finite_mandatory_breakpoint():
    with pytest.raises(TranslationError, match="must be finite"):
        build(smooth, 0.0, 1.0, mandatory_breakpoints=(np.nan,))


def test_build_rejects_a_sampler_that_returns_the_wrong_length():
    with pytest.raises(TranslationError, match="one value per position"):
        build(lambda _z: np.zeros(3), 0.0, 1.0)


def test_build_rejects_a_sampler_that_returns_an_empty_value_shape():
    with pytest.raises(TranslationError, match="empty value shape"):
        build(lambda z: np.zeros((z.size, 0)), 0.0, 1.0)


def test_build_rejects_a_non_finite_sample():
    def diverges(z):
        return np.where(z > 0.9, np.inf, z)

    with pytest.raises(TranslationError, match="non-finite value at z"):
        build(diverges, 0.0, 1.0)


def test_the_representation_rejects_an_empty_interval_set():
    with pytest.raises(TranslationError, match="at least one interval"):
        PiecewiseChebyshev(
            np.zeros((0, 2)),
            np.zeros(1),
            np.zeros(0, dtype=np.intp),
            np.zeros(0),
        )


def test_the_representation_rejects_non_finite_breakpoints():
    with pytest.raises(TranslationError, match="must be finite"):
        PiecewiseChebyshev(
            np.zeros((1, 2)),
            np.asarray([0.0, np.inf]),
            np.zeros(1, dtype=np.intp),
            np.zeros(0),
        )


def test_the_representation_rejects_breakpoints_that_do_not_increase():
    with pytest.raises(TranslationError, match="strictly increasing"):
        PiecewiseChebyshev(
            np.zeros((2, 2)),
            np.asarray([0.0, 0.5, 0.5]),
            np.zeros(2, dtype=np.intp),
            np.zeros(0),
        )


def test_the_representation_rejects_a_single_precision_dtype():
    with pytest.raises(TranslationError, match="float64 or complex128"):
        PiecewiseChebyshev(
            np.zeros((1, 2), dtype=np.float32),
            np.asarray([0.0, 1.0]),
            np.zeros(1, dtype=np.intp),
            np.zeros(0),
        )


def test_the_representation_rejects_a_hard_kink_that_is_not_a_breakpoint():
    with pytest.raises(TranslationError, match="not an interior breakpoint"):
        PiecewiseChebyshev(
            np.zeros((2, 2)),
            np.asarray([0.0, 0.5, 1.0]),
            np.zeros(2, dtype=np.intp),
            np.asarray([0.3]),
        )


def test_the_representation_rejects_a_hard_kink_when_there_is_no_interior_edge():
    with pytest.raises(TranslationError, match="not an interior breakpoint"):
        PiecewiseChebyshev(
            np.zeros((1, 2)),
            np.asarray([0.0, 1.0]),
            np.zeros(1, dtype=np.intp),
            np.asarray([0.5]),
        )


# --------------------------------------------------------------------- budget and logs


def test_exceeding_the_interval_budget_sets_the_flag_and_warns(caplog):
    def chatter(z):
        return np.sign(np.sin(400.0 * z))

    with caplog.at_level(logging.WARNING, logger="solprop.translate.chebyshev"):
        representation = build(chatter, 0.0, 1.0, max_intervals=16)

    assert representation.budget_exceeded
    assert representation.degrees.size <= 16
    assert "exceeded its budget" in caplog.text
    # The flag survives into anything derived from the representation, because the
    # caller decides on it after integrating.
    assert representation.integral().budget_exceeded
    assert representation.derivative().budget_exceeded


def test_mandatory_breakpoints_outrank_the_interval_budget():
    representation = build(
        smooth, 0.0, 1.0, mandatory_breakpoints=(0.25, 0.5, 0.75), max_intervals=1
    )

    # A declared edge is a statement about the operator, not a preference: honouring it
    # can never be traded against the budget.
    np.testing.assert_array_equal(
        representation.breakpoints, [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    assert not representation.budget_exceeded


def test_a_resolved_build_does_not_set_the_flag():
    assert not build(smooth, 0.0, 1.0).budget_exceeded


def test_the_build_logs_its_interval_count_and_degrees_at_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="solprop.translate.chebyshev"):
        build(np.sqrt, 0.0, 1.0, min_width=1e-3)

    assert "piecewise Chebyshev on" in caplog.text
    assert "intervals, degrees" in caplog.text


# --------------------------------------------------------------------- the error bound


@st.composite
def sums_of_gaussians(draw):
    """Random smooth functions: 1 to 6 Gaussians, wide enough to be resolvable."""
    count = draw(st.integers(min_value=1, max_value=6))
    amplitudes = draw(
        st.lists(
            st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
            min_size=count,
            max_size=count,
        )
    )
    centres = draw(
        st.lists(
            st.floats(-0.2, 1.2, allow_nan=False, allow_infinity=False),
            min_size=count,
            max_size=count,
        )
    )
    widths = draw(
        st.lists(
            st.floats(0.03, 0.5, allow_nan=False, allow_infinity=False),
            min_size=count,
            max_size=count,
        )
    )
    return np.asarray(amplitudes), np.asarray(centres), np.asarray(widths)


@pytest.mark.medium
@given(parameters=sums_of_gaussians())
@settings(
    derandomize=True,
    deadline=None,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_the_build_meets_its_uniform_error_bound(parameters):
    amplitudes, centres, widths = parameters
    tol = 1e-10

    def gaussians(z):
        offsets = (np.asarray(z)[..., None] - centres) / widths
        return np.sum(amplitudes * np.exp(-(offsets**2)), axis=-1)

    representation = build(gaussians, 0.0, 1.0, tol=tol)

    probe = np.linspace(0.0, 1.0, 2001)
    exact = gaussians(probe)
    scale = float(np.max(np.abs(exact)))
    if scale == 0.0:
        return
    values = np.array([float(representation(float(z))) for z in probe])
    error = float(np.max(np.abs(values - exact)))
    # The acceptance test bounds the neglected coefficients by tol/2 relative to the
    # value scale, and Chebyshev aliasing at most doubles that in the interpolant
    # (Trefethen, ATAP theorem 8.2), so tol bounds the uniform relative error itself.
    assert 2.0 * error <= tol * scale


@settings(max_examples=60, deadline=None)
@given(
    degree=st.integers(0, 66),
    width=st.sampled_from([(), (1,), (3,), (2, 9)]),
    complex_values=st.booleans(),
    x=st.floats(-1.0, 1.0),
    exponent=st.integers(-20, 20),
)
def test_evaluation_is_numpys_chebval(degree, width, complex_values, x, exponent):
    """The compiled Clenshaw pass gives NumPy's values, in their shape.

    The two are independent Clenshaw passes, held to :func:`clenshaw_bound` per
    element; the shape, the type and the read-only flag
    are structural and exact.
    """
    rng = np.random.default_rng(degree * 97 + len(width))
    shape = (1, degree + 1, *width)
    coefficients = rng.standard_normal(shape) * 10.0**exponent
    if complex_values:
        coefficients = coefficients + 1j * rng.standard_normal(shape)
    table = PiecewiseChebyshev(
        coefficients, np.array([0.0, 2.0]), np.array([degree]), np.empty(0)
    )
    z = 1.0 + x
    expected = chebyshev_basis.chebval((2.0 * z - 0.0 - 2.0) / 2.0, coefficients[0])
    value = table(z)
    assert value.shape == np.shape(expected)
    assert value.dtype == coefficients.dtype
    bound = np.broadcast_to(clenshaw_bound(coefficients[0]), np.shape(expected))
    error = np.abs(np.asarray(value) - np.asarray(expected))
    assert np.all(error <= bound)
    assert not value.flags.writeable


def _one_at_a_time(table, positions):
    """Return ``table(z)`` at each position, each formed afresh rather than kept."""
    values = []
    for z in positions:
        table.__dict__["_recent"].clear()
        values.append(np.array(table(float(z))))
    table.__dict__["_recent"].clear()
    return np.stack(values)


@settings(max_examples=40, deadline=None)
@given(
    degree=st.integers(0, 66),
    width=st.sampled_from([(), (1,), (3,), (2, 9)]),
    complex_values=st.booleans(),
    offset=st.sampled_from([0.0, 1.0e3, 4.2, -7.5e5]),
)
def test_positions_taken_together_are_the_values_taken_alone(
    degree, width, complex_values, offset
):
    """A batch of positions is one call per position, bit for bit (SEP 25 phase 2).

    Three intervals, so the batch picks a different coefficient block per point;
    positions on every breakpoint (an interior one belongs to the interval starting
    there), an ulp either side of each, within the edge slack outside both ends, a
    pair a relative 1e-12 apart, and anchors far from the origin, where the map onto
    ``[-1, 1]`` cancels most of the position's digits.
    """
    rng = np.random.default_rng(degree * 31 + len(width))
    shape = (3, degree + 1, *width)
    coefficients = rng.standard_normal(shape)
    if complex_values:
        coefficients = coefficients + 1j * rng.standard_normal(shape)
    edges = offset + np.array([0.0, 0.25, 0.3, 1.0])
    table = PiecewiseChebyshev(
        coefficients, edges, np.full(3, degree), np.array([edges[1]])
    )
    slack = 0.5 * chebyshev._EDGE_SLACK_FRACTION * (edges[-1] - edges[0])
    inner = rng.uniform(edges[0], edges[-1], 7)
    positions = np.concatenate(
        [
            edges,
            np.nextafter(edges[1:], -np.inf),
            np.nextafter(edges[:-1], np.inf),
            [edges[0] - slack, edges[-1] + slack],
            inner,
            inner * (1.0 + 1e-12),
        ]
    )
    expected = _one_at_a_time(table, positions)
    values = table.at_positions(positions)
    assert values.shape == (positions.size, *width)
    assert values.dtype == coefficients.dtype
    assert not values.flags.writeable
    assert np.array_equal(values.view(np.uint8), expected.view(np.uint8))


def test_positions_taken_together_are_kept_for_the_next_call():
    """The last positions of a batch are kept, so the next anchor is not re-read."""
    table = build(smooth, 0.0, 1.0)
    positions = np.linspace(0.0, 1.0, 11)
    values = table.at_positions(positions)
    kept = table(float(positions[-1]))
    assert np.shares_memory(kept, values)


def test_positions_taken_together_refuse_the_outside_of_the_span():
    """The batch is as strict as one position: the first offender is named."""
    table = build(smooth, 0.0, 1.0)
    with pytest.raises(TranslationError, match=r"z = 1\.5 is outside"):
        table.at_positions(np.array([0.5, 1.5, -1.0]))


@pytest.mark.parametrize("compiled", [False, True])
def test_the_batch_kernel_stops_at_the_first_offender_and_files_nan_last(compiled):
    """The plain loop and its compiled instance answer alike, position by position.

    A position outside the widened span stops the loop and is returned, its row and
    those after it unwritten; a NaN position passes the span test, takes the last
    interval as ``searchsorted`` files it, and gives a NaN row, as one call does.
    """
    table = build(smooth, 0.0, 1.0, mandatory_breakpoints=(0.5,))
    flat = table.coefficients.reshape(*table.coefficients.shape[:2], -1)
    kernel = chebyshev._clenshaw_batch() if compiled else chebyshev._clenshaw_points
    positions = np.array([0.25, math.nan, 1.5, 0.75])
    out = np.full((positions.size, flat.shape[2]), 7.0)
    assert kernel(positions, table.breakpoints, 0.0, 1.0, flat, out) == 2
    assert np.array_equal(out[:1], _one_at_a_time(table, positions[:1]).reshape(1, -1))
    assert np.isnan(out[1]).all() and np.isnan(table(math.nan)).all()
    assert (out[2:] == 7.0).all()


@pytest.mark.parametrize("compiled", [False, True], ids=["plain", "compiled"])
def test_a_degree_one_series_is_one_product_and_one_sum(compiled):
    # The recursion's two-term case, `c_0 + c_1 x`, on real and complex columns:
    # exact here, since every product and sum is exact in binary at x = 0.5.
    columns = chebyshev._clenshaw() if compiled else chebyshev._clenshaw_columns
    for coefficients in (
        np.array([[1.0, -2.0], [2.0, 0.25]]),
        np.array([[1.0 - 0.5j, 3.0j], [2.0 + 4.0j, -1.0]]),
    ):
        out = np.empty(2, dtype=coefficients.dtype)
        columns(0.5, coefficients, out)
        np.testing.assert_array_equal(out, coefficients[0] + 0.5 * coefficients[1])


@pytest.mark.parametrize("shape", [(), (18,), (3, 2)])
@pytest.mark.parametrize("complex_valued", [False, True])
def test_clenshaw_before_numba_loads_gives_the_compiled_values(
    shape, complex_valued, monkeypatch
):
    # Until numba is imported, clenshaw evaluates with NumPy's chebval (a run on the
    # native backend never imports numba); the values must be the compiled ones.
    pytest.importorskip("numba")
    rng = np.random.default_rng(7)
    coefficients = rng.standard_normal((33, *shape))
    if complex_valued:
        coefficients = coefficients + 1j * rng.standard_normal(coefficients.shape)
    points = rng.uniform(-1.0, 1.0, 16)
    compiled = [chebyshev.clenshaw(x, coefficients) for x in points]
    monkeypatch.delitem(sys.modules, "numba")
    uncompiled = [chebyshev.clenshaw(x, coefficients) for x in points]
    for one, other in zip(compiled, uncompiled, strict=True):
        assert one.shape == other.shape == shape
        assert one.dtype == other.dtype
        np.testing.assert_array_equal(one, other)
