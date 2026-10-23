# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the Runge-Kutta tableaux and their order conditions.

The two DOPRI5 traps of ``docs/dev/stepper.md`` section 10 are tested directly: the
propagated solution must be the 5th-order one (Luna's ``dopri.jl`` had the pair labelled
the wrong way round until PR #436), and the FSAL row must equal the solution weights.
"""

import itertools

import numpy as np
import pytest

from convergence_records import Column, Study, record
from solprop.stepper.tableau import (
    DOPRI5,
    TSIT5,
    Tableau,
    check_order_conditions,
)

TABLEAUX = [DOPRI5, TSIT5]
TABLEAU_IDS = [tableau.name for tableau in TABLEAUX]

# The required bound. It is above the worst residual either tableau produces
# (8.0e-16, in Tsit5's order-5 trees, whose coefficients are published decimals rather
# than exact rationals) and eleven orders below the residual of a condition that is
# genuinely not met (8.7e-4 for both embedded companions at order 5).
ORDER_CONDITION_TOLERANCE = 1e-15


def rk_step(tableau, weights, rhs, z, y, dz):
    """Advance one explicit Runge-Kutta step with an arbitrary weight vector."""
    stages = np.zeros(tableau.n_stages)
    for index in range(tableau.n_stages):
        increment = float(tableau.matrix[index, :index] @ stages[:index])
        stages[index] = rhs(z + tableau.nodes[index] * dz, y + dz * increment)
    return y + dz * float(weights @ stages)


def integrate_decay(tableau, weights, n_steps):
    """Integrate ``y' = -y`` over [0, 1] uniformly; return the error at the end."""
    dz = 1.0 / n_steps
    y = 1.0
    for step in range(n_steps):
        y = rk_step(tableau, weights, lambda _z, value: -value, step * dz, y, dz)
    return abs(y - np.exp(-1.0))


def ladder(tableau, weights, counts):
    """Return the errors of a fixed-step ladder and the orders between them."""
    errors = [integrate_decay(tableau, weights, count) for count in counts]
    # The round-off floor of this integration is about n * eps * |y|; 1e-13 keeps
    # every level of the ladders used here an order of magnitude clear of it.
    assert min(errors) > 1e-14, "ladder must stay above the round-off floor"
    orders = [
        np.log(coarse / fine) / np.log(2.0)
        for coarse, fine in itertools.pairwise(errors)
    ]
    return errors, orders


def observed_orders(tableau, weights, counts):
    """Return the observed order between each consecutive pair of step counts."""
    return ladder(tableau, weights, counts)[1]


# The record of the ladder below, drawn as the first figure of `docs/dev/accuracy.md`.
# Both tolerances are set by the finest rung, which is a *difference* of two numbers
# near exp(-1): Tsit5's solution weights reach 1.6e-14 there, about sixty ulps of 0.37,
# so a seven-element dot product summed in another order -- a different BLAS on another
# platform -- moves it by a few per cent while leaving the ladder's claim untouched.
# `atol` carries that floor and `rtol` stays tight above it, which is the whole ladder
# except its last rung. The order tolerance is what a six per cent move in one rung
# costs a log2 ratio; the test's own claim, that the deviations fall monotonically and
# the finest is within 0.15 of the stated order, is far tighter than either.
TABLEAU_ORDER = Study(
    name="tableau_order",
    title="Global order of the DOPRI5 and Tsit5 weights on a scalar decay",
    case={
        "problem": "y' = -y on [0, 1], y(0) = 1",
        "reference": "exp(-1), the exact solution",
        "schedule": "uniform fixed steps",
    },
    axis={"name": "n_steps", "unit": "steps", "label": "uniform steps over [0, 1]"},
    columns=(
        Column(
            name="error",
            unit="1 (absolute, on a solution of order unity)",
            why="atol is about four ulps of exp(-1), the round-off floor the finest "
            "rung sits on; rtol is tight above it, where the ladder is truncation",
            rtol=1e-6,
            atol=1e-15,
        ),
        Column(
            name="order",
            unit="1",
            why="what a six per cent move in one rung costs a log2 ratio, that being "
            "the floor tolerance above; null on the coarsest rung, which has no pair",
            rtol=0.0,
            atol=0.2,
        ),
    ),
)


@pytest.mark.parametrize("tableau", TABLEAUX, ids=TABLEAU_IDS)
def test_order_conditions_hold_to_the_stated_tolerance(tableau):
    report = check_order_conditions(tableau, tolerance=ORDER_CONDITION_TOLERANCE)
    assert report.max_residual < ORDER_CONDITION_TOLERANCE
    assert report.matches_declared_orders


@pytest.mark.parametrize("tableau", TABLEAUX, ids=TABLEAU_IDS)
def test_the_solution_weights_are_fifth_order_and_the_companion_fourth(tableau):
    report = check_order_conditions(tableau)
    assert (report.solution_order, report.embedded_order) == (5, 4)
    # The separation must be gross, not marginal: the order-5 conditions the companion
    # fails do so by ~1e-4, so no plausible tolerance could confuse the two vectors.
    worst_fifth = max(
        report.embedded_residuals[label]
        for label in ("b.c4", "b.Ac3", "b.AAAc", "b.(Ac)2")
    )
    assert worst_fifth > 1e-5


@pytest.mark.parametrize("tableau", TABLEAUX, ids=TABLEAU_IDS)
def test_the_propagated_solution_is_the_fifth_order_one(tableau):
    # Luna trap: with b and b_hat swapped the propagated solution is 4th order and
    # nothing else in the tableau notices. A *global* ladder is required -- a single
    # step measures the local error and reads one order too high.
    #
    # The band is half an order. At these step sizes the observed order still carries
    # the next term of the expansion and sits above its asymptote (measured +0.37 at
    # the coarsest level for Tsit5, falling monotonically; see the slow test). Half an
    # order is far wider than that drift and far narrower than the whole order that
    # separates the two weight vectors, which is what this test has to discriminate.
    for order in observed_orders(tableau, tableau.solution_weights, (8, 16, 32)):
        assert 4.5 < order < 5.5
    for order in observed_orders(tableau, tableau.embedded_weights, (8, 16, 32)):
        assert 3.5 < order < 4.5


@pytest.mark.parametrize("tableau", TABLEAUX, ids=TABLEAU_IDS)
def test_the_observed_order_converges_onto_the_stated_order(tableau, request):
    """The ladder of `docs/dev/accuracy.md` section 1, and the record it is drawn from.

    Four levels, so three observed orders: each must be closer to the stated order
    than the last, and the finest must be within 0.15 of it. 64 steps is as fine as
    this ladder goes -- below 1e-14 the round-off floor of ~n*eps*|y| takes over and
    the ratio stops measuring truncation at all. Measured, DOPRI5's solution weights:
    3.84e-9, 1.08e-10, 3.21e-12, 9.79e-14, so 5.149, 5.076 and 5.036.
    """
    counts = (8, 16, 32, 64)
    rows = []
    for kind, weights, stated in (
        ("solution", tableau.solution_weights, 5.0),
        ("embedded", tableau.embedded_weights, 4.0),
    ):
        errors, orders = ladder(tableau, weights, counts)
        deviations = [abs(order - stated) for order in orders]
        assert deviations == sorted(deviations, reverse=True)
        assert deviations[-1] < 0.15
        rows += [
            {
                "series": f"{tableau.name} {kind}",
                "axis": count,
                "error": float(error),
                # An order belongs to a pair, so the coarsest rung has none.
                "order": None if index == 0 else float(orders[index - 1]),
            }
            for index, (count, error) in enumerate(zip(counts, errors, strict=True))
        ]
    record(TABLEAU_ORDER, request=request, rows=rows)


@pytest.mark.parametrize("tableau", TABLEAUX, ids=TABLEAU_IDS)
def test_the_fsal_row_equals_the_solution_weights(tableau):
    # Bitwise, not approximate: the coefficients are written out twice, independently,
    # so any drift between them is a typing error rather than a rounding difference.
    assert tableau.first_same_as_last
    assert np.array_equal(tableau.matrix[-1, :-1], tableau.solution_weights[:-1])
    assert tableau.solution_weights[-1] == 0.0


def test_dopri5_matches_the_published_coefficients():
    assert DOPRI5.nodes[1] == 1 / 5
    assert DOPRI5.matrix[4, 0] == 19372 / 6561
    assert DOPRI5.solution_weights[0] == 35 / 384
    # b_hat is recovered by subtraction, which round-trips the decision to store the
    # difference; one subtraction of nearly equal doubles costs at most a few ulp.
    assert np.allclose(
        DOPRI5.embedded_weights,
        [
            5179 / 57600,
            0.0,
            7571 / 16695,
            393 / 640,
            -92097 / 339200,
            187 / 2100,
            1 / 40,
        ],
        rtol=0.0,
        atol=4 * np.finfo(np.float64).eps,
    )


def test_tsit5_matches_the_published_coefficients():
    assert TSIT5.nodes[1] == 0.161
    assert TSIT5.nodes[4] == 0.9800255409045097
    assert TSIT5.matrix[2, 0] == -0.008480655492356989
    assert TSIT5.solution_weights[1] == 0.01
    # The one error weight with an exact rational form; the rest are published decimals.
    assert TSIT5.error_weights[6] == 1 / 66
    # Luna's tsit.jl stores the error vector where the weights belong. Its sum is the
    # giveaway: a weight vector sums to 1, an error vector to 0.
    assert abs(TSIT5.solution_weights.sum() - 1.0) < 1e-15
    assert abs(TSIT5.error_weights.sum()) < 1e-15


def make_tableau(**overrides):
    """Build Heun's 2(1) pair, a valid non-FSAL tableau, with fields overridden."""
    fields = {
        "name": "Heun21",
        "nodes": np.array([0.0, 1.0]),
        "matrix": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "solution_weights": np.array([1 / 2, 1 / 2]),
        "error_weights": np.array([-1 / 2, 1 / 2]),
        "order": 2,
        "embedded_order": 1,
        "first_same_as_last": False,
    }
    return Tableau(**(fields | overrides))


def test_a_non_fsal_pair_is_accepted_and_measured():
    heun = make_tableau()
    assert heun.n_stages == 2
    assert not heun.first_same_as_last
    np.testing.assert_array_equal(heun.embedded_weights, [1.0, 0.0])
    report = check_order_conditions(heun)
    assert (report.solution_order, report.embedded_order) == (2, 1)
    assert report.matches_declared_orders


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"matrix": np.zeros((3, 3))}, "matrix inconsistent"),
        ({"solution_weights": np.zeros(3)}, "solution_weights inconsistent"),
        ({"error_weights": np.zeros(3)}, "error_weights inconsistent"),
        ({"nodes": np.array([0.0, np.nan])}, "nodes has a non-finite entry"),
        ({"nodes": np.array([0.5, 1.0])}, "nodes\\[0\\] == 0"),
        (
            {"matrix": np.array([[0.0, 0.0], [0.5, 0.5]])},
            "not strictly lower triangular",
        ),
        (
            {"matrix": np.array([[0.0, 0.0], [0.9, 0.0]])},
            "rows do not sum to the nodes",
        ),
        ({"solution_weights": np.array([0.5, 0.6])}, "sum to 1 \\+"),
        ({"error_weights": np.array([0.5, 0.5])}, "error weights sum to"),
        # Passes the sum rule; the estimate would be zero on every step (SEP 24).
        ({"error_weights": np.zeros(2)}, "every error weight is zero"),
        ({"embedded_order": 2}, "0 < embedded_order < order"),
        ({"first_same_as_last": True}, "first_same_as_last requires"),
    ],
)
def test_tableau_validation_rejects_a_broken_definition(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_tableau(**overrides)


def test_fsal_is_rejected_when_the_last_weight_is_not_zero():
    # The last row does equal the weights here, so only the trailing-zero rule can
    # catch it; without that rule the reused stage would be counted twice.
    with pytest.raises(ValueError, match="first_same_as_last requires"):
        make_tableau(
            nodes=np.array([0.0, 0.5]),
            matrix=np.array([[0.0, 0.0], [0.5, 0.0]]),
            first_same_as_last=True,
        )


def test_the_arrays_of_a_tableau_cannot_be_written_through():
    heun = make_tableau()
    with pytest.raises(ValueError, match="read-only"):
        heun.nodes[0] = 1.0


def test_a_tableau_does_not_adopt_the_caller_s_arrays():
    nodes = np.array([0.0, 1.0])
    heun = make_tableau(nodes=nodes)
    nodes[1] = 99.0
    assert heun.nodes[1] == 1.0


def test_an_order_condition_missed_by_a_hair_lowers_the_reported_order():
    # 5e-15 passes the 1e-14 structural check but fails the 1e-15 order-1 condition,
    # so the report must say order 0 rather than trusting the declaration.
    barely_wrong = make_tableau(solution_weights=np.array([1 / 2, 1 / 2 + 5e-15]))
    report = check_order_conditions(barely_wrong)
    assert report.solution_order == 0
    assert not report.matches_declared_orders
