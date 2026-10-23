# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The variable-step Adams algebra of SEP 13 section 4, against exact moments."""

import inspect
import math

import numpy as np
import pytest

from bounds import gamma
from solprop.exceptions import StepperError
from solprop.stepper.multistep import (
    ComparisonMultistep,
    Multistep,
    _basis,
    _basis_rows,
    defect_integral,
    integrated_weights,
    lagrange_coefficients,
    milne_factor,
)


@pytest.mark.parametrize(
    ("nodes", "expected"),
    [
        # Constant-step Adams-Bashforth 1-4 and Adams-Moulton 2-4, Hairer, Noersett
        # and Wanner I, tables III.1.1 and III.1.2.
        ([0], [1]),
        ([0, -1], [3 / 2, -1 / 2]),
        ([0, -1, -2], [23 / 12, -16 / 12, 5 / 12]),
        ([0, -1, -2, -3], [55 / 24, -59 / 24, 37 / 24, -9 / 24]),
        ([1, 0], [1 / 2, 1 / 2]),
        ([1, 0, -1], [5 / 12, 8 / 12, -1 / 12]),
        ([1, 0, -1, -2], [9 / 24, 19 / 24, -5 / 24, 1 / 24]),
    ],
)
def test_integrated_weights_constant_step_adams(nodes, expected):
    # Each weight is a few polynomial evaluations of order one: round-off only.
    np.testing.assert_allclose(integrated_weights(nodes), expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("count", [1, 3, 5, 7])
def test_integrated_weights_exact_on_irregular_nodes(seed, count):
    rng = np.random.default_rng(seed)
    # A history behind the anchor with step ratios between 1/2 and 2, plus the anchor.
    steps = rng.uniform(0.5, 2.0, count - 1)
    nodes = np.concatenate(([0.0], -np.cumsum(steps)))
    upper = rng.uniform(0.2, 1.0)
    _assert_scaled_moments(integrated_weights(nodes, upper), nodes, upper)


def _assert_scaled_moments(weights, nodes, upper):
    # Monomial scaling leaves the polynomial exactness invariant unchanged, while
    # avoiding the million-sized degree-six terms of the unscaled history. fsum
    # removes the platform-specific BLAS reduction from this algebraic witness.
    count = len(nodes)
    scale = max(1.0, float(np.max(np.abs(nodes))))
    for degree in range(count):
        terms = weights * (nodes / scale) ** degree
        exact = upper * (upper / scale) ** degree / (degree + 1)
        # All roots are nonpositive and upper is positive: the cardinal
        # polynomials have nonnegative coefficients before their signed common
        # denominator. Their O(count**2) construction/evaluation therefore has no
        # internal cancellation. Include powers, products and the final moment
        # reduction in a conservative eight-operation-per-coefficient budget.
        operations = 8 * count**2
        epsilon = np.finfo(float).eps
        gamma = operations * epsilon / (1.0 - operations * epsilon)
        bound = gamma * max(math.fsum(np.abs(terms)), abs(exact))
        assert math.fsum(terms) == pytest.approx(exact, rel=0, abs=bound)


@pytest.mark.parametrize("defect", ["coefficient", "order"])
def test_irregular_moment_witness_rejects_wrong_weights(defect):
    nodes = np.array([0.0, -0.5, -1.25, -2.5, -3.0, -4.5, -5.25])
    upper = 0.75
    if defect == "coefficient":
        weights = integrated_weights(nodes, upper)
        weights[0] *= 1.0 + 1e-6
    else:
        # One missing history value still integrates every lower degree exactly.
        weights = np.append(integrated_weights(nodes[:-1], upper), 0.0)
    with pytest.raises(AssertionError):
        _assert_scaled_moments(weights, nodes, upper)


def test_defect_constants_and_milne_factor():
    # 251/720 and -19/720 times 4!, and the corrector's defect -19/270 of v_C - v_P.
    assert defect_integral([0, -1, -2, -3]) / math.factorial(4) == pytest.approx(
        251 / 720, rel=1e-13
    )
    assert defect_integral([1, 0, -1, -2]) / math.factorial(4) == pytest.approx(
        -19 / 720, rel=1e-13
    )
    assert milne_factor([0, -1, -2, -3], [1, 0, -1, -2]) == pytest.approx(
        -19 / 270, rel=1e-13
    )


def test_fifth_difference_is_the_ab4_defect():
    # The quartic and cubic predictors differ by 251/720 times the fourth backward
    # difference (SEP 13 section 4.3).
    wider = integrated_weights([0, -1, -2, -3, -4])
    narrower = np.append(integrated_weights([0, -1, -2, -3]), 0.0)
    np.testing.assert_allclose(
        (wider - narrower) * 720 / 251, [1, -4, 6, -4, 1], rtol=0, atol=1e-12
    )


@pytest.mark.parametrize(
    ("method", "name"),
    [
        (ComparisonMultistep(corrector="none", order=4), "adams-ab4"),
        (ComparisonMultistep(corrector="pec", order=3), "adams-pec3"),
        (
            ComparisonMultistep(corrector="pec", order=1, corrector_order=2),
            "adams-pec1/2",
        ),
        (
            ComparisonMultistep(corrector="pece", order=5, corrector_order=5),
            "adams-pece5",
        ),
        (
            ComparisonMultistep(order=4, corrector_order=5, variable_order=True),
            "adams-pece4/5-variable",
        ),
    ],
)
def test_multistep_name(method, name):
    assert method.name == name


@pytest.mark.parametrize(
    "fields",
    [
        {"order": 0},
        {"order": 7},
        {"order": 3, "corrector_order": 5},
        {"max_growth": 1.0},
        {"min_factor": 1.0},
        {"safety": 0.0},
        {"order": 4, "variable_order": True},
        {"corrector": "none", "variable_order": True},
    ],
)
def test_multistep_refuses_out_of_range(fields):
    with pytest.raises(StepperError):
        ComparisonMultistep(**fields)


def test_algebra_refuses_degenerate_input():
    with pytest.raises(StepperError, match="distinct"):
        integrated_weights([0.0, 0.0])
    with pytest.raises(StepperError, match="equal node counts"):
        milne_factor([0, -1], [1, 0, -1])
    with pytest.raises(StepperError, match="coincide"):
        milne_factor([1, 0], [1, 0])
    with pytest.raises(StepperError, match="non-empty and finite"):
        integrated_weights([])
    with pytest.raises(StepperError, match="non-empty and finite"):
        integrated_weights([0.0, float("nan")])


@pytest.mark.parametrize("order", range(1, 6))
def test_one_order_higher_pec_predicts_with_the_next_adams_bashforth(order):
    # At constant step, PEC with a corrector one order higher advances its *predicted*
    # states by Adams-Bashforth of order + 1: the corrector over [n, n+1] minus the
    # predictor over [n, n+1] plus the next predictor over [n+1, n+2] is AB(q+1) over
    # [n+1, n+2]. On the derivatives at n+1, n, ..., n-q+1:
    behind = -np.arange(order, dtype=float)
    corrector = integrated_weights(np.concatenate(([1.0], behind)))
    previous = np.concatenate(([0.0], integrated_weights(behind)))
    following = np.append(integrated_weights(behind), 0.0)
    wider = integrated_weights(-np.arange(order + 1, dtype=float))
    # Weights of order one in magnitude, summed over at most seven terms: round-off.
    np.testing.assert_allclose(
        corrector - previous + following, wider, rtol=0, atol=1e-12
    )


def test_selected_method_defaults():
    # SEP 13's selection: one-call PEC, corrector one order higher, variable order to 5,
    # in SEP 16's exponential formulation.
    method = Multistep()
    assert method.name == "exponential-adams-pec5/6-variable"
    assert method.formulation == "exponential"
    assert method.corrector == "pec"
    assert method.corrector_nodes(3) == 4
    interaction = Multistep(order=3, variable_order=False, formulation="interaction")
    assert interaction.name == "adams-pec3/4"


def test_the_representation_names_the_formulation():
    # A run's record hashes the representation (output schema 15): it names the
    # formulation whichever it is, and the comparison variants keep the interaction one.
    assert repr(Multistep()).endswith("formulation='exponential')")
    assert "formulation='interaction'" in repr(Multistep(formulation="interaction"))
    assert ComparisonMultistep(order=4).formulation == "interaction"


@pytest.mark.parametrize(
    "fields",
    [{"order": 0}, {"order": 7}, {"max_calls": 0}, {"max_growth": 1.0}],
)
def test_selected_method_refuses_out_of_range(fields):
    with pytest.raises(StepperError):
        Multistep(**fields)


@pytest.mark.parametrize("order", range(1, 7))
def test_pec_real_interval_is_the_recurrence_spectral_limit(order):
    # PEC q/(q+1) for u' = lambda u at constant step: the state u_n and the q
    # predicted-state samples advance by a companion matrix; the tabulated interval
    # is the largest -h lambda for which its spectral radius stays at most one.
    def radius(z):
        predictor = integrated_weights([-i for i in range(order)])
        corrector = integrated_weights([1.0] + [-i for i in range(order)])
        size = order + 1
        predicted = np.zeros(size)
        predicted[0], predicted[1:] = 1.0, predictor
        sample = z * predicted
        advanced = np.zeros(size)
        advanced[0], advanced[1:] = 1.0, corrector[1:]
        advanced += corrector[0] * sample
        matrix = np.zeros((size, size))
        matrix[0], matrix[1] = advanced, sample
        for row in range(2, size):
            matrix[row, row - 1] = 1.0
        return max(abs(np.linalg.eigvals(matrix)))

    interval = Multistep(order=order).real_stability_interval
    # Rounded down by at most 1e-3 relative: stable just inside, unstable 1 % out.
    assert radius(-interval) <= 1.0 + 1e-9
    assert radius(-1.01 * interval / (1 - 1e-3)) > 1.0


def _python_basis(points, upper):
    """The pure-Python loops the compiled basis replaced, kept as its reference."""
    weights = np.empty(len(points))
    rows = np.zeros((len(points), len(points)))
    for index, node in enumerate(points):
        coefficients = [1.0]
        denominator = 1.0
        for other, root in enumerate(points):
            if other == index:
                continue
            product = [0.0, *coefficients]
            for power, value in enumerate(coefficients):
                product[power] -= root * value
            coefficients = product
            denominator *= node - root
        integral = 0.0
        for power in range(len(coefficients) - 1, -1, -1):
            integral = (integral + coefficients[power] / (power + 1)) * upper
        weights[index] = integral / denominator
        rows[index] = np.asarray(coefficients) / denominator
    return rows, weights


def _majorant(points, upper):
    """The same loops on the magnitudes: every term the rounding bound scales with.

    Each coefficient of ``prod (x - r) / d`` is a sum of signed products of roots;
    with every root and denominator replaced by its modulus the recurrence adds the
    terms' moduli instead, and the Horner integral of those moduli over ``[0, upper]``
    majorises the weight's terms the same way.
    """
    weights = np.empty(len(points))
    rows = np.zeros((len(points), len(points)))
    for index, node in enumerate(points):
        coefficients = [1.0]
        denominator = 1.0
        for other, root in enumerate(points):
            if other == index:
                continue
            product = [0.0, *coefficients]
            for power, value in enumerate(coefficients):
                product[power] += abs(root) * value
            coefficients = product
            denominator *= abs(node - root)
        integral = 0.0
        for power in range(len(coefficients) - 1, -1, -1):
            integral = (integral + coefficients[power] / (power + 1)) * abs(upper)
        weights[index] = integral / denominator
        rows[index] = np.asarray(coefficients) / denominator
    return rows, weights


def assert_basis_close(got, expected, majorant, count):
    """Assert a compiled basis is the loops' within both evaluations' rounding.

    The compiled kernel may contract its multiply-adds (SEP 25), so it is not the
    loops bit for bit. Each coefficient takes ``count - 1`` multiply-adds and a
    division, each weight a Horner pass of two roundings a term and a division: at
    most ``4 count + 4`` roundings, against the majorant, for each of the two
    evaluations (:func:`bounds.gamma` counts in ``eps = 2 u``, which covers both).
    """
    bound = gamma(4 * count + 4) * np.abs(majorant)
    assert np.all(np.abs(np.asarray(got) - np.asarray(expected)) <= bound)


@pytest.mark.parametrize("seed", range(40))
def test_the_compiled_basis_is_the_python_loops_to_rounding(seed):
    """Weights and coefficients as the scalar loops computed them, to rounding."""
    rng = np.random.default_rng(seed)
    count = int(rng.integers(1, 8))
    nodes = np.sort(rng.uniform(-6.0, 1.2, count))[::-1].copy()
    upper = float(rng.uniform(0.0, 1.0)) if seed % 2 else 1.0
    rows, weights = _python_basis(nodes.tolist(), upper)
    row_bound, weight_bound = _majorant(nodes.tolist(), upper)
    got_weights = integrated_weights(nodes, upper, check=False)
    got_rows = lagrange_coefficients(nodes, check=False)
    assert_basis_close(got_weights, weights, weight_bound, count)
    assert_basis_close(got_rows, rows, row_bound, count)


@pytest.mark.parametrize("integrate", [False, True])
def test_the_compiled_basis_is_the_python_one_to_rounding(integrate):
    # The weights and coefficients every Adams step takes are the compiled form of
    # the Python loops: the same operations in the same order, which a compiled
    # kernel may contract, so they agree within both evaluations' rounding.
    points = np.array([0.0, -0.7, -1.9, -2.6, -4.1])
    python = inspect.unwrap(_basis_rows)(points, 0.85, integrate)
    compiled = _basis()(points, 0.85, integrate)
    rows, weights = _majorant(points.tolist(), 0.85)
    if integrate:
        assert_basis_close(compiled[1], python[1], weights, points.size)
        assert_basis_close(
            integrated_weights(points, 0.85), python[1], weights, points.size
        )
    else:
        assert_basis_close(compiled[0], python[0], rows, points.size)
