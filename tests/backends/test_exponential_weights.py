# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The phi functions and exponential Adams weights against independent references.

SEP 16 section 7.1: moments and complete weights checked against augmented matrix
exponentials and against quadrature of their defining integrals, at zero, on both sides
of the evaluator's join, at large imaginary arguments, under strong damping, at
resonance (``exp(x) = 1``), on unequal nodes and on ill-conditioned histories; and the
ordinary Adams weights recovered at ``x = 0``.
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from solprop.backends.exponential import (
    MAX_PHI_ORDER,
    PHI_JOIN,
    exponential_weights,
    phi_functions,
)
from solprop.exceptions import StepperError
from solprop.stepper.multistep import integrated_weights, lagrange_coefficients

EPS = np.finfo(float).eps

# Arguments x = h D: zero, tiny, both sides of the join in several directions, large
# imaginary (dispersion), strong damping, gain within range, and resonances exp(x) = 1.
ARGUMENTS = np.array(
    [
        0.0,
        1e-12j,
        -1e-9,
        0.3 - 0.2j,
        0.5j,
        -2.0 + 3.0j,
        PHI_JOIN * (1 - 1e-12) * 1j,
        PHI_JOIN * (1 + 1e-12) * 1j,
        -PHI_JOIN * (1 - 1e-12),
        -PHI_JOIN * (1 + 1e-12),
        PHI_JOIN * np.exp(0.7j) * (1 - 1e-12),
        PHI_JOIN * np.exp(0.7j) * (1 + 1e-12),
        5.0,
        40j,
        -3.0 - 1000j,
        -50.0,
        -700.0 + 20j,
        2j * np.pi,
        4j * np.pi,
        2j * np.pi * 51,
    ]
)


def augmented_phi(x: complex, order: int) -> np.ndarray:
    """Return phi_0..phi_order from the first row of an augmented exponential.

    ``exp`` of the upper bidiagonal matrix with ``x`` at (0, 0) and ones on the
    superdiagonal has ``phi_k(x)`` at (0, k) (Hochbruck and Ostermann 2010, section 2).
    """
    matrix = np.zeros((order + 1, order + 1), dtype=complex)
    matrix[0, 0] = x
    for k in range(order):
        matrix[k, k + 1] = 1.0
    return expm(matrix)[0]


def quadrature_phi(x: complex, order: int) -> complex:
    """Return phi_order(x) by adaptive quadrature of its defining integral.

    ``phi_r(x) = exp(x) int_0^1 exp(-theta Re x) exp(-i theta Im x) theta^(r-1) /
    (r-1)! dtheta``, the oscillatory factor handled by QUADPACK's weighted rules.
    """
    real, imag = float(np.real(x)), float(np.imag(x))
    factor = math.factorial(order - 1)

    def envelope(theta: float) -> float:
        return math.exp((1.0 - theta) * real) * theta ** (order - 1) / factor

    if imag == 0.0:
        return complex(quad(envelope, 0.0, 1.0, epsabs=0.0, epsrel=1e-12, limit=400)[0])
    # exp(i (1 - theta) Im x) = cos(Im x - theta Im x) + i sin(...).
    cosine = quad(
        envelope,
        0.0,
        1.0,
        weight="cos",
        wvar=imag,
        epsabs=1e-17,
        epsrel=1e-11,
        limit=400,
    )[0]
    sine = quad(
        envelope,
        0.0,
        1.0,
        weight="sin",
        wvar=imag,
        epsabs=1e-17,
        epsrel=1e-11,
        limit=400,
    )[0]
    rotation = complex(math.cos(imag), math.sin(imag))
    return rotation * complex(cosine, -sine)


def natural_scale(x: complex, order: int) -> float:
    """Return a bound on the size of phi_order's terms: the scale of its error."""
    # |phi_r(x)| <= phi_r(Re x) <= max(1, exp(Re x)) / r!, and for |x| > 1 the function
    # falls as 1/|x|; its rounding is relative to the larger of the two.
    return max(1.0, math.exp(x.real)) / math.factorial(order)


@pytest.mark.parametrize("x", ARGUMENTS, ids=lambda one: f"{one:.3g}")
def test_phi_functions_agree_with_the_augmented_exponential(x):
    order = MAX_PHI_ORDER
    values = phi_functions(np.array([x]), order)[:, 0]
    reference = augmented_phi(complex(x), order)
    for k in range(order + 1):
        # The evaluator: a 24-term series and at most four doublings, or k upward
        # recurrence steps each dividing by |x| > 8; the reference: scaling and
        # squaring with a Pade approximant. Each is accurate to tens of ulp of the
        # function's natural scale; 256 ulp is their sum with a margin, and is still
        # 1e3 below the coarsest coefficient error that would show in a run.
        assert abs(values[k] - reference[k]) <= 256 * EPS * natural_scale(x, k), k


@pytest.mark.parametrize(
    "x", ARGUMENTS[np.real(ARGUMENTS) > -100], ids=lambda one: f"{one:.3g}"
)
@pytest.mark.parametrize("order", [1, 3, MAX_PHI_ORDER])
def test_phi_functions_agree_with_quadrature(x, order):
    value = phi_functions(np.array([x]), order)[order, 0]
    reference = quadrature_phi(complex(x), order)
    # Requested at 1e-11 relative (tighter requests trip QUADPACK's round-off
    # detection at |Im x| = 1000); measured below 1e-18 of the natural scale.
    assert abs(value - reference) <= 1e-13 * natural_scale(complex(x), order)


def test_phi_functions_keep_the_shape_and_vanish_at_resonance():
    x = np.array([[0.0, 2j * np.pi], [-1.0, 1.0]])
    values = phi_functions(x, 3)
    assert values.shape == (4, 2, 2)
    np.testing.assert_allclose(values[:, 0, 0], [1, 1, 1 / 2, 1 / 6], rtol=0, atol=EPS)
    # exp(2 pi i) = 1, so phi_1 vanishes; its rounding is that of exp(x) - 1 over |x|.
    assert abs(values[1, 0, 1]) <= 4 * EPS / (2 * np.pi)


@pytest.mark.parametrize(
    ("x", "order"),
    [(np.array([np.nan]), 2), (np.array([701.0]), 2), (np.array([0.0]), 0)],
    ids=["nan", "overflow", "order"],
)
def test_phi_functions_refuse(x, order):
    with pytest.raises(StepperError):
        phi_functions(x, order)


def augmented_integral(coefficients: np.ndarray, x: complex, upper: float) -> complex:
    """Return int_0^upper exp((upper - theta) x) p(theta) dtheta by one exponential.

    ``s' = x s + sum_k a_k theta^k`` with the monomials generated by the nilpotent
    system ``v_k = theta^k / k!``, ``v_0' = 0``, ``v_k' = v_{k-1}``: ``s(upper)`` is one
    entry of ``exp(upper A)`` applied to ``(0, 1, 0, ...)``. Independent of the Lagrange
    coefficients and of the phi evaluator.
    """
    degree = len(coefficients) - 1
    size = degree + 2
    matrix = np.zeros((size, size), dtype=complex)
    matrix[0, 0] = x
    for k, value in enumerate(coefficients):
        matrix[0, 1 + k] = value * math.factorial(k)
    for k in range(1, degree + 1):
        matrix[1 + k, k] = 1.0
    start = np.zeros(size, dtype=complex)
    start[1] = 1.0
    return complex((expm(upper * matrix) @ start)[0])


NODE_SETS = {
    "equal predictor": [0.0, -1.0, -2.0, -3.0, -4.0],
    "equal corrector": [1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
    "unequal": [1.0, 0.0, -0.4, -1.7, -2.1, -3.9],
    "growing ratio 2": [1.0, 0.0, -0.5, -0.75, -0.875, -0.9375],
    "clustered": [1.0, 0.0, -0.01, -0.02, -2.0],
    "one node": [0.0],
}


@pytest.mark.parametrize("nodes", NODE_SETS.values(), ids=NODE_SETS.keys())
@pytest.mark.parametrize("x", [0.0, 0.7j, -2.0 + 9j, 30j, -40.0, 2j * np.pi])
@pytest.mark.parametrize("upper", [1.0, 0.37])
def test_weights_integrate_polynomials_exactly(nodes, x, upper):
    weights = exponential_weights(nodes, np.array([x]), upper)[:, 0]
    theta = np.asarray(nodes)
    rng = np.random.default_rng(len(nodes))
    coefficients = rng.normal(size=len(nodes))
    samples = np.polynomial.polynomial.polyval(theta, coefficients)
    exact = augmented_integral(coefficients, complex(x), upper)
    # The weights carry the Lagrange coefficients' conditioning: the moment errors
    # (a few ulp of each phi's natural scale, which a resonance does not lower) are
    # multiplied by sum_k |c_jk| k! and summed with |p(theta_j)|. 64 ulp of that sum
    # bounds the reduction and the moments; the reference is one Pade exponential of a
    # matrix of norm |x| plus the coefficients.
    count = len(nodes)
    phi = np.array([natural_scale(complex(upper * x), k + 1) for k in range(count)])
    magnitude = np.abs(lagrange_coefficients(nodes)) @ (
        phi * np.array([math.factorial(k) * upper ** (k + 1) for k in range(count)])
    )
    bound = 64 * EPS * float(np.sum(magnitude * np.abs(samples)) + abs(exact))
    assert abs(np.dot(weights, samples) - exact) <= bound


@pytest.mark.parametrize("nodes", NODE_SETS.values(), ids=NODE_SETS.keys())
@pytest.mark.parametrize("upper", [1.0, 0.37])
def test_weights_reduce_to_ordinary_adams_at_zero(nodes, upper):
    exponential = exponential_weights(nodes, np.zeros(3), upper)
    ordinary = integrated_weights(nodes, upper)
    for column in range(3):
        # Two constructions of the same integrals over the same coefficients: a few
        # ulp of each term's size, which the clustered nodes raise to about 1e3.
        scale = float(np.max(np.abs(lagrange_coefficients(nodes))))
        np.testing.assert_allclose(
            np.real(exponential[:, column]), ordinary, rtol=0, atol=64 * EPS * scale
        )
        assert np.all(np.imag(exponential[:, column]) == 0.0)


def test_weights_of_a_single_node_are_phi_one():
    x = np.array([0.1j, -3.0, 25j])
    np.testing.assert_allclose(
        exponential_weights([0.0], x)[0], phi_functions(x, 1)[1], rtol=4 * EPS
    )


def test_weights_refuse_a_fraction_outside_the_step():
    with pytest.raises(StepperError, match="fraction"):
        exponential_weights([0.0, -1.0], 1j, 1.5)
    with pytest.raises(StepperError, match="distinct"):
        exponential_weights([0.0, 0.0], 1j)


def test_lagrange_coefficients_are_cardinal():
    nodes = np.array([1.0, 0.0, -0.4, -1.7, -2.1, -3.9])
    values = np.polynomial.polynomial.polyval(nodes, lagrange_coefficients(nodes).T)
    # polyval of degree five at |theta| <= 3.9 with coefficients up to about 1e1.
    np.testing.assert_allclose(values, np.eye(len(nodes)), rtol=0, atol=1e-12)
