# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent finite-contour and continuous-series bounds for the raw QDHT.

Derivations are in docs/guide/free-space/hankel.md. None of the bound routines
forms T @ T or transforms the qualification field. Their results precede
those measurements.
"""

import math
from functools import cache

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_laguerre, hankel1e, jn_zeros, jv, jve

EPS = np.finfo(float).eps
QUADRATURE_BUDGET = 32 * EPS


def gamma(count):
    return count * EPS / (1 - count * EPS)


@cache
def truncation_bound(n_r):
    """Integrate the absolute imaginary contour remainder, independently of T."""
    zeros = jn_zeros(0, n_r + 1)
    edge = float(zeros[-1])
    ratios = zeros[:-1] / edge
    j1 = np.abs(jv(1, zeros[:-1]))

    def integrand(y):
        z = complex(edge, y)
        # Scaled special functions leave only exp(-(2-a_i-a_j)*y); the
        # contour is exponentially convergent without overflow or a tail cutoff.
        vector = jve(0, ratios * z) * np.exp((ratios - 1) * y) / j1
        ratio = (
            hankel1e(0, np.complex128(z)) * np.exp(1j * edge) / jve(0, np.complex128(z))
        )
        values = 2 * z / edge**2 * ratio * np.outer(vector, vector)
        return float(np.max(np.abs(values.imag)))

    value, error = quad(
        integrand, 0, np.inf, epsabs=QUADRATURE_BUDGET / 8, epsrel=0, limit=500
    )
    assert 0 <= error <= QUADRATURE_BUDGET, (n_r, error, QUADRATURE_BUDGET)
    return value + error


def construction_error(transform):
    """Bound conditioned argument, Bessel-value and diagonal-scaling roundoff."""
    # The standard floating-point oracle model budgets eight eps for a Bessel
    # root and absolute J0 evaluation, then gamma(32) for both root arguments
    # and the common/diagonal scaling. |J0'| <= 1 on the real axis makes the
    # argument contribution absolute, including near-zero matrix entries.
    # J1'(alpha) = -J1(alpha)/alpha at a J0 root, so root perturbations have
    # relative condition one in the denominator. This model, like the FFT
    # envelope, is a library-accuracy assumption, not a correct-rounding claim.
    edge = transform.zeros[-1]
    denominator = edge * np.outer(transform.j1, transform.j1)
    return 2 / denominator * (gamma(32) * edge + 8 * EPS) + gamma(32) * np.abs(
        transform.matrix
    )


def gram_bound(transform):
    """Add construction and GEMM roundoff to the contour truncation bound."""
    truncation = truncation_bound(transform.n_r)
    absolute = np.abs(transform.matrix)
    error = construction_error(transform)
    floating = (
        absolute @ error
        + error @ absolute
        + error @ error
        + gamma(transform.n_r) * (absolute @ absolute)
    )
    return truncation, floating


def gaussian_bound(transform, waist):
    """Bound the finite Fourier-Bessel expansion before testing the Gaussian."""
    radius, size = transform.radius, transform.n_r
    power = 8
    wall = np.exp(-((radius / waist) ** 2))

    def root_tail(exponent):
        return np.pi ** (-exponent) * (size - 0.25) ** (1 - exponent) / (exponent - 1)

    residual = wall
    for degree in range(1, power):
        boundary = abs(
            (-4 / waist**2) ** degree
            * math.factorial(degree)
            * eval_laguerre(degree, (radius / waist) ** 2)
            * wall
        )
        residual += (
            2
            * np.sqrt(np.pi)
            * radius ** (2 * degree)
            * boundary
            * root_tail(2 * degree + 0.5)
        )
    integral = waist**2 / 2 * (4 / waist**2) ** power * math.factorial(power)
    residual += (
        2 * np.pi * radius ** (2 * power - 2) * integral * root_tail(2 * power - 1)
    )
    reference = waist**2 / 2 * np.exp(-((transform.k_r * waist) ** 2) / 4)
    finite_radius = wall * (
        waist**2 / 2 + radius**2 * transform.j1 / transform.zeros[:-1]
    )
    forward = transform.forward_scale * (
        transform.j1[:, None] * transform.matrix / transform.j1[None, :]
    )
    truncation = (
        transform.j1
        * truncation_bound(size)
        * np.sum((np.abs(reference) + finite_radius) / transform.j1)
    )
    spatial_tail = np.sum(np.abs(forward), axis=1) * residual + finite_radius
    # f is bounded by one. Include diagonal scalings and the length-N product;
    # this uses that bound, not the as-yet untransformed Gaussian samples.
    floating = (
        transform.forward_scale
        * transform.j1
        * (
            construction_error(transform) @ (1 / transform.j1)
            + gamma(size + 8) * (np.abs(transform.matrix) @ (1 / transform.j1))
        )
    )
    return reference, truncation + spatial_tail + floating


def complex_gaussian_bound(transform, coefficient):
    """Bound exp(-a*r^2) for Re(a)>0, including its complex phase."""
    radius, size = transform.radius, transform.n_r
    a = complex(coefficient)
    assert a.real > 0
    power = 8
    wall = np.exp(-a.real * radius**2)

    def root_tail(exponent):
        return np.pi ** (-exponent) * (size - 0.25) ** (1 - exponent) / (exponent - 1)

    def laguerre(degree, argument):
        return sum(
            math.comb(degree, j) * (-argument) ** j / math.factorial(j)
            for j in range(degree + 1)
        )

    residual = wall
    for degree in range(1, power):
        boundary = (
            abs(
                (-4 * a) ** degree
                * math.factorial(degree)
                * laguerre(degree, a * radius**2)
            )
            * wall
        )
        residual += (
            2
            * np.sqrt(np.pi)
            * radius ** (2 * degree)
            * boundary
            * root_tail(2 * degree + 0.5)
        )
    # Triangle inequality on the Laguerre polynomial; each monomial integrates
    # exactly against exp(-Re(a)*r^2). No transformed samples enter this bound.
    integral = (
        (4 * abs(a)) ** power
        * math.factorial(power)
        * (1 + abs(a) / a.real) ** power
        / (2 * a.real)
    )
    residual += (
        2 * np.pi * radius ** (2 * power - 2) * integral * root_tail(2 * power - 1)
    )
    reference = np.exp(-(transform.k_r**2) / (4 * a)) / (2 * a)
    finite_radius = wall * (
        1 / (2 * a.real) + radius**2 * transform.j1 / transform.zeros[:-1]
    )
    forward = transform.forward_scale * (
        transform.j1[:, None] * transform.matrix / transform.j1[None, :]
    )
    truncation = (
        transform.j1
        * truncation_bound(size)
        * np.sum((np.abs(reference) + finite_radius) / transform.j1)
    )
    spatial_tail = np.sum(np.abs(forward), axis=1) * residual + finite_radius
    floating = (
        transform.forward_scale
        * transform.j1
        * (
            construction_error(transform) @ (1 / transform.j1)
            + gamma(size + 8) * (np.abs(transform.matrix) @ (1 / transform.j1))
        )
    )
    return reference, truncation + spatial_tail + floating


def gaussian_axis_bound(transform, coefficient):
    """Bound the finite Fourier-Bessel series at r=0 from continuous coefficients."""
    radius, size = transform.radius, transform.n_r
    b = complex(coefficient)
    assert b.real > 0
    power = 8
    wall = np.exp(-b.real * radius**2)
    residual = wall
    for degree in range(1, power):
        polynomial = sum(
            math.comb(degree, j) * (-b * radius**2) ** j / math.factorial(j)
            for j in range(degree + 1)
        )
        boundary = abs((-4 * b) ** degree * math.factorial(degree) * polynomial) * wall
        exponent = 2 * degree + 0.5
        tail = np.pi ** (-exponent) * (size - 0.25) ** (1 - exponent) / (exponent - 1)
        residual += 2 * np.sqrt(np.pi) * radius ** (2 * degree) * boundary * tail
    integral = (
        (4 * abs(b)) ** power
        * math.factorial(power)
        * (1 + abs(b) / b.real) ** power
        / (2 * b.real)
    )
    exponent = 2 * power - 1
    tail = np.pi ** (-exponent) * (size - 0.25) ** (1 - exponent) / (exponent - 1)
    residual += 2 * np.pi * radius ** (2 * power - 2) * integral * tail
    # Difference between the finite-disk coefficients and infinite Gaussian
    # coefficients, plus subtraction of the constant wall value for Dirichlet data.
    finite_radius = wall * (
        1 / (2 * b.real) + radius**2 * transform.j1 / transform.zeros[:-1]
    )
    weights = transform.spectral_weights / (2 * np.pi)
    coefficient_magnitudes = np.exp(-(transform.k_r**2) * np.real(1 / b) / 4) / (
        2 * abs(b)
    )
    rounding = gamma(32 + size) * np.sum(weights * coefficient_magnitudes)
    return residual + np.sum(weights * finite_radius) + rounding
