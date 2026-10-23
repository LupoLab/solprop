# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Derive the complete plasma column envelope before evaluating its error.

The exact same scalar rate pass supplies the inputs. The serial trace below is
used to price absolute sensitivity, not as a fitted numerical tolerance. Keep
the independent 192 epsilon ceil(log2(n)) parity cap alongside this envelope.
"""

import math

import numpy as np
from plasma_scan import PlasmaWork

from solprop.backends.bodies import PLAIN

EPS = np.finfo(float).eps
__all__ = ["allocate_work", "column_budget", "gamma"]


def gamma(count):
    scaled = count * EPS
    return scaled / (1.0 - scaled)


def outward(value):
    """Inflate positive bound arithmetic and retain subnormal endpoints."""
    return np.nextafter(value * (1 + 16 * EPS), np.inf)


def absolute_prefix(values):
    n = values.shape[-1]
    return outward(np.cumsum(np.abs(values), axis=-1) / (1 - gamma(n - 1)))


def scan_budget(values, uncertainty, depth):
    n = values.shape[-1]
    return outward(
        (gamma(depth) + gamma(n - 1)) * absolute_prefix(values)
        + (1 + gamma(depth)) * absolute_prefix(uncertainty)
    )


def trapezoids(values, dt):
    result = np.zeros_like(values)
    result[:, 1:] = dt * (values[:, 1:] + values[:, :-1]) / 2.0
    return result


def trapezoid_budget(values, uncertainty, dt):
    result = np.zeros_like(values)
    result[:, 1:] = outward(
        (abs(dt) / 2)
        * (
            (1 + gamma(2)) * (uncertainty[:, 1:] + uncertainty[:, :-1])
            + 2 * gamma(2) * (np.abs(values[:, 1:]) + np.abs(values[:, :-1]))
        )
    )
    return result


def product_budget(value, uncertainty, factor):
    return outward(np.abs(factor) * ((1 + EPS) * uncertainty + 2 * EPS * np.abs(value)))


def allocate_work(shape):
    """Allocate all prototype scratch outside every compiled and timed call."""
    segments = min(16, shape[-1])
    return PlasmaWork(
        *(np.empty(shape) for _ in range(5)),
        np.empty((shape[0], segments)),
        np.empty((shape[0], segments)),
    )


def column_budget(nodes, rates, window, scalars, *, raman):
    """Propagate the actual partition/carry error through all three plasma scans."""
    n = nodes.shape[-1]
    segments = min(16, n)
    depth = 2 * math.ceil(math.log2(n // segments)) + segments + 3
    exposure_increments = trapezoids(rates, scalars.dt)
    exposure = np.cumsum(exposure_increments, axis=-1)
    exposure_error = scan_budget(
        exposure_increments, np.zeros_like(exposure_increments), depth
    )
    fraction = np.fromiter(
        (PLAIN.fraction_at(float(x), scalars.preionisation) for x in exposure.flat),
        dtype=float,
        count=exposure.size,
    ).reshape(exposure.shape)
    # The exact map has derivative <=1. Eight eps covers both scalar exp paths
    # (the existing one-ulp body contract), multiplication and subtraction. This
    # is an explicit floating-libm assumption, not a proved vendor error bound.
    fraction_error = outward(exposure_error + 8 * EPS)
    weight = fraction * scalars.charge_mass_ratio
    weight_error = product_budget(fraction, fraction_error, scalars.charge_mass_ratio)
    integrand = weight * nodes
    integrand_error = product_budget(weight, weight_error, nodes)
    squared = np.abs(nodes) * np.abs(nodes)
    positive = (rates > 0) & (squared > 0)
    coefficient = scalars.ionisation_potential * rates
    guard = np.zeros_like(nodes)
    np.divide(coefficient * (1 - fraction), squared, out=guard, where=positive)
    sensitivity = np.zeros_like(nodes)
    np.divide(np.abs(coefficient), squared, out=sensitivity, where=positive)
    guard_error = outward(
        (1 + gamma(3)) * sensitivity * fraction_error
        + 2 * gamma(3) * (np.abs(guard) + sensitivity * fraction_error)
    )
    guard_error[~positive] = 0.0
    loss = guard * nodes
    loss_error = product_budget(guard, guard_error, nodes)
    free_increments = trapezoids(integrand, scalars.dt)
    free_error = scan_budget(
        free_increments,
        trapezoid_budget(integrand, integrand_error, scalars.dt),
        depth,
    )
    free = np.cumsum(free_increments, axis=-1)
    current = free + loss
    current_error = outward(
        (1 + EPS) * (free_error + loss_error) + 2 * EPS * (np.abs(free) + np.abs(loss))
    )
    outer_increments = trapezoids(current, scalars.dt)
    outer_error = scan_budget(
        outer_increments,
        trapezoid_budget(current, current_error, scalars.dt),
        depth,
    )
    outer = np.cumsum(outer_increments, axis=-1)
    plasma = scalars.density * outer
    plasma_error = product_budget(outer, outer_error, scalars.density)
    kerr = (scalars.kerr_factor * (nodes * nodes)) * nodes
    total = plasma + kerr
    total_error = outward(
        (1 + EPS) * plasma_error + 2 * EPS * (np.abs(plasma) + np.abs(kerr))
    )
    if not raman:
        total_error = product_budget(total, total_error, window)
        total = total * window
    arrays = {
        "column_bound": total_error,
        "fraction_bound": fraction_error[:, -1],
        "serial_trace": total,
        "serial_fraction": fraction[:, -1],
    }
    summary = {
        "samples": n,
        "segments": segments,
        "addition_depth": depth,
        "serial_gamma": gamma(n - 1),
        "candidate_gamma": gamma(depth),
        "column_absolute_bound_max": float(np.max(total_error)),
        "fraction_absolute_bound_max": float(np.max(fraction_error[:, -1])),
        "complete_chain_relative_cap": 192 * EPS * math.ceil(math.log2(n)),
        "scalar_exp_assumption": (
            "existing one-ulp body contract; 8 epsilon two-path allowance"
        ),
        "maximum_exposure": float(np.max(exposure)),
        "positive_guard_samples": int(np.count_nonzero(positive)),
        "rate_nonzero_samples": int(np.count_nonzero(rates)),
    }
    return arrays, summary
