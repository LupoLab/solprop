# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent frozen-cycle experiments; no production response or kernel imports.

The adaptive oracle integrates Cartesian real fields. The candidate uses ellipse
axes and split Gauss-Legendre rules. Their integrands and coordinates differ.
"""

import functools
import itertools
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import integrate, interpolate, special

from rate_oracles import adk_oracle
from solprop.exceptions import ConvergenceError
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.stepper.schedule import CONCLUSIVE_FACTOR, ORDER_MARGIN, observed_order
from solprop.translate.ionisation import build_rate_table

BUDGETS = json.loads(
    Path(__file__)
    .with_name("data")
    .joinpath("envelope_plasma_oracle/budgets.json")
    .read_text()
)


def cartesian_oracle(envelope, rate):
    """Return Wbar and j/(Ip rho survival), integrating the full physical cycle."""
    envelope = np.asarray(envelope, dtype=np.complex128)
    scale = float(np.linalg.norm(envelope))
    if scale == 0:
        return 0.0, np.zeros_like(envelope), 0.0
    rate_scale = max(float(rate(scale)), 1e-300)

    def integrand(theta):
        field = (envelope * np.exp(-1j * theta)).real
        square = float(field @ field)
        weight = float(rate(math.sqrt(square))) / rate_scale
        current = np.zeros_like(envelope)
        if square > 0:
            current = 2 * weight * field * np.exp(1j * theta) * scale / square
        return np.concatenate(([weight], current))

    result, error = integrate.quad_vec(
        integrand,
        0.0,
        2 * np.pi,
        epsabs=1e-13,
        epsrel=BUDGETS["adaptive_angular_rtol"],
        limit=2000,
        points=np.linspace(0, 2 * np.pi, 17),
    )
    return (
        float(result[0].real * rate_scale / (2 * np.pi)),
        result[1:] * rate_scale / (2 * np.pi * scale),
        float(error * rate_scale / (2 * np.pi)),
    )


@functools.lru_cache
def angular_rule(order):
    return special.roots_legendre(order)


def axis_moments(major, ratio, rate, order, *, edges=()):
    """Return <W c²/d>, <W s²/d>; split at rate threshold and clipping crossings."""
    if major == 0:
        return np.zeros(2)
    boundaries = [0.0, np.pi / 2]
    if ratio < 1:
        for edge in edges:
            square = (edge / major) ** 2
            if ratio**2 < square < 1:
                boundaries.append(
                    math.acos(math.sqrt((square - ratio**2) / (1 - ratio**2)))
                )
    boundaries.sort()
    nodes, weights = angular_rule(order)
    total = np.zeros(2)
    for lower, upper in itertools.pairwise(boundaries):
        theta = lower + (nodes + 1) * (upper - lower) / 2
        cosine, sine = np.cos(theta) ** 2, np.sin(theta) ** 2
        denominator = cosine + ratio**2 * sine
        values = np.asarray(rate(major * np.sqrt(denominator))) / denominator
        total += (
            (upper - lower)
            / np.pi
            * np.array(
                [
                    np.dot(weights, values * cosine),
                    np.dot(weights, values * sine),
                ]
            )
        )
    return total


def reconstructed_current(envelope, rate, order=128):
    """Apply the spectral function of the real 2x2 Gram matrix, without an angle."""
    envelope = np.asarray(envelope)
    basis = np.column_stack((envelope.real, envelope.imag))
    eigenvalues, eigenvectors = np.linalg.eigh(basis.T @ basis)
    major = math.sqrt(max(float(eigenvalues[1]), 0))
    if major == 0:
        return 0.0, np.zeros_like(envelope)
    ratio = math.sqrt(max(float(eigenvalues[0]), 0)) / major
    moments = axis_moments(major, ratio, rate, order)
    matrix = (eigenvectors * (moments[::-1] / major**2)) @ eigenvectors.T
    return float(moments[0] + ratio**2 * moments[1]), 2 * basis @ matrix @ np.array(
        [1, 1j]
    )


def polynomial_experiments():
    rows = []
    for minor in (0.0, 1e-12, 0.7, 2 * (1 - 1e-12), 2.0):
        major = 2.0
        envelope = np.array([major, 1j * minor])
        exact_rate = (3 * major**4 + 2 * major**2 * minor**2 + 3 * minor**4) / 8
        exact_current = (
            np.array(
                [
                    major * (3 * major**2 + minor**2),
                    1j * minor * (major**2 + 3 * minor**2),
                ]
            )
            / 4
        )
        average, current, error = cartesian_oracle(envelope, lambda field: field**4)
        transformed = []
        for phase in (0.0, 0.317, 1.8):
            rotation = np.array(
                [[np.cos(0.61), -np.sin(0.61)], [np.sin(0.61), np.cos(0.61)]]
            )
            rotated = rotation @ envelope * np.exp(1j * phase)
            _, actual = reconstructed_current(rotated, lambda field: field**4)
            transformed.append(
                float(
                    np.linalg.norm(
                        actual - rotation @ exact_current * np.exp(1j * phase)
                    )
                )
            )
        collinear = 2 * average * envelope / np.vdot(envelope, envelope).real
        rows.append(
            {
                "minor": minor,
                "rate_error": abs(average - exact_rate),
                "current_error": float(np.linalg.norm(current - exact_current)),
                "adaptive_error_estimate": error,
                "covariance_error": max(transformed),
                "work_error": abs(float(np.vdot(envelope, current).real / 2) - average),
                "collinear_relative_error": float(
                    np.linalg.norm(collinear - current) / np.linalg.norm(current)
                ),
            }
        )
    return rows


def population(rate, dt, preionisation):
    exposure = integrate.cumulative_trapezoid(rate, dx=dt, initial=0)
    survival = (1 - preionisation) * np.exp(-exposure)
    fraction = preionisation + (1 - preionisation) * (-np.expm1(-exposure))
    return fraction, survival


def order_record(errors, order, floor):
    """Use the project's observed order and p+3/floor policy, on all three rungs."""
    errors = [float(value) for value in errors]
    conclusive = min(errors) > CONCLUSIVE_FACTOR * floor
    orders = [observed_order(errors[i], errors[i + 1]) for i in range(2)]
    finite_orders = [value if math.isfinite(value) else None for value in orders]
    return {
        "errors": errors,
        "orders": finite_orders,
        "floor": floor,
        "status": "inconclusive: floor limited"
        if not conclusive
        else (
            "pass"
            if all(order - 0.2 < value <= order + ORDER_MARGIN for value in orders)
            else "fail"
        ),
    }


def population_experiments():
    errors = []
    for count in (33, 65, 129):
        time = np.linspace(0, 1, count)
        fraction, _ = population(time**2, float(time[1]), 0.2)
        exact = 0.2 + 0.8 * (-np.expm1(-(time**3) / 3))
        errors.append(float(np.max(np.abs(fraction - exact))))
    cases = []
    for initial in (0, 0.2, 1):
        for exposure in (0, 1e-18, 1, 1000):
            fraction, survival = population(np.full(3, exposure), 0.5, initial)
            cases.append(
                {
                    "f0": initial,
                    "H": exposure,
                    "first": float(fraction[0]),
                    "last": float(fraction[-1]),
                    "survival": float(survival[-1]),
                }
            )
    return {
        "trapezoid": order_record(errors, 2, 64 * np.finfo(float).eps),
        "cases": cases,
    }


def relative_moments(actual, reference):
    """Compare positive moments in s^-1 with the declared negligible floor."""
    return float(
        np.max(
            np.abs(actual - reference)
            / np.maximum(reference, BUDGETS["negligible_rate_s_inverse"])
        )
    )


def refined_moments(major, ratio, rate, edges):
    """Exercise the proposed runtime refinement and explicit failure policy."""
    previous = axis_moments(major, ratio, rate, 64, edges=edges)
    for order in (128, 256, 512):
        current = axis_moments(major, ratio, rate, order, edges=edges)
        if relative_moments(current, previous) <= BUDGETS["angular_rtol"]:
            return current, order
        previous = current
    raise ConvergenceError("cycle moments did not converge by 512 points")


def cutoff_experiment():
    """Isolate the physical difference caused by an arbitrary table lower bound."""
    # optmat's named PPT form replaces solprop's former PPT (deleted, SEP 3 Phase 4b).
    rate = OptmatRate("Ar", 800e-9, model="ppt_ilkov")
    table = build_rate_table(
        rate, (1e9, 1e12), rtol=BUDGETS["rate_interpolation_log_atol"], use_cache=False
    )
    major, ratio = 1.001e9, 0.35
    direct = axis_moments(major, ratio, rate, 1024, edges=(rate.threshold,))
    capped = axis_moments(
        major, ratio, table, 1024, edges=(table.field_min, table.field_max)
    )
    return {
        "species": "Ar",
        "rate_parameters": rate.parameters,
        "major_V_per_m": major,
        "ellipticity": ratio,
        "table_field_min_V_per_m": table.field_min,
        "table_key": table.key,
        "direct_moments_s_inverse": direct.tolist(),
        "capped_moments_s_inverse": capped.tolist(),
        "relative_moment_change": ((direct - capped) / direct).tolist(),
        "meaning": "cutoff model difference; not a spline interpolation error",
    }


def averaging_experiments():
    """Compare quadrature and bicubic moment tables at independent points."""
    rows = []
    random = np.random.default_rng(3901)
    for species in ("He", "Ar"):
        # The ADK test oracle is solprop's former ADK; optmat's named PPT form
        # replaces its former PPT (both deleted, SEP 3 Phase 4b).
        for name, rate in (
            ("ADK", adk_oracle(species)),
            ("PPT", OptmatRate(species, 800e-9, model="ppt_ilkov")),
        ):
            started = perf_counter()
            table = build_rate_table(
                rate,
                (rate.threshold, 1e12),
                rtol=BUDGETS["rate_interpolation_log_atol"],
                use_cache=False,
            )
            rate_seconds = perf_counter() - started
            # Include the rate table's floor and ceiling.
            amplitudes = np.exp(
                random.uniform(
                    np.log(table.field_min * 1.001), np.log(2 * table.field_max), 32
                )
            )
            ratios = random.uniform(0, 1, 32)
            points = list(zip(amplitudes, ratios, strict=True)) + [
                (amplitude, ratio)
                for amplitude in (
                    table.field_min * (1 - 1e-8),
                    table.field_min * (1 + 1e-8),
                    table.field_min * 1.3,
                    table.field_max * (1 - 1e-8),
                    table.field_max * (1 + 1e-8),
                    table.field_max * 2,
                )
                for ratio in (0, 1e-10, 0.35, 1 - 1e-10, 1)
            ]
            edges = (table.field_min, table.field_max)
            reference = np.array(
                [axis_moments(a, q, table, 1024, edges=edges) for a, q in points]
            )

            def bounded_rate(field, model=rate, compiled=table):
                # Isolate interpolation from the existing table's explicit cutoffs.
                return np.where(
                    field < compiled.field_min,
                    0,
                    model(np.minimum(field, compiled.field_max)),
                )

            direct_reference = np.array(
                [axis_moments(a, q, bounded_rate, 1024, edges=edges) for a, q in points]
            )
            oracle_checks = []
            for amplitude, ratio in points[:8]:
                average, current, estimate = cartesian_oracle(
                    np.array([amplitude, 1j * ratio * amplitude]),
                    table,
                )
                moments = axis_moments(amplitude, ratio, table, 1024, edges=edges)
                actual = np.array(
                    [
                        moments[0] + ratio**2 * moments[1],
                        2 * moments[0],
                        2 * ratio * moments[1],
                    ]
                )
                expected = np.array(
                    [average, current[0].real * amplitude, current[1].imag * amplitude]
                )
                oracle_checks.append(
                    {
                        "relative_error": relative_moments(actual, expected),
                        "quad_error_estimate_s_inverse": estimate,
                    }
                )
            # The high-amplitude hold is intentional, not interpolation error.
            interior = [
                i
                for i, (a, _) in enumerate(points)
                if a <= table.field_max and a >= table.field_min * 1.001
            ]
            quadrature = []
            for order in BUDGETS["angular_orders"]:
                started = perf_counter()
                actual = np.array(
                    [axis_moments(a, q, table, order, edges=edges) for a, q in points]
                )
                quadrature.append(
                    {
                        "order": order,
                        "seconds": perf_counter() - started,
                        "relative_error": relative_moments(actual, reference),
                    }
                )
            # No hidden repair near threshold in this simple 2-D alternative.
            started = perf_counter()
            refined = [refined_moments(a, q, table, edges) for a, q in points]
            refined_seconds = perf_counter() - started
            alternatives = []
            for count in (17, 33, 65):
                started = perf_counter()
                x = np.linspace(
                    np.log(table.field_min), np.log(2 * table.field_max), count
                )
                q = np.linspace(0, 1, count)
                values = np.array(
                    [
                        [axis_moments(np.exp(a), b, table, 128, edges=edges) for b in q]
                        for a in x
                    ]
                )
                splines = [
                    interpolate.RectBivariateSpline(
                        x, q, np.log(np.maximum(values[:, :, k], 1e-300))
                    )
                    for k in range(2)
                ]
                build_seconds = perf_counter() - started
                inside = [
                    (a, b, i) for i, (a, b) in enumerate(points) if a >= table.field_min
                ]
                started = perf_counter()
                estimates = np.column_stack(
                    [
                        np.exp(
                            spline.ev(
                                np.log([a for a, _, _ in inside]),
                                [b for _, b, _ in inside],
                            )
                        )
                        for spline in splines
                    ]
                )
                alternatives.append(
                    {
                        "knots_per_axis": count,
                        "build_seconds": build_seconds,
                        "evaluation_seconds": perf_counter() - started,
                        "bytes": sum(spline.get_coeffs().nbytes for spline in splines)
                        + x.nbytes
                        + q.nbytes,
                        "relative_error": relative_moments(
                            estimates, reference[[i for _, _, i in inside]]
                        ),
                    }
                )
            rows.append(
                {
                    "species": species,
                    "model": name,
                    "rate_parameters": dict(rate.parameters),
                    "field_min": table.field_min,
                    "field_max": table.field_max,
                    "rate_table_key": table.key,
                    "rate_table_build_seconds": rate_seconds,
                    "rate_table_bytes": table.coefficients.nbytes
                    + table.log_rate.nbytes,
                    "rate_table_reported_error": table.error,
                    "off_knot_moment_interpolation_error": relative_moments(
                        reference[interior], direct_reference[interior]
                    ),
                    "points": len(points),
                    "recommended_refinement": {
                        "seconds": refined_seconds,
                        "maximum_order": max(order for _, order in refined),
                        "relative_error": relative_moments(
                            np.array([value for value, _ in refined]), reference
                        ),
                    },
                    "cartesian_oracle_checks": oracle_checks,
                    "threshold_rate_s_inverse": float(rate(rate.threshold)),
                    "lower_cutoff_field_margin": table.field_min / rate.threshold - 1,
                    "quadrature": quadrature,
                    "bicubic": alternatives,
                }
            )
    return rows
