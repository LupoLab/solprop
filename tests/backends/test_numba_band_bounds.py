# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The multipass plasma column against ``numpy_ref``, to absolute rounding bounds.

SEP 25 section 4.6: the column's integrands are signed and cancel over carrier cycles,
so a relative bound says nothing where the polarisation passes through zero. Each
sample is held instead to an **absolute** budget carried through the whole chain --
the rate, its filter, the exposure's scan, the fraction, the guard and the weight, the
free current's filter and scan, the current and the density -- with
:func:`bounds.stencil_bound` for every filter and :func:`bounds.prefix_bound` for every
scan, at the serial scan's depth ``n_to``. The budget is one computation's distance
from exact arithmetic; two computations differ by at most twice it. It is fixed from
the inputs before either side runs, never from the difference it then accepts.

The logarithm and exponential are assumed accurate to one epsilon relative. Their
errors are propagated through the stored cubic coefficients, including the
conditioning of the local logarithmic coordinate. This replaces SEP 25's fixed
16-epsilon rate allowance, which missed the tiny nonzero rate in the pre-ionised
oscillatory case. The current exposes its first sample; the historical outer
integral was exactly zero there. The stress inputs and production methods remain
unchanged. See SEP 27's implementation record for the approved correction.
"""

import numpy as np
import pytest

from bounds import EPS, assert_within, gamma, prefix_bound, stencil_bound
from legacy_geometry import LEGACY_GRID
from mixture_cases import mixture_case
from solprop.backends.numba_cpu import kernels
from solprop.backends.numba_cpu.backend import _channel_rate_arrays, _rate_arrays
from solprop.backends.numpy_ref import passes
from solprop.grid import FieldGrid
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import PlasmaResponse
from solprop.translate.plasma_integration import design_running_integral
from solprop.translate.responses import plasma_coefficients

GRID = FieldGrid.from_limits(
    1.0, 800e-9, (200e-9, 2e-6), 120e-15, ffac=6.0, **LEGACY_GRID
)
DENSITY = 2e25


def rule(name):
    """The grid's band rule, or the band machinery with the trapezoid's one tap."""
    if name == "K=0":
        return RunningIntegral(
            "band", np.ones(1), dt=GRID.dt_fine, band_edge=GRID.omega_max_window
        )
    return design_running_integral(GRID, "band")


def trapezoids(values):
    """The running trapezoid's increments, zero first."""
    out = np.zeros_like(values)
    out[..., 1:] = GRID.dt_fine * (values[..., 1:] + values[..., :-1]) / 2.0
    return out


def stencil_carry(taps, errors):
    """Bound how far a stencil moves input errors: ``3 sum |c_k| e`` around a sample."""
    taps = np.abs(np.asarray(taps))
    total = taps[0] * errors
    for k in range(1, taps.size):
        shifted = np.zeros_like(errors)
        shifted[..., k:] += errors[..., :-k]
        shifted[..., :-k] += errors[..., k:]
        # The reflection reads the end sample twice and another once.
        shifted[..., :k] += 3 * np.max(errors[..., : k + 1], axis=-1, keepdims=True)
        shifted[..., -k:] += 3 * np.max(errors[..., -k - 1 :], axis=-1, keepdims=True)
        total = total + taps[k] * shifted
    return total


def stencil_rounding(taps, values):
    """A stencil's own rounding, with its reflected samples' magnitudes."""
    half = len(taps) - 1
    if half == 0:
        return np.zeros_like(values)
    head = 2.0 * np.abs(values[..., :1]) + np.abs(values[..., 1 : half + 1][..., ::-1])
    tail = 2.0 * np.abs(values[..., -1:]) + np.abs(
        values[..., -half - 1 : -1][..., ::-1]
    )
    extended = np.concatenate([head, np.abs(values), tail], axis=-1)
    return stencil_bound(taps, extended)[..., half:-half] * 2.0


def scan(values, errors, taps):
    """Filter and scan `values` whose own errors are `errors`; return both bounds."""
    n_to = values.shape[-1]
    filtered = RunningIntegral("band", taps, dt=GRID.dt_fine, band_edge=1.0).filter(
        values
    )
    filtered_error = stencil_carry(taps, errors) + stencil_rounding(taps, values)
    increments = trapezoids(filtered)
    increment_error = np.zeros_like(increments)
    increment_error[..., 1:] = GRID.dt_fine * (
        (filtered_error[..., 1:] + filtered_error[..., :-1]) / 2
        + 2 * EPS * (np.abs(filtered[..., 1:]) + np.abs(filtered[..., :-1]))
    )
    total = np.cumsum(increments, axis=-1)
    return total, prefix_bound(increments, n_to, uncertainty=increment_error)


def rate_error_bound(real, table, rate):
    """Bound one evaluation of exp(cubic(log |E|)) from its actual operands.

    With normal field components, gamma(8) bounds the norm's relative error.
    Assuming one-epsilon log/exp, delta(log E) <= gamma(8) + eps |log E|.
    Forming x = log E - (log_min + index*d) adds gamma(3) times the sum
    of those absolute operands. The cubic contributes its derivative bound
    times delta(x), plus gamma(6) times its absolute Horner polynomial.
    Exponentiation turns delta(p) into expm1(delta(p)) relative error.

    Near an interval boundary, either adjacent stored polynomial can be
    selected. Their evaluated difference and both evaluation bounds cover
    any rounding discontinuity without assuming exact spline continuity.
    These stress inputs stay away from the table's discontinuous lower clip.
    No compiled result or observed discrepancy enters this calculation.
    """
    magnitude = np.sqrt(np.sum(real * real, axis=1))
    out = np.zeros_like(rate)
    uncertainty = gamma(8) * magnitude
    assert np.all(abs(magnitude - table.field_min) > uncertainty)
    assert np.all(abs(magnitude - table.field_max) > uncertainty)
    inside = (magnitude >= table.field_min) & (magnitude <= table.field_max)
    logarithm = np.log(magnitude[inside])
    log_error = gamma(8) + EPS * abs(logarithm)
    coordinate = (logarithm - table.log_field_min) / table.d_log_field
    coordinate_error = (
        log_error + gamma(2) * (abs(logarithm) + abs(table.log_field_min))
    ) / table.d_log_field
    last = table.coefficients.shape[1] - 1
    index = np.clip(coordinate.astype(np.intp), 0, last)

    def polynomial(indices):
        centre = table.log_field_min + indices * table.d_log_field
        local = logarithm - centre
        local_error = log_error + gamma(3) * (
            abs(logarithm) + abs(table.log_field_min) + abs(indices * table.d_log_field)
        )
        c = table.coefficients[:, indices]
        radius = abs(local) + local_error
        condition = (
            (abs(c[0]) * radius + abs(c[1])) * radius + abs(c[2])
        ) * radius + abs(c[3])
        derivative = (3 * abs(c[0]) * radius + 2 * abs(c[1])) * radius + abs(c[2])
        value = ((c[0] * local + c[1]) * local + c[2]) * local + c[3]
        return value, local_error * derivative + gamma(6) * condition

    value, base_error = polynomial(index)
    error = base_error.copy()
    for neighbour in (coordinate - coordinate_error, coordinate + coordinate_error):
        alternate = np.clip(np.floor(neighbour).astype(np.intp), 0, last)
        other_value, other_error = polynomial(alternate)
        error = np.maximum(
            error,
            np.where(
                alternate == index,
                base_error,
                abs(other_value - value) + other_error + base_error,
            ),
        )
    # Account for rounding of the reference exponential used to express the
    # absolute scale, and round the bound outward after its own arithmetic.
    out[inside] = rate[inside] * (np.expm1(error) + EPS) / (1 - EPS)
    return np.nextafter(out * (1 + gamma(32)), np.inf)


def budget(real, rate, plasma, initial, taps):
    """One computation's absolute error bound on the current, per sample."""
    rate_error = rate_error_bound(real, plasma.rate, rate)
    exposure, exposure_error = scan(rate, rate_error, taps)
    floored = np.maximum(exposure, 0.0)
    fraction = 1.0 - (1.0 - initial) * np.exp(-floored)
    # d f / d X = (1 - f0) exp(-X) <= 1; the map's own rounding is a few eps.
    fraction_error = exposure_error + 4 * EPS
    squared = np.sum(real * real, axis=1)
    positive = (rate > 0) & (squared > 0)
    inverse = np.zeros_like(squared)
    np.divide(1.0, squared, out=inverse, where=positive)
    work = plasma.ionisation_potential * rate * (1.0 - fraction)
    guard = work * inverse
    guard_error = plasma.ionisation_potential * inverse * (
        (1.0 - fraction) * rate_error + rate * fraction_error
    ) + 6 * EPS * np.abs(guard)
    weight = fraction * plasma.charge_mass_ratio
    weight_error = plasma.charge_mass_ratio * fraction_error + EPS * np.abs(weight)
    out = np.zeros_like(real)
    for component in range(real.shape[1]):
        field = real[:, component]
        magnitude = np.abs(field)
        integrand = weight * field
        integrand_error = magnitude * weight_error + EPS * np.abs(integrand)
        free, free_error = scan(integrand, integrand_error, taps)
        loss = guard * field
        current = free + loss
        current_error = (
            free_error
            + magnitude * guard_error
            + 2 * EPS * (np.abs(free) + np.abs(loss))
        )
        out[:, component] = DENSITY * current_error + EPS * DENSITY * np.abs(current)
    return out


def coefficients(gas):
    return plasma_coefficients(
        PlasmaResponse(gas, OptmatRate(gas, 800e-9)), (1e8, 3e11), z_max=1.0
    )


def stress(name, n_pol):
    """One of SEP 25's stress fields, three nodes, real, (3, n_pol, n_to)."""
    n_to = GRID.n_to
    t = (np.arange(n_to) - n_to // 2) * GRID.dt_fine
    carrier = 2.35e15
    out = np.zeros((3, n_pol, n_to))
    for node, scale in enumerate((0.6, 1.0, 1.3)):
        if name == "preionised_oscillation":
            # Keep the original input: tiny but nonzero rates at some samples.
            envelope = 1e9 * scale * np.ones(n_to)
        elif name == "zero_crossings":
            envelope = 8e10 * scale * np.exp(-((t / 8e-15) ** 2))
        elif name == "tiny_exposure":
            envelope = 2.2e10 * scale * np.exp(-((t / 5e-15) ** 2))
        elif name == "near_threshold":
            envelope = 1.0e10 * scale * np.exp(-((t / 20e-15) ** 2))
        elif name == "long_cancelling_prefix":
            envelope = 7e10 * scale * np.exp(-((t / 40e-15) ** 4))
        elif name == "leading_edge":
            envelope = 9e10 * scale * np.exp(-(((t - t[0]) / 6e-15) ** 2))
        elif name == "trailing_edge":
            envelope = 9e10 * scale * np.exp(-(((t - t[-1]) / 6e-15) ** 2))
        else:
            raise ValueError(name)
        # Floored far below anything physical, so no product reaches gradual
        # underflow, which needs an absolute term of its own (SEP 25 section 4.1).
        envelope = np.maximum(envelope, 1e-20 * envelope.max())
        out[node, 0] = envelope * np.cos(carrier * t + 0.3 * node)
        if n_pol == 2:
            out[node, 1] = 0.5 * envelope * np.sin(carrier * t + 0.3 * node)
    return out


CASES = [
    ("preionised_oscillation", 0.2),
    ("zero_crossings", 0.0),
    ("tiny_exposure", 0.0),
    ("near_threshold", 0.0),
    ("long_cancelling_prefix", 0.05),
    ("leading_edge", 0.0),
    ("trailing_edge", 0.1),
]


@pytest.mark.parametrize(("case", "initial"), CASES)
@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("name", ["K=4", "K=0"])
def test_the_column_is_within_its_absolute_budget(
    case, initial, n_pol, name, record_property
):
    integral = rule(name)
    plasma = coefficients("Ar")
    real = stress(case, n_pol)
    rate = passes.ionisation_rate(real, plasma.rate)
    fraction, free = passes.plasma_scan(
        real, rate, plasma, preionisation=initial, dt=GRID.dt_fine, integral=integral
    )
    expected = DENSITY * passes._plasma_total_current(
        real,
        rate,
        fraction,
        free,
        plasma,
    )
    bound = 2.0 * budget(real, rate, plasma, initial, integral.taps)
    spec = kernels.ColumnSpec("field_real", True, False, False, False, n_pol, False)
    spec = spec._replace(band=True)
    rows = real.reshape(-1, GRID.n_to)
    out = np.zeros_like(rows)
    kernels.make_band_plasma_column(spec)(
        rows,
        out,
        _rate_arrays(plasma),
        integral.taps,
        plasma.charge_mass_ratio,
        plasma.ionisation_potential,
        DENSITY,
        initial,
        GRID.dt_fine,
        np.zeros((1, 2)),
        np.zeros((3, kernels.band_work_rows(spec), GRID.n_to)),
    )
    assert np.any(expected), "a nontrivial reference"
    margin = assert_within(
        out.reshape(expected.shape), expected, bound, elementwise=True, what=case
    )
    record_property("current_bound_fraction", margin)
    assert margin <= 1.0


def test_two_channels_are_within_their_budget():
    """A mixture: each channel's budget, weighted by its density, summed."""
    integral = rule("K=4")
    problem = mixture_case(kerr=False)[-1]
    channels = problem.responses.plasma_channels
    real = stress("zero_crossings", 2)
    densities, initial = (1.5e25, 0.5e25), (0.0, 0.05)
    expected = passes._plasma_channels_current(
        real, channels, densities, initial, GRID.dt_fine, integral
    )
    bound = np.zeros_like(real)
    for channel, density, start in zip(channels, densities, initial, strict=True):
        rate = channel.rate(np.sqrt(np.sum(real * real, axis=1)))
        bound += (
            2.0
            * budget(real, rate, channel, start, integral.taps)
            * (density / DENSITY)
        )
    spec = kernels.ColumnSpec("field_real", True, False, False, False, 2, False)
    spec = spec._replace(n_channels=2, band=True)
    storage, metadata = _channel_rate_arrays(problem)
    rows = real.reshape(-1, GRID.n_to)
    out = np.zeros_like(rows)
    kernels.make_band_plasma_column(spec)(
        rows,
        out,
        (storage, metadata),
        integral.taps,
        np.array([c.charge_mass_ratio for c in channels]),
        np.array([c.ionisation_potential for c in channels]),
        np.array(densities),
        np.array(initial),
        GRID.dt_fine,
        np.zeros((3, 4)),
        np.zeros((3, kernels.band_work_rows(spec), GRID.n_to)),
    )
    assert np.any(expected)
    margin = assert_within(
        out.reshape(expected.shape), expected, bound, elementwise=True
    )
    assert margin <= 1.0
