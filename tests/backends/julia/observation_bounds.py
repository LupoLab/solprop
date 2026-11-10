# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bound same-field scalar envelope statistics, separately from propagation.

The fixture has one mode and one polarisation, a constant unit power weight and
normal-scale binary64 data. Each inverse transform has the declared componentwise
envelope ``gamma(32*n_t)*sum(abs(spectrum))/n_t``. The linear operation count is
deliberately larger than the FFT's logarithmic depth; it includes normalization,
twiddle rounding and pointwise setup. This is a CPU transform qualification
hypothesis, not a bound for an arbitrary FFT library or reduced precision.

Both flux operands may be transformed independently. Their errors are propagated
through the real Hermitian product, maxima, and the actual linear interpolation of
half-maximum crossings. Width fixtures must certify the peak, every threshold-side
decision and both local slopes before either backend's statistic is inspected.
Exact rational arithmetic prevents the bound calculation rounding inward; only its
final positive result is rounded upwards. No measured discrepancy sets a constant.

The reduction model is Higham's ``gamma_k = k*u/(1-k*u)``, with binary64
``u=2**-53`` (Accuracy and Stability of Numerical Algorithms, 2nd ed., sections 3.1
and 24.1, https://doi.org/10.1137/1.9780898718027). The width perturbation inequality
is derived beside its implementation. These are same-input rounding bounds, not
adaptive accuracy envelopes or bounds for material population scans.
"""

import math
from fractions import Fraction

import numpy as np
from jaxtyping import Complex, Float

from solprop.grid import TimeGrid
from solprop.stats import StatsSpec
from solprop.translate.problem import PropagationProblem

__all__ = ["scalar_envelope_bounds"]

_NAMES = frozenset(
    {
        "omega0",
        "energy",
        "energy_window",
        "peak_power",
        "peak_power_window",
        "peak_power_total",
        "fwhm_t",
        "fwhm_t_outer",
    }
)


def _gamma(operations: int) -> Fraction:
    assert 0 < operations < 2**53
    return Fraction(operations, 2**53 - operations)


def _upper(value: Fraction) -> float:
    result = math.nextafter(float(value), math.inf)
    assert value > 0 and math.isfinite(result) and result > 0
    return result


def _squares(spectrum):
    return [
        Fraction(float(value.real)) ** 2 + Fraction(float(value.imag)) ** 2
        for value in spectrum
    ]


def _spectral_bounds(spectrum, omega, dt):
    powers = _squares(spectrum)
    total = sum(powers, start=Fraction())
    assert total > 0, "the fixture must carry nonzero spectral energy"
    # Refuse destructive underflow/overflow in the density route used by NumPy.
    # Its common dt^2/(2*pi) factor cancels from the exact centroid, but not from
    # the rounded individual densities that the reduction actually receives.
    density_scale = Fraction(dt) ** 2 / (2 * Fraction(math.pi))
    tiny, largest = map(Fraction, (np.finfo(float).tiny, np.finfo(float).max))
    assert all(tiny < density_scale * power < largest for power in powers if power)
    moment = sum(
        (
            abs(Fraction(float(frequency))) * power
            for frequency, power in zip(omega, powers, strict=True)
        ),
        start=Fraction(),
    )
    assert max(total, moment) < largest
    assert density_scale * max(total, moment) < largest
    # Each energy computation has <=n additions, abs2 and Parseval scaling.
    energy = 2 * _gamma(spectrum.size + 8) * Fraction(dt) * total / spectrum.size
    # For D=sum|a|^2 and A=sum|omega| |a|^2, each numerator/denominator has
    # errors <=g*A and <=g*D. Thus |N'/D' - N/D| <=2*g*A/(D*(1-g)); include
    # division rounding and both implementations. The signed moment may cancel.
    reduction = _gamma(spectrum.size + 16)
    centroid = (
        2
        * (2 * reduction + _gamma(1) * (1 + reduction))
        * moment
        / (total * (1 - reduction))
    )
    return energy, centroid


def _temporal_envelope(spectrum):
    # |Re|+|Im| bounds |a| without needing a rounded square root. The stated
    # componentwise FFT bound gives at most twice that bound in complex modulus.
    norm = sum(
        (
            abs(Fraction(float(value.real))) + abs(Fraction(float(value.imag)))
            for value in spectrum
        ),
        start=Fraction(),
    )
    assert norm > 0, "each temporal fixture/window must carry nonzero energy"
    one_fft = 2 * _gamma(32 * spectrum.size) * norm / spectrum.size
    # This transform locates/certifies crossings; it is not either backend's
    # observed statistic. Each candidate operand differs from it by <=2*one_fft.
    centred = np.fft.ifft(spectrum)
    assert np.all(np.isfinite(centred))
    difference = 2 * one_fft
    powers, errors = [], []
    for value in centred:
        real, imag = Fraction(float(value.real)), Fraction(float(value.imag))
        amplitude = abs(real) + abs(imag)
        powers.append(real**2 + imag**2)
        # |Re(a'*conj(b'))-|a|^2| <=2*|a|*d+d^2, with both operands within d.
        # Eight further real roundings cover either abs2 or the Hermitian pair.
        errors.append(
            2 * amplitude * difference
            + difference**2
            + _gamma(8) * (amplitude + difference) ** 2
        )
    assert max(powers) > max(errors) > 0
    assert max(powers) + max(errors) < Fraction(np.finfo(float).max)
    return powers, errors


def _crossing_bound(time, powers, errors, level, level_error, pair):
    outside, inside = pair
    denominator = powers[inside] - powers[outside]
    denominator_error = errors[inside] + errors[outside]
    assert denominator > denominator_error > 0, "uncertified crossing slope"
    numerator = level - powers[outside]
    quotient = numerator / denominator
    assert 0 < quotient < 1
    first, second = Fraction(float(time[outside])), Fraction(float(time[inside]))
    span = abs(second - first)
    # With q=A/B, |delta q| <=(delta A+|q|*delta B)/(B-delta B).
    # The two real power samples and the half-maximum level all may move.
    variation = (
        span
        * (level_error + errors[outside] + quotient * denominator_error)
        / (denominator - denominator_error)
    )
    # Subtractions, multiply, divide and final add in x1+(H-y1)*(x2-x1)/(y2-y1).
    # The perturbed exact quotient remains in (0,1) by the certified brackets.
    rounding = _gamma(8) * (abs(first) + 2 * span)
    return variation + rounding


def _width_bound(time, powers, errors, *, outer):
    apex = max(range(len(powers)), key=powers.__getitem__)
    peak = powers[apex]
    assert all(
        peak - power > errors[apex] + errors[index]
        for index, power in enumerate(powers)
        if index != apex
    ), "the half-width fixture has no certified unique peak"
    level, level_error = peak / 2, max(errors) / 2
    assert all(
        abs(power - level) > error + level_error
        for power, error in zip(powers, errors, strict=True)
    ), "a sample can cross the half-maximum threshold under the rounding envelope"
    above = [index for index, power in enumerate(powers) if power > level]
    low, high = above[0], above[-1]
    if not outer:
        low = high = apex
        while low > 0 and powers[low - 1] > level:
            low -= 1
        while high + 1 < len(powers) and powers[high + 1] > level:
            high += 1
    assert 0 < low <= high < len(powers) - 1, (
        "the width must have two interior brackets"
    )
    endpoints = _crossing_bound(
        time, powers, errors, level, level_error, (low - 1, low)
    ) + _crossing_bound(time, powers, errors, level, level_error, (high + 1, high))
    coordinate = max(abs(Fraction(float(value))) for value in time)
    # Both widths differ from the same exact nominal crossing difference. Bound
    # its final subtraction too, including each endpoint's prior uncertainty.
    return 2 * (endpoints + _gamma(1) * (2 * coordinate + endpoints))


def scalar_envelope_bounds(
    problem: PropagationProblem,
    grid: TimeGrid,
    spec: StatsSpec,
    field: Complex[np.ndarray, "1 1 n_omega"],
) -> dict[str, Float[np.ndarray, "*stat_shape"]]:
    """Return absolute comparison bounds for the same scalar envelope spectrum.

    Parameters
    ----------
    problem : PropagationProblem
        One-mode, one-polarisation binary64 problem with constant unit power weight.
    grid : TimeGrid
        Matching coarse envelope grid, with times in seconds and frequencies in rad/s.
    spec : StatsSpec
        Requested ordinary scalar statistics and hard half-open frequency windows.
        An unavailable ``quadrature_error`` entry is omitted from the returned bounds.
    field : numpy.ndarray
        Identical internal spectrum supplied to both backends, shape ``(1,1,n_omega)``.

    Returns
    -------
    dict of str to numpy.ndarray
        Positive finite absolute bounds, with each statistic's public result shape.
        Compute these before comparing the backend results. They do not include any
        propagation error; widths refuse rather than hide an uncertified crossing.

    Raises
    ------
    AssertionError
        If the fixture falls outside the scalar unit-weight, finite-range or width
        conditioning hypotheses. Material populations and other names are unsupported.
    """
    assert problem.grid_kind == grid.kind == "envelope"
    assert problem.precision.state == problem.precision.source == "double"
    assert problem.shape == field.shape == (1, 1, grid.n_t)
    assert field.dtype == np.complex128 and np.all(np.isfinite(field))
    assert math.isfinite(grid.dt) and grid.dt > 0
    assert grid.dt == problem.grids.dt
    assert np.array_equal(grid.omega, problem.grids.omega)
    assert np.all(np.isfinite(grid.t)) and np.all(np.diff(grid.t) > 0)
    weight = problem.linear.power_weight_constant
    assert weight is not None and np.all(weight == 1.0)
    assert problem.linear.power_weight_expansion is None
    names = set(spec.names) - {"quadrature_error"}
    assert names <= _NAMES
    spectrum = field[0, 0]
    energy, centroid = _spectral_bounds(spectrum, grid.omega, grid.dt)
    result = {}
    if "energy" in names:
        result["energy"] = np.full((1, 1), _upper(energy))
    if "omega0" in names:
        result["omega0"] = np.full((1, 1), _upper(centroid))
    temporal = {"peak_power", "peak_power_total", "fwhm_t", "fwhm_t_outer"}
    if names & temporal:
        powers, errors = _temporal_envelope(spectrum)
        peak = _upper(2 * max(errors))
        for name in names & {"peak_power", "peak_power_total"}:
            result[name] = np.full((1,) if name == "peak_power" else (), peak)
        for name in names & {"fwhm_t", "fwhm_t_outer"}:
            result[name] = np.asarray(
                _upper(
                    _width_bound(grid.t, powers, errors, outer=name == "fwhm_t_outer")
                )
            )
    if names & {"energy_window", "peak_power_window"}:
        assert spec.n_windows > 0
        window_energy, window_peak = [], []
        for low, high in spec.energy_windows:
            gated = spectrum * ((grid.omega >= low) & (grid.omega < high))
            if "energy_window" in names:
                value, _ = _spectral_bounds(gated, grid.omega, grid.dt)
                window_energy.append(_upper(value))
            if "peak_power_window" in names:
                _, errors = _temporal_envelope(gated)
                window_peak.append(_upper(2 * max(errors)))
        if "energy_window" in names:
            result["energy_window"] = np.asarray(window_energy).reshape(-1, 1, 1)
        if "peak_power_window" in names:
            result["peak_power_window"] = np.asarray(window_peak).reshape(-1, 1)
    return result
