# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise prescribed histories and the discrete work/decomposition identities."""

import math

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.special import erf
from tests.backends.test_held_plasma import EPS, grid_case, rounding_budget, source

from solprop.translate.plasma_integration import design_running_integral


@pytest.mark.parametrize("rule_name", ["band", "trapezoid"])
@pytest.mark.parametrize("phase", [0.0, math.pi / 4, math.pi / 2])
def test_prescribed_ionisation_recovers_the_analytic_drift(rule_name, phase):
    grid = grid_case()
    rule = design_running_integral(grid, rule_name)
    times = grid.t_fine
    width, sigma, carrier = 4e-15, 0.7e-15, grid.omega0
    initial, freed = 0.2, 0.4
    gaussian = np.exp(-((times / width) ** 2))
    angle = carrier * times + phase
    # A0 = 1, and current in units of e^2/m_e: no dimensional constants enter
    # this prescribed-history test. E = -dA/dt in the interface convention.
    field = gaussian * (2 * times / width**2 * np.sin(angle) - carrier * np.cos(angle))
    fraction = initial + freed * (1 + erf(times / sigma)) / 2
    current = rule.integrate(fraction * field, grid.dt_fine)
    exact = (
        freed
        * width
        / math.sqrt(width**2 + sigma**2)
        * math.exp(-(carrier**2) * width**2 * sigma**2 / (4 * (width**2 + sigma**2)))
        * math.sin(phase)
    )
    # Poisson summation on fE, analytic in the strip |Im t| <= sigma.
    # |erf(x+iy)| <= 1 + 2|y|exp(y^2)/sqrt(pi); shifting the Fourier
    # contour by sigma bounds every alias by M exp(-|m| omega_s sigma).
    # Integrating |E(t+i sigma)| gives the explicit M below. The band
    # filter preserves DC; its recorded DC defect and absolute tap sum
    # account for coefficient rounding and the vanishing reflected tails.
    magnitude = (
        (initial + freed * (1 + math.e / math.sqrt(math.pi)))
        * math.exp((sigma / width) ** 2 + carrier * sigma)
        * (
            2
            + 2 * sigma * math.sqrt(math.pi) / width
            + carrier * width * math.sqrt(math.pi)
        )
    )
    alias_ratio = math.exp(-2 * math.pi * sigma / grid.dt_fine)
    alias = 2 * magnitude * alias_ratio / (1 - alias_ratio)
    tap_sum = rule.taps[0] + 2 * np.sum(rule.taps[1:])
    tap_norm = abs(rule.taps[0]) + 2 * np.sum(abs(rule.taps[1:]))
    absolute_integral = np.sum(abs(fraction * field)) * grid.dt_fine
    end = min(abs(times[0]), abs(times[-1])) - 2 * len(rule.taps) * grid.dt_fine
    tails = (
        grid.n_to
        * grid.dt_fine
        * tap_norm
        * (initial + freed)
        * (2 * abs(times[0]) / width**2 + carrier)
        * math.exp(-((end / width) ** 2))
    )
    bound = tap_norm * alias + tails + abs(tap_sum - 1) * absolute_integral
    bound += 32 * EPS * grid.n_to * tap_norm * absolute_integral
    assert 0 < bound < 1e-8
    assert abs(current[-1] - exact) <= bound
    if phase == 0:
        # No final drift, but the integrated transient leaves a polarisation.
        polarisation = rule.integrate(current, grid.dt_fine)
        assert abs(polarisation[-1]) > 100 * EPS * np.max(abs(polarisation))


@pytest.mark.parametrize("rule_name", ["band", "trapezoid"])
def test_residual_polarisation_edge_is_removed(rule_name):
    grid = grid_case()
    times = grid.t_fine
    sigma = 1e-15
    current = np.exp(-((times / sigma) ** 2))
    rule = design_running_integral(grid, rule_name)
    polarisation = rule.integrate(current, grid.dt_fine)
    active, crop = grid.active_mask, grid.crop_index[grid.active_mask]
    omega = grid.omega[active]
    exact = (
        (-1.0) ** crop
        * math.sqrt(math.pi)
        * sigma
        * np.exp(-((omega * sigma) ** 2) / 4)
        / (grid.dt_fine * 1j * omega)
    )
    # Gaussian tails at the plateau and aliases at Nyquist are far below rounding
    # for this resolved 2048-sample lattice. The two retained terms make that
    # assumption explicit, independent of either numerical source.
    distance = grid.t_collar.one_high
    tail = 4 * grid.n_to * math.exp(-((distance / sigma) ** 2)) / omega
    alias = 4 * grid.n_to * math.exp(-((np.pi * sigma / grid.dt_fine) ** 2) / 4) / omega
    allowance = tail + alias + rounding_budget(grid, current)[active]
    allowance += 16 * EPS * abs(exact)
    held = source(grid, current)[active]
    gated = np.fft.rfft(grid.t_window_fine * polarisation)[crop]
    assert np.all(abs(held - exact) <= allowance)
    assert np.linalg.norm(gated - held) > 1e3 * np.linalg.norm(allowance)


def test_compact_current_and_polarisation_have_no_collar_difference():
    grid = grid_case()
    sigma = 1e-15
    polarisation = np.exp(-((grid.t_fine / sigma) ** 2))
    current = -2 * grid.t_fine / sigma**2 * polarisation
    active = grid.active_mask
    crop, omega = grid.crop_index[active], grid.omega[active]
    gate, slope = grid.t_window_fine, grid.t_collar.window_derivative(grid.t_fine)
    # Analytic P and J=P' avoid conflating a running-integral closure defect
    # with the collar edge. Both Gaussian tails and all sampled aliases are
    # bounded before comparing the two source equations.
    distance = min(abs(grid.t_collar.one_low), grid.t_collar.one_high)
    tail = 4 * grid.n_to * (1 + distance / sigma) * math.exp(-((distance / sigma) ** 2))
    nyquist = np.pi * sigma / grid.dt_fine
    aliases = 4 * grid.n_to * (1 + nyquist**2) * math.exp(-(nyquist**2) / 4)
    edge = np.sum(abs(slope * current)) / omega**2
    rounding = rounding_budget(grid, current)[active]
    rounding += 16 * EPS * grid.n_to * np.linalg.norm(polarisation)
    allowance = tail + aliases + edge + rounding
    gated = np.fft.rfft(gate * polarisation)[crop]
    assert np.all(abs(source(grid, current)[active] - gated) <= allowance)


@pytest.mark.parametrize("rule_name", ["band", "trapezoid"])
def test_polarisation_form_closes_only_after_its_discrete_defect(rule_name):
    grid = grid_case()
    current = (1 + erf((grid.t_fine + 5e-15) / 1e-15)) / 2
    polarisation = design_running_integral(grid, rule_name).integrate(
        current, grid.dt_fine
    )
    active, crop = grid.active_mask, grid.crop_index[grid.active_mask]
    q = 1j * grid.omega[active]
    gate, slope = grid.t_window_fine, grid.t_collar.window_derivative(grid.t_fine)
    v = np.fft.rfft(gate * polarisation)[crop]
    x = np.fft.rfft(gate * current)[crop]
    y = np.fft.rfft(slope * polarisation)[crop]
    z = np.fft.rfft(slope * current)[crop]
    defect = q * v - x - y
    corrected_polarisation = v - y / q - z / q**2
    # This is a discrete algebraic identity, not an accuracy claim about P'=J.
    # Both expressions reuse the same FFT results; 16 rounded operations bound
    # each absolute operand through the divisions and final subtraction.
    allowance = 16 * EPS * (abs(v) + abs(y / q) + abs(z / q**2) + abs(x / q))
    assert np.all(
        abs(corrected_polarisation - source(grid, current)[active] - defect / q)
        <= allowance
    )
    assert np.linalg.norm(defect / q) > 0


@pytest.mark.parametrize("centre", [0.0, 45e-15])
def test_discrete_work_identity_includes_the_taper_and_window_terms(
    centre, record_property
):
    grid = grid_case()
    times, count, dt = grid.t_fine, grid.n_to, grid.dt_fine
    current = (1 + erf((times + 5e-15) / 1e-15)) / 2
    envelope = np.exp(-(((times - centre) / 3e-15) ** 2))
    if centre == 0:
        envelope[abs(times) >= grid.t_collar.one_high] = 0
    field = envelope * np.cos(grid.omega0 * times)
    # Remove DC with a profile of the same support: the plateau case remains
    # exactly zero on both tapers, unlike subtracting a constant everywhere.
    field -= envelope * (np.sum(field) / np.sum(envelope))
    gate = grid.t_window_fine
    edge = grid.t_collar.window_derivative(times) * current
    spectrum = np.fft.rfft(edge)
    omega = 2 * np.pi * np.fft.rfftfreq(count, dt)
    primitive_spectrum = np.zeros_like(spectrum)
    primitive_spectrum[1:] = spectrum[1:] / (1j * omega[1:])
    periodic = np.fft.irfft(primitive_spectrum, n=count)
    jump = dt * np.sum(edge)
    ordinary = periodic - periodic[0] + np.arange(count) / count * jump
    held_tail = -jump
    current_spectrum = np.fft.rfft(gate * current) - primitive_spectrum
    electric = np.fft.rfft(field)
    weights = np.full(electric.size, 2.0)
    weights[[0, -1]] = 1.0
    active = np.zeros(electric.size, dtype=bool)
    active[grid.crop_index[grid.active_mask]] = True
    work = (
        dt
        / count
        * np.sum(
            weights[active]
            * np.real(np.conj(electric[active]) * current_spectrum[active])
        )
    )
    out_of_band = (
        dt
        / count
        * np.sum(
            weights[~active]
            * np.real(np.conj(electric[~active]) * current_spectrum[~active])
        )
    )
    physical = dt * np.dot(field, current)
    taper = dt * np.dot(field, (gate - 1) * current)
    edge_work = dt * np.dot(field, ordinary)
    moment = dt * np.dot(field, np.arange(count) / count)
    expected = physical + taper - edge_work - held_tail * moment - out_of_band
    # Parseval plus five reductions, products and the two FFTs; absolute operand
    # sums include cancellation in work and the arbitrary primitive constant.
    allowance = (
        64
        * EPS
        * count
        * dt
        * np.sum(
            abs(field)
            * (abs(current) + abs(ordinary) + abs(held_tail) + abs(periodic[0]))
        )
    )
    assert allowance > 0
    assert abs(work - expected) <= allowance
    if centre == 0:
        assert taper == 0
        assert abs(edge_work) <= allowance
    for name, value in (
        ("work", work),
        ("physical", physical),
        ("taper", taper),
        ("edge_work", edge_work),
        ("window", held_tail * moment),
        ("out_of_band", out_of_band),
        ("rounding_allowance", allowance),
    ):
        record_property(name, float(value))


@pytest.mark.parametrize("rule_name", ["band", "trapezoid"])
@pytest.mark.parametrize("phase", [0.0, math.pi / 4, math.pi / 2])
def test_prescribed_history_switch_off_splits_into_drift_and_residual(  # noqa: PLR0915 - keep the a priori derivation beside its identity
    rule_name, phase, record_property
):
    """T4(a,c): the trailing affine P has two independently known amplitudes."""
    grid = grid_case()
    rule = design_running_integral(grid, rule_name)
    times, dt = grid.t_fine, grid.dt_fine
    width, sigma, carrier = 4e-15, 0.7e-15, grid.omega0
    initial, freed = 0.2, 0.4

    def integrand(time):
        angle = carrier * time + phase
        field = np.exp(-((time / width) ** 2)) * (
            2 * time / width**2 * np.sin(angle) - carrier * np.cos(angle)
        )
        fraction = initial + freed * (1 + erf(time / sigma)) / 2
        return fraction * field

    def moments(position):
        value = integrand(position * width) * width
        return np.asarray([value, -position * value])

    # Integration by parts gives H=int fE and B=-int t fE: after the pulse
    # P=Ht+B. The dimensionless quadrature avoids an absolute tolerance in
    # seconds swallowing B; its two coefficients have comparable scales.
    moment, error = quad_vec(moments, -12, 12, epsabs=1e-13, epsrel=1e-13)
    drift = (
        freed
        * width
        / math.sqrt(width**2 + sigma**2)
        * math.exp(-(carrier**2) * width**2 * sigma**2 / (4 * (width**2 + sigma**2)))
        * math.sin(phase)
    )
    residual = width * moment[1]
    values = integrand(times)
    tap_sum = rule.taps[0] + 2 * np.sum(rule.taps[1:])
    tap_norm = abs(rule.taps[0]) + 2 * np.sum(abs(rule.taps[1:]))
    # Poisson contour bound on fE and t*fE, in |Im t|<=sigma. Symmetric
    # filtering preserves both zeroth and first moments (apart from tap_sum).
    # A second trapezoid scan therefore has the same affine-tail intercept
    # as the sampled first moment; there is no fitted discretisation allowance.
    common = (initial + freed * (1 + math.e / math.sqrt(math.pi))) * math.exp(
        (sigma / width) ** 2 + carrier * sigma
    )
    norm = common * (
        2
        + 2 * sigma * math.sqrt(math.pi) / width
        + carrier * width * math.sqrt(math.pi)
    )
    first_norm = common * (
        math.sqrt(math.pi) * width
        + 4 * sigma
        + 2 * sigma**2 * math.sqrt(math.pi) / width
        + carrier * (width**2 + sigma * width * math.sqrt(math.pi))
    )
    ratio = math.exp(-2 * math.pi * sigma / dt)
    alias = 2 * ratio / (1 - ratio)
    absolute = dt * np.sum(abs(values))
    duration = grid.n_to * dt
    # The pulse is at least 7 widths from either taper; its Gaussian tail
    # and the independent quadrature's 12-width truncation enter additively.
    distance = (
        min(abs(grid.t_collar.one_low), grid.t_collar.one_high)
        - 2 * len(rule.taps) * dt
    )
    tail = (
        (initial + freed)
        * (2 + carrier * duration)
        * math.exp(-((distance / width) ** 2))
    )
    rounding = 64 * EPS * grid.n_to * tap_norm**2 * absolute
    drift_error = (
        tap_norm * norm * alias + tail + abs(tap_sum - 1) * absolute + rounding
    )
    residual_error = (
        tap_norm**2 * first_norm * alias
        + width * error
        + duration * (tail + abs(tap_sum**2 - 1) * absolute + rounding)
    )
    assert drift_error > 0 and residual_error > 0
    current = rule.integrate(values, dt)
    polarisation = rule.integrate(current, dt)
    trailing = times >= grid.t_collar.one_high
    leading = times <= grid.t_collar.one_low
    current_error = drift_error + abs(tap_sum - 1) * absolute
    polarisation_error = residual_error + duration * current_error
    assert np.max(abs(current[trailing] - drift)) <= current_error
    assert (
        np.max(abs(polarisation[trailing] - drift * times[trailing] - residual))
        <= polarisation_error
    )
    assert np.max(abs(current[leading])) <= current_error
    assert np.max(abs(polarisation[leading])) <= polarisation_error

    active, crop = grid.active_mask, grid.crop_index[grid.active_mask]
    q = 1j * grid.omega[active]
    gate, slope = grid.t_window_fine, grid.t_collar.window_derivative(times)
    v = np.fft.rfft(gate * polarisation)[crop]
    x = np.fft.rfft(gate * current)[crop]
    y = np.fft.rfft(slope * polarisation)[crop]
    defect = q * v - x - y
    trailing_slope = np.where(trailing, slope, 0)
    edge = np.fft.rfft(trailing_slope)[crop]
    moment_edge = np.fft.rfft(trailing_slope * times)[crop]
    drift_part = drift * (moment_edge / q + edge / q**2)
    residual_part = residual * edge / q
    expected = drift_part + residual_part + defect / q
    # The source's two edge operands amplify only the independently bounded
    # affine-tail errors. C/q is reported separately as the third scan's
    # closure defect, as required for both integration rules by T6.
    allowance = np.sum(abs(slope)) * (
        polarisation_error / abs(q) + current_error / abs(q) ** 2
    )
    allowance += rounding_budget(grid, current)[active]
    allowance += 32 * EPS * grid.n_to * (abs(v) + abs(expected) + abs(y / q))
    assert np.all(abs(v - source(grid, current)[active] - expected) <= allowance)
    record_property("drift", float(drift))
    record_property("residual_polarisation", float(residual))
    record_property("drift_switch_norm", float(np.linalg.norm(drift_part)))
    record_property("residual_switch_norm", float(np.linalg.norm(residual_part)))
    record_property("closure_norm", float(np.linalg.norm(defect / q)))
    assert np.linalg.norm(drift_part + residual_part) > 100 * np.linalg.norm(allowance)
