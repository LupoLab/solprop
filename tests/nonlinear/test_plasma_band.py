# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The plasma column's band integration rule (SEP 18, section 7 items 2, 3, 6 and 7).

The trapezoid's own identities are in ``test_plasma.py``; these are the band rule's:
the interior Drude susceptibility across the band, the finite-window budget and its
boundary coefficient, the abrupt switch-on probes, the precursor's support, the
closed-form and energy ladders, the totals, and the range and undershoot of the
fraction.
"""

import math

import numpy as np
import pytest
from jaxtyping import TypeCheckError

from convergence_records import Column, Study, record
from plasma_cases import (
    CHARGE_MASS_RATIO,
    DENSITY,
    constant_field_reference,
    constant_rate,
    gaussian_pulse_and_derivative,
    power_rate,
    zero_rate,
)
from solprop.backends.numpy_ref.passes import (
    plasma_integration_diagnostics,
    plasma_integration_statistics,
)
from solprop.exceptions import GridError, MaterialError
from solprop.grid import FieldGrid
from solprop.nonlinear.ionisation import RunningIntegral, ionisation_fraction
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import PlasmaResponse
from solprop.physdata import Gas, ionisation_potential
from solprop.translate.plasma_integration import (
    design_running_integral,
)
from solprop.translate.responses import plasma_coefficients


@pytest.fixture(scope="module")
def grid():
    """The default field grid of section 2: 90 nm to 4 um, ffac 6."""
    return FieldGrid.from_limits(1.0, 800e-9, (90e-9, 4e-6), 400e-15)


@pytest.fixture(scope="module")
def rule(grid):
    """The band rule the default grid designs: K = 4."""
    return design_running_integral(grid, "band")


def unit_rule(rule, grid):
    """The same taps at unit sample spacing, for the dimensionless probes."""
    return RunningIntegral(
        "band", rule.taps, dt=1.0, band_edge=grid.omega_max_window * grid.dt_fine
    )


def rescaled(rule, grid, dt):
    """The default grid's taps at another spacing: the same theta_b, a finer window."""
    edge = grid.omega_max_window * grid.dt_fine / dt
    return RunningIntegral("band", rule.taps, dt=dt, band_edge=edge)


def twice(rule, values):
    """Two running integrals of `values` at unit spacing."""
    return rule.integrate(rule.integrate(values, 1.0), 1.0)


# ------------------------------------------------------- the Drude limit (item 2)

# A cosine pre-ionised tone, fitted on the interior (3K samples clear of each end) as a
# constant, a secular ramp and the oscillation, which is -offset times H^2; the bound
# on |H^2 - 1| is 2 eps + eps^2 for the certified eps. Measured |c + 1|: 1.47e-6,
# 3.59e-6 and 9.804e-6 at 0.5, 1.0 and 1.0999 omega_max against the bound 9.849e-6,
# with fit residuals below 1e-11 of the offset at 512 samples; the trapezoid's are
# 4.5e-2, 0.1775 and 0.2134. The ramp is the switch-on's (item 2's probes below).
FIT_ROUNDOFF = 1e-10


@pytest.mark.parametrize("top", [0.5, 1.0, 1.0999])
def test_the_band_rule_has_the_drude_susceptibility_across_the_band(grid, rule, top):
    dt = grid.dt_fine
    n = np.arange(512.0)
    theta = top * grid.omega_max * dt
    field = 1e9 * np.cos(theta * n)
    plasma = PlasmaResponse(Gas.Ar, zero_rate, preionisation=0.3, integration="band")
    got = plasma.evaluate(field[None, :], DENSITY, dt=dt, kind="field", integral=rule)
    offset = DENSITY * 0.3 * CHARGE_MASS_RATIO * 1e9 / (theta / dt) ** 2
    interior = slice(3 * rule.half_width, n.size - 3 * rule.half_width)
    basis = np.stack([np.ones_like(n), n, np.cos(theta * n), np.sin(theta * n)], 1)
    coefficients, *_ = np.linalg.lstsq(
        basis[interior], got[0, interior] / offset, rcond=None
    )
    eps = rule.certified_error
    assert abs(coefficients[2] + 1.0) <= 2 * eps + eps**2 + FIT_ROUNDOFF
    # Symmetric taps add no phase: no sine.
    assert abs(coefficients[3]) < FIT_ROUNDOFF
    fitted = basis[interior] @ coefficients
    assert np.max(np.abs(fitted - got[0, interior] / offset)) < FIT_ROUNDOFF


# The finite-window budget (section 4.4): a band-limited periodic field whose Gaussian
# envelope is lambda of its peak at the window's start, against the exact causal double
# integral of that trigonometric polynomial, on 4096 samples at theta = pi/3. Measured
# whole-window error: 3.75e-6 at lambda = 1e-8 (the multiplier level), and 3.34e-5,
# 3.01e-3 at 1e-6 and 1e-4, so kappa = 33 and 30; the trapezoid's is 0.18 throughout.
WINDOW = 4096


def _exact_double(spectrum, theta):
    """Causal double integral from n = 0 of the trigonometric polynomial (dt = 1)."""
    n = np.arange(WINDOW, dtype=float)
    out = (spectrum[0].real / WINDOW) * n**2 / 2
    for k in np.flatnonzero(spectrum[1:]) + 1:
        amplitude = 2 * spectrum[k] / WINDOW
        omega = theta[k]
        term = (
            amplitude / (1j * omega) * ((np.exp(1j * omega * n) - 1) / (1j * omega) - n)
        )
        out = out + term.real
    return out


def boundary_error(rule, grid, level):
    """Whole-window error of two band integrals of a tone starting at `level`."""
    n = np.arange(WINDOW, dtype=float)
    width = (WINDOW / 2) / math.sqrt(-2 * math.log(level))
    tone = np.exp(-((n - WINDOW / 2) ** 2) / (2 * width**2)) * np.cos(math.pi / 3 * n)
    spectrum = np.fft.rfft(tone)
    theta = 2 * math.pi * np.fft.rfftfreq(WINDOW)
    spectrum[theta > grid.omega_max_window * grid.dt_fine] = 0.0
    values = np.fft.irfft(spectrum, WINDOW)
    exact = _exact_double(spectrum, theta)
    return float(np.max(np.abs(twice(rule, values) - exact)) / np.max(np.abs(exact)))


PLASMA_BAND = Study(
    name="plasma_band",
    title="The plasma column's band rule: boundary budget and ladders",
    case={
        "boundary": "two band integrals (the default grid's K = 4 taps) of a "
        "band-limited tone at theta = pi/3, 4096 samples, whose envelope is lambda of "
        "its peak at the window's start, against the exact causal double integral",
        "closed form": "argon, a constant 1e9 V/m field and a constant 1e14 1/s rate "
        "over 10 fs, against the closed-form polarisation, with the default grid's "
        "taps at every rung (theta_b fixed)",
        "energy identity": "the work the field does on the medium against Ip times "
        "the electrons, on a Gaussian pulse, He minus Ar sharing one rate, with the "
        "default grid's taps at every rung",
    },
    axis={"name": "rung", "unit": "1", "label": "lambda or samples in the window"},
    columns=(
        Column(
            name="deviation",
            unit="1 (relative)",
            why="sums of a few thousand terms, compared well above their round-off; "
            "atol is that floor, about 8193 eps",
            rtol=1e-6,
            atol=2e-12,
        ),
        Column(
            name="ratio",
            unit="1",
            why="the deviation over lambda on the boundary series (kappa), its fall "
            "per doubling on the ladders; null on a ladder's coarsest rung; rtol "
            "covers the finest closed-form rung, about 1e4 eps above round-off, "
            "whose last bits differ between platforms",
            rtol=1e-3,
        ),
    ),
)


def test_the_whole_window_error_is_the_boundary_budget(request, grid, rule):
    unit = unit_rule(rule, grid)
    levels = (1e-8, 1e-6, 1e-4)
    errors = [boundary_error(unit, grid, level) for level in levels]
    # At 1e-8 the start-up term is below the multiplier tolerance.
    assert errors[0] <= 1e-5
    # kappa, the boundary coefficient of the budget eps + kappa lambda: about 30.
    for level, error in zip(levels[1:], errors[1:], strict=True):
        assert 20.0 < error / level < 45.0
    trapezoid = RunningIntegral.trapezoid()
    assert boundary_error(trapezoid, grid, 1e-8) > 0.1
    record(
        PLASMA_BAND,
        request=request,
        rows=[
            {
                "series": "boundary",
                "axis": f"{level:.0e}",
                "deviation": error,
                "ratio": error / level,
            }
            for level, error in zip(levels, errors, strict=True)
        ],
    )


# The review's abrupt-start probes (section 4.4's table): the ramp coefficient per unit
# tone of two integrals at theta = pi/3 over 512 samples, fitted on 40 < n < 470. No
# local closure can meet the tolerance for them, so they are recorded values, not
# accuracy claims.
@pytest.mark.parametrize(
    ("start", "ramp"),
    [("cosine", -0.025408548708), ("sine", 1.712861332e-6), ("constant", 0.0)],
)
def test_the_abrupt_switch_on_probes_keep_their_recorded_ramps(grid, rule, start, ramp):
    n = np.arange(512.0)
    theta = math.pi / 3
    values, exact = {
        "cosine": (np.cos(theta * n), (1 - np.cos(theta * n)) / theta**2),
        "sine": (np.sin(theta * n), (theta * n - np.sin(theta * n)) / theta**2),
        "constant": (np.ones_like(n), n**2 / 2),
    }[start]
    error = twice(unit_rule(rule, grid), values) - exact
    interior = (n > 40) & (n < 470)
    slope = np.polyfit(n[interior], error[interior], 1)[0]
    assert slope == pytest.approx(ramp, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize(
    ("start", "current", "polarisation"),
    [("cosine", 0.0446, 1.51e-4), ("sine", 8.04e-3, 3.41e-5)],
)
def test_the_precursor_is_confined_to_the_filter_s_reach(
    grid, rule, start, current, polarisation
):
    """Section 4.6: each band integral reads K samples ahead, and no further."""
    unit = unit_rule(rule, grid)
    reach = unit.half_width
    n = np.arange(512.0)
    switch = 200
    shape = np.cos if start == "cosine" else np.sin
    values = np.where(n >= switch, shape(math.pi / 3 * (n - switch)), 0.0)
    first = unit.integrate(values, 1.0)
    second = unit.integrate(first, 1.0)
    # Exactly nothing before the reach: the ends see a zero integrand.
    # The sine is zero at the switch, so its first non-zero sample is one later.
    onset = switch if start == "cosine" else switch + 1
    assert np.flatnonzero(first)[0] == onset - reach
    assert np.flatnonzero(second)[0] == onset - 2 * reach
    early_first = np.max(np.abs(first[:switch])) / np.max(np.abs(first))
    early_second = np.max(np.abs(second[:switch])) / np.max(np.abs(second))
    assert early_first == pytest.approx(current, rel=1e-2)
    assert early_second == pytest.approx(polarisation, rel=1e-2)


# ------------------------------------------------------ ladders (section 7 item 3)


def test_the_band_closed_form_converges_at_least_at_second_order(request, grid, rule):
    """Measured 8.67e-11, 1.61e-11 and 3.34e-12: a thousand times the trapezoid's
    9.45e-8, 2.36e-8 and 5.91e-9, falling by 5.4 and 4.8 per doubling."""
    amplitude, rate, duration = 1e9, 1e14, 1e-14
    plasma = PlasmaResponse(Gas.Ar, constant_rate(rate), integration="band")
    potential = ionisation_potential(Gas.Ar)

    def deviation(exponent):
        t = np.linspace(0.0, duration, 2**exponent + 1)
        dt = float(t[1] - t[0])
        field = np.full(t.size, amplitude)
        got = plasma.evaluate(
            field[None, :], 1.0, dt=dt, kind="field", integral=rescaled(rule, grid, dt)
        )[0]
        _, expected = constant_field_reference(t, amplitude, rate, potential, 0.0)
        return float(np.max(np.abs(got - expected)) / np.max(np.abs(expected)))

    deviations = [deviation(k) for k in (10, 11, 12)]
    assert deviations[0] < 1e-9
    assert deviations[0] / deviations[1] > 3.6
    # The finest rung is within a factor of two of the sums' round-off floor.
    assert deviations[1] / deviations[2] > 3.0
    _record_ladder(request, "closed form", (1025, 2049, 4097), deviations)


def band_energy_deviation(grid, rule, n_samples):
    """Relative deviation of the loss energy from Ip times the electrons (band)."""
    _, dt, field, derivative = gaussian_pulse_and_derivative(n_samples)
    integral = rescaled(rule, grid, dt)
    rate = power_rate()

    def work(gas):
        plasma = PlasmaResponse(gas, rate, integration="band")
        polarisation = plasma.evaluate(
            field[None, :], DENSITY, dt=dt, kind="field", integral=integral
        )[0]
        return -float(np.trapezoid(derivative * polarisation, dx=dt)) / DENSITY

    fraction = PlasmaResponse(Gas.Ar, rate, integration="band").ionised_fraction(
        field[None, :], dt=dt, kind="field", integral=integral
    )
    electrons = float(fraction[-1] - fraction[0])
    potentials = ionisation_potential(Gas.He) - ionisation_potential(Gas.Ar)
    return abs((work(Gas.He) - work(Gas.Ar)) / (potentials * electrons) - 1.0)


def test_the_band_energy_identity_converges_at_second_order(request, grid, rule):
    """Measured 4.39e-8, 1.11e-8 and 2.77e-9 (the trapezoid's: 9.9e-5, 2.5e-5,
    6.2e-6): the by-parts work is a plain trapezoid on both rules, so the identity
    stays second order, two thousand times smaller."""
    deviations = [band_energy_deviation(grid, rule, 2**k + 1) for k in (13, 14, 15)]
    assert deviations[0] < 1e-7
    assert 3.6 < deviations[0] / deviations[1] < 4.4
    assert 3.6 < deviations[1] / deviations[2] < 4.4
    _record_ladder(request, "energy identity", (8193, 16385, 32769), deviations)


def _record_ladder(request, series, counts, deviations):
    """Record one ladder of the band rule."""
    record(
        PLASMA_BAND,
        request=request,
        rows=[
            {
                "series": series,
                "axis": count,
                "deviation": deviation,
                "ratio": None if index == 0 else deviations[index - 1] / deviation,
            }
            for index, (count, deviation) in enumerate(
                zip(counts, deviations, strict=True)
            )
        ],
    )


# ------------------------------------------------------------ totals (item 6)


def test_the_filtered_total_is_the_trapezoid_s_for_a_vanishing_integrand(grid, rule):
    """C(0) = 1: an integrand zero within 2K samples of both ends keeps its total."""
    n = np.arange(400.0)
    values = np.exp(-(((n - 200) / 30) ** 2)) * np.cos(0.9 * n)
    values[: 2 * rule.half_width] = 0.0
    values[-2 * rule.half_width :] = 0.0
    unit = unit_rule(rule, grid)
    total = unit.integrate(values, 1.0)[-1]
    plain = RunningIntegral.trapezoid().integrate(values, 1.0)[-1]
    # A sum of 400 terms of order one, weighted by taps of sum one.
    assert total == pytest.approx(plain, abs=400 * 4 * np.finfo(float).eps)


# ---------------------------------------------------- range and undershoot (item 7)


def test_an_isolated_rate_spike_is_floored_and_counted(grid, rule):
    """Section 4.5: one sampled spike with dt W = 1 undershoots by 0.166, the fraction
    stays at or above its initial value, and the final exposure is still exactly one,
    because the floor is pointwise and accumulates nothing."""
    unit = unit_rule(rule, grid)
    rate = np.zeros(400)
    rate[200] = 1.0
    exposure = unit.integrate(rate, 1.0)
    steps = np.diff(exposure)
    undershoot = -steps.min() / steps.max()
    assert undershoot == pytest.approx(0.16622, rel=1e-4)
    assert exposure[-1] == pytest.approx(1.0, rel=1e-15)
    fraction = ionisation_fraction(
        lambda field_amplitude: np.asarray(field_amplitude, dtype=float),
        rate,
        1.0,
        preionisation=0.0,
        integral=unit,
    )
    assert fraction.min() == 0.0
    assert np.all(fraction < 1.0)
    assert fraction[-1] == pytest.approx(1.0 - math.exp(-1.0), rel=1e-15)


def test_the_diagnostics_name_the_spike(grid, rule):
    """The numpy_ref statistics pass finds the same undershoot on a compiled channel.

    One sample of a strong field in a quiet window: its tabulated rate, times dt, sets
    the exposure the spike leaves, and the filter's undershoot is the spike's 0.166
    whatever that is.
    """
    coefficients = plasma_coefficients(
        PlasmaResponse(Gas.Ar, OptmatRate(Gas.Ar, 800e-9)), (1e9, 1e11), z_max=1.0
    )
    field = np.zeros((1, 1, 400))
    field[..., 200] = 5e10
    parts = plasma_integration_diagnostics(
        field, (coefficients,), (0.0,), grid.dt_fine, rule
    )
    undershoot, _, boundary = plasma_integration_statistics(parts)
    defect, freed = parts[2], parts[3]
    assert undershoot == pytest.approx(0.16622, rel=1e-4)
    assert freed > 0.0
    assert boundary == 0.0
    # The loss current pays at the spike alone, the fraction moves around it.
    assert defect > 0.0


def test_the_band_fraction_stays_in_range_on_an_ionising_pulse(grid, rule):
    """f in [f0, 1) and non-decreasing to within the recorded undershoot."""
    dt = grid.dt_fine
    t = (np.arange(4096) - 2048) * dt
    field = 6e10 * np.exp(-((t / 1.5e-15) ** 2)) * np.cos(2.35e15 * t)
    plasma = PlasmaResponse(Gas.He, power_rate(), preionisation=0.1, integration="band")
    fraction = plasma.ionised_fraction(
        field[None, :], dt=dt, kind="field", integral=rule
    )
    # 1 - 0.9 exp(-0) is 0.1 to within its own rounding.
    assert fraction.min() >= 0.1 - 2e-16
    assert fraction.max() < 1.0
    steps = np.diff(fraction)
    assert -steps.min() <= 1e-3 * steps.max()


# ------------------------------------------------------------- the interface


def test_the_band_rule_without_its_integral_is_refused(grid):
    plasma = PlasmaResponse(Gas.Ar, zero_rate, integration="band")
    field = np.zeros((1, 16))
    with pytest.raises(GridError, match="design_running_integral"):
        plasma.evaluate(field, DENSITY, dt=grid.dt_fine, kind="field")
    with pytest.raises(GridError, match="design_running_integral"):
        plasma.ionised_fraction(field, dt=grid.dt_fine, kind="field")


def test_an_integral_for_another_rule_or_spacing_is_refused(grid, rule):
    field = np.zeros((1, 16))
    legacy = PlasmaResponse(Gas.Ar, zero_rate, integration="trapezoid")
    with pytest.raises(GridError, match="asks for integration='trapezoid'"):
        legacy.evaluate(field, DENSITY, dt=grid.dt_fine, kind="field", integral=rule)
    band = PlasmaResponse(Gas.Ar, zero_rate, integration="band")
    with pytest.raises(GridError, match="designed for dt"):
        band.evaluate(
            field, DENSITY, dt=grid.dt_fine * 1.001, kind="field", integral=rule
        )


def test_an_unknown_integration_rule_is_refused():
    # The runtime type hook refuses the literal first under the test session; without
    # it the response's own check names the two rules.
    with pytest.raises((TypeCheckError, MaterialError), match="band"):
        PlasmaResponse(Gas.Ar, zero_rate, integration="simpson")  # type: ignore[arg-type]
