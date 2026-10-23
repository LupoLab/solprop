# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The band rule's compiled kernels against the reference passes (SEP 18, phase 2).

The band plasma column and the band statistics column are the reference's
arithmetic in a lagged traversal: the filter, the reflection's virtual samples, the
same running trapezoid. On one chunk of nodes they reproduce ``numpy_ref``'s passes
within the band column's rounding budget, ``64 (2K + 1) n_to eps`` relative (the
mixture column's, :func:`band_bound`), compiled or interpreted alike. These tests run
in the coverage lane's interpreted pass too (``NUMBA_DISABLE_JIT``), which is where the
kernels' lines are counted.
"""

import numpy as np
import pytest

from bounds import assert_within, stencil_bound
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

GRID = FieldGrid.from_limits(1.0, 800e-9, (200e-9, 2e-6), 60e-15, ffac=6.0)
"""A short default-oversampled window: K = 4, a few hundred samples."""

DENSITY = 2e25


def rules():
    """The band rule of the grid, and the band rule at K = 0."""
    band = design_running_integral(GRID, "band")
    flat = RunningIntegral(
        "band", np.ones(1), dt=GRID.dt_fine, band_edge=GRID.omega_max_window
    )
    return {"K=4": band, "K=0": flat}


def coefficients(gas):
    """One channel's compiled coefficients over the fields used here."""
    return plasma_coefficients(
        PlasmaResponse(gas, OptmatRate(gas, 800e-9)), (1e8, 2e11), z_max=1.0
    )


def fields(n_pol, analytic=False):
    """Three nodes of an ionising pulse, one of them weak, two components or one."""
    t = (np.arange(GRID.n_to) - GRID.n_to // 2) * GRID.dt_fine
    out = np.zeros((3, n_pol, GRID.n_to))
    for node, amplitude in enumerate((6e10, 9e10, 1.2e10)):
        envelope = amplitude * np.exp(-((t / 4e-15) ** 2))
        out[node, 0] = envelope * np.cos(2.35e15 * t)
        if n_pol == 2:
            out[node, 1] = 0.4 * envelope * np.sin(2.35e15 * t)
    if analytic:
        # The analytic family hands the kernel complex samples whose real part is
        # the field; the imaginary part must not reach the plasma.
        return out, out + 1j * np.roll(out, 3, axis=-1)
    return out, out


def scratch(spec, n_nodes):
    """Return the multipass column's per-node work rows."""
    return np.zeros((n_nodes, kernels.band_work_rows(spec), GRID.n_to))


def band_bound(rule):
    """Bound a band column against the reference passes, relative to its largest output.

    The mixture column's budget: 64 eps per fine sample per tap of the ``2K + 1``
    tap filter, for the exposure, the fraction's exponential and the two currents'
    running sums, compiled or interpreted (``math.exp`` against NumPy's).
    """
    width = 2 * rule.half_width + 1
    return 64 * width * GRID.n_to * np.finfo(float).eps


@pytest.mark.parametrize("name", ["K=4", "K=0"])
@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize("initial", [0.0, 0.1])
def test_the_band_column_is_the_reference(name, n_pol, analytic, initial):
    """Pre-ionised or not: un-ionised, the free-current windows are exact zeros."""
    rule = rules()[name]
    plasma = coefficients("He")
    real, samples = fields(n_pol, analytic)
    rate = passes.ionisation_rate(real, plasma.rate)
    fraction, free = passes.plasma_scan(
        real, rate, plasma, preionisation=initial, dt=GRID.dt_fine, integral=rule
    )
    expected = DENSITY * passes._plasma_total_current(
        real,
        rate,
        fraction,
        free,
        plasma,
    )
    family = "field_analytic" if analytic else "field_real"
    spec = kernels.ColumnSpec(family, True, False, False, False, n_pol, False)
    stashing = initial > 0.0
    spec = spec._replace(stash=stashing, band=True)
    out = np.zeros(real.reshape(-1, GRID.n_to).shape)
    stash = np.zeros((3, 2))
    work = scratch(spec, 3)
    kernels.make_band_plasma_column(spec)(
        samples.reshape(-1, GRID.n_to),
        out,
        _rate_arrays(plasma),
        rule.taps,
        plasma.charge_mass_ratio,
        plasma.ionisation_potential,
        DENSITY,
        initial,
        GRID.dt_fine,
        stash,
        work,
    )
    got = out.reshape(expected.shape)
    bound = band_bound(rule)
    scale = float(np.max(np.abs(expected)))
    assert scale > 0
    assert np.max(np.abs(got - expected)) <= bound * scale
    if stashing:
        assert stash[:, 0] == pytest.approx(fraction[:, -1], rel=bound)
        assert stash[:, 1] == pytest.approx(rate.max(axis=-1), rel=bound)
    else:
        # Structural: a column without the stash never writes it.
        assert not np.any(stash)


def test_the_band_column_carries_several_channels():
    rule = rules()["K=4"]
    problem = mixture_case(kerr=False)[-1]
    channels = problem.responses.plasma_channels
    real, _ = fields(2)
    densities, initial = (1.5e25, 0.5e25), (0.0, 0.05)
    expected = passes._plasma_channels_current(
        real, channels, densities, initial, GRID.dt_fine, rule
    )
    spec = kernels.ColumnSpec("field_real", True, False, False, False, 2, False)
    spec = spec._replace(n_channels=2, stash=True, band=True)
    storage, metadata = _channel_rate_arrays(problem)
    out = np.zeros(real.reshape(-1, GRID.n_to).shape)
    stash = np.zeros((3, 4))
    work = scratch(spec, 3)
    kernels.make_band_plasma_column(spec)(
        real.reshape(-1, GRID.n_to),
        out,
        (storage, metadata),
        rule.taps,
        np.array([c.charge_mass_ratio for c in channels]),
        np.array([c.ionisation_potential for c in channels]),
        np.array(densities),
        np.array(initial),
        GRID.dt_fine,
        stash,
        work,
    )
    # As the mixture's own column: the per-channel exponentials and the weighted sums
    # agree to their rounding.
    bound = band_bound(rule)
    scale = float(np.max(np.abs(expected)))
    assert np.max(np.abs(out.reshape(expected.shape) - expected)) <= bound * scale
    _, fractions = passes.plasma_channel_fractions(
        real, channels, initial, GRID.dt_fine, rule
    )
    assert stash[:, 0] == pytest.approx(fractions[0, :, -1], rel=bound)
    assert stash[:, 2] == pytest.approx(fractions[1, :, -1], rel=bound)


@pytest.mark.parametrize("name", ["K=4", "K=0"])
def test_the_band_statistics_column_is_the_reference(name):
    """The statistics column's fraction, largest rate and diagnostics."""
    rule = rules()[name]
    plasma = coefficients("Ar")
    real, _ = fields(2)
    spec = kernels.ColumnSpec("field_real", True, False, False, False, 2, False)
    partials = np.zeros((2 + kernels.BAND_DIAGNOSTICS, 3))
    rings = np.zeros((3, kernels.BAND_STATS_ROWS, GRID.n_to))
    kernels.make_band_stats_column(spec)(
        real.reshape(-1, GRID.n_to),
        _rate_arrays(plasma),
        rule.taps,
        0.2,
        GRID.dt_fine,
        partials,
        rings,
    )
    rate = passes.ionisation_rate(real, plasma.rate)
    fraction, _ = passes.plasma_scan(
        real, rate, plasma, preionisation=0.2, dt=GRID.dt_fine, integral=rule
    )
    bound = band_bound(rule)
    assert partials[0] == pytest.approx(fraction[:, -1], rel=bound)
    assert partials[1] == pytest.approx(rate.max(axis=-1), rel=bound)
    expected = passes.plasma_integration_diagnostics(
        real, (plasma,), (0.2,), GRID.dt_fine, rule
    )
    measured = partials[2:].max(axis=1)
    # The fraction-side parts are formed by the same operations, to the column's
    # rounding. The exposure's steps are the increments themselves here and
    # differences of the cumulative sum in the reference, and the work defect's sums
    # run in another order (a pairwise trapezoid there), so those agree to theirs.
    assert measured[3:] == pytest.approx(expected[3:], rel=bound)
    assert measured[:2] == pytest.approx(expected[:2], rel=1e-9)
    assert measured[2] == pytest.approx(expected[2], rel=1e-3, abs=1e-15)


def test_the_band_statistics_column_carries_several_channels():
    rule = rules()["K=4"]
    problem = mixture_case(kerr=False)[-1]
    channels = problem.responses.plasma_channels
    real, _ = fields(1)
    spec = kernels.ColumnSpec("field_real", True, False, False, False, 1, False)
    spec = spec._replace(n_channels=2)
    storage, metadata = _channel_rate_arrays(problem)
    partials = np.zeros((4 + kernels.BAND_DIAGNOSTICS, 3))
    rings = np.zeros((3, kernels.BAND_STATS_ROWS, GRID.n_to))
    kernels.make_band_stats_column(spec)(
        real.reshape(-1, GRID.n_to),
        (storage, metadata),
        rule.taps,
        np.array([0.0, 0.05]),
        GRID.dt_fine,
        partials,
        rings,
    )
    _, fractions = passes.plasma_channel_fractions(
        real, channels, (0.0, 0.05), GRID.dt_fine, rule
    )
    bound = 64 * GRID.n_to * np.finfo(float).eps
    assert partials[0] == pytest.approx(fractions[0, :, -1], rel=bound)
    assert partials[2] == pytest.approx(fractions[1, :, -1], rel=bound)


def test_the_band_filter_reflects_both_ends():
    """At the ends the virtual samples are the reference's point reflection."""
    rule = rules()["K=4"]
    values = np.random.default_rng(18).normal(size=12)
    expected = rule.filter(values)
    filter_range = kernels._make_band_passes()[0]
    got = np.empty_like(values)
    filter_range(values, got, 0, values.size, rule.taps, 0, values.size)
    # Each sample sums the reference's terms in the reference's order; contraction
    # may round each product-sum once: gamma_{2K+2} of its magnitudes (SEP 25 4.6),
    # where a reflected sample 2 y_0 - y_m enters with magnitude 2 |y_0| + |y_m|.
    half = len(rule.taps) - 1
    magnitudes = np.abs(values)
    extended = np.concatenate(
        [
            2.0 * magnitudes[:1] + magnitudes[1 : half + 1][::-1],
            magnitudes,
            2.0 * magnitudes[-1:] + magnitudes[-half - 1 : -1][::-1],
        ]
    )
    bound = stencil_bound(rule.taps, extended)[half:-half]
    margin = assert_within(got, expected, bound * 4 + 1e-300, elementwise=True)
    assert margin <= 1.0


@pytest.mark.parametrize("n_pol", [1, 2])
def test_the_preloaded_fused_column_starts_from_the_band_polarisation(n_pol):
    """Preloaded: each sample's total is what ``out`` held, then the Kerr term, then
    the collar, in the order the reference sums the plasma and the cubic."""
    real, _ = fields(n_pol)
    rows = real.reshape(-1, GRID.n_to)
    held = np.random.default_rng(18).normal(scale=1e-6, size=rows.shape)
    window = np.linspace(0.5, 1.0, GRID.n_to)
    spec = kernels.ColumnSpec("field_real", False, False, False, False, n_pol, False)

    def run(preloaded, start):
        out = start.copy()
        kernels.make_response_column(spec._replace(preloaded=preloaded))(
            rows,
            out,
            out[:1],
            window,
            None,
            np.zeros(GRID.n_to, complex),
            1e-30,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            GRID.dt_fine,
            np.zeros((1, 2)),
        )
        return out

    plain = run(False, np.zeros_like(rows))
    preloaded = run(True, held)
    # first = held + kerr, then times the window: the plain run's kerr * window is
    # (0 + kerr) * window, so the difference is held * window to one rounding.
    assert np.allclose(preloaded, (held + plain / window) * window, rtol=1e-14, atol=0)
