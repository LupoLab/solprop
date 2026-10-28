# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify SEP 27's direct-current source against analytic, independent oracles."""

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.integrate import quad_vec
from scipy.special import erf, erfc

from solprop.backends.numpy_ref import passes
from solprop.exceptions import TranslationError
from solprop.grid import (
    FieldGrid,
    planck_taper,
    planck_taper_derivative,
    planck_taper_derivative_error,
)
from solprop.translate.masks import build_masks
from solprop.translate.plasma_source import plasma_source_record

EPS = np.finfo(float).eps


def grid_case(limits=(90e-9, 1.6e-6), window=60e-15, fft_size="pow2"):
    # The certificate's cases were sized on the collar inside the window, a twentieth
    # of it at each edge, as solprop built it before SEP 21 (LEGACY_GRID's collar).
    return FieldGrid.from_limits(
        1.0,
        800e-9,
        limits,
        window,
        fft_size=fft_size,
        collar_fraction=0.05,
        collar_placement="inside",
    )


def source(grid, current):
    masks = build_masks(grid)
    return passes.held_plasma_spectrum(
        grid.t_window_fine * current,
        masks.t_window_fine_derivative * current,
        masks.omega,
        masks.crop_index,
        masks.active_mask,
    )


def rounding_budget(grid, current):
    """Propagate SEP 27 4.6 through products, two FFTs, division and subtraction."""
    collar = grid.t_collar
    gate, slope = grid.t_window_fine, collar.window_derivative(grid.t_fine)
    slope_error = planck_taper_derivative_error(
        grid.t_fine, collar.zero_low, collar.one_low, collar.one_high, collar.zero_high
    )
    count = grid.n_to
    x, z = gate * current, slope * current
    # Gate formation and product: 16 operations, with the analytic derivative's
    # operand-dependent bound (19 + 3 |zeta|) eps |g'| supplied independently.
    dx = 16 * EPS * np.abs(x)
    dz = np.abs(current) * slope_error + EPS * np.abs(z)
    fft = 8 * EPS * math.ceil(math.log2(count))
    bx = math.sqrt(count) * (np.linalg.norm(dx) + fft * np.linalg.norm(x))
    bz = math.sqrt(count) * (np.linalg.norm(dz) + fft * np.linalg.norm(z))
    active = grid.active_mask
    omega = grid.omega[active]
    crop = grid.crop_index[active]
    spectrum, edge = np.fft.rfft(x)[crop], np.fft.rfft(z)[crop]
    out = np.zeros(grid.n_omega)
    out[active] = (
        bx / omega
        + bz / omega**2
        + 4 * EPS * (np.abs(spectrum) / omega + np.abs(edge) / omega**2)
    )
    return out


def steady_residual(grid):
    active = grid.active_mask
    omega = grid.omega[active]
    crop = grid.crop_index[active]
    x = np.fft.rfft(grid.t_window_fine)[crop]
    z = np.fft.rfft(grid.t_collar.window_derivative(grid.t_fine))[crop]
    return np.abs(x - z / (1j * omega)) / omega


@pytest.mark.parametrize(
    "limits,window",
    [
        ((90e-9, 1.6e-6), 60e-15),
        ((90e-9, 1.6e-6), 100e-15),
        ((200e-9, 2e-6), 40e-15),
        ((400e-9, 2e-6), 1e-12),
    ],
)
def test_steady_current_is_limited_by_the_certificate_and_rounding(
    limits, window, record_property
):
    grid = grid_case(limits, window)
    current = np.ones(grid.n_to)
    active = grid.active_mask
    allowance = steady_residual(grid) + rounding_budget(grid, current)[active]
    assert np.all(np.abs(source(grid, current)[active]) <= allowance)
    record = plasma_source_record(build_masks(grid))
    assert isinstance(record["certificate"], float)
    assert isinstance(record["limit"], float)
    assert record["certificate"] <= record["limit"]
    gated = np.fft.rfft(grid.t_window_fine * (grid.t_fine - grid.t_fine[0]))[
        grid.crop_index[active]
    ]
    assert np.linalg.norm(gated) > 0
    record_property("gated_steady_norm", float(np.linalg.norm(gated)))
    record_property("held_steady_norm", float(np.linalg.norm(source(grid, current))))


def test_zero_current_and_inactive_bins_are_structurally_zero():
    grid = grid_case()
    assert_array_equal(source(grid, np.zeros(grid.n_to)), np.zeros(grid.n_omega))
    assert_array_equal(source(grid, np.ones(grid.n_to))[~grid.active_mask], 0.0)


def product_sampling_bounds(grid, centre, sigma):
    """Bound each sampled product against independent continuous quadrature.

    Integration uses a dimensionless period, with breaks at all four analytic
    collar boundaries and the rise. The returned quadrature error is added to
    both products, without letting their errors cancel. This is a per-case
    sampling allowance, not the production steady-current certificate.
    """
    period = grid.n_to * grid.dt_fine
    active = grid.active_mask
    bins = grid.crop_index[active]
    collar = grid.t_collar

    def integrand(position):
        times = np.asarray([position * period])
        current = (1 + erf((times[0] - centre) / sigma)) / 2
        phase = np.exp(-2j * np.pi * bins * (position + 0.5))
        return np.stack(
            (
                collar.window(times)[0] * current * phase,
                period * collar.window_derivative(times)[0] * current * phase,
            )
        )

    points = (
        np.asarray(
            [
                collar.zero_low,
                collar.one_low,
                centre,
                collar.one_high,
                collar.zero_high,
            ]
        )
        / period
    )
    continuous, error = quad_vec(
        integrand,
        -0.5,
        0.5,
        points=points,
        epsabs=1e-13,
        epsrel=1e-13,
        norm="max",
        limit=10000,
    )
    current = (1 + erf((grid.t_fine - centre) / sigma)) / 2
    sampled = np.stack(
        (
            np.fft.rfft(grid.t_window_fine * current)[bins],
            period * np.fft.rfft(collar.window_derivative(grid.t_fine) * current)[bins],
        )
    )
    # Quadrature error in raw DFT units. The FFT and product rounding is
    # separately charged in rounding_budget, including the continuous oracle's
    # finite precision. No term depends on the assembled source being tested.
    errors = abs(sampled - grid.n_to * continuous) + grid.n_to * error
    return errors[0], errors[1] / period


@pytest.mark.parametrize(
    "limits,window", [((90e-9, 1.6e-6), 60e-15), ((200e-9, 2e-6), 40e-15)]
)
@pytest.mark.parametrize(
    "centre,sigma", [(-5e-15, 1e-15), (3e-15, 2e-15), (-3e-15, 0.8e-15)]
)
@pytest.mark.parametrize("fft_size", ["pow2", "smooth"])
def test_erf_drift_has_the_continuous_held_spectrum(
    limits, window, centre, sigma, fft_size
):
    grid = grid_case(limits, window, fft_size)
    if limits[0] == 200e-9 and fft_size == "smooth":
        with pytest.raises(TranslationError, match="collar certificate"):
            plasma_source_record(build_masks(grid))
        return
    plasma_source_record(build_masks(grid))
    active = grid.active_mask
    omega = grid.omega[active]
    current = (1 + erf((grid.t_fine - centre) / sigma)) / 2
    expected = (
        (-1.0) ** grid.crop_index[active]
        * np.exp(-(omega**2) * sigma**2 / 4 - 1j * omega * centre)
        / (grid.dt_fine * (1j * omega) ** 2)
    )
    collar = grid.t_collar
    distance = min(centre - collar.one_low, collar.one_high - centre)
    # The two-edge steady certificate can cancel per bin and cannot bound a
    # one-sided drift. Independently bound each product's sampling error instead
    # (the approved T2 correction in SEP 27 Part II); retain analytic tails and
    # operand-based rounding without a relative floor on weak bins.
    tails = (
        erfc(distance / sigma) + math.exp(-((np.pi * sigma / grid.dt_fine) ** 2) / 4)
    ) / (grid.dt_fine * omega**2)
    sampled_current, sampled_edge = product_sampling_bounds(grid, centre, sigma)
    allowance = (
        tails
        + sampled_current / omega
        + sampled_edge / omega**2
        + rounding_budget(grid, current)[active]
        + 16 * EPS * np.abs(expected)
    )
    assert np.all(np.abs(source(grid, current)[active] - expected) <= allowance)


@pytest.mark.parametrize("samples", [0.25, 1, 4, 10, 20, 40])
def test_underresolved_tapers_are_refused(samples):
    grid = grid_case()
    masks = build_masks(grid)
    low, high = grid.t_fine[0], grid.t_fine[-1]
    width = samples * grid.dt_fine
    collar = (low, low + width, high - width, high)
    broken = dataclasses.replace(
        masks,
        t_window_fine=planck_taper(grid.t_fine, *collar),
        t_window_fine_derivative=planck_taper_derivative(grid.t_fine, *collar),
    )
    with pytest.raises(TranslationError, match=r"taper samples.*wider"):
        plasma_source_record(broken)


def test_certificate_uses_the_full_fine_transform():
    grid = grid_case((400e-9, 2e-6), 1e-12)
    masks = build_masks(grid)
    gate = np.fft.rfft(masks.t_window_fine)
    slope = np.fft.rfft(masks.t_window_fine_derivative)
    crop, omega = masks.crop_index[masks.active_mask], masks.omega[masks.active_mask]
    expected = np.linalg.norm(gate[crop] - slope[crop] / (1j * omega)) / np.linalg.norm(
        gate
    )
    # Two independently evaluated FFT/subtraction expressions, SEP 27 T7.
    certificate = plasma_source_record(masks)["certificate"]
    assert isinstance(certificate, float)
    assert abs(certificate - expected) <= 10 * EPS * math.log2(grid.n_to)


def test_former_short_spatial_fixture_requires_wider_tapers():
    grid = grid_case((600e-9, 1.2e-6), 150e-15)
    with pytest.raises(TranslationError, match="taper samples"):
        plasma_source_record(build_masks(grid))
    wider = grid_case((600e-9, 1.2e-6), 200e-15)
    certificate = plasma_source_record(build_masks(wider))["certificate"]
    assert isinstance(certificate, float)
    assert certificate < 1e-10
