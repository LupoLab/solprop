# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import replace

import numpy as np
import pytest

from solprop import fourier
from solprop.boundaries import absorber
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.noise import input_noise, modified_noise
from solprop.nonlinear.responses import thg_carrier
from solprop.translate.masks import (
    DEFAULT_ABSORBER_SCHEME,
    Grids,
    build_masks,
    noise_spectrum,
    temporal_rates,
)

GRIDS = ["field", "envelope"]


@pytest.fixture(scope="module", params=GRIDS, ids=GRIDS)
def grid(request):
    builder = FieldGrid if request.param == "field" else EnvelopeGrid
    return builder.from_limits(1.0, 800e-9, (400e-9, 2e-6), 400e-15)


def test_every_mask_is_the_grid_s_own(grid):
    masks = build_masks(grid)
    assert np.array_equal(masks.omega, grid.omega)
    assert np.array_equal(masks.active_mask, grid.active_mask)
    assert np.array_equal(masks.inband_mask, grid.inband_mask)
    assert np.array_equal(masks.t_window_fine, grid.t_window_fine)
    assert np.array_equal(
        masks.t_window_fine_derivative, grid.t_collar.window_derivative(grid.t_fine)
    )
    assert not masks.t_window_fine_derivative.flags.writeable
    # The polarisation taper is off by default: the rate absorber on the field
    # is inside the operator, and the taper would set a floor on a broad pulse.
    assert np.all(masks.omega_window == 1.0)
    assert masks.omega_window.shape == grid.omega_window.shape
    tapered = build_masks(grid, polarisation_taper=True)
    assert np.array_equal(tapered.omega_window, grid.omega_window)
    assert not np.all(tapered.omega_window == 1.0)
    assert masks.omega0 == grid.omega0
    assert (masks.dt, masks.dt_fine, masks.ffac) == (grid.dt, grid.dt_fine, grid.ffac)
    assert (masks.n_t, masks.n_to, masks.n_omega) == (
        grid.n_t,
        grid.n_to,
        grid.n_omega,
    )


def test_the_crop_and_pad_data_are_the_fourier_module_s_own(grid):
    masks = build_masks(grid)
    assert np.array_equal(
        masks.crop_index, fourier.crop_index(grid.n_t, grid.n_to, kind=grid.kind)
    )
    assert masks.pad_scale == fourier.pad_scale(grid.n_t, grid.n_to)
    assert masks.crop_scale == fourier.crop_scale(grid.n_t, grid.n_to)
    # The two factors cancel the different 1/N of the two inverse transforms.
    assert masks.pad_scale * masks.crop_scale == 1.0


def test_the_crop_index_selects_the_coarse_bins_of_the_fine_spectrum(grid):
    masks = build_masks(grid)
    coarse = np.arange(grid.n_omega, dtype=complex)
    fine = fourier.oversample(coarse, grid.n_to, kind=grid.kind)
    assert np.array_equal(fine[masks.crop_index] / masks.pad_scale, coarse)


def test_the_arrays_are_frozen(grid):
    masks = build_masks(grid)
    for name in (
        "omega",
        "crop_index",
        "inband_mask",
        "active_mask",
        "omega_window",
        "t_window_fine",
    ):
        values = getattr(masks, name)
        assert not values.flags.writeable, name
        assert values.flags.c_contiguous, name


def test_the_in_band_region_is_inside_the_active_band(grid):
    masks = build_masks(grid)
    assert np.all(masks.active_mask[masks.inband_mask])
    assert int(np.count_nonzero(masks.active_mask)) > int(
        np.count_nonzero(masks.inband_mask)
    )


# ------------------------------------------------------------------ temporal absorber


def test_the_temporal_rates_are_the_boundary_module_s_own(grid):
    collar = absorber(grid)
    coarse, fine = temporal_rates(collar)
    assert coarse is not None
    assert fine is not None
    assert np.array_equal(coarse, collar.temporal_rate)
    assert np.array_equal(fine, collar.temporal_rate_fine)
    assert coarse.shape == (grid.n_t,)
    assert fine.shape == (grid.n_to,)
    assert not coarse.flags.writeable


def test_no_absorber_means_no_rates():
    assert temporal_rates(None) == (None, None)


def test_both_schemes_carry_both_rates(grid):
    """Switching between the absorber's two applications is not a retranslation."""
    collar = absorber(grid)
    split = temporal_rates(collar, scheme="split_step")
    inside = temporal_rates(collar, scheme="nonlinear")
    assert np.array_equal(np.asarray(split[0]), np.asarray(inside[0]))
    assert np.array_equal(np.asarray(split[1]), np.asarray(inside[1]))
    # ADR-025 (docs/dev/decisions.md) chose the in-stage term: the split factor is a
    # Lie splitting of a term that does not commute with the linear flow, so it caps
    # the observed order at one wherever the collar does any work.
    assert DEFAULT_ABSORBER_SCHEME == "nonlinear"


def test_an_unknown_scheme_is_refused(grid):
    with pytest.raises(TranslationError, match="absorber scheme must be"):
        temporal_rates(absorber(grid), scheme="mask")


# ----------------------------------------------------------------------------- noise


def test_a_realisation_reaches_the_canonical_layout(grid):
    draw = modified_noise(grid, np.random.default_rng(5))
    spectrum = noise_spectrum(draw, (1, 1, 1, grid.n_omega))
    assert spectrum.shape == (1, 1, 1, grid.n_omega)
    assert np.array_equal(spectrum[0], draw.spectrum)
    assert not spectrum.flags.writeable
    assert spectrum.dtype == np.complex128


def test_one_realisation_is_shared_across_a_batch(grid):
    draw = input_noise(grid, np.random.default_rng(5))
    spectrum = noise_spectrum(draw, (3, 1, 1, grid.n_omega))
    assert spectrum.shape == (3, 1, 1, grid.n_omega)
    for member in spectrum:
        assert np.array_equal(member, draw.spectrum)


def test_a_realisation_that_does_not_fit_is_refused(grid):
    draw = modified_noise(grid, np.random.default_rng(5), n_modes=3)
    with pytest.raises(TranslationError, match="does not fit the state shape"):
        noise_spectrum(draw, (1, 1, 1, grid.n_omega))


# ------------------------------------------------------------------------ validation


def masks_with(grid, **overrides):
    """Build a Grids group from a real grid with some fields replaced."""
    fields = {
        "omega": grid.omega,
        "omega0": grid.omega0,
        "dt": grid.dt,
        "dt_fine": grid.dt_fine,
        "n_t": grid.n_t,
        "n_to": grid.n_to,
        "ffac": grid.ffac,
        "crop_index": grid.crop_index,
        "pad_scale": grid.pad_scale,
        "crop_scale": grid.crop_scale,
        "inband_mask": grid.inband_mask,
        "active_mask": grid.active_mask,
        "norm_mask": grid.active_mask,
        "omega_window": grid.omega_window,
        "t_window_fine": grid.t_window_fine,
        "t_window_fine_derivative": grid.t_collar.window_derivative(grid.t_fine),
    }
    return Grids(**(fields | overrides))


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("crop_index", np.intp),
        ("inband_mask", np.bool_),
        ("active_mask", np.bool_),
        ("omega_window", np.float64),
    ],
)
def test_an_array_of_the_wrong_length_is_refused(grid, name, dtype):
    with pytest.raises(TranslationError, match="to match omega"):
        masks_with(grid, **{name: np.zeros(3, dtype=dtype)})


@pytest.mark.parametrize("name", ["t_window_fine", "t_window_fine_derivative"])
def test_a_fine_window_of_the_wrong_length_is_refused(grid, name):
    with pytest.raises(TranslationError, match="to match n_to"):
        masks_with(grid, **{name: np.zeros(3)})


def test_sizes_that_do_not_nest_are_refused(grid):
    with pytest.raises(TranslationError, match="positive multiple"):
        masks_with(grid, n_t=3, n_to=grid.n_to, t_window_fine=grid.t_window_fine)


def test_a_non_positive_scale_is_refused(grid):
    with pytest.raises(TranslationError, match="must be positive"):
        masks_with(grid, pad_scale=0.0)


# ---------------------------------------------------------------- the harmonic carrier


def test_no_harmonic_carrier_by_default(grid):
    assert build_masks(grid).carrier_2w0 is None


def test_the_harmonic_carrier_is_thg_carrier_on_the_fine_time_axis(grid):
    """The contract carries the physics layer's carrier itself, exactly (a copy)."""
    carried = build_masks(grid, harmonic_carrier=True).carrier_2w0
    assert carried is not None
    assert carried.shape == (grid.n_to,)
    assert carried.dtype == np.complex128
    assert not carried.flags.writeable
    np.testing.assert_array_equal(carried, thg_carrier(grid.omega0, grid.t_fine))
    # And the expression the reference backend used to build for itself: the grid's
    # fine axis is that expression to a rounding of each time, which moves the phase
    # by at most a few eps of the largest phase 2 omega_0 max|t| (4 eps, with the
    # exponential's own rounding).
    built = (np.arange(grid.n_to) - grid.n_to / 2) * (grid.dt_fine)
    phase = 2 * grid.omega0 * float(np.max(np.abs(built)))
    np.testing.assert_allclose(
        carried,
        np.exp(2j * grid.omega0 * built),
        rtol=0,
        atol=4 * np.finfo(float).eps * max(phase, 1.0),
    )


def test_a_carrier_of_the_wrong_length_is_refused(grid):
    masks = build_masks(grid)
    with pytest.raises(TranslationError, match="carrier_2w0 has"):
        replace(masks, carrier_2w0=np.ones(masks.n_to + 1, dtype=complex))
