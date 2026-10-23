# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the shot-noise realisations: one photon per mode per bin, on both grids."""

import logging

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from solprop import fourier
from solprop.constants import HBAR
from solprop.exceptions import NoiseError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.noise import (
    NoiseRealisation,
    _amplitude,
    input_noise,
    modified_noise,
    photons_per_bin,
    transformed_noise,
)
from solprop.pulses import GaussianPulse

EPS = np.finfo(np.float64).eps

# A 4096-bin grid, so the uniformity test has enough samples for a Kolmogorov-Smirnov
# statistic to mean something, and small enough to build in milliseconds.
LIMITS = (400e-9, 2e-6)
TRANGE = 2e-12

GRIDS = ("field_grid", "envelope_grid")
DRAWS = (input_noise, modified_noise)


@pytest.fixture
def field_grid():
    return FieldGrid.from_limits(1.0, 800e-9, LIMITS, TRANGE)


@pytest.fixture
def envelope_grid():
    return EnvelopeGrid.from_limits(1.0, 800e-9, LIMITS, TRANGE)


# --- one photon per bin ---------------------------------------------------------------


@pytest.mark.parametrize("grid_name", GRIDS)
@pytest.mark.parametrize("draw", DRAWS, ids=["input", "modified"])
def test_every_active_bin_carries_exactly_one_photon(draw, grid_name, request):
    grid = request.getfixturevalue(grid_name)
    realisation = draw(grid, np.random.default_rng(7))
    photons = photons_per_bin(grid, realisation.spectrum)[0, 0]
    # The amplitude is deterministic and only the phase is random, so this is not a
    # statistic: it is a square root, a square and a division, and the residual is the
    # round-off of those three, well under ten epsilon.
    assert_allclose(photons[grid.active_mask], 1.0, rtol=10 * EPS, atol=0.0)


@pytest.mark.parametrize("grid_name", GRIDS)
@pytest.mark.parametrize("draw", DRAWS, ids=["input", "modified"])
def test_the_noise_is_zero_outside_the_active_band(draw, grid_name, request):
    grid = request.getfixturevalue(grid_name)
    realisation = draw(grid, np.random.default_rng(7))
    outside = realisation.spectrum[0, 0][~grid.active_mask]
    # Real-grid noise must be masked like the envelope's, so this is the regression that
    # keeps solprop's two grids consistent with one another.
    assert np.all(outside == 0.0)
    assert np.any(~grid.active_mask)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_spectral_density_of_the_noise_is_hbar_omega_per_bin(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    realisation = input_noise(grid, np.random.default_rng(13))
    density = fourier.spectral_density(
        realisation.spectrum[0, 0], grid.dt, kind=grid.kind
    )
    active = grid.active_mask
    # The pairing that fixes the amplitudes of conventions.md section 4: the density
    # times the bin width is one photon's energy. Same three operations as above.
    assert_allclose(
        density[active] * grid.d_omega,
        HBAR * grid.omega[active],
        rtol=10 * EPS,
        atol=0.0,
    )


@pytest.mark.parametrize("grid_name", GRIDS)
def test_photons_per_bin_reports_zero_where_there_is_no_photon(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    spectrum = np.ones((1, 1, grid.n_omega), dtype=complex)
    photons = photons_per_bin(grid, spectrum)[0, 0]
    assert np.all(photons[grid.omega > 0.0] > 0.0)
    assert np.all(photons[grid.omega <= 0.0] == 0.0)


def test_a_real_pulse_sits_far_above_the_quantum_floor(envelope_grid):
    spectrum = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(
        envelope_grid
    )
    photons = photons_per_bin(envelope_grid, spectrum)[0, 0]
    # A microjoule at 800 nm is 4e12 photons; spread over the few hundred bins the
    # pulse occupies, the peak bin must hold many orders of magnitude more than one.
    assert float(np.max(photons)) > 1e8


# --- randomness -----------------------------------------------------------------------


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_phases_are_uniform(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    realisation = input_noise(grid, np.random.default_rng(19))
    active = grid.active_mask
    phases = np.angle(realisation.spectrum[0, 0][active]) % (2.0 * np.pi)
    result = stats.kstest(phases / (2.0 * np.pi), "uniform")
    # Seeded, so this is a fixed number rather than a random one: with seed 19 the
    # p-value is 0.4, two orders above the 0.005 threshold asserted here. The threshold
    # exists to catch a broken draw, not to police a particular realisation.
    assert result.pvalue > 5e-3
    assert phases.size > 1000


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_modulus_is_the_same_for_every_realisation(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    first = input_noise(grid, np.random.default_rng(2)).spectrum
    second = input_noise(grid, np.random.default_rng(3)).spectrum
    # Only the phase is random (conventions.md section 4), so two realisations differ
    # in phase and agree bin for bin in modulus, to the round-off of one exponential.
    assert_allclose(np.abs(first), np.abs(second), rtol=8 * EPS, atol=0.0)
    assert not np.allclose(np.angle(first), np.angle(second))


@pytest.mark.parametrize("grid_name", GRIDS)
def test_a_realisation_is_reproducible_from_its_seed(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    first = input_noise(grid, np.random.default_rng(23))
    again = input_noise(grid, np.random.default_rng(first.seed))
    assert first.seed == 23
    # The same draw (the generator's stream is exact) through one exponential and one
    # product per bin: the realisation agrees to their rounding, 8 eps.
    assert_allclose(again.spectrum, first.spectrum, rtol=8 * EPS, atol=0.0)


def test_an_unseeded_generator_still_records_the_entropy_it_drew(field_grid):
    rng = np.random.default_rng()
    realisation = input_noise(field_grid, rng)
    assert isinstance(realisation.seed, int)
    replayed = input_noise(field_grid, np.random.default_rng(realisation.seed))
    # The same draw, to the rounding of one exponential and one product per bin.
    assert_allclose(replayed.spectrum, realisation.spectrum, rtol=8 * EPS, atol=0.0)


def test_a_seed_that_is_not_one_integer_is_not_recorded(field_grid):
    # A generator seeded with a sequence has no single integer to record, so the
    # realisation says None rather than inventing one that would not replay.
    realisation = input_noise(field_grid, np.random.default_rng([1, 2, 3]))
    assert realisation.seed is None
    assert np.any(realisation.spectrum != 0.0)


@pytest.mark.parametrize("grid_name", GRIDS)
def test_modes_and_polarisations_are_independent(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    realisation = input_noise(grid, np.random.default_rng(29), n_modes=3, n_pol=2)
    assert realisation.spectrum.shape == (3, 2, grid.n_omega)
    active = grid.active_mask
    phases = np.angle(realisation.spectrum[..., active]).reshape(6, -1)
    for i in range(6):
        for j in range(i + 1, 6):
            assert not np.allclose(phases[i], phases[j])
            correlation = float(np.corrcoef(np.cos(phases[i]), np.cos(phases[j]))[0, 1])
            # Independent draws over n bins have a sample correlation of order
            # 1/sqrt(n); with more than a thousand active bins, 0.2 is many standard
            # deviations away and still catches a repeated stream.
            assert abs(correlation) < 0.2


def test_the_phase_is_drawn_over_the_whole_axis(field_grid):
    # One draw per bin, not one per active bin, so moving the band on a grid of the
    # same size does not shift the stream and a seed keeps meaning one realisation.
    realisation = input_noise(field_grid, np.random.default_rng(31))
    drawn = np.random.default_rng(31).uniform(
        0.0, 2.0 * np.pi, size=(1, 1, field_grid.n_omega)
    )
    expected = _amplitude(field_grid) * np.exp(1j * drawn[0, 0])
    # The same two operations on the same two arrays: an exponential and a product,
    # each a few ulps, so 8 eps per bin.
    assert_allclose(realisation.spectrum[0, 0], expected, rtol=8 * EPS, atol=0.0)


# --- the two models -------------------------------------------------------------------


@pytest.mark.parametrize("grid_name", GRIDS)
def test_the_two_models_share_one_draw(grid_name, request):
    grid = request.getfixturevalue(grid_name)
    first = input_noise(grid, np.random.default_rng(37))
    second = modified_noise(grid, np.random.default_rng(37))
    # ADR-008: the models differ in how the driver consumes the realisation, never in
    # how it is drawn. A parity test between them is a parity test of that claim: the
    # same draw, to the rounding of one exponential and one product per bin.
    assert_allclose(second.spectrum, first.spectrum, rtol=8 * EPS, atol=0.0)
    assert (first.model, second.model) == ("input", "modified")


def test_a_realisation_is_frozen(field_grid):
    realisation = input_noise(field_grid, np.random.default_rng(41))
    assert not realisation.spectrum.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        realisation.spectrum[0, 0, 0] = 0.0


def test_a_realisation_can_be_built_by_hand(field_grid):
    spectrum = np.zeros((1, 1, field_grid.n_omega), dtype=complex)
    realisation = NoiseRealisation(spectrum=spectrum, seed=None, model="input")
    assert realisation.seed is None
    # The constructor copies, so the caller's array and the realisation cannot drift.
    spectrum[0, 0, 0] = 1.0
    assert realisation.spectrum[0, 0, 0] == 0.0


def test_the_draw_is_logged_at_debug_level(field_grid, caplog):
    with caplog.at_level(logging.DEBUG, logger="solprop.noise"):
        input_noise(field_grid, np.random.default_rng(43))
    assert "input shot noise" in caplog.text


# --- error paths ----------------------------------------------------------------------


@pytest.mark.parametrize("draw", DRAWS, ids=["input", "modified"])
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_modes": 0}, "n_modes must be a positive count"),
        ({"n_modes": -2}, "n_modes must be a positive count"),
        ({"n_pol": 0}, "n_pol must be a positive count"),
    ],
)
def test_a_bad_count_is_rejected(draw, overrides, message, field_grid):
    with pytest.raises(NoiseError, match=message):
        draw(field_grid, np.random.default_rng(47), **overrides)


# -------------------------------------- the draw carried into a declared basis


CIRCULAR = np.array([[1.0, 1.0], [-1j, 1j]]) / np.sqrt(2.0)
"""The circular partners of a parity pair, as `circular_pair` builds them."""


def test_a_carried_draw_keeps_its_total_photon_number_in_every_bin():
    """`sum_m |d_m|^2 = sum_n |c_n|^2` for a unitary transform, bin by bin."""
    grid = FieldGrid.from_limits(1.0, 800e-9, (200e-9, 2e-6), 400e-15)
    drawn = input_noise(grid, np.random.default_rng(17), n_modes=2)
    turned = transformed_noise(drawn, CIRCULAR)
    before = photons_per_bin(grid, drawn.spectrum).sum(axis=(0, 1))
    after = photons_per_bin(grid, turned.spectrum).sum(axis=(0, 1))
    assert_allclose(after, before, rtol=1e-12, atol=0.0)
    assert_allclose(after[grid.active_mask], 2.0, rtol=1e-12, atol=0.0)


def test_a_carried_draw_redistributes_the_channels_and_keeps_its_seed():
    """A fixed-modulus random phase is not invariant under unitary mixing."""
    grid = FieldGrid.from_limits(1.0, 800e-9, (200e-9, 2e-6), 400e-15)
    drawn = input_noise(grid, np.random.default_rng(17), n_modes=2)
    turned = transformed_noise(drawn, CIRCULAR)
    assert turned.seed == drawn.seed and turned.model == drawn.model
    active = photons_per_bin(grid, turned.spectrum)[:, 0][:, grid.active_mask]
    # Per channel the moduli are no longer all one, which is the point: the draw
    # is defined in the reference basis and carried, not drawn again here.
    assert float(np.max(np.abs(active - 1.0))) > 0.1
    # `U.H` and not `U`: the amplitudes carry the inverse of the fields' transform.
    assert_allclose(
        turned.spectrum,
        np.einsum("mn,npk->mpk", CIRCULAR.conj().T, drawn.spectrum),
        rtol=0.0,
        atol=0.0,
    )


def test_a_carried_draw_comes_back_under_the_inverse_transform():
    grid = FieldGrid.from_limits(1.0, 800e-9, (200e-9, 2e-6), 400e-15)
    drawn = input_noise(grid, np.random.default_rng(5), n_modes=2)
    back = transformed_noise(transformed_noise(drawn, CIRCULAR), CIRCULAR.conj().T)
    assert_allclose(back.spectrum, drawn.spectrum, rtol=1e-14, atol=0.0)


def test_a_transform_of_the_wrong_width_is_refused_by_name():
    grid = FieldGrid.from_limits(1.0, 800e-9, (200e-9, 2e-6), 400e-15)
    drawn = input_noise(grid, np.random.default_rng(5), n_modes=2)
    with pytest.raises(NoiseError, match="square transform of that width"):
        transformed_noise(drawn, np.eye(3, dtype=complex))
