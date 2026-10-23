# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test the medium axial-flux measure of free-space planes (SEP 34)."""

import math

import numpy as np
import pytest

from solprop import fourier
from solprop.constants import EPS0, HBAR, C
from solprop.exceptions import GridError, MaterialError, PulseError
from solprop.grid import EnvelopeGrid, FieldGrid, LineGrid, RadialGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.noise import spatial_noise
from solprop.pulses import GaussianPulse, PlanarPulse, PulseSum, SpatialPulse
from solprop.spatial import (
    SpatialFlux,
    SpatialMedium,
    beam_radius,
    critical_power,
    on_axis_spectrum,
    planar_fluence,
    planar_mode_energy,
    spatial_energy,
    spatial_fluence,
    spatial_intensity,
    spatial_power,
)

EPS = np.finfo(float).eps


def _envelope() -> EnvelopeGrid:
    """Return a small envelope grid around 800 nm."""
    return EnvelopeGrid.from_limits(0.1, 800e-9, (650e-9, 1000e-9), 600e-15)


def _gaussian(grid: EnvelopeGrid, space: RadialGrid) -> np.ndarray:
    """Return a Gaussian beam's internal spectrum with a 30 fs Gaussian envelope."""
    profile = np.exp(-((space.r / 2e-4) ** 2))
    envelope = np.exp(-((grid.t / 30e-15) ** 2)) + 0j
    spectrum = fourier.forward(
        profile[:, None, None] * envelope[None, None, :], kind=grid.kind
    )
    return np.moveaxis(space.hankel.forward(np.moveaxis(spectrum, 0, -1)), -1, 0)


def test_energy_scales_with_the_admittance() -> None:
    """A fixed field carries n times its vacuum energy, up to obliquity."""
    grid = _envelope()
    space = RadialGrid(1e-3, 64)
    spectrum = _gaussian(grid, space)
    vacuum = spatial_energy(spectrum, SpatialFlux.resolve(grid, space, SpatialMedium()))
    glass = spatial_energy(
        spectrum, SpatialFlux.resolve(grid, space, SpatialMedium(1.5))
    )
    # A 200 um waist at 800 nm has angular spread k_perp/k ~ 1e-3, so the
    # obliquity cos(theta) departs from one by ~1e-6 of the energy, less in glass.
    assert glass / vacuum == pytest.approx(1.5, rel=1e-5)
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    # In vacuum the admittance is the obliquity c Re(k_z)/omega <= 1, exactly one on
    # axis; the old vacuum measure is its paraxial limit.
    old = (
        0.5
        * EPS0
        * C
        * np.sum(
            space.spectral_weights[:, None]
            * weights
            * np.sum(np.abs(spectrum) ** 2, axis=1)
        )
    )
    assert vacuum <= old
    assert vacuum == pytest.approx(old, rel=1e-5)


def test_energy_fluence_and_power_agree() -> None:
    """Energy, the area integral of fluence and the time integral of power agree."""
    grid = _envelope()
    space = RadialGrid(1e-3, 64)
    spectrum = _gaussian(grid, space)
    flux = SpatialFlux.resolve(grid, space, SpatialMedium("Ar", pressure=2.0))
    energy = spatial_energy(spectrum, flux)
    fluence = spatial_fluence(spectrum, flux)
    power = spatial_power(spectrum, flux)
    intensity = spatial_intensity(spectrum, flux)
    # Discrete Parseval identities of a fixed map: a few hundred rounding units of
    # sums over n_r * n_omega terms.
    assert np.sum(fluence * space.radial_weights) == pytest.approx(energy, rel=1e-12)
    assert np.sum(power) * grid.dt == pytest.approx(energy, rel=1e-12)
    assert grid.dt * np.sum(intensity, axis=-1) == pytest.approx(fluence, rel=1e-10)
    axis = np.sum(on_axis_spectrum(spectrum, flux)) * grid.d_omega
    assert axis == pytest.approx(fluence[0], rel=1e-3)  # first node off axis


def test_field_grid_endpoints_use_constant_admittance_identity() -> None:
    """On a field grid the analytic-signal weights pair with the admittance."""
    grid = FieldGrid.from_limits(0.05, 800e-9, (500e-9, 2000e-9), 400e-15)
    space = RadialGrid(1e-3, 32)
    profile = np.exp(-((space.r / 2e-4) ** 2))
    field = np.cos(grid.omega0 * grid.t) * np.exp(-((grid.t / 20e-15) ** 2))
    spectrum = fourier.forward(profile[:, None, None] * field, kind=grid.kind)
    spectrum = np.moveaxis(space.hankel.forward(np.moveaxis(spectrum, 0, -1)), -1, 0)
    flux = SpatialFlux.resolve(grid, space, SpatialMedium(1.2))
    assert flux.admittance()[:, 0].tolist() == [0.0] * 32  # no optical DC port
    energy = spatial_energy(spectrum, flux)
    power = spatial_power(spectrum, flux)
    assert np.sum(power) * grid.dt == pytest.approx(energy, rel=1e-12)


def test_complex_admittance_cross_term() -> None:
    """Local flux keeps the imaginary admittance's two-angle interference."""
    grid = _envelope()
    # Two oblique plane waves at 0.27 k and 0.8 k, whose admittances differ, in a
    # strongly absorbing medium.
    space = LineGrid(2e-6, 64)
    medium = SpatialMedium(1.5 + 0.2j)
    flux = SpatialFlux.resolve(grid, space, medium)
    k_index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    spectrum = np.zeros((space.n_columns, 1, grid.n_omega), complex)
    spectrum[2, 0, k_index] = 1.0 + 0.5j
    spectrum[6, 0, k_index] = 0.3 - 0.8j
    # Independent reference in the optics convention: the stored field is the
    # complex conjugate of the optics one, and each plane wave has
    # H = (k_z / omega mu0) E, with k_z depending on |k_x| alone.
    omega = grid.omega[k_index]
    k = omega * medium.index(np.array([omega]))[0] / C
    eta = C * np.sqrt(k**2 - space.kperp_sq) / omega
    optics = np.conj(np.fft.ifft(spectrum[:, 0, k_index]))
    plane_waves = np.fft.fft(optics)
    weight = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)[k_index]

    def local(admittance: np.ndarray) -> np.ndarray:
        h_field = np.fft.ifft(plane_waves * admittance)
        return 0.5 * EPS0 * C * weight * np.real(optics * np.conj(h_field))

    expected = local(eta)
    fluence = spatial_fluence(spectrum, flux)
    # One frequency, two plane waves: a few rounding units of the fluence scale.
    scale = np.max(np.abs(expected))
    assert fluence == pytest.approx(expected, rel=1e-12, abs=1e-12 * scale)
    # Discarding the imaginary admittance changes the local flux ...
    assert np.max(np.abs(local(np.real(eta)) - expected)) > 1e-2 * scale
    # ... but not the energy, which is the real-weight quadratic form.
    assert np.sum(fluence) * space.spatial_weights[0] == pytest.approx(
        spatial_energy(spectrum, flux), rel=1e-12
    )


def test_launch_energy_and_peak_power_in_medium() -> None:
    """A launch carries its requested physical energy or peak power."""
    grid = _envelope()
    space = RadialGrid(1e-3, 64)
    medium = SpatialMedium(1.45)
    flux = SpatialFlux.resolve(grid, space, medium)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 2e-4
    )
    spectrum = pulse.build(grid, space, medium=medium)
    # Normalised to rounding of the energy sum.
    assert spatial_energy(spectrum, flux) == pytest.approx(1e-6, rel=1e-12)
    peaked = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, peak_power=1e9), 2e-4
    )
    spectrum = peaked.build(grid, space, medium=medium)
    assert np.max(spatial_power(spectrum, flux)) == pytest.approx(1e9, rel=1e-12)
    vacuum = pulse.build(grid, space, medium=SpatialMedium())
    glass = pulse.build(grid, space, medium=medium)
    # Same energy: the field in glass is lower by sqrt(n0), up to the ~1e-6
    # obliquity of a 200 um waist.
    assert np.max(np.abs(glass)) / np.max(np.abs(vacuum)) == pytest.approx(
        1 / math.sqrt(1.45), rel=1e-5
    )


def test_launch_pulse_sum_members_keep_their_energy() -> None:
    """Each member of a pulse sum carries its own energy in the medium."""
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (500e-9, 1200e-9), 600e-15)
    space = RadialGrid(1e-3, 32)
    medium = SpatialMedium("Ar", pressure=5.0)
    members = (
        GaussianPulse(wavelength=700e-9, fwhm=30e-15, energy=1e-6),
        GaussianPulse(wavelength=900e-9, fwhm=30e-15, energy=2e-6),
    )
    spectrum = SpatialPulse(PulseSum(members), 2e-4).build(grid, space, medium=medium)
    flux = SpatialFlux.resolve(grid, space, medium)
    # Spectrally separated members: their cross term is below 1e-9 of the energy.
    assert spatial_energy(spectrum, flux) == pytest.approx(3e-6, rel=1e-9)


def test_planar_metric_and_launch() -> None:
    """The slab metric weighs each mode by its index; the launch is incident energy."""
    grid = _envelope()
    space = PlanarGrid(
        [SlabMode(63.5e-6, "Ar", 1.0, order=p) for p in (1, 3)], 1e-3, 16
    )
    medium = SpatialMedium("Ar", pressure=1.0)
    flux = SpatialFlux.resolve(grid, space, medium)
    pulse = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
        250e-6,
        waist_x=46.7e-6,
    )
    spectrum = pulse.build(grid, space, medium=medium)
    modes = planar_mode_energy(spectrum, flux)
    assert np.sum(modes) == pytest.approx(spatial_energy(spectrum, flux), rel=1e-13)
    # The coupled share into p = 1 at waist_x = 0.735 a is 98.8 % (Arnold et al.).
    assert modes[0] / 1e-3 == pytest.approx(0.988, abs=2e-3)
    fluence = planar_fluence(spectrum, flux)
    weights = np.tile(space.x_weights, space.n_y) * space.dy
    # The x rule integrates the modes' products to its quadrature accuracy.
    assert np.sum(fluence * weights) == pytest.approx(np.sum(modes), rel=1e-6)


def test_noise_is_one_photon_per_admitted_bin() -> None:
    """Each admitted column and bin carries hbar omega in the medium's measure."""
    grid = _envelope()
    space = RadialGrid(1e-3, 8)
    medium = SpatialMedium(1.5)
    flux = SpatialFlux.resolve(grid, space, medium)
    draw = spatial_noise(flux, np.random.default_rng(3))
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    energy = (
        0.5
        * EPS0
        * C
        * space.spectral_weights[:, None]
        * np.real(flux.admittance())
        * weights
        * np.abs(draw.spectrum[:, 0]) ** 2
    )
    live = flux.admittance() != 0
    photon = HBAR * np.broadcast_to(grid.omega, energy.shape)
    # One photon per admitted bin, to the rounding of a few products per bin.
    assert energy[live] == pytest.approx(photon[live], rel=1e-12)
    assert np.all(draw.spectrum[:, 0][~live] == 0)


def test_flux_refusals() -> None:
    """Widths refuse a signed profile; a side needs a jump; the index is required."""
    grid = _envelope()
    space = RadialGrid(1e-3, 16)
    with pytest.raises(MaterialError, match="no declared jump"):
        SpatialFlux.resolve(grid, space, SpatialMedium(), side="before")
    with pytest.raises(MaterialError, match="finite position"):
        SpatialFlux.resolve(grid, space, SpatialMedium(), z=math.inf)
    with pytest.raises(MaterialError, match="index"):
        critical_power(800e-9, 1e-25, index=0.0)
    # Evaluated through logarithms: a few rounding units of exp and log.
    assert critical_power(800e-9, 1e-25, index=2.0) == pytest.approx(
        2 * critical_power(800e-9, 1e-25, index=1.0), rel=1e-14
    )
    # A normal and a near-cutoff plane wave in an absorbing medium interfere to a
    # locally negative axial flux (about 0.5 % of its peak).
    line = LineGrid(2e-6, 64)
    flux = SpatialFlux.resolve(grid, line, SpatialMedium(1.5 + 0.2j))
    k_index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    spectrum = np.zeros((64, 1, grid.n_omega), complex)
    spectrum[[0, 7], 0, k_index] = 1.0
    with pytest.raises(GridError, match="nonnegative fluence"):
        beam_radius(spectrum, flux)
    # A metal-like index, Re(n**2) < 0, has no propagating column to carry energy.
    with pytest.raises(PulseError, match="no positive forward flux"):
        SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 2e-4
        ).build(grid, space, medium=SpatialMedium(0.1 + 2j))
