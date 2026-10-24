# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Temporal remapping of a field plane against the direct finite-time sum (SEP 31)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.coupling import (
    FieldPlane,
    PlaneBasis,
    PlaneGeometry,
    PlaneMetric,
    PlaneReference,
    basis_identity,
    regrid,
)
from solprop.exceptions import CouplingError
from solprop.fourier import spectrum_to_optics
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.pulses import GaussianPulse, PulseSum, SpatialPulse
from solprop.spatial import SpatialMedium

EPS = np.finfo(float).eps
LIMITS = (600e-9, 1.2e-6)


def envelope(trange=1e-12, wavelength=800e-9, limits=LIMITS):
    return EnvelopeGrid.from_limits(1.0, wavelength, limits, trange)


def field(trange=1e-12, limits=LIMITS):
    return FieldGrid.from_limits(1.0, 800e-9, limits, trange)


def plane_of(pulse, grid, **changes):
    samples = spectrum_to_optics(pulse.build(grid), kind=grid.kind)
    basis = PlaneBasis("modal", basis_identity("modal", {"modes": ["test"]}))
    return FieldPlane.from_samples(
        samples,
        grid,
        PlaneGeometry(0.0),
        exterior=changes.pop("exterior", "zero"),
        basis=basis,
        **changes,
    )


def q_factor(grid):
    factor = np.ones(grid.n_omega)
    if grid.kind == "field":
        factor[1:-1] = 2.0
    return factor


def direct_sum(plane, grid, shift=0.0, phase=0.0):
    """Evaluate the finite-time Fourier sum as written, one double sum, no FFT."""
    source = plane.grid
    spectrum = plane.samples[:, :, :]
    analytic = (
        np.einsum(
            "mpk,kj->mpj",
            spectrum * q_factor(source),
            np.exp(-1j * np.outer(source.omega, source.t)),
        )
        / source.n_t
    )
    active = grid.omega[grid.active_mask]
    half = 0.5 * source.d_omega
    inside = (
        (grid.omega >= active.min())
        & (grid.omega <= active.max())
        # Beyond the source lattice the sum repeats: aliases, not light.
        & (grid.omega >= source.omega.min() - half)
        & (grid.omega <= source.omega.max() + half)
    )
    kernel = np.exp(1j * np.outer(source.t + shift, grid.omega))
    result = np.einsum("mpj,jk->mpk", analytic, kernel) * source.dt * np.exp(1j * phase)
    result /= grid.dt * q_factor(grid)
    result[..., ~inside] = 0.0
    return result


def gaussian(energy=1e-6, wavelength=800e-9, t0=0.0):
    return GaussianPulse(wavelength=wavelength, fwhm=30e-15, energy=energy, t0=t0)


def reference_budget(plane, grid):
    """Rounding of two sums of N terms each, relative to the largest sample."""
    return 64 * (plane.grid.n_t + grid.n_t) * EPS * float(np.max(np.abs(plane.samples)))


def test_an_identical_lattice_copies_and_the_direct_sum_agrees():
    plane = plane_of(gaussian(), envelope())
    copied, report = regrid(plane, plane.grid)
    assert report.method == "copy"
    # The identity map is an exact structural contract.
    assert np.array_equal(copied.samples, plane.samples)
    direct, report = regrid(plane, plane.grid, method="direct")
    assert report.method == "direct"
    assert_allclose(
        direct.samples, plane.samples, rtol=0, atol=reference_budget(plane, plane.grid)
    )
    assert report.energy_out == pytest.approx(report.energy_in, rel=1e-12)


@pytest.mark.parametrize(
    "target",
    [
        envelope(2e-12),
        envelope(1.5e-12, limits=(500e-9, 1.5e-6)),
        field(2e-12),
        envelope(2e-12, wavelength=900e-9, limits=(550e-9, 1.4e-6)),
    ],
    ids=["longer", "finer", "field", "carrier"],
)
@pytest.mark.parametrize("source_kind", ["envelope", "field"])
def test_unequal_lattices_follow_the_direct_sum_and_the_continuous_pulse(
    target, source_kind
):
    source = envelope() if source_kind == "envelope" else field()
    pulse = gaussian()
    plane = plane_of(pulse, source)
    moved, report = regrid(plane, target)
    assert report.method == "direct"
    assert_allclose(
        moved.samples,
        direct_sum(plane, target),
        rtol=0,
        atol=reference_budget(plane, target),
    )
    # The same localised, band-limited pulse built on the target: two DFT samplings
    # of one continuous spectrum differ by its aliased tails, below 1e-12 here.
    expected = spectrum_to_optics(pulse.build(target), kind=target.kind)
    scale = float(np.max(np.abs(expected)))
    assert_allclose(moved.samples, expected, rtol=0, atol=1e-12 * scale)
    # No renormalisation: the energy out is what the map gives, and it agrees.
    assert moved.energy() == pytest.approx(plane.energy(), rel=1e-10)
    assert report.band_loss < 1e-12 and report.window_loss == 0.0


def test_a_delay_moves_the_record_and_leaves_the_absolute_spectrum():
    grid = envelope()
    plane = plane_of(gaussian(), grid, reference=PlaneReference(0.3, 2e-12))
    step = 40 * grid.dt
    moved, report = regrid(plane, grid, delay=2e-12 - step)
    assert report.shift == pytest.approx(step, rel=1e-15)
    assert_allclose(
        moved.samples,
        direct_sum(plane, grid, shift=step),
        rtol=0,
        atol=reference_budget(plane, grid),
    )
    # The absolute spectrum is invariant under a change of reference.
    assert_allclose(
        moved.absolute(), plane.absolute(), rtol=0, atol=reference_budget(plane, grid)
    )
    # A whole-sample delay of a localised record is a shift of its samples: the
    # pulse peaks 40 samples later in the window.
    synthesis = np.exp(-1j * np.outer(grid.omega, grid.t)) / grid.n_t
    before = np.abs(plane.samples[0, 0] @ synthesis)
    after = np.abs(moved.samples[0, 0] @ synthesis)
    assert int(np.argmax(after)) - int(np.argmax(before)) == 40


def test_a_carrier_phase_change_is_a_phase_of_the_stored_spectrum():
    grid = field()
    plane = plane_of(gaussian(), grid)
    moved, report = regrid(plane, grid, phase=-0.7)
    assert report.phase_shift == pytest.approx(0.7)
    active = grid.active_mask
    # A copy times one exponential, against the direct sum's rounding.
    assert_allclose(
        moved.samples[..., active],
        np.exp(0.7j) * plane.samples[..., active],
        rtol=0,
        atol=reference_budget(plane, grid),
    )
    assert_allclose(
        moved.absolute(), plane.absolute(), rtol=0, atol=reference_budget(plane, grid)
    )


def test_separated_bands_and_spectral_zeros_are_carried_without_interpolation():
    source = envelope(1e-12, limits=(500e-9, 1.6e-6))
    target = field(1.4e-12, limits=(450e-9, 1.8e-6))
    pulse = PulseSum((gaussian(wavelength=650e-9), gaussian(wavelength=1.3e-6)))
    plane = plane_of(pulse, source)
    moved, _ = regrid(plane, target)
    expected = spectrum_to_optics(pulse.build(target), kind=target.kind)
    scale = float(np.max(np.abs(expected)))
    assert_allclose(moved.samples, expected, rtol=0, atol=1e-12 * scale)
    # Between the bands the spectrum is at the level of the tails, not of a fit.
    middle = (target.omega > 2.0e15) & (target.omega < 2.35e15)
    assert float(np.max(np.abs(moved.samples[..., middle]))) < 1e-12 * scale
    # A spectral zero with a sign change: the sum carries the complex value, so the
    # pi phase jump is exact where an unwrapped-phase interpolation would smear it.
    odd = plane.samples * (source.omega - source.omega0)[None, None, :]
    signed = FieldPlane(
        "modal",
        odd,
        source,
        plane.units,
        plane.basis,
        plane.reference,
        plane.geometry,
        plane.metric,
    )
    moved_odd, _ = regrid(signed, target)
    assert_allclose(
        moved_odd.samples,
        direct_sum(signed, target),
        rtol=0,
        atol=reference_budget(signed, target),
    )


def test_an_occupied_band_the_target_cannot_hold_is_refused_or_reported():
    plane = plane_of(gaussian(), envelope(limits=(500e-9, 1.6e-6)))
    # A field grid holds every frequency up to its Nyquist one, but its active band
    # here starts at twice 400 nm, the pulse's own carrier: about half is outside.
    narrow = field(limits=(300e-9, 400e-9))
    with pytest.raises(CouplingError, match="outside the target's active band"):
        regrid(plane, narrow)
    moved, report = regrid(plane, narrow, crop_band=True)
    assert report.band_loss > 0.1
    # Nothing restores the removed light.
    assert report.energy_out is not None and report.energy_in is not None
    assert report.energy_out < report.energy_in * (1 - 0.5 * report.band_loss)
    assert moved.energy() == report.energy_out


def test_a_window_that_cannot_hold_the_record_is_refused_or_cut():
    plane = plane_of(gaussian(t0=300e-15), envelope(1e-12))
    short = envelope(0.4e-12)
    with pytest.raises(CouplingError, match="outside the target window"):
        regrid(plane, short)
    _, report = regrid(plane, short, truncate=True)
    assert report.window_loss > 0.5 and report.truncated


def test_a_nonlocalised_or_unknown_record_is_not_extended_without_truncation():
    grid = envelope(0.2e-12)
    edge = plane_of(GaussianPulse(wavelength=800e-9, fwhm=80e-15, energy=1e-6), grid)
    longer = envelope(1e-12)
    # A record cut by its window leaks across the band too; that loss is allowed
    # here so that the window's refusal is the one met.
    with pytest.raises(CouplingError, match="temporal collar"):
        regrid(edge, longer, crop_band=True)
    _, report = regrid(edge, longer, truncate=True, crop_band=True)
    assert report.edge_share > 1e-10
    unknown = plane_of(gaussian(), envelope(1e-12), exterior="unknown")
    with pytest.raises(CouplingError, match="exterior is unknown"):
        regrid(unknown, envelope(2e-12))
    # The same window is no extension, so an unknown exterior is not consulted.
    same, _ = regrid(unknown, envelope(1e-12), method="direct")
    assert same.exterior == "unknown"


def test_regrid_refuses_an_invalid_tolerance():
    plane = plane_of(gaussian(), envelope())
    with pytest.raises(CouplingError, match="tolerance"):
        regrid(plane, plane.grid, tolerance=1.0)


def test_a_spatial_metric_follows_the_plane_to_its_new_lattice():
    source = FieldGrid.from_limits(1e-3, 800e-9, LIMITS, 400e-15)
    target = FieldGrid.from_limits(1e-3, 800e-9, LIMITS, 800e-15)
    space = RadialGrid(4e-4, 32)
    medium = SpatialMedium("Ar", 1.0)
    beam = SpatialPulse(gaussian(energy=3e-9), 6e-5)
    plane = FieldPlane.from_pulse(
        beam, source, PlaneGeometry(0.0), space=space, medium=medium
    )
    moved, report = regrid(plane, target)
    assert moved.metric.kind == "admittance" and moved.metric.flux is not None
    assert report.energy_out == pytest.approx(report.energy_in, rel=1e-10)
    bare = FieldPlane(
        plane.kind,
        plane.samples,
        plane.grid,
        plane.units,
        plane.basis,
        plane.reference,
        plane.geometry,
        PlaneMetric(
            "admittance",
            admittance=plane.metric.admittance,
            weights=plane.metric.weights,
            scale=plane.metric.scale,
        ),
    )
    unknown, report = regrid(bare, target)
    assert unknown.metric.kind == "unknown" and report.energy_out is None


def test_a_zero_plane_moves_to_zero_with_nothing_lost():
    source = envelope()
    basis = PlaneBasis("modal", basis_identity("modal", {"modes": ["test"]}))
    plane = FieldPlane.from_samples(
        np.zeros((1, 1, source.n_omega), complex),
        source,
        PlaneGeometry(0.0),
        exterior="zero",
        basis=basis,
    )
    moved, report = regrid(plane, envelope(trange=2e-12, limits=(500e-9, 1.5e-6)))
    # An empty record has no norm to share: no loss is reported, and the sum of
    # zeros is exactly zero.
    assert report.method == "direct" and report.record_norm == 0.0
    assert report.band_loss == 0.0 and report.window_loss == 0.0
    assert not np.any(moved.samples)
