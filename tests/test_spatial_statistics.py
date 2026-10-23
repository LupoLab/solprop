# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent Bessel-field and Gaussian oracles for physical spatial statistics."""

import dataclasses
import inspect
import math
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import j0

from hankel_oracles import gamma, gaussian_bound
from solprop.constants import EPS0, C
from solprop.exceptions import GridError, OutputError
from solprop.fourier import energy_omega
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.hankel import QDHT
from solprop.output.schema import SavedPlane
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import (
    SpatialFlux,
    SpatialMedium,
    beam_radius,
    fwhm_r,
    on_axis_intensity,
    on_axis_spectrum,
    radial_fluence_widths,
    spatial_energy,
    spatial_fluence,
    spatial_intensity,
)
from solprop.stats import analytic_field, collect, default_spec


@pytest.fixture(params=["field", "envelope"])
def grid(request):
    constructor = (
        FieldGrid.from_limits
        if request.param == "field"
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    return constructor(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)


def _vacuum(grid, space):
    """Return the vacuum measure of the plane at z = 0."""
    return SpatialFlux.resolve(grid, space, SpatialMedium())


def _obliquity(grid, k_r):
    """Return the vacuum admittance c k_z / omega of a column, independently.

    ``k_z = sqrt((omega/c)**2 - k_r**2)`` on the forward branch, so the admittance is
    the obliquity ``sqrt(1 - (c k_r / omega)**2)``; only active positive-frequency
    propagating bins are power ports, and every other bin has none (SEP 34 §4.2).
    """
    omega = grid.omega
    live = grid.active_mask & (omega > C * k_r)
    eta = np.zeros(grid.n_omega)
    eta[live] = np.sqrt(1 - (C * k_r / omega[live]) ** 2)
    return eta


@pytest.fixture
def bessel_case(grid):
    space = RadialGrid(6e-4, 64)
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9).build(grid)
    # Circular analytic light occupies positive active frequencies. Imaginary DC
    # and Nyquist components are not real-field degrees of freedom.
    temporal = temporal * grid.active_mask
    temporal *= np.sqrt(
        3e-9 / float(energy_omega(temporal[0, 0], grid.dt, kind=grid.kind))
    )
    state = np.zeros((space.n_r, 2, grid.n_omega), dtype=np.complex128)
    index = 3
    state[index, 0] = temporal[0, 0] / np.sqrt(2)
    state[index, 1] = 1j * temporal[0, 0] / np.sqrt(2)
    return space, temporal, state, index


def test_bessel_column_has_the_physical_bessel_profile(grid, bessel_case):
    space, temporal, state, index = bessel_case
    flux = _vacuum(grid, space)
    # Inverse Hankel series coefficient: 2/(R^2 J1(alpha)^2). J0 gives the
    # entire independent real-space profile; temporal Jones normalization is one.
    coefficient = space.spectral_weights[index] / (2 * np.pi)
    profile = coefficient * j0(space.k_r[index] * space.r)
    # One column's magnetic closure is the field filtered by its vacuum admittance,
    # so the axial flux pairs E_a with the analytic signal of eta * E:
    # S_z = eps0 c / 2 Re(E_a conj(G_a)) (SEP 34 §4.2). Both Jones components carry
    # half of it.
    eta = _obliquity(grid, space.k_r[index])
    field = analytic_field(grid, temporal)[0, 0]
    closure = analytic_field(grid, eta * temporal)[0, 0]
    power = np.real(field * np.conj(closure))
    expected = 0.5 * EPS0 * C * profile[:, None] ** 2 * power
    bound = gamma(64 + 16 * int(np.log2(grid.n_t))) * float(expected.max())
    measured = spatial_intensity(state, flux)
    assert_allclose(measured, expected, rtol=0, atol=bound)
    assert_allclose(
        spatial_fluence(state, flux),
        grid.dt * expected.sum(axis=1),
        rtol=0,
        atol=bound * grid.n_t * grid.dt,
    )
    assert_allclose(
        on_axis_intensity(state, flux),
        0.5 * EPS0 * C * coefficient**2 * power.max(),
        rtol=gamma(64 + 16 * int(np.log2(grid.n_t))),
        atol=0,
    )


def test_bessel_energy_and_on_axis_spectral_fluence(grid, bessel_case):
    space, temporal, state, index = bessel_case
    flux = _vacuum(grid, space)
    weight = space.spectral_weights[index]
    # The real admittance weighs each bin's temporal energy: the column carries the
    # temporal energy of sqrt(eta) times its pulse, not the vacuum 3 nJ.
    eta = _obliquity(grid, space.k_r[index])
    weighted = float(
        energy_omega(np.sqrt(eta) * temporal[0, 0], grid.dt, kind=grid.kind)
    )
    expected_energy = 0.5 * EPS0 * C * weight * weighted
    expected_axis = 0.5 * EPS0 * C * (weight / (2 * np.pi)) ** 2 * weighted
    bound = gamma(8 * grid.n_omega + 32)
    assert_allclose(spatial_energy(state, flux), expected_energy, rtol=bound, atol=0)
    assert_allclose(
        on_axis_spectrum(state, flux).sum() * grid.d_omega,
        expected_axis,
        rtol=bound,
        atol=0,
    )


def test_gaussian_inverse_against_the_dual_hankel_remainder(grid):
    space = RadialGrid(6e-4, 64)
    waist = 1e-4
    # B_R is exactly F_{S/R}. Apply the independently derived Gaussian Hankel
    # remainder to exp(-w^2 k^2/4), before evaluating the reconstruction.
    dual = QDHT(space.hankel.zeros[-1] / space.radius, space.n_r)
    _, transform_bound = gaussian_bound(dual, 2 / waist)
    amplitude_bound = waist**2 / 2 * transform_bound
    profile = np.exp(-((space.r / waist) ** 2))
    intensity_bound = 2 * profile * amplitude_bound + amplitude_bound**2
    # Only the active band is a forward power port: the flux has no closure
    # outside it, so the pulse is confined to it.
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9).build(grid)
    temporal = temporal * grid.active_mask
    field = analytic_field(grid, temporal)[0, 0]
    power = np.abs(field) ** 2
    spectrum = waist**2 / 2 * np.exp(-((space.k_r * waist) ** 2) / 4)
    # The vacuum closure G applies the obliquity sqrt(1 - (c k_r/omega)^2). To first
    # order it is 1 - (c k_r/omega)^2/2, and k_r^2 is -laplacian, so the Gaussian's
    # G = E (1 + alpha(r)/omega^2) with alpha = 2 c^2 (r^2/w^2 - 1)/w^2. The second
    # order is ~(c/(omega w))^4 ~ 1e-11 of the peak, inside the Hankel budget below.
    omega = grid.omega
    inverse_square = np.zeros(grid.n_omega)
    live = grid.active_mask & (omega > 0)
    inverse_square[live] = omega[live] ** -2.0
    alpha = 2 * C**2 * ((space.r / waist) ** 2 - 1) / waist**2
    correction = np.real(
        field * np.conj(analytic_field(grid, inverse_square * temporal)[0, 0])
    )
    expected = (
        0.5
        * EPS0
        * C
        * profile[:, None] ** 2
        * (power + alpha[:, None] * correction[None, :])
    )
    bound = 0.5 * EPS0 * C * intensity_bound[:, None] * power
    bound += gamma(64 + 16 * int(np.log2(grid.n_t))) * expected.max()
    measured = spatial_intensity(
        spectrum[:, None, None] * temporal, _vacuum(grid, space)
    )
    assert np.all(np.abs(measured - expected) <= bound)


def test_fluence_widths_integrate_time_instead_of_selecting_the_peak(grid, bessel_case):
    space, _, state, _ = bessel_case
    # Two coherent Bessel fields with disjoint frequency columns have no cross
    # term after integration over the complete temporal period (Parseval).
    bins = np.flatnonzero(grid.active_mask)[[1, 4]]
    state[:] = 0
    state[0, 0, bins[0]] = 1
    state[1, 0, bins[1]] = 2j
    coefficients = space.spectral_weights[:2] / (2 * np.pi)
    # Each column's fluence is weighted by its vacuum admittance at its one bin.
    eta = [_obliquity(grid, space.k_r[j])[bins[j]] for j in range(2)]
    expected_profile = eta[0] * (coefficients[0] * j0(space.k_r[0] * space.r)) ** 2
    expected_profile += 4 * eta[1] * (coefficients[1] * j0(space.k_r[1] * space.r)) ** 2
    expected_radius = np.sqrt(
        2
        * np.sum(space.radial_weights * space.r**2 * expected_profile)
        / np.sum(space.radial_weights * expected_profile)
    )
    bound = gamma(16 * space.n_r + 16 * int(np.log2(grid.n_t)))
    assert_allclose(
        beam_radius(state, _vacuum(grid, space)), expected_radius, rtol=bound, atol=0
    )


def test_a_flat_top_keeps_its_interpolated_widths(grid):
    # A truncated Fourier-Bessel series puts the first samples of a flat top a
    # ripple above the separately reconstructed axis value. The thresholds are
    # fractions of the peak, so the widths stay defined; the analytic 1/e^2 radius
    # of a super-Gaussian is its waist at every order, and its FWHM follows from
    # exp(-2 (r/w)^(2 n)) = 1/2. Both are asserted to one radial sample spacing,
    # the interpolation's own resolution.
    space = RadialGrid(1.2e-3, 256)
    waist, order = 200e-6, 8
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9)
    state = SpatialPulse(temporal, waist, order=order).build(
        grid, space, medium=SpatialMedium()
    )
    flux = _vacuum(grid, space)
    fluence = spatial_fluence(state, flux)
    axis = float(np.sum(on_axis_spectrum(state, flux)) * grid.d_omega)
    assert float(np.max(fluence)) > axis, "the ripple this test exists for is absent"
    spacing = float(np.max(np.diff(space.r)))
    assert abs(beam_radius(state, flux, method="1/e2") - waist) <= spacing
    expected_fwhm = 2 * waist * (math.log(2) / 2) ** (1 / (2 * order))
    assert abs(fwhm_r(state, flux) - expected_fwhm) <= spacing


@pytest.mark.parametrize(
    "fluence, axis, expected",
    [
        ([0.0, 0.0, 0.0], 0.0, (np.nan, np.nan)),
        ([1.0, 2.0, 1.0], 0.0, (np.nan, np.nan)),
        ([1.0, 1.0, 1.0], 1.0, (np.nan, np.nan)),
        ([0.75, 0.5, 0.0], 1.0, (4 - 4 * np.exp(-2), 4.0)),
        # A mild ring: the thresholds are fractions of the peak 1.2, the axis lies
        # above both, and the crossings are found beyond the ring.
        ([1.2, 0.5, 0.0], 1.0, (4 - 4.8 * np.exp(-2), 2 * (1 + 0.6 / 0.7))),
        # Hollow at half maximum only: the 1/e^2 radius is defined, the FWHM is not.
        ([1.2, 0.5, 0.0], 0.3, (4 - 4.8 * np.exp(-2), np.nan)),
    ],
)
def test_interpolated_fluence_widths_and_undefined_limits(fluence, axis, expected):
    # Piecewise-linear profiles make the interpolation oracle exact algebra.
    _, radius, full = radial_fluence_widths(
        np.array([1.0, 2.0, 4.0]), np.asarray(fluence), np.ones(3), axis
    )
    # Eight operations in the interpolation, with NaN preserved for undefined widths.
    assert_allclose((radius, full), expected, rtol=gamma(8), atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "radius,fluence,weights,axis",
    [
        ([1.0], [1.0], [1.0], 1.0),
        ([2.0, 1.0], [1.0, 0.0], [1.0, 1.0], 1.0),
        ([1.0, 2.0], [1.0], [1.0, 1.0], 1.0),
        ([1.0, 2.0], [1.0, -1.0], [1.0, 1.0], 1.0),
        ([1.0, 2.0], [1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 2.0], [1.0, np.nan], [1.0, 1.0], 1.0),
        ([1.0, 2.0], [1.0, 0.0], [1.0, 1.0], -1.0),
        ([1.0, 2.0], [1.0, 0.0], [1.0, 1.0], np.inf),
        ([1.0, 2.0], [1e308, 1e308], [1e308, 1e308], 1e308),
        (["bad", "input"], [1.0, 0.0], [1.0, 1.0], 1.0),
        ([1.0 + 1j, 2.0], [1.0, 0.0], [1.0, 1.0], 1.0),
    ],
)
def test_radial_fluence_widths_refuse_invalid_or_unrepresentable_samples(
    radius, fluence, weights, axis
):
    with pytest.raises(GridError, match="radial fluence"):
        radial_fluence_widths(radius, fluence, weights, axis)


def test_empty_spatial_widths_and_statistic_dependencies(grid, bessel_case):
    space, _, state, _ = bessel_case
    flux = _vacuum(grid, space)
    state[:] = 0
    assert np.isnan(beam_radius(state, flux))
    assert np.isnan(beam_radius(state, flux, method="1/e2"))
    assert np.isnan(fwhm_r(state, flux))
    values = collect(
        default_spec("radial"), grid, SavedPlane(z=0.0, field=state), flux=flux
    )
    assert values["energy"].shape == ()
    assert values["energy"] == 0
    assert values["beam_radius"].shape == (2,)
    with pytest.raises(OutputError, match="spatial geometry"):
        collect(
            dataclasses.replace(default_spec("radial"), names=("fwhm_r",)),
            grid,
            SavedPlane(z=0.0, field=state),
        )
    with pytest.raises(GridError, match="method"):
        # Exercise production validation beneath the test session's Literal hook.
        inspect.unwrap(beam_radius)(state, flux, method="peak")


@pytest.mark.parametrize("damage", ["shape", "nan"])
def test_spatial_statistic_refuses_invalid_spectrum(grid, bessel_case, damage):
    space, _, state, _ = bessel_case
    if damage == "shape":
        state = state[:-1]
    else:
        state[0, 0, 0] = np.nan
    with pytest.raises(GridError, match="shape"):
        spatial_energy(state, _vacuum(grid, space))
