# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cartesian Gaussian diffraction with independent lattice and exact-kz budgets."""

import math
from functools import partial

import numpy as np
import pytest
from scipy.special import erfc

from backends.test_radial_diffraction import diffraction_reference
from bounds import gamma
from solprop.constants import C
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.spatial import SpatialFlux, SpatialMedium, beam_radius
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble
from spatial_records import record_metric


def _vacuum(grid, space):
    """Return the vacuum measure of the plane at z = 0 (SEP 34)."""
    return SpatialFlux.resolve(grid, space, SpatialMedium())


def gaussian_tail(a, spacing, first):
    """Bound a decreasing positive Gaussian lattice by first term plus integral."""
    return math.exp(-a * first**2) + math.sqrt(math.pi / a) * erfc(
        math.sqrt(a) * first
    ) / (2 * spacing)


def gaussian_reference(space, waist, wave, distance):
    """Bound a finite analytic Fourier series against the continuum Gaussian."""
    a = waist**2 / 4 - 1j * distance / (2 * wave)
    scale = waist**2 / (4 * a)
    continuum = scale * np.exp(-(space.r**2) / (4 * a))
    dx, dy = np.diff(space.x)[0], np.diff(space.y)[0]
    lx, ly = dx * space.n_x, dy * space.n_y
    dkx, dky = 2 * math.pi / lx, 2 * math.pi / ly
    # Poisson summation: the infinite sampled spectrum reconstructs the sum
    # of spatial Gaussian images. At any sampled point |x|<=L/2, every
    # image is at least (m-1/2)L away. Product of the two 1-D image bounds.
    real_decay = float(np.real(1 / (4 * a)))
    image_x = 2 * gaussian_tail(real_decay, lx, lx / 2)
    image_y = 2 * gaussian_tail(real_decay, ly, ly / 2)
    images = abs(scale) * (image_x + image_y + image_x * image_y)
    # Omitted reciprocal-lattice coefficients start no earlier than pi/dx
    # or pi/dy; a union bound counts the corner twice, conservatively.
    spectral_decay = waist**2 / 4
    tail_x = 2 * gaussian_tail(spectral_decay, dkx, math.pi / dx)
    tail_y = 2 * gaussian_tail(spectral_decay, dky, math.pi / dy)
    sum_x = 1 + 2 * gaussian_tail(spectral_decay, dkx, dkx)
    sum_y = 1 + 2 * gaussian_tail(spectral_decay, dky, dky)
    truncation = math.pi * waist**2 / (lx * ly) * (tail_x * sum_y + tail_y * sum_x)
    phase_origin = np.exp(
        1j * (space.k_y[:, None] * space.y[0] + space.k_x[None, :] * space.x[0])
    ).ravel()
    spectrum = (
        math.pi
        * waist**2
        / (dx * dy)
        * np.exp(-(waist**2) * space.kperp_sq / 4)
        * phase_origin
    )
    # Taylor remainder of sqrt(1-s) on s<=1/4. Outside that interval,
    # both exact/clamped and paraxial propagators have modulus at most one.
    q2 = space.kperp_sq
    phase_error = np.where(
        q2 <= wave**2 / 4,
        np.minimum(distance * q2**2 / (8 * wave**3 * 0.75**1.5), 2),
        2,
    )
    nonparaxial = float(np.sum(np.abs(spectrum) * phase_error) / space.n_columns)
    arithmetic = gamma(64 * space.n_columns) + gamma(64) * distance * wave
    return continuum, scale, spectrum, images + truncation + nonparaxial + arithmetic


def propagate(spatial_backend_class, grid, space, start, distance):
    """Use the real driver and an eight-step schedule on either spatial grid."""
    stop = max(distance, 1e-5)
    problem = assemble(
        grid,
        None,
        [],
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[stop],
        dz_max=stop,
        dz_init=stop / 8,
        schedule=FixedSchedule.uniform(0, stop, stop / 8),
    )
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        if distance:
            backend.propagate_to(distance)
        return backend.observe().require_field()[0]
    finally:
        backend.close()


@pytest.fixture(params=["field", "envelope"])
def diffraction_grid(request):
    constructor = (
        FieldGrid.from_limits
        if request.param == "field"
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    return constructor(0.1, 800e-9, (600e-9, 1.2e-6), 150e-15)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_cartesian_gaussian_width_gouy_exact_diffraction(
    diffraction_grid, spatial_backend_class, fraction, request
):
    grid = diffraction_grid
    space = FreeGrid(6e-4, 48, 7.5e-4, 60, window_factor=0)
    index = int(np.argmin(abs(grid.omega - grid.omega0)))
    wave, waist = grid.omega[index] / C, 80e-6
    distance = fraction * wave * waist**2 / 2
    continuum, axis, spectrum, bound = gaussian_reference(space, waist, wave, distance)
    phase_bound = math.asin(bound / abs(axis))
    expected_width = waist * math.sqrt(1 + fraction**2)
    # Positive sums propagate the pointwise amplitude envelope into the moment.
    radius_bound = moment_bound(space, continuum, bound, expected_width)
    assert abs(flux_width(waist, wave, distance) / expected_width - 1) < 2e-6
    expected_width = flux_width(waist, wave, distance)
    # The width model's next order, (k0 w)^-4 ~ 6e-12 of the width.
    radius_bound += 1e-11 * expected_width
    start = np.zeros((space.n_columns, 1, grid.n_omega), complex)
    start[:, 0, index] = spectrum
    state = propagate(spatial_backend_class, grid, space, start, distance)
    physical = np.fft.ifft2(state[:, 0, index].reshape(space.n_y, space.n_x)).ravel()
    assert np.max(np.abs(physical - continuum)) < bound
    measured_axis = np.sum(space.on_axis * state[:, 0, index])
    assert abs(np.angle(measured_axis / axis)) < phase_bound
    assert abs(beam_radius(state, _vacuum(grid, space)) - expected_width) < radius_bound
    for label, value, limit in (
        ("amplitude", np.max(np.abs(physical - continuum)), bound),
        ("Gouy phase (rad)", abs(np.angle(measured_axis / axis)), phase_bound),
        (
            "moment width (m)",
            abs(beam_radius(state, _vacuum(grid, space)) - expected_width),
            radius_bound,
        ),
    ):
        record_metric(
            request,
            f"Cartesian diffraction {label}",
            f"{grid.kind}, z/zR={fraction}",
            value,
            limit,
            spatial_backend_class.name,
        )


def flux_width(waist, wave, distance):
    """Return a Gaussian's second-moment width of axial-flux fluence in vacuum.

    The fluence is Re(E conj(G)), G the field filtered by the obliquity
    eta = 1 - beta k^2 + O(k^4), beta = 1/(2 k0^2) (SEP 34). For the angular
    spectrum X = exp(-p k^2), with |p|^2 and Re(p) of the paraxial Gaussian,
    Parseval gives <r^2> = [4|p|^2 (<k^2> - beta <k^4>) + 4 beta Re(p) <k^2>]
    / (1 - beta <k^2>), with <k^2> = 2/w^2 and <k^4> = 8/w^4 of |X|^2. At the waist
    this is w (1 + 1/(2 k0^2 w^2)), 1.3e-6 wider for 80 um at 800 nm; the
    neglected order is (k0 w)^-4 of the width.
    """
    p = complex(waist**2 / 4, distance / (2 * wave))
    beta = 1 / (2 * wave**2)
    k2, k4 = 2 / waist**2, 8 / waist**4
    moment = (4 * abs(p) ** 2 * (k2 - beta * k4) + 4 * beta * p.real * k2) / (
        1 - beta * k2
    )
    return math.sqrt(2 * moment)


def moment_bound(space, profile, amplitude_error, width):
    """Bound the Cartesian positive-sum second moment, including sampling."""
    intensity = np.abs(profile) ** 2
    error = 2 * np.abs(profile) * amplitude_error + amplitude_error**2
    weights = space.spatial_weights
    mass = np.dot(weights, intensity)
    moment = np.dot(weights * space.r**2, intensity)
    mass_error = np.dot(weights, error)
    moment_error = np.dot(weights * space.r**2, error)
    assert mass_error < mass and moment_error < moment
    discrete = math.sqrt(2 * moment / mass)
    ratio = (moment_error / moment + mass_error / mass) / (1 - mass_error / mass)
    # The exact analytic Gaussian sampled on these axes fixes the quadrature
    # defect before propagation. Its centred symmetry fixes the centroid.
    return (
        discrete * ratio + abs(discrete - width) + gamma(16 * space.n_columns) * width
    )


def test_cartesian_and_radial_agree_with_resampling_budget(
    diffraction_grid, spatial_backend_class, request
):
    grid = diffraction_grid
    space = FreeGrid(6e-4, 48, 7.5e-4, 60, window_factor=0)
    radial = RadialGrid(1.2e-3, 64, window_factor=0)
    index = int(np.argmin(abs(grid.omega - grid.omega0)))
    wave, waist = grid.omega[index] / C, 80e-6
    distance = wave * waist**2 / 2
    _, _, spectrum, free_bound = gaussian_reference(space, waist, wave, distance)
    _, amplitude, radial_bound, _, radial_spectrum = diffraction_reference(
        grid, radial, index, waist, distance
    )
    # Linear radial interpolation: |f''| <= |A| (1/(2|a|)+r²/(4|a|²))
    # times exp(-b r²); r² exp(-b r²)<=1/(b e). Error is h² sup|f''|/8.
    a = waist**2 / 4 - 1j * distance / (2 * wave)
    decay = np.real(1 / (4 * a))
    curvature = abs(amplitude) * (
        1 / (2 * abs(a)) + 1 / (4 * abs(a) ** 2 * decay * math.e)
    )
    interpolation = np.max(np.diff(radial.r)) ** 2 * curvature / 8
    bound = free_bound + float(np.max(radial_bound)) + interpolation
    starts = []
    for transverse, values in ((space, spectrum), (radial, radial_spectrum)):
        start = np.zeros((transverse.n_columns, 1, grid.n_omega), complex)
        start[:, 0, index] = values
        starts.append(
            propagate(spatial_backend_class, grid, transverse, start, distance)
        )
    physical = np.fft.ifft2(
        starts[0][:, 0, index].reshape(space.n_y, space.n_x)
    ).ravel()
    radial_physical = radial.hankel.inverse(starts[1][:, 0, index])
    valid = (space.r >= radial.r[0]) & (space.r <= radial.r[-1])
    resampled = np.interp(space.r[valid], radial.r, radial_physical)
    assert np.max(np.abs(physical[valid] - resampled)) < bound
    record_metric(
        request,
        "Cartesian/radial resampled amplitude",
        grid.kind,
        np.max(np.abs(physical[valid] - resampled)),
        bound,
        spatial_backend_class.name,
    )
