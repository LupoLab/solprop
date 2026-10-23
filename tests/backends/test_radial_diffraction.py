# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exact-kz diffraction against Gaussian paraxial fields with a derived remainder."""

import math
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hankel_oracles import complex_gaussian_bound, gamma, gaussian_axis_bound
from solprop.constants import C
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.hankel import QDHT
from solprop.spatial import SpatialMedium, beam_radius, fwhm_r
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble
from spatial_records import record_metric


@pytest.fixture(params=["field", "envelope"])
def case(request):
    constructor = (
        FieldGrid.from_limits
        if request.param == "field"
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    grid = constructor(0.1, 800e-9, (600e-9, 1.2e-6), 150e-15)
    return grid, RadialGrid(1.2e-3, 64, window_factor=0.0)


def diffraction_reference(grid, space, index, waist, distance):
    """Derive reference fields and absolute errors before any propagation."""
    wave = grid.omega[index] / C
    a = waist**2 / 4 - 1j * distance / (2 * wave)
    amplitude = (waist**2 / 4) / a
    profile = amplitude * np.exp(-(space.r**2) / (4 * a))
    dual = QDHT(space.hankel.zeros[-1] / space.radius, space.n_r)
    _, inverse_error = complex_gaussian_bound(dual, a)
    inverse_error *= waist**2 / 2
    axis_error = abs(amplitude) * gaussian_axis_bound(space.hankel, 1 / (4 * a))
    spectrum = waist**2 / 2 * np.exp(-(waist**2) * space.k_r**2 / 4)
    # Taylor's theorem for sqrt(1-s), with second derivative bounded on
    # [0,1/4], yields |kz-k+q^2/(2k)| <= q^4/[8k^3(3/4)^(3/2)].
    # |exp(ix)-exp(iy)| <= min(|x-y|,2), |J0|<=1. Beyond q=k/2,
    # both propagators have modulus <=1, including the evanescent clamp.
    phase_bound = np.where(
        space.k_r <= wave / 2,
        np.minimum(distance * space.k_r**4 / (8 * wave**3 * 0.75**1.5), 2),
        2.0,
    )
    nonparaxial = float(
        np.sum(space.spectral_weights / (2 * np.pi) * spectrum * phase_bound)
    )
    arithmetic = gamma(64) * distance * wave + gamma(64 + space.n_r)
    return (
        profile,
        amplitude,
        inverse_error + nonparaxial + arithmetic,
        axis_error + nonparaxial + arithmetic,
        spectrum,
    )


def width_bounds(space, profile, amplitude_error, width):
    """Propagate bounded intensity perturbations into the second moment."""
    intensity = np.abs(profile) ** 2
    error = 2 * np.abs(profile) * amplitude_error + amplitude_error**2
    mass = float(np.sum(space.radial_weights * intensity))
    moment = float(np.sum(space.radial_weights * space.r**2 * intensity))
    mass_error = float(np.sum(space.radial_weights * error))
    moment_error = float(np.sum(space.radial_weights * space.r**2 * error))
    assert mass_error < mass and moment_error < moment
    discrete_width = math.sqrt(2 * moment / mass)
    relative = (moment_error / moment + mass_error / mass) / (1 - mass_error / mass)
    # The sampled *analytic* Gaussian gives its deterministic quadrature defect.
    # It is evaluated before the propagated field; it cannot absorb a solver error.
    sampling = abs(discrete_width - width) + gamma(16 * space.n_r) * width
    return discrete_width * relative + sampling


@pytest.mark.parametrize("rayleigh_fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_gaussian_width_gouy_and_axis_intensity_include_nonparaxial_error(
    case, spatial_backend_class, rayleigh_fraction, n_pol, request
):
    grid, space = case
    index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    waist = 80e-6
    rayleigh = grid.omega[index] / C * waist**2 / 2
    distance = rayleigh_fraction * rayleigh
    profile, amplitude, error, axis_error, spectrum = diffraction_reference(
        grid, space, index, waist, distance
    )
    width = waist * math.sqrt(1 + rayleigh_fraction**2)
    radius_bound = width_bounds(space, profile, error, width)
    assert axis_error < abs(amplitude)
    phase_bound = math.asin(axis_error / abs(amplitude))
    axis_intensity_bound = 2 * abs(amplitude) * axis_error + axis_error**2
    # Linear interpolation on a Gaussian: max |I''| <= 4 I(0)/w^2.
    # Add pointwise intensity perturbations and invert the minimum derivative
    # on the enclosing cell. The independently known crossing locates that cell.
    interpolation_bounds = []
    positions = np.r_[0, space.r]
    max_intensity_error = max(
        float(np.max(2 * np.abs(profile) * error + error**2)), axis_intensity_bound
    )
    for fraction, diameter in ((math.exp(-2), 1), (0.5, 2)):
        root = width * math.sqrt(-math.log(fraction) / 2)
        right = int(np.searchsorted(positions, root))
        left = right - 1
        h = positions[right] - positions[left]
        derivative = (
            4
            * positions[left : right + 1]
            / width**2
            * abs(amplitude) ** 2
            * np.exp(-2 * positions[left : right + 1] ** 2 / width**2)
        )
        slope = float(np.min(derivative))
        assert slope > 0
        interpolation_bounds.append(
            diameter
            * (
                h**2 * abs(amplitude) ** 2 / (2 * width**2)
                + (1 + fraction) * max_intensity_error
            )
            / slope
        )
    start = np.zeros((space.n_r, n_pol, grid.n_omega), dtype=np.complex128)
    start[:, 0, index] = spectrum
    if n_pol == 2:
        start[:, 1, index] = 1j * spectrum
        start /= math.sqrt(2)
    stop = max(distance, rayleigh / 8)
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
        schedule=FixedSchedule.uniform(0.0, stop, stop / 8),
    )
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        if distance:
            backend.propagate_to(distance)
        state = backend.observe().require_field()[0]
        radial_spectrum = state[:, 0, index] * math.sqrt(n_pol)
        measured = space.hankel.inverse(radial_spectrum)
        assert np.all(np.abs(measured - profile) <= error)
        axis = np.sum(space.spectral_weights / (2 * np.pi) * radial_spectrum)
        assert abs(np.angle(axis / amplitude)) <= phase_bound
        assert abs(abs(axis) ** 2 - abs(amplitude) ** 2) <= axis_intensity_bound
        assert_allclose(
            beam_radius(grid, space, state), width, rtol=0, atol=radius_bound
        )
        assert_allclose(
            beam_radius(grid, space, state, method="1/e2"),
            width,
            rtol=0,
            atol=interpolation_bounds[0],
        )
        assert_allclose(
            fwhm_r(grid, space, state),
            width * math.sqrt(2 * math.log(2)),
            rtol=0,
            atol=interpolation_bounds[1],
        )
        label = f"{grid.kind}, p={n_pol}, z/zR={rayleigh_fraction:g}"
        metrics = (
            ("Gaussian profile (1)", np.max(np.abs(measured - profile)), np.max(error)),
            ("Gouy phase (rad)", abs(np.angle(axis / amplitude)), phase_bound),
            (
                "axis intensity / launch (1)",
                abs(abs(axis) ** 2 - abs(amplitude) ** 2),
                axis_intensity_bound,
            ),
            (
                "second-moment radius (m)",
                abs(beam_radius(grid, space, state) - width),
                radius_bound,
            ),
            (
                "1/e² radius (m)",
                abs(beam_radius(grid, space, state, method="1/e2") - width),
                interpolation_bounds[0],
            ),
            (
                "FWHM diameter (m)",
                abs(fwhm_r(grid, space, state) - width * math.sqrt(2 * math.log(2))),
                interpolation_bounds[1],
            ),
        )
        for name, deviation, bound in metrics:
            record_metric(
                request, name, label, deviation, bound, spatial_backend_class.name
            )
    finally:
        backend.close()
