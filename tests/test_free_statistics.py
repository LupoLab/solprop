# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Centroid, angular-average and field-reconstruction spatial statistics."""

import math

import numpy as np
import pytest

from hankel_oracles import gamma
from solprop.exceptions import GridError
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import (
    SpatialFlux,
    SpatialMedium,
    beam_radius,
    cartesian_fluence_widths,
    on_axis_spectrum,
    spatial_energy,
    spatial_fluence,
)
from spatial_records import record_metric


@pytest.mark.parametrize("size", [48, 96, 192])
def test_cartesian_widths_of_shifted_gaussian_converge(size, request):
    waist, radius = 80e-6, 400e-6
    space = FreeGrid(radius, size, 1.2 * radius, size + 8, window_factor=0)
    x0, y0 = 0.13 * radius, -0.11 * radius
    profile = np.exp(
        -2 * ((space.x[None, :] - x0) ** 2 + (space.y[:, None] - y0) ** 2) / waist**2
    ).ravel()
    dx, dy = space.x[1] - space.x[0], space.y[1] - space.y[0]
    # For exp(-2r²/w²), |F_xx| and |F_yy| <=4/w². Bilinear error
    # <=(dx²+dy²)/(2w²), radial interpolation <=h²/(2w²). Include
    # the central-level error again, then divide by a conservative lower
    # slope 0.3/w for both threshold neighbourhoods [0.5w,1.2w].
    # Centroid/moment Gaussian lattice tails are exponentially small:
    # at least 4w to the edge and reciprocal exponent >=(pi*w/dmax)²/2.
    delta = (dx**2 + dy**2) / (2 * waist**2)
    interpolation_bound = (
        (2 * delta + (min(dx, dy) / 2) ** 2 / (2 * waist**2)) * waist / 0.3
    )
    lattice = math.exp(-0.5 * (np.pi * waist / max(dx, dy)) ** 2)
    tail = math.exp(-2 * ((radius - abs(x0)) / waist) ** 2)
    moment_bound = waist * (128 * (lattice + tail) + gamma(64 * space.n_columns))
    widths = cartesian_fluence_widths(space.x, space.y, profile)
    assert abs(widths[0] - waist) < moment_bound
    assert abs(widths[1] - waist) < interpolation_bound
    assert abs(widths[2] - waist * math.sqrt(2 * math.log(2))) < 2 * interpolation_bound
    for name, actual, expected, bound in zip(
        ("second moment", "1/e² radius", "FWHM"),
        widths,
        (waist, waist, waist * math.sqrt(2 * math.log(2))),
        (moment_bound, interpolation_bound, 2 * interpolation_bound),
        strict=True,
    ):
        record_metric(
            request,
            f"Cartesian shifted Gaussian {name} (m)",
            size,
            abs(actual - expected),
            bound,
        )


def test_cartesian_second_moment_uses_centroid_and_both_axes(request):
    space = FreeGrid(1.0, 12, 1.5, 16, window_factor=0)
    profile = np.zeros((space.n_y, space.n_x))
    profile[5, 4], profile[5, 8], profile[9, 4], profile[9, 8] = 1, 1, 1, 1
    expected = math.sqrt(
        2 * ((space.x[8] - space.x[4]) ** 2 + (space.y[9] - space.y[5]) ** 2) / 4
    )
    widths = cartesian_fluence_widths(space.x, space.y, profile.ravel())
    assert abs(widths[0] / expected - 1) < gamma(64 * space.n_columns)
    record_metric(
        request,
        "Cartesian four-point centroid radius (relative)",
        "12x16",
        abs(widths[0] / expected - 1),
        gamma(64 * space.n_columns),
    )
    assert all(math.isnan(one) for one in widths[1:])
    assert all(
        math.isnan(one)
        for one in cartesian_fluence_widths(space.x, space.y, np.zeros(space.n_columns))
    )
    uniform = cartesian_fluence_widths(space.x, space.y, np.ones(space.n_columns))
    assert all(math.isnan(one) for one in uniform[1:])


@pytest.mark.parametrize(
    "change", ["complex", "shape", "negative", "nonfinite", "axis"]
)
def test_cartesian_widths_refuse_invalid_data(change):
    space = FreeGrid(1, 8, 2, 6, window_factor=0)
    x, profile = space.x, np.ones(space.n_columns)
    if change == "complex":
        profile = profile.astype(complex)
    elif change == "shape":
        profile = profile[:-1]
    elif change == "negative":
        profile[0] = -1
    elif change == "nonfinite":
        profile[0] = np.nan
    else:
        x = x[::-1]
    with pytest.raises(GridError, match="Cartesian"):
        cartesian_fluence_widths(x, space.y, profile)


def test_cartesian_fluence_integrates_to_beam_energy(request):
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = FreeGrid(3e-4, 32, 4e-4, 40, window_factor=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9),
        80e-6,
        jones=(1, 0.4j),
    )
    medium = SpatialMedium()
    spectrum = pulse.build(grid, space, medium=medium)
    flux = SpatialFlux.resolve(grid, space, medium)
    bound = gamma(128 * (space.n_columns + grid.n_t))
    fluence = spatial_fluence(spectrum, flux)
    energy = spatial_energy(spectrum, flux)
    # The launch normalises the same measure, so the requested energy returns.
    assert abs(energy / 3e-9 - 1) < bound
    assert abs(np.dot(space.spatial_weights, fluence) / energy - 1) < bound
    assert beam_radius(spectrum, flux) > 0
    axis = float(np.sum(on_axis_spectrum(spectrum, flux)) * grid.d_omega)
    center = fluence.reshape(space.n_y, space.n_x)[space.n_y // 2, space.n_x // 2]
    assert abs(axis / center - 1) < bound
    for name, defect in (
        ("fluence energy", abs(np.dot(space.spatial_weights, fluence) / energy - 1)),
        ("on-axis fluence", abs(axis / center - 1)),
    ):
        record_metric(request, f"Cartesian {name} (relative)", "32x40", defect, bound)


@pytest.mark.parametrize("size", [48, 96])
def test_line_widths_of_a_shifted_gaussian(size):
    waist, radius = 80e-6, 400e-6
    line = LineGrid(radius, size, window_factor=0)
    x0 = 0.13 * radius
    profile = np.exp(-2 * (line.x - x0) ** 2 / waist**2)
    dx = line.x[1] - line.x[0]
    # The plane's bounds (above) with dy = 0: linear interpolation of
    # exp(-2x²/w²), |F''| <= 4/w², errs by at most dx²/(2w²), the half-pixel
    # radii add (dx/2)²/(2w²), over a lower slope 0.3/w near both levels. The
    # moment's lattice and tail terms are the plane's.
    interpolation_bound = (
        (2 * dx**2 / (2 * waist**2) + (dx / 2) ** 2 / (2 * waist**2)) * waist / 0.3
    )
    lattice = math.exp(-0.5 * (np.pi * waist / dx) ** 2)
    tail = math.exp(-2 * ((radius - abs(x0)) / waist) ** 2)
    moment_bound = waist * (128 * (lattice + tail) + gamma(64 * size))
    second, radius_e2, width = cartesian_fluence_widths(line.x, line.y, profile)
    assert abs(second - waist) < moment_bound
    assert abs(radius_e2 - waist) < interpolation_bound
    assert abs(width - waist * math.sqrt(2 * math.log(2))) < 2 * interpolation_bound


def test_line_widths_of_empty_and_edge_profiles():
    line = LineGrid(1.0, 8, window_factor=0)
    assert all(
        math.isnan(one) for one in cartesian_fluence_widths(line.x, line.y, np.zeros(8))
    )
    edge = np.zeros(8)
    edge[0] = 1.0
    second, radius_e2, width = cartesian_fluence_widths(line.x, line.y, edge)
    assert second == 0.0 and math.isnan(radius_e2) and math.isnan(width)


def test_line_widths_report_a_nonrepresentable_moment():
    # Every input is finite: the centroid sits on the first sample, so the far
    # sample's offset is 1.6e308 and twice it overflows, as the plane's test does.
    fluence = np.array([1.0, 0.0, 0.0])
    with pytest.raises(GridError, match="line fluence moments are not representable"):
        cartesian_fluence_widths(
            np.array([-8e307, 0.0, 8e307]), np.array([0.0]), fluence
        )
