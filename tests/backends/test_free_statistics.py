# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Packed Cartesian reductions and fluence-width perturbation bounds."""

import dataclasses
import math
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.test_radial_nonlinear import make_problem
from bounds import gamma
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid
from solprop.output.schema import SavedPlane
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium, spatial_fluence
from solprop.stats import StatsSpec, collect, default_spec
from spatial_records import record_metric


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_free_packed_statistics_match_host_definitions(
    spatial_backend_class, kind, n_pol, request
):
    constructor = (
        FieldGrid.from_limits
        if kind == "field"
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    grid = constructor(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space, waist = FreeGrid(4e-4, 32, 5e-4, 40, window_factor=0), 100e-6
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9),
        waist,
        jones=(1,) if n_pol == 1 else (1, 0.6j),
    )
    # make_problem's medium is the vacuum.
    medium = SpatialMedium()
    flux = SpatialFlux.resolve(grid, space, medium)
    initial = pulse.build(grid, space, medium=medium) * grid.active_mask
    spec = default_spec("free")
    reference = collect(spec, grid, SavedPlane(z=0.0, field=initial), flux=flux)
    # FFTs plus positive space/time reductions bound the fluence perturbation.
    # All input-dependent conditioning is computed before backend construction.
    relative = gamma(256 * (space.n_columns + grid.n_to))
    fluence = spatial_fluence(initial, flux)
    maximum, mass = float(fluence.max()), float(fluence.sum())
    perturbation = relative * maximum
    mass_ratio = space.n_columns * perturbation / mass
    extent = float(space.r.max())
    center_error = 2 * extent * mass_ratio / (1 - mass_ratio)
    variance_error = (
        extent**2 * mass_ratio / (1 - mass_ratio)
        + 4 * extent * center_error
        + center_error**2
    )
    radius = float(reference["beam_radius"][0])
    moment_error = 2 * variance_error / radius
    dx, dy = space.x[1] - space.x[0], space.y[1] - space.y[0]
    samples = fluence.reshape(space.n_y, space.n_x)
    lipschitz = math.hypot(
        float(np.max(abs(np.diff(samples, axis=1)))) / dx,
        float(np.max(abs(np.diff(samples, axis=0)))) / dy,
    )
    profile_error = (
        perturbation
        + (lipschitz + 2 * perturbation * math.hypot(1 / dx, 1 / dy)) * center_error
    )
    # Every Gaussian crossing lies in [.5w,1.25w]. At least one Cartesian
    # coordinate is >=r/sqrt(2); its interpolation-cell left edge is >=
    # (.5/sqrt(2)-h/w)w. The Gaussian slope there has this lower bound,
    # using the enclosing radius 1.5w. Bilinear weights are positive.
    h = max(dx, dy)
    slope = (
        4
        * (0.5 / math.sqrt(2) - h / waist)
        * math.exp(-2 * 1.5**2)
        * maximum
        / (math.sqrt(2) * waist)
    )
    assert slope > 0
    bounds = {
        name: np.asarray(relative * float(np.max(np.abs(values))))
        for name, values in reference.items()
    }
    bounds["beam_radius"] = np.array(
        [moment_error, (1 + math.exp(-2)) * profile_error / slope]
    )
    bounds["fwhm_r"] = 3 * profile_error / slope
    problem = make_problem(grid, space, initial, [])
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=7)
    )
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        actual = backend.stats(spec)
        for name, expected in reference.items():
            errors = np.abs(actual[name][0] - expected)
            assert np.all(errors <= bounds[name]), (name, errors, bounds[name])
            pairs = (
                [(float(np.max(errors)), float(np.max(bounds[name])))]
                if name == "on_axis_spectrum"
                else list(
                    zip(
                        np.ravel(errors),
                        np.broadcast_to(bounds[name], expected.shape).ravel(),
                        strict=True,
                    )
                )
            )
            for i, (value, limit) in enumerate(pairs):
                record_metric(
                    request,
                    f"Cartesian packed {name}[{i}] (catalogue units)",
                    f"{kind}, p={n_pol}",
                    value,
                    limit,
                    spatial_backend_class.name,
                )
            single = backend.stats(StatsSpec(names=(name,)))
            assert_allclose(single[name], actual[name], rtol=0, atol=0)
    finally:
        backend.close()
