# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify saved and periodic spatial reductions against the host definitions."""

import dataclasses
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hankel_oracles import gamma
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.spatial import collect_spatial
from solprop.exceptions import BackendError
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.output.schema import SavedPlane
from solprop.profiles import Constant
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium, spatial_fluence
from solprop.stats import StatsSpec, collect, default_spec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble
from solprop.translate.radial import radial_transform
from spatial_records import record_metric


@pytest.fixture(params=["field", "envelope"])
def case(request):
    constructor = (
        FieldGrid.from_limits
        if request.param == "field"
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    grid = constructor(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = RadialGrid(6e-4, 32, collar_fraction=0.0)
    return grid, space


def statistic_bounds(grid, space, state, reference, flux):
    """Propagate FFT/dot-product bounds through the fluence interpolation."""
    # At most two transforms, polarisation sum, and a fixed transverse reduction.
    relative = gamma(
        128 * space.n_r + 32 * int(np.ceil(np.log2(grid.n_t))) + 8 * grid.n_t
    )
    bounds = {
        name: relative * float(np.max(np.abs(value)))
        for name, value in reference.items()
    }
    fluence = spatial_fluence(state, flux)
    axis = float(reference["on_axis_spectrum"].sum() * grid.d_omega)
    perturbation = relative * max(float(fluence.max()), axis)
    positions = np.r_[0, space.r]
    values = np.r_[axis, fluence]
    crossing_bounds = []
    for fraction, diameter in ((np.exp(-2), 1), (0.5, 2)):
        right = int(np.flatnonzero(values <= fraction * axis)[0])
        left = right - 1
        difference = values[left] - values[right]
        assert difference > 2 * perturbation
        # A linear crossing is r0+h*(f0-q*axis)/(f0-f1). Each endpoint
        # moves by delta; numerator by (1+q)delta, denominator by 2delta.
        crossing_bounds.append(
            diameter
            * (positions[right] - positions[left])
            * (3 + fraction)
            * perturbation
            / (difference - 2 * perturbation)
        )
    mass = np.sum(space.radial_weights * fluence)
    moment = np.sum(space.radial_weights * space.r**2 * fluence)
    mass_error = perturbation * np.sum(space.radial_weights)
    moment_error = perturbation * np.sum(space.radial_weights * space.r**2)
    assert mass > mass_error and moment > moment_error
    # Ratio perturbation followed by sqrt: |delta sqrt(x)| <= |delta x|/sqrt(x).
    second_relative = (moment_error / moment + mass_error / mass) / (
        1 - mass_error / mass
    )
    bounds["beam_radius"] = np.array(
        [
            float(reference["beam_radius"][0]) * second_relative
            + bounds["beam_radius"],
            crossing_bounds[0] + bounds["beam_radius"],
        ]
    )
    bounds["fwhm_r"] += crossing_bounds[1]
    return bounds


@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("steps", [0, 8])
def test_radial_statistics_match_the_host_without_field_observation(
    case, spatial_backend_class, n_pol, steps, request
):
    grid, space = case
    jones = (1 + 0j,) if n_pol == 1 else (1 + 0j, 0.6j)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 80e-6, jones=jones
    )
    start = pulse.build(grid, space, medium=SpatialMedium())
    flux = SpatialFlux.resolve(grid, space, SpatialMedium())
    spec = default_spec("radial")
    problem = assemble(
        grid,
        None,
        [],
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length / 8,
        schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / 8),
        stats_spec=spec,
    )
    # The stats test uses exactly the state at the measured plane; propagation
    # already has an independent rationalized dispersion oracle in radial_runs.

    oracle = NumpyRefBackend(problem, grid=grid)
    backend = spatial_backend_class(problem, grid=grid)
    try:
        oracle.set_initial(problem.initial.field0, None)
        if steps:
            oracle.propagate_to(grid.length)
        state = oracle.observe().require_field()[0]
        reference = collect(spec, grid, SavedPlane(z=0.0, field=state), flux=flux)
        bounds = statistic_bounds(grid, space, state, reference, flux)
        backend.set_initial(state[None], None)
        # Calling stats never calls observe(); the native lane transfers only its
        # packed reductions. All exposed statistics are requested together, then singly.
        actual = backend.stats(spec)
        for name, expected in reference.items():
            assert actual[name].shape == (1, *expected.shape)
            assert np.all(np.abs(actual[name][0] - expected) <= bounds[name]), name
            for index, (deviation, limit) in enumerate(
                zip(
                    np.abs(actual[name][0] - expected).reshape(-1),
                    np.broadcast_to(bounds[name], expected.shape).reshape(-1),
                    strict=True,
                )
            ):
                # A spectrum has one common absolute bound; store its maximum
                # instead of repeating every temporal bin in the accuracy figure.
                if name == "on_axis_spectrum":
                    record_metric(
                        request,
                        f"spatial statistic {name} (J s/m²)",
                        f"{grid.kind}, p={n_pol}, steps={steps}",
                        np.max(np.abs(actual[name][0] - expected)),
                        np.max(bounds[name]),
                        spatial_backend_class.name,
                    )
                    break
                record_metric(
                    request,
                    f"spatial statistic {name}[{index}] (catalogue units)",
                    f"{grid.kind}, p={n_pol}, steps={steps}",
                    deviation,
                    limit,
                    spatial_backend_class.name,
                )
            single = backend.stats(StatsSpec(names=(name,)))
            assert_allclose(single[name], actual[name], rtol=0, atol=0, equal_nan=True)
        periodic = dataclasses.replace(
            spec,
            names=("energy", "on_axis_intensity", "peak_intensity"),
            periodic=("energy", "on_axis_intensity", "peak_intensity"),
        )
        values = backend.stats(periodic)
        for name in periodic.names:
            assert_allclose(values[name], actual[name], rtol=0, atol=0)
        if backend.name == "native":
            ledger = backend.memory_ledger()
            assert ledger["spatial_statistics_geometry"] == space.n_r * 24
            assert ledger["spatial_statistics_partials"] == space.n_r * 3 * 8
    finally:
        backend.close()
        oracle.close()


def test_empty_spatial_widths_and_intensities(case, spatial_backend_class):
    grid, space = case
    spec = default_spec("radial")
    problem = assemble(
        grid,
        None,
        [],
        np.zeros((space.n_r, 1, grid.n_omega), dtype=np.complex128),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
    )
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        values = backend.stats(spec)
        for name in (
            "energy",
            "on_axis_intensity",
            "peak_intensity",
            "on_axis_spectrum",
        ):
            np.testing.assert_array_equal(values[name], np.zeros_like(values[name]))
        assert np.all(np.isnan(values["beam_radius"]))
        assert np.all(np.isnan(values["fwhm_r"]))
    finally:
        backend.close()


def test_spatial_statistics_require_geometry_and_named_dependencies(case):
    grid, space = case
    transform = radial_transform(space, grid, n_pol=1, z_max=grid.length)
    state = np.zeros((space.n_r, 1, grid.n_omega), dtype=np.complex128)
    spec = StatsSpec(names=("density",))
    with pytest.raises(BackendError, match="spatial geometry"):
        collect_spatial(
            spec,
            grid,
            dataclasses.replace(transform, geometry=None, kind="modal"),
            state,
            z=0.0,
            flux=None,
        )
    for profiles in (None, {}):
        with pytest.raises(BackendError, match="named profile"):
            collect_spatial(
                spec, grid, transform, state, z=0.0, flux=None, profiles=profiles
            )
    values = collect_spatial(
        spec,
        grid,
        transform,
        state,
        z=0.0,
        flux=None,
        profiles={"density": Constant(2.0)},
    )
    assert values["density"] == 2.0
    with pytest.raises(BackendError, match="not spatial saved-field"):
        collect_spatial(
            StatsSpec(names=("zdw",)), grid, transform, state, z=0.0, flux=None
        )
    # A field statistic is measured by the plane's flux, which it must be given.
    with pytest.raises(BackendError, match="plane measure"):
        collect_spatial(
            StatsSpec(names=("energy",)), grid, transform, state, z=0.0, flux=None
        )
