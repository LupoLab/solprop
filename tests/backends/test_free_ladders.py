# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cartesian response and statistics ladders with bounded physical-column buffers."""

import dataclasses
import math
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.test_radial_nonlinear import (
    CHI3,
    LENGTH,
    QuadraticRate,
    derivative,
    make_problem,
    matched_parity_grids,
)
from bounds import gamma, ladder_bound
from legacy_geometry import LEGACY_GRID
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.constants import C
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.translate.ionisation import build_rate_table
from spatial_records import record_equivalence, record_metric


@pytest.fixture(
    params=["field_kerr", "envelope_kerr", "field_plasma", "envelope_raman"]
)
def free_case(request):
    family = request.param
    constructor = (
        FieldGrid.from_limits
        if family.startswith("field")
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    # Resolve the held-current collar while retaining the short non-plasma grids.
    window = 400e-15 if family.endswith("plasma") else 150e-15
    # The former geometry, which the stored records' bounds were derived on.
    grid = constructor(LENGTH, 800e-9, (600e-9, 1.2e-6), window, **LEGACY_GRID)
    space = FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7),
        80e-6,
        jones=(1,) if family.endswith("raman") else (1, 0.5j),
    )
    density = 1.0
    if family.endswith("plasma"):
        responses = [
            PlasmaResponse("Ar", build_rate_table(QuadraticRate(), (1e3, 1e10)))
        ]
        density = 1e22
    elif family.endswith("raman"):
        responses = [
            RamanResponse(
                (DampedOscillator(1e-8, 2e14, undamped_dephasing()),), thg=False
            )
        ]
    else:
        responses = [KerrResponse(CHI3, thg=False)]
    problem = make_problem(
        grid,
        space,
        pulse.build(grid, space, medium=SpatialMedium()),
        responses,
        density=density,
    )
    plasma_names = (
        ("electron_density", "peak_ionisation_rate")
        if family.endswith("plasma")
        else ()
    )
    problem = dataclasses.replace(
        problem,
        transform=dataclasses.replace(problem.transform, node_chunk=7),
        stats_spec=StatsSpec(names=plasma_names, periodic=plasma_names),
    )
    return family, grid, space, problem


def amplitude_bound(family, grid, space, backend, steps):
    """Count temporal and transverse FFT depths plus plasma scan roundoff."""
    depth = (
        math.ceil(math.log2(grid.n_to))
        + math.ceil(math.log2(space.n_x))
        + math.ceil(math.log2(space.n_y))
    )
    per_call = 32 * np.finfo(float).eps * depth
    if family.endswith("plasma"):
        per_call += gamma(12 * grid.n_to)
    evaluations = max(1, 7 * steps)
    return ladder_bound(backend, evaluations * per_call, grid.n_to, evaluations)


def recorded_family(family):
    """Distinguish the held-current fixture and its resolved collar from history."""
    return "field_plasma held current, 400 fs" if family == "field_plasma" else family


@pytest.mark.parametrize("steps", [0, 1, 8])
def test_cartesian_nonlinear_ladder(free_case, spatial_backend_class, steps, request):
    family, grid, space, problem = free_case
    bound = amplitude_bound(family, grid, space, spatial_backend_class.name, steps)
    assert bound > 0
    if steps == 0:
        reference = derivative(NumpyRefBackend, problem, grid)
        actual = derivative(spatial_backend_class, problem, grid)
    else:
        schedule = problem.stepper.schedule
        assert schedule is not None
        problem = dataclasses.replace(
            problem,
            stepper=dataclasses.replace(
                problem.stepper,
                schedule=type(schedule).uniform(0, LENGTH, LENGTH / steps),
            ),
        )
        values = []
        for implementation in (NumpyRefBackend, spatial_backend_class):
            backend = implementation(problem, grid=grid)
            try:
                backend.set_initial(problem.initial.field0, None)
                backend.propagate_to(LENGTH)
                values.append(backend.observe().require_field().copy())
            finally:
                backend.close()
        reference, actual = values
    assert_allclose(actual, reference, rtol=0, atol=bound * np.max(np.abs(reference)))
    record_equivalence(
        request,
        ("free", recorded_family(family)),
        spatial_backend_class.name,
        "N" if steps == 0 else steps,
        np.max(np.abs(actual - reference)) / np.max(np.abs(reference)),
        bound,
    )

    record_metric(
        request,
        f"Cartesian {recorded_family(family)} ladder (relative)",
        "N" if steps == 0 else steps,
        np.max(np.abs(actual - reference)) / np.max(np.abs(reference)),
        bound,
        spatial_backend_class.name,
    )


@pytest.mark.parametrize("steps", [0, 8])
def test_cartesian_statistics_fresh_and_cached(
    free_case, spatial_backend_class, steps, request
):
    family, grid, space, problem = free_case
    # Fresh statistics use the same initial state on both backends. Propagated
    # statistics add twice the amplitude ladder envelope for squared fields.
    bound = 2 * amplitude_bound(family, grid, space, spatial_backend_class.name, steps)
    bound += gamma(128 * space.n_columns + 32 * grid.n_to)
    names = ("energy", "peak_intensity", "on_axis_intensity", "on_axis_spectrum")
    if family.endswith("plasma"):
        names += ("electron_density", "peak_ionisation_rate")
    spec = StatsSpec(names=names)
    rows = []
    for implementation in (NumpyRefBackend, spatial_backend_class):
        backend = implementation(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            if steps:
                backend.propagate_to(LENGTH)
            # Read cached reductions before observe invalidates that route.
            rows.append(backend.stats(spec))
            state = backend.observe().require_field().copy()
            backend.set_initial(state, None)
            fresh = backend.stats(spec)
            for name in names:
                assert_allclose(
                    fresh[name],
                    rows[-1][name],
                    rtol=0,
                    atol=bound * np.max(np.abs(rows[-1][name])),
                )
        finally:
            backend.close()
    for name in names:
        assert_allclose(
            rows[1][name],
            rows[0][name],
            rtol=0,
            atol=bound * np.max(np.abs(rows[0][name])),
        )
        record_metric(
            request,
            f"Cartesian {recorded_family(family)} {name} (relative)",
            steps,
            np.max(np.abs(rows[1][name] - rows[0][name]))
            / np.max(np.abs(rows[0][name])),
            bound,
            spatial_backend_class.name,
        )


def test_cartesian_analytic_family_plasma_statistics_scan_the_whole_field(
    spatial_backend_class,
):
    """One-channel electron density and peak rate on the analytic family, fresh.

    Kerr without its harmonic puts a field grid on the analytic family, and the
    statistics column must then pack the analytic signal as the step path does.
    Packing the real projection instead halved the field and quartered the peak
    rate; the plasma ladder above has no Kerr term and so never compiles this family.
    """
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (600e-9, 1.2e-6), 400e-15)
    space = FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7),
        80e-6,
        jones=(1, 0.5j),
    )
    responses = [
        KerrResponse(CHI3, thg=False),
        PlasmaResponse("Ar", build_rate_table(QuadraticRate(), (1e3, 1e10))),
    ]
    problem = make_problem(
        grid,
        space,
        pulse.build(grid, space, medium=SpatialMedium()),
        responses,
        density=1e22,
    )
    # Kerr without its harmonic on a field grid: the analytic family (backends.md 3.1a).
    assert problem.flags.has_plasma and problem.flags.n_channels == 1
    kerr = problem.responses.kerr_coefficients
    assert kerr is not None and kerr.analytic_signal
    names = ("electron_density", "peak_ionisation_rate", "species_electron_density")
    problem = dataclasses.replace(
        problem,
        transform=dataclasses.replace(problem.transform, node_chunk=7),
        stats_spec=StatsSpec(names=names),
    )
    rows = {}
    for implementation in (NumpyRefBackend, spatial_backend_class):
        backend = implementation(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            rows[implementation.name] = backend.stats(StatsSpec(names=names))
        finally:
            backend.close()
    # Fresh statistics: the transforms and the two reductions, no step ladder.
    bound = gamma(128 * space.n_columns + 32 * grid.n_to)
    reference, candidate = rows["numpy_ref"], rows[spatial_backend_class.name]
    for name in names:
        scale = np.max(np.abs(reference[name]))
        assert scale > 0, name
        assert_allclose(candidate[name], reference[name], rtol=0, atol=bound * scale)
    # The one channel is the whole electron density, on both backends.
    for row in rows.values():
        assert_allclose(
            row["electron_density"],
            np.sum(row["species_electron_density"], axis=0),
            rtol=0,
            atol=bound * np.max(np.abs(row["electron_density"])),
        )


@pytest.mark.parametrize("steps", [1, 8])
def test_cartesian_field_envelope_and_scalar_vector_parity(
    spatial_backend_class, steps, request
):
    field_grid, envelope_grid, carrier_index = matched_parity_grids()
    carrier = field_grid.omega[carrier_index]
    space = FreeGrid(3e-4, 12, 4e-4, 16, collar_fraction=0)
    waist = 80e-6
    spatial = np.fft.fft2(
        (2e8 * np.exp(-((space.r / waist) ** 2))).reshape(space.n_y, space.n_x)
    ).ravel()
    duration = 30e-15 / math.sqrt(2 * math.log(2))
    temporal = np.exp(-((field_grid.t_fine / duration) ** 2))
    carrier_phase = (-1) ** carrier_index
    # Envelope and field share the temporal period, exact carrier-bin shift and
    # physical band. The real-field opposite-frequency tail is bounded by the
    # Gaussian Fourier tail plus the finite temporal wall, before either run.
    tail = 4 * math.exp(-((duration * carrier / 2) ** 2))
    tail += 4 * math.exp(-((field_grid.t_fine[0] / duration) ** 2))
    evaluations = 7 * steps
    rounding = evaluations * (
        16 * np.finfo(float).eps * math.ceil(math.log2(field_grid.n_to))
        + 2 * gamma(space.n_columns)
    )
    rounding += gamma(64) * carrier / C * LENGTH + gamma(16 * field_grid.n_to)
    bound = ladder_bound(
        spatial_backend_class.name, rounding + tail, field_grid.n_to, evaluations
    )
    rows = []
    statistics = []
    for grid, n_pol in (
        (field_grid, 1),
        (envelope_grid, 1),
        (field_grid, 2),
        (envelope_grid, 2),
    ):
        if grid.kind == "field":
            samples = (
                temporal
                * np.cos(2 * np.pi * carrier_index * np.arange(grid.n_to) / grid.n_to)
                * carrier_phase
            )
            spectrum = np.fft.rfft(samples)[grid.crop_index] * grid.crop_scale
        else:
            spectrum = np.fft.fft(temporal)[grid.crop_index] * grid.crop_scale
        start = np.zeros((space.n_columns, n_pol, grid.n_omega), complex)
        start[:, 0] = spatial[:, None] * spectrum
        problem = make_problem(
            grid, space, start, [KerrResponse(CHI3, thg=False)], steps=steps
        )
        backend = spatial_backend_class(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(LENGTH)
            state = backend.observe().require_field()[0]
            if n_pol == 2:
                np.testing.assert_array_equal(state[:, 1], np.zeros_like(state[:, 1]))
            indices = np.rint((grid.omega - carrier) / grid.d_omega).astype(int)
            active = grid.active_mask
            ordering = np.argsort(indices[active])
            values = state[:, 0, active][:, ordering] / grid.n_t
            if grid.kind == "field":
                values *= 2 * carrier_phase
            rows.append((indices[active][ordering], values))
            statistics.append(
                backend.stats(
                    StatsSpec(names=("energy", "on_axis_intensity", "peak_intensity"))
                )
            )
        finally:
            backend.close()
    for label, (indices, values) in zip(
        ("envelope scalar", "field vector", "envelope vector"), rows[1:], strict=True
    ):
        np.testing.assert_array_equal(indices, rows[0][0])
        assert_allclose(
            values, rows[0][1], rtol=0, atol=bound * np.max(np.abs(rows[0][1]))
        )
        record_metric(
            request,
            f"Cartesian field-scalar parity with {label} (relative)",
            steps,
            np.max(np.abs(values - rows[0][1])) / np.max(np.abs(rows[0][1])),
            bound,
            spatial_backend_class.name,
        )
    for result in statistics[1:]:
        for name, reference in statistics[0].items():
            assert_allclose(result[name], reference, rtol=2 * bound, atol=0)
