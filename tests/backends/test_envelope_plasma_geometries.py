# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise independent envelope populations across host geometry and chunk routes."""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from bounds import assert_within
from mixture_cases import AnalyticRate, mixture_case
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.backends.numpy_ref.envelope_plasma import envelope_populations
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.modes import MarcatiliMode, ModeCollection, transverse_quadrature
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import assemble
from solprop.translate.stage import scalars


@dataclass(frozen=True)
class InstantaneousRate(AnalyticRate):
    """Declare the synthetic quadratic magnitude rate's physical convention."""

    convention: str = "instantaneous-isotropic"


@pytest.fixture(scope="module")
def envelope_mixture():
    mixture, _, _, _, _ = mixture_case(kerr=False)
    responses = tuple(
        EnvelopePlasmaResponse(
            component.species,
            build_rate_table(
                InstantaneousRate(scale=(index + 1) * 1e13),
                (1e7, 1e11),
                rtol=1e-7,
            ),
            preionisation=component.preionisation,
            species=component.label,
        )
        for index, component in enumerate(mixture.components)
    )
    return mixture, responses


def geometry_case(envelope_mixture, geometry, n_pol):
    mixture, responses = envelope_mixture
    length = 8e-6
    grid = EnvelopeGrid.from_limits(length, 800e-9, (500e-9, 1.2e-6), 80e-15)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-7)
    jones = (1,) if n_pol == 1 else (1, 0.35j)
    kwargs = {}
    if geometry in ("radial", "free"):
        space = (
            RadialGrid(6e-4, 8, window_factor=0.0)
            if geometry == "radial"
            else FreeGrid(3e-4, 4, 4e-4, 4, window_factor=0.0)
        )
        medium = SpatialMedium(mixture)
        initial = SpatialPulse(pulse, 80e-6, jones=jones).build(
            grid, space, medium=medium
        )
        mode = None
        kwargs = {"spatial_grid": space, "medium": medium}
    else:
        mode = MarcatiliMode(30e-6, mixture=mixture, cladding=1.45, loss=False)
        initial = pulse.build(grid) * np.asarray(jones)[None, :, None]
    problem = assemble(
        grid,
        mode,
        responses,
        initial,
        controller=PIController(rtol=1e-7, atol=1e-14),
        save_planes=[length],
        dz_max=length / 2,
        dz_init=length / 2,
        schedule=FixedSchedule.uniform(0.0, length, length / 2),
        **kwargs,
    )
    if geometry != "modal":
        problem = replace(problem, transform=replace(problem.transform, node_chunk=3))
        assert problem.transform.n_nodes % 3 != 0
    return problem, grid


def run(backend_class, problem, grid, *, threads=1):
    backend = (
        NumpyRefBackend(problem, grid=grid)
        if backend_class is NumpyRefBackend
        else NumbaCpuBackend(
            problem,
            grid=grid,
            plan_flag="estimate",
            threads=threads,
            blas_threads=1,
            fftw_threads=1,
            regime="large" if threads > 1 else "small",
        )
    )
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(grid.length)
        assert summary.met_tolerance and summary.n_steps == 2
        return backend.observe().require_field().copy(), backend.memory_ledger()
    finally:
        backend.close()


@pytest.mark.medium
@pytest.mark.parametrize("geometry", ["modal", "radial", "free"])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_two_population_host_geometries_and_partial_chunks(
    envelope_mixture, geometry, n_pol
):
    problem, grid = geometry_case(envelope_mixture, geometry, n_pol)
    reference, _ = run(NumpyRefBackend, problem, grid)
    actual, ledger = run(NumbaCpuBackend, problem, grid)
    assert problem.flags.n_channels == 2
    assert np.all(np.isfinite(actual))
    assert ledger["envelope_sources"] == (
        2 * problem.transform.node_chunk * n_pol * grid.n_to * 16
    )
    # The polynomial rate's integrals converge to rounding; FFTs, two-channel
    # scans and projections contribute < 128 eps per sample per RK stage.
    bound = 128 * grid.n_to * 7 * 2 * np.finfo(float).eps
    assert bound > 0
    np.testing.assert_allclose(
        actual, reference, rtol=bound, atol=bound * np.max(np.abs(reference))
    )


@pytest.mark.medium
def test_host_sources_are_repeatable_and_thread_invariant(envelope_mixture):
    """A repeat and a four-thread run of the two fixed Runge-Kutta steps.

    One computation on one schedule, its sums ordered by the threads: held to the
    geometry ladder's budget below, not to its bits.
    """
    problem, grid = geometry_case(envelope_mixture, "free", 2)
    original, _ = run(NumbaCpuBackend, problem, grid)
    bound = 128 * grid.n_to * 7 * 2 * np.finfo(float).eps
    for threads in (1, 4):
        repeated, _ = run(NumbaCpuBackend, problem, grid, threads=threads)
        assert_within(repeated, original, bound, what=f"{threads} threads")


@pytest.mark.medium
@pytest.mark.parametrize("backend_class", [NumpyRefBackend, NumbaCpuBackend])
def test_vector_mode_rows_match_state_polarisation(envelope_mixture, backend_class):
    mixture, responses = envelope_mixture
    original, grid = geometry_case(envelope_mixture, "modal", 2)
    mode = MarcatiliMode(30e-6, mixture=mixture, cladding=1.45, loss=False)
    results = []
    for layout in ("state", "rows"):
        modes = ModeCollection([mode])
        initial = original.initial.field0[0]
        if layout == "rows":
            modes = ModeCollection.with_polarisations([mode])
            initial = initial.transpose(1, 0, 2).copy()
        rule = transverse_quadrature(modes, n_r=4, radial_rule="gauss_kronrod")
        problem = assemble(
            grid,
            modes,
            responses,
            initial,
            quadrature=rule,
            controller=original.stepper.controller,
            save_planes=[grid.length],
            dz_init=grid.length / 2,
            dz_max=grid.length / 2,
            schedule=original.stepper.schedule,
        )
        problem = replace(problem, transform=replace(problem.transform, node_chunk=3))
        result, _ = run(backend_class, problem, grid)
        results.append(result.reshape(2, -1))
    # The two representations sum identical Cartesian sources with different
    # matrix strides; the bound covers all temporal and quadrature reductions.
    bound = 128 * grid.n_to * rule.n_nodes * 7 * 2 * np.finfo(float).eps
    np.testing.assert_allclose(
        results[1], results[0], rtol=bound, atol=bound * np.max(np.abs(results[0]))
    )


@pytest.mark.medium
def test_taper_and_density_paths_reach_each_envelope_population(envelope_mixture):
    mixture, responses = envelope_mixture
    original, grid = geometry_case(envelope_mixture, "modal", 2)
    mixture = replace(
        mixture,
        components=tuple(
            replace(
                component,
                density=Linear(
                    float(component.density(np.array([0.0]))[0]), 0.0, grid.length
                ),
            )
            for component in mixture.components
        ),
    )
    mode = MarcatiliMode(
        Linear(30e-6, 27e-6, grid.length),
        mixture=mixture,
        cladding=1.45,
        loss=False,
    )
    problem = assemble(
        grid,
        mode,
        responses,
        original.initial.field0[0],
        controller=original.stepper.controller,
        save_planes=[grid.length],
        dz_max=grid.length / 2,
        dz_init=grid.length / 2,
        schedule=original.stepper.schedule,
    )
    start = scalars(problem, 0.0, 0.0)
    end = scalars(problem, grid.length, grid.length)
    assert start.taper_scale != end.taper_scale
    assert all(density > 0 for density in start.plasma_densities)
    np.testing.assert_array_equal(end.plasma_densities, 0)
    reference, _ = run(NumpyRefBackend, problem, grid)
    actual, _ = run(NumbaCpuBackend, problem, grid)
    # Two stages of profile-table interpolation add to the same 128-eps/sample
    # source and Fourier budget used by the other geometry witnesses.
    bound = 128 * grid.n_to * 7 * 2 * np.finfo(float).eps
    np.testing.assert_allclose(
        actual, reference, rtol=bound, atol=bound * np.max(np.abs(reference))
    )


@pytest.mark.medium
@pytest.mark.parametrize("geometry", ["modal", "radial", "free"])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_envelope_statistics_use_cycle_rates_on_all_host_geometries(
    envelope_mixture, geometry, n_pol
):
    problem, grid = geometry_case(envelope_mixture, geometry, n_pol)
    names = (
        "electron_density",
        "peak_ionisation_rate",
        "species_ionisation",
        "species_electron_density",
        "species_peak_ionisation_rate",
        "species_density",
        "species_ionisation_energy",
    )
    periodic = names[:4]
    spec = StatsSpec(names=names, periodic=periodic, period=1)
    # Roundoff in SVD/Gram axes, FFTs and the two-stage RK sums is below this bound;
    # interpolation and angular budgets are shared by both.
    bound = 128 * grid.n_to * 7 * 2 * np.finfo(float).eps
    problem = replace(problem, stats_spec=spec)
    results = []
    for cls in (NumpyRefBackend, NumbaCpuBackend):
        backend = cls(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            initial = backend.stats(spec)
            backend.propagate_to(grid.length)
            final = backend.stats(spec)
            rows = backend.observe().step_stats
            for name in periodic:
                # The final periodic row and saved-plane scan see the same field, so
                # they are one evaluation repeated: within the statistics' budget.
                np.testing.assert_allclose(
                    rows.values[name][-1],
                    final[name],
                    rtol=bound,
                    atol=bound * max(float(np.max(np.abs(final[name]))), 1e-300),
                )
            results.append((initial, final))
        finally:
            backend.close()
    for reference, actual in zip(results[0], results[1], strict=True):
        for name in names:
            np.testing.assert_allclose(
                actual[name],
                reference[name],
                rtol=bound,
                atol=bound * max(float(np.max(np.abs(reference[name]))), 1e-300),
            )
    assert np.all(results[0][0]["species_ionisation"] >= 0)


def test_envelope_population_scan_matches_independent_response(envelope_mixture):
    problem, grid = geometry_case(envelope_mixture, "modal", 2)
    assembly = problem.responses.envelope_plasma
    assert assembly is not None
    modal = passes.to_time(
        problem.initial.field0[0],
        problem.grids.crop_index,
        grid.n_to,
        kind="envelope",
    )
    field = np.einsum("nm,mpt->npt", problem.transform.synthesis, modal)
    initials = tuple(float(c.preionisation(0.0)) for c in assembly.channels)
    fractions, peaks = envelope_populations(
        field, assembly, initials, problem.grids.dt_fine
    )
    for index, channel in enumerate(assembly.channels):
        response = EnvelopePlasmaResponse(
            envelope_mixture[0].components[index].species,
            channel.rate,
            preionisation=initials[index],
        )
        oracle = response.sources(field[0], 1.0, dt=problem.grids.dt_fine)
        # The production angular budget is 2e-6; this off-knot polynomial
        # witness also includes the independent Cartesian adaptive integration.
        np.testing.assert_allclose(
            fractions[index, 0], oracle.fraction[-1], rtol=2e-6, atol=1e-15
        )
        np.testing.assert_allclose(
            peaks[index, 0], np.max(oracle.ionisation_rate), rtol=2e-6, atol=1e-30
        )
