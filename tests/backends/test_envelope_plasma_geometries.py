# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise independent envelope populations across host geometry and chunk routes."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from backends.forced_threads import assert_threaded
from backends.host_cases_envelope_plasma import (
    build_envelope_mixture,
    geometry_case,
    geometry_options,
)
from backends.matched import native_thread_fields
from bounds import assert_within, device_backend
from sessions.runs import run as session_run
from solprop.backends.node_chunk import plan_node_chunk
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.backends.numpy_ref.envelope_plasma import envelope_populations
from solprop.backends.session import ExecutionOptions
from solprop.modes import MarcatiliMode, ModeCollection, transverse_quadrature
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.profiles import Linear
from solprop.stats import StatsSpec
from solprop.translate.problem import assemble
from solprop.translate.stage import scalars

PROBE = Path(__file__).with_name("host_cases_envelope_plasma.py")


@pytest.fixture(scope="module")
def envelope_mixture():
    return build_envelope_mixture()


def run(backend_class, problem, run_request, grid, options=None):
    backend = backend_class(problem, run_request, grid=grid, options=options)
    try:
        result = session_run(backend, problem)
        # The run's one save plane is its end: its summary spans the run.
        summary = result.observations[-1].summary
        assert summary.met_tolerance and summary.n_steps == 2
        return result.fields[-1], backend.memory_ledger()
    finally:
        backend.close()


@pytest.mark.medium
@pytest.mark.parametrize("geometry", ["modal", "radial", "free"])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_two_population_host_geometries_and_partial_chunks(
    thread_budget_lifted, envelope_mixture, geometry, n_pol, equivalence_backend_class
):
    problem, run_request, grid = geometry_case(envelope_mixture, geometry, n_pol)
    options = geometry_options(geometry)
    reference, ledger = run(NumpyRefBackend, problem, run_request, grid, options)
    actual, _ = run(equivalence_backend_class, problem, run_request, grid, options)
    assert problem.flags.n_channels == 2
    assert np.all(np.isfinite(actual))
    chunk = plan_node_chunk(problem, options).chunk
    assert chunk == (problem.transform.n_nodes if options is None else 3)
    assert ledger["envelope_source_streams"] == 2 * chunk * n_pol * grid.n_to * 16
    # The polynomial rate's integrals converge to rounding; FFTs, two-channel
    # scans and projections contribute < 128 eps per sample per RK stage.
    bound = 128 * grid.n_to * 7 * 2 * np.finfo(float).eps
    assert bound > 0
    np.testing.assert_allclose(
        actual, reference, rtol=bound, atol=bound * np.max(np.abs(reference))
    )


@pytest.mark.medium
def test_host_sources_are_repeatable_and_thread_invariant(tmp_path, envelope_mixture):
    """A repeat and a four-thread run of the two fixed Runge-Kutta steps.

    One computation on one schedule, its sums ordered by the threads: held to the
    geometry ladder's budget below, not to its bits. Each run is its own process, whose
    Kokkos pool has that thread count.
    """
    pytest.importorskip("solprop_native")
    if device_backend("native") != "none":
        pytest.skip("a device build's host space is serial")
    _, _, grid = geometry_case(envelope_mixture, "free", 2)
    names = ["field", "steps", "met_tolerance"]
    original, *others = [
        native_thread_fields(PROBE, threads, names, tmp_path) for threads in (1, 1, 4)
    ]
    assert_threaded(
        others[1]["sizing"], 4, "kernels.stage", "kernels.response", "fft.fft"
    )
    bound = 128 * grid.n_to * 7 * 2 * np.finfo(float).eps
    for threads, repeated in zip((1, 4), others, strict=True):
        assert repeated["met_tolerance"] and int(repeated["steps"]) == 2
        assert_within(
            repeated["field"], original["field"], bound, what=f"{threads} threads"
        )
    assert original["met_tolerance"] and int(original["steps"]) == 2


@pytest.mark.medium
def test_vector_mode_rows_match_state_polarisation(
    thread_budget_lifted, envelope_mixture, backend_class
):
    mixture, responses = envelope_mixture
    original, original_request, grid = geometry_case(envelope_mixture, "modal", 2)
    mode = MarcatiliMode(30e-6, mixture=mixture, cladding=1.45, loss=False)
    results = []
    for layout in ("state", "rows"):
        modes = ModeCollection([mode])
        initial = original.initial.field0
        if layout == "rows":
            modes = ModeCollection.with_polarisations([mode])
            initial = initial.transpose(1, 0, 2).copy()
        rule = transverse_quadrature(modes, n_r=4, radial_rule="gauss_kronrod")
        problem, run_request = assemble(
            grid,
            modes,
            responses,
            initial,
            quadrature=rule,
            controller=original_request.stepper.controller,
            save_planes=[grid.length],
            dz_init=grid.length / 2,
            dz_max=grid.length / 2,
            schedule=original_request.stepper.schedule,
        )
        result, _ = run(
            backend_class, problem, run_request, grid, ExecutionOptions(node_chunk=3)
        )
        results.append(result.reshape(2, -1))
    # The two representations sum identical Cartesian sources with different
    # matrix strides; the bound covers all temporal and quadrature reductions.
    bound = 128 * grid.n_to * rule.n_nodes * 7 * 2 * np.finfo(float).eps
    np.testing.assert_allclose(
        results[1], results[0], rtol=bound, atol=bound * np.max(np.abs(results[0]))
    )


@pytest.mark.medium
def test_taper_and_density_paths_reach_each_envelope_population(
    thread_budget_lifted, envelope_mixture, equivalence_backend_class
):
    mixture, responses = envelope_mixture
    original, original_request, grid = geometry_case(envelope_mixture, "modal", 2)
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
    problem, run_request = assemble(
        grid,
        mode,
        responses,
        original.initial.field0,
        controller=original_request.stepper.controller,
        save_planes=[grid.length],
        dz_max=grid.length / 2,
        dz_init=grid.length / 2,
        schedule=original_request.stepper.schedule,
    )
    start = scalars(problem, 0.0, 0.0)
    end = scalars(problem, grid.length, grid.length)
    assert start.taper_scale != end.taper_scale
    assert all(density > 0 for density in start.plasma_densities)
    np.testing.assert_array_equal(end.plasma_densities, 0)
    reference, _ = run(NumpyRefBackend, problem, run_request, grid)
    actual, _ = run(equivalence_backend_class, problem, run_request, grid)
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
    envelope_mixture, geometry, n_pol, equivalence_backend_class
):
    problem, run_request, grid = geometry_case(envelope_mixture, geometry, n_pol)
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
    run_request = run_request.for_run(stats=spec)
    results = []
    for cls in (NumpyRefBackend, equivalence_backend_class):
        backend = cls(
            problem,
            run_request,
            grid=grid,
            options=geometry_options(geometry),
        )
        try:
            result = session_run(backend, problem, export=None)
            # The launch plane's statistics and the end's, the run's one save plane.
            initial = result.observations[0].stats
            final = result.observations[-1].stats
            taken = result.joined[1]
            for name in periodic:
                # The final periodic row and saved-plane scan see the same field, so
                # they are one evaluation repeated: within the statistics' budget.
                np.testing.assert_allclose(
                    taken.values[name][-1],
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
    problem, _, grid = geometry_case(envelope_mixture, "modal", 2)
    assembly = problem.responses.envelope_plasma
    assert assembly is not None
    modal = passes.to_time(
        problem.initial.field0,
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
