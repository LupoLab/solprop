# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify short complete scalar spatial CPU sessions through the managed bridge."""

import math
import os
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.spatial_cases import spatial_case, spatial_collar
from backends.julia.spatial_observation_bounds import spatial_observation_bounds
from bounds import gamma
from sessions.executors import IDENTITY, Started, drive
from solprop.backends.events import ObservationReady, Outcome, join_attempts
from solprop.backends.julia import _calls
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.registry import select
from solprop.backends.session import ExecutionOptions, Start
from solprop.backends.spatial import collect_spatial
from solprop.constants import C
from solprop.exceptions import BackendError
from solprop.stats import StatsSpec
from solprop.translate.linop import delta_phi, unitary_phase
from solprop.translate.problem import Initial, Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def spatial_run(case, grid, *, julia, chunk=3):
    """Drive one Julia-owned or reference session with the identical fixed request."""
    if julia:
        setup_id = os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]
        owner = JuliaBackend(
            case.problem,
            case.request,
            grid=grid,
            options=ExecutionOptions(
                node_chunk=chunk, backend_options={"setup_id": setup_id}
            ),
        )
    else:
        owner = select("numpy_ref", case.problem, case.request, grid=grid)
    try:
        prepared = owner.prepare(case.problem, case.request)
        with prepared.start(Start.entrance(case.problem)) as session:
            result = drive(
                Started(session), export=IDENTITY, work_limit=2, deadline=0.0
            )
            assert result.status.outcome is Outcome.COMPLETED
            return result
    finally:
        owner.close()


def linear_budget(case):
    """Bound the two represented vacuum flows independently of measured outputs."""
    problem, request = case.problem, case.request
    assert request.replay is not None
    assert not problem.responses.raman_terms
    assert problem.responses.kerr_coefficients is None
    assert problem.responses.temporal_absorber is None
    geometry = problem.transform.geometry
    assert geometry is not None and not np.any(geometry.absorber_rate)
    steps = request.replay.schedule.n_steps
    length = float(request.replay.schedule.z_end)
    active = problem.grids.active_mask
    wave = problem.grids.omega[active] / C
    # These vacuum fixtures stay well away from the grazing square-root branch.
    # Two products/subtractions, sqrt and the frame subtraction are bounded in
    # absolute wave-number units, so cancellation in D never sets the tolerance.
    base = problem.route.flux.base
    assert np.max(base.kperp_sq) < 0.5 * np.min(wave**2)
    root_error = gamma(32) * float(np.max(wave)) * length
    phase = delta_phi(problem.linear, 0.0, length)
    assert np.max(np.real(phase)) <= 0
    # Fewer than 256 real operations per accepted linear flow, including the
    # exponential and exact-zero source folds in the RK/low-storage methods.
    rounding = gamma(256 * (steps + 1)) * (1 + float(np.max(abs(phase))))
    relative = 2 * (root_error + rounding)
    assert 0 < relative < 1e-8
    return relative * np.linalg.norm(problem.initial.field0.ravel())


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
@pytest.mark.parametrize("method", ["rk", "adams", "low_storage"])
@pytest.mark.parametrize("steps", [1, 8])
def test_scalar_spatial_linear_sessions_match_the_represented_flow(
    kind, geometry, method, steps
):
    case, grid = spatial_case(kind, geometry, method=method, steps=steps)
    bound = linear_budget(case)
    actual = spatial_run(case, grid, julia=True)
    expected = spatial_run(case, grid, julia=False)
    assert actual.status.attempts == expected.status.attempts == steps
    assert actual.status.rejected == expected.status.rejected == 0
    for field, reference in zip(actual.fields, expected.fields, strict=True):
        assert np.all(np.isfinite(field))
        assert np.linalg.norm((field - reference).ravel()) <= bound
        assert np.all(field[..., ~case.problem.grids.active_mask] == 0)


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_scalar_spatial_kerr_replays_meet_a_converged_reference_envelope(
    kind, geometry
):
    # A fixed global accuracy envelope, checked against three progressively
    # refined reference runs. No local-controller tolerance bounds this comparison.
    case, grid = spatial_case(kind, geometry, material=True, steps=8)
    references = []
    for steps in (8, 16, 32):
        refined, refined_grid = spatial_case(kind, geometry, material=True, steps=steps)
        references.append(spatial_run(refined, refined_grid, julia=False).fields[-1])
    scale = np.linalg.norm(references[-1].ravel())
    assert math.isfinite(scale) and scale > 0
    for coarse in references[:-1]:
        assert np.linalg.norm((coarse - references[-1]).ravel()) < 1e-11 * scale
    actual = spatial_run(case, grid, julia=True).fields[-1]
    assert np.linalg.norm((actual - references[-1]).ravel()) < 1e-8 * scale


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_material_free_spatial_absorption_keeps_the_physical_source(kind, geometry):
    # A represented positive physical loss still needs source evaluations when
    # every material response is absent. Use three short reference refinements
    # to certify the fixed global envelope, independently of Julia's result.
    references = []
    for steps in (8, 16, 32):
        refined, refined_grid = spatial_case(kind, geometry, steps=steps)
        references.append(
            spatial_run(spatial_collar(refined), refined_grid, julia=False).fields[-1]
        )
    scale = np.linalg.norm(references[-1].ravel())
    assert math.isfinite(scale) and scale > 0
    for coarse in references[:-1]:
        assert np.linalg.norm((coarse - references[-1]).ravel()) < 1e-10 * scale
    base, grid = spatial_case(kind, geometry)
    case = spatial_collar(base)
    actual = spatial_run(case, grid, julia=True).fields[-1]
    assert np.linalg.norm((actual - references[-1]).ravel()) < 1e-8 * scale
    weights = case.problem.transform.geometry.spectral_weights[:, None, None]
    initial_energy = np.sum(weights * abs(case.problem.initial.field0) ** 2)
    final_energy = np.sum(weights * abs(actual) ** 2)
    assert 0 < final_energy < (1 - 1e-5) * initial_energy


@pytest.mark.parametrize("geometry", ["cartesian", "line"])
@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_spatial_input_noise_exports_the_absolute_launch_phase(kind, geometry):
    case, grid = spatial_case(kind, geometry)
    launch = 2e-6
    noise = np.full(case.problem.shape, 0.125 + 0.0625j)
    noise[..., ~case.problem.grids.active_mask] = 0
    case = Translation(
        replace(
            case.problem,
            initial=Initial(case.problem.initial.field0, noise, "input", launch),
            flags=replace(case.problem.flags, has_noise=True),
        ),
        case.request,
    )
    owner = JuliaBackend(
        case.problem,
        case.request,
        grid=grid,
        options=ExecutionOptions(
            backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]}
        ),
    )
    try:
        prepared = owner.prepare(case.problem, case.request)
        with prepared.start(Start.entrance(case.problem)) as session:
            event = session.advance().events[0]
            assert isinstance(event, ObservationReady) and event.z == launch
            expected = case.problem.initial.field0 + noise
            expected[..., ~case.problem.grids.active_mask] = 0
            np.testing.assert_array_equal(
                session.export(event.ticket, IDENTITY), expected
            )
            phase = session.noise_phase(event.ticket)
            reference = unitary_phase(case.problem.linear, launch)[0]
            assert phase is not None and np.any(reference != 0)
            # Absolute root/frame error in this admitted vacuum fixture is
            # bounded by gamma(32) times the uncancelled wave number per engine.
            active = case.problem.grids.active_mask
            wave = case.problem.grids.omega[active] / C
            bound = 2 * gamma(32) * float(np.max(wave)) * launch
            assert bound > 0 and np.all(abs(phase - reference) <= bound)
    finally:
        owner.close()


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("method", ["rk", "adams", "low_storage"])
def test_spatial_observations_use_the_selected_field_and_physical_flux(
    kind, geometry, method
):
    case, grid = spatial_case(kind, geometry, method=method, resolved=True)
    names = (
        "energy",
        "peak_intensity",
        "on_axis_intensity",
        "on_axis_spectrum",
        "beam_radius",
        "fwhm_r",
        "quadrature_error",
    )
    spec = StatsSpec(names=names, periodic=("energy",), period=3)
    case = Translation(
        case.problem,
        replace(
            case.request, observation=replace(case.request.observation, stats=spec)
        ),
    )
    result = spatial_run(case, grid, julia=True)
    planes = [
        event
        for batch in result.batches
        for event in batch.events
        if isinstance(event, ObservationReady)
    ]
    for event, field in zip(planes, result.fields, strict=True):
        bounds = spatial_observation_bounds(case.problem, grid, field, event.z)
        reference = collect_spatial(
            StatsSpec(names=names[:-1], periodic=()),
            grid,
            case.problem.transform,
            field,
            z=event.z,
            flux=case.problem.route.flux,
            node_chunk=3,
        )
        assert tuple(event.stats) == names
        for name, bound in bounds.items():
            actual, expected = event.stats[name], reference[name]
            assert actual.shape == expected.shape
            assert not actual.flags.writeable
            assert np.all(np.isfinite(actual))
            assert np.all(abs(actual - expected) <= bound), name
        assert np.isnan(event.stats["quadrature_error"])
    _, rows = join_attempts(
        [
            (getattr(event, "history", None), getattr(event, "step_stats", None))
            for batch in result.batches
            for event in batch.events
        ]
    )
    assert tuple(rows.values) == ("energy",)
    assert rows.values["energy"].shape == (2,)
    assert np.all(np.diff(rows.z) > 0)
    assert not rows.values["energy"].flags.writeable


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
def test_spatial_exports_preserve_the_measure_and_reject_foreign_preparation(geometry):
    case, grid = spatial_case("envelope", geometry)
    options = ExecutionOptions(
        backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]}
    )
    owner = JuliaBackend(case.problem, case.request, grid=grid, options=options)
    other = JuliaBackend(case.problem, case.request, grid=grid, options=options)
    try:
        prepared = owner.prepare(case.problem, case.request)
        foreign = other.prepare(case.problem, case.request)
        with prepared.start(Start.entrance(case.problem)) as session:
            batch = session.advance()
            event = batch.events[0]
            assert isinstance(event, ObservationReady)
            raw = session.export(event.ticket, IDENTITY)
            power = session.export(event.ticket, replace(IDENTITY, power=True))
            # The spatial export contract ignores the modal normalization flag.
            np.testing.assert_array_equal(raw.view(np.uint64), power.view(np.uint64))
            destination = np.full(case.problem.shape, 7 + 2j)
            with pytest.raises(BackendError, match="another numerical preparation"):
                _calls.call_void(
                    session._runtime,
                    "export_session!",
                    destination,
                    session._engine,
                    event.ticket.generation,
                    foreign._preparation,
                    False,
                    False,
                )
            assert np.all(destination == 7 + 2j)
            session.acknowledge(batch.receipt)
    finally:
        owner.close()
        other.close()
