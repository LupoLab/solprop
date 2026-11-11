# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify short internal Julia binary32 runs against settled binary64 references.

These tests deliberately construct JuliaBackend directly. They do not change the
public single-precision backend registry or qualify a device or DF32 wide arithmetic.
"""

import os
from dataclasses import replace
from functools import cache
from typing import Any, cast

import numpy as np
import pytest
from tests.bounds import UNIT_ROUNDOFF_32, higham_gamma

from backends.julia.cases import cpu_case, material_case
from backends.julia.material_routes import material_route, refined_request
from backends.julia.split_cases import split_case
from backends.julia.test_transverse_records import edge_case
from sessions.executors import IDENTITY, Started, drive, fibre, soliton, window
from solprop.backends.events import ObservationReady, Outcome, join_attempts
from solprop.backends.julia._session import JuliaSession
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.registry import select
from solprop.backends.session import ExecutionOptions, Start
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.stats import StatsSpec
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.tableau import TSIT5
from solprop.translate.precision import SINGLE
from solprop.translate.problem import Translation
from solprop.translate.run_request import MethodKind

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _method(request, method):
    integration = request.integration
    if method == "tsit5":
        integration = replace(integration, tableau=TSIT5)
    elif method == "adams":
        integration = replace(integration, kind=MethodKind.ADAMS, method=Multistep())
    elif method == "low_storage":
        integration = replace(
            integration, kind=MethodKind.LOW_STORAGE, method=LowStorageRungeKutta()
        )
    else:
        assert method == "dopri5"
    return replace(request, integration=integration)


def _single(case, method=None, *, rtol=1e-6):
    request = case.request if method is None else _method(case.request, method)
    request = replace(
        request,
        error_control=replace(
            request.error_control,
            controller=replace(request.error_control.controller, rtol=rtol),
        ),
    )
    return Translation(replace(case.problem, precision=SINGLE), request)


def _execute(case, grid, modes, *, julia, work_limit=2):
    if julia:
        setup_id = os.environ.get("SOLPROP_TEST_JULIA_SETUP_ID")
        assert setup_id, "the enabled Julia lane must publish a managed setup identity"
        owner = JuliaBackend(
            case.problem,
            case.request,
            grid=grid,
            modes=modes,
            options=ExecutionOptions(
                node_chunk=3, backend_options={"setup_id": setup_id}
            ),
        )
    else:
        owner = select("numpy_ref", case.problem, case.request, grid=grid, modes=modes)
    original = case.problem.initial.field0.tobytes()
    try:
        with owner.prepare(case.problem, case.request).start(
            Start.entrance(case.problem)
        ) as session:
            result = drive(
                Started(session), export=IDENTITY, work_limit=work_limit, deadline=60.0
            )
            assert result.status.outcome is Outcome.COMPLETED, result.batches[
                -1
            ].terminal
            if case.request.replay is not None:
                assert result.status.rejected == 0
            # Execution and monitor metadata are heterogeneous JSON records; the
            # assertions below validate each numerical field used by these tests.
            record = cast("dict[str, Any]", dict(session.execution_record()))
            if julia:
                assert isinstance(session, JuliaSession)
                assert record["precision"] == case.problem.precision.name
                assert record["wide"] == "binary64"
                size = case.problem.initial.field0.size
                assert session.memory_ledger()["session.field"] == size * 8
            assert case.problem.initial.field0.tobytes() == original
            watched = cast("dict[str, Any] | None", session.monitor_record())
            return result, record, watched
    finally:
        owner.close()


def _statistics(result):
    return next(
        event.stats
        for event in reversed(result.observations)
        if isinstance(event, ObservationReady)
    )


def _bands(field, grid):
    active = grid.active_mask
    omega = grid.omega[active]
    weights = np.sum(np.abs(field[..., active]) ** 2, axis=(0, 1))
    edges = np.linspace(float(np.min(omega)), float(np.max(omega)), 33)
    return np.histogram(omega, bins=edges, weights=weights)[0]


@cache
def _reference(kind, geometry, gradient=False, taper=False):
    case, grid, modes = material_route(kind, geometry, gradient=gradient, taper=taper)
    runs = []
    for steps in (16, 32):
        refined = Translation(case.problem, refined_request(case.request, steps))
        result, _, _ = _execute(refined, grid, modes, julia=False)
        assert result.status.attempts == steps
        runs.append(result)
    return tuple(runs)


def _assert_selected_accuracy(case, actual, coarse, fine):
    # SEP33's accepted metrics (section 7.3/Part II) fix field 1e-3,
    # >=-40 dB spectral bands 1%, total energy 1e-4 and ionised fraction 1e-3.
    # Ordinary replay and variable-order Adams need accuracy envelopes, not a
    # claim of identical discrete histories. The same-grid binary64 DOPRI5 runs
    # are independently refined 16→32; settlement is at least 100 times tighter.
    # These limits are fixed before comparison and are not the controller rtol.
    got, reference = actual.fields[-1], fine.fields[-1]
    previous = coarse.fields[-1]
    scale = float(np.linalg.norm(reference.ravel()))
    assert scale > 0.0 and np.all(np.isfinite(got))
    assert got.dtype == np.complex128  # Physical exports widen binary32 storage.
    assert np.linalg.norm((reference - previous).ravel()) <= 1e-9 * scale
    assert np.linalg.norm((got - reference).ravel()) <= 1e-3 * scale
    assert np.count_nonzero(got[..., ~case.problem.grids.active_mask]) == 0

    bands = _bands(reference, case.problem.grids)
    significant = bands >= 1e-4 * np.max(bands)
    assert np.any(significant)
    for values, allowance in ((previous, 1e-4), (got, 1e-2)):
        error = np.abs(_bands(values, case.problem.grids) - bands)
        assert np.all(error[significant] <= allowance * bands[significant])

    statistics = tuple(_statistics(run) for run in (actual, coarse, fine))
    for rows in statistics:
        assert all(np.all(np.isfinite(values)) for values in rows.values())
    energy = float(np.sum(statistics[-1]["energy"]))
    assert energy > 0.0
    assert abs(float(np.sum(statistics[1]["energy"])) - energy) <= 1e-8 * energy
    assert abs(float(np.sum(statistics[0]["energy"])) - energy) <= 1e-4 * energy
    for name in ("species_ionisation", "species_electron_density", "electron_density"):
        expected = np.asarray(statistics[-1][name])
        # Every synthetic species starts at >=.03 ionised: the accepted population
        # metric's 1e-6 minimum signal is exceeded without a post-hoc cutoff.
        assert np.all(expected > 0.0)
        assert np.all(np.abs(statistics[1][name] - expected) <= 1e-8 * expected)
        assert np.all(np.abs(statistics[0][name] - expected) <= 1e-3 * expected)


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize(
    "geometry", ["averaged", "multimode", "radial", "cartesian", "line"]
)
def test_binary32_materials_on_each_admitted_route(kind, geometry):
    case, grid, modes = material_route(kind, geometry)
    actual, record, _ = _execute(_single(case), grid, modes, julia=True)
    assert actual.status.attempts == 8
    assert record["source_evaluations"] == 8 * len(
        case.request.integration.tableau.nodes
    )
    coarse, fine = _reference(kind, geometry)
    _assert_selected_accuracy(case, actual, coarse, fine)


@pytest.mark.parametrize("method", ["tsit5", "adams", "low_storage"])
@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_binary32_materials_across_march_families(kind, method):
    case, grid, modes = material_route(kind, "cartesian")
    actual, record, _ = _execute(_single(case, method), grid, modes, julia=True)
    assert actual.status.attempts == 8
    assert (
        record["source_evaluations"]
        == {"tsit5": 56, "adams": 9, "low_storage": 32}[method]
    )
    coarse, fine = _reference(kind, "cartesian")
    _assert_selected_accuracy(case, actual, coarse, fine)


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("geometry", ["averaged", "multimode"])
def test_binary32_independent_profiles_and_uncoupled_taper(kind, geometry):
    case, grid, modes = material_route(kind, geometry, gradient=True, taper=True)
    actual, _, _ = _execute(_single(case), grid, modes, julia=True)
    assert actual.status.attempts == 8
    coarse, fine = _reference(kind, geometry, True, True)
    _assert_selected_accuracy(case, actual, coarse, fine)


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("response", ["kerr", "raman", "plasma"])
def test_binary32_individual_material_sources(kind, response):
    case, grid, mode = material_case(kind, response, steps=8)
    actual, record, _ = _execute(_single(case), grid, (mode,), julia=True)
    references = []
    for steps in (16, 32):
        reference, reference_grid, reference_mode = material_case(
            kind, response, steps=steps
        )
        result, _, _ = _execute(
            reference, reference_grid, (reference_mode,), julia=False
        )
        references.append(result.fields[-1])
    coarse, fine = references
    scale = float(np.linalg.norm(fine.ravel()))
    # The existing binary64 component matrix establishes settlement at 1e-11 on
    # these eight-micrometre windows; use the accepted SEP33 field envelope for
    # the independently rounded binary32 transforms, response and state.
    assert scale > 0.0
    assert np.linalg.norm((fine - coarse).ravel()) <= 1e-11 * scale
    assert np.linalg.norm((actual.fields[-1] - fine).ravel()) <= 1e-3 * scale
    assert actual.status.attempts == 8 and record["source_evaluations"] == 56


@pytest.mark.parametrize("rtol", [1e-6, 1e-5])
@pytest.mark.parametrize("method", ["dopri5", "tsit5", "adams"])
@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_binary32_adaptive_materials_at_both_accepted_tolerances(kind, method, rtol):
    case, grid, modes = material_route(kind, "averaged")
    length = float(case.request.observation.save_planes[-1])
    adaptive = Translation(
        case.problem,
        replace(
            case.request,
            replay=None,
            error_control=replace(
                case.request.error_control, dz_init=length / 8, dz_max=length / 8
            ),
        ),
    )
    actual, _, _ = _execute(
        _single(adaptive, method, rtol=rtol), grid, modes, julia=True
    )
    assert actual.status.attempts - actual.status.rejected >= 8
    coarse, fine = _reference(kind, "averaged")
    _assert_selected_accuracy(case, actual, coarse, fine)


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize(
    "geometry", ["averaged", "multimode", "radial", "cartesian", "line"]
)
def test_binary32_physical_peak_and_width_readouts(kind, geometry):
    case, grid, modes = material_route(kind, geometry)
    names = (
        ("peak_power",)
        if geometry in ("averaged", "multimode")
        else ("peak_intensity", "beam_radius", "fwhm_r")
    )
    case = Translation(
        case.problem,
        replace(
            case.request,
            observation=replace(
                case.request.observation, stats=StatsSpec(names=names, periodic=())
            ),
        ),
    )
    actual, _, _ = _execute(_single(case), grid, modes, julia=True)
    references = []
    for steps in (16, 32):
        reference = Translation(case.problem, refined_request(case.request, steps))
        result, _, _ = _execute(reference, grid, modes, julia=False)
        references.append(_statistics(result))
    assert actual.status.attempts == 8
    coarse, fine = references
    for name, values in _statistics(actual).items():
        expected = np.asarray(fine[name])
        assert np.all(np.isfinite(values)) and np.all(expected > 0.0), name
        assert not values.flags.writeable
        # SEP33 fixes peak power/intensity and spatial width envelopes at 1e-3.
        # The same-grid binary64 refinement must settle five orders more tightly;
        # no slope-independent rounding claim is made for threshold widths.
        np.testing.assert_allclose(coarse[name], expected, rtol=1e-8, atol=0.0)
        np.testing.assert_allclose(values, expected, rtol=1e-3, atol=0.0)


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("method", ["rk", "adams", "low_storage"])
def test_binary32_split_collar_and_temporal_services(kind, method):
    case, grid, mode = split_case(
        kind, method, rate="uniform", observers=("statistics", "monitor")
    )
    monitor = case.request.boundary.monitor
    assert monitor is not None
    monitor.check = replace(monitor.check, quiet_collar=True)
    actual, record, watched = _execute(_single(case), grid, (mode,), julia=True)
    assert actual.status.attempts == 8
    assert record["source_evaluations"] == (
        0 if method == "adams" else 32 if method == "low_storage" else 56
    )
    assert watched is not None
    assert watched["steps"] + watched["skipped"] == 8
    assert watched["planes"]["z"] == [0.0, 4e-6, 8e-6]
    assert watched["collar_quiet"] is True
    assert monitor.summary()["launch_energy"] is None
    initial = np.where(case.problem.grids.active_mask, case.problem.initial.field0, 0.0)
    operator = case.problem.linear.phi_constant
    assert operator is not None and np.all(np.real(operator) == 0)
    # Uniform power loss alpha=.2/L commutes with this lossless spectral flow.
    # This exact solution avoids a reference march: the already accepted SEP33
    # field and total-energy envelopes apply at every delivered plane.
    launch_energy = float(np.sum(actual.observations[0].stats["energy"]))
    for event, field in zip(actual.observations, actual.fields, strict=True):
        expected = initial * np.exp(operator * event.z - 0.1 * event.z / 8e-6)
        scale = float(np.linalg.norm(expected.ravel()))
        assert scale > 0.0
        assert np.linalg.norm((field - expected).ravel()) <= 1e-3 * scale
        energy = launch_energy * np.exp(-0.2 * event.z / 8e-6)
        assert abs(float(np.sum(event.stats["energy"])) - energy) <= 1e-4 * energy
    assert np.linalg.norm(actual.fields[-1].ravel()) < 0.95 * np.linalg.norm(
        initial.ravel()
    )


@cache
def _dense_reference():
    planes = np.linspace(0.02 / 16, 0.02, 16)
    planes.setflags(write=False)
    runs = []
    for steps in (32, 64):
        case = cpu_case(steps=steps)
        request = replace(
            case.request,
            observation=replace(
                case.request.observation,
                save_planes=planes,
                stats=StatsSpec(names=("energy",), periodic=()),
            ),
        )
        result, _, _ = _execute(
            Translation(case.problem, request), window(), (fibre().mode,), julia=False
        )
        runs.append(result)
    return tuple(runs)


@pytest.mark.parametrize("method", ["dopri5", "adams"])
def test_binary32_dense_observation_and_periodic_services(method, request):
    case = soliton(
        integrator=Multistep() if method == "adams" else None,
        interpolate=True,
        planes=np.linspace(0.02 / 16, 0.02, 16),
    )
    run_request = replace(
        case.request,
        error_control=replace(
            case.request.error_control, dz_init=0.0025, dz_max=0.0025
        ),
        observation=replace(
            case.request.observation,
            stats=StatsSpec(names=("energy",), periodic=("energy",), period=1),
        ),
    )
    monitor = BoundaryMonitor.build(window(), None, BoundaryCheck(budget=1.0))
    case = Translation(case.problem, run_request.for_run(monitor=monitor))
    actual, record, watched = _execute(
        _single(case, method), window(), (fibre().mode,), julia=True, work_limit=1
    )
    accepted = actual.status.attempts - actual.status.rejected
    # The adaptive controller may reject or shorten a step; dz_max only supplies
    # this lower bound. Exact periodic cadence is checked against actual history.
    assert accepted >= 8
    request.node.user_properties.extend(
        [("accepted", accepted), ("rejected", actual.status.rejected)]
    )
    assert len(actual.observations) == 17
    if method == "adams":
        assert record["response_calls"]["dense_observations"] > 0
    history, periodic = join_attempts(actual.chunks)
    np.testing.assert_array_equal(periodic.z, history.z[history.accepted])
    assert watched is not None
    assert watched["planes"]["z"] == [event.z for event in actual.observations]
    coarse, fine = _dense_reference()
    for got, previous, reference in zip(
        actual.fields, coarse.fields, fine.fields, strict=True
    ):
        scale = float(np.linalg.norm(reference.ravel()))
        assert scale > 0.0
        assert np.linalg.norm((reference - previous).ravel()) <= 1e-9 * scale
        assert np.linalg.norm((got - reference).ravel()) <= 1e-3 * scale
    for got, previous, reference in zip(
        actual.observations, coarse.observations, fine.observations, strict=True
    ):
        energy = float(np.sum(reference.stats["energy"]))
        assert energy > 0.0
        assert abs(float(np.sum(previous.stats["energy"])) - energy) <= 1e-8 * energy
        assert abs(float(np.sum(got.stats["energy"])) - energy) <= 1e-4 * energy


@pytest.mark.parametrize("geometry", ["radial", "cartesian", "line"])
def test_binary32_transverse_monitor_keeps_physical_launch_and_cadence(geometry):
    case, grid, edge = edge_case(geometry)
    actual, _, watched = _execute(_single(case), grid, None, julia=True)
    assert actual.status.attempts == 8
    assert watched is not None
    transverse = watched["transverse"]
    clean = np.where(case.problem.grids.active_mask, case.problem.initial.field0, 0.0)
    reading = edge.reduction.reading(clean, full=True)
    assert reading.totals is not None and reading.faces is not None
    energy = float(reading.totals[0])
    assert energy > 0.0
    # Launch energy uses the accepted SEP33 physical-energy envelope. Face shares
    # use the same declared coarse-transform/positive-reduction CPU bound as the
    # binary64 transverse bridge test, recomputed at binary32 unit roundoff. This
    # includes normalization and the launch division, retains total launch energy
    # as the absolute scale, and makes no tiny-face relative or decision claim.
    assert abs(transverse["launch_energy"] - energy) <= 1e-4 * energy
    expected = reading.faces[: len(edge._faces)] / energy
    bound = 8 * higham_gamma(64 * (clean.shape[0] + clean.shape[-1]), UNIT_ROUNDOFF_32)
    assert 0 < bound < 0.01
    np.testing.assert_allclose(
        transverse["edge_baseline"], expected, rtol=0.0, atol=bound
    )
    assert transverse["reads"] > 0
    assert transverse["faces"] == [face.name for face in edge._faces]
    assert watched["steps"] + watched["skipped"] == 8
    assert edge.summary()["launch_energy"] is None
