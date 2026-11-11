# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify dense observation transport separately from the committed march."""

import os
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.cases import cpu_case
from backends.julia.observation_bounds import scalar_envelope_bounds
from backends.julia.test_cpu_monitor import _record, _start
from backends.julia.test_cpu_session import _run
from sessions.executors import IDENTITY, Started, drive, fibre, soliton, window
from solprop.backends.events import (
    ObservationReady,
    Outcome,
    ServiceKind,
    ServiceRequired,
)
from solprop.backends.export import ExportRecipe, apply_recipe
from solprop.backends.registry import select
from solprop.backends.session import SessionError, Start
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.problem import Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _case(*, weighted=False, method="adams"):
    case = soliton(
        integrator=Multistep() if method == "adams" else None,
        interpolate=True,
        planes=np.linspace(0.02 / 16, 0.02, 16),
    )
    request = replace(
        case.request,
        error_control=replace(
            case.request.error_control, controller=PIController(rtol=1e-8, atol=1e-16)
        ),
        observation=replace(
            case.request.observation,
            stats=StatsSpec(names=("energy",), periodic=("energy",), period=1),
        ),
    )
    monitor = BoundaryMonitor.build(window(), None, BoundaryCheck(budget=1.0))
    problem = case.problem
    if weighted:
        # On the actual descriptor interval, w(z)=1+z/span_end. This exercises
        # the observation coordinate independently of the committed endpoint.
        shape = problem.shape
        table = PiecewiseChebyshev(
            np.stack((np.full(shape, 1.5), np.full(shape, 0.5)))[None],
            np.array(problem.linear.z_span),
            np.array([1], dtype=np.intp),
            np.empty(0),
        )
        for array in (
            table.coefficients,
            table.breakpoints,
            table.degrees,
            table.hard_kinks,
        ):
            array.setflags(write=False)
        problem = replace(
            problem,
            linear=replace(
                problem.linear,
                power_weight_constant=None,
                power_weight_representation="table",
                power_weight=table,
            ),
        )
    return Translation(problem, request.for_run(monitor=monitor))


@pytest.mark.parametrize("work_limit", [1, 256])
@pytest.mark.parametrize("method", ["rk", "adams"])
def test_dense_fields_meet_a_converged_landed_reference(work_limit, method):
    case = _case(method=method)
    with _start(case) as session:
        actual = drive(Started(session), export=IDENTITY, work_limit=work_limit)
        assert actual.status.outcome is Outcome.COMPLETED
        if method == "adams":
            calls = session.execution_record()["response_calls"]
            assert isinstance(calls, dict) and calls["dense_observations"] > 0
    references = []
    for steps in (128, 256):
        base = cpu_case(steps=steps)
        reference = Translation(
            base.problem,
            replace(
                base.request,
                observation=replace(
                    base.request.observation,
                    save_planes=case.request.observation.save_planes,
                ),
            ),
        )
        references.append(_run(reference, julia=False))
    assert len(actual.fields) == len(references[0].fields) == len(references[1].fields)
    for got, coarse, fine in zip(
        actual.fields, references[0].fields, references[1].fields, strict=True
    ):
        scale = np.linalg.norm(fine.ravel())
        # Different adaptive/dense histories use the existing short soliton's
        # fixed global1e-8 envelope; RK refinement must settle1000times tighter.
        assert np.linalg.norm((fine - coarse).ravel()) <= 1e-11 * scale
        assert np.linalg.norm((got - fine).ravel()) <= 1e-8 * scale


@pytest.mark.parametrize("method", ["rk", "adams"])
def test_dense_statistics_snapshots_and_monitor_keep_distinct_coordinates(method):
    case = _case(method=method)
    oracle = select(
        "numpy_ref", case.problem, case.request, grid=window(), modes=(fibre().mode,)
    )
    dense_positions, snapshot_positions, observed_positions = [], [], []
    accepted, periodic = [], []
    with closing(oracle), _start(case) as session:
        while not session.finished:
            batch = session.advance(work_limit=1, deadline=60.0)
            status = session.status()
            for event in batch.events:
                history, rows = event.history, event.step_stats
                if history is not None:
                    accepted.extend(history.z[history.accepted].tolist())
                if rows is not None:
                    periodic.extend(rows.z.tolist())
                if isinstance(event, ObservationReady):
                    observed_positions.append(event.z)
                    raw = session.export(event.ticket, IDENTITY)
                    bound = scalar_envelope_bounds(
                        case.problem, window(), case.request.observation.stats, raw
                    )["energy"]
                    oracle.set_initial(raw, None, z=event.z, resume=True)
                    expected = oracle.stats(case.request.observation.stats)["energy"]
                    assert np.all(np.abs(event.stats["energy"] - expected) <= bound)
                    if event.z < status.z:
                        dense_positions.append(event.z)
                        assert session.march_snapshot(event.ticket) is None
                        assert session.march_snapshot_nbytes() is None
                        session.request_snapshot()
                elif isinstance(event, ServiceRequired):
                    assert event.kind is ServiceKind.SNAPSHOT
                    assert event.z == status.z
                    snapshot_positions.append(event.z)
                    assert event.ticket is not None
                    history_snapshot = session.march_snapshot(event.ticket)
                    if method == "adams":
                        assert history_snapshot is not None
                        assert history_snapshot.z == status.z
                    else:
                        assert history_snapshot is None
                    assert observed_positions[1:] == [
                        z for z in case.request.observation.save_planes if z <= status.z
                    ]
            session.acknowledge(batch.receipt)
        assert session.status().outcome is Outcome.COMPLETED
        assert dense_positions and snapshot_positions
        assert periodic == accepted
        assert _record(session)["planes"]["z"] == observed_positions


@pytest.mark.parametrize("method", ["rk", "adams"])
def test_dense_power_export_uses_observation_weight_and_retains_owned_arrays(method):
    case = _case(weighted=True, method=method)
    recipe = ExportRecipe(power=True, optics=True)
    retained = []
    interior = 0
    with _start(case) as session:
        while not session.finished:
            batch = session.advance(deadline=60.0)
            for event in batch.events:
                if not isinstance(event, ObservationReady):
                    continue
                raw = session.export(event.ticket, IDENTITY)
                weight = np.full(
                    case.problem.shape, 1.0 + event.z / case.problem.linear.z_span[1]
                )
                expected = apply_recipe(raw, recipe, weight=weight)
                value = session.export(event.ticket, recipe)
                # Both evaluate the same affine weight.32u covers its mapping,
                # Clenshaw sum, square root and final complex product roundings.
                bound = 32 * np.finfo(float).eps / 2 * np.linalg.norm(expected.ravel())
                assert bound > 0
                assert np.linalg.norm((value - expected).ravel()) <= bound
                destination = np.empty(case.problem.shape, dtype=np.complex128)
                assert session.export(event.ticket, recipe, destination) is destination
                assert destination.tobytes() == value.tobytes()
                retained.append((value, value.tobytes()))
                interior += event.z < session.status().z
            session.acknowledge(batch.receipt)
            if isinstance(batch.events[0], ObservationReady):
                with pytest.raises(SessionError):
                    session.export(batch.events[0].ticket, IDENTITY)
        assert session.status().outcome is Outcome.COMPLETED
    assert interior > 0
    assert all(value.tobytes() == saved for value, saved in retained)


@pytest.mark.parametrize("method", ["rk", "adams"])
def test_dense_plane_restarts_from_its_field_without_future_history(method):
    case = _case(method=method)
    restart = None
    with _start(case) as session:
        while restart is None:
            batch = session.advance(deadline=60.0)
            for event in batch.events:
                if isinstance(event, ObservationReady) and event.z < session.status().z:
                    assert session.march_snapshot(event.ticket) is None
                    restart = Start(
                        field=session.export(event.ticket, IDENTITY).copy(),
                        z=event.z,
                        plane_index=event.plane_index + 1,
                        plane_pending=False,
                        resume=True,
                    )
            session.acknowledge(batch.receipt)
            assert restart is not None or not session.finished
    with _start(case, restart) as resumed:
        actual = drive(Started(resumed), export=IDENTITY)
        assert actual.status.outcome is Outcome.COMPLETED
    coarse = _run(cpu_case(steps=128), julia=False).fields[-1]
    fine = _run(cpu_case(steps=256), julia=False).fields[-1]
    scale = np.linalg.norm(fine.ravel())
    # A field restart changes the discrete history. Use the same fixed global
    # soliton envelope and independent reference refinement as the unsplit run.
    assert np.linalg.norm((coarse - fine).ravel()) <= 1e-11 * scale
    assert np.linalg.norm((actual.fields[-1] - fine).ravel()) <= 1e-8 * scale
