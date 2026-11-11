# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep actual raw Adams coordinates through DF32 managed-start boundaries."""

import math
import os
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.cases import adams_case
from sessions.executors import IDENTITY, Started, drive
from solprop.backends.events import ObservationReady, Outcome
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.session import ExecutionOptions, Start
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.precision import SINGLE
from solprop.translate.problem import Translation
from solprop.translate.responses import ResponseStage
from solprop.translate.run_request import ReplaySpec

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _case(*, compensated=False, source_absent=False):
    original = adams_case(method=Multistep(order=3, variable_order=False), steps=2)
    problem = replace(original.problem, precision=SINGLE)
    if compensated:
        problem = replace(
            problem,
            linear=replace(
                problem.linear, phi_representation="table", phi_constant=None
            ),
        )
    if source_absent:
        problem = replace(
            problem,
            responses=replace(
                problem.responses,
                response_order=(ResponseStage.ACCUMULATE,),
                kerr_coefficients=None,
            ),
        )
    return Translation(
        problem,
        replace(
            original.request,
            integration=replace(
                original.request.integration, propagator_mode="recompute"
            ),
        ),
    )


@contextmanager
def _factory(case, *, wide="df32"):
    owner = JuliaBackend(
        case.problem,
        case.request,
        options=ExecutionOptions(
            backend_options={
                "setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"],
                "wide": wide,
            }
        ),
    )
    try:
        yield owner.prepare(case.problem, case.request)
    finally:
        owner.close()


def _checkpoint(case):
    with (
        _factory(case, wide="binary64") as prepared,
        prepared.start(Start.entrance(case.problem), owns=False) as session,
    ):
        while not session.finished:
            batch = session.advance(work_limit=1)
            for event in batch.events:
                if isinstance(event, ObservationReady) and event.z > 0:
                    snapshot = session.march_snapshot(event.ticket)
                    assert snapshot is not None
                    field = session.export(event.ticket, IDENTITY).copy()
                    return snapshot, field
            session.acknowledge(batch.receipt)
    pytest.fail("the managed binary32 reference did not publish its first history")


def _launch_history(session):
    batch = session.advance(work_limit=1)
    event = next(event for event in batch.events if isinstance(event, ObservationReady))
    snapshot = session.march_snapshot(event.ticket)
    assert snapshot is not None
    assert session.execution_record()["source_evaluations"] == 0
    session.acknowledge(batch.receipt)
    return snapshot


@pytest.mark.parametrize("compensated", [False, True])
def test_df32_factory_keeps_compatible_future_sample_beyond_nearby_end(compensated):
    original = _case(compensated=compensated)
    snapshot, field = _checkpoint(original)
    ulp = math.ulp(snapshot.z)
    step = 16 * ulp
    finish = snapshot.z + 2 * step
    positions = snapshot.positions.copy()
    positions[-1] = snapshot.z + 512 * ulp
    selected = replace(snapshot, positions=positions)
    assert snapshot.z < finish < positions[-1]
    assert math.isclose(positions[-1], snapshot.z, rel_tol=1e-12, abs_tol=0)
    # This is the shared restore's coordinate-admission boundary, not a claim
    # that changing a saved sample's coordinate reproduces a physical trajectory.
    # Binding and an owed launch must retain it exactly without taking a trial.
    case = Translation(
        original.problem,
        replace(
            original.request,
            observation=replace(
                original.request.observation, save_planes=np.array([finish])
            ),
            replay=ReplaySpec(FixedSchedule(np.array([step, step]), snapshot.z)),
        ),
    )
    start = Start(field=field, z=snapshot.z, march=selected, resume=True)
    expected_positions = positions.tobytes()
    with (
        _factory(case) as prepared,
        prepared.start(start, owns=False) as session,
    ):
        assert start.field is None
        assert session.march_refusal is None
        # Mutation after successful adoption cannot rewrite the selected snapshot.
        positions[-1] = snapshot.z
        restored = _launch_history(session)
        assert restored.positions.tobytes() == expected_positions
        assert restored.slots.tobytes() == snapshot.slots.tobytes()
        assert restored.identities == snapshot.identities
        assert restored.state.tobytes() == snapshot.state.tobytes()
        assert set(restored.rows) == set(snapshot.rows)
        for name, row in snapshot.rows.items():
            assert restored.rows[name].tobytes() == row.tobytes()


@pytest.mark.parametrize("compensated", [False, True])
def test_df32_raw_predicted_endpoint_above_canonical_finish_remains_exportable(
    compensated,
):
    case = _case(compensated=compensated)
    finish = 0.02
    start_z = float(np.nextafter(0.01, np.inf))
    raw_endpoint = start_z + 0.01
    assert raw_endpoint > finish
    # Replay retains its recorded destination while applying the supplied step
    # at the actual resumed anchor, admitted within the shared edge tolerance.
    start = Start(
        field=case.problem.initial.field0.copy(),
        z=start_z,
        plane_index=1,
        resume=True,
    )
    with (
        _factory(case) as prepared,
        prepared.start(start, owns=False) as session,
    ):
        retained = None
        while not session.finished:
            batch = session.advance(work_limit=1)
            for event in batch.events:
                if isinstance(event, ObservationReady) and event.z == finish:
                    retained = session.march_snapshot(event.ticket)
            if batch.terminal is not None:
                assert batch.terminal.outcome is Outcome.COMPLETED, batch.terminal
            session.acknowledge(batch.receipt)
        assert retained is not None
        assert retained.z == finish < retained.positions[-1] == raw_endpoint
        assert retained.positions[0] == start_z
        assert retained.identities[-1] == "predicted"
        assert session.execution_record()["source_evaluations"] == 2


@pytest.mark.parametrize("represented", [False, True])
def test_df32_completed_source_free_start_has_no_fake_history_or_trial(represented):
    case = _case(compensated=represented, source_absent=True)
    finish = float(case.request.observation.save_planes[-1])
    start = Start(
        field=case.problem.initial.field0.copy(),
        z=finish,
        plane_index=len(case.request.observation.save_planes),
        resume=True,
    )
    with (
        _factory(case) as prepared,
        prepared.start(start, owns=False) as session,
    ):
        batch = session.advance(work_limit=1)
        event = next(
            event for event in batch.events if isinstance(event, ObservationReady)
        )
        assert event.z == finish
        assert session.march_snapshot(event.ticket) is None
        assert session.execution_record()["source_evaluations"] == 0
        assert all(
            size == 0
            for name, size in session.memory_ledger().items()
            if "plans.banks" in name
        )
        session.acknowledge(batch.receipt)
        result = drive(Started(session), export=IDENTITY, work_limit=1)
        assert result.status.outcome is Outcome.COMPLETED
        assert result.status.attempts == 0
        assert session.execution_record()["source_evaluations"] == 0
