# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify portable binary64 Adams continuation through the actual managed bridge."""

import os
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.bounds import adams_rounding_budget
from backends.julia.cases import adams_case
from backends.julia.test_cpu_session import _prepared
from sessions.executors import IDENTITY, Started, drive, fibre, window
from solprop.backends.events import ObservationReady, Outcome
from solprop.backends.julia import _calls
from solprop.backends.julia._history import HistoryBuffers
from solprop.backends.multistep import CallLedger
from solprop.backends.registry import select
from solprop.backends.session import SessionError, Start
from solprop.exceptions import BackendError, ResponseBudgetError
from solprop.stepper.multistep import Multistep
from solprop.translate.problem import Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _session(case, *, backend, start=None):
    if backend == "julia":
        _, prepared = _prepared(case)
    else:
        owner = select(
            backend, case.problem, case.request, grid=window(), modes=[fibre().mode]
        )
        prepared = owner.prepare(case.problem, case.request)
    return prepared.start(Start.entrance(case.problem) if start is None else start)


def _first_plane(session):
    while not session.finished:
        batch = session.advance(deadline=60.0)
        for event in batch.events:
            if isinstance(event, ObservationReady) and event.z > 0:
                snapshot = session.march_snapshot(event.ticket)
                assert snapshot is not None
                return batch, event, snapshot
        session.acknowledge(batch.receipt)
    pytest.fail("the short history witness ended without a save plane")


def _saved(case, *, backend):
    session = _session(case, backend=backend)
    with session:
        batch, event, snapshot = _first_plane(session)
        assert (
            snapshot.state.tobytes() == session.export(event.ticket, IDENTITY).tobytes()
        )
        assert session.march_snapshot_nbytes() == snapshot.nbytes
        calls = session.execution_record()["response_calls"]
        next_plane = event.plane_index + 1
        session.acknowledge(batch.receipt)
        with pytest.raises(SessionError):
            session.march_snapshot(event.ticket)
        result = drive(Started(session), export=IDENTITY)
        assert result.status.outcome is Outcome.COMPLETED
        return (
            snapshot,
            calls,
            next_plane,
            result.fields[-1],
            session.execution_record(),
        )


@pytest.mark.parametrize(
    ("source", "destination"),
    [
        ("julia", "julia"),
        ("numpy_ref", "julia"),
        ("julia", "numpy_ref"),
        ("native", "julia"),
        ("julia", "native"),
    ],
)
def test_portable_history_continues_without_an_extra_startup_charge(
    source, destination
):
    case = adams_case(method=Multistep(order=5, variable_order=False))
    bound = adams_rounding_budget(case, maximum_order=5)
    snapshot, calls, next_plane, expected, final_record = _saved(case, backend=source)
    start = Start(
        field=snapshot.state.copy(),
        z=snapshot.z,
        plane_index=next_plane,
        march=snapshot,
        ledgers={"logical": calls},
        plane_pending=False,
        resume=True,
    )
    resumed = _session(case, backend=destination, start=start)
    assert start.field is None
    with resumed:
        assert getattr(resumed, "march_refusal", None) is None
        result = drive(Started(resumed), export=IDENTITY, work_limit=1)
        assert result.status.outcome is Outcome.COMPLETED, result.batches[-1].terminal
        # This uses the same discrete eight-step method and retained history as the
        # unsplit path. Its operation-derived bound is fixed before either run.
        assert bound > 0
        assert np.linalg.norm((result.fields[-1] - expected).ravel()) <= bound
        assert (
            resumed.execution_record()["response_calls"]
            == final_record["response_calls"]
        )
        if destination == "julia":
            assert resumed.execution_record()["source_evaluations"] == 4


@pytest.mark.parametrize("keep_history", [False, True])
def test_resume_budget_uses_saved_charges_before_any_new_trial(keep_history):
    method = Multistep(order=5, variable_order=False)
    case = adams_case(method=method)
    snapshot, calls, next_plane, _, _ = _saved(case, backend="julia")
    assert isinstance(calls, dict)
    spent = sum(int(calls[name]) for name in ("startup", "predictor", "corrected"))
    limited = adams_case(method=replace(method, max_calls=spent))
    start = Start(
        field=snapshot.state.copy(),
        z=snapshot.z,
        plane_index=next_plane,
        march=snapshot if keep_history else None,
        ledgers={"logical": calls},
        plane_pending=False,
        resume=True,
    )
    with _session(limited, backend="julia", start=start) as resumed:
        result = drive(Started(resumed), export=IDENTITY)
        assert result.status.outcome is Outcome.FAILED
        terminal = result.batches[-1].terminal
        assert terminal is not None and isinstance(terminal.error, ResponseBudgetError)
        assert terminal.error.calls == spent
        assert result.status.attempts == 0
        assert resumed.execution_record()["source_evaluations"] == 0
        assert resumed.execution_record()["response_calls"] == calls


@pytest.mark.parametrize("compensated", [False, True])
def test_snapshot_restore_keeps_raw_state_and_every_inactive_store_row(compensated):
    case = adams_case(method=Multistep(order=5, variable_order=False))
    if compensated:
        # The same cumulative polynomial in its full table representation selects
        # the sampled-state/rate history. This is a transfer witness, not a new model.
        case = Translation(
            replace(
                case.problem,
                linear=replace(
                    case.problem.linear, phi_representation="table", phi_constant=None
                ),
            ),
            case.request,
        )
    snapshot, calls, next_plane, _, _ = _saved(case, backend="julia")
    assert set(snapshot.rows) == (
        {"store", "states", "rates"} if compensated else {"store"}
    )
    # The history's own accepted state overrides a separately reconstructed field.
    # No propagation is used for this exact storage contract.
    field = snapshot.state.copy()
    field.flat[0] += 0.5
    start = Start(
        field=field,
        z=snapshot.z,
        plane_index=next_plane,
        march=snapshot,
        ledgers={"logical": calls},
        plane_pending=True,
        resume=True,
    )
    with _session(case, backend="julia", start=start) as resumed:
        batch = resumed.advance()
        event = batch.events[0]
        assert isinstance(event, ObservationReady)
        restored = resumed.march_snapshot(event.ticket)
        assert restored is not None
        assert restored.state.tobytes() == snapshot.state.tobytes()
        assert restored.identities == snapshot.identities
        assert restored.slots.tobytes() == snapshot.slots.tobytes()
        assert restored.positions.tobytes() == snapshot.positions.tobytes()
        for name in snapshot.rows:
            assert restored.rows[name].tobytes() == snapshot.rows[name].tobytes()
        # A retained caller snapshot remains independent of the running engine.
        restored.state.fill(9)
        assert (
            resumed.export(event.ticket, IDENTITY).tobytes() == snapshot.state.tobytes()
        )
        resumed.acknowledge(batch.receipt)


def test_foreign_history_rows_produce_a_recorded_field_reset():
    case = adams_case(method=Multistep(order=5, variable_order=False))
    snapshot, calls, next_plane, _, _ = _saved(case, backend="julia")
    invalid = replace(
        snapshot, rows={**snapshot.rows, "applied": snapshot.rows["store"]}
    )
    start = Start(
        field=snapshot.state.copy(),
        z=snapshot.z,
        plane_index=next_plane,
        march=invalid,
        ledgers={"logical": calls},
        plane_pending=False,
        resume=True,
    )
    with _session(case, backend="julia", start=start) as resumed:
        assert "applied" in str(getattr(resumed, "march_refusal", None))
        result = drive(Started(resumed), export=IDENTITY)
        assert result.status.outcome is Outcome.COMPLETED
        record = resumed.execution_record()["response_calls"]
        assert isinstance(record, dict) and isinstance(calls, dict)
        assert record["startup"] == calls["startup"] + 1
        assert record["accepted_steps"] == 8


def test_foreign_overlapping_history_destinations_refuse_before_writes():
    case = adams_case(method=Multistep(order=5, variable_order=False))
    _, prepared = _prepared(case)
    with prepared.start(Start.entrance(case.problem)) as session:
        batch, event, snapshot = _first_plane(session)
        shared = np.full(snapshot.rows["store"].shape, 11 + 3j, dtype=np.complex128)
        buffers = HistoryBuffers(
            shared[1, 0],
            np.empty(snapshot.positions.shape, dtype=np.float64),
            np.empty(len(snapshot.identities), dtype=np.uint8),
            np.empty(snapshot.slots.shape, dtype=np.int64),
            shared,
        )
        with pytest.raises(BackendError, match="disjoint"):
            _calls.call_void(
                session._runtime,
                "export_adams_history!",
                buffers,
                session._engine,
                event.ticket.generation,
                event.z,
            )
        assert np.all(shared == 11 + 3j)
        session.acknowledge(batch.receipt)


def test_restored_counter_capacity_fails_before_source_work_without_wrapping():
    case = adams_case()
    calls = CallLedger(startup=2**63 - 1).to_record()
    start = Start(
        field=np.array(case.problem.initial.field0, copy=True),
        z=0.0,
        ledgers={"logical": calls},
        plane_pending=False,
        resume=True,
    )
    with _session(case, backend="julia", start=start) as session:
        result = drive(Started(session), export=IDENTITY)
        assert result.status.outcome is Outcome.FAILED
        terminal = result.batches[-1].terminal
        assert terminal is not None and "overflow" in str(terminal.error).lower()
        assert result.status.attempts == 0
        assert session.execution_record()["source_evaluations"] == 0
        assert session.execution_record()["response_calls"] == calls
