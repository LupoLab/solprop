# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify complete DF32 Adams sessions and selected portable history."""

import os
from contextlib import ExitStack, contextmanager
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.cases import adams_case
from backends.julia.test_cpu_single_groups import _group, _scaled
from sessions.executors import IDENTITY, Started, drive, fibre, window
from solprop.backends.events import ObservationReady, Outcome, Terminal
from solprop.backends.export import ExportRecipe
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.registry import select
from solprop.backends.session import ExecutionOptions, Start
from solprop.stepper.multistep import Multistep
from solprop.translate.precision import SINGLE
from solprop.translate.problem import Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _case(*, order=5, compensated=False, single=True, steps=8):
    case = adams_case(
        method=Multistep(order=order, variable_order=steps is None),
        steps=steps,
        rtol=1e-6,
        dz_max=0.0025 if steps is None else 1e-5,
    )
    problem = replace(case.problem, precision=SINGLE) if single else case.problem
    if compensated:
        # The same cumulative operator in its table representation selects sampled
        # states/rates. This changes the execution representation, not the physics.
        problem = replace(
            problem,
            linear=replace(
                problem.linear, phi_representation="table", phi_constant=None
            ),
        )
    return Translation(
        problem,
        replace(
            case.request,
            integration=replace(case.request.integration, propagator_mode="recompute"),
        ),
    )


@contextmanager
def _session(case, *, wide="df32", start=None):
    if wide is None:
        owner = select(
            "numpy_ref", case.problem, case.request, grid=window(), modes=[fibre().mode]
        )
    else:
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
        prepared = owner.prepare(case.problem, case.request)
        with prepared.start(
            Start.entrance(case.problem) if start is None else start
        ) as session:
            yield session
    finally:
        owner.close()


def _run(case, *, wide="df32", work_limit=256, recipe=IDENTITY):
    with _session(case, wide=wide) as session:
        result = drive(Started(session), export=recipe, work_limit=work_limit)
        assert result.status.outcome is Outcome.COMPLETED, result.batches[-1].terminal
        assert all(np.isfinite(field).all() for field in result.fields)
        return result


@pytest.mark.parametrize("order", [1, 3, 6])
@pytest.mark.parametrize("compensated", [False, True])
def test_df32_adams_landed_sessions_meet_refined_binary64_envelopes(order, compensated):
    recipe = ExportRecipe(power=True, optics=False)
    actual = _run(
        _case(order=order, compensated=compensated), recipe=recipe, work_limit=1
    )
    reference = _run(
        _case(order=order, compensated=compensated, single=False, steps=32),
        wide=None,
        recipe=recipe,
    )
    coarse = _run(
        _case(order=order, compensated=compensated, single=False, steps=16),
        wide=None,
        recipe=recipe,
    )
    assert actual.status.attempts == 8 and actual.status.rejected == 0
    for found, expected, previous in zip(
        actual.fields, reference.fields, coarse.fields, strict=True
    ):
        # Ordinary replay uses the SEP33 field/energy accuracy envelopes, with
        # step-size refinement settling the independent binary64 witness 100x tighter.
        scale = np.linalg.norm(expected)
        assert scale > 0
        assert np.linalg.norm(previous - expected) / scale < 1e-5
        assert np.linalg.norm(found - expected) / scale < 1e-3
        energy = float(np.vdot(expected, expected).real)
        assert energy > 0
        assert abs(float(np.vdot(previous, previous).real) / energy - 1) < 1e-6
        assert abs(float(np.vdot(found, found).real) / energy - 1) < 1e-4


@pytest.mark.parametrize("compensated", [False, True])
def test_df32_adams_adaptive_session_agrees_with_refined_binary64(compensated):
    actual = _run(_case(compensated=compensated, steps=None), work_limit=1)
    reference = _run(_case(compensated=compensated, single=False, steps=32), wide=None)
    coarse = _run(_case(compensated=compensated, single=False, steps=16), wide=None)
    for found, expected, previous in zip(
        actual.fields, reference.fields, coarse.fields, strict=True
    ):
        # Adaptive and prescribed histories differ: use the fixed global field
        # envelope with a separately settled reference, not a rounding comparison.
        scale = np.linalg.norm(expected)
        assert scale > 0
        assert np.linalg.norm(previous - expected) / scale < 1e-5
        assert np.linalg.norm(found - expected) / scale < 1e-3


def _saved(case, wide, *, final=False):
    with _session(case, wide=wide) as session:
        selected = None
        while not session.finished:
            batch = session.advance()
            for event in batch.events:
                if (
                    isinstance(event, ObservationReady)
                    and event.z > 0
                    and (selected is None or final)
                ):
                    snapshot = session.march_snapshot(event.ticket)
                    assert snapshot is not None
                    selected = (
                        snapshot,
                        session.export(event.ticket, IDENTITY).copy(),
                        session.execution_record()["response_calls"],
                        event.plane_index + 1,
                    )
            session.acknowledge(batch.receipt)
            if selected is not None and not final:
                result = drive(Started(session), export=IDENTITY)
                assert result.status.outcome is Outcome.COMPLETED
                return (*selected, result.fields[-1], session.execution_record())
        assert selected is not None
        return (*selected, selected[1], session.execution_record())


@pytest.mark.parametrize("compensated", [False, True])
@pytest.mark.parametrize(
    ("source", "destination"),
    [
        ("df32", "df32"),
        ("binary64", "df32"),
        ("df32", "binary64"),
        (None, "df32"),
        ("df32", None),
    ],
)
def test_df32_adams_restores_same_precision_history_without_startup(
    compensated, source, destination
):
    case = _case(compensated=compensated)
    snapshot, field, calls, next_plane, expected, record = _saved(case, source)
    assert set(snapshot.rows) == (
        {"store", "states", "rates"} if compensated else {"store"}
    )
    start = Start(
        field=field,
        z=snapshot.z,
        plane_index=next_plane,
        march=snapshot,
        ledgers={"logical": calls},
        plane_pending=False,
        resume=True,
    )
    with _session(case, wide=destination, start=start) as session:
        assert getattr(session, "march_refusal", None) is None
        result = drive(Started(session), export=IDENTITY, work_limit=1)
        assert result.status.outcome is Outcome.COMPLETED, result.batches[-1].terminal
        # Changing the wide graph or backend uses the SEP33 accuracy envelope;
        # portable metadata and source charging remain exact structural contracts.
        scale = np.linalg.norm(expected)
        assert scale > 0 and np.linalg.norm(result.fields[-1] - expected) / scale < 1e-3
        assert session.execution_record()["response_calls"] == record["response_calls"]
        if destination is not None:
            assert session.execution_record()["source_evaluations"] == 4


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("compensated", [False, True])
def test_df32_adams_raw_history_restore_is_lossless_and_completed_start_is_idle(
    compensated, final
):
    case = _case(compensated=compensated)
    snapshot, field, calls, next_plane, _, _ = _saved(case, "binary64", final=final)
    # A raw history is authoritative over a reconstructed physical launch field.
    field.flat[0] += 0.25
    start = Start(
        field=field,
        z=snapshot.z,
        plane_index=next_plane,
        march=snapshot,
        ledgers={"logical": calls},
        plane_pending=True,
        resume=True,
    )
    with _session(case, start=start) as session:
        batch = session.advance()
        event = next(
            event for event in batch.events if isinstance(event, ObservationReady)
        )
        restored = session.march_snapshot(event.ticket)
        assert restored is not None
        assert restored.state.tobytes() == snapshot.state.tobytes()
        assert restored.positions.tobytes() == snapshot.positions.tobytes()
        assert restored.slots.tobytes() == snapshot.slots.tobytes()
        assert restored.identities == snapshot.identities
        assert all(
            restored.rows[name].tobytes() == row.tobytes()
            for name, row in snapshot.rows.items()
        )
        assert session.execution_record()["source_evaluations"] == 0
        session.acknowledge(batch.receipt)
        if final:
            result = drive(Started(session), export=IDENTITY)
            assert result.status.outcome is Outcome.COMPLETED
            assert result.status.attempts == 0
            assert session.execution_record()["source_evaluations"] == 0
            assert all(
                size == 0
                for name, size in session.memory_ledger().items()
                if "plans.banks" in name
            )


def test_df32_adams_invalid_history_resets_before_binding_its_old_coordinates():
    case = _case(compensated=True)
    snapshot, field, calls, next_plane, _, _ = _saved(case, "binary64")
    positions, slots = snapshot.positions.copy(), snapshot.slots.copy()
    positions[0] = -1e10
    slots[0] = slots[-1]
    invalid = replace(snapshot, positions=positions, slots=slots)
    start = Start(
        field=field,
        z=snapshot.z,
        plane_index=next_plane,
        march=invalid,
        ledgers={"logical": calls},
        plane_pending=False,
        resume=True,
    )
    with _session(case, start=start) as session:
        assert "repeats a history slot" in str(session.march_refusal)
        result = drive(Started(session), export=IDENTITY)
        assert result.status.outcome is Outcome.COMPLETED
        restored = session.execution_record()["response_calls"]
        assert restored["startup"] == calls["startup"] + 1


def test_df32_adams_group_preserves_independent_plain_and_compensated_members():
    specifications = ((False, 0.75), (True, 1.0))
    cases = [
        _scaled(_case(compensated=kind), factor) for kind, factor in specifications
    ]
    fields = [[], []]
    recipe = ExportRecipe(power=True, optics=False)
    with ExitStack() as cleanup:
        group, _ = _group(cases, "df32", cleanup)
        while not group.finished:
            batch = group.advance()
            for index, events in enumerate(batch.members):
                for event in events:
                    if isinstance(event, ObservationReady):
                        fields[index].append(group.export(index, event.ticket, recipe))
            group.acknowledge(batch.receipt)
        assert batch.outcome is Outcome.COMPLETED
        for index, (kind, factor) in enumerate(specifications):
            status = group.member_status(index)
            assert status.attempts == 8 and status.rejected == 0
            record = group.execution_record(index)
            assert record["wide"] == "df32" and record["precision"] == "single"
            reference = _run(
                _scaled(_case(compensated=kind, single=False, steps=32), factor),
                wide=None,
                recipe=recipe,
            )
            coarse = _run(
                _scaled(_case(compensated=kind, single=False, steps=16), factor),
                wide=None,
                recipe=recipe,
            )
            for found, expected, previous in zip(
                fields[index], reference.fields, coarse.fields, strict=True
            ):
                # Independent members use the same SEP33 global accuracy envelope
                # as scalar execution, with separate binary64 step-size refinement.
                scale = np.linalg.norm(expected)
                assert scale > 0 and np.isfinite(found).all()
                assert np.linalg.norm(previous - expected) / scale < 1e-5
                assert np.linalg.norm(found - expected) / scale < 1e-3
                energy = float(np.vdot(expected, expected).real)
                assert energy > 0
                assert abs(float(np.vdot(previous, previous).real) / energy - 1) < 1e-6
                assert abs(float(np.vdot(found, found).real) / energy - 1) < 1e-4


def test_df32_adams_group_phase_refusal_precedes_seed_source_and_history_work():
    case = _case()
    constant = case.problem.linear.phi_constant
    assert constant is not None
    changed = np.full_like(constant, 1j * 2**40)
    changed.setflags(write=False)
    refused = Translation(
        replace(
            case.problem, linear=replace(case.problem.linear, phi_constant=changed)
        ),
        case.request,
    )
    with ExitStack() as cleanup:
        group, _ = _group([case, refused], "df32", cleanup)
        launch = group.advance()
        event = launch.members[1][0]
        assert isinstance(event, ObservationReady)
        original = group.export(1, event.ticket, IDENTITY).copy()
        group.acknowledge(launch.receipt)
        failed = group.advance()
        assert failed.outcome is Outcome.FAILED
        assert "DF32 function phase range exceeds 2^20" in str(failed.error)
        terminal = failed.members[1][0]
        assert isinstance(terminal, Terminal)
        assert terminal.ticket is not None
        np.testing.assert_array_equal(
            group.export(1, terminal.ticket, IDENTITY), original
        )
        record = group.execution_record(1)
        assert record["source_evaluations"] == 0
        assert record["response_calls"]["startup"] == 0
        assert group.member_status(1).attempts == 0
        group.acknowledge(failed.receipt)
