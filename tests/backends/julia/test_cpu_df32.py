# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify actual DF32 session attachment against settled binary64 references."""

import os
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.cases import cpu_case, low_storage_case
from sessions.executors import IDENTITY, Started, drive, fibre, window
from solprop.backends.events import ObservationReady, Outcome
from solprop.backends.export import ExportRecipe
from solprop.backends.julia._session import JuliaSession
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.registry import select
from solprop.backends.session import ExecutionOptions, Start
from solprop.translate.precision import SINGLE
from solprop.translate.problem import Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _case(low, steps, *, single=False):
    factory = low_storage_case if low else cpu_case
    # RK replay may exceed the adaptive ceiling; the established low-storage
    # contract instead records dz_max as a true ceiling on its prescribed steps.
    case = factory(steps=steps, dz_max=0.02 / steps if low else 1e-5)
    return Translation(
        replace(case.problem, precision=SINGLE) if single else case.problem,
        replace(
            case.request,
            integration=replace(case.request.integration, propagator_mode="recompute"),
        ),
    )


def _run(case, *, wide=None, work_limit=256, export=None):
    if export is None:
        export = ExportRecipe(power=True, optics=False)
    if wide is None:
        owner = select(
            "numpy_ref", case.problem, case.request, grid=window(), modes=[fibre().mode]
        )
    else:
        setup_id = os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]
        owner = JuliaBackend(
            case.problem,
            case.request,
            options=ExecutionOptions(
                backend_options={"setup_id": setup_id, "wide": wide}
            ),
        )
    prepared = owner.prepare(case.problem, case.request)
    start = Start.entrance(case.problem)
    with prepared.start(start) as session:
        result = drive(
            Started(session),
            export=export,
            work_limit=work_limit,
        )
        assert result.status.outcome is Outcome.COMPLETED
        assert all(np.isfinite(field).all() for field in result.fields)
        if wide is not None:
            assert isinstance(session, JuliaSession)
            assert prepared.metadata()["wide"] == wide
            assert any("plans.banks" in name for name in session.memory_ledger())
        return result


@pytest.mark.parametrize("low", [False, True])
@pytest.mark.parametrize("work_limit", [1, 256])
def test_df32_actual_sessions_meet_settled_binary64_envelopes(low, work_limit):
    actual = _run(_case(low, 8, single=True), wide="df32", work_limit=work_limit)
    reference = _run(_case(low, 32))
    coarse = _run(_case(low, 16))
    assert actual.status.attempts == 8 and actual.status.rejected == 0
    assert len(actual.fields) == len(reference.fields) == len(coarse.fields)
    # SEP 33 section 7.3 fixes field L2 <=1e-3 and energy <=1e-4. These short
    # ordinary-replay histories use an accuracy envelope (SEP 25), not a matched
    # arithmetic bound. Require the binary64 refinement to settle 100 times tighter.
    # Power-normalized envelope exports make the common Parseval energy prefactor
    # cancel exactly from the ratio; no vacuum/modal-weight approximation is used.
    for found, expected, previous in zip(
        actual.fields, reference.fields, coarse.fields, strict=True
    ):
        scale = np.linalg.norm(expected)
        assert scale > 0
        assert np.linalg.norm(previous - expected) / scale < 1e-5
        assert np.linalg.norm(found - expected) / scale < 1e-3
        energy = float(np.vdot(expected, expected).real)
        assert energy > 0
        assert abs(float(np.vdot(previous, previous).real) / energy - 1) < 1e-6
        assert abs(float(np.vdot(found, found).real) / energy - 1) < 1e-4


def _prepared(case):
    owner = JuliaBackend(
        case.problem,
        case.request,
        options=ExecutionOptions(
            backend_options={
                "setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"],
                "wide": "df32",
            }
        ),
    )
    return owner.prepare(case.problem, case.request)


def _large_phase_case(low):
    case = _case(low, 8, single=True)
    constant = case.problem.linear.phi_constant
    assert constant is not None
    # The imaginary rate is finite and representable, but an actual replay step
    # gives a phase beyond the derived DF32 transcendental domain of 2^20 rad.
    changed = np.full_like(constant, 1j * 2**40)
    changed.setflags(write=False)
    return Translation(
        replace(
            case.problem, linear=replace(case.problem.linear, phi_constant=changed)
        ),
        case.request,
    )


@pytest.mark.parametrize("low", [False, True])
def test_df32_actual_phase_refusal_preserves_field_before_source_work(low):
    case = _large_phase_case(low)
    prepared = _prepared(case)
    with prepared.start(Start.entrance(case.problem)) as session:
        launch = session.advance()
        observation = next(
            event for event in launch.events if isinstance(event, ObservationReady)
        )
        original = session.export(observation.ticket, IDENTITY).copy()
        session.acknowledge(launch.receipt)
        failed = session.advance()
        terminal = failed.terminal
        assert terminal is not None and terminal.outcome is Outcome.FAILED
        assert "wide certificate refused" in str(terminal.error)
        assert session.in_flight is None
        ticket = session.failure_ticket()
        assert ticket is not None
        # The certificate precedes every numerical mutation; this is a structural
        # preservation assertion, not a cross-run floating-point comparison.
        np.testing.assert_array_equal(session.export(ticket, IDENTITY), original)
        record = session.execution_record()
        if low:
            assert record["attempted_source_evaluations"] == 0
            assert record["norm_evaluations"] == 0
        assert record["source_evaluations"] == 0
        session.acknowledge(failed.receipt)


@pytest.mark.parametrize("low", [False, True])
def test_df32_completed_start_does_not_invent_a_phase_or_source_call(low):
    case = _large_phase_case(low)
    planes = case.request.observation.save_planes
    start = Start(
        field=case.problem.initial.field0.copy(),
        z=float(planes[-1]),
        plane_index=len(planes),
        plane_pending=False,
        resume=True,
    )
    prepared = _prepared(case)
    with prepared.start(start) as session:
        result = drive(Started(session), export=IDENTITY)
        assert result.status.outcome is Outcome.COMPLETED
        assert result.status.attempts == 0 and result.status.rejected == 0
        record = session.execution_record()
        assert record["source_evaluations"] == 0
        if low:
            assert record["attempted_source_evaluations"] == 0
        assert all(
            size == 0
            for name, size in session.memory_ledger().items()
            if "plans.banks" in name
        )


@pytest.mark.parametrize("low", [False, True])
def test_df32_field_restart_binds_remaining_replay_at_the_actual_start(low):
    case = _case(low, 8, single=True)
    complete = _run(case, wide="df32", export=IDENTITY)
    event = complete.observations[1]
    assert 0.0 < event.z < case.request.observation.save_planes[-1]
    start = Start(
        field=complete.fields[1].copy(),
        z=event.z,
        plane_index=event.plane_index + 1,
        plane_pending=False,
        resume=True,
    )
    prepared = _prepared(case)
    with prepared.start(start) as session:
        resumed = drive(Started(session), export=IDENTITY, work_limit=1)
        assert resumed.status.outcome is Outcome.COMPLETED
        assert resumed.status.attempts == 4 and resumed.status.rejected == 0
        assert start.field is None
        # A field restart resets method state and rounds its exported physical
        # field on import. The accepted SEP33 field envelope bounds this comparison;
        # the independent reference settlement is covered above.
        expected = complete.fields[-1]
        scale = np.linalg.norm(expected)
        assert scale > 0.0
        assert np.linalg.norm(resumed.fields[-1] - expected) / scale < 1e-3
