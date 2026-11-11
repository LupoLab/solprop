# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify per-start input noise, phase tickets and continuation through JuliaCall."""

import os
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.cases import adams_case, cpu_case, low_storage_case, material_case
from backends.julia.noise_bounds import temporal_noise_floor_bound
from backends.julia.test_cpu_dense import _case as dense_case
from backends.julia.test_cpu_monitor import _record, _start
from backends.julia.test_cpu_session import _prepared, _run
from sessions.executors import IDENTITY, Started, drive, fibre, window
from solprop.backends.events import ObservationReady, Outcome
from solprop.backends.julia import _calls
from solprop.backends.registry import select
from solprop.backends.session import SessionError, Start
from solprop.exceptions import BackendError
from solprop.noise import NoiseRealisation
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.linop import unitary_phase
from solprop.translate.problem import Initial, Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _noisy(case, *, launch=0.0, scale=1.0):
    frequencies = case.problem.shape[-1]
    draw = scale * (0.125 + 0.0625j) * np.exp(1j * np.arange(frequencies) / 8)
    draw *= case.problem.grids.active_mask
    initial = Initial(case.problem.initial.field0, draw[None, None], "input", launch)
    return Translation(
        replace(
            case.problem,
            initial=initial,
            flags=replace(case.problem.flags, has_noise=True),
        ),
        case.request,
    )


@pytest.mark.parametrize("factory", [cpu_case, adams_case, low_storage_case])
@pytest.mark.parametrize("resume", [False, True])
def test_input_draw_is_added_only_at_a_fresh_launch(factory, resume):
    case = _noisy(factory())
    source = np.array(case.problem.initial.field0)
    override = NoiseRealisation(np.full(case.problem.shape, 0.25 - 0.5j), 7, "modified")
    start = Start(field=source.copy(), z=0.0, noise=override, resume=resume)
    _, prepared = _prepared(case)
    with prepared.start(start) as session:
        launch = session.advance().events[0]
        assert isinstance(launch, ObservationReady)
        expected = source if resume else source + override.spectrum
        expected = np.where(case.problem.grids.active_mask[None, None], expected, 0.0)
        # The draw-only override metadata cannot switch the active input model.
        # Addition and hard masking are the exact same structural launch operation.
        np.testing.assert_array_equal(session.export(launch.ticket, IDENTITY), expected)
        assert session.execution_record()["source_evaluations"] == 0
        phase = session.noise_phase(launch.ticket)
        assert phase is not None and not phase.flags.writeable
        assert np.count_nonzero(phase) == 0
        assert session.noise_phase(launch.ticket) is phase
        ledger = session.memory_ledger()
        assert any("noise.spectrum" in name for name in ledger)
        assert ledger["input_noise.phase_export"] == phase.nbytes
    assert start.field is None
    assert np.count_nonzero(phase) == 0


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_nonzero_problem_launch_adds_once_without_resetting_absolute_phase(kind):
    base, _, _ = material_case(kind, "kerr")
    case = _noisy(base, launch=2e-6)
    _, prepared = _prepared(case)
    with prepared.start(Start.entrance(case.problem)) as session:
        launch = session.advance().events[0]
        assert isinstance(launch, ObservationReady) and launch.z == 2e-6
        expected = case.problem.initial.field0 + case.problem.initial.noise0
        expected = np.where(case.problem.grids.active_mask[None, None], expected, 0.0)
        np.testing.assert_array_equal(session.export(launch.ticket, IDENTITY), expected)
        phase = session.noise_phase(launch.ticket)
        reference = unitary_phase(case.problem.linear, launch.z)[0]
        assert phase is not None
        # One multiplication per entry in this constant operator, independently
        # evaluated. The positive tiny term covers structurally zero components.
        bound = (
            4
            * np.finfo(float).eps
            * np.maximum(np.abs(reference), np.finfo(float).tiny)
        )
        assert np.all(np.abs(phase - reference) <= bound)


def test_later_fresh_start_keeps_its_supplied_field_without_another_draw():
    case = _noisy(cpu_case(steps=None))
    start = Start(field=np.array(case.problem.initial.field0), z=0.005)
    _, prepared = _prepared(case)
    with prepared.start(start) as session:
        launch = session.advance().events[0]
        assert isinstance(launch, ObservationReady)
        expected = np.where(
            case.problem.grids.active_mask[None, None], case.problem.initial.field0, 0.0
        )
        np.testing.assert_array_equal(session.export(launch.ticket, IDENTITY), expected)


@pytest.mark.parametrize("representation", ["table", "expansion"])
def test_represented_phase_keeps_the_absolute_cumulative_offset(representation):
    case = cpu_case(steps=None)
    shape = (1, *case.problem.shape)
    span = np.array(case.problem.linear.z_span)
    launch = float(span[0] + (span[-1] - span[0]) / 4)
    first = np.full(shape, -0.25 + 0.375j, dtype=np.complex128)
    second = (np.arange(np.prod(shape)).reshape(shape) % 17 - 8) * (0.125 + 0.25j)
    coefficients = np.stack((first, second))[None]
    weights = PiecewiseChebyshev(
        np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        span,
        np.array([1], dtype=np.intp),
        np.empty(0),
    )
    terms = np.stack((first, second))
    expansion = ParameterExpansion(
        ("test",), np.array([[0.0, 1.0]]), np.array([1]), terms, weights
    )
    for value in (
        coefficients,
        span,
        terms,
        weights.coefficients,
        weights.degrees,
        weights.hard_kinks,
    ):
        value.setflags(write=False)
    linear = replace(
        case.problem.linear,
        phi_representation=representation,
        phi_breakpoints=span,
        phi_coefficients=coefficients if representation == "table" else None,
        phi_degrees=weights.degrees,
        phi_constant=None,
        phi_expansion=expansion if representation == "expansion" else None,
    )
    case = _noisy(
        Translation(replace(case.problem, linear=linear), case.request), launch=launch
    )
    _, prepared = _prepared(case)
    start = Start(field=case.problem.initial.field0, z=launch, plane_index=0)
    with prepared.start(start) as session:
        event = session.advance().events[0]
        assert isinstance(event, ObservationReady)
        phase = session.noise_phase(event.ticket)
        expected = unitary_phase(linear, launch)[0]
        assert phase is not None and np.any(expected != 0)
        # Degree-one Clenshaw and a two-term expansion use fewer than32 real
        # operations per component. Sum the absolute input coefficients so this
        # comparison remains bounded at cancellation, independent of the result.
        magnitude = np.abs(first.imag) + np.abs(second.imag)
        u = np.finfo(float).eps / 2
        bound = 2 * 32 * u / (1 - 32 * u) * magnitude[0]
        assert np.all(bound > 0)
        assert np.all(np.abs(phase - expected) <= bound)


def test_dense_phase_uses_observed_z_and_invalid_tickets_do_not_reenter(monkeypatch):
    case = _noisy(dense_case())
    retained = []
    dense = 0
    with _start(case) as session:
        while not session.finished:
            batch = session.advance(deadline=60.0)
            for event in batch.events:
                if not isinstance(event, ObservationReady):
                    continue
                phase = session.noise_phase(event.ticket)
                expected = unitary_phase(case.problem.linear, event.z)[0]
                assert phase is not None
                bound = (
                    4
                    * np.finfo(float).eps
                    * np.maximum(np.abs(expected), np.finfo(float).tiny)
                )
                assert np.all(np.abs(phase - expected) <= bound)
                retained.append((phase, phase.tobytes()))
                dense += event.z < session.status().z
            session.acknowledge(batch.receipt)
            if isinstance(batch.events[0], ObservationReady):

                def unexpected(*args):
                    pytest.fail("an invalid phase ticket entered Julia")

                with monkeypatch.context() as patch:
                    patch.setattr(_calls, "call_void", unexpected)
                    with pytest.raises(SessionError):
                        session.noise_phase(batch.events[0].ticket)
        assert session.status().outcome is Outcome.COMPLETED
    assert dense > 0
    assert all(value.tobytes() == saved for value, saved in retained)


def test_noise_floor_matches_reference_and_is_retained_on_resume():
    case = _noisy(dense_case())
    reference = (
        select(
            "numpy_ref",
            case.problem,
            case.request,
            grid=window(),
            modes=(fibre().mode,),
        )
        .prepare(case.problem, case.request)
        .start(Start.entrance(case.problem))
    )
    with reference:
        launch = reference.advance()
        reference.acknowledge(launch.receipt)
        # The reference installs its monitor on the next advance after the launch
        # acknowledgment; its floor then continues to describe that launch.
        reference.advance(work_limit=1)
        expected = _record(reference)["noise_floor"]
    with _start(case) as session:
        launch = session.advance()
        event = launch.events[0]
        assert isinstance(event, ObservationReady)
        field = session.export(event.ticket, IDENTITY).copy()
        session.acknowledge(launch.receipt)
        # Julia also publishes the post-launch monitor record in its next batch.
        session.advance(work_limit=1)
        record = _record(session)
        actual = record["noise_floor"]
    assert expected > 0
    noise = case.problem.initial.noise0
    assert noise is not None
    bound = temporal_noise_floor_bound(noise, field)
    assert bound > 0
    assert abs(actual - expected) <= bound
    restart = Start(
        field=field,
        z=0.0,
        resume=True,
        monitor={"record": record, "plane": True, "unwatched": []},
    )
    with _start(case, restart) as resumed:
        launch = resumed.advance()
        event = launch.events[0]
        assert isinstance(event, ObservationReady)
        np.testing.assert_array_equal(resumed.export(event.ticket, IDENTITY), field)
        resumed.acknowledge(launch.receipt)
        resumed.advance(work_limit=1)
        assert _record(resumed)["noise_floor"] == actual


def test_nonfinite_override_refuses_without_consuming_start_or_entering_julia(
    monkeypatch,
):
    case = _noisy(cpu_case())
    owner, prepared = _prepared(case)
    try:
        field = np.array(case.problem.initial.field0)
        start = Start(
            field=field,
            z=0.0,
            noise=NoiseRealisation(np.full(case.problem.shape, np.nan), None, "input"),
        )

        def unexpected(*args, **kwargs):
            pytest.fail("invalid noise entered Julia")

        monkeypatch.setattr(prepared._runtime, "own_call", unexpected)
        with pytest.raises(BackendError, match="finite"):
            prepared.start(start)
        assert start.field is field and not owner.closed
    finally:
        owner.close()


def test_noisy_adams_history_resume_matches_a_refined_reference():
    case = _noisy(adams_case(steps=None, rtol=1e-10))
    _, prepared = _prepared(case)
    with prepared.start(Start.entrance(case.problem)) as session:
        while True:
            batch = session.advance(deadline=60.0)
            event = batch.events[0]
            if isinstance(event, ObservationReady) and event.z > 0:
                snapshot = session.march_snapshot(event.ticket)
                assert snapshot is not None
                restart = Start(
                    field=snapshot.state.copy(),
                    z=snapshot.z,
                    plane_index=event.plane_index + 1,
                    plane_pending=False,
                    resume=True,
                    march=snapshot,
                    ledgers={"logical": session.execution_record()["response_calls"]},
                )
                break
            session.acknowledge(batch.receipt)
    _, prepared = _prepared(case)
    with prepared.start(restart) as resumed:
        assert resumed.march_refusal is None
        result = drive(Started(resumed), export=IDENTITY)
        assert result.status.outcome is Outcome.COMPLETED
    coarse = _run(_noisy(cpu_case(steps=256)), julia=False).fields[-1]
    fine = _run(_noisy(cpu_case(steps=512)), julia=False).fields[-1]
    scale = np.linalg.norm(fine.ravel())
    # Different histories use a fixed global envelope and a separately refined RK
    # reference; the full original draw remains in the resumed accepted field.
    assert np.linalg.norm((fine - coarse).ravel()) <= 1e-11 * scale
    assert np.linalg.norm((result.fields[-1] - fine).ravel()) <= 1e-8 * scale
