# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the managed Julia bridge with short complete binary64 sessions."""

import math
import os
import threading
from dataclasses import replace

import numpy as np
import pytest

from backends.julia.bounds import adams_rounding_budget, rk_rounding_budget
from backends.julia.cases import adams_case, cpu_case, low_storage_case
from sessions.executors import (
    IDENTITY,
    Started,
    drive,
    fibre,
    window,
)
from solprop.backends.events import (
    ObservationReady,
    Outcome,
    ServiceRequired,
    join_attempts,
)
from solprop.backends.export import ExportRecipe, apply_recipe
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.registry import select
from solprop.backends.session import ExecutionOptions, SessionError, Start, StopToken
from solprop.driver.run import run_problem
from solprop.driver.simulation import metadata_for
from solprop.exceptions import BackendError, ResponseBudgetError
from solprop.log import NullReporter
from solprop.output import MemoryOutput
from solprop.stepper.multistep import Multistep
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.problem import Translation
from solprop.translate.responses import ResponseStage

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _prepared(case):
    setup_id = os.environ.get("SOLPROP_TEST_JULIA_SETUP_ID")
    assert setup_id, "the enabled Julia lane must publish a managed setup identity"
    owner = JuliaBackend(
        case.problem,
        case.request,
        options=ExecutionOptions(backend_options={"setup_id": setup_id}),
    )
    return owner, owner.prepare(case.problem, case.request)


def _run(case, *, julia, work_limit=256, deadline=0.1):
    if julia:
        _, prepared = _prepared(case)
    else:
        owner = select(
            "numpy_ref", case.problem, case.request, grid=window(), modes=[fibre().mode]
        )
        prepared = owner.prepare(case.problem, case.request)
    session = prepared.start(Start.entrance(case.problem))
    with session:
        result = drive(
            Started(session), export=IDENTITY, work_limit=work_limit, deadline=deadline
        )
        assert result.status.outcome is Outcome.COMPLETED, result.batches[-1].terminal
        return result


@pytest.mark.parametrize("work_limit", [1, 256])
def test_eight_replayed_steps_match_the_reference_with_a_rounding_budget(work_limit):
    case = cpu_case(dz_max=1e-5)
    bound = rk_rounding_budget(case)
    actual = _run(case, julia=True, work_limit=work_limit, deadline=0.0)
    reference = _run(case, julia=False)
    assert bound > 0
    assert actual.status.outcome is Outcome.COMPLETED
    assert actual.status.attempts == reference.status.attempts == 8
    assert actual.status.rejected == reference.status.rejected == 0
    assert len(actual.fields) == len(reference.fields) == 3
    for got, expected in zip(actual.fields, reference.fields, strict=True):
        assert np.linalg.norm((got - expected).ravel()) <= bound
        assert np.all(np.isfinite(got))
    chunks = [
        (getattr(event, "history", None), getattr(event, "step_stats", None))
        for batch in actual.batches
        for event in batch.events
    ]
    history, rows = join_attempts(chunks)
    assert history.n_steps == 8
    assert case.request.replay is not None
    assert np.array_equal(history.dz, case.request.replay.schedule.steps)
    assert np.all(np.diff(history.wall_time) >= 0)
    assert rows.z.size == 0


def test_adaptive_session_meets_a_converged_reference_envelope():
    actual = _run(cpu_case(steps=None, rtol=1e-8), julia=True)
    coarse = _run(cpu_case(steps=128), julia=False)
    fine = _run(cpu_case(steps=256), julia=False)
    scale = np.linalg.norm(fine.fields[-1].ravel())
    # The independently refined reference must settle far inside the fixed global
    # accuracy envelope; a local rtol is not used as a cross-run equality tolerance.
    assert np.linalg.norm((fine.fields[-1] - coarse.fields[-1]).ravel()) < 1e-11 * scale
    assert np.linalg.norm((actual.fields[-1] - fine.fields[-1]).ravel()) < 1e-8 * scale
    assert actual.status.outcome is Outcome.COMPLETED


@pytest.mark.parametrize("unit_weight", [False, True])
def test_destination_exports_match_exact_recipe_bits_and_retain_arrays(unit_weight):
    case = cpu_case()
    weight = (
        np.ones(case.problem.shape)
        if unit_weight
        else np.linspace(0.4, 2.0, math.prod(case.problem.shape)).reshape(
            case.problem.shape
        )
    )
    table = PiecewiseChebyshev(
        weight[None, None],
        np.array(case.problem.linear.z_span),
        np.array([0], dtype=np.intp),
        np.empty(0),
    )
    for values in (
        weight,
        table.coefficients,
        table.breakpoints,
        table.degrees,
        table.hard_kinks,
    ):
        values.setflags(write=False)
    problem = replace(
        case.problem,
        linear=replace(
            case.problem.linear, power_weight_constant=weight, power_weight=table
        ),
    )
    case = Translation(problem, case.request)
    _, prepared = _prepared(case)
    state = np.array(problem.initial.field0)
    parts = state.view(np.float64).reshape(*state.shape, 2)
    parts[..., ::3, 0] = -0.0
    parts[..., 1::4, 1] = -0.0
    session = prepared.start(Start(state, 0.0))
    with session:
        launch = session.advance().events[0]
        assert isinstance(launch, ObservationReady)
        raw = session.export(launch.ticket, IDENTITY)
        retained = raw.copy()
        for power in (False, True):
            for optics in (False, True):
                recipe = ExportRecipe(power=power, optics=optics)
                expected = apply_recipe(raw, recipe, weight=weight)
                out = np.empty_like(raw)
                assert session.export(launch.ticket, recipe, out) is out
                assert out.tobytes() == expected.tobytes()
                cached = session.export(launch.ticket, recipe)
                assert not cached.flags.writeable
                assert cached is session.export(launch.ticket, recipe)
        wrong = np.zeros(raw.shape, dtype=np.complex64)
        with pytest.raises(BackendError, match="complex128"):
            session.export(launch.ticket, IDENTITY, wrong)
        assert np.all(wrong == 0)
        session.acknowledge(launch.ticket.receipt)
        with pytest.raises(SessionError, match="expired"):
            session.export(launch.ticket, IDENTITY)
        following = session.advance(work_limit=1)
        assert raw.tobytes() == retained.tobytes()
        session.acknowledge(following.receipt)


def test_real_snapshot_barrier_and_foreign_thread_stop_leave_valid_exports():
    case = cpu_case()
    _, prepared = _prepared(case)
    token = StopToken()
    session = prepared.start(Start.entrance(case.problem), stop=token)
    with session:
        first = session.request_snapshot()
        assert session.request_snapshot().request == first.request
        launch = session.advance()
        session.acknowledge(launch.receipt)
        batch = session.advance(deadline=60.0)
        snapshot = batch.events[0]
        assert isinstance(snapshot, ServiceRequired)
        assert snapshot.z == 0.0025
        assert snapshot.ticket is not None
        retained = session.export(snapshot.ticket, IDENTITY)
        assert session.march_snapshot(snapshot.ticket) is None
        session.acknowledge(batch.receipt)
        worker = threading.Thread(target=token.set)
        worker.start()
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        stopped = session.advance().terminal
        assert stopped is not None and stopped.outcome is Outcome.STOPPED
        assert stopped.z == snapshot.z
        assert stopped.ticket is not None
        # No numerical operation lies between these two observations of one state.
        assert session.export(stopped.ticket, IDENTITY).tobytes() == retained.tobytes()


def test_real_failed_listener_delivers_its_held_attempt_exactly_once():
    case = cpu_case()
    _, prepared = _prepared(case)

    def listener(*_):
        raise ValueError("injected consumer failure")

    session = prepared.start(Start.entrance(case.problem), listener=listener)
    with session:
        launch = session.advance()
        session.acknowledge(launch.receipt)
        with pytest.raises(ValueError, match="consumer failure"):
            session.advance()
        ticket = session.failure_ticket()
        assert ticket is not None
        chunk, _ = session.undelivered(ticket)
        assert chunk.n_steps == 1
        assert session.undelivered(ticket)[0].n_steps == 0
        assert np.all(np.isfinite(session.export(ticket, IDENTITY)))


@pytest.mark.parametrize("fields", ["all", "none"])
def test_shared_service_loop_uses_only_the_execution_owner(fields):
    case = cpu_case()
    owner, _ = _prepared(case)
    output = MemoryOutput()
    starts = []

    class Reporter(NullReporter):
        def started(self, z_start, z_end, n_planes):
            assert output.n_saved == 1
            starts.append((z_start, z_end, n_planes))

    try:
        run_problem(
            case.problem,
            owner,
            output,
            start=Start.entrance(case.problem),
            metadata=metadata_for(
                case.problem, case.request, window(), modes=[fibre().mode]
            ),
            stats_spec=case.request.observation.stats,
            progress=Reporter(),
            fields=fields,
        )
        assert starts == [(0.0, 0.02, 2)]
        assert output.n_saved == 3
        assert np.count_nonzero(output["step_history/accepted"]) == 8
    finally:
        owner.close()


def _material_run(kind, response, *, julia, steps):
    from backends.julia.cases import material_case  # noqa: PLC0415

    case, grid, mode = material_case(kind, response, steps=steps)
    if julia:
        _, prepared = _prepared(case)
    else:
        owner = select("numpy_ref", case.problem, case.request, grid=grid, modes=[mode])
        prepared = owner.prepare(case.problem, case.request)
    session = prepared.start(Start.entrance(case.problem))
    with session:
        result = drive(Started(session), export=IDENTITY)
        assert result.status.outcome is Outcome.COMPLETED, result.batches[-1].terminal
        assert result.status.attempts == steps
        assert np.all(np.isfinite(result.fields[-1]))
        return result.fields[-1]


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("response", ["kerr", "raman", "plasma"])
def test_material_sessions_meet_short_converged_reference_envelopes(kind, response):
    actual = _material_run(kind, response, julia=True, steps=8)
    coarse = _material_run(kind, response, julia=False, steps=16)
    fine = _material_run(kind, response, julia=False, steps=32)
    scale = np.linalg.norm(fine.ravel())
    # These are different discrete histories. The independent reference must settle
    # three orders inside the fixed1e-8 global accuracy envelope over eight micrometres.
    assert np.linalg.norm((fine - coarse).ravel()) <= 1e-11 * scale
    assert np.linalg.norm((actual - fine).ravel()) <= 1e-8 * scale


@pytest.mark.parametrize("representation", ["expansion", "table"])
def test_varying_field_operator_and_prefactor_use_complete_representations(
    representation, capillary_problem, tabulated_problem, field_grid
):
    from solprop.stats import StatsSpec  # noqa: PLC0415
    from solprop.stepper.schedule import FixedSchedule  # noqa: PLC0415
    from solprop.translate.run_request import ReplaySpec  # noqa: PLC0415

    original = (
        capillary_problem() if representation == "expansion" else tabulated_problem
    )
    length = 8e-6

    def run(steps, julia):
        # The represented descriptor spans later material kinks. Retain those
        # itinerary edges for admission; the sole observation ends this short run.
        later = original.problem.linear.hard_kinks
        edges = np.linspace(0.0, length, steps + 1)
        for kink in later[later > length]:
            # Unused future steps must still respect the temporal collar's
            # admission bound; only the first eight micrometres are executed.
            count = max(1, int(np.ceil((kink - edges[-1]) / 1e-3)))
            edges = np.concatenate((edges, np.linspace(edges[-1], kink, count + 1)[1:]))
        request = replace(
            original.request,
            replay=ReplaySpec(FixedSchedule(np.diff(edges))),
            observation=replace(
                original.request.observation,
                save_planes=np.array([length]),
                stats=StatsSpec(names=(), periodic=()),
            ),
        )
        case = Translation(original.problem, request)
        if julia:
            _, prepared = _prepared(case)
        else:
            owner = select("numpy_ref", case.problem, request, grid=field_grid)
            prepared = owner.prepare(case.problem, request)
        session = prepared.start(Start.entrance(case.problem))
        with session:
            result = drive(Started(session), export=IDENTITY)
            terminal = result.batches[-1].terminal
            assert result.status.outcome is Outcome.COMPLETED, (
                None if terminal is None else str(terminal.error)
            )
            return result.fields[-1]

    actual, coarse, fine = run(8, True), run(16, False), run(32, False)
    scale = np.linalg.norm(fine.ravel())
    # Varying represented operators are compared to the refined reference using
    # the same fixed short-window accuracy envelope as the material compositions.
    assert np.linalg.norm((fine - coarse).ravel()) <= 1e-11 * scale
    assert np.linalg.norm((actual - fine).ravel()) <= 1e-8 * scale


@pytest.mark.parametrize("representation", ["table", "expansion"])
def test_power_export_evaluates_weights_at_the_restart_position(representation):
    from solprop.translate.expansion import ParameterExpansion  # noqa: PLC0415

    case = cpu_case(steps=None)
    shape = case.problem.shape
    span = np.array(case.problem.linear.z_span)
    midpoint = float(np.mean(span))
    # The exact midpoint of this affine representation gives weight1.5. The
    # destination recipe is then a structural test of one state's fixed conversion.
    table = PiecewiseChebyshev(
        np.stack((np.full(shape, 1.5), np.full(shape, 0.5)))[None],
        span.copy(),
        np.array([1], dtype=np.intp),
        np.empty(0),
    )
    for value in (
        table.coefficients,
        table.breakpoints,
        table.degrees,
        table.hard_kinks,
    ):
        value.setflags(write=False)
    expansion = None
    if representation == "expansion":
        weights = PiecewiseChebyshev(
            np.array([[[1.0, 0.5], [0.0, 0.5]]]),
            span.copy(),
            np.array([1], dtype=np.intp),
            np.empty(0),
        )
        for value in (
            weights.coefficients,
            weights.breakpoints,
            weights.degrees,
            weights.hard_kinks,
        ):
            value.setflags(write=False)
        expansion = ParameterExpansion(
            ("test",),
            np.array([[0.0, 1.0]]),
            np.array([1]),
            np.ones((2, *shape)),
            weights,
        )
    problem = replace(
        case.problem,
        linear=replace(
            case.problem.linear,
            power_weight_constant=None,
            power_weight_representation=representation,
            power_weight=table if expansion is None else None,
            power_weight_expansion=expansion,
        ),
    )
    _, prepared = _prepared(Translation(problem, case.request))
    with prepared.start(
        Start(np.array(problem.initial.field0), midpoint, plane_index=1)
    ) as session:
        launch = session.advance().events[0]
        assert isinstance(launch, ObservationReady)
        assert launch.z == midpoint
        raw = session.export(launch.ticket, IDENTITY)
        recipe = ExportRecipe(power=True, optics=True)
        expected = apply_recipe(raw, recipe, weight=np.full(shape, 1.5))
        assert session.export(launch.ticket, recipe).tobytes() == expected.tobytes()


def test_adaptive_adams_meets_an_independently_converged_rk_reference():
    actual = _run(adams_case(steps=None, rtol=1e-8), julia=True)
    coarse = _run(cpu_case(steps=128), julia=False)
    fine = _run(cpu_case(steps=256), julia=False)
    scale = np.linalg.norm(fine.fields[-1].ravel())
    # The RK refinement settles three orders below the global accuracy envelope;
    # no equality of adaptive attempts or local error readings is assumed.
    assert np.linalg.norm((fine.fields[-1] - coarse.fields[-1]).ravel()) < 1e-11 * scale
    assert np.linalg.norm((actual.fields[-1] - fine.fields[-1]).ravel()) < 1e-8 * scale
    assert actual.status.outcome is Outcome.COMPLETED


@pytest.mark.parametrize("order", [1, 5, 6])
@pytest.mark.parametrize("work_limit", [1, 256])
def test_fixed_adams_histories_match_with_conditioned_rounding_budget(
    order, work_limit
):
    case = adams_case(method=Multistep(order=order, variable_order=False))
    bound = adams_rounding_budget(case, order)
    assert np.isfinite(bound) and bound > 0
    runs, records = [], []
    for julia in (True, False):
        if julia:
            _, prepared = _prepared(case)
        else:
            owner = select(
                "numpy_ref",
                case.problem,
                case.request,
                grid=window(),
                modes=[fibre().mode],
            )
            prepared = owner.prepare(case.problem, case.request)
        with prepared.start(Start.entrance(case.problem)) as session:
            result = drive(Started(session), export=IDENTITY, work_limit=work_limit)
            assert result.status.outcome is Outcome.COMPLETED
            runs.append(result)
            records.append(session.execution_record()["response_calls"])
    # Identical histories/order choices are an explicit hypothesis of this bound;
    # ordinary replay or an adaptive trajectory would need an accuracy envelope.
    assert records[0] == records[1]
    assert records[0]["accepted_steps"] == 8
    assert records[0]["rejected_steps"] == 0
    assert records[0]["restarts"] == 0
    for actual, expected in zip(runs[0].fields, runs[1].fields, strict=True):
        assert np.linalg.norm((actual - expected).ravel()) <= bound


@pytest.mark.parametrize("work_limit", [1, 256])
@pytest.mark.parametrize("absent", [False, True])
def test_adams_budget_retains_the_accepted_field_and_exact_logical_counts(
    work_limit, absent
):
    case = adams_case(method=Multistep(order=5, variable_order=False, max_calls=4))
    if absent:
        responses = replace(
            case.problem.responses,
            response_order=(ResponseStage.ACCUMULATE,),
            kerr_coefficients=None,
        )
        case = Translation(replace(case.problem, responses=responses), case.request)
    _, prepared = _prepared(case)
    with prepared.start(Start.entrance(case.problem)) as session:
        result = drive(Started(session), export=IDENTITY, work_limit=work_limit)
        error = result.terminal.error
        assert result.status.outcome is Outcome.FAILED
        assert isinstance(error, ResponseBudgetError)
        assert error.calls == 4
        assert case.request.replay is not None
        assert error.z == case.request.replay.schedule.edges[3]
        calls = session.execution_record()["response_calls"]
        assert calls == {
            "startup": 1,
            "predictor": 3,
            "corrected": 0,
            "rejected": 0,
            "accepted_steps": 3,
            "rejected_steps": 0,
            "restarts": 0,
            "failure_restarts": 0,
            "dense_observations": 0,
            "orders": [[1, 1], [2, 1], [3, 1]],
        }
        assert session.execution_record()["source_evaluations"] == (0 if absent else 4)
        # The exception owns the last accepted internal field, independent of close.
        field = np.array(error.field)
        assert field.shape == case.problem.shape
        assert np.all(np.isfinite(field))
    assert isinstance(error.field, np.ndarray)
    assert np.array_equal(error.field, field)
    owner = select(
        "numpy_ref", case.problem, case.request, grid=window(), modes=[fibre().mode]
    )
    with owner.prepare(case.problem, case.request).start(
        Start.entrance(case.problem)
    ) as reference:
        witness = drive(Started(reference), export=IDENTITY)
        other = witness.terminal.error
        assert isinstance(other, ResponseBudgetError)
        assert str(error) == str(other)
        assert calls == reference.execution_record()["response_calls"]


@pytest.mark.parametrize("work_limit", [1, 256])
def test_low_storage_replay_matches_the_reference_and_committed_counts(work_limit):
    case = low_storage_case()
    bound = rk_rounding_budget(case)
    _, prepared = _prepared(case)
    with prepared.start(Start.entrance(case.problem)) as session:
        actual = drive(Started(session), export=IDENTITY, work_limit=work_limit)
        record = session.execution_record()
        assert actual.status.outcome is Outcome.COMPLETED
        assert record["source_evaluations"] == 32
        assert record["attempted_source_evaluations"] == 32
        assert record["norm_evaluations"] == 8
        assert "response_calls" not in record
        history, _ = join_attempts(actual.chunks)
        assert history.n_steps == 8
        assert np.all(history.error_norm == 0)  # The method declares 'not estimated'.
        assert np.all(history.accepted)
        assert case.request.replay is not None
        assert np.array_equal(history.dz, case.request.replay.schedule.steps)
    reference = _run(case, julia=False)
    assert bound > 0
    for got, expected in zip(actual.fields, reference.fields, strict=True):
        assert np.linalg.norm((got - expected).ravel()) <= bound


def test_low_storage_source_failure_poisoning_retains_attempted_work():
    from solprop.backends.events import Eligibility  # noqa: PLC0415

    case = low_storage_case()
    assert case.request.replay is not None
    kerr = case.problem.responses.kerr_coefficients
    assert kerr is not None
    problem = replace(
        case.problem,
        responses=replace(
            case.problem.responses,
            kerr_coefficients=replace(kerr, coefficient=1e300),
        ),
    )
    _, prepared = _prepared(Translation(problem, case.request))
    with prepared.start(Start.entrance(problem)) as session:
        launch = session.advance()
        session.acknowledge(launch.receipt)
        failed = session.advance()
        terminal = failed.terminal
        assert terminal is not None and terminal.outcome is Outcome.FAILED
        assert terminal.eligibility is Eligibility.POISONED
        assert terminal.ticket is None
        assert session.failure_ticket() is None
        assert session.in_flight == (0.0, case.request.replay.schedule.steps[0])
        record = session.execution_record()
        assert record["source_evaluations"] == 0
        assert record["attempted_source_evaluations"] == 1
        assert record["norm_evaluations"] == 0
        session.acknowledge(failed.receipt)
