# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep descriptor lowering owned, lazy, cached and explicit about admission."""

import gc
import weakref
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest
from benchmarks.worker import _Started

from backends.julia.cases import adams_case, cpu_case, low_storage_case
from solprop.backends.julia import backend
from solprop.backends.julia._descriptors import (
    TableTransfer,
    lower_modal,
    lower_table,
    transfer_memory,
)
from solprop.backends.julia._lower import lower_run
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.session import ExecutionOptions, Start
from solprop.exceptions import BackendError
from solprop.stepper.multistep import ComparisonMultistep, Multistep, march_policy
from solprop.stepper.tableau import DOPRI5, continuous_extension
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.precision import SINGLE
from solprop.translate.responses import ResponseStage


def test_identical_quadrature_projection_has_no_embedded_diagnostic():
    case = cpu_case()
    transform = case.problem.transform
    assert np.array_equal(transform.projection, transform.projection_coarse)
    assert lower_modal(case.problem).coarse_projection is None
    represented = replace(transform, projection_coarse=0.5 * transform.projection)
    changed = lower_modal(
        replace(case.problem, route=replace(case.problem.route, transform=represented))
    )
    np.testing.assert_array_equal(
        changed.coarse_projection, represented.projection_coarse
    )


def test_lowering_caches_owned_read_only_arrays_and_releases_its_problem():
    case = cpu_case()
    problem = case.problem
    transfer = lower_modal(problem)
    assert lower_modal(problem) is transfer
    assert transfer.operator.values is not None
    assert transfer.operator.values.shape == (problem.shape[-1], problem.shape[0])
    assert not np.shares_memory(transfer.synthesis, problem.transform.synthesis)
    assert not np.shares_memory(transfer.projection, problem.transform.projection)
    assert not np.shares_memory(transfer.crop_index, problem.grids.crop_index)

    def check(value):
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            assert value.flags.owndata
            assert value.flags.f_contiguous
        elif is_dataclass(value):
            for descriptor in fields(value):
                check(getattr(value, descriptor.name))
        elif isinstance(value, tuple):
            for item in value:
                check(item)

    check(transfer)
    assert len(transfer_memory(transfer)) >= 10
    assert transfer.kerr is not None
    assert problem.responses.kerr_coefficients is not None
    assert transfer.kerr.coefficient == problem.responses.kerr_coefficients.coefficient
    original = weakref.ref(problem)
    del case, problem
    gc.collect()
    assert original() is None


def test_run_lowering_preserves_actual_replay_lengths_beyond_the_adaptive_ceiling():
    case = cpu_case(dz_max=1e-5)
    transfer = lower_run(case.problem, case.request)
    assert case.request.replay is not None
    assert np.array_equal(transfer.steps, case.request.replay.schedule.steps)
    assert np.array_equal(transfer.edges, case.request.replay.schedule.edges)
    assert not np.shares_memory(transfer.steps, case.request.replay.schedule.steps)
    assert np.all(transfer.steps > transfer.control.dz_max)
    assert np.array_equal(transfer.active, case.problem.grids.active_mask)
    assert transfer.active.dtype == np.bool_
    assert np.all(transfer.weights == 1.0)
    assert not transfer.matrix.flags.writeable


@pytest.mark.parametrize("factory", [cpu_case, adams_case, low_storage_case])
@pytest.mark.parametrize("mode", ["auto", "recompute", "tabulate"])
def test_run_lowering_resolves_actual_flow_table_policy(factory, mode):
    case = factory()
    request = replace(
        case.request,
        integration=replace(case.request.integration, propagator_mode=mode),
    )
    transfer = lower_run(case.problem, request)
    expected = factory is low_storage_case and (
        mode == "tabulate" or (mode == "auto" and case.problem.flags.constant_linop)
    )
    assert transfer.tabulate_flow is expected
    owner = JuliaBackend(case.problem, request)
    assert owner.metadata()["propagator_mode"] == (
        "tabulate" if expected else "recompute"
    )
    owner.close()


def test_dense_adams_separates_observation_planes_from_step_landings():
    case = adams_case(steps=None)
    request = replace(
        case.request, observation=replace(case.request.observation, dense=True)
    )
    transfer = lower_run(case.problem, request)
    assert transfer.adams is not None and transfer.adams.interpolate
    assert transfer.adams.save_planes == ()
    np.testing.assert_array_equal(transfer.planes, request.observation.save_planes)
    assert not transfer.planes.flags.writeable


def test_dense_runge_kutta_copies_only_the_shared_extension_weights():
    case = cpu_case(steps=None)
    request = replace(
        case.request, observation=replace(case.request.observation, dense=True)
    )
    transfer = lower_run(case.problem, request)
    shared = continuous_extension(DOPRI5)
    assert transfer.dense_weights is not None and shared is not None
    np.testing.assert_array_equal(transfer.dense_weights, shared)
    assert not transfer.dense_weights.flags.writeable
    assert not np.shares_memory(transfer.dense_weights, shared)
    assert lower_run(case.problem, case.request).dense_weights is None


@pytest.mark.parametrize("order", [1, 5, 6])
@pytest.mark.parametrize("interval_cap", [False, True])
def test_adams_lowering_uses_shared_policy_and_preserves_replay(order, interval_cap):
    method = Multistep(order=order, interval_cap=interval_cap, max_calls=13)
    case = adams_case(method=method, dz_max=1e-5)
    transfer = lower_run(case.problem, case.request)
    control, observation = case.request.error_control, case.request.observation
    expected = march_policy(
        method,
        dz_min=control.dz_min,
        dz_max=control.dz_max,
        stretch=control.stretch,
        z_end=float(observation.save_planes[-1]),
        save_planes=observation.save_planes.tolist(),
        hard_kinks=case.problem.linear.hard_kinks.tolist(),
        breakpoints=case.problem.linear.phi_breakpoints.tolist(),
        interpolate=False,
    )
    # Scalar control is an exact policy-transfer contract, not a numerical comparison.
    assert transfer.adams == expected
    assert not transfer.source_absent
    assert np.all(transfer.steps > control.dz_max)
    assert case.request.replay is not None
    assert np.array_equal(transfer.steps, case.request.replay.schedule.steps)
    assert not transfer.steps.flags.writeable


@pytest.mark.parametrize(
    ("method", "refusal"),
    [
        (Multistep(formulation="interaction"), "interaction"),
        (Multistep(carrier_frame="scalar"), "carrier frame"),
        (ComparisonMultistep(), "comparison"),
    ],
)
def test_unadmitted_adams_variants_refuse_before_initialization(
    method, refusal, monkeypatch
):
    case = adams_case(method=method)

    def no_runtime(**_):
        pytest.fail("method admission must precede Julia initialization")

    monkeypatch.setattr(backend, "get_runtime", no_runtime)
    with pytest.raises(BackendError, match=refusal):
        JuliaBackend(case.problem, case.request)


@pytest.mark.parametrize("source", ["absent", "zero_coefficient", "absorber"])
def test_adams_source_absence_is_structural_and_includes_the_collar(source):
    case = adams_case()
    responses = case.problem.responses
    if source == "zero_coefficient":
        assert responses.kerr_coefficients is not None
        responses = replace(
            responses,
            kerr_coefficients=replace(responses.kerr_coefficients, coefficient=0.0),
        )
    else:
        responses = replace(
            responses,
            response_order=(ResponseStage.ACCUMULATE,),
            kerr_coefficients=None,
        )
    if source == "absorber":
        coarse = np.ones(case.problem.grids.n_t)
        fine = np.ones(case.problem.grids.n_to)
        coarse.setflags(write=False)
        fine.setflags(write=False)
        responses = replace(
            responses,
            temporal_absorber=coarse,
            temporal_absorber_fine=fine,
            absorber_scheme="nonlinear",
        )
    transfer = lower_run(replace(case.problem, responses=responses), case.request)
    assert transfer.source_absent is (source == "absent")


def test_adams_rejects_disagreeing_resolved_interval_topology():
    case = adams_case(method=Multistep(interval_cap=True))
    request = replace(
        case.request,
        error_control=replace(case.request.error_control, interval_cap=False),
    )
    with pytest.raises(BackendError, match="interval-cap policy disagree"):
        lower_run(case.problem, request)


def test_low_storage_lowers_the_recorded_steps_without_an_adaptive_policy():
    case = low_storage_case()
    transfer = lower_run(case.problem, case.request)
    assert transfer.low_storage and transfer.adams is None
    assert case.request.replay is not None
    assert np.array_equal(transfer.steps, case.request.replay.schedule.steps)
    assert np.array_equal(transfer.edges, case.request.replay.schedule.edges)
    assert np.all(transfer.steps <= transfer.control.dz_max)
    assert not transfer.steps.flags.writeable


def test_low_storage_refuses_an_adaptive_request_before_runtime_entry(monkeypatch):
    case = low_storage_case(steps=None)

    def no_runtime(**_):
        pytest.fail("a fixed itinerary must be required before initialization")

    monkeypatch.setattr(backend, "get_runtime", no_runtime)
    with pytest.raises(BackendError, match="requires a fixed replay schedule"):
        JuliaBackend(case.problem, case.request)


def test_binary32_lowering_stays_lazy_and_preserves_wide_coefficients(monkeypatch):
    case = cpu_case()

    def no_runtime(**_):
        pytest.fail("lowering must not initialize Julia")

    monkeypatch.setattr(backend, "get_runtime", no_runtime)
    problem = replace(case.problem, precision=SINGLE)
    owner = JuliaBackend(problem, case.request)
    transfer = lower_modal(problem)
    assert transfer.arithmetic.name == "single"
    assert transfer.arithmetic.state_exponent == transfer.arithmetic.field_exponent
    assert transfer.synthesis.dtype == np.float64
    assert transfer.operator.values is not None
    assert transfer.operator.values.dtype == np.complex128
    assert not transfer.operator.values.flags.writeable
    assert not np.shares_memory(transfer.synthesis, problem.transform.synthesis)
    owner.close()


def test_df32_method_admission_is_lazy_and_retains_independent_operator_storage(
    monkeypatch,
):
    case = cpu_case()

    def no_runtime(**_):
        pytest.fail("DF32 method admission must not initialize Julia")

    monkeypatch.setattr(backend, "get_runtime", no_runtime)
    problem = replace(case.problem, precision=SINGLE)
    owner = JuliaBackend(
        problem,
        case.request,
        options=ExecutionOptions(backend_options={"wide": "df32"}),
    )
    assert owner.metadata()["wide"] == "df32"
    assert owner._df32_operator is not None
    source = lower_modal(problem).operator.values
    bound = owner._df32_operator.data.values
    assert source is not None and bound is not None
    assert not np.shares_memory(source, bound) and not bound.flags.writeable
    assert owner.memory_ledger()["wide.transfer.data.values"] == bound.nbytes
    with pytest.raises(BackendError, match="DF32 timing probes"):
        owner.timing_probe(Start.entrance(problem))
    owner.close()


def test_binary32_input_noise_refuses_before_coefficient_lowering(monkeypatch):
    case = cpu_case()
    problem = replace(
        case.problem,
        precision=SINGLE,
        initial=replace(
            case.problem.initial,
            noise_model="input",
            noise0=np.zeros(case.problem.shape, dtype=np.complex128),
        ),
        flags=replace(case.problem.flags, has_noise=True),
    )

    def no_runtime(**_):
        pytest.fail("binary32 noise admission must precede Julia initialization")

    monkeypatch.setattr(backend, "get_runtime", no_runtime)
    with pytest.raises(BackendError, match="binary32 input noise"):
        JuliaBackend(problem, case.request)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"unknown": True}, "unsupported|does not support"),
        ({"wide": 1}, "wide"),
        ({"wide": "double"}, "wide"),
        ({"setup_id": 4}, "setup_id"),
        ({"setup_id": ""}, "setup_id"),
        ({"wide": "df32"}, "binary32"),
    ],
)
def test_execution_options_refuse_before_preparation(options, message):
    case = cpu_case()
    with pytest.raises(BackendError, match=message):
        JuliaBackend(
            case.problem,
            case.request,
            options=ExecutionOptions(backend_options=options),
        )


def test_execution_owner_copies_options_without_loading_julia():
    case = cpu_case()
    options = ExecutionOptions(backend_options={"wide": "binary64", "setup_id": "one"})
    owner = JuliaBackend(case.problem, case.request, options=options)
    owned_options = options.backend_options
    assert isinstance(owned_options, dict)
    owned_options["setup_id"] = "two"
    assert owner.metadata()["backend_options"] == {
        "wide": "binary64",
        "setup_id": "one",
    }
    assert "julia_threads" not in owner.metadata()
    assert (
        owner.memory_ledger()["transfer.operator.values"]
        == case.problem.initial.field0.nbytes
    )
    owner.close()
    assert owner.closed


def test_foreign_numerical_request_refuses_before_runtime_selection(monkeypatch):
    case = cpu_case()
    owner = JuliaBackend(case.problem, case.request)

    def no_runtime(**_):
        pytest.fail("a foreign request must refuse before Julia initialization")

    monkeypatch.setattr(backend, "get_runtime", no_runtime)
    foreign = replace(
        case.request, error_control=replace(case.request.error_control, dz_max=1e-4)
    )
    with pytest.raises(BackendError, match="error control"):
        owner.prepare(case.problem, foreign)
    owner.close()


def test_benchmark_start_proxy_accepts_an_execution_only_owner_without_loading():
    case = cpu_case()
    owner = JuliaBackend(case.problem, case.request)
    try:
        timing = _Started(owner)
        assert timing.engine.request is owner.request
        assert timing.engine.metadata() == owner.metadata()
        assert timing.started_at is None
        assert owner._runtime is None
    finally:
        owner.close()


def test_delivery_memory_omits_views_of_arrays_already_counted_by_members():
    shared = np.arange(12, dtype=np.float64).reshape(3, 4)
    private = shared.copy()
    buffers = {"borrowed": shared[:, ::2], "owned": private, "repeated": private.T}
    ledger = transfer_memory(buffers, exclude=(shared.T,))
    assert sum(ledger.values()) == private.nbytes
    assert len(ledger) == 1 and "owned" in next(iter(ledger))


def _evaluate_table(transfer, z):
    # Independent NumPy Chebyshev evaluation of the transferred coefficient order.
    interval = np.clip(
        np.searchsorted(transfer.edges, z, side="right") - 1, 0, len(transfer.edges) - 2
    )
    lo, hi = transfer.edges[interval : interval + 2]
    coordinate = (2 * z - lo - hi) / (hi - lo)
    return np.polynomial.chebyshev.chebval(
        coordinate, transfer.coefficients[:, :, interval].T
    )


def test_represented_transfer_keeps_axis_order_interior_knots_and_edge_slack():
    coefficients = np.arange(2 * 4 * 2 * 1 * 3, dtype=float).reshape(2, 4, 2, 1, 3)
    table = PiecewiseChebyshev(
        coefficients, np.array([0.0, 0.25, 1.0]), np.array([3, 3]), np.array([0.25])
    )
    transfer = lower_table(table)
    assert transfer.lower == table.packed()[1]
    assert transfer.upper == table.packed()[2]
    assert transfer.coefficients.shape == (6, 4, 2)
    for z in (transfer.lower, 0.0, 0.125, 0.25, 0.75, 1.0, transfer.upper):
        actual = _evaluate_table(transfer, z)
        # Both evaluate the identical four-term polynomial. This is a rounding
        # budget for the short recurrence, scaled by absolute coefficients.
        bound = 32 * np.finfo(float).eps * np.sum(np.abs(coefficients))
        assert bound > 0
        assert np.max(np.abs(actual - table(z).ravel())) <= bound
    coefficients.fill(0)
    assert np.any(transfer.coefficients != 0)


def test_modal_transfer_preserves_expanded_operator_and_prefactor(
    synthetic_expansion, expanded_prefactor, substitute
):
    case = cpu_case()
    n_omega = case.problem.shape[-1]
    terms = [np.full(n_omega, -0.2j), np.linspace(0.1j, 0.4j, n_omega)]
    linear = synthetic_expansion(
        case.problem, terms, lambda z: np.column_stack((z, z * z))
    )
    problem = substitute(
        case.problem,
        run_request=case.request,
        linear=linear,
        responses=expanded_prefactor(case.problem.responses),
    )
    transfer = lower_modal(problem)
    assert transfer.operator.table is None
    assert transfer.operator.weights is not None
    assert transfer.prefactor.weights is not None
    assert transfer.operator.values is not None
    assert transfer.operator.values.shape == (n_omega, 1, 2)
    for z in (0.0, 0.006, 0.02):
        weights = _evaluate_table(transfer.operator.weights, z)
        value = np.einsum("fct,t->fc", transfer.operator.values, weights)
        expected = terms[0] * z + terms[1] * z * z
        # Two fixed terms and a short weight recurrence; the represented polynomial
        # itself is exact here. A fixed absolute budget includes the zero entrance.
        bound = 64 * np.finfo(float).eps * max(abs(expected).max(), 1.0)
        assert abs(value[:, 0] - expected).max() <= bound


def test_full_modal_tables_are_lowered_without_sampling_variation(tabulated_problem):
    problem, _ = tabulated_problem
    transfer = lower_modal(problem)
    assert isinstance(transfer.operator.table, TableTransfer)
    assert isinstance(transfer.prefactor.table, TableTransfer)
    assert transfer.operator.values is None
    assert transfer.prefactor.values is None
    assert transfer.temporal_rate is not None
    assert not np.shares_memory(
        transfer.temporal_rate, problem.responses.temporal_absorber_fine
    )
    assert transfer.kind == "field"
    assert transfer.n_time == problem.grids.n_to
    assert transfer.dt == problem.grids.dt_fine
    assert transfer.taper == 1.0


def test_explicit_node_chunk_is_owned_without_runtime_entry():
    case = cpu_case()
    owner = JuliaBackend(
        case.problem, case.request, options=ExecutionOptions(node_chunk=3)
    )
    try:
        assert owner.metadata()["node_chunk"] == 1
    finally:
        owner.close()
