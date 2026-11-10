# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep descriptor lowering owned, lazy, cached and explicit about admission."""

import gc
import weakref
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest
from benchmarks.records import SuiteError
from benchmarks.worker import _Loaded

from backends.julia.cases import adams_case, cpu_case
from solprop.backends.julia import backend
from solprop.backends.julia._descriptors import (
    TableTransfer,
    lower_modal,
    lower_table,
    transfer_memory,
)
from solprop.backends.julia._lower import lower_run
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.session import ExecutionOptions
from solprop.exceptions import BackendError
from solprop.stepper.multistep import ComparisonMultistep, Multistep, march_policy
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.precision import SINGLE
from solprop.translate.responses import ResponseStage


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


def test_binary32_refuses_before_a_runtime_is_selected(monkeypatch):
    case = cpu_case()

    def no_runtime(**_):
        pytest.fail("admission must precede Julia initialization")

    monkeypatch.setattr(backend, "get_runtime", no_runtime)
    with pytest.raises(BackendError, match="binary32"):
        JuliaBackend(replace(case.problem, precision=SINGLE), case.request)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"unknown": True}, "unsupported|does not support"),
        ({"wide": 1}, "wide"),
        ({"wide": "double"}, "wide"),
        ({"setup_id": 4}, "setup_id"),
        ({"setup_id": ""}, "setup_id"),
        ({"wide": "df32"}, "binary64"),
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


def test_legacy_benchmark_window_refuses_an_execution_only_owner():
    case = cpu_case()
    owner = JuliaBackend(case.problem, case.request)
    try:
        with pytest.raises(SuiteError, match="driver load boundary"):
            _Loaded(owner)
    finally:
        owner.close()


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
