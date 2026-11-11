# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep whole-expression admission inputs bound to the actual start."""

import inspect
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from jaxtyping import Complex

from solprop.backends.julia._descriptors import (
    CoefficientTransfer,
    TableTransfer,
    transfer_memory,
)
from solprop.backends.julia._lower import RunTransfer
from solprop.backends.julia._precision import ArithmeticTransfer
from solprop.backends.julia._wide import (
    DF32OperatorBinding,
    admit_df32_flow,
    bind_df32_start,
)
from solprop.exceptions import BackendError
from solprop.output.schema import MarchSnapshot
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep, march_policy
from solprop.translate.run_request import ErrorControlSpec


def _run(*, replay=False, order=None, nodes=(0.0, 1.0)):
    method = None if order is None else Multistep(order=order)
    policy = (
        None
        if method is None
        else march_policy(
            method,
            dz_min=0.01,
            dz_max=0.01,
            stretch=1.2,
            z_end=1.0,
            save_planes=[1.0],
            hard_kinks=[],
            breakpoints=[0.0, 1.0],
            interpolate=False,
        )
    )
    return RunTransfer(
        ErrorControlSpec(PIController(1e-6, 1e-9), 0.01, 0.01, 0.01, 1.2, False, True),
        np.array(nodes),
        np.zeros((len(nodes), len(nodes))),
        np.ones(len(nodes)),
        np.zeros(len(nodes)),
        np.ones(1, dtype=bool),
        np.ones(1, dtype=bool),
        np.ones(1),
        np.array([1.0]),
        np.array([0.0, 1.0]),
        np.empty(0),
        np.array([1.0]) if replay else np.empty(0),
        np.array([0.0, 1.0]) if replay else np.empty(0),
        0,
        2.0,
        adams=policy,
        history_key=None if method is None else method.history_key,
        arithmetic=ArithmeticTransfer("single", 8, 8),
    )


def _operator(coefficients=None, *, expansion=False):
    if coefficients is None:
        return CoefficientTransfer(np.array([[1j]]), None, None)
    values = np.asarray(coefficients)
    if values.ndim == 1:
        values = values[None, :, None]
    table = TableTransfer(np.array([0.0, 1.0]), 0.0, 1.0 + 1e-12, values)
    return (
        CoefficientTransfer(np.array([[[1j]]]), None, table)
        if expansion
        else CoefficientTransfer(None, table, None)
    )


def _history(run, *, compensated=False):
    assert run.adams is not None
    store = np.zeros((run.adams.order + 2, 1, 1, 1, 1), dtype=np.complex64)
    rows: dict[str, Complex[np.ndarray, "n_slots 1 n_modes n_pol n_omega"]] = {
        "store": store
    }
    if compensated:
        rows.update(states=store.copy(), rates=store.astype(np.complex128))
        rows["rates"][0] = 100 + 200j
    return MarchSnapshot(
        z=0.5,
        state=np.ones((1, 1, 1), dtype=np.complex128),
        proposal=0.01,
        method="M()",
        order=1,
        rejections=0,
        last_rejected=False,
        positions=np.array([0.0, 0.5]),
        identities=("accepted", "predicted"),
        slots=np.array([0, 1], dtype=np.int64),
        rows=rows,
        history_key=run.history_key,
    )


@pytest.mark.parametrize("kind", ["constant", "table", "expansion"])
def test_operator_binding_owns_even_readonly_aliases_and_shares_only_its_copy(kind):
    origin = np.array([[1j, 2j, 3j]])
    readonly = origin[:, 1:2]
    readonly.setflags(write=False)
    if kind == "constant":
        operator = CoefficientTransfer(readonly, None, None)
    elif kind == "table":
        operator = _operator(readonly[:, :, None])
    else:
        operator = _operator(np.real(readonly)[:, :, None], expansion=True)
    prepared = DF32OperatorBinding(operator)
    first = bind_df32_start(prepared, _run(replay=True), 0.0)
    second = bind_df32_start(prepared, _run(replay=True), 0.0)
    before = first.operands
    origin[:] = 2.0**21 * 1j
    assert first.operator is second.operator is prepared
    assert first.operands == before
    assert not first.run.nodes.flags.writeable
    assert not np.shares_memory(first.run.nodes, second.run.nodes)
    arrays = transfer_memory(first)
    assert sum(arrays.values()) > 0
    assert not any(
        path.startswith("operator.")
        for path in transfer_memory(first, exclude=prepared)
    )
    retained = prepared.data.values
    if kind == "table":
        assert prepared.data.table is not None
        retained = prepared.data.table.coefficients
    assert (
        retained is not None and retained.flags.owndata and not retained.flags.writeable
    )
    assert not np.shares_memory(retained, origin)


def test_actual_replay_ceiling_and_each_start_arrays_are_bound_before_origin_mutation():
    run = _run(replay=True)
    weights = np.array([2.0])
    run = replace(
        run,
        control=replace(
            run.control, controller=replace(run.control.controller, weights=weights)
        ),
        dense_weights=np.array([0.25, 0.5]),
    )
    operator = CoefficientTransfer(np.array([[2.0**21 * 1j]]), None, None)
    binding = bind_df32_start(operator, run, 0.0)
    assert run.control.dz_max * 2.0**21 < 2.0**20
    assert binding.operands.spans.step_max == 1.0
    assert binding.operands.step_argument.imag > 2.0**20
    run.steps[:] = 0.01
    run.nodes[:] = 99
    weights[:] = 99
    assert run.dense_weights is not None
    run.dense_weights[:] = 99
    np.testing.assert_array_equal(binding.run.steps, [1.0])
    np.testing.assert_array_equal(binding.run.nodes, [0.0, 1.0])
    np.testing.assert_array_equal(binding.run.control.controller.weights, [2.0])
    np.testing.assert_array_equal(binding.run.dense_weights, [0.25, 0.5])
    assert binding.run.arithmetic == ArithmeticTransfer("single", 8, 8)


def test_represented_interior_extremum_is_not_replaced_by_endpoint_sampling():
    # Exact Chebyshev w=1-T2 is zero at both endpoints and two at the midpoint.
    operator = _operator([1.0, 0.0, -1.0], expansion=True)
    assert operator.values is not None
    operator.values[:] *= 2.0**20
    binding = bind_df32_start(operator, _run(replay=True), 0.0)
    assert binding.operands.operator.cumulative is not None
    assert binding.operands.operator.cumulative.imag >= 2.0**21
    assert binding.operands.step_argument.imag >= 2.0**21


@pytest.mark.parametrize("compensated", [False, True])
@pytest.mark.parametrize("origin", [None, "binary64", "df32"])
def test_plain_and_compensated_histories_keep_old_age_maximum_order_and_actual_rates(
    compensated, origin
):
    run = _run(order=6)
    history = _history(run, compensated=compensated)
    operator = _operator([0.0j, 0.5j]) if compensated else _operator()
    binding = bind_df32_start(operator, run, 0.5, history=history, origin_wide=origin)
    assert binding.history is not None
    assert binding.history.order == 1 and run.adams is not None and run.adams.order == 6
    assert binding.operands.spans.transport_span >= 0.5 + 7 * 0.01
    assert binding.origin_wide == origin
    history.positions[:] = [0.4, 0.5]
    history.state[:] = 9
    history.rows["store"][:] = 8
    np.testing.assert_array_equal(binding.history.positions, [0.0, 0.5])
    np.testing.assert_array_equal(binding.history.state, np.ones((1, 1, 1)))
    assert not np.any(binding.history.rows["store"])
    assert not binding.history.rows["store"].flags.writeable
    assert not hasattr(binding.history.rows, "__setitem__")
    if compensated:
        assert binding.operands.restored_rates is not None
        history.rows["rates"][0] = 0
        assert binding.history.rows["rates"][0, 0, 0, 0, 0] == 100 + 200j
        assert binding.operands.restored_rates.imag >= 200
    else:
        assert binding.operands.restored_rates is None


def test_adaptive_jump_has_no_uniform_positive_minimum_step_certificate():
    coefficients = np.array([[[0.0j, 1.0e-8j]]])
    table = TableTransfer(np.array([0.0, 0.5, 1.0]), 0.0, 1.0 + 1e-12, coefficients)
    binding = bind_df32_start(
        CoefficientTransfer(None, table, None), _run(order=6), 0.0
    )
    assert binding.operands.operator.rate.imag == 0
    assert binding.operands.operator.jumps.imag >= 1e-8
    assert binding.operands.transported_argument is None
    h = Fraction(1, 2**54)
    assert (h + Fraction(1, 4)) * Fraction.from_float(1e-8) / h > 2**20


def test_completed_replay_preserves_zero_future_work():
    binding = bind_df32_start(_operator(), _run(replay=True), 1.0)
    assert binding.operands.spans.start == binding.operands.spans.end == 1.0
    assert binding.operands.spans.step_max == 0.0
    assert binding.operands.spans.step_min is None


def test_new_binding_cannot_reuse_a_summary_from_changed_caller_coefficients():
    operator = _operator()
    first = bind_df32_start(operator, _run(replay=True), 0.0)
    assert operator.values is not None
    operator.values[:] = 2.0**21 * 1j
    second = bind_df32_start(operator, _run(replay=True), 0.0)
    assert (
        first.operands.step_argument.imag < 2.0**20 < second.operands.step_argument.imag
    )
    assert first.operator.data.values is not second.operator.data.values


def test_binding_refuses_foreign_precision_and_does_not_silently_discard_history():
    run = _run(order=3)
    with pytest.raises(BackendError, match="binary32 scientific"):
        bind_df32_start(_operator(), replace(run, arithmetic=ArithmeticTransfer()), 0.0)
    with pytest.raises(BackendError, match="fresh"):
        bind_df32_start(_operator(), run, 0.0, origin_wide="binary64")
    # Exercise the record validator beneath the optional import-hook Literal check.
    with pytest.raises(BackendError, match="origin"):
        inspect.unwrap(bind_df32_start)(_operator(), run, 0.0, origin_wide="exact")
    history = _history(run)
    bad = replace(history, rows={"store": history.rows["store"].astype(np.complex128)})
    with pytest.raises(BackendError, match="raw binary32 history"):
        bind_df32_start(_operator(), run, 0.5, history=bad)
    with pytest.raises(BackendError, match="already-compatible"):
        bind_df32_start(_operator(), _run(order=2), 0.5, history=history)
    np.testing.assert_array_equal(history.positions, [0.0, 0.5])


def test_df32_execution_admits_only_qualified_landed_method_policies():
    run = _run(replay=True)
    assert admit_df32_flow(run) is None
    assert admit_df32_flow(replace(run, low_storage=True)) is None
    adams = _run(order=3)
    assert adams.adams is not None
    assert admit_df32_flow(adams) is None
    for refused, name in (
        (replace(run, arithmetic=ArithmeticTransfer()), "binary32"),
        (
            replace(adams, adams=replace(adams.adams, interpolate=True)),
            "dense observations",
        ),
        (replace(run, dense_weights=np.ones(2)), "dense observations"),
        (replace(run, tabulate_flow=True), "tabulated flow"),
    ):
        with pytest.raises(BackendError, match=name):
            admit_df32_flow(refused)
