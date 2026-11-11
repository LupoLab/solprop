# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check portable history admission and compact cumulative restart ledgers."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from backends.julia.cases import adams_case, cpu_case, low_storage_case
from solprop.backends.julia._descriptors import lower_modal
from solprop.backends.julia._history import (
    CALL_COUNTERS,
    HistoryBuffers,
    _lower_history,
    lower_call_restore,
    preflight_history,
)
from solprop.backends.julia._lower import lower_run
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.julia._session import JuliaSession
from solprop.backends.julia._wide import bind_df32_start
from solprop.backends.multistep import CallLedger
from solprop.backends.session import Start
from solprop.exceptions import BackendError
from solprop.output.schema import MarchSnapshot
from solprop.translate.precision import SINGLE


def _snapshot():
    case = adams_case()
    transfer = lower_run(case.problem, case.request)
    assert transfer.adams is not None
    state = np.array(case.problem.initial.field0, dtype=np.complex128, copy=True)
    # Signed zeros belong to the raw portable representation, even in inactive bins.
    state.real.flat[0] = -0.0
    shape = (transfer.adams.order + 2, 1, *case.problem.shape)
    rows = {"store": np.full(shape, 0.125 - 0.25j, dtype=np.complex128)}
    snapshot = MarchSnapshot(
        z=0.01,
        state=state,
        proposal=0.0025,
        method=transfer.method,
        order=3,
        rejections=0,
        last_rejected=False,
        positions=np.array([0.005, 0.0075, 0.01]),
        identities=("accepted", "predicted", "predicted"),
        slots=np.array([2, 0, 1], dtype=np.int64),
        rows=rows,
        history_key=transfer.history_key,
    )
    return case, transfer, snapshot


def test_restart_ledger_matches_shared_schema_without_expanding_orders():
    saved = CallLedger(startup=1, predictor=4, accepted_steps=4, orders=[1, 2, 2, 3])
    record = saved.to_record()
    restored = lower_call_restore({"logical": record}, adams=True)
    assert restored is not None
    assert restored.counts == tuple(getattr(saved, name) for name in CALL_COUNTERS)
    assert restored.orders.tolist() == record["orders"]
    record["orders"] = [[2, 10**12]]
    record["accepted_steps"] = 10**12
    huge = lower_call_restore({"logical": record}, adams=True)
    assert huge is not None
    assert huge.orders.tolist() == [[2, 10**12]]
    assert huge.orders.nbytes == 16
    # Later mutation of a saved record cannot change a session's cumulative orders.
    record["orders"][0][1] = 1
    assert restored.orders.tolist() == [[1, 1], [2, 2], [3, 1]]
    assert huge.orders.tolist() == [[2, 10**12]]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("startup", True),
        ("startup", -1),
        ("startup", 1.5),
        ("startup", 2**63),
        ("orders", [1, 2]),
        ("orders", [[1, 0]]),
        ("orders", [[True, 1]]),
        ("orders", [[1, 2]]),
        ("orders", "invalid"),
    ],
)
def test_restart_ledger_refuses_malformed_counters_and_order_runs(name, value):
    record = CallLedger(accepted_steps=1, orders=[1]).to_record()
    record[name] = value
    with pytest.raises(BackendError, match="logical call"):
        lower_call_restore({"logical": record}, adams=True)


def test_ledger_schema_atomic_refusal_and_charge_overflow_are_explicit():
    for record in ({"unexpected": 1}, [1]):
        with pytest.raises(BackendError, match="schema"):
            lower_call_restore({"logical": record}, adams=True)
    with pytest.raises(BackendError, match="atomic"):
        lower_call_restore({"atomic": {"propagations": 1}}, adams=False)
    record = CallLedger(startup=2**63 - 1, predictor=1).to_record()
    with pytest.raises(BackendError, match="total exceeds"):
        lower_call_restore({"logical": record}, adams=True)
    assert lower_call_restore(None, adams=True) is None
    assert lower_call_restore({"logical": {}}, adams=True) is None
    assert lower_call_restore({"logical": record}, adams=False) is None


def test_history_import_owns_disjoint_contiguous_lossless_buffers():
    case, transfer, snapshot = _snapshot()
    snapshot.state.flags.writeable = False
    snapshot = replace(
        snapshot, rows={"store": np.asfortranarray(snapshot.rows["store"])}
    )
    result = _lower_history(snapshot, transfer, case.problem.shape)
    assert isinstance(result, HistoryBuffers)
    for actual, expected in (
        (result.state, snapshot.state),
        (result.positions, snapshot.positions),
        (result.slots, snapshot.slots),
        (result.store, snapshot.rows["store"]),
    ):
        assert actual.flags.c_contiguous and actual.flags.writeable
        assert not np.shares_memory(actual, expected)
        assert actual.tobytes() == expected.tobytes()
    assert result.identities.tolist() == [0, 1, 1]
    assert result.states is result.rates is None
    result.store.fill(0)
    assert np.all(snapshot.rows["store"] == 0.125 - 0.25j)


@pytest.mark.parametrize("method", [cpu_case, low_storage_case])
def test_field_only_methods_report_an_explicit_history_reset(method):
    case, _, snapshot = _snapshot()
    destination = method()
    transfer = lower_run(destination.problem, destination.request)
    reason = _lower_history(snapshot, transfer, case.problem.shape)
    assert isinstance(reason, str) and "keeps no history" in reason


@pytest.mark.parametrize("key", [None, "foreign-history"])
def test_history_key_mismatch_preserves_raw_field_restart(key):
    case, transfer, snapshot = _snapshot()
    reason = _lower_history(
        replace(snapshot, history_key=key), transfer, case.problem.shape
    )
    assert isinstance(reason, str) and "history" in reason


@pytest.mark.parametrize("name", ["store", "states", "rates"])
def test_single_precision_stores_refuse_without_narrowing_or_upcasting(name):
    case, transfer, snapshot = _snapshot()
    rows = {key: snapshot.rows["store"].copy() for key in ("store", "states", "rates")}
    rows[name] = rows[name].astype(np.complex64)
    reason = _lower_history(replace(snapshot, rows=rows), transfer, case.problem.shape)
    assert isinstance(reason, str) and "binary64 history" in reason


@pytest.mark.parametrize("name", ["applied", "weights_0", "unknown"])
def test_foreign_store_names_refuse_before_any_partial_restore(name):
    case, transfer, snapshot = _snapshot()
    rows = {**snapshot.rows, name: snapshot.rows["store"].copy()}
    reason = _lower_history(replace(snapshot, rows=rows), transfer, case.problem.shape)
    assert isinstance(reason, str) and name in reason


@pytest.mark.parametrize(
    "refusal", [None, "the snapshot newest sample is not at its accepted position"]
)
def test_history_preflight_owns_the_exact_rows_and_retains_reset_reason(
    setup_record, refusal
):
    case, transfer, snapshot = _snapshot()
    seen = []

    def check(preparation, run, z, selected, buffers):
        assert preparation == "prepared" and run is transfer and z == snapshot.z
        assert selected is not snapshot and selected.state is buffers.state
        assert selected.rows["store"] is buffers.store
        assert selected.positions is buffers.positions
        seen.append(selected)
        return refusal

    runtime = JuliaRuntime(
        setup_record, 1, None, SimpleNamespace(preflight_adams_history=check)
    )
    preparation = runtime.owners.retain("prepared")
    start = Start(field=snapshot.state.copy(), z=snapshot.z, march=snapshot)
    try:
        selected = preflight_history(
            runtime, preparation, start, transfer, case.problem.shape
        )
        assert selected.refusal == refusal
        assert len(seen) == 1 and start.field is not None
        if refusal is None:
            assert selected.snapshot is seen[0]
            assert selected.snapshot.identities == snapshot.identities
            expected = snapshot.rows["store"].copy()
            snapshot.rows["store"].fill(9)
            np.testing.assert_array_equal(selected.snapshot.rows["store"], expected)
        else:
            assert selected.snapshot is None
    finally:
        preparation.close()


def test_history_preflight_skips_absent_or_structurally_incompatible_rows(setup_record):
    case, transfer, snapshot = _snapshot()
    runtime = JuliaRuntime(setup_record, 1, None, SimpleNamespace())
    preparation = runtime.owners.retain("prepared")
    start = Start.entrance(case.problem)
    try:
        selected = preflight_history(
            runtime, preparation, start, transfer, case.problem.shape
        )
        assert selected.snapshot is None and selected.refusal is None
        start.march = replace(snapshot, history_key="foreign")
        selected = preflight_history(
            runtime, preparation, start, transfer, case.problem.shape
        )
        assert selected.snapshot is None and "history" in str(selected.refusal)
    finally:
        preparation.close()


def test_history_preflight_bridge_failure_cannot_become_a_field_reset(setup_record):
    case, transfer, snapshot = _snapshot()

    def fail(*_):
        raise RuntimeError("controlled preflight failure")

    runtime = JuliaRuntime(
        setup_record, 1, None, SimpleNamespace(preflight_adams_history=fail)
    )
    preparation = runtime.owners.retain("prepared")
    start = Start(field=snapshot.state.copy(), z=snapshot.z, march=snapshot)
    try:
        with pytest.raises(BackendError, match="controlled preflight failure"):
            preflight_history(runtime, preparation, start, transfer, case.problem.shape)
        assert start.field is not None
    finally:
        preparation.close()


def test_df32_unselected_history_refuses_before_constructing_an_engine(setup_record):
    case, _, snapshot = _snapshot()
    problem = replace(case.problem, precision=SINGLE)
    transfer = lower_run(problem, case.request)
    wide = bind_df32_start(lower_modal(problem).operator, transfer, snapshot.z)
    runtime = JuliaRuntime(setup_record, 1, None, SimpleNamespace())
    preparation = runtime.owners.retain("prepared")
    field = snapshot.state.copy()
    start = Start(field=field, z=snapshot.z, march=snapshot)
    try:
        with pytest.raises(BackendError, match="history was not selected"):
            JuliaSession(
                runtime,
                preparation,
                wide.run,
                start,
                problem.shape,
                record={},
                released=lambda _: None,
                wide=wide,
            )
        assert start.field is field
        assert preparation.borrow() == "prepared"
    finally:
        preparation.close()
