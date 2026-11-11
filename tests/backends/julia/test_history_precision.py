# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preserve portable history dtypes and refuse cross-precision rows before copying."""

from dataclasses import replace

import numpy as np
import pytest

from backends.julia.test_history import _snapshot
from solprop.backends.julia._history import HistoryBuffers, _lower_history
from solprop.backends.julia._precision import ArithmeticTransfer


@pytest.mark.parametrize(
    "precision,dtype", [("double", np.complex128), ("single", np.complex64)]
)
@pytest.mark.parametrize("compensated", [False, True])
def test_portable_rows_keep_raw_state_precision_and_wide_rates(
    precision, dtype, compensated
):
    case, transfer, snapshot = _snapshot()
    transfer = replace(transfer, arithmetic=ArithmeticTransfer(precision, 3, 3))
    rows = {"store": snapshot.rows["store"].astype(dtype)}
    if compensated:
        rows.update(states=rows["store"].copy(), rates=snapshot.rows["store"].copy())
    raw = np.array(snapshot.state, dtype=dtype).astype(np.complex128)
    candidate = replace(snapshot, state=raw, rows=rows)
    buffers = _lower_history(candidate, transfer, case.problem.shape)
    assert isinstance(buffers, HistoryBuffers)
    assert buffers.state.dtype == np.complex128
    assert buffers.state.tobytes() == raw.tobytes()
    assert buffers.store.dtype == dtype
    assert buffers.store.tobytes() == rows["store"].tobytes()
    assert not np.shares_memory(buffers.state, raw)
    assert not np.shares_memory(buffers.store, rows["store"])
    if compensated:
        assert buffers.states is not None and buffers.rates is not None
        assert buffers.states.dtype == dtype
        assert buffers.rates.dtype == np.complex128
        assert buffers.rates.tobytes() == rows["rates"].tobytes()


@pytest.mark.parametrize(
    "precision,wrong", [("double", np.complex64), ("single", np.complex128)]
)
def test_cross_precision_source_rows_refuse_without_conversion(precision, wrong):
    case, transfer, snapshot = _snapshot()
    transfer = replace(transfer, arithmetic=ArithmeticTransfer(precision))
    candidate = replace(snapshot, rows={"store": snapshot.rows["store"].astype(wrong)})
    before = candidate.rows["store"].tobytes()
    reason = _lower_history(candidate, transfer, case.problem.shape)
    assert isinstance(reason, str) and "history" in reason
    assert candidate.rows["store"].tobytes() == before


def test_binary32_history_refuses_narrow_rates():
    case, transfer, snapshot = _snapshot()
    transfer = replace(transfer, arithmetic=ArithmeticTransfer("single"))
    rows = {
        name: snapshot.rows["store"].astype(np.complex64)
        for name in ("store", "states", "rates")
    }
    reason = _lower_history(replace(snapshot, rows=rows), transfer, case.problem.shape)
    assert isinstance(reason, str) and "rates" in reason and "binary64" in reason
