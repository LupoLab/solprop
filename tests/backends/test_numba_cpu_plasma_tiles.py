# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the tiled plasma traversal against the ordered column, inside its budget.

The tiled traversal performs each node's arithmetic in the column's order, but it is
a different compiled kernel, which may contract and vectorise differently (SEP 25): it
is held to the complete column's derived envelope (``prototypes/plasma_bounds.py``) and
its fixed relative cap, as the blocked column is (``test_numba_cpu_plasma_scan.py``).
"""

import importlib
import math
from pathlib import Path

import numpy as np
import pytest
from numba import config, get_num_threads, set_num_threads

from plasma_prototypes import (
    PlasmaScalars,
    historical_column,
    inputs,
    make_blocked_plasma,
    make_tiled_plasma,
    reference_arguments,
)
from solprop.backends.bodies import RateArrays
from solprop.backends.numba_cpu.kernels import ColumnSpec
from solprop.exceptions import BackendError


@pytest.fixture
def budgets(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "prototypes"))
    return importlib.import_module("plasma_bounds")


def column_bounds(budgets, spec, field, table, scalars, window):
    """Return the column's absolute envelope, fraction envelope and relative cap."""
    work = budgets.allocate_work(field.shape)
    make_blocked_plasma(spec, field.shape[-1]).rates(field, table, work)
    bound, summary = budgets.column_budget(
        field, work.rates, window, scalars, raman=spec.raman
    )
    return bound, summary["complete_chain_relative_cap"]


def within(actual, expected, absolute_bound, cap):
    """Assert the absolute envelope elementwise and the relative cap normwise."""
    assert np.all(np.isfinite(actual))
    error = np.abs(actual - expected)
    assert np.all(error <= absolute_bound)
    scale = float(np.max(np.abs(expected)))
    assert float(np.max(error)) <= cap * scale
    return float(np.max(error))


def assert_column(outputs, expected, bound, cap, *, stash=True):
    """Hold the polarisation, the Raman drive and the stash to the column's budget.

    The drive is one product of each sample, rounded once on either side, so it is
    exact; the stash's fraction has its own envelope, and its largest rate, a maximum
    of rates evaluated apart, the chain's relative cap. A column without the stash
    never writes it.
    """
    largest = within(outputs[0], expected[0], bound["column_bound"], cap)
    np.testing.assert_array_equal(
        outputs[1].view(np.uint64), expected[1].view(np.uint64)
    )
    if not stash:
        np.testing.assert_array_equal(outputs[2], expected[2])
        return largest
    within(outputs[2][:, 0], expected[2][:, 0], bound["fraction_bound"], cap)
    within(outputs[2][:, 1], expected[2][:, 1], np.inf, cap)
    return largest


def packed_run(kernel, args, scalars, capacity):
    count, samples = args[0].shape
    packed = np.full((samples, capacity), np.nan)
    output = np.full_like(packed, np.nan)
    history = np.full((8, (capacity + 7) // 8, 8), np.nan)
    packed[:, :count] = args[0].T
    kernel(packed, output, args[2], args[3], args[4], scalars, args[-1], history, count)
    assert np.all(np.isfinite(output[:, :count]))
    assert np.all(np.isnan(packed[:, count:]))
    assert np.all(np.isnan(output[:, count:]))
    for node in range(count):
        assert np.all(np.isfinite(history[:, node // 8, node % 8]))
    for node in range(count, history.shape[1] * 8):
        assert np.all(np.isnan(history[:, node // 8, node % 8]))
    return output[:, :count].T.copy()


@pytest.mark.parametrize(
    "geometry",
    [(1, 1, 1), (7, 8, 128), (8, 8, 128), (9, 16, 128), (21, 24, 128), (33, 40, 128)],
)
@pytest.mark.parametrize("preionisation", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("response", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("parallel", [False, True])
def test_tiled_column_agrees_with_the_ordered_column(
    budgets, geometry, preionisation, response, parallel, record_property
):
    count, capacity, samples = geometry
    raman, stash = response
    field, table, scalars, window = inputs(samples, count, preionisation)
    unchanged = field.copy()
    spec = ColumnSpec("field_real", True, raman, raman, False, 1, parallel, stash)
    args = reference_arguments(field, table, scalars, window)
    historical_column(spec)(*args)
    expected = [args[1].copy(), args[2].copy(), args[15].copy()]
    kernel = make_tiled_plasma(spec)
    bound, cap = column_bounds(budgets, spec, field, table, scalars, window)
    output = packed_run(kernel, args, scalars, capacity)
    largest = assert_column(
        (output, args[2], args[-1]), expected, bound, cap, stash=stash
    )
    # Structural: the input field is read, never written.
    np.testing.assert_array_equal(field.view(np.uint64), unchanged.view(np.uint64))
    record_property(
        "tiled_plasma",
        {
            "nodes": count,
            "capacity": capacity,
            "samples": samples,
            "parallel": parallel,
            "maximum_absolute_error": largest,
            "bound": float(np.max(bound["column_bound"])),
        },
    )


def test_tiled_column_agrees_across_worker_counts(budgets, record_property):
    """Every worker count, twice, against the serial column, inside its budget."""
    field, table, scalars, window = inputs(256, 21, 0.2)
    spec = ColumnSpec("field_real", True, True, True, False, 1, True, True)
    bound, cap = column_bounds(budgets, spec, field, table, scalars, window)
    args = reference_arguments(field, table, scalars, window)
    historical_column(spec._replace(parallel=False))(*args)
    expected = [args[1].copy(), args[2].copy(), args[15].copy()]
    capacity = int(vars(config)["NUMBA_NUM_THREADS"])
    counts = [n for n in (1, 3, 5, 8, 18) if n <= capacity]
    original = get_num_threads()
    kernel = make_tiled_plasma(spec)
    try:
        for count in counts:
            set_num_threads(count)
            assert get_num_threads() == count
            for _ in range(2):
                output = packed_run(kernel, args, scalars, 24)
                assert_column((output, args[2], args[-1]), expected, bound, cap)
    finally:
        set_num_threads(original)
    record_property("realized_workers", counts)


def test_tiled_column_preserves_the_positive_rate_underflow_guard(budgets):
    field = np.full((9, 32), 1e-170)
    table = RateArrays(
        np.zeros((4, 1)), math.log(1e-180), math.log(1e20), 1e-180, 1e-160, 1.0
    )
    scalars = PlasmaScalars(0.0, 3.0, 0.2, 2.0, 0.0, 1e-3)
    spec = ColumnSpec("field_real", True, False, False, False, 1, True, True)
    window = np.ones(32)
    bound, cap = column_bounds(budgets, spec, field, table, scalars, window)
    args = reference_arguments(field, table, scalars, window)
    historical_column(spec)(*args)
    expected = args[1].copy()
    output = packed_run(make_tiled_plasma(spec), args, scalars, 16)
    within(output, expected, bound["column_bound"], cap)


@pytest.mark.parametrize(
    "family,plasma,n_pol",
    [
        ("envelope", True, 1),
        ("field_analytic", True, 1),
        ("field_real", True, 2),
        ("field_real", False, 1),
    ],
)
def test_tiled_plasma_refuses_unsupported_families(family, plasma, n_pol):
    spec = ColumnSpec(family, plasma, False, False, False, n_pol, False)
    with pytest.raises(BackendError, match="requires scalar field_real plasma"):
        make_tiled_plasma(spec)


def test_tiled_plasma_refuses_inconsistent_real_raman_drive():
    spec = ColumnSpec("field_real", True, True, False, False, 1, False)
    with pytest.raises(BackendError, match="requires real-field Raman drive"):
        make_tiled_plasma(spec)
