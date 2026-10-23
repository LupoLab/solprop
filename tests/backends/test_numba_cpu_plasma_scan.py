# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the complete blocked plasma-scan prototype before measuring its costs."""

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
    reference_arguments,
)
from solprop.backends.bodies import RateArrays
from solprop.backends.numba_cpu.kernels import ColumnSpec
from solprop.exceptions import BackendError


@pytest.fixture
def budgets(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "prototypes"))
    return importlib.import_module("plasma_bounds")


def compare(actual, expected, absolute_bound, cap):
    error = np.abs(actual - expected)
    assert np.all(np.isfinite(actual))
    assert np.all(error <= absolute_bound)
    scale = float(np.max(np.abs(expected)))
    largest = float(np.max(error))
    relative = largest / scale if scale > 0 else largest
    if scale == 0:
        assert largest == 0
    assert relative <= cap
    return {"error": relative, "cap": cap, "absolute_error": largest}


@pytest.mark.parametrize("geometry", [(1, 1), (16, 3), (32, 7), (256, 3), (1024, 1)])
@pytest.mark.parametrize("preionisation", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("response", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("parallel", [False, True])
def test_complete_blocked_column_respects_derived_envelope_and_fixed_cap(
    budgets, geometry, preionisation, response, parallel, record_property
):
    samples, nodes = geometry
    raman, stash = response
    field, table, scalars, window = inputs(samples, nodes, preionisation)
    unchanged = field.copy()
    spec = ColumnSpec("field_real", True, raman, raman, False, 1, parallel, stash)
    pipeline = make_blocked_plasma(spec, samples)
    work = budgets.allocate_work(field.shape)
    for array in work:
        array[:] = np.nan
    pipeline.rates(field, table, work)
    assert np.all(np.isfinite(work.rates))
    bound, summary = budgets.column_budget(
        field, work.rates, window, scalars, raman=raman
    )
    cap = summary["complete_chain_relative_cap"]
    old_args = reference_arguments(field, table, scalars, window)
    historical_column(spec)(*old_args)
    out, drive, stashed = old_args[1].copy(), old_args[2].copy(), old_args[-1].copy()
    out[:] = np.nan
    if stash:
        stashed[:] = np.nan
    pipeline.column(field, out, drive, window, table, scalars, stashed, work)
    result = compare(out, old_args[1], bound["column_bound"], cap)
    if stash:
        result["fraction"] = compare(
            stashed[:, 0], old_args[-1][:, 0], bound["fraction_bound"], cap
        )
        # The largest rate is a maximum of rates the two passes evaluate apart:
        # their libm and contraction paths, inside the chain's fixed cap.
        result["rate"] = compare(stashed[:, 1], old_args[-1][:, 1], np.inf, cap)
    else:
        # Structural: a column without the stash never writes it.
        np.testing.assert_array_equal(stashed, old_args[-1])
    # Structural: the Raman drive is one product of each sample, rounded once on
    # either side, and the input field is read, never written.
    np.testing.assert_array_equal(drive.view(np.uint64), old_args[2].view(np.uint64))
    np.testing.assert_array_equal(field.view(np.uint64), unchanged.view(np.uint64))
    assert sum(array.nbytes for array in work) == 8 * (
        5 * nodes * samples + 2 * nodes * min(16, samples)
    )
    record_property("blocked_plasma", {**summary, **result})


def test_blocked_complete_column_agrees_across_worker_counts(budgets, record_property):
    """Every worker count, twice, against the serial column, inside its budget.

    A worker count changes the partition of the scans and so the order of their
    sums: the column's derived envelope (:func:`plasma_bounds.column_budget`) and
    its fixed relative cap bound the difference, not its bits. The Raman drive is one
    product per sample and stays exact.
    """
    field, table, scalars, window = inputs(256, 7, 0.2)
    spec = ColumnSpec("field_real", True, True, True, False, 1, True, True)
    pipeline = make_blocked_plasma(spec, 256)
    work = budgets.allocate_work(field.shape)
    output = np.empty_like(field)
    drive = np.zeros((7, 512))
    stashed = np.empty((7, 2))
    capacity = int(vars(config)["NUMBA_NUM_THREADS"])
    counts = [n for n in (1, 4, 8, 12, 18) if n <= capacity]
    original = get_num_threads()
    serial = make_blocked_plasma(spec._replace(parallel=False), 256)
    serial.column(field, output, drive, window, table, scalars, stashed, work)
    expected = [array.copy() for array in (output, drive, stashed)]
    assert all(np.all(np.isfinite(array)) for array in expected)
    pipeline.rates(field, table, work)
    bound, summary = budgets.column_budget(
        field, work.rates, window, scalars, raman=True
    )
    cap = summary["complete_chain_relative_cap"]
    try:
        for count in counts:
            set_num_threads(count)
            assert get_num_threads() == count
            for _ in range(2):
                for array in (*work, output, stashed):
                    array[:] = np.nan
                pipeline.column(
                    field, output, drive, window, table, scalars, stashed, work
                )
                assert all(
                    np.all(np.isfinite(array)) for array in (output, drive, stashed)
                )
                compare(output, expected[0], bound["column_bound"], cap)
                np.testing.assert_array_equal(
                    drive.view(np.uint64), expected[1].view(np.uint64)
                )
                compare(stashed[:, 0], expected[2][:, 0], bound["fraction_bound"], cap)
                compare(stashed[:, 1], expected[2][:, 1], np.inf, cap)
    finally:
        set_num_threads(original)
    record_property("realized_workers", counts)


def test_blocked_column_preserves_positive_rate_underflow_guard(
    budgets, record_property
):
    field = np.full((1, 32), 1e-170)
    table = RateArrays(
        np.zeros((4, 1)), math.log(1e-180), math.log(1e20), 1e-180, 1e-160, 1.0
    )
    scalars = PlasmaScalars(0.0, 3.0, 0.2, 2.0, 0.0, 1e-3)
    window = np.ones(32)
    spec = ColumnSpec("field_real", True, False, False, False, 1, False, True)
    pipeline = make_blocked_plasma(spec, 32)
    work = budgets.allocate_work(field.shape)
    pipeline.rates(field, table, work)
    bounds, summary = budgets.column_budget(
        field, work.rates, window, scalars, raman=False
    )
    assert np.all(work.rates > 0) and np.all(field * field == 0)
    args = reference_arguments(field, table, scalars, window)
    historical_column(spec)(*args)
    out = np.empty_like(field)
    pipeline.column(field, out, args[2], window, table, scalars, args[-1], work)
    result = compare(
        out, args[1], bounds["column_bound"], summary["complete_chain_relative_cap"]
    )
    assert np.all(work.loss == 0)
    record_property("blocked_plasma_guard", {**summary, **result})


@pytest.mark.parametrize(
    "family,plasma,n_pol",
    [
        ("envelope", True, 1),
        ("field_analytic", True, 1),
        ("field_real", True, 2),
        ("field_real", False, 1),
    ],
)
def test_blocked_plasma_refuses_outside_its_prototype_domain(family, plasma, n_pol):
    spec = ColumnSpec(family, plasma, False, False, False, n_pol, False)
    with pytest.raises(BackendError, match="requires scalar field_real plasma"):
        make_blocked_plasma(spec, 32)


def test_blocked_plasma_refuses_inconsistent_real_raman_drive():
    spec = ColumnSpec("field_real", True, True, False, False, 1, False)
    with pytest.raises(BackendError, match="requires real-field Raman drive"):
        make_blocked_plasma(spec, 32)


@pytest.mark.parametrize("stash", [False, True])
def test_parallel_rate_pass_writes_every_actual_chunk_sample(budgets, stash):
    field, table, _, _ = inputs(8192, 33, 0.0)
    field[:] = 0.0
    field[::2] = 8.0
    spec = ColumnSpec("field_real", True, True, True, False, 1, True, stash)
    pipeline = make_blocked_plasma(spec, 8192)
    work = budgets.allocate_work(field.shape)
    for array in work:
        array[:] = np.nan
    pipeline.rates(field, table, work)
    # Both branches are exact table endpoints, independent of exp/log rounding:
    # structural, so exact.
    expected = np.zeros_like(field)
    expected[::2] = table.rate_max
    np.testing.assert_array_equal(work.rates, expected)
    expected_maxima = np.zeros_like(work.maxima)
    if stash:
        expected_maxima[::2] = table.rate_max
    np.testing.assert_array_equal(work.maxima, expected_maxima)
