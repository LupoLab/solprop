# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify repeated-exposure fraction reuse against the ordered column.

The reusing column is another specialisation of the same arithmetic, so it is held to
the column chain's rounding budget (:func:`matched.assert_columns_agree`); the input
field is never written, exactly.
"""

import numpy as np
import pytest
from numba import config, get_num_threads, set_num_threads

from backends.matched import assert_columns_agree
from bounds import scan_bound
from plasma_column_inputs import inputs, reference_arguments
from solprop.backends.numba_cpu.kernels import ColumnSpec, make_response_column


def column_arguments(family, n_pol, preionisation, samples=32):
    field, table, scalars, window = inputs(samples, 9 * n_pol, preionisation)
    if family == "field_analytic":
        field = field.astype(complex) * (1 + 0.3j)
    args = reference_arguments(field, table, scalars, window)
    return (
        field,
        np.full(field.shape, np.nan),
        np.zeros((9, 2 * samples)),
        window,
        table,
        args[5],
        scalars.kerr_factor,
        0.75,
        0.25,
        0.0,
        scalars.charge_mass_ratio,
        scalars.ionisation_potential,
        scalars.density,
        scalars.preionisation,
        scalars.dt,
        np.full((9, 2), -1.0),
        None,
        np.zeros(field.shape),
    )


def assert_outputs(actual, expected, samples):
    assert_columns_agree(actual, expected, samples)


@pytest.mark.parametrize("family", ["field_real", "field_analytic"])
@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize(
    "response",
    [
        (False, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("preionisation", [0.0, 1e-20, 0.2, 1.0])
def test_fraction_reuse_keeps_the_column_drive_and_stash(
    family, n_pol, response, parallel, preionisation, record_property
):
    plasma, raman, stash = response
    spec = ColumnSpec(
        family, plasma, raman, family == "field_real", False, n_pol, parallel, stash
    )
    args = column_arguments(family, n_pol, preionisation)
    unchanged = args[0].copy()
    make_response_column(spec)(*args)
    expected = [args[1].copy(), args[2].copy(), args[15].copy(), args[17].copy()]
    make_response_column(spec, reuse_unchanged_fraction=True)(*args)
    samples = args[0].shape[-1]
    assert_outputs([args[i] for i in (1, 2, 15, 17)], expected, samples)
    # Structural: the input field is read, never written.
    np.testing.assert_array_equal(args[0].view(np.uint64), unchanged.view(np.uint64))
    record_property(
        "fraction_reuse",
        {
            "bound": scan_bound(samples),
            "family": family,
            "n_pol": n_pol,
            "preionisation": preionisation,
        },
    )


def test_fraction_reuse_always_evaluates_first_sample_and_is_thread_invariant(
    thread_budget_lifted,
):
    spec = ColumnSpec("field_real", True, True, True, False, 1, True, True)
    args = column_arguments("field_real", 1, 1e-20, 64)
    args[0][:] = 0.0
    args[0][:, 20:30] = 2.0
    make_response_column(spec._replace(parallel=False))(*args)
    expected = [args[1].copy(), args[2].copy(), args[15].copy(), args[17].copy()]
    kernel = make_response_column(spec, reuse_unchanged_fraction=True)
    old = get_num_threads()
    try:
        for count in (1, 4, 8, 12, 18):
            if count <= int(vars(config)["NUMBA_NUM_THREADS"]):
                set_num_threads(count)
                for _ in range(2):
                    kernel(*args)
                    assert_outputs([args[i] for i in (1, 2, 15, 17)], expected, 64)
    finally:
        set_num_threads(old)


def test_fraction_reuse_one_sample_preserves_rounding_of_preionisation():
    spec = ColumnSpec("field_real", True, False, False, False, 1, False, True)
    args = column_arguments("field_real", 1, 1e-20, 1)
    make_response_column(spec)(*args)
    expected = [args[1].copy(), args[2].copy(), args[15].copy(), args[17].copy()]
    assert np.all(args[15][:, 0] == 0.0)
    make_response_column(spec, reuse_unchanged_fraction=True)(*args)
    assert_outputs([args[i] for i in (1, 2, 15, 17)], expected, 1)
