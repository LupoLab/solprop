# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep the Kerr sums when an unused plasma magnitude is omitted, to their rounding.

The gather without the magnitude is another specialisation of the same arithmetic, so
its sums are held to their few roundings and the column's outputs to the column
chain's budget (:func:`matched.assert_columns_agree`).
"""

import numpy as np
import pytest

from backends.matched import assert_columns_agree
from bounds import assert_within
from plasma_column_inputs import inputs, reference_arguments
from solprop.backends.numba_cpu import kernels


def _arguments(family):
    field, table, scalars, window = inputs(32, 18, 0.2)
    if family != "field_real":
        field = field.astype(complex) * (1 + 0.3j)
    args = reference_arguments(field, table, scalars, window)
    output = np.full(field.shape, np.nan, dtype=field.dtype)
    if family == "field_analytic":
        output = np.full(field.shape, np.nan)
    return (
        field,
        output,
        np.zeros((9, 64)),
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


@pytest.mark.parametrize("family", ["field_real", "field_analytic", "envelope"])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("required", [False, True])
def test_gather_omission_preserves_kerr_sums_and_default_magnitude(
    family, components, required
):
    spec = kernels.ColumnSpec(
        family, False, False, False, False, components, False, False
    )
    nodes = _arguments(family)[0]
    expected = kernels._make_gather(spec)(nodes, 0, 2)
    result = kernels._make_gather(spec, require_magnitude=required)(nodes, 0, 2)
    # The Kerr sums are sums of at most two products of each sample: a few roundings.
    rounding = 4 * np.finfo(float).eps
    assert_within(np.asarray(result[:4]), np.asarray(expected[:4]), rounding)
    assert np.isfinite(result).all()
    assert expected[4] > 0.0
    if required or components == 1:
        assert result[4] == pytest.approx(expected[4], rel=rounding)
    else:
        # Structural: an omitted magnitude is not computed at all.
        assert result[4] == 0.0


@pytest.mark.parametrize("family", ["field_real", "field_analytic", "envelope"])
@pytest.mark.parametrize("plasma", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("raman_stash", [False, True])
def test_unused_magnitude_keeps_column_drive_and_stash_bits(
    family, plasma, parallel, raman_stash, monkeypatch
):
    spec = kernels.ColumnSpec(
        family, plasma, raman_stash, False, False, 2, parallel, raman_stash
    )
    args = _arguments(family)
    original_gather = kernels._make_gather

    def always_gather(spec, *, require_magnitude=True):
        assert require_magnitude is spec.plasma
        return original_gather(spec)

    with monkeypatch.context() as original:
        original.setattr(kernels, "_make_gather", always_gather)
        before = kernels.make_response_column(spec, reuse_unchanged_fraction=True)
    after = kernels.make_response_column(spec, reuse_unchanged_fraction=True)
    before(*args)
    arrays = (args[1], args[2], args[15], args[17])
    expected = [values.copy() for values in arrays]
    after(*args)
    assert_columns_agree(arrays, expected, args[0].shape[-1])


@pytest.mark.parametrize("family", ["field_real", "field_analytic", "envelope"])
def test_statistics_reader_still_requests_the_vector_magnitude(family, monkeypatch):
    spec = kernels.ColumnSpec(family, True, False, False, False, 2, False, False)
    calls = []
    original_gather = kernels._make_gather

    def observed_gather(spec, *, require_magnitude=True):
        calls.append(require_magnitude)
        return original_gather(spec, require_magnitude=require_magnitude)

    monkeypatch.setattr(kernels, "_make_gather", observed_gather)
    kernels.make_stats_column(spec)
    assert calls == [True]
