# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Two-channel fused columns against the independent NumPy passes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from backends.matched import assert_columns_agree
from mixture_cases import mixture_case
from solprop.backends.numba_cpu import kernels
from solprop.backends.numba_cpu.backend import _channel_rate_arrays
from solprop.backends.numpy_ref import passes
from solprop.translate.stage import scalars


@pytest.mark.parametrize(
    "family,n_pol",
    [
        ("field_real", 1),
        ("field_real", 2),
        ("field_analytic", 1),
        ("field_analytic", 2),
    ],
)
@pytest.mark.parametrize("parallel", [False, True])
def test_two_channel_column_matches_the_shared_integral_oracle(family, n_pol, parallel):
    problem = mixture_case(kerr=False)[-1]
    bundle = scalars(problem, 0.0, 0.0)
    n_nodes, n_to = 3, 513
    dt = 60e-15 / (n_to - 1)
    t = np.arange(n_to) * dt - 30e-15
    analytic = np.array(
        [
            (1 + row / 10) * 3e10 * np.exp(-((t / 10e-15) ** 2)) * np.exp(2.4e15j * t)
            for row in range(n_nodes * n_pol)
        ]
    )
    nodes = analytic.real.copy() if family == "field_real" else analytic
    field = nodes.real.reshape(n_nodes, n_pol, n_to)
    expected = passes._plasma_channels_current(
        field,
        problem.responses.plasma_channels,
        bundle.plasma_densities,
        bundle.plasma_preionisation,
        dt,
    )
    # Each of the two shared trapezoid integrals accumulates n_to terms. Rate
    # interpolation adds log/exp error; allow 64 eps per time sample per channel,
    # against the absolute polarisation envelope, before evaluating the column.
    bound = (
        64
        * n_to
        * problem.flags.n_channels
        * np.finfo(float).eps
        * float(np.max(np.abs(expected)))
    )
    assert bound > 0
    storage, metadata = _channel_rate_arrays(problem)
    tables = (storage, metadata)
    # Structural: the packed tables are a lossless copy of the channel's coefficients.
    assert_array_equal(
        storage[0, : problem.responses.plasma_channels[0].rate.coefficients.size],
        problem.responses.plasma_channels[0].rate.coefficients.ravel(),
    )
    spec = kernels.ColumnSpec(
        family, True, False, False, False, n_pol, parallel, True, 2, 0
    )
    column = kernels.make_response_column(spec)
    output = np.zeros(nodes.shape)
    polarisation = np.full(nodes.shape, np.nan)
    stash = np.zeros((n_nodes, 5))
    scratch = np.zeros((2, n_nodes, 4))
    args = (
        nodes,
        polarisation,
        np.zeros((n_nodes, 2 * n_to)),
        np.ones(n_to),
        tables,
        np.zeros(n_to, dtype=complex),
        0.0,
        0.0,
        0.0,
        0.0,
        np.array([c.charge_mass_ratio for c in problem.responses.plasma_channels]),
        np.array([c.ionisation_potential for c in problem.responses.plasma_channels]),
        np.array(bundle.plasma_densities),
        np.array(bundle.plasma_preionisation),
        dt,
        stash,
        scratch,
        output,
    )
    column(*args)
    assert np.count_nonzero(polarisation) == 0
    assert_allclose(output.reshape(expected.shape), expected, rtol=0, atol=bound)
    rates, fractions = passes.plasma_channel_fractions(
        field, problem.responses.plasma_channels, bundle.plasma_preionisation, dt
    )
    # The same rate interpolation can differ by a few libm ulps across compilers;
    # cumulative exposure amplifies it by at most n_to, independent of density.
    fraction_bound = 64 * n_to * np.finfo(float).eps
    assert_allclose(stash[:, :4:2].T, fractions[:, :, -1], rtol=0, atol=fraction_bound)
    first = output.copy()
    column(*args)
    # The same column repeated: within the column's budget.
    assert_allclose(output, first, rtol=0, atol=bound)
    unchanged_stash, unchanged_scratch = stash.copy(), scratch.copy()
    # The one-channel option cannot reuse another channel's state, so this
    # specialisation performs the same operations: the column chain's budget.
    kernels.make_response_column(spec, reuse_unchanged_fraction=True)(*args)
    assert_columns_agree(
        (output, stash, scratch), (first, unchanged_stash, unchanged_scratch), n_to
    )
    statistics = kernels.make_stats_column(spec)
    partials = np.zeros((5, n_nodes))
    statistics(nodes, tables, np.array(bundle.plasma_preionisation), dt, partials)
    assert_allclose(partials[:4:2], fractions[:, :, -1], rtol=0, atol=fraction_bound)
    # Peak rates use the same interpolant without a cumulative sum. The fraction
    # allowance also bounds their libm discrepancy after normalisation.
    assert_allclose(partials[1:4:2], rates.max(axis=-1), rtol=fraction_bound, atol=0)
