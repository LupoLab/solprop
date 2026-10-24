# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the fixed tree and carry before pricing a plasma scan candidate."""

import math

import numpy as np
import pytest
from numba import config, get_num_threads, set_num_threads

import plasma_prototypes as prototypes
from plasma_prototypes import make_scan_kernels, make_tiled_plasma
from solprop.backends.numba_cpu.kernels import ColumnSpec
from solprop.exceptions import BackendError


def gamma(count):
    scaled = count * np.finfo(float).eps
    return scaled / (1.0 - scaled)


def execute(kernels, increments, prefix, totals):
    kernels.local(increments, prefix, totals)
    kernels.carries(totals)
    kernels.apply(prefix, totals)


@pytest.mark.parametrize("samples", [1, 2, 16, 32, 128, 1024])
@pytest.mark.parametrize("parallel", [False, True])
def test_blocked_scan_matches_serial_with_derived_absolute_budget(
    samples, parallel, record_property
):
    kernels = make_scan_kernels(samples, parallel=parallel)
    expected_segments = min(16, samples)
    depth = (
        2 * math.ceil(math.log2(samples // expected_segments)) + expected_segments + 3
    )
    assert kernels.segments == expected_segments and kernels.depth == depth
    rng = np.random.default_rng(383)
    increments = rng.normal(size=(3, samples))
    increments[1] *= np.where(np.arange(samples) % 2, 1e-8, 1e8)
    increments[2] = 0.0
    unchanged = increments.copy()
    prefix = np.full_like(increments, np.nan)
    totals = np.full((3, kernels.segments), np.nan)
    # Both algorithms' addition paths act on the same increments.
    # Inflate the computed absolute prefix for its own serial rounding, then
    # outward-round the budget; eight eps covers its few scalar arithmetic steps.
    absolute = np.cumsum(np.abs(increments), axis=-1) / (1 - gamma(samples - 1))
    budget = np.nextafter(
        (gamma(depth) + gamma(samples - 1)) * absolute * (1 + 8 * np.finfo(float).eps),
        np.inf,
    )
    expected = np.cumsum(increments, axis=-1)
    execute(kernels, increments, prefix, totals)
    error = np.abs(prefix - expected)
    assert np.all(np.isfinite(prefix)) and np.all(error <= budget)
    # Structural: the increments are read, never written; a sum of zeros is zero in
    # any order; and one sample is a copy, with no addition to round.
    np.testing.assert_array_equal(increments.view(np.uint64), unchanged.view(np.uint64))
    np.testing.assert_array_equal(prefix[2], np.zeros(samples))
    if samples == 1:
        np.testing.assert_array_equal(prefix, increments)
    if samples == 1024:
        assert np.max(error) > 0.0
    record_property(
        "blocked_scan",
        {
            "samples": samples,
            "segments": kernels.segments,
            "depth": depth,
            "maximum_absolute_error": float(np.max(error)),
            "maximum_absolute_budget": float(np.max(budget)),
            "maximum_entrywise_budget_ratio": float(np.max(error / budget)),
            "owned_work_bytes": prefix.nbytes + totals.nbytes,
        },
    )


def test_blocked_scan_agrees_across_workers_and_reuse(
    thread_budget_lifted, record_property
):
    """Every worker count, twice, against the first run, inside the scan's budget.

    Each run is the blocked scan of depth ``D`` on the same increments, so each is
    within ``gamma(D)`` of the absolute prefix of the exact sum and two runs within
    twice that (SEP 25 section 4.6), whatever order a worker count gives the sums.
    """
    kernels = make_scan_kernels(256, parallel=True)
    increments = np.random.default_rng(384).normal(size=(5, 256))
    absolute = np.cumsum(np.abs(increments), axis=-1) / (1 - gamma(255))
    budget = 2 * gamma(kernels.depth) * absolute * (1 + 8 * np.finfo(float).eps)
    prefix = np.empty_like(increments)
    totals = np.empty((5, kernels.segments))
    original_threads = get_num_threads()
    # Numba populates environment-derived values in its module namespace.
    capacity = int(vars(config)["NUMBA_NUM_THREADS"])
    counts = [count for count in (1, 4, 8, 12, 18) if count <= capacity]
    reference = None
    try:
        for count in counts:
            set_num_threads(count)
            assert get_num_threads() == count
            for _ in range(2):
                prefix[:] = np.nan
                totals[:] = np.nan
                execute(kernels, increments, prefix, totals)
                if reference is None:
                    reference = prefix.copy()
                assert np.all(np.isfinite(prefix))
                assert np.all(np.abs(prefix - reference) <= budget)
    finally:
        set_num_threads(original_threads)
    record_property("realized_workers", counts)


@pytest.mark.parametrize(
    "samples, segments", [(0, 16), (-1, 16), (3, 16), (8, 0), (8, -1), (8, 3)]
)
def test_blocked_scan_refuses_invalid_partition(samples, segments):
    with pytest.raises(BackendError, match="positive power of two"):
        make_scan_kernels(samples, segments=segments)


def _disk_cached(dispatcher) -> bool:
    """Return whether numba writes this dispatcher's compilations to its disk cache."""
    return type(dispatcher._cache).__name__ != "NullCache"


@pytest.mark.parametrize("parallel", [False, True])
def test_a_parallel_prototype_closure_is_never_disk_cached(parallel):
    """The order-dependent failure of the blocked plasma column, prevented.

    Numba names a parfor's gufunc from its IR node's address and a per-process
    counter, so two variants of one parallel closure compiled in different processes
    can carry the same symbol; loaded from the disk cache into one process, the second
    binds to the first's gufunc and runs with the other's constants (the blocked
    column then left all but a few outputs unwritten). A serial closure has no gufunc
    and keeps its cache.
    """
    kernels = make_scan_kernels(16, parallel=parallel)
    spec = ColumnSpec("field_real", True, False, False, False, 1, parallel, False)
    dispatchers = (
        kernels.local,
        kernels.carries,
        kernels.apply,
        prototypes._scan._make_rates(spec, 16, 16),
        prototypes._scan._make_increments(spec, 16, 16),
        prototypes._scan._make_finish(spec, 16, 16),
        make_tiled_plasma(spec),
    )
    assert [_disk_cached(one) for one in dispatchers] == [not parallel] * len(
        dispatchers
    )
