# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``numba_cpu``'s accepted-stage species stash and its thread pools."""

import dataclasses

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from backends.host_cases_mixture_threads import NAMES, fixed_schedule
from backends.test_mixture_ladder import ladder_bound
from mixture_cases import mixture_case
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.stats import StatsSpec


def test_periodic_species_stash_matches_fresh_scans_and_threads(thread_budget_lifted):
    """The stash against a fresh scan of the same state, and one thread against four.

    The stash and the fresh scan are one statistics evaluation twice, held to the
    fresh-backend budget of the next test (64 eps a fine sample); the two thread
    counts replay eight fixed steps,
    so their rows stand at the same positions, exactly, with values within the
    ladder's budget.
    """
    _, grid, _, _, original = mixture_case()
    periodic = ("species_ionisation", "species_electron_density")
    problem = dataclasses.replace(
        fixed_schedule(original, grid),
        stats_spec=StatsSpec(names=periodic, periodic=periodic, period=1),
    )
    fresh_bound = 64 * grid.n_to * np.finfo(float).eps
    bound = ladder_bound(grid, 8)
    outputs = []
    for threads in (1, 4):
        backend = NumbaCpuBackend(problem, grid=grid, threads=threads)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(grid.length)
            assert backend._stash_current
            held = backend._require_state()[0]
            stashed = backend._species_statistics(held)
            backend._stash_current = False
            fresh = backend._species_statistics(held)
            for name in NAMES:
                assert_allclose(
                    stashed[name], fresh[name], rtol=fresh_bound, atol=0, err_msg=name
                )
            rows = backend.observe().step_stats
            assert rows.n_rows > 0
            assert backend._stash.shape == (
                problem.transform.n_nodes,
                1 + 2 * problem.flags.n_channels,
            )
            outputs.append(rows)
            backend._begin_plane()
            assert not backend._stash_current
        finally:
            backend.close()
    assert_array_equal(outputs[0].z, outputs[1].z)
    for name in periodic:
        absolute = bound if name == "species_ionisation" else 0.0
        assert_allclose(
            outputs[1].values[name],
            outputs[0].values[name],
            rtol=bound,
            atol=absolute,
            err_msg=name,
        )
