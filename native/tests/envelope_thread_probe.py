# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write envelope fields and populations for the independent-process thread test.

Four routes of a two-component envelope mixture, eight prescribed steps each. Usage:
``envelope_thread_probe.py <out.npz>``.
"""

# ruff: noqa: I001 -- Kokkos loads first, before any other OpenMP runtime on macOS.

from __future__ import annotations

import json
import sys
import warnings

import solprop_native

import numpy as np

from solprop.exceptions import MaterialRangeWarning
from session_runs import committed_steps, run
from test_envelope_plasma import NAMES, ROUTES, envelope_case, envelope_options

arrays: dict[str, np.ndarray] = {}
counts: list[int] = []
sizings: list[object] = []
for route in ROUTES:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem, run_request, grid = envelope_case(route, 2)
    owner = solprop_native.NativeBackend(
        problem, run_request, grid=grid, options=envelope_options(route)
    )
    try:
        result = run(owner, problem, run_request)
        last = result.all_planes()[-1]
        assert last.z == grid.length
        arrays[f"{route}_field"] = result.fields[-1]
        for name in NAMES:
            arrays[f"{route}_plane_{name}"] = last.stats[name]
        _, rows = result.joined
        for name in run_request.observation.stats.periodic:
            arrays[f"{route}_rows_{name}"] = rows.values[name]
        counts.append(committed_steps(result))
        sizings.append(owner.metadata()["execution_sizing"])
    finally:
        owner.close()
np.savez(sys.argv[1], allow_pickle=False, **arrays)

print(
    json.dumps(
        {
            "threads": solprop_native.build_info().thread_count,
            "steps": counts,
            "sizing": sizings,
        }
    )
)
