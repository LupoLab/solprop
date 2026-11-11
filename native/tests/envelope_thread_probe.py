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
from session_runs import finish, joined, steps
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
        planes = finish(owner, problem, run_request, owns=False).planes
        assert planes[-1]["z"] == grid.length
        arrays[f"{route}_field"] = planes[-1]["field"]
        for name in NAMES:
            arrays[f"{route}_plane_{name}"] = planes[-1]["stats"][name]
        _, rows = joined(planes)
        for name in run_request.observation.stats.periodic:
            arrays[f"{route}_rows_{name}"] = rows[name]
        counts.append(steps(planes)[0])
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
