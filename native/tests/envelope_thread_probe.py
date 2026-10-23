# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write envelope fields and populations for the independent-process thread test.

Four routes of a two-component envelope mixture, eight prescribed steps each. Usage:
``envelope_thread_probe.py <out.npz>``.
"""

# ruff: noqa: I001 -- Kokkos must load before solprop imports pyFFTW on macOS.

from __future__ import annotations

import json
import sys
import warnings

import solprop_native

import numpy as np

from solprop.exceptions import MaterialRangeWarning
from test_envelope_plasma import NAMES, ROUTES, envelope_case

arrays: dict[str, np.ndarray] = {}
steps: list[int] = []
for route in ROUTES:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem, grid = envelope_case(route, 2)
    owner = solprop_native.NativeBackend(problem, grid=grid)
    try:
        owner.set_initial(problem.initial.field0, None)
        summary = owner.propagate_to(grid.length)
        observation = owner.observe()
        values = owner.stats(problem.stats_spec)
        arrays[f"{route}_field"] = observation.require_field()
        for name in NAMES:
            arrays[f"{route}_plane_{name}"] = np.asarray(values[name])
        for name in problem.stats_spec.periodic:
            arrays[f"{route}_rows_{name}"] = observation.step_stats.values[name]
        steps.append(summary.n_steps)
    finally:
        owner.close()
np.savez(sys.argv[1], allow_pickle=False, **arrays)

print(
    json.dumps(
        {
            "threads": solprop_native.build_info().thread_count,
            "steps": steps,
        }
    )
)
