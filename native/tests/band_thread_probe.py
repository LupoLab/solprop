# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write band-rule fields and statistics for the thread-count test.

solprop SEP 18's band plasma column on two routes -- two components on the radial route
and a two-population capillary mixture -- for two fixed steps each, and two
single-node windows of 4096 samples (a preionised field below threshold, and a mixture
cut by the window's end) for ten, whose rows a host team of several threads splits;
every plasma statistic and the run diagnostics are read at the end. Usage:
``band_thread_probe.py <out.npz>``.
"""

# ruff: noqa: I001 -- Kokkos loads first, before any other OpenMP runtime on macOS.

from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Mapping

import solprop_native

import numpy as np
from solprop.stats import StatsSpec
from test_band_plasma import STATISTICS, march, modal, spatial
from test_band_stress import case

from solprop_native import NativeBackend

# Each march's backend metadata, for its ``execution_sizing``.
records: list[Mapping[str, object]] = []

arrays: dict[str, np.ndarray] = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for label, (grid, problem, run_request) in (
        ("radial", spatial("radial")),
        ("mixture", modal("mixture", "band")),
        ("long_quiet", case("preionised_quiet", 10)),
        ("long_mixture", case("mixture_edge", 10)),
    ):
        run_request = run_request.for_run(stats=StatsSpec(names=STATISTICS))
        field, stats = march(NativeBackend, problem, run_request, grid, record=records)
        arrays[f"{label}_field"] = field
        for name in STATISTICS:
            arrays[f"{label}_{name}"] = np.asarray(stats[name])
np.savez(sys.argv[1], allow_pickle=False, **arrays)

print(
    json.dumps(
        {
            "threads": solprop_native.build_info().thread_count,
            "sizing": [record["execution_sizing"] for record in records],
        }
    )
)
