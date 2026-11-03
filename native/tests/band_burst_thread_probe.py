# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write the compact burst's fields and statistics for the thread-count test.

Two marches of ``test_band_burst_threads.burst`` in one process, whose team reuses the
fraction over an unchanged exposure in every sample outside the burst. Usage:
``band_burst_thread_probe.py <out.npz>``.
"""

# ruff: noqa: I001 -- Kokkos loads first, before any other OpenMP runtime on macOS.

from __future__ import annotations

import json
import sys
import warnings

import solprop_native

import numpy as np
from test_band_burst_threads import STEPS, burst
from test_band_plasma import STATISTICS, march

from solprop_native import NativeBackend

arrays: dict[str, np.ndarray] = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grid, problem = burst(STEPS)
    for call in range(2):
        field, stats = march(NativeBackend, problem, grid)
        arrays[f"field_{call}"] = field
        for name in STATISTICS:
            arrays[f"{name}_{call}"] = np.asarray(stats[name])
    backend = NativeBackend(problem, grid=grid)
    try:
        sizing = backend.metadata()["execution_sizing"]
    finally:
        backend.close()
np.savez(sys.argv[1], allow_pickle=False, **arrays)

print(
    json.dumps({"threads": solprop_native.build_info().thread_count, "sizing": sizing})
)
