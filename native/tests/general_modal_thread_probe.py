# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write ten general modal steps per named case for the thread-count test.

Usage: ``general_modal_thread_probe.py <out.npz> <case> [<case> ...]``.
"""

# ruff: noqa: I001 -- Kokkos loads first, before any other OpenMP runtime on macOS.

from __future__ import annotations

import json
import sys

import solprop_native

import numpy as np
from session_runs import finish
from solprop.backends.session import Start
from test_general_modal import CASES, Z_END

from solprop_native import NativeBackend

fields: dict[str, np.ndarray] = {}
sizings: list[object] = []
for name in sys.argv[2:]:
    case = CASES[name]()
    backend = NativeBackend(case.problem, case.run_request, grid=case.simulation.grid)
    try:
        start = Start(field=case.state, z=0.0)
        run = finish(backend, case.problem, case.run_request, start, owns=False)
        assert run.planes[-1]["z"] == Z_END
        fields[name] = run.planes[-1]["field"]
        sizings.append(backend.metadata()["execution_sizing"])
    finally:
        backend.close()
np.savez(sys.argv[1], allow_pickle=False, **fields)
print(
    json.dumps(
        {"threads": solprop_native.build_info().thread_count, "sizing": sizings},
        sort_keys=True,
    )
)
