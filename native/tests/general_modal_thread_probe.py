# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write ten general modal steps per named case for the thread-count test.

Usage: ``general_modal_thread_probe.py <out.npz> <case> [<case> ...]``.
"""

# ruff: noqa: I001 -- Kokkos must load before solprop imports pyFFTW on macOS.

from __future__ import annotations

import json
import sys

import solprop_native

import numpy as np
from test_general_modal import CASES, Z_END

from solprop_native import NativeBackend

fields: dict[str, np.ndarray] = {}
for name in sys.argv[2:]:
    case = CASES[name]()
    backend = NativeBackend(case.problem, grid=case.simulation.grid)
    try:
        backend.set_initial(case.state, None)
        backend.propagate_to(Z_END)
        fields[name] = np.array(backend.observe().require_field(), copy=True)
    finally:
        backend.close()
np.savez(sys.argv[1], allow_pickle=False, **fields)
print(
    json.dumps(
        {"threads": solprop_native.build_info().thread_count},
        sort_keys=True,
    )
)
