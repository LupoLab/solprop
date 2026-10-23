# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ten steps of solprop's general modal route on ``numpy_ref`` and Kokkos.

Each case of ``tests/test_general_modal.py`` propagates the same ten fixed steps on
both backends in this process; the script prints one JSON object per case with the
derived parity bound, the measured departure, the window's nonlinear change and the
propagation times (every repeat, ``propagate_to`` only, construction excluded).
Run it with the thread environment fixed, for example ``SOLPROP_NATIVE_THREADS=1
OMP_PROC_BIND=false python tests/general_modal_benchmark.py``.
It lives beside the tests, as the thread probes do, because it propagates their cases.
"""

# ruff: noqa: I001 -- Kokkos must load before solprop imports pyFFTW on macOS.

from __future__ import annotations

import argparse
import json
import time

import solprop_native

import numpy as np
from solprop.backends.numpy_ref import NumpyRefBackend

from test_general_modal import (
    CASES,
    Z_END,
    Case,
    _linear,  # pyright: ignore[reportPrivateUsage]
    _parity_bound,  # pyright: ignore[reportPrivateUsage]
)

from solprop_native import NativeBackend


def _timed(
    backend_class: type[NumpyRefBackend] | type[NativeBackend], case: Case, repeats: int
) -> tuple[np.ndarray, list[float], int]:
    seconds: list[float] = []
    field = np.empty(0, dtype=np.complex128)
    ledger = 0
    for _ in range(repeats):
        backend = backend_class(case.problem, grid=case.simulation.grid)
        try:
            backend.set_initial(case.state, None)
            started = time.perf_counter()
            backend.propagate_to(Z_END)
            seconds.append(time.perf_counter() - started)
            field = backend.observe().require_field()
            ledger = sum(
                value
                for name, value in backend.memory_ledger().items()
                if name.startswith("general_")
            )
        finally:
            backend.close()
    return field, seconds, ledger


def main() -> None:
    """Print one JSON record per case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", default=sorted(CASES))
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    info = solprop_native.build_info()
    for name in arguments.cases:
        case = CASES[name]()
        expected, reference_seconds, reference_bytes = _timed(
            NumpyRefBackend, case, arguments.repeats
        )
        actual, native_seconds, native_bytes = _timed(
            NativeBackend, case, arguments.repeats
        )
        linear = _linear(case)
        scale = float(np.max(np.abs(case.state)))
        tables = case.problem.transform.modal_tables
        assert tables is not None
        record = {
            "case": name,
            "shape": list(case.problem.shape),
            "n_to": case.problem.grids.n_to,
            "nodes": int(tables.region_index.size),
            "threads": info.thread_count,
            "nonlinear_change": float(np.max(np.abs(expected - linear))) / scale,
            "bound_relative": _parity_bound(case, expected, linear) / scale,
            "departure_relative": float(np.max(np.abs(actual - expected))) / scale,
            "numpy_ref_seconds": reference_seconds,
            "native_seconds": native_seconds,
            "numpy_ref_general_bytes": reference_bytes,
            "native_general_bytes": native_bytes,
        }
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
