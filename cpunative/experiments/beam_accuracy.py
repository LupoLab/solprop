# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare the complete standard Cartesian case against its independent oracle."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from benchmarks.builders import build
from benchmarks.registry import CASES
from solprop.backends.registry import select


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reference", choices=("numpy_ref", "numba_cpu"), default="numpy_ref"
    )
    args = parser.parse_args()
    case = CASES["kerr_beam_384"]
    simulation = build(case.configuration("default", 1, "numpy_ref", 1.0))
    problem = simulation.translate()
    geometry = problem.transform.geometry
    assert geometry is not None
    # The same transform/four-step allowance used by the small independent tests,
    # fixed from grid dimensions before either backend executes.
    bound = 128 * np.finfo(float).eps * (problem.grids.n_t + sum(geometry.shape))
    records = {}
    fields = []
    statistics = []
    for name in (args.reference, "cpunative"):
        backend = select(name, problem, grid=simulation.grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            start = time.perf_counter()
            backend.propagate_to(case.length)
            elapsed = time.perf_counter() - start
            fields.append(backend.observe().require_field())
            statistics.append(backend.stats(problem.stats_spec))
            records[name] = dict(seconds=elapsed, metadata=backend.metadata())
        finally:
            backend.close()
        del backend
        gc.collect()
    expected, actual = (field.reshape(-1, problem.grids.n_t) for field in fields)
    error_squared = 0.0
    scale_squared = 0.0
    # Bound temporary storage; the complete standard volume contains 110M values.
    for first in range(0, len(expected), 1024):
        reference = expected[first : first + 1024]
        candidate = actual[first : first + 1024]
        assert np.isfinite(candidate).all()
        assert np.count_nonzero(candidate[:, ~problem.grids.active_mask]) == 0
        difference = candidate - reference
        error_squared += float(np.vdot(difference, difference).real)
        scale_squared += float(np.vdot(reference, reference).real)
    relative_error = np.sqrt(error_squared / scale_squared)
    assert relative_error < bound, (relative_error, bound)
    stat_errors = {}
    for name, reference in statistics[0].items():
        scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
        candidate = statistics[1][name]
        np.testing.assert_allclose(
            candidate, reference, rtol=bound, atol=bound * scale, err_msg=name
        )
        stat_errors[name] = float(np.max(np.abs(candidate - reference))) / scale
    record = dict(
        case=case.name,
        reference=args.reference,
        shape=problem.shape,
        relative_field_error=relative_error,
        relative_bound=bound,
        scaled_statistic_errors=stat_errors,
        runs=records,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in record.items() if key != "runs"}))


if __name__ == "__main__":
    main()
