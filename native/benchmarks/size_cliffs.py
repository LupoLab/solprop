# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure the cost per source evaluation over 2-3-5-smooth and power-of-two grid sizes.

The four-mode helium RDW capillary (125 um, 0.8 bar falling to vacuum, 7.5 fs and
275 uJ at 800 nm, band 90 nm to 4 um) is built on field grids whose window is the
case's 400 fs times each factor, with ``fft_size="smooth"`` and ``"pow2"``. Each size
runs short and long Runge-Kutta windows of capped steps; the cost of one evaluation
is their difference in wall time over their difference in evaluations, which removes
construction, planning and compilation. The minimum over repeats is kept.

A size whose cost per coarse sample exceeds that of the nearest power of two (in
log2 n_t) by more than ``--limit`` is a cliff: thread widths or node chunks that change
at a size boundary rather than with the work. The nearest power of two, not the
cheapest, because fixed per-evaluation costs make small grids dearer per sample.
``--check`` exits non-zero on one. Run on a quiet machine, with the threads pinned::

    OMP_PLACES=cores OMP_PROC_BIND=close python native/benchmarks/size_cliffs.py --check
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import warnings
from typing import Any, cast

FACTORS = (1.0, 1.125, 1.25, 1.375, 1.5, 1.75, 2.0)
LENGTH = 1.5
STEP = LENGTH / 4000
SHORT, LONG = 10, 40


def simulation(factor: float, fft_size: str, backend: str, steps: int) -> Any:
    """Return the RDW case on one grid, with `steps` capped Runge-Kutta steps."""
    from solprop.driver import Simulation
    from solprop.grid import FieldGrid
    from solprop.stepper import RungeKutta

    grid = FieldGrid.from_limits(
        LENGTH, 800e-9, (90e-9, 4e-6), factor * 400e-15, fft_size=fft_size
    )
    built = Simulation.from_capillary(
        125e-6,
        steps * STEP,
        "He",
        (0.8, 0.0),
        grid=grid,
        fwhm=7.5e-15,
        energy=275e-6,
        modes=4,
        wavelength=800e-9,
        rtol=1e-6,
        atol=1e-10,
        noise=None,
        n_save=2,
        backend=backend,
        integrator=RungeKutta(),
    )
    built.settings = dataclasses.replace(built.settings, dz_max=STEP, dz_init=STEP)
    return built


def timed(
    factor: float, fft_size: str, backend: str, steps: int
) -> tuple[int, int, float]:
    """Return the samples, source evaluations and wall seconds of one run."""
    built = simulation(factor, fft_size, backend, steps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        output = built.run(progress="none")
        wall = time.perf_counter() - start
    execution = cast("dict[str, int]", output.outcome.execution or {})
    evaluations = int(execution["source_evaluations"])
    output.close()
    return int(built.grid.n_t), evaluations, wall


def main() -> None:
    """Time every size once warm and report the cost per evaluation and per sample."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numba_cpu", "native"), default="native")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=float, default=1.25)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for factor in FACTORS:
        for fft_size in ("smooth", "pow2"):
            n_t = int(simulation(factor, fft_size, arguments.backend, 1).grid.n_t)
            if n_t in seen:
                continue
            seen.add(n_t)
            timed(factor, fft_size, arguments.backend, SHORT)
            best = float("inf")
            for _ in range(arguments.repeats):
                _, short_calls, short_wall = timed(
                    factor, fft_size, arguments.backend, SHORT
                )
                _, long_calls, long_wall = timed(
                    factor, fft_size, arguments.backend, LONG
                )
                if long_calls > short_calls:
                    best = min(
                        best, (long_wall - short_wall) / (long_calls - short_calls)
                    )
            rows.append(
                {"factor": factor, "fft_size": fft_size, "n_t": n_t, "seconds": best}
            )
            if not arguments.json:
                print(
                    f"n_t {n_t:6d} {fft_size:6s} {1e3 * best:7.3f} ms/evaluation "
                    f"{1e9 * best / n_t:7.1f} ns/sample",
                    flush=True,
                )
    powers = {
        row["n_t"]: row["seconds"] / row["n_t"]
        for row in rows
        if row["fft_size"] == "pow2"
    }
    for row in rows:
        nearest = min(powers, key=lambda n: abs(math.log2(n / row["n_t"])))
        row["ratio"] = row["seconds"] / row["n_t"] / powers[nearest]
    cliffs = [row for row in rows if row["ratio"] > arguments.limit]
    if arguments.json:
        print(json.dumps({"rows": rows, "cliffs": [row["n_t"] for row in cliffs]}))
    else:
        for row in cliffs:
            print(
                f"cliff: n_t {row['n_t']} costs {row['ratio']:.2f} x the nearest power"
                " of two per sample"
            )
    if arguments.check and cliffs:
        sys.exit(1)


if __name__ == "__main__":
    main()
