# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure the usual observations on identical stored propagation fields."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from benchmarks.builders import capillary
from solprop.backends.registry import select


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("rdw", "ctc", "argon_blueshift"), required=True
    )
    parser.add_argument("--backend", choices=("native", "cpunative"), required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    simulation = capillary(args.case)
    problem = simulation.translate()
    stored = np.load(args.trajectory, allow_pickle=False)
    backend = select(
        args.backend,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    try:
        for plane, z in enumerate(stored["z"]):
            backend.set_initial(
                stored["field"][plane].reshape(problem.shape), None, z=z
            )
            for _ in range(2):
                backend.observe(field=True)
                backend.stats(problem.stats_spec)
            field_seconds, statistics_seconds = [], []
            for _ in range(args.repeats):
                start = time.perf_counter()
                backend.observe(field=True)
                field_seconds.append(time.perf_counter() - start)
                start = time.perf_counter()
                backend.stats(problem.stats_spec)
                statistics_seconds.append(time.perf_counter() - start)
            print(
                json.dumps(
                    dict(
                        case=args.case,
                        backend=args.backend,
                        plane=plane,
                        z=float(z),
                        field_seconds=field_seconds,
                        statistics_seconds=statistics_seconds,
                    )
                ),
                flush=True,
            )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
