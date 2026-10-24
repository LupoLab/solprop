# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Screen one unchanged SEP 30 case in a fresh process, outside full qualification."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from benchmarks.builders import build
from benchmarks.registry import CASES
from benchmarks.worker import runtime

from solprop.backends import select


def main() -> None:
    """Save a first window, warm windows, final field and observation statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=tuple(CASES))
    parser.add_argument("backend")
    parser.add_argument("--track", default="fixed-rk")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = runtime(args.backend)
    configuration = CASES[args.case].measurement(args.track) | {"backend": args.backend}
    begin = time.perf_counter()
    sim = build(configuration)
    problem = sim.translate()
    translation = time.perf_counter() - begin
    begin = time.perf_counter()
    engine = select(
        args.backend, problem, grid=sim.grid, modes=sim.modes, profiles=sim.profiles
    )
    setup = time.perf_counter() - begin
    times = []
    try:
        for _ in range(args.repeats + 1):
            engine.set_initial(problem.initial.field0, None)
            if configuration["start"]:
                engine.propagate_to(configuration["start"])
            before = engine.metadata()
            begin = time.perf_counter()
            engine.propagate_to(configuration["stop"])
            times.append(time.perf_counter() - begin)
        begin = time.perf_counter()
        field = engine.observe().field
        stats = engine.stats(problem.stats_spec)
        observation = time.perf_counter() - begin
        assert field is not None and np.all(np.isfinite(field))
        assert all(
            np.all(np.isfinite(a))
            for key, a in stats.items()
            if key != "quadrature_error"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output.with_suffix(".npz"), field=field, **stats)
        record = {
            "case": args.case,
            "backend": args.backend,
            "track": args.track,
            "translation_s": translation,
            "setup_s": setup,
            "first_s": times[0],
            "warm_s": times[1:],
            "observation_s": observation,
            "shape": problem.shape,
            "fine_samples": problem.grids.n_to,
            "metadata": engine.metadata(),
            "before": before,
            "provenance": provenance,
            "owned_bytes": sum(engine.memory_ledger().values()),
        }
        args.output.write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
        sys.stdout.write(
            json.dumps(
                {k: v for k, v in record.items() if k not in {"provenance", "before"}}
            )
            + "\n"
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
