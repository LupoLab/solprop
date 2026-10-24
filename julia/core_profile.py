# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sample a warmed serial SEP 30 window in Python and Julia."""

import argparse
import cProfile
import json
import os
from pathlib import Path

from benchmarks.builders import build
from benchmarks.registry import CASES
from benchmarks.worker import measure, prepare

from solprop.backends import select


def main() -> None:
    """Keep construction, compilation, reset and lead-in outside the profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=tuple(CASES))
    parser.add_argument("--track", default="fixed-rk")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--interval", type=float, default=0.005)
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The Xeon threaded sampler terminated on SIGUSR1. Keep the demonstrated
    # serial path; parallel attribution uses explicit operation timers.
    if not args.python_only and int(os.environ.get("SOLPROP_JULIA_THREADS", "1")) != 1:
        parser.error("Julia signal sampling is qualified only with one Julia thread")
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = CASES[args.case].measurement(args.track) | {"backend": "julia"}
    simulation = build(configuration)
    problem = simulation.translate()
    engine = select(
        "julia",
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    try:
        output = prepare(simulation, problem, engine, configuration)
        measure(engine, problem, configuration, output)
        output.close()
        main = None
        if not args.python_only:
            import juliacall  # noqa: PLC0415 - private runtime is already initialised

            main = juliacall.Main  # type: ignore[attr-defined] - initialised dynamically
            main.seval("using Profile")
            main.Profile.init(n=40_000_000, delay=args.interval)
            main.Profile.clear()
        profiler = cProfile.Profile()
        records = []
        for _ in range(args.repeats):
            output = prepare(simulation, problem, engine, configuration)
            # These are the exact timer operations used by Profile.@profile;
            # starting externally also includes the shared Python march policy.
            if main is not None:
                main.Profile.start_timer()
            try:
                records.append(
                    profiler.runcall(measure, engine, problem, configuration, output)
                )
            finally:
                if main is not None:
                    main.Profile.stop_timer()
                output.close()
        profiler.dump_stats(args.output / "python.prof")
        (args.output / "windows.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        if main is None:
            return
        for form in ("flat", "tree"):
            for label, thread_filter in (("", ""), ("-main", ", threads=1")):
                report = main.seval(
                    "sprint(io -> Profile.print(io; "
                    f"format=:{form}, C=true, sortedby=:count, mincount=3"
                    f"{thread_filter}))"
                )
                (args.output / f"julia-{form}{label}.txt").write_text(
                    str(report), encoding="utf-8"
                )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
