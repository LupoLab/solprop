# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What the transverse edge and the safety ramp cost a run, on one implementation.

SEP 40 section 7.7's control: each case is built by the harness's builder and run four
ways in one process, interleaved round by round, against itself rather than an older
build. Each run keeps the temporal monitor, so the comparison is the ramp's and the
edge's alone:

- ``neither``: no ramp and no reads;
- ``ramp``: the ramp alone;
- ``edge``: the reads alone;
- ``both``: the default.

Usage::

    python -m benchmarks.edge_overhead --case line --case planar --rounds 9 \\
        --backend native --width 18 --out overhead.jsonl

One JSON line per run, with its wall time and what it took: the march's calls, source
calls and reads. Building and translating a variant is timed apart, as construction. A
summary follows on stdout. Pin the process and set the threads yourself; the module
says what it found in its first line.
"""

import argparse
import dataclasses
import json
import os
import random
import statistics
import time
import warnings
from pathlib import Path
from typing import Any

from benchmarks.builders import build
from benchmarks.registry import CASES
from solprop.driver import Simulation
from solprop.exceptions import BoundaryWarning

__all__ = ["main"]

VARIANTS = {
    "neither": (False, False),
    "ramp": (True, False),
    "edge": (False, True),
    "both": (True, True),
}
"""Each variant's ramp and reads."""


def _variant(
    name: str, backend: str, width: int, fraction: float, ramp: bool, edge: bool
) -> tuple[Simulation, float]:
    """Return a case's simulation with the ramp and the reads as asked, and its build time.

    Parameters
    ----------
    name : str
        The case.
    backend : str
        The backend.
    width : int
        The threads.
    fraction : float
        The share of its length to run.
    ramp, edge : bool
        Whether to keep the safety ramp, and the reads of the edge.

    Returns
    -------
    tuple
        The simulation, translated, and the seconds that took.
    """
    started = time.perf_counter()
    simulation = build(CASES[name].configuration("default", width, backend, fraction))
    if not ramp:
        simulation.settings = dataclasses.replace(
            simulation.settings, spatial_absorber=False
        )
    if not edge:
        # Keep the temporal monitor: the control is the edge's cost, not the whole check's.
        simulation._transverse_edge = lambda *_: None  # type: ignore[method-assign, assignment]  # noqa: SLF001
    simulation.translate()
    return simulation, time.perf_counter() - started


def _took(output: object) -> dict[str, Any]:
    """Return what a run's record says it took.

    Parameters
    ----------
    output : object
        What the run returned.

    Returns
    -------
    dict
        The march's logical calls, the source calls, and the edge's reads and status
        where it has them.
    """
    outcome = getattr(output, "outcome", None)
    execution = {} if outcome is None else dict(outcome.execution)
    monitor = execution.get("boundary_monitor") or {}
    edge = monitor.get("transverse") or {}
    return {
        "logical": dict(getattr(outcome, "logical_calls", {}) or {}),
        "calls": execution.get("source_evaluations"),
        "reads": edge.get("reads"),
        "status": edge.get("transverse_status"),
    }


def main(argv: list[str] | None = None) -> None:
    """Measure the four variants of each case and print a summary.

    Parameters
    ----------
    argv : list of str or None, optional
        The arguments, by default the process's.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, choices=sorted(CASES))
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--backend", default="native")
    parser.add_argument("--width", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args(argv)
    shuffled = random.Random(arguments.seed)  # noqa: S311 - the order, not a secret
    lines: list[dict[str, Any]] = [
        {
            "section": "record",
            "backend": arguments.backend,
            "width": arguments.width,
            "fraction": arguments.fraction,
            "rounds": arguments.rounds,
            "cpus": sorted(os.sched_getaffinity(0)),
            "load": os.getloadavg(),
            "threads": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "SOLPROP_THREADS")
            },
        }
    ]
    print(json.dumps(lines[0]), flush=True)  # noqa: T201
    with warnings.catch_warnings():
        # A case whose light reaches its edge warns on purpose: its status is reported.
        warnings.simplefilter("ignore", BoundaryWarning)
        for case in arguments.case:
            variants: dict[str, Simulation] = {}
            for variant, (ramp, edge) in VARIANTS.items():
                variants[variant], seconds = _variant(
                    case,
                    arguments.backend,
                    arguments.width,
                    arguments.fraction,
                    ramp,
                    edge,
                )
                line = {
                    "section": "construction",
                    "case": case,
                    "variant": variant,
                    "seconds": round(seconds, 4),
                }
                lines.append(line)
                print(json.dumps(line), flush=True)  # noqa: T201
            for index in range(arguments.rounds):
                order = list(variants)
                shuffled.shuffle(order)
                for variant in order:
                    started = time.perf_counter()
                    output = variants[variant].run(progress="none")
                    wall = time.perf_counter() - started
                    line = {
                        "section": "run",
                        "case": case,
                        "variant": variant,
                        "round": index,
                        "wall_s": round(wall, 4),
                        **_took(output),
                    }
                    lines.append(line)
                    print(json.dumps(line), flush=True)  # noqa: T201
    if arguments.out is not None:
        arguments.out.write_text(
            "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
        )
    _summary(lines, arguments.case)


def _summary(lines: list[dict[str, Any]], cases: list[str]) -> None:
    """Print each case's medians, with each variant against ``neither``.

    Parameters
    ----------
    lines : list of dict
        What was measured.
    cases : list of str
        The cases.
    """
    for case in cases:
        runs = [
            line
            for line in lines
            if line["section"] == "run" and line.get("case") == case
        ]
        walls = {
            variant: [line["wall_s"] for line in runs if line["variant"] == variant]
            for variant in VARIANTS
        }
        base = statistics.median(walls["neither"])
        parts = []
        for variant in VARIANTS:
            median = statistics.median(walls[variant])
            spread = statistics.median(abs(wall - median) for wall in walls[variant])
            taken = next(line for line in runs if line["variant"] == variant)
            parts.append(
                f"{variant} {median:.3f}s ±{spread:.3f} "
                f"({100 * (median / base - 1):+.1f}%) "
                f"calls {taken['calls']} reads {taken['reads']}"
            )
        print(f"{case}: " + "; ".join(parts), flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
