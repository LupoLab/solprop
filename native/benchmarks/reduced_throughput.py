# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Throughput of a fibre run on the native backend three ways (solprop SEP 36 6.6).

One soliton fibre on a fixed schedule, at ``2**k`` samples for each requested ``k``,
run

- solo, one state in one process;
- as a lockstep batch of ``--batch`` members in one process (``select_batch``);
- as ``--processes`` concurrent processes, each running the case solo.

Each reports samples times steps per second, aggregated over the members or the
processes: ``members * n_samples * steps / seconds``, the seconds those of the march
alone (construction and a warm-up run are excluded; concurrent processes from the
first start to the last finish). The median of ``--repeats`` timed runs is kept.

Run it on a device build to measure the device; on a CPU build it measures the CPU.
Thread budgets are the caller's (the processes inherit the environment).

usage::

    python native/benchmarks/reduced_throughput.py --precision single \\
        --exponents 13 14 15 16 --steps 64 --batch 8 --processes 1 2 4 8 \\
        --json throughput.json
"""

# ruff: noqa: I001 -- Kokkos must load before solprop can import pyFFTW on macOS.

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any, cast

import solprop_native  # noqa: F401 - loads Kokkos first

from solprop.backends.protocol import Backend
from solprop.backends.registry import select, select_batch
from solprop.driver import Simulation
from solprop.stepper import RungeKutta
from solprop.stepper.schedule import FixedSchedule

# The case: a 1024-sample soliton fibre's window, scaled by a power of two in time.
_BASE_EXPONENT = 10
_BASE_TRANGE = 1e-12
_LENGTH = 0.05


def case(precision: str, exponent: int, steps: int) -> Simulation:
    """Return the fibre at ``2**exponent`` samples on ``steps`` uniform steps."""
    # Slightly under the scaled window, so that the power-of-two rounding lands on it.
    trange = _BASE_TRANGE * 2.0 ** (exponent - _BASE_EXPONENT) * 0.99
    return Simulation.from_gnlse(
        0.11, _LENGTH, [0.0, 0.0, -1.183e-26, 8.1e-41], lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6), trange=trange, width=50e-15,
        peak_power=100.0, pulseshape="sech", save_planes=[_LENGTH],
        backend="native", precision=precision,  # pyright: ignore[reportArgumentType]
        integrator=RungeKutta(),
        schedule=FixedSchedule.uniform(0.0, _LENGTH, _LENGTH / steps),
    )  # fmt: skip


def loaded(simulation: Simulation, cleanup: ExitStack) -> Backend:
    """Return a native state with the launch loaded."""
    problem = simulation.translate()
    backend = select("native", problem, grid=simulation.grid)
    cleanup.callback(backend.close)
    backend.set_initial(problem.initial.field0, None)
    return backend


def timed_solo(simulation: Simulation) -> float:
    """Return one solo march's seconds, after an untimed warm-up march."""
    with ExitStack() as cleanup:
        loaded(simulation, cleanup).propagate_to(_LENGTH)
    with ExitStack() as cleanup:
        backend = loaded(simulation, cleanup)
        started = time.perf_counter()
        backend.propagate_to(_LENGTH)
        return time.perf_counter() - started


def timed_batch(simulation: Simulation, members: int) -> float:
    """Return one lockstep batch march's seconds, after a warm-up march."""
    for warm in (True, False):
        with ExitStack() as cleanup:
            owners = [loaded(simulation, cleanup) for _ in range(members)]
            runner = select_batch(cast(Sequence[Backend], owners))
            cleanup.callback(runner.close)
            started = time.perf_counter()
            runner.propagate_to(_LENGTH)
            seconds = time.perf_counter() - started
        if not warm:
            return seconds
    raise AssertionError("unreachable")


def worker(precision: str, exponent: int, steps: int) -> None:
    """One concurrent process: report its march's start and end on stdout."""
    simulation = case(precision, exponent, steps)
    with ExitStack() as cleanup:
        loaded(simulation, cleanup).propagate_to(_LENGTH)
    with ExitStack() as cleanup:
        backend = loaded(simulation, cleanup)
        started = time.time()
        backend.propagate_to(_LENGTH)
        finished = time.time()
    print(json.dumps({"start": started, "end": finished}))


def timed_processes(precision: str, exponent: int, steps: int, count: int) -> float:
    """Return the span from the first process's march start to the last's end."""
    command = [
        sys.executable, __file__, "--worker", "--precision", precision,
        "--exponents", str(exponent), "--steps", str(steps),
    ]  # fmt: skip
    processes = [
        subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        for _ in range(count)
    ]
    spans = []
    for process in processes:
        output, _ = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"a worker failed with {process.returncode}")
        spans.append(json.loads(output.strip().splitlines()[-1]))
    return max(span["end"] for span in spans) - min(span["start"] for span in spans)


def report(
    arguments: argparse.Namespace,
    exponent: int,
    samples: int,
    mode: str,
    members: int,
    seconds: list[float],
) -> dict[str, Any]:
    """Print and return one measurement's row: the median and its throughput."""
    median = statistics.median(seconds)
    throughput = members * samples * arguments.steps / median
    print(
        f"{arguments.precision:6s} 2^{exponent} ({samples}) {mode:9s} x{members:<2d} "
        f"{median * 1e3:9.1f} ms {throughput:.3e} samples*steps/s",
        flush=True,
    )
    return {
        "precision": arguments.precision, "exponent": exponent, "samples": samples,
        "steps": arguments.steps, "mode": mode, "members": members,
        "seconds": median, "samples_steps_per_second": throughput,
    }  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--precision", default="single",
                        choices=("double", "single", "mixed"))  # fmt: skip
    parser.add_argument("--exponents", type=int, nargs="+", default=[13, 14, 15, 16])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--processes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments.precision, arguments.exponents[0], arguments.steps)
        return
    results: list[dict[str, Any]] = []
    for exponent in arguments.exponents:
        simulation = case(arguments.precision, exponent, arguments.steps)
        samples = simulation.grid.n_t

        def record(
            mode: str,
            members: int,
            seconds: list[float],
            *,
            exponent: int = exponent,
            samples: int = samples,
        ) -> None:
            results.append(report(arguments, exponent, samples, mode, members, seconds))

        repeats = range(arguments.repeats)
        record("solo", 1, [timed_solo(simulation) for _ in repeats])
        record("batch", arguments.batch,
               [timed_batch(simulation, arguments.batch) for _ in repeats])  # fmt: skip
        for count in arguments.processes:
            record(
                "processes",
                count,
                [
                    timed_processes(
                        arguments.precision, exponent, arguments.steps, count
                    )
                    for _ in repeats
                ],
            )
    if arguments.json:
        with open(arguments.json, "w", encoding="utf-8") as stream:
            json.dump(results, stream, indent=2)


if __name__ == "__main__":
    main()
