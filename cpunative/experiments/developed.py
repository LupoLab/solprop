# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure matched source calls on stored standard-case fields."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("rdw", "ctc", "argon_blueshift", "radial_filament_256"),
        required=True,
    )
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--expected-threads", type=int)
    parser.add_argument("--chunk", type=int)
    args = parser.parse_args()
    cpus = tuple(sorted(os.sched_getaffinity(0))) if sys.platform == "linux" else ()
    if args.extension is not None:
        spec = importlib.util.spec_from_file_location(
            "solprop_cpunative._core", args.extension
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        import solprop_cpunative

        # Injecting an experimental extension loads OpenMP before the package can
        # capture its CPU grant. Supply the same pre-import record as __init__.
        solprop_cpunative.PROCESS_CPUS = cpus
    import numpy as np
    from benchmarks.builders import build, capillary
    from benchmarks.registry import CASES

    from solprop_cpunative.backend import CPUNativeBackend, _stage

    case = CASES[args.case]
    simulation = (
        build(
            dict(
                case=case.name,
                backend="cpunative",
                stepper="default",
                stop=case.length,
                planes=case.planes,
                fields="planes",
            )
        )
        if case.builder == "spatial"
        else capillary(args.case)
    )
    problem = simulation.translate()
    if args.chunk is not None:
        problem = dataclasses.replace(
            problem,
            transform=dataclasses.replace(problem.transform, node_chunk=args.chunk),
        )
    if args.trajectory is None:
        stored = {"z": [0.0], "field": problem.initial.field0[None]}
        planes = [0]
    else:
        stored = np.load(args.trajectory, allow_pickle=False)
        peak = (
            np.asarray(stored["stat_peak_intensity"])
            .reshape(len(stored["z"]), -1)
            .max(axis=1)
        )
        planes = sorted({0, int(np.argmax(peak)), len(peak) - 1})
    backend = CPUNativeBackend(problem)
    try:
        if args.expected_threads is not None:
            assert backend._cpu.threads == args.expected_threads, (
                backend._cpu.threads,
                args.expected_threads,
            )
        for plane in planes:
            z = float(stored["z"][plane])
            field = stored["field"][plane].reshape(problem.shape)
            backend.set_initial(field, None, z=z)
            stage = _stage(backend._stage_scalars(z, z))
            for _ in range(5):
                backend._cpu.evaluate(stage)
            before = np.asarray(backend._cpu.timings)
            response_before = np.asarray(backend._cpu.response_timings)
            start = time.perf_counter()
            for _ in range(args.calls):
                result = backend._cpu.evaluate(stage)
            elapsed = time.perf_counter() - start
            assert np.isfinite(result).all()
            phases = dict(
                zip(
                    backend.metadata()["kernel_seconds"],
                    (np.asarray(backend._cpu.timings) - before) / args.calls,
                    strict=True,
                )
            )
            print(
                json.dumps(
                    dict(
                        case=args.case,
                        plane=plane,
                        z=z,
                        threads=backend._cpu.threads,
                        chunk=problem.transform.node_chunk,
                        workspace_bytes=sum(backend.memory_ledger().values()),
                        build=backend.metadata()["cpunative"],
                        seconds=elapsed / args.calls,
                        phases=phases,
                        response_work=dict(
                            zip(
                                backend.metadata()["response_work_seconds"],
                                (
                                    np.asarray(backend._cpu.response_timings)
                                    - response_before
                                )
                                / args.calls,
                                strict=True,
                            )
                        ),
                    )
                ),
                flush=True,
            )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
