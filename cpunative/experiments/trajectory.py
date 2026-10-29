# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Save a standard-case trajectory for independent accuracy and developed inputs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from benchmarks.builders import build
from benchmarks.registry import CASES
from solprop.backends.registry import select
from solprop.driver.run import run_problem
from solprop.output import MemoryOutput


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("rdw", "ctc", "argon_blueshift", "radial_filament_256"),
        required=True,
    )
    parser.add_argument(
        "--backend",
        choices=("numpy_ref", "numba_cpu", "native", "cpunative"),
        required=True,
    )
    parser.add_argument("--stepper", choices=("default", "rk"), required=True)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    case = CASES[args.case]
    sim = build(
        dict(
            case=case.name,
            backend=args.backend,
            stepper=args.stepper,
            stop=case.length,
            planes=case.planes,
            fields="all",
        )
    )
    if args.rtol is not None:
        sim.settings = dataclasses.replace(
            sim.settings,
            controller=dataclasses.replace(sim.settings.controller, rtol=args.rtol),
        )
    problem = sim.translate()
    started = time.perf_counter()
    backend = select(
        args.backend, problem, grid=sim.grid, modes=sim.modes, profiles=sim.profiles
    )
    construction = time.perf_counter() - started
    output = MemoryOutput()
    try:
        backend.set_initial(problem.initial.field0, None, z=problem.initial.z)
        started = time.perf_counter()
        run_problem(
            problem,
            backend,
            output,
            metadata=sim.metadata(),
            stats_spec=problem.stats_spec,
            progress="log",
            resume_history=False,
            fields="all",
            field_planes=(),
        )
        elapsed = time.perf_counter() - started
        arrays = dict(
            z=np.asarray(output["z"]),
            field=np.stack(
                [output.plane_field(i, "fft") for i in range(output.n_saved)]
            ),
        )
        for name in problem.stats_spec.names:
            arrays["stat_" + name] = np.asarray(output["stats/" + name])
        arrays["accepted"] = np.asarray(output["step_history/accepted"])
        arrays["h"] = np.asarray(output["step_history/h"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
        record = dict(
            case=args.case,
            backend=args.backend,
            stepper=args.stepper,
            rtol=problem.stepper.controller.rtol,
            atol=problem.stepper.controller.atol,
            seconds=elapsed,
            construction_seconds=construction,
            accepted=int(arrays["accepted"].sum()),
            rejected=int((~arrays["accepted"]).sum()),
            shape=list(problem.shape),
            n_t=problem.grids.n_t,
            n_to=problem.grids.n_to,
            metadata=backend.metadata(),
        )
        args.out.with_suffix(".json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {key: value for key, value in record.items() if key != "metadata"}
            ),
            flush=True,
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
