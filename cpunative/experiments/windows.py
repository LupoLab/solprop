# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare eight prescribed steps from identical launch and developed fields."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from benchmarks.builders import build
from benchmarks.registry import CASES
from benchmarks.worker import source_calls
from solprop.backends.registry import select
from solprop.stepper import FixedSchedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("rdw", "ctc", "argon_blueshift"), required=True
    )
    parser.add_argument(
        "--backend", choices=("numpy_ref", "native", "cpunative"), required=True
    )
    parser.add_argument("--stepper", choices=("default", "rk"), required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    case = CASES[args.case]
    simulation = build(
        dict(
            case=case.name,
            backend=args.backend,
            stepper=args.stepper,
            stop=case.length,
            planes=case.planes,
            fields="all",
        )
    )
    original = simulation.translate()
    stored = np.load(args.trajectory, allow_pickle=False)
    peak = stored["stat_peak_intensity"].reshape(len(stored["z"]), -1).max(axis=1)
    # Keep room after the chosen field: stages must stay within translated tables.
    planes = sorted({0, int(np.argmax(peak[:-1])), len(peak) - 2})
    fields, records = [], []
    for plane in planes:
        z = float(stored["z"][plane])
        dz = case.length * 1e-6
        target = z + 8 * dz
        # Retain the full domain's hard endpoints without an unexecuted giant
        # step that would violate the ordinary collar-stability admission check.
        spacing = min(original.stepper.dz_init, case.length * 1e-4)
        prefix = FixedSchedule.uniform(0.0, z, spacing).steps if z else np.empty(0)
        suffix = FixedSchedule.uniform(target, case.length, spacing).steps
        schedule = FixedSchedule(np.r_[prefix, np.full(8, dz), suffix])
        problem = dataclasses.replace(
            original,
            initial=dataclasses.replace(
                original.initial,
                field0=stored["field"][plane].reshape(original.shape),
                z=z,
            ),
            stepper=dataclasses.replace(
                original.stepper, schedule=schedule, save_planes=[target]
            ),
        )
        backend = select(
            args.backend,
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        try:
            backend.set_initial(problem.initial.field0, None, z=z)
            backend.propagate_to(target)
            backend.set_initial(problem.initial.field0, None, z=z)
            before = source_calls(backend)
            start = time.perf_counter()
            backend.propagate_to(target)
            elapsed = time.perf_counter() - start
            after = source_calls(backend)
            fields.append(backend.observe().require_field())
            records.append(
                dict(
                    case=args.case,
                    backend=args.backend,
                    stepper=args.stepper,
                    plane=plane,
                    z=z,
                    target=target,
                    steps=8,
                    dz=dz,
                    seconds=elapsed,
                    source_evaluations=None
                    if before is None or after is None
                    else after - before,
                )
            )
        finally:
            backend.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), planes=planes, fields=fields)
    args.out.with_suffix(".json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
