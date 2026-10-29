# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Interleave runtime policies, backends and short standard propagation windows."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from benchmarks.ab import arm_order
from benchmarks.records import atomic_record
from benchmarks.registry import CASES
from benchmarks.runner import (
    freeze_harness,
    launch_worker,
    plan_rows,
    prepared_python,
    receipt,
)
from benchmarks.system import THREADING, eligible_cpus, guard, thread_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--baseline-backend", default="native")
    parser.add_argument("--candidate-backend", default="cpunative")
    parser.add_argument("--threads", default="8,36")
    parser.add_argument("--policies", default=",".join(THREADING))
    parser.add_argument("--cases", default="rdw,ctc,argon_blueshift")
    parser.add_argument("--steppers", default="default,rk")
    parser.add_argument("--blas-threads", type=int)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--fraction", type=float, default=0.0025)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retain-fields", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    policies = args.policies.split(",")
    if not set(policies) <= THREADING.keys():
        parser.error("unknown runtime policy")
    if (
        args.rounds < 1
        or args.timeout <= 0
        or (args.blas_threads is not None and args.blas_threads < 1)
    ):
        parser.error("rounds, timeout, and BLAS threads must be positive")
    if args.out.exists():
        parser.error("use a new output directory")
    pythons = {
        "baseline": prepared_python(str(args.baseline_python)),
        "candidate": prepared_python(str(args.candidate_python)),
    }
    backends = {
        "baseline": args.baseline_backend,
        "candidate": args.candidate_backend,
    }
    rows, skipped = plan_rows(
        tuple(CASES[name] for name in args.cases.split(",")),
        tuple(args.steppers.split(",")),
        args.threads,
        max(int(n) for n in args.threads.split(",")),
        backend="native",
        fraction=args.fraction,
        device=False,
    )
    if skipped:
        parser.error(str(skipped))
    harness = args.out / "harness"
    atomic_record(
        args.out / "plan.json",
        dict(
            pythons=pythons,
            rows=rows,
            policies=policies,
            blas_threads=args.blas_threads,
            rounds=args.rounds,
            timeout_seconds=args.timeout,
            harness=freeze_harness(harness),
        ),
        immutable=True,
    )
    allocation = eligible_cpus()
    start = time.perf_counter()
    samples = []
    for round_index in range(args.rounds):
        configurations = [(policy, row) for policy in policies for row in rows]
        random.Random(2016 + round_index).shuffle(configurations)
        for policy, row in configurations:
            width = row["configuration"]["threads"]
            environment = thread_environment(width, dict(os.environ), policy)
            if args.blas_threads is not None:
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "SOLPROP_BLAS_THREADS",
                ):
                    environment[name] = str(args.blas_threads)
            for arm in arm_order(round_index):
                identity = f"r{round_index}-{policy}-{arm}-{row['id']}"
                quiet = guard(allocation)["quiet"]
                events, elapsed = launch_worker(
                    pythons[arm],
                    dict(
                        configuration=row["configuration"]
                        | dict(backend=backends[arm]),
                        repeats=1,
                        fields=str((args.out / "fields" / f"{identity}.npy").resolve())
                        if args.retain_fields and round_index == 0
                        else None,
                    ),
                    environment,
                    args.out / "raw" / identity,
                    args.timeout,
                    harness,
                )
                sample = dict(
                    receipt(events),
                    id=row["id"],
                    round=round_index,
                    policy=policy,
                    arm=arm,
                    quiet=quiet,
                    wall_seconds=elapsed,
                )
                samples.append(sample)
                atomic_record(args.out / "results.json", dict(samples=samples))
                if sample["status"] != "ok":
                    raise RuntimeError(f"{identity}: {sample.get('error')}")
                if sample["effective_threads"] != width:
                    raise RuntimeError(
                        f"{identity}: requested {width} threads, "
                        f"observed {sample['effective_threads']}"
                    )
                print(
                    json.dumps(
                        {
                            k: sample[k]
                            for k in ("id", "round", "policy", "arm", "status", "quiet")
                        }
                    ),
                    flush=True,
                )
    atomic_record(
        args.out / "results.json",
        dict(samples=samples, collection_seconds=time.perf_counter() - start),
    )


if __name__ == "__main__":
    main()
