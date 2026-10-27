# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare two builds quickly: alternating workers per configuration and round."""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
import time
import uuid
from pathlib import Path

from benchmarks.records import (
    LOCAL_ROOT,
    Record,
    SuiteError,
    atomic_record,
    canonical,
    now,
)
from benchmarks.registry import select
from benchmarks.runner import (
    freeze_harness,
    launch_worker,
    plan_rows,
    prepared_python,
    receipt,
)
from benchmarks.system import eligible_cpus, guard, physical_cores, thread_environment
from benchmarks.track import selection_options, steppers

__all__ = [
    "ARMS",
    "ab",
    "ab_options",
    "arm_order",
    "field_difference",
    "show_summary",
    "summarise",
]

ARMS = ("baseline", "candidate")


def ab_options(parser: argparse.ArgumentParser) -> None:
    """Declare the A/B selection, builds, length, rounds and repetitions."""
    selection_options(parser)
    parser.add_argument("--baseline-python", required=True)
    parser.add_argument("--candidate-python", default=sys.executable)
    parser.add_argument(
        "--length",
        type=float,
        default=1.0,
        help="fraction of each guide to run, for quick iteration",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=30, help="shuffles the case order")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--timeout", type=float, default=3600.0, help="per-worker limit, seconds"
    )
    parser.add_argument("--json", action="store_true")


def arm_order(round_index: int) -> tuple[str, ...]:
    """Alternate which build runs first, so drift does not favour either."""
    return ARMS if round_index % 2 == 0 else ARMS[::-1]


def field_difference(baseline: Path, candidate: Path) -> float | None:
    """Return the relative L2 difference of two builds' final fields."""
    if not (baseline.exists() and candidate.exists()):
        return None
    import numpy as np  # noqa: PLC0415 - only after both workers saved fields

    reference = np.load(baseline)
    found = np.load(candidate)
    if reference.shape != found.shape:
        return math.inf
    scale = float(np.linalg.norm(reference))
    difference = float(np.linalg.norm(found - reference))
    return difference / scale if scale else difference


def summarise(rows: list[Record], samples: list[Record], fields: Path) -> list[Record]:
    """Pair each round's medians and summarise the candidate/baseline ratios."""
    result = []
    for row in rows:
        values = row["configuration"]
        mine = [item for item in samples if item["id"] == row["id"]]
        failures = [item for item in mine if item["status"] != "ok"]
        medians: dict[str, dict[int, float]] = {arm: {} for arm in ARMS}
        work: dict[str, set[tuple[int, int]]] = {arm: set() for arm in ARMS}
        for item in mine:
            if item["status"] == "ok":
                times = [rep["seconds"] for rep in item["repetitions"]]
                medians[item["arm"]][item["round"]] = statistics.median(times)
                work[item["arm"]].update(
                    (rep["accepted"], rep["rejected"]) for rep in item["repetitions"]
                )
        rounds = sorted(set(medians["baseline"]) & set(medians["candidate"]))
        ratios = [medians["candidate"][i] / medians["baseline"][i] for i in rounds]
        result.append(
            {
                "id": row["id"],
                "case": values["case"],
                "stepper": values["stepper"],
                "threads": values["threads"],
                "rounds": len(rounds),
                "baseline_seconds": statistics.median(medians["baseline"].values())
                if medians["baseline"]
                else None,
                "candidate_seconds": statistics.median(medians["candidate"].values())
                if medians["candidate"]
                else None,
                "ratio": statistics.median(ratios) if ratios else None,
                "ratio_min": min(ratios, default=None),
                "ratio_max": max(ratios, default=None),
                "ratios": ratios,
                "same_work": work["baseline"] == work["candidate"]
                and len(work["baseline"]) == 1,
                "work": {arm: sorted(work[arm]) for arm in ARMS},
                "field_difference": field_difference(
                    fields / "baseline" / f"{row['id']}.npy",
                    fields / "candidate" / f"{row['id']}.npy",
                ),
                "failures": [item.get("error") for item in failures],
            }
        )
    return result


def show_summary(summary: list[Record]) -> None:
    """Print one line per configuration; the ratio is exploratory, not a verdict."""
    print(
        f"\n{'case':26} {'stepper':12} {'N':>3} {'base / s':>9} {'cand / s':>9} "
        f"{'ratio':>7} {'range':>13} {'field diff':>10} work"
    )
    for row in summary:

        def seconds(value: float | None) -> str:
            return "-" if value is None else f"{value:.3f}"

        ratio = "-" if row["ratio"] is None else f"{row['ratio']:.4f}"
        spread = (
            "-"
            if row["ratio"] is None
            else f"{row['ratio_min']:.3f}-{row['ratio_max']:.3f}"
        )
        field = (
            "-" if row["field_difference"] is None else f"{row['field_difference']:.1e}"
        )
        print(
            f"{row['case']:26} {row['stepper']:12} {row['threads']:3} "
            f"{seconds(row['baseline_seconds']):>9} "
            f"{seconds(row['candidate_seconds']):>9} "
            f"{ratio:>7} {spread:>13} {field:>10} "
            f"{'same' if row['same_work'] else 'differs'}"
        )
        for error in row["failures"]:
            print(f"  failed: {error}")
    ratios = [row["ratio"] for row in summary if row["ratio"] is not None]
    if ratios:
        mean = math.exp(statistics.fmean(math.log(value) for value in ratios))
        print(f"Geometric mean candidate/baseline: {mean:.4f}")
    print(
        "Ratios are medians of per-round paired medians (below 1: candidate faster); "
        "range is over rounds. Exploratory: no adoption verdict."
    )


def ab(args: argparse.Namespace) -> int:
    """Collect and summarise a paired comparison of two prepared builds."""
    if args.rounds < 1 or args.repeats < 1 or args.timeout <= 0:
        raise SuiteError("rounds, repeats and timeout must be positive")
    started = time.perf_counter()
    rows, skipped = plan_rows(
        select(tuple(args.case), tuple(args.set), all_cases=args.all),
        steppers(args.stepper),
        args.threads,
        args.production or physical_cores(),
        backend=args.backend,
        fraction=args.length,
    )
    for reason in skipped:
        print(f"skipped {reason}")
    if not rows:
        raise SuiteError("no configuration in the selection")
    output: Path = args.out or LOCAL_ROOT / "ab" / uuid.uuid4().hex
    if output.exists() and any(output.iterdir()):
        raise SuiteError("output already exists; choose a new --out")
    pythons = {
        "baseline": prepared_python(args.baseline_python),
        "candidate": prepared_python(args.candidate_python),
    }
    harness = output / "harness"
    atomic_record(
        output / "plan.json",
        {
            "created": now(),
            "purpose": "exploratory A/B",
            "pythons": pythons,
            "rounds": args.rounds,
            "repeats": args.repeats,
            "length": args.length,
            "threading": args.threading,
            "seed": args.seed,
            "rows": rows,
            "skipped": skipped,
            "harness_files": freeze_harness(harness),
        },
        immutable=True,
    )
    samples: list[Record] = []
    provenance: dict[str, list[Record]] = {arm: [] for arm in ARMS}
    allocation = eligible_cpus()
    for round_index in range(args.rounds):
        shuffled = list(rows)
        random.Random(args.seed + round_index).shuffle(shuffled)
        shuffled.sort(key=lambda row: row["configuration"]["threads"])
        for row in shuffled:
            width = row["configuration"]["threads"]
            for arm in arm_order(round_index):
                quiet = guard(allocation)["quiet"]
                events, elapsed = launch_worker(
                    pythons[arm],
                    {
                        "configuration": row["configuration"],
                        "repeats": args.repeats,
                        "fields": str(
                            (output / "fields" / arm / f"{row['id']}.npy").resolve()
                        )
                        if round_index == 0
                        else None,
                    },
                    thread_environment(width, dict(os.environ), args.threading),
                    output / "raw" / f"r{round_index}-{arm}-{row['id']}",
                    args.timeout,
                    harness,
                )
                result = receipt(events)
                if result.get("provenance"):
                    provenance[arm].append(result["provenance"])
                samples.append(
                    {
                        **{k: v for k, v in result.items() if k != "provenance"},
                        "id": row["id"],
                        "arm": arm,
                        "round": round_index,
                        "quiet": quiet,
                    }
                )
                print(
                    f"round {round_index + 1}/{args.rounds} {row['id']} {arm}: "
                    f"{result['status']} in {elapsed:.1f} s"
                    + ("" if quiet else " (machine busy at launch)"),
                    flush=True,
                )
    summary = summarise(rows, samples, output / "fields")
    atomic_record(
        output / "results.json",
        {
            "samples": samples,
            "summary": summary,
            "provenance": provenance,
            "collection_seconds": time.perf_counter() - started,
            "busy_samples": sum(not item["quiet"] for item in samples),
        },
    )
    if args.json:
        print(canonical(summary))
    else:
        show_summary(summary)
        same = pythons["baseline"] == pythons["candidate"]
        print(
            ("A/A control: the same interpreter for both arms. " if same else "")
            + (f"Results: {output.resolve()} ({time.perf_counter() - started:.0f} s)")
        )
    return 0 if all(item["status"] == "ok" for item in samples) else 1
