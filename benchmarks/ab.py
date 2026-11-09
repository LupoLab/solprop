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
    calibrate_builds,
    freeze_harness,
    launch_worker,
    plan_rows,
    prepared_python,
    receipt,
    worker_calibration,
)
from benchmarks.system import eligible_cpus, guard, physical_cores, thread_environment
from benchmarks.track import BACKENDS, calibration_option, selection_options, steppers

__all__ = [
    "ARMS",
    "ab",
    "ab_options",
    "arm_configuration",
    "arm_environment",
    "arm_order",
    "field_difference",
    "same_work",
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
    parser.add_argument(
        "--baseline-backend",
        choices=BACKENDS,
        help="the baseline arm's backend; default: --backend",
    )
    parser.add_argument(
        "--candidate-backend",
        choices=BACKENDS,
        help="the candidate arm's backend; default: --backend",
    )
    calibration_option(parser)
    for arm in ARMS:
        parser.add_argument(
            f"--{arm}-env",
            action="append",
            default=[],
            metavar="NAME=VALUE",
            help=f"an environment variable for the {arm} arm alone, over the "
            "threading preset (repeatable); SOLPROP_CALIBRATION=off against =use "
            "compares the built-in table with a machine calibration (SEP 46)",
        )


def arm_environment(settings: list[str]) -> dict[str, str]:
    """Return an arm's own variables from ``NAME=VALUE`` settings."""
    environment: dict[str, str] = {}
    for setting in settings:
        name, separator, value = setting.partition("=")
        if not separator or not name:
            raise SuiteError(f"an arm's environment takes NAME=VALUE, got {setting!r}")
        environment[name] = value
    return environment


def arm_configuration(configuration: Record, backend: str) -> Record:
    """Return one arm's configuration: its backend, in double precision."""
    return {**configuration, "backend": backend, "precision": "double"}


def arm_order(round_index: int) -> tuple[str, ...]:
    """Alternate which build runs first, so drift does not favour either."""
    return ARMS if round_index % 2 == 0 else ARMS[::-1]


def field_difference(baseline: Path, candidate: Path) -> float | None:
    """Return the relative L2 difference of two builds' final fields.

    ``None`` when either build saved none, or when their grids differ (a build that
    changes the time window or transform size), where no difference is defined.
    """
    if not (baseline.exists() and candidate.exists()):
        return None
    import numpy as np  # noqa: PLC0415 - only after both workers saved fields

    reference = np.load(baseline)
    found = np.load(candidate)
    if reference.shape != found.shape:
        return None
    scale = float(np.linalg.norm(reference))
    difference = float(np.linalg.norm(found - reference))
    return difference / scale if scale else difference


def same_work(
    baseline: set[tuple[int | None, int | None]],
    candidate: set[tuple[int | None, int | None]],
) -> bool | None:
    """Whether both builds took one and the same (accepted, rejected) steps.

    ``None`` when a repetition did not count its steps (a group on ``Batch``, which
    watches no point's run): nothing was compared, which is not a match.
    """
    if any(None in steps for steps in baseline | candidate):
        return None
    return baseline == candidate and len(baseline) == 1


def summarise(rows: list[Record], samples: list[Record], fields: Path) -> list[Record]:
    """Pair each round's medians and summarise the candidate/baseline ratios."""
    result = []
    for row in rows:
        values = row["configuration"]
        mine = [item for item in samples if item["id"] == row["id"]]
        failures = [item for item in mine if item["status"] != "ok"]
        medians: dict[str, dict[int, float]] = {arm: {} for arm in ARMS}
        work: dict[str, set[tuple[int | None, int | None]]] = {
            arm: set() for arm in ARMS
        }
        counters: dict[str, Record | None] = dict.fromkeys(ARMS)
        for item in mine:
            if item["status"] == "ok":
                if counters[item["arm"]] is None and item.get("counted"):
                    counters[item["arm"]] = item["counted"].get("counters")
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
                "same_work": same_work(work["baseline"], work["candidate"]),
                "work": {arm: sorted(work[arm], key=str) for arm in ARMS},
                "field_difference": field_difference(
                    fields / "baseline" / f"{row['id']}.npy",
                    fields / "candidate" / f"{row['id']}.npy",
                ),
                "failures": [item.get("error") for item in failures],
                # Each arm's structural counters, from its first counted run
                # (--counters): what each build returned to Python for.
                "counters": counters,
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
            + {True: "same", False: "differs", None: "unchecked"}[row["same_work"]]
        )
        for error in row["failures"]:
            print(f"  failed: {error}")
        for arm in ARMS:
            found = (row.get("counters") or {}).get(arm)
            if found:
                returns = ", ".join(
                    f"{name} {count}" for name, count in found["native_returns"].items()
                )
                print(f"  {arm} returns to Python: {returns or 'none'}")
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
    backends = {
        "baseline": args.baseline_backend or args.backend,
        "candidate": args.candidate_backend or args.backend,
    }
    rows, skipped = plan_rows(
        select(tuple(args.case), tuple(args.set), all_cases=args.all),
        steppers(args.stepper),
        args.threads,
        args.production or physical_cores(),
        backend=args.backend,
        fraction=args.length,
        device=args.device,
        noise=args.noise,
        fft_size=args.fft_size,
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
    environments = {
        "baseline": arm_environment(args.baseline_env),
        "candidate": arm_environment(args.candidate_env),
    }
    harness = output / "harness"
    atomic_record(
        output / "plan.json",
        {
            "created": now(),
            "purpose": "exploratory A/B",
            "pythons": pythons,
            "backends": backends,
            "environments": environments,
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
    calibrate_builds(pythons, args.calibrate, output, environments)
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
                        "configuration": arm_configuration(
                            row["configuration"], backends[arm]
                        ),
                        "repeats": args.repeats,
                        "counters": args.counters,
                        "fields": str(
                            (output / "fields" / arm / f"{row['id']}.npy").resolve()
                        )
                        if round_index == 0
                        else None,
                    },
                    thread_environment(width, dict(os.environ), args.threading)
                    | worker_calibration(dict(os.environ) | environments[arm])
                    | environments[arm],
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
        same = (
            pythons["baseline"] == pythons["candidate"]
            and backends["baseline"] == backends["candidate"]
            and environments["baseline"] == environments["candidate"]
        )
        print(
            ("A/A control: the same interpreter for both arms. " if same else "")
            + (f"Results: {output.resolve()} ({time.perf_counter() - started:.0f} s)")
        )
    return 0 if all(item["status"] == "ok" for item in samples) else 1
