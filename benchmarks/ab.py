# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare two builds quickly: one worker per build per round, cases built once."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

from benchmarks.collection import freeze_harness, prepared_python
from benchmarks.planning import Selection, matrix
from benchmarks.records import (
    LOCAL_ROOT,
    Record,
    SuiteError,
    atomic_record,
    canonical,
    now,
)
from benchmarks.registry import SETS, TRACKS, select
from benchmarks.system import admission, guard, profile_for, thread_environment

__all__ = [
    "ARMS",
    "ab",
    "ab_options",
    "admitted",
    "arm_order",
    "field_difference",
    "launch_worker",
    "show_summary",
    "summarise",
]

ARMS = ("baseline", "candidate")


def ab_options(parser: argparse.ArgumentParser) -> None:
    """Declare the exploratory A/B selection, builds, rounds and repetitions."""
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--set", action="append", choices=SETS, default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("numpy_ref", "numba_cpu", "native", "julia"),
        default="native",
    )
    parser.add_argument("--threads")
    parser.add_argument("--track", action="append", choices=TRACKS, default=[])
    parser.add_argument("--machine-profile", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--baseline-python", required=True)
    parser.add_argument("--candidate-python", default=sys.executable)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--repeats", type=int, default=5, help="timed windows per case per worker"
    )
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--timeout", type=float, default=1800.0, help="worker limit, seconds"
    )
    parser.add_argument("--json", action="store_true")


def arm_order(round_index: int) -> tuple[str, ...]:
    """Alternate which build runs first, so drift does not favour either."""
    return ARMS if round_index % 2 == 0 else ARMS[::-1]


def admitted(plan: Record) -> tuple[list[Record], list[str]]:
    """Keep ready singleton rows that fit in memory; explain every other row."""
    rows: list[Record] = []
    skipped: list[str] = []
    for row in plan["rows"]:
        values = row["configuration"]
        label = f"{values['case']} {values['track']} {values['threads']} threads"
        if row["status"] != "ready":
            skipped.append(f"{label}: {row['reason']}")
        elif values["case"] == "batch":
            skipped.append(f"{label}: the batch exercise needs the standard runner")
        elif not (memory := admission(row["estimate_bytes"], None))["admitted"]:
            skipped.append(f"{label}: {memory['reason']}")
        else:
            rows.append(row)
    return rows, skipped


def launch_worker(
    python: str,
    request: Record,
    environment: dict[str, str],
    directory: Path,
    timeout: float,
) -> list[Record]:
    """Run one worker to completion and return its protocol events."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "request.json"
    atomic_record(path, request, immutable=True)
    harness = Path(request["harness_root"]).resolve()
    with (
        (directory / "stdout.log").open("w", encoding="utf-8") as stdout,
        (directory / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        try:
            subprocess.run(
                [python, "-m", "benchmarks.ab_worker", str(path.resolve())],
                cwd=harness,
                env={**environment, "PYTHONPATH": str(harness)},
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            stdout.write(canonical({"event": "failed", "error": "worker timeout"}))
    events = []
    for line in (directory / "stdout.log").read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and "event" in event:
            events.append(event)
    return events


def field_difference(baseline: Path, candidate: Path) -> float | None:
    """Return the relative L2 difference of two builds' retained fields."""
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
                "track": values["track"],
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
        f"\n{'case':22} {'track':20} {'N':>3} {'base / ms':>10} {'cand / ms':>10} "
        f"{'ratio':>7} {'range':>15} {'field diff':>10} work"
    )
    for row in summary:

        def ms(value: float | None) -> str:
            return "-" if value is None else f"{1e3 * value:.3f}"

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
            f"{row['case']:22} {row['track']:20} {row['threads']:3} "
            f"{ms(row['baseline_seconds']):>10} {ms(row['candidate_seconds']):>10} "
            f"{ratio:>7} {spread:>15} {field:>10} "
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
    """Collect and summarise a fast paired comparison of two prepared builds."""
    if args.rounds < 1 or args.repeats < 1 or args.timeout <= 0:
        raise SuiteError("rounds, repeats and timeout must be positive")
    started = time.perf_counter()
    cases = select(tuple(args.case), tuple(args.set), all_cases=args.all)
    profile = profile_for(args.machine_profile, standard=False)
    plan = matrix(
        cases,
        profile,
        Selection(
            "quick",
            args.backend,
            args.threads,
            tuple(args.track),
            None,
            args.diagnostic,
        ),
    )
    rows, skipped = admitted(plan)
    for reason in skipped:
        print(f"skipped {reason}")
    if not rows:
        raise SuiteError("no admitted configuration in the selection")
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
            "seed": args.seed,
            "profile": plan["profile"],
            "rows": rows,
            "skipped": skipped,
            "harness_files": freeze_harness(harness),
        },
        immutable=True,
    )
    samples: list[Record] = []
    provenance: dict[str, list[Record]] = {arm: [] for arm in ARMS}
    diagnostics: list[Record] = []
    widths = list(dict.fromkeys(row["configuration"]["threads"] for row in rows))
    for round_index in range(args.rounds):
        shuffled = list(rows)
        random.Random(args.seed + round_index).shuffle(shuffled)
        for width in widths:
            selected = [
                row for row in shuffled if row["configuration"]["threads"] == width
            ]
            for arm in arm_order(round_index):
                quiet = guard(tuple(profile.allocation), standard=False)
                request = {
                    "rows": selected,
                    "repeats": args.repeats,
                    "harness_root": str(harness.resolve()),
                    "fields": str((output / "fields" / arm).resolve())
                    if round_index == 0
                    else None,
                }
                launched = time.perf_counter()
                events = launch_worker(
                    pythons[arm],
                    request,
                    thread_environment(width, profile.policy, dict(os.environ)),
                    output / "raw" / f"r{round_index}-t{width}-{arm}",
                    args.timeout,
                )
                elapsed = time.perf_counter() - launched
                received = {e["id"]: e for e in events if e["event"] == "row"}
                failure = next(
                    (e["error"] for e in events if e["event"] == "failed"),
                    "no receipt from the worker",
                )
                provenance[arm].extend(
                    e["provenance"] for e in events if e["event"] == "provenance"
                )
                diagnostics.extend(
                    {"arm": arm, "round": round_index, **item}
                    for e in events
                    if e["event"] == "diagnostics"
                    for item in e["diagnostics"]
                )
                for row in selected:
                    event = received.get(
                        row["id"], {"status": "failed", "error": failure}
                    )
                    samples.append(
                        {
                            **{k: v for k, v in event.items() if k != "event"},
                            "id": row["id"],
                            "arm": arm,
                            "round": round_index,
                            "quiet": quiet["quiet"],
                        }
                    )
                done = sum(row["id"] in received for row in selected)
                print(
                    f"round {round_index + 1}/{args.rounds}, {width} threads, "
                    f"{arm}: {done}/{len(selected)} cases in {elapsed:.1f} s"
                    + ("" if quiet["quiet"] else " (machine busy at launch)"),
                    flush=True,
                )
    summary = summarise(rows, samples, output / "fields")
    atomic_record(
        output / "results.json",
        {
            "samples": samples,
            "summary": summary,
            "provenance": provenance,
            "diagnostics": diagnostics,
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
