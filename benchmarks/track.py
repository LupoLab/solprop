# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Time the tracking set's complete runs and record them per hardware (SEP 38)."""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from benchmarks import history
from benchmarks.records import LOCAL_ROOT, Record, SuiteError, atomic_record, now
from benchmarks.registry import SETS, STEPPERS, select
from benchmarks.runner import (
    ROOT,
    launch_worker,
    plan_rows,
    receipt,
    show_plan,
)
from benchmarks.system import (
    THREADING,
    eligible_cpus,
    git_receipt,
    guard,
    hardware,
    physical_cores,
    thread_environment,
)

__all__ = [
    "history_rows",
    "selection_options",
    "show_results",
    "steppers",
    "summarise",
    "track",
    "track_options",
]


def selection_options(parser: argparse.ArgumentParser) -> None:
    """Declare the case, stepper and width selection shared by every command."""
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--set", action="append", choices=SETS, default=[])
    parser.add_argument("--all", action="store_true", help="every registered case")
    parser.add_argument(
        "--stepper",
        default="default",
        help="default, multistep, multistep-interaction or rk; comma-separated to "
        "compare steppers in one run",
    )
    parser.add_argument(
        "--threads", help="widths, e.g. 1,production; default: each case's own"
    )
    parser.add_argument(
        "--production", type=int, help="production width; default: physical cores"
    )
    parser.add_argument(
        "--backend", choices=("numpy_ref", "numba_cpu", "native"), default="native"
    )
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per case")
    parser.add_argument(
        "--threading",
        choices=tuple(THREADING),
        default="passive",
        help="OpenMP placement and waiting (benchmarks/system.py)",
    )


def track_options(parser: argparse.ArgumentParser) -> None:
    """Declare the tracking run's output and history options."""
    selection_options(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--instance-type", help="cloud instance type, if any")
    parser.add_argument(
        "--append-history",
        action="store_true",
        help="add the run to benchmarks/history/<hardware>.csv",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="with --append-history, accept uncommitted changes",
    )
    parser.add_argument("--archive", type=Path, help="copy the run's bundle here")
    parser.add_argument(
        "--timeout", type=float, default=3600.0, help="per-worker limit, seconds"
    )


def steppers(value: str) -> tuple[str, ...]:
    """Parse a comma-separated stepper list."""
    names = tuple(dict.fromkeys(item.strip() for item in value.split(",")))
    if unknown := [name for name in names if name not in STEPPERS]:
        raise SuiteError(f"unknown stepper: {', '.join(unknown)}")
    return names


def summarise(
    row: Record, result: Record, busy: bool, threading: str = "passive"
) -> Record:
    """Reduce one worker's receipt to the tracked statistics."""
    values = row["configuration"]
    base = {
        "id": row["id"],
        "case": values["case"],
        "revision": values["revision"],
        "stepper": values["stepper"],
        "threads": values["threads"],
        "threading": threading,
        "status": result["status"],
        "busy": busy,
    }
    if result["status"] != "ok":
        return {**base, "error": result.get("error")}
    times = [item["seconds"] for item in result["repetitions"]]
    last = result["repetitions"][-1]
    return {
        **base,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "repeats": len(times),
        "accepted": last["accepted"],
        "rejected": last["rejected"],
        "same_steps": len(
            {(r["accepted"], r["rejected"]) for r in result["repetitions"]}
        )
        == 1,
        "source_evaluations": last["source_evaluations"],
        "peak_rss_gib": result["peak_rss_bytes"] / 2**30,
        "construction_s": result["construction_seconds"] + result["backend_seconds"],
        "warmup_s": result["warmup_seconds"],
        "integrator": result["integrator"],
        "rtol": result["rtol"],
        "atol": result["atol"],
        "noise": result["noise"],
        "finite": result["finite"],
    }


def show_results(rows: list[Record]) -> None:
    """Print one line per configuration."""
    print(
        f"\n{'case':26} {'integrator':28} {'N':>3} {'median s':>9} "
        f"{'range %':>8} {'steps':>7} {'GiB':>6}"
    )
    for row in rows:
        if row["status"] != "ok":
            print(
                f"{row['case']:26} {'':28} {row['threads']:>3} failed: {row['error']}"
            )
            continue
        spread = 100 * (row["max_s"] - row["min_s"]) / row["median_s"]
        print(
            f"{row['case']:26} {row['integrator']:28} {row['threads']:>3} "
            f"{row['median_s']:9.2f} {spread:8.1f} {row['accepted']:7d} "
            f"{row['peak_rss_gib']:6.2f}"
            + ("" if row["same_steps"] else "  (steps differ between repeats)")
            + ("  (busy)" if row["busy"] else "")
        )


def history_rows(
    rows: list[Record], machine: Record, source: Record, provenance: Record
) -> list[Record]:
    """Shape a complete run's results as history rows."""
    build = provenance.get("build_info") or {}
    date = datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ")
    return [
        {
            "date": date,
            "solprop": provenance["packages"]["solprop"],
            "commit": (source.get("revision") or "")[:12],
            "native": build.get("package_version"),
            "native_build": "; ".join(
                str(build[key])
                for key in ("build_profile", "compiler", "fft_backend", "blas_backend")
                if build.get(key)
            )
            or None,
            "slug": machine["slug"],
            "cpu_model": machine["cpu_model"],
            "instance_type": machine["instance_type"],
            "system": f"{machine['system']} {machine['machine']}",
            "threads": row["threads"],
            "threading": row["threading"],
            "case": row["case"],
            "revision": row["revision"],
            "stepper": row["stepper"],
            "integrator": row["integrator"],
            "rtol": row["rtol"],
            "atol": row["atol"],
            "noise": row["noise"],
            "median_s": f"{row['median_s']:.4f}",
            "min_s": f"{row['min_s']:.4f}",
            "max_s": f"{row['max_s']:.4f}",
            "repeats": row["repeats"],
            "accepted": row["accepted"],
            "rejected": row["rejected"],
            "source_evaluations": row["source_evaluations"],
            "peak_rss_gib": f"{row['peak_rss_gib']:.3f}",
            "construction_s": f"{row['construction_s']:.2f}",
            "busy": int(row["busy"]),
        }
        for row in rows
    ]


def track(args: argparse.Namespace) -> int:
    """Run the selected configurations, one worker each, and record the results."""
    if args.repeats < 1 or args.timeout <= 0:
        raise SuiteError("repeats and timeout must be positive")
    production = args.production or physical_cores()
    rows, skipped = plan_rows(
        select(tuple(args.case), tuple(args.set), all_cases=args.all),
        steppers(args.stepper),
        args.threads,
        production,
        backend=args.backend,
    )
    if not rows:
        raise SuiteError("no configuration in the selection")
    show_plan(rows, skipped, args.repeats)
    if args.dry_run:
        return 0
    machine = hardware(production, args.instance_type)
    source = git_receipt(ROOT)
    if args.append_history:
        if args.backend != "native":
            raise SuiteError("the history records the native backend only")
        if source.get("dirty") and not args.allow_dirty:
            raise SuiteError(
                "uncommitted changes: commit them, or pass --allow-dirty to record "
                "this run anyway"
            )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output: Path = args.out or LOCAL_ROOT / "track" / f"{stamp}-{machine['slug']}"
    if output.exists() and any(output.iterdir()):
        raise SuiteError("output already exists; choose a new --out")
    atomic_record(
        output / "plan.json",
        {
            "created": now(),
            "hardware": machine,
            "source": source,
            "repeats": args.repeats,
            "threading": args.threading,
            "rows": rows,
            "skipped": skipped,
        },
        immutable=True,
    )
    started = time.perf_counter()
    results: list[Record] = []
    provenance: Record = {}
    allocation = eligible_cpus()
    for index, row in enumerate(rows):
        busy = not guard(allocation)["quiet"]
        width = row["configuration"]["threads"]
        events, elapsed = launch_worker(
            sys.executable,
            {"configuration": row["configuration"], "repeats": args.repeats},
            thread_environment(width, dict(os.environ), args.threading),
            output / "raw" / row["id"],
            args.timeout,
        )
        result = receipt(events)
        provenance = result.get("provenance") or provenance
        results.append(summarise(row, result, busy, args.threading))
        state = (
            f"{results[-1]['median_s']:.2f} s"
            if result["status"] == "ok"
            else result["status"]
        )
        print(
            f"[{index + 1}/{len(rows)}] {row['id']}: {state} "
            f"({elapsed:.0f} s with construction)" + (" busy" if busy else ""),
            flush=True,
        )
    atomic_record(
        output / "results.json",
        {
            "rows": results,
            "provenance": provenance,
            "collection_seconds": time.perf_counter() - started,
        },
    )
    show_results(results)
    complete = all(row["status"] == "ok" and row["finite"] for row in results)
    print(
        f"Results: {output.resolve()} ({(time.perf_counter() - started) / 60:.0f} min)"
    )
    if args.append_history:
        if not complete:
            print("Not added to the history: a configuration failed.", file=sys.stderr)
            return 1
        path = history.HISTORY / f"{machine['slug']}.csv"
        history.append(path, history_rows(results, machine, source, provenance))
        print(f"History: {path}")
    if args.archive is not None:
        target = args.archive / output.name
        shutil.copytree(output, target)
        print(f"Archived: {target}")
    return 0 if complete else 1
