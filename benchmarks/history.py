# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep and summarise the per-hardware tracking history (SEP 38, section 5.3).

Each hardware set has one CSV under ``benchmarks/history/``, named by its slug, with
one row per run, case, stepper and width. Rows compare only within one hardware set,
width, case revision and resolved integrator.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

from benchmarks.records import Record, SuiteError
from benchmarks.runner import ROOT

__all__ = ["COLUMNS", "HISTORY", "append", "comparable", "read", "show", "summary"]

HISTORY = ROOT / "benchmarks" / "history"
COLUMNS = (
    "date",
    "solprop",
    "commit",
    "dirty",
    "native",
    "native_commit",
    "native_build",
    "slug",
    "cpu_model",
    "gpu_model",
    "instance_type",
    "system",
    "numa_domains",
    "threads",
    "threading",
    "case",
    "revision",
    "case_hash",
    "stepper",
    "precision",
    "integrator",
    "rtol",
    "atol",
    "noise",
    "median_s",
    "min_s",
    "max_s",
    "repeats",
    "accepted",
    "rejected",
    "source_evaluations",
    "peak_rss_gib",
    "device_gib",
    "construction_s",
    "busy",
)
"""History columns. New ones go anywhere; older files gain them, empty, on append."""


def append(path: Path, rows: Sequence[Record]) -> None:
    """Append rows, giving an older file any columns it lacks.

    A file whose header has a column this version does not know is refused: it
    was written by a later version.
    """
    for row in rows:
        if set(row) != set(COLUMNS):
            raise SuiteError("a history row must give exactly the declared columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[Record] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if unknown := set(reader.fieldnames or ()) - set(COLUMNS):
                raise SuiteError(
                    f"{path}: columns {', '.join(sorted(unknown))} are from a later "
                    "version of the benchmarks"
                )
            if tuple(reader.fieldnames or ()) != COLUMNS:
                existing = list(reader)
    mode = "w" if existing or not path.exists() else "a"
    with path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(COLUMNS), lineterminator="\n")
        if mode == "w":
            writer.writeheader()
        for row in (*existing, *rows):
            writer.writerow(
                {key: "" if row.get(key) is None else row[key] for key in COLUMNS}
            )


def read(path: Path) -> list[Record]:
    """Read a history file's rows as strings."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def comparable(row: Record) -> tuple[str, ...]:
    """Key the rows that measure the same thing on one hardware set.

    Two rows compare only with the same memory domains spanned, width, threading,
    case definition, stepper, precision, resolved integrator, tolerances and noise
    model: a changed library default is a new configuration, not a speed-up.
    """
    return tuple(
        str(row.get(key) or "")
        for key in (
            "numa_domains",
            "threads",
            "threading",
            "case",
            "revision",
            "stepper",
            "precision",
            "integrator",
            "rtol",
            "atol",
            "noise",
        )
    )


def _ratio(
    times: dict[tuple[str, ...], float], before: dict
) -> tuple[int, float | None]:
    shared = sorted(set(times) & set(before))
    ratios = [times[key] / before[key] for key in shared]
    mean = math.exp(statistics.fmean(map(math.log, ratios))) if ratios else None
    return len(shared), mean


def summary(rows: Sequence[Record]) -> list[Record]:
    """Compare each recorded run with the first and with the previous, per width.

    A run is the rows sharing a date and commit. Each change is the geometric
    mean, over the configurations the two runs share, of the ratio of median
    times; below one is faster. Busy rows count, and are counted.
    """
    runs: dict[tuple[str, str, str], dict[tuple[str, ...], float]] = {}
    labels: dict[tuple[str, str, str], tuple[str, int]] = {}
    for row in rows:
        key = (str(row["threads"]), row["date"], row["commit"])
        runs.setdefault(key, {})[comparable(row)] = float(row["median_s"])
        label, busy = labels.get(key, (row["solprop"], 0))
        labels[key] = (label, busy + int(str(row.get("busy") or "0") == "1"))
    result = []
    first: dict[str, dict[tuple[str, ...], float]] = {}
    previous: dict[str, dict[tuple[str, ...], float]] = {}
    for (threads, date, commit), times in runs.items():
        shared_first, versus_first = _ratio(times, first.setdefault(threads, times))
        shared_previous, versus_previous = _ratio(times, previous.get(threads, times))
        previous[threads] = times
        label, busy = labels[(threads, date, commit)]
        result.append(
            {
                "threads": threads,
                "date": date,
                "commit": commit,
                "solprop": label,
                "cases": len(times),
                "busy": busy,
                "shared_first": shared_first,
                "ratio_first": versus_first,
                "shared_previous": shared_previous,
                "ratio_previous": versus_previous,
            }
        )
    return result


def show(path: Path, threads: str | None = None) -> None:
    """Print one line per recorded run and width."""
    print(f"{path.stem}")
    print(
        f"{'threads':>7}  {'date':18} {'solprop':26} {'cases':>5} {'busy':>4} "
        f"{'vs first':>13} {'vs previous':>13}"
    )
    for row in summary(read(path)):
        if threads is not None and row["threads"] != threads:
            continue

        def change(ratio: float | None, shared: int) -> str:
            return "-" if ratio is None else f"{ratio:.3f} ({shared})"

        print(
            f"{row['threads']:>7}  {row['date']:18} {row['solprop']:26} "
            f"{row['cases']:>5} {row['busy']:>4} "
            f"{change(row['ratio_first'], row['shared_first']):>13} "
            f"{change(row['ratio_previous'], row['shared_previous']):>13}"
        )
    print(
        "Changes are geometric means of median-time ratios (below 1: faster) over "
        "the configurations both runs share, counted in brackets."
    )
