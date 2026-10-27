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
    "native",
    "native_build",
    "slug",
    "cpu_model",
    "instance_type",
    "system",
    "threads",
    "threading",
    "case",
    "revision",
    "stepper",
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
    "construction_s",
    "busy",
)


def append(path: Path, rows: Sequence[Record]) -> None:
    """Append rows, writing the header to a new file and refusing a changed one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    if not new:
        with path.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream), None)
        if tuple(header or ()) != COLUMNS:
            raise SuiteError(f"{path}: columns differ from this version's")
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(COLUMNS), lineterminator="\n")
        if new:
            writer.writeheader()
        for row in rows:
            if set(row) != set(COLUMNS):
                raise SuiteError("a history row must give exactly the declared columns")
            writer.writerow(
                {key: "" if row[key] is None else row[key] for key in COLUMNS}
            )


def read(path: Path) -> list[Record]:
    """Read a history file's rows as strings."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def comparable(row: Record) -> tuple[str, ...]:
    """Key the rows that measure the same thing on one hardware set."""
    return (
        str(row["threads"]),
        row["threading"],
        row["case"],
        str(row["revision"]),
        row["stepper"],
        row["integrator"],
    )


def summary(rows: Sequence[Record]) -> list[Record]:
    """Compare each recorded run with the first one, per width.

    A run is the rows sharing a date and commit. Its change is the geometric mean,
    over the configurations it shares with the first run at that width, of the
    ratio of median times; below one is faster.
    """
    runs: dict[tuple[str, str, str], dict[tuple[str, ...], float]] = {}
    labels: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = (str(row["threads"]), row["date"], row["commit"])
        runs.setdefault(key, {})[comparable(row)] = float(row["median_s"])
        labels[key] = row["solprop"]
    result = []
    first: dict[str, dict[tuple[str, ...], float]] = {}
    for (threads, date, commit), times in runs.items():
        reference = first.setdefault(threads, times)
        shared = sorted(set(times) & set(reference))
        ratios = [times[key] / reference[key] for key in shared]
        result.append(
            {
                "threads": threads,
                "date": date,
                "commit": commit,
                "solprop": labels[(threads, date, commit)],
                "cases": len(times),
                "shared": len(shared),
                "ratio": math.exp(statistics.fmean(map(math.log, ratios)))
                if ratios
                else None,
            }
        )
    return result


def show(path: Path, threads: str | None = None) -> None:
    """Print one line per recorded run and width."""
    print(f"{path.stem}")
    print(f"{'threads':>7}  {'date':25} {'solprop':26} {'cases':>5} {'vs first':>9}")
    for row in summary(read(path)):
        if threads is not None and row["threads"] != threads:
            continue
        ratio = "-" if row["ratio"] is None else f"{row['ratio']:.3f}"
        print(
            f"{row['threads']:>7}  {row['date']:25} {row['solprop']:26} "
            f"{row['cases']:>5} {ratio:>9}"
        )
    print("vs first: geometric mean of median-time ratios over shared cases.")
