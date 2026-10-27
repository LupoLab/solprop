# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Expose the benchmark commands without importing scientific libraries (SEP 38)."""

from __future__ import annotations

import argparse
import sys

from benchmarks import history
from benchmarks.ab import ab, ab_options
from benchmarks.records import SuiteError
from benchmarks.registry import select
from benchmarks.runner import plan_rows, show_plan
from benchmarks.system import physical_cores
from benchmarks.track import selection_options, steppers, track, track_options

__all__ = ["main", "parser_for"]


def parser_for() -> argparse.ArgumentParser:
    """Declare the list, track, ab and history commands."""
    parser = argparse.ArgumentParser(
        description="SEP 38: track complete runs per hardware, or A/B two builds."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    selection_options(
        commands.add_parser("list", help="print the plan and its estimated time")
    )
    track_options(
        commands.add_parser("track", help="time complete runs; record per hardware")
    )
    ab_options(commands.add_parser("ab", help="compare two builds; exploratory"))
    recorded = commands.add_parser("history", help="summarise recorded runs")
    recorded.add_argument("slug", nargs="?", help="hardware slug; default: all")
    recorded.add_argument("--threads", help="one width only")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one command; with none, plan the default tracking run."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        tokens = ["list"]
    args = parser_for().parse_args(tokens)
    try:
        if args.command == "track":
            return track(args)
        if args.command == "ab":
            return ab(args)
        if args.command == "history":
            paths = (
                [history.HISTORY / f"{args.slug}.csv"]
                if args.slug
                else sorted(history.HISTORY.glob("*.csv"))
            )
            if not paths:
                print("No recorded runs.")
            for path in paths:
                history.show(path, args.threads)
            return 0
        rows, skipped = plan_rows(
            select(tuple(args.case), tuple(args.set), all_cases=args.all),
            steppers(args.stepper),
            args.threads,
            args.production or physical_cores(),
            backend=args.backend,
        )
        show_plan(rows, skipped, args.repeats)
        return 0
    except (SuiteError, OSError, ValueError) as error:
        print(f"benchmarks: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
