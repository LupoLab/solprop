# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Expose the standard benchmark command without importing scientific libraries."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import uuid
from pathlib import Path

from benchmarks.ab import ab, ab_options
from benchmarks.collection import Collection, collect, prepared_python
from benchmarks.comparison import compare, completed
from benchmarks.planning import Selection, matrix
from benchmarks.records import (
    LOCAL_ROOT,
    Record,
    SuiteError,
    attempts,
    canonical,
    distribution,
    read_record,
    write_csv,
)
from benchmarks.registry import ADAPTIVE, CASES, SETS, TRACKS, select
from benchmarks.system import profile_for, save_profile

__all__ = [
    "compare_paths",
    "coverage_options",
    "describe",
    "diagnose",
    "diagnostic_options",
    "main",
    "manage",
    "new_collection",
    "parser_for",
    "resume_request",
    "selected_plan",
    "selectors",
    "show_comparison",
    "show_results",
]


def selectors(parser: argparse.ArgumentParser) -> None:
    """Share selectors and measurement policy between planning and collection."""
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--set", action="append", choices=SETS, default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=("quick", "standard"), default="quick")
    parser.add_argument(
        "--backend",
        choices=("numpy_ref", "numba_cpu", "native", "julia"),
        default="native",
    )
    parser.add_argument("--threads")
    parser.add_argument("--track", action="append", choices=TRACKS, default=[])
    parser.add_argument("--machine-profile", type=Path)
    parser.add_argument("--references", type=Path, default=LOCAL_ROOT / "references")
    parser.add_argument("--diagnostic", action="store_true")


def diagnostic_options(parser: argparse.ArgumentParser) -> None:
    """Declare the separate thread/binding diagnostic's short workload and limits."""
    parser.add_argument("--case", default="smoke", choices=tuple(CASES))
    parser.add_argument(
        "--backend",
        choices=("numpy_ref", "numba_cpu", "native", "julia"),
        default="native",
    )
    parser.add_argument("--threads", default="1,production")
    parser.add_argument("--machine-profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--memory-limit", type=int)


def parser_for() -> argparse.ArgumentParser:
    """Declare the CLI and its explicit qualification/profile operations."""
    parser = argparse.ArgumentParser(description="SEP 30: bounded standard benchmarks.")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser(
        "list", help="plan without scientific imports or translation"
    )
    selectors(listing)
    listing.add_argument("--json", action="store_true")
    run = commands.add_parser(
        "run", help="collect quick or standard fresh-process samples"
    )
    selectors(run)
    run.add_argument("--out", type=Path)
    run.add_argument("--resume", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--baseline-python")
    run.add_argument("--candidate-python", default=sys.executable)
    run.add_argument("--seed", type=int, default=30)
    run.add_argument(
        "--timeout", type=float, default=1800.0, help="attempt limit, seconds"
    )
    run.add_argument("--memory-limit", type=int, help="attempt memory limit, bytes")
    run.add_argument("--allow-unestimated", action="store_true")
    run.add_argument(
        "--allow-accuracy-failure",
        action="store_true",
        help="retain failed accuracy checks and continue timing; exploratory only",
    )
    run.add_argument(
        "--force",
        action="store_true",
        help="skip guard waiting; exploratory results only",
    )
    comparison = commands.add_parser(
        "compare", help="compare complete stored standard selections"
    )
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path, nargs="?")
    comparison.add_argument("--aa-control", type=Path)
    comparison.add_argument("--json", action="store_true")
    publication = commands.add_parser(
        "export", help="write a sanitized public bundle without publishing it"
    )
    publication.add_argument("source", type=Path)
    publication.add_argument("--out", type=Path, required=True)
    publication.add_argument("--alias", required=True, help="public machine alias")
    diagnostic = commands.add_parser(
        "diagnose",
        help="inspect threads and binding separately from timing comparisons",
    )
    diagnostic_options(diagnostic)
    profile = commands.add_parser(
        "profile", help="save a declared production width before collecting"
    )
    profile.add_argument("out", type=Path)
    profile.add_argument("--alias", required=True)
    profile.add_argument("--production", type=int, required=True)
    profile.add_argument(
        "--policy",
        choices=("budget", "all-pools-N", "blas1", "as-installed"),
        default="budget",
    )
    qualification = commands.add_parser(
        "qualify", help="prepare/check numerical references separately from timing"
    )
    selectors(qualification)
    qualification.add_argument("--out", type=Path)
    qualification.add_argument("--timeout", type=float, default=3600.0)
    qualification.add_argument("--memory-limit", type=int)
    qualification.add_argument(
        "--accuracy-bound",
        type=float,
        help="reviewed field/band envelope; defaults to approved case adaptive bounds",
    )
    qualification.add_argument("--reference-revision", type=int)
    qualification.add_argument("--policy-revision", type=int)
    qualification.add_argument(
        "--recheck",
        type=Path,
        help="reassess retained reference evidence with a revised bound; no simulation",
    )
    qualification.add_argument(
        "--supersedes",
        type=Path,
        help="previous reference bundle; preserves its receipts",
    )
    qualification.add_argument(
        "--revision-note", help="reason for replacing a reference or accuracy policy"
    )
    qualification.add_argument(
        "--reviewed",
        action="store_true",
        help="with --recheck, confirm retained reset/history/statistics evidence "
        "was reviewed for standard comparison",
    )
    ab_options(
        commands.add_parser(
            "ab", help="time two builds on a few cases quickly; exploratory only"
        )
    )
    coverage_options(
        commands.add_parser(
            "coverage", help="collect separate Clang CPU coverage and execution labels"
        )
    )
    return parser


def coverage_options(coverage: argparse.ArgumentParser) -> None:
    """Declare the explicit native coverage diagnostic."""
    coverage.add_argument("--case", action="append", default=[])
    coverage.add_argument("--set", action="append", choices=SETS, default=[])
    coverage.add_argument("--all", action="store_true")
    coverage.add_argument("--track", action="append", choices=TRACKS, default=[])
    coverage.add_argument("--threads", default="1")
    coverage.add_argument("--machine-profile", type=Path)
    coverage.add_argument("--diagnostic", action="store_true")
    coverage.add_argument("--native-root", type=Path, required=True)
    coverage.add_argument("--llvm-bin", type=Path, required=True)
    coverage.add_argument("--python", default=sys.executable)
    coverage.add_argument(
        "--out",
        type=Path,
        help="local report directory; defaults beneath ignored .benchmarks/",
    )
    coverage.add_argument("--timeout", type=float, default=1800)
    coverage.add_argument("--memory-limit", type=int)


def describe(plan: dict) -> None:
    """Print admitted and excluded configurations without pretending costs are known."""
    print(
        f"{plan['mode']}: production width {plan['profile']['production']}; "
        f"{plan['rounds']} round(s), at most {plan['maximum_rounds']}"
    )
    for row in plan["rows"]:
        values = row["configuration"]
        estimate = row["estimate_bytes"]
        memory = "unknown" if estimate is None else f"{estimate / 2**30:.2f} GiB"
        detail = row["reason"] or f"reference {row['reference']}; cost unknown"
        print(
            f"{values['case']:24} {values['track']:20} {values['threads']:3} threads "
            f"{row['status']}: {detail}; memory budget {memory}"
        )
    print(
        "Estimates require compatible pilot receipts; "
        "reference preparation is separate."
    )


def show_results(plan: Record, rows: list[Record]) -> None:
    """Show all-sample timings and failures without hiding unqualified results."""
    groups: dict[tuple[str, str], list[Record]] = {}
    for row in completed(rows):
        groups.setdefault((row["configuration_id"], row["arm"]), []).append(row)
    print("\nWarm times are all-sample medians; startup includes the first window.")
    print(
        f"{'case':24} {'track':20} {'N':>3} {'arm':9} "
        f"{'warm / s':>10} {'start / s':>10} {'RSS / GiB':>10} {'clean':>7} check"
    )
    for item in plan["rows"]:
        configuration = item["configuration"]
        for (key, arm), group in groups.items():
            if key != item["id"]:
                continue
            timed = [row for row in group if row.get("seconds") is not None]
            warm = distribution([row["seconds"] for row in timed]).median
            start = distribution(
                [row["startup_seconds"] for row in timed if row.get("startup_seconds")]
            ).median
            peaks = [
                row["peak_rss_bytes"] for row in timed if row.get("peak_rss_bytes")
            ]
            peak = max(peaks) / 2**30 if peaks else None
            values = [
                "-" if value is None else f"{value:.4f}"
                for value in (warm, start, peak)
            ]
            clean = sum(row.get("clean") is True for row in group)
            checks = sorted(
                {row.get("accuracy", {}).get("status", row["status"]) for row in group}
            )
            print(
                f"{configuration['case']:24} {configuration['track']:20} "
                f"{configuration['threads']:3} {arm:9} "
                f"{values[0]:>10} {values[1]:>10} {values[2]:>10} "
                f"{clean:3}/{len(group):<3} {','.join(checks)}"
            )
            for row in group:
                if row["status"] != "ok":
                    detail = row.get("error", row.get("admission", "see accuracy.csv"))
                    print(f"  {row['status']}: {detail}")
    print(
        "Missing references do not block timing. "
        "Quick results have no adoption verdict."
    )


def show_comparison(result: Record) -> None:
    """Print descriptive changes even when adoption evidence is unavailable."""
    print(f"Comparison: {result['status']}")
    print("Changes use all-sample medians; negative means less time.")
    for row in result.get("configurations", []):
        ratio = row.get("all_ratio")
        change = "unavailable" if ratio is None else f"{100 * (ratio - 1):+.2f}%"
        print(
            f"{row['case']:24} {row['track']:20} {row['threads']:3} threads "
            f"{change:>12} {row['status']}"
        )
    if result.get("reason"):
        print(result["reason"])


def compare_paths(
    baseline: Path,
    candidate: Path | None,
    aa_path: Path | None = None,
    *,
    as_json: bool = False,
) -> int:
    """Refuse incompatible manifests before comparing their durable samples."""
    before = read_record(baseline / "manifest.json")["plan"]
    rows = attempts(baseline)
    if candidate is not None:
        after = read_record(candidate / "manifest.json")["plan"]
        if before["profile"] != after["profile"] or [
            row["id"] for row in before["rows"]
        ] != [row["id"] for row in after["rows"]]:
            print(
                canonical(
                    {
                        "status": "incompatible",
                        "reason": "selected workload or machine profile differs",
                    }
                )
            )
            return 1
        rows = [{**row, "arm": "baseline"} for row in rows] + [
            {**row, "arm": "candidate"} for row in attempts(candidate)
        ]
        before = {
            **before,
            "exploratory": before.get("exploratory", False)
            or after.get("exploratory", False),
        }
    bands = None
    if aa_path is not None:
        control = read_record(aa_path / "aa.json")
        environment = before["environments"].get(
            "baseline", before["environments"]["candidate"]
        )
        if (
            control["profile"] != before["profile"]
            or control["baseline"] != environment
        ):
            raise SuiteError("A/A profile or baseline environment identity differs")
        bands = control["configurations"]
    result = compare(before, rows, paired=candidate is None, aa_bands=bands)
    if as_json:
        print(canonical(result))
    else:
        show_comparison(result)
    usable = result["status"] in ("passed", "exploratory") and all(
        row["status"] not in ("incomplete", "incompatible", "numerically-invalid")
        for row in result.get("configurations", [])
    )
    return 0 if usable else 1


def resume_request(
    args: argparse.Namespace, tokens: list[str]
) -> tuple[Record, Collection]:
    """Restore a saved request while refusing explicit changes to its policy."""
    path: Path = args.resume
    saved = read_record(path / "manifest.json")["plan"]
    explicit = {token.split("=", 1)[0] for token in tokens if token.startswith("--")}
    selection = {
        "--case",
        "--set",
        "--all",
        "--mode",
        "--backend",
        "--threads",
        "--track",
        "--machine-profile",
        "--diagnostic",
        "--out",
    }
    if explicit & selection:
        raise SuiteError(
            "--resume restores the saved selection and profile; "
            "use a new output to change them"
        )
    options = dict(saved["options"])
    for key, value in options.items():
        if f"--{key.replace('_', '-')}" not in explicit:
            continue
        supplied = getattr(args, key)
        if key == "references":
            supplied = str(supplied.resolve())
        elif key in ("baseline_python", "candidate_python"):
            supplied = prepared_python(supplied)
        if supplied != value:
            raise SuiteError(f"resume {key} differs; use a new output")
    options["output"] = path
    options["references"] = Path(options["references"])
    options["resume"] = True
    return saved, Collection(**options)


def selected_plan(args: argparse.Namespace) -> Record:
    """Resolve fresh selectors without constructing a simulation."""
    cases = select(tuple(args.case), tuple(args.set), all_cases=args.all)
    profile = profile_for(args.machine_profile, standard=args.mode == "standard")
    plan = matrix(
        cases,
        profile,
        Selection(
            args.mode,
            args.backend,
            args.threads,
            tuple(args.track),
            args.references,
            args.diagnostic,
        ),
    )
    plan["fixed_assessment"] = bool(args.track) and all(
        track.startswith("fixed-") for track in args.track
    )
    return plan


def new_collection(args: argparse.Namespace) -> Collection:
    """Validate execution limits before admitting a new collection."""
    if args.timeout <= 0 or (args.memory_limit is not None and args.memory_limit <= 0):
        raise SuiteError("timeout and memory limit must be positive")
    if args.allow_unestimated and args.memory_limit is None:
        raise SuiteError("--allow-unestimated requires --memory-limit")
    return Collection(
        args.out or LOCAL_ROOT / "runs" / uuid.uuid4().hex,
        args.baseline_python,
        args.candidate_python,
        args.seed,
        args.timeout,
        args.memory_limit,
        args.force,
        args.allow_unestimated,
        False,
        args.references,
        args.allow_accuracy_failure,
    )


def diagnose(args: argparse.Namespace) -> int:
    """Collect one short diagnostic per width without an adoption comparison."""
    if args.timeout <= 0 or (args.memory_limit is not None and args.memory_limit <= 0):
        raise SuiteError("timeout and memory limit must be positive")
    case = CASES[args.case]
    profile = profile_for(args.machine_profile, standard=False)
    plan = matrix(
        (case,),
        profile,
        Selection(
            backend=args.backend,
            threads=args.threads,
            tracks=(ADAPTIVE[1] if ADAPTIVE[1] in case.tracks else case.tracks[0],),
            diagnostic=True,
        ),
    )
    plan.update(purpose="thread-binding-diagnostic", thread_diagnostics=True)
    result = collect(
        plan,
        Collection(
            args.out, timeout=args.timeout, memory_limit=args.memory_limit, force=True
        ),
    )
    rows = attempts(args.out)
    write_csv(
        args.out / "threads.csv",
        [
            {
                "attempt_id": row["attempt_id"],
                "configuration_id": row["configuration_id"],
                **thread,
            }
            for row in rows
            for thread in (row.get("thread_diagnostics") or {}).get("threads", [])
        ],
    )
    print(f"Thread/binding diagnostics: {args.out.resolve()}")
    return 0 if result["execution_ok"] else 1


def manage(args: argparse.Namespace) -> int | None:
    """Handle stored-bundle and profile commands before constructing a plan."""
    if args.command == "export":
        from benchmarks.publication import export_public  # noqa: PLC0415

        export_public(args.source, args.out, args.alias)
        print(f"Public export: {args.out.resolve()}")
        return 0
    if args.command == "profile":
        print(
            canonical(
                dataclasses.asdict(
                    save_profile(args.out, args.alias, args.production, args.policy)
                )
            )
        )
        return 0
    if args.command == "compare":
        return compare_paths(
            args.baseline, args.candidate, args.aa_control, as_json=args.json
        )
    if args.command == "diagnose":
        return diagnose(args)
    if args.command == "coverage":
        from benchmarks.native_coverage import coverage  # noqa: PLC0415

        return coverage(args)
    return None


def main(argv: list[str] | None = None) -> int:
    """Default to a quick collection, with explicit planning and review commands."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or (tokens[0].startswith("-") and tokens[0] not in ("-h", "--help")):
        tokens.insert(0, "run")
    args = parser_for().parse_args(tokens)
    try:
        managed = manage(args)
        if managed is not None:
            return managed
        if args.command == "ab":
            return ab(args)
        options = None
        if args.command == "run" and args.resume is not None:
            plan, options = resume_request(args, tokens)
        else:
            plan = selected_plan(args)
        if args.command == "list" or (args.command == "run" and args.dry_run):
            if getattr(args, "json", False):
                print(canonical(plan))
            else:
                describe(plan)
            return 0
        if args.command == "qualify":
            from benchmarks.qualification import qualify  # noqa: PLC0415

            return qualify(plan, args)
        if options is None:
            options = new_collection(args)
        result = collect(plan, options)
        show_results(plan, attempts(options.output))
        print(f"Results: {options.output.resolve()} ({result['status']})")
        return 0 if result["execution_ok"] else 1
    except (SuiteError, OSError, ValueError) as error:
        print(f"benchmark suite: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
