# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Collect native CPU profiles and separate Kokkos execution-label evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from benchmarks.collection import launch, prepared_python
from benchmarks.coverage_reports import (
    capability_tags,
    coverage_tables,
    project_file,
    source_digest,
)
from benchmarks.planning import Selection, matrix
from benchmarks.records import (
    LOCAL_ROOT,
    Record,
    SuiteError,
    atomic_record,
    file_hash,
    run_lock,
    write_csv,
)
from benchmarks.registry import CASES, select
from benchmarks.system import admission, profile_for, thread_environment

__all__ = [
    "CoverageOptions",
    "collect_coverage",
    "command",
    "coverage",
    "profile_groups",
    "read_labels",
    "render_report",
    "toolchain",
    "verified_profiles",
]


@dataclass(frozen=True)
class CoverageOptions:
    """Fix the diagnostic binary, tools, source tree and execution limits."""

    output: Path
    native_root: Path
    llvm_bin: Path
    python: str
    timeout: float
    memory_limit: int | None


def command(arguments: list[str], directory: Path, name: str) -> str:
    """Retain tool output and refuse failed or mismatched profile processing."""
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        arguments, capture_output=True, text=True, check=False, timeout=300
    )
    (directory / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
    (directory / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode or re.search(
        r"mismatch|malformed|out of date|truncated|error:", result.stderr, re.I
    ):
        raise SuiteError(f"{name} failed; see {directory / f'{name}.stderr.log'}")
    return result.stdout


def toolchain(directory: Path, compiler: str) -> Record:
    """Require matching LLVM releases for Clang's versioned raw-profile format."""
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", compiler)
    if match is None:
        raise SuiteError("coverage compiler has no identifiable LLVM version")
    result = {}
    for name in ("llvm-profdata", "llvm-cov"):
        path = directory / name
        version = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, check=True
        ).stdout
        found = re.search(r"version (\d+\.\d+\.\d+)", version)
        if found is None or found[1] != match[1]:
            raise SuiteError(f"{name} must match coverage compiler {match[1]}")
        result[name] = {
            "path": str(path),
            "version": found[1],
            "sha256": file_hash(path),
        }
    return result


def verified_profiles(rows: list[Record], binary_hash: str) -> list[Path]:
    """Admit successful raw profiles only with their recorded binary and hashes."""
    result = []
    for row in rows:
        if row["status"] != "ok":
            continue
        if row["extension_hash"] != binary_hash:
            raise SuiteError("coverage profile belongs to a different binary")
        if not row["profiles"]:
            raise SuiteError("successful coverage worker has no raw profiles")
        for profile in row["profiles"]:
            path = Path(profile["path"])
            if (
                not path.is_file()
                or not path.stat().st_size
                or file_hash(path) != profile["sha256"]
            ):
                raise SuiteError("raw coverage profile is missing, empty or changed")
            result.append(path)
    return sorted(set(result))


def profile_groups(rows: list[Record]) -> dict[str, list[Record]]:
    """Union successful configurations by case and declared set, without repeats."""
    groups: dict[str, list[Record]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        case = CASES[row["configuration"]["case"]]
        for key in (
            "union",
            f"cases/{case.name}",
            *(f"sets/{name}" for name in case.sets),
        ):
            groups.setdefault(key, []).append(row)
    return groups


def render_report(
    options: CoverageOptions, name: str, rows: list[Record], binary: Record
) -> Record:
    """Merge a verified profile union and export HTML plus normalized CSV views."""
    directory = options.output / "reports" / name
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(binary["path"])
    if file_hash(path) != binary["sha256"]:
        raise SuiteError("coverage binary snapshot changed")
    if source_digest(options.native_root) != binary["source_hash"]:
        raise SuiteError("native source changed since the coverage build")
    profiles = verified_profiles(rows, binary["sha256"])
    if not profiles:
        raise SuiteError("coverage report has no successful profiles")
    profile = directory / "merged.profdata"
    command(
        [
            str(options.llvm_bin / "llvm-profdata"),
            "merge",
            "-sparse",
            "--failure-mode=any",
            *map(str, profiles),
            "-o",
            str(profile),
        ],
        directory,
        "merge",
    )
    base = [
        str(options.llvm_bin / "llvm-cov"),
        "export",
        str(path),
        f"-instr-profile={profile}",
    ]
    exported = json.loads(command(base, directory, "export"))
    lcov = command([*base, "-format=lcov"], directory, "lcov")
    summary = coverage_tables(exported, lcov, options.native_root, directory)
    sources = sorted(
        {
            item["filename"]
            for data in exported["data"]
            for item in data["files"]
            if project_file(item["filename"], options.native_root) is not None
        }
    )
    command(
        [
            str(options.llvm_bin / "llvm-cov"),
            "show",
            str(path),
            f"-instr-profile={profile}",
            "-format=html",
            f"-output-dir={directory / 'html'}",
            "-show-line-counts-or-regions",
            "-show-branches=count",
            "-show-instantiations",
            *sources,
        ],
        directory,
        "html",
    )
    result = {
        "group": name,
        "binary": binary,
        "configuration_ids": [row["id"] for row in rows],
        **summary,
    }
    atomic_record(directory / "summary.json", result, immutable=True)
    return result


def _environment(configuration: Record, directory: Path) -> dict[str, str]:
    environment = thread_environment(
        configuration["threads"], configuration["policy"], dict(os.environ)
    )
    # Neither inherited tools nor an old profile destination may contaminate this
    # process. Labels get their own second pass and raw-profile directory.
    environment.pop("KOKKOS_TOOLS_LIBS", None)
    environment.pop("SOLPROP_NATIVE_LABEL_FILE", None)
    raw = directory / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    environment["LLVM_PROFILE_FILE"] = str((raw / "%p-%m.profraw").resolve())
    return environment


def _one_case(
    row: Record, plan: Record, options: CoverageOptions, binary: Record
) -> Record:
    configuration = row["configuration"]
    directory = options.output / "configurations" / row["id"]
    receipt = {
        "id": row["id"],
        "configuration": configuration,
        "extension_hash": binary["sha256"],
        "profiles": [],
        "declared_capabilities": capability_tags(configuration["case"]),
    }
    admit = admission(row["estimate_bytes"], options.memory_limit)
    if row["status"] != "ready" or not admit["admitted"]:
        receipt.update(
            status="excluded" if row["status"] != "ready" else "refused",
            reason=row["reason"],
            admission=admit,
        )
    else:
        request = {
            "configuration": configuration,
            "coverage": True,
            "extension_hash": binary["sha256"],
        }
        result = launch(
            options.python,
            request,
            _environment(configuration, directory),
            directory / "worker",
            (options.timeout, admit["limit_bytes"]),
            tuple(plan["profile"]["allocation"]),
        )
        receipt.update(status=result["status"], result=result)
        receipt["capability_status"] = (
            "executed"
            if result["status"] == "ok"
            else "unsupported"
            if "UnsupportedFeatureError:" in result.get("error", "")
            else "failed"
        )
        profiles = sorted((directory / "raw").glob("*.profraw"))
        receipt["profiles"] = [
            {"path": str(path.resolve()), "sha256": file_hash(path)}
            for path in profiles
        ]
        if receipt["status"] == "ok" and (
            not profiles or any(not path.stat().st_size for path in profiles)
        ):
            receipt.update(
                status="failed", error="missing or empty raw coverage profiles"
            )
        if receipt["status"] == "ok":
            if file_hash(Path(binary["labels_library"])) != binary["labels_hash"]:
                raise SuiteError("Kokkos label module changed during collection")
            labels = directory / "labels"
            environment = _environment(configuration, labels)
            environment.update(
                KOKKOS_TOOLS_LIBS=binary["labels_library"],
                SOLPROP_NATIVE_LABEL_FILE=str((labels / "events.jsonl").resolve()),
            )
            receipt["label_result"] = launch(
                options.python,
                request,
                environment,
                labels / "worker",
                (options.timeout, admit["limit_bytes"]),
                tuple(plan["profile"]["allocation"]),
            )
            events = labels / "events.jsonl"
            try:
                receipt["labels"] = read_labels(events)
            except (OSError, ValueError) as error:
                receipt["labels"] = []
                receipt["labels_error"] = str(error)
            receipt["labels_status"] = (
                "ok"
                if not receipt.get("labels_error")
                and receipt["label_result"]["status"] == "ok"
                else "failed"
            )
    atomic_record(directory / "receipt.json", receipt, immutable=True)
    return receipt


def read_labels(path: Path) -> list[Record]:
    """Reject malformed execution-label receipts without interpreting them as zero."""
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("label"), str)
            or not isinstance(item.get("kind"), str)
            or type(item.get("count")) is not int
            or item["count"] < 1
        ):
            raise SuiteError("invalid Kokkos execution-label receipt")
        result.append(item)
    if not result:
        raise SuiteError("Kokkos label collector produced no execution labels")
    return result


def collect_coverage(plan: Record, options: CoverageOptions) -> Record:
    """Collect each selected configuration serially, retaining failures as rows."""
    with run_lock(options.output):
        if (options.output / "manifest.json").exists():
            raise SuiteError("coverage output already exists; choose a new bundle")
        inventory_configuration = {
            "case": "smoke",
            "backend": "native",
            "threads": 1,
            "policy": "budget",
        }
        directory = options.output / "inventory"
        inventory = launch(
            options.python,
            {
                "configuration": inventory_configuration,
                "inventory": True,
                "coverage": True,
            },
            _environment(inventory_configuration, directory),
            directory / "worker",
            (options.timeout, options.memory_limit),
            tuple(plan["profile"]["allocation"]),
        )
        atomic_record(directory / "receipt.json", inventory, immutable=True)
        if inventory["status"] != "ok":
            raise SuiteError(f"coverage inventory failed: {inventory.get('error')}")
        provenance = inventory["provenance"]
        build = provenance["build_info"]
        digest = source_digest(options.native_root)
        if digest != build.get("coverage_source_hash"):
            raise SuiteError("native source does not match the instrumented binary")
        tools = toolchain(options.llvm_bin, build["compiler"])
        extension = Path(provenance["extension_path"])
        libraries = list(extension.parent.glob("solprop_native_labels.*"))
        if len(libraries) != 1:
            raise SuiteError("coverage build must provide one Kokkos label module")
        snapshot = options.output / f"native-core{extension.suffix}"
        shutil.copyfile(extension, snapshot)
        binary = {
            "path": str(snapshot.resolve()),
            "sha256": file_hash(snapshot),
            "source_hash": digest,
            "labels_library": str(libraries[0]),
            "labels_hash": file_hash(libraries[0]),
        }
        if binary["sha256"] != provenance["extension_hash"]:
            raise SuiteError("native binary changed during coverage inventory")
        atomic_record(
            options.output / "manifest.json",
            {
                "schema": "solprop.native-coverage.v1",
                "plan": plan,
                "provenance": provenance,
                "binary": binary,
                "tools": tools,
                "scope": "construction, lead-in, one window and observations; "
                "separate label pass; no timing samples",
            },
            immutable=True,
        )
        rows = []
        for row in plan["rows"]:
            print(
                f"coverage {row['configuration']['case']} "
                f"{row['configuration']['track']} "
                f"{row['configuration']['threads']} threads",
                flush=True,
            )
            try:
                rows.append(_one_case(row, plan, options, binary))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                failure = {
                    "id": row["id"],
                    "configuration": row["configuration"],
                    "status": "failed",
                    "error": str(error),
                }
                atomic_record(
                    options.output / "configurations" / row["id"] / "failure.json",
                    failure,
                    immutable=True,
                )
                rows.append(failure)
        reports = {}
        for name, selected in profile_groups(rows).items():
            try:
                reports[name] = {
                    "status": "ok",
                    **render_report(options, name, selected, binary),
                }
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                reports[name] = {"status": "failed", "error": str(error)}
                atomic_record(
                    options.output / "reports" / name / "failure.json",
                    reports[name],
                    immutable=True,
                )
        write_csv(
            options.output / "configurations.csv",
            [
                {
                    "id": r["id"],
                    **r["configuration"],
                    "status": r["status"],
                    "labels_status": r.get("labels_status"),
                    "capabilities": r.get("declared_capabilities", ()),
                    "capability_status": r.get("capability_status", r["status"]),
                }
                for r in rows
            ],
        )
        write_csv(
            options.output / "labels.csv",
            [
                {
                    "id": r["id"],
                    "case": r["configuration"]["case"],
                    "declared_capabilities": r.get("declared_capabilities", ()),
                    **label,
                }
                for r in rows
                for label in r.get("labels", [])
            ],
        )
        result = {
            "status": "complete"
            if all(report["status"] == "ok" for report in reports.values())
            and all(
                r["status"] in ("ok", "excluded")
                and r.get("labels_status", "ok") == "ok"
                for r in rows
            )
            else "incomplete",
            "rows": rows,
            "reports": reports,
            "instrumented_cpu_only": True,
            "compiled_out_capabilities": ["CUDA", "HIP"],
            "uninstantiated_templates": None,
            "uninstantiated_reason": "not enumerated by LLVM coverage maps; "
            "not counted as missed executable code",
        }
        atomic_record(options.output / "summary.json", result, immutable=True)
        return result


def coverage(args: argparse.Namespace) -> int:
    """Resolve coverage selections independently of timing modes and rounds."""
    profile = profile_for(args.machine_profile, standard=False)
    plan = matrix(
        select(tuple(args.case), tuple(args.set), all_cases=args.all),
        profile,
        Selection(
            mode="standard",
            backend="native",
            threads=args.threads,
            tracks=tuple(args.track),
            diagnostic=args.diagnostic,
        ),
    )
    for key in ("mode", "rounds", "maximum_rounds", "noise_extension_cost"):
        plan.pop(key, None)
    options = CoverageOptions(
        (args.out or LOCAL_ROOT / "coverage" / uuid.uuid4().hex).resolve(),
        args.native_root.resolve(),
        args.llvm_bin.resolve(),
        prepared_python(args.python),
        args.timeout,
        args.memory_limit,
    )
    result = collect_coverage(plan, options)
    union = result["reports"].get("union", {})
    if union.get("status") == "ok":
        print("Native project coverage (compiled CPU paths):")
        for name in ("lines", "functions", "branches"):
            metric = union["totals"][name]
            percent = (
                f"{metric['percent']:.2f}%"
                if metric["percent"] is not None
                else "unavailable"
            )
            print(f"  {name}: {metric['covered']}/{metric['count']} ({percent})")
    failures = sum(row["status"] not in ("ok", "excluded") for row in result["rows"])
    if failures:
        print(
            f"  {failures} configuration(s) failed or were refused; "
            "retained in the report"
        )
    print(f"Native coverage: {options.output} ({result['status']})")
    return 0 if result["status"] == "complete" else 1
