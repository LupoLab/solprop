# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Filter LLVM coverage to project sources and preserve its counting semantics."""

from __future__ import annotations

import hashlib
from pathlib import Path

from benchmarks.records import Record, SuiteError, file_hash, write_csv
from benchmarks.registry import CASES

__all__ = [
    "capability_tags",
    "coverage_tables",
    "lcov_lines",
    "project_file",
    "source_digest",
]


def capability_tags(case: str) -> tuple[str, ...]:
    """Declare intended exercise capabilities, separately from observed labels."""
    definition = CASES[case]
    tags = {definition.builder, *definition.sets}
    if case.startswith("kerr_beam"):
        tags.update(("cartesian", "envelope", "kerr", "rk4ip"))
    if "filament" in case:
        tags.update(("plasma", "field", "kerr"))
    if "averaged" in case or case.startswith("gnlse"):
        tags.add("mode-averaged")
    if definition.builder == "advanced":
        tags.add(
            "general-modal"
            if case.startswith(("grin", "step_index", "moving"))
            else "coupled-flow"
        )
    special = {
        "rdw": ("field", "modal", "plasma"),
        "ctc": ("field", "modal", "raman"),
        "vector_raman": ("tensor-raman", "envelope", "vector"),
        "envelope_plasma": ("envelope", "plasma"),
        "mixture_gradient": ("mixture", "z-dependent", "plasma"),
        "dispersive_kerr": ("dispersive-kerr",),
        "plasma_band": ("plasma", "band-integration"),
        "plasma_trapezoid": ("plasma", "trapezoid-integration"),
        "batch": ("batch", "member-statistics"),
        "noise": ("physical-noise",),
        "dense_output": ("dense-output",),
        "absorber_schemes": ("split-step-absorber",),
        "ordinary_stats": ("ordinary-statistics", "plasma"),
        "far_field_stats": (
            "far-field",
            "apertures",
            "spectral-regions",
            "periodic-statistics",
        ),
    }
    tags.update(special.get(case, ()))
    return tuple(sorted(tags))


def source_digest(root: Path) -> str:
    """Match the coverage build's ordered C++ source/header identity."""
    paths = sorted((*root.glob("src/**/*.cpp"), *root.glob("include/**/*.hpp")))
    if not paths:
        raise SuiteError("native source root contains no project C++ sources")
    record = "".join(
        f"{p.relative_to(root).as_posix()}:{file_hash(p)}\n" for p in paths
    )
    return hashlib.sha256(record.encode()).hexdigest()


def project_file(name: str, root: Path) -> str | None:
    """Accept core, bindings and project headers; exclude dependencies and tools."""
    path = Path(name).resolve()
    if not path.is_relative_to(root.resolve()):
        return None
    relative = path.relative_to(root.resolve())
    if relative.parts[0] == "src" and relative.suffix == ".cpp":
        if relative.name in ("benchmark_gemm.cpp", "coverage_labels.cpp"):
            return None
        return relative.as_posix()
    if (
        relative.parts[:2] == ("include", "solprop_native")
        and relative.suffix == ".hpp"
    ):
        return relative.as_posix()
    return None


def lcov_lines(lcov: str, root: Path) -> list[Record]:
    """Read LLVM line aggregation while excluding dependency files."""
    lines = []
    selected = None
    for line in lcov.splitlines():
        if line.startswith("SF:"):
            selected = project_file(line[3:], root)
        elif line == "end_of_record":
            selected = None
        elif selected is not None and line.startswith("DA:"):
            number, count, *_ = line[3:].split(",")
            lines.append({"file": selected, "line": int(number), "count": int(count)})
    return lines


def coverage_tables(exported: Record, lcov: str, root: Path, output: Path) -> Record:
    """Write exact LLVM line counts and function/branch views with project totals.

    LCOV supplies LLVM's line-count aggregation, avoiding a second interpretation
    of nested coverage segments. Branch rows retain true and false counts. LLVM
    omits constant-folded conditions and uninstantiated templates from its maps;
    their absence is reported explicitly, never as zero coverage.
    """
    if exported.get("type") != "llvm.coverage.json.export":
        raise SuiteError("unsupported LLVM coverage export")
    files, functions, branches = [], [], []
    for data in exported["data"]:
        for item in data["files"]:
            name = project_file(item["filename"], root)
            if name is None:
                continue
            files.append({"file": name, **item["summary"]})
            for index, branch in enumerate(item.get("branches", [])):
                branches.append(
                    {
                        "file": name,
                        "branch": index,
                        "line": branch[0],
                        "column": branch[1],
                        "end_line": branch[2],
                        "end_column": branch[3],
                        "true_count": branch[4],
                        "false_count": branch[5],
                    }
                )
        for item in data["functions"]:
            name = project_file(item["filenames"][0], root)
            if name is not None:
                functions.append(
                    {
                        "file": name,
                        "function": item["name"],
                        "count": item["count"],
                        "status": "covered" if item["count"] else "uncovered",
                        "line": item["regions"][0][0] if item["regions"] else None,
                    }
                )
    lines = lcov_lines(lcov, root)
    if not files or not lines:
        raise SuiteError("coverage export contains no mapped project lines")
    for name, rows in (
        ("lines", lines),
        ("functions", functions),
        ("branches", branches),
    ):
        write_csv(output / f"{name}.csv", rows)
    totals = {}
    for metric in ("lines", "functions", "instantiations", "regions", "branches"):
        count = sum(item[metric]["count"] for item in files)
        covered = sum(item[metric]["covered"] for item in files)
        totals[metric] = {
            "count": count,
            "covered": covered,
            "percent": 100 * covered / count if count else None,
        }
    return {
        "files": files,
        "totals": totals,
        "uninstantiated": {
            "count": None,
            "reason": "templates absent from LLVM maps have no measurable denominator",
        },
        "compiled_out": {
            "count": None,
            "reason": "preprocessor and constant-folded paths are absent from the map; "
            "inspect build capabilities",
        },
        "unmapped_sources": [
            p.relative_to(root).as_posix()
            for p in sorted(
                (*root.glob("src/*.cpp"), *root.glob("include/solprop_native/*.hpp"))
            )
            if project_file(str(p), root) is not None
            and p.relative_to(root).as_posix() not in {f["file"] for f in files}
        ],
    }
