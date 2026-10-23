# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check native report semantics and failed-profile refusal without a campaign."""

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.__main__ import parser_for
from benchmarks.coverage_reports import (
    capability_tags,
    coverage_tables,
    project_file,
    source_digest,
)
from benchmarks.native_coverage import (
    CoverageOptions,
    command,
    profile_groups,
    read_labels,
    render_report,
    toolchain,
    verified_profiles,
)
from benchmarks.records import SuiteError, file_hash
from benchmarks.worker import runtime


def test_llvm_views_preserve_hits_misses_and_exclude_dependency_templates(tmp_path):
    root = tmp_path / "native"
    source = root / "src/core.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("hit\nmiss\n", encoding="utf-8")
    dependency = root / "build/_deps/kokkos.cpp"
    metrics = {
        key: {"count": 2, "covered": 1}
        for key in ("lines", "functions", "instantiations", "regions", "branches")
    }
    exported = {
        "type": "llvm.coverage.json.export",
        "data": [
            {
                "files": [
                    {
                        "filename": str(p),
                        "summary": metrics,
                        "branches": [[1, 1, 1, 5, 3, 0, 0, 0, 4]],
                    }
                    for p in (source, dependency)
                ],
                "functions": [
                    {
                        "name": name,
                        "count": count,
                        "filenames": [str(p)],
                        "regions": [[1, 1, 1, 5, count, 0, 0, 0]],
                    }
                    for name, count, p in (
                        ("hit", 3, source),
                        ("miss", 0, source),
                        ("dependency", 100, dependency),
                    )
                ],
            }
        ],
    }
    lcov = (
        f"SF:{source}\nDA:1,3\nDA:2,0\nend_of_record\n"
        f"SF:{dependency}\nDA:1,100\nend_of_record\n"
    )
    output = tmp_path / "report"
    output.mkdir()
    result = coverage_tables(exported, lcov, root, output)
    assert result["totals"]["lines"] == {"count": 2, "covered": 1, "percent": 50}
    for name in ("lines", "functions", "branches"):
        with (output / f"{name}.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert {row["file"] for row in rows} == {"src/core.cpp"}
    assert result["uninstantiated"]["count"] is None
    assert result["compiled_out"]["count"] is None
    with pytest.raises(SuiteError, match="no mapped"):
        coverage_tables(exported, "", root, output)
    with pytest.raises(SuiteError, match="unsupported"):
        coverage_tables({}, lcov, root, output)


def test_sources_and_profile_receipts_bind_the_exact_build(tmp_path):
    root = tmp_path / "native"
    root.mkdir()
    with pytest.raises(SuiteError, match="no project"):
        source_digest(root)
    source = root / "include/solprop_native/core.hpp"
    source.parent.mkdir(parents=True)
    source.write_text("first", encoding="utf-8")
    original = source_digest(root)
    source.write_text("changed", encoding="utf-8")
    assert source_digest(root) != original
    assert project_file(str(source), root) == "include/solprop_native/core.hpp"
    assert project_file(str(root / "src/coverage_labels.cpp"), root) is None
    assert project_file(str(tmp_path / "other.cpp"), root) is None
    assert project_file(str(root / "src/python.py"), root) is None
    profile = tmp_path / "123-module.profraw"
    profile.write_bytes(b"profile")
    row = {
        "status": "ok",
        "extension_hash": "binary",
        "profiles": [{"path": str(profile), "sha256": file_hash(profile)}],
    }
    assert verified_profiles([row, row, {"status": "failed"}], "binary") == [profile]
    with pytest.raises(SuiteError, match="different binary"):
        verified_profiles([row], "other")
    with pytest.raises(SuiteError, match="no raw profiles"):
        verified_profiles([row | {"profiles": []}], "binary")
    profile.write_bytes(b"changed")
    with pytest.raises(SuiteError, match="changed"):
        verified_profiles([row], "binary")


def test_case_set_union_does_not_turn_failures_into_coverage():
    rows = [
        {"status": "ok", "configuration": {"case": "rdw"}},
        {"status": "ok", "configuration": {"case": "ctc"}},
        {"status": "failed", "configuration": {"case": "line"}},
    ]
    groups = profile_groups(rows)
    assert groups["union"] == rows[:2]
    assert groups["sets/core"] == rows[:2]
    assert groups["cases/rdw"] == rows[:1]
    assert "cases/line" not in groups
    assert "tensor-raman" in capability_tags("vector_raman")
    assert "general-modal" in capability_tags("grin_linear")
    assert "rk4ip" in capability_tags("kerr_beam_192")


def test_tool_versions_and_failure_logs_are_explicit(tmp_path, monkeypatch):
    for name in ("llvm-cov", "llvm-profdata"):
        (tmp_path / name).write_text("tool", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, "LLVM version 23.1.3", ""
        ),
    )
    assert toolchain(tmp_path, "Clang 23.1.3")["llvm-cov"]["version"] == "23.1.3"
    with pytest.raises(SuiteError, match="must match"):
        toolchain(tmp_path, "Clang 22.0.0")
    with pytest.raises(SuiteError, match="no identifiable"):
        toolchain(tmp_path, "unknown")
    assert command(["tool"], tmp_path, "success") == "LLVM version 23.1.3"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 1, "", "corrupt profile"
        ),
    )
    with pytest.raises(SuiteError, match="failed"):
        command(["tool"], tmp_path, "merge")
    assert (tmp_path / "merge.stderr.log").read_text(
        encoding="utf-8"
    ) == "corrupt profile"


def test_binary_and_source_changes_refuse_report_before_merging(tmp_path):
    binary = tmp_path / "native.so"
    binary.write_bytes(b"native")
    options = CoverageOptions(
        tmp_path / "output", tmp_path, tmp_path, "python", 30, None
    )
    with pytest.raises(SuiteError, match="snapshot changed"):
        render_report(options, "union", [], {"path": str(binary), "sha256": "other"})
    source = tmp_path / "src/core.cpp"
    source.parent.mkdir()
    source.write_text("source", encoding="utf-8")
    record = {"path": str(binary), "sha256": file_hash(binary), "source_hash": "other"}
    with pytest.raises(SuiteError, match="source changed"):
        render_report(replace(options, output=tmp_path / "second"), "union", [], record)


@pytest.mark.parametrize(
    "value", ["", "broken", "{}", '{"kind":"for","label":"x","count":0}']
)
def test_bad_labels_are_unknown_not_zero(tmp_path, value):
    path = tmp_path / "labels.jsonl"
    path.write_text(value, encoding="utf-8")
    with pytest.raises(ValueError):
        read_labels(path)


def test_coverage_is_an_explicit_command_and_labels_are_lossless(tmp_path):
    args = parser_for().parse_args(
        [
            "coverage",
            "--case",
            "smoke",
            "--native-root",
            ".",
            "--llvm-bin",
            ".",
            "--out",
            "output",
        ]
    )
    assert args.threads == "1"
    assert not hasattr(args, "mode")
    assert not hasattr(args, "repetitions")
    label = {"kind": "parallel_for", "label": 'a\\b"\n', "count": 2}
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(label) + "\n", encoding="utf-8")
    assert read_labels(path) == [label]


@pytest.mark.parametrize("instrumented", [False, True])
def test_timing_and_coverage_refuse_the_wrong_build_before_construction(
    monkeypatch, instrumented
):
    @dataclass
    class Build:
        coverage_instrumented: bool = instrumented

    package = SimpleNamespace(build_info=Build, _core=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "solprop_native", package)
    with pytest.raises(SuiteError, match="instrumented"):
        runtime("native", coverage=not instrumented)


@pytest.mark.medium
def test_real_clang_known_hits_misses_and_invalid_profile(tmp_path):
    location = os.environ.get("SOLPROP_COVERAGE_LLVM")
    if not location:
        pytest.skip(
            "explicit matching LLVM toolchain required for coverage integration"
        )
    tools = Path(location)
    root = tmp_path / "native"
    source = root / "src/core.cpp"
    source.parent.mkdir(parents=True)
    source.write_text(
        "int choose(int n) {\n  if (n > 1) return 2;\n  return 1;\n}\n"
        "int unused(int n) { return n * 4; }\n"
        "int main(int argc, char**) { return choose(argc) == 1 ? 0 : 1; }\n",
        encoding="utf-8",
    )
    binary = tmp_path / "fixture"
    command(
        [
            str(tools / "clang++"),
            "-O1",
            "-fprofile-instr-generate",
            "-fcoverage-mapping",
            "-fprofile-update=atomic",
            str(source),
            "-o",
            str(binary),
        ],
        tmp_path,
        "compile",
    )
    profile = tmp_path / "fixture.profraw"
    subprocess.run(
        [str(binary)], env={**os.environ, "LLVM_PROFILE_FILE": str(profile)}, check=True
    )
    digest = file_hash(binary)
    row = {
        "id": "fixture",
        "status": "ok",
        "extension_hash": digest,
        "profiles": [{"path": str(profile), "sha256": file_hash(profile)}],
    }
    options = CoverageOptions(
        tmp_path / "output", root, tools, sys.executable, 60, None
    )
    summary = render_report(
        options,
        "union",
        [row],
        {"path": str(binary), "sha256": digest, "source_hash": source_digest(root)},
    )
    assert summary["totals"]["functions"]["count"] == 3
    assert summary["totals"]["functions"]["covered"] == 2
    assert (
        0
        < summary["totals"]["branches"]["covered"]
        < summary["totals"]["branches"]["count"]
    )
    assert (options.output / "reports/union/html/index.html").is_file()
    bad = tmp_path / "bad.profraw"
    bad.write_bytes(b"invalid profile")
    with pytest.raises(SuiteError, match="failed"):
        command(
            [
                str(tools / "llvm-profdata"),
                "merge",
                str(bad),
                "-o",
                str(tmp_path / "bad.profdata"),
            ],
            tmp_path,
            "bad-profile",
        )
