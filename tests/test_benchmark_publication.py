# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Verify public export omissions, failure retention and missing telemetry."""

import csv
import dataclasses
import time

import pytest
from benchmarks.__main__ import main
from benchmarks.collection import Monitor
from benchmarks.planning import Selection, matrix
from benchmarks.publication import export_public, public_record, summaries
from benchmarks.records import (
    SuiteError,
    atomic_record,
    attempts,
    canonical,
    read_record,
)
from benchmarks.registry import select
from benchmarks.system import Profile
from benchmarks.views import write_views


@pytest.fixture
def local_bundle(tmp_path):
    source = tmp_path / "local"
    plan = matrix(
        select(("smoke",)),
        Profile("private-machine", 1, 1, (0,), "topology"),
        Selection(backend="numpy_ref"),
    )
    plan["environments"] = {
        "candidate": {
            "solprop_path": "/private/source/solprop",
            "python_executable": "/private/python",
            "packages": {"numpy": "2.5.2"},
            "source": {"revision": "abc", "location": "/private/source"},
            "build_info": {
                "execution_space": "OpenMP",
                "compiler_flags": "-I/private/include",
            },
        }
    }
    plan["system"]["hostname"] = "private-host"
    plan["options"] = {"references": "/private/reference"}
    manifest = {"created": "2026-10-21T12:00:00Z", "plan": plan}
    atomic_record(source / "manifest.json", manifest)
    for index, status in enumerate(("ok", "failed")):
        atomic_record(
            source / "attempts" / f"{index}.json",
            {
                "attempt_id": index,
                "sample_id": str(index),
                "arm": "candidate",
                "round": index,
                "configuration_id": plan["rows"][0]["id"],
                "status": status,
                "phase": "measurement",
                "seconds": 1.0 if status == "ok" else None,
                "clean": False,
                "peak_rss_bytes": None,
                "error": "/private/worker crashed on private-host",
                "diagnostics": [{"message": "/private/log"}],
                "process": {"pid": 123, "command": "/private/command"},
                "repetitions": [{"seconds": 1.0, "source_evaluations": 7}]
                if status == "ok"
                else [],
            },
        )
    return source


def test_public_export_omits_private_details_and_preserves_failures(
    local_bundle, tmp_path
):
    destination = tmp_path / "public"
    original = read_record(local_bundle / "manifest.json")
    result = export_public(local_bundle, destination, "public-machine")
    contents = canonical(result)
    assert "private" not in contents.replace("private bundle", "").replace(
        "private paths", ""
    )
    assert "pid" not in contents
    assert result["environments"]["candidate"]["build"]["execution_space"] == "OpenMP"
    assert [row["status"] for row in result["samples"]] == ["ok", "failed"]
    assert all(row["peak_rss_bytes"] is None for row in result["samples"])
    assert all(row["peak_rss_unavailable_reason"] for row in result["samples"])
    assert read_record(local_bundle / "manifest.json") == original
    summary = summaries(result["samples"])
    assert summary[0]["count"] == 1 and summary[0]["mad"] is None
    assert summary[1]["count"] == 0 and summary[1]["median"] is None
    assert all(row["failed_samples"] == 1 for row in summary)
    for name in (
        "samples",
        "summary",
        "configurations",
        "cases",
        "runs",
        "system",
        "repetitions",
    ):
        assert (destination / f"{name}.csv").exists()
    with pytest.raises(SuiteError, match="empty or new"):
        export_public(local_bundle, destination, "public-machine")
    with pytest.raises(SuiteError, match="must differ"):
        export_public(local_bundle, local_bundle, "public-machine")


def test_export_command_and_public_alias_validation(local_bundle, tmp_path):
    assert (
        main(
            [
                "export",
                str(local_bundle),
                "--out",
                str(tmp_path / "export"),
                "--alias",
                "cpu",
            ]
        )
        == 0
    )
    manifest = read_record(local_bundle / "manifest.json")
    with pytest.raises(SuiteError, match="public alias"):
        public_record(manifest, [], "../private")
    del manifest["plan"]["system"]
    assert public_record(manifest, [], "cpu")["hardware_unavailable_reason"]


def test_missing_process_telemetry_is_unknown_not_zero():
    monitor = Monitor(time.monotonic(), (0,))
    monitor.sample(None, (60.0, None))
    record = dataclasses.asdict(monitor)
    assert record["own_cpu"] is None
    assert record["sampled_peak"] is None
    assert record["missed_samples"] == 1


def test_local_views_link_attempts_and_explain_missing_telemetry(local_bundle):
    rows = attempts(local_bundle)
    rows[0]["process"] = {"pid": 123, "created": 123.5, "cpu_seconds": 0.25}
    rows[0]["provenance"] = {
        "libraries": [{"internal_api": "openblas", "num_threads": 1}]
    }
    write_views(local_bundle, rows)

    def table(name):
        with (local_bundle / f"{name}.csv").open(encoding="utf-8") as source:
            return list(csv.DictReader(source))

    assert table("runs")[0]["collection_seconds_reason"] == "collection incomplete"
    assert table("configurations")[0]["case"] == "smoke"
    assert table("allocations")[0]["cpu"] == "0"
    assert table("processes")[0]["cpu_seconds"] == "0.25"
    assert table("libraries")[0]["unavailable_reason"] == "no supported pool observed"
    assert table("libraries")[1]["scope"] == "after measurement"
    assert table("coverage")[0]["status"] == "not-collected"
    assert table("system")[0]["run_id"] == table("runs")[0]["run_id"]
