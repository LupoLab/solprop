# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check topology identities, missing telemetry and separate binding diagnostics."""

from types import SimpleNamespace

import psutil
import pytest
from benchmarks import __main__ as cli
from benchmarks import telemetry
from benchmarks.records import SuiteError, atomic_record
from benchmarks.system import Profile


def test_linux_topology_distinguishes_smt_socket_and_missing_nodes(tmp_path):
    for cpu, core, socket, siblings in (
        (0, 0, 0, "0,4"),
        (4, 0, 0, "0,4"),
        (5, 1, 1, "1,5"),
    ):
        directory = tmp_path / f"cpu{cpu}" / "topology"
        directory.mkdir(parents=True)
        for name, value in (
            ("core_id", core),
            ("physical_package_id", socket),
            ("thread_siblings_list", siblings),
        ):
            (directory / name).write_text(str(value), encoding="utf-8")
        (directory.parent / f"node{socket}").mkdir()
    rows = telemetry.linux_topology(tmp_path, (0, 4, 5, 8))
    assert [(row["socket"], row["core"]) for row in rows[:3]] == [
        (0, 0),
        (0, 0),
        (1, 1),
    ]
    assert rows[0]["siblings"] == [0, 4]
    assert rows[2]["numa_nodes"] == [1]
    assert rows[3]["core"] is None and rows[3]["core_reason"]
    assert rows[3]["numa_nodes"] is None and rows[3]["numa_reason"]
    assert telemetry.cpu_list("0-2,2,5") == [0, 1, 2, 5]
    for invalid in ("", "-1", "3-1", "0-a"):
        with pytest.raises(SuiteError):
            telemetry.cpu_list(invalid)


def test_numa_mapping_counts_use_each_mapping_page_size_and_omit_paths():
    contents = (
        "abc default file=/private/library N0=2 N1=3 kernelpagesize_kB=4\n"
        "def bind:1 anon=1 N1=1 kernelpagesize_kB=2048\n"
        "ghi default file=/private/unknown N0=10\n"
    )
    assert telemetry.numa_pages(contents) == [
        {"node": 0, "resident_mapping_bytes": 8192},
        {"node": 1, "resident_mapping_bytes": 2109440},
    ]


def test_missing_hardware_and_process_counters_keep_reasons(monkeypatch):
    monkeypatch.setattr(telemetry.platform, "system", lambda: "unknown")
    record = telemetry.hardware_details((0,))
    assert record["cpu_model"] is None and record["cpu_model_reason"]
    assert record["cpu_topology"] is None and record["cpu_topology_reason"]
    monkeypatch.setattr(telemetry, "_read", lambda _: None)

    def denied():
        raise psutil.AccessDenied()

    monkeypatch.setattr(telemetry.psutil, "Process", denied)
    record = telemetry.process_binding()
    assert record["threads"] == [] and record["threads_reason"]
    assert record["numa"] == [] and record["numa_reason"]
    assert record["allocation"]["allowed_cpus"] is None
    assert record["residency"] is None and record["residency_reason"]


def test_binding_snapshot_does_not_claim_execution_residency(monkeypatch):
    monkeypatch.setattr(telemetry, "_read", lambda _: None)
    thread = SimpleNamespace(id=123, user_time=1.0, system_time=0.25)
    monkeypatch.setattr(
        telemetry.psutil, "Process", lambda: SimpleNamespace(threads=lambda: [thread])
    )
    monkeypatch.setattr(
        telemetry.os, "sched_getaffinity", lambda _: {2, 4}, raising=False
    )
    record = telemetry.process_binding()
    assert record["threads"][0]["allowed_cpus"] == [2, 4]
    assert record["residency"] is None


def test_diagnostic_command_uses_serial_exploratory_quick_collection(
    tmp_path, monkeypatch
):
    seen = []
    monkeypatch.setattr(
        cli,
        "profile_for",
        lambda *_args, **_kw: Profile("test", 1, 2, (0, 1), "topology"),
    )

    def collect(plan, options):
        seen.append((plan, options))
        atomic_record(
            options.output / "attempts" / "0.json",
            {
                "attempt_id": 0,
                "configuration_id": plan["rows"][0]["id"],
                "thread_diagnostics": {
                    "threads": [{"thread_id": 1, "allowed_cpus": [0, 1]}]
                },
            },
        )
        return {"execution_ok": True}

    monkeypatch.setattr(cli, "collect", collect)
    assert cli.main(["diagnose", "--out", str(tmp_path / "diagnostic")]) == 0
    plan, options = seen[0]
    assert plan["purpose"] == "thread-binding-diagnostic" and plan["thread_diagnostics"]
    assert plan["mode"] == "quick" and plan["rounds"] == 1
    assert [row["configuration"]["threads"] for row in plan["rows"]] == [1, 2]
    assert options.force
    assert (options.output / "threads.csv").exists()
    assert (
        cli.main(["diagnose", "--out", str(tmp_path / "invalid"), "--timeout", "0"])
        == 2
    )
