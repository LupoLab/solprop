# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check topology identities and missing telemetry."""

import pytest
from benchmarks import telemetry
from benchmarks.records import SuiteError


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


def test_missing_hardware_and_allocation_details_keep_reasons(monkeypatch):
    monkeypatch.setattr(telemetry.platform, "system", lambda: "unknown")
    record = telemetry.hardware_details((0,))
    assert record["cpu_model"] is None and record["cpu_model_reason"]
    assert record["cpu_topology"] is None and record["cpu_topology_reason"]
    monkeypatch.setattr(telemetry, "_read", lambda _: None)
    record = telemetry.allocation_details()
    assert record["allowed_cpus"] is None and record["allowed_cpus_reason"]
