# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The machine a run records, and what it records the run spent."""

import contextlib
import json
import platform
import sys
import types
from typing import Any

import pytest

from solprop import _host


def test_the_machine_record_is_plain_json_and_names_no_host():
    record: Any = _host.machine()
    assert json.loads(json.dumps(record)) == record
    assert {"system", "machine", "cpu_model", "logical_cpus", "usable_cpus"} <= set(
        record
    )
    assert record["usable_cpus"] >= 1
    assert isinstance(record["memory_bytes"], int) and record["memory_bytes"] > 2**20
    node = platform.node()
    assert not node or node not in json.dumps(record)
    # Asked once per process: the second call is the first's record.
    assert _host.machine() is record


def test_a_slurm_job_is_named(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    _host.machine.cache_clear()
    try:
        assert _host.machine()["slurm_job_id"] == "4242"
    finally:
        _host.machine.cache_clear()


def test_linux_names_its_cpu_from_cpuinfo_then_lscpu(monkeypatch, tmp_path):
    monkeypatch.setattr(_host.platform, "system", lambda: "Linux")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: AMD EPYC 9R45\n")
    monkeypatch.setattr(_host, "Path", lambda _: cpuinfo)
    assert _host.cpu_model() == "AMD EPYC 9R45"
    cpuinfo.write_text("processor\t: 0\nCPU implementer\t: 0x41\n")
    lscpu = "Vendor ID: ARM\nModel name: Neoverse-V2\n"
    monkeypatch.setattr(_host, "_command", lambda arguments: lscpu)
    assert _host.cpu_model() == "ARM Neoverse-V2"


def test_an_unknown_system_or_a_failed_query_names_no_cpu(monkeypatch):
    monkeypatch.setattr(_host.platform, "system", lambda: "Plan9")
    assert _host.cpu_model() is None
    monkeypatch.setattr(_host.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_host, "_command", lambda arguments: None)
    assert _host.cpu_model() is None


def test_a_query_returns_its_output_or_none():
    assert _host._command([sys.executable, "-c", "print('M')"]) == "M\n"
    assert _host._command([sys.executable, "-c", "raise SystemExit(1)"]) is None
    assert _host._command(["solprop-no-such-command"]) is None


def test_an_unreadable_cpuinfo_falls_back_to_lscpu(monkeypatch, tmp_path):
    monkeypatch.setattr(_host.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_host, "Path", lambda _: tmp_path / "missing")
    monkeypatch.setattr(_host, "_command", lambda arguments: None)
    assert _host.cpu_model() is None


def test_windows_names_its_cpu_from_the_registry(monkeypatch):
    @contextlib.contextmanager
    def open_key(root, path):
        yield path

    registry = types.SimpleNamespace(
        HKEY_LOCAL_MACHINE=0,
        OpenKey=open_key,
        QueryValueEx=lambda key, name: (" AMD Ryzen 9 9950X ", 1),
    )
    monkeypatch.setitem(sys.modules, "winreg", registry)
    monkeypatch.setattr(_host.platform, "system", lambda: "Windows")
    assert _host.cpu_model() == "AMD Ryzen 9 9950X"
    # Off Windows the module is absent, and no name is read.
    monkeypatch.setitem(sys.modules, "winreg", None)
    assert _host._windows_model() is None


def test_memory_that_cannot_be_read_is_none(monkeypatch):
    def refuse(name):
        raise ValueError(name)

    monkeypatch.setattr(_host.os, "sysconf", refuse)
    assert _host._memory_bytes() is None
    monkeypatch.setattr(_host.os, "sysconf", lambda name: -1)
    assert _host._memory_bytes() is None


def test_usage_counts_from_the_clock_and_reports_the_peak_in_bytes():
    clock = _host.RunClock.start()
    block = bytearray(64 * 2**20)
    block[::4096] = b"\x01" * len(block[::4096])
    spent: Any = _host.usage(clock)
    assert 0.0 <= spent["wall_time_s"] < 60.0
    assert 0.0 <= spent["cpu_time_s"] < 60.0
    peak = spent["peak_rss_bytes"]
    # The scale is right: the process holds the 64 MiB just touched, and no machine
    # this runs on has a terabyte resident.
    assert peak is not None and 64 * 2**20 <= peak < 2**40
    del block


@pytest.mark.parametrize(
    ("output", "model"),
    [
        ("Vendor ID: ARM\nModel name: Neoverse-V2\n", "ARM Neoverse-V2"),
        ("Vendor ID: AuthenticAMD\nModel name: AMD EPYC 9R45\n", "AMD EPYC 9R45"),
        ("Vendor ID: ARM\nModel name: -\n", None),
    ],
)
def test_lscpu_names_a_core_by_vendor_and_part(output, model):
    assert _host.lscpu_model(output) == model
