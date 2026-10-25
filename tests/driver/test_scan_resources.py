# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise allocation limits and worker failure cleanup without large jobs."""

import ctypes
import importlib.util
import multiprocessing
import os
import subprocess
import sys
from collections.abc import Generator
from multiprocessing import connection
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import cast
from unittest.mock import Mock

import pytest

from solprop import _process
from solprop.driver.executors import pool
from solprop.driver.executors.queue import Queue
from solprop.driver.executors.slurm import Slurm
from solprop.exceptions import DriverError


@pytest.mark.parametrize("value", ["max", "0", "4294967296"])
def test_linux_memory_limits_ignore_unlimited_or_zero_cgroups(
    tmp_path, monkeypatch, value
):
    record = tmp_path / "memory.max"
    record.write_text(value)
    monkeypatch.setattr(pool, "Path", lambda name: record)
    monkeypatch.setattr(pool.sys, "platform", "linux")
    monkeypatch.setattr(
        pool.os,
        "sysconf",
        lambda key: 2**20 if key == "SC_PHYS_PAGES" else 4096,
        raising=False,  # Windows has no sysconf; this test simulates Linux
    )
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    assert pool._memory_limit() == 2**32


@pytest.mark.parametrize("offset,count", [(-1, 1), (0, 0)])
def test_cpu_instance_refuses_invalid_slice(monkeypatch, offset, count):
    monkeypatch.setattr(_process, "cgroup_cpus", lambda: None)
    monkeypatch.setenv("SOLPROP_SCAN_CPU_OFFSET", str(offset))
    monkeypatch.setenv("SOLPROP_SCAN_CPU_COUNT", str(count))
    with pytest.raises(DriverError, match="offsets/counts"):
        pool._cpus()


def test_unlimited_cpu_quota_preserves_allocated_affinity(monkeypatch):
    monkeypatch.setattr(_process, "cgroup_cpus", lambda: None)
    monkeypatch.delenv("SOLPROP_THREADS", raising=False)
    monkeypatch.setattr(
        pool.os, "sched_getaffinity", lambda pid: {1, 3, 5, 7}, raising=False
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("SOLPROP_SCAN_CPU_OFFSET", "1")
    monkeypatch.setenv("SOLPROP_SCAN_CPU_COUNT", "2")
    assert pool._cpus() == (3, 5)


@pytest.mark.parametrize("limit", ["quota", "variable"])
def test_the_controllers_thread_budget_limits_the_cpus_shared_out(monkeypatch, limit):
    """A CPU quota or ``SOLPROP_THREADS`` holds the scan's workers within the budget."""
    monkeypatch.setattr(
        pool.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False
    )
    monkeypatch.setattr(
        _process, "cgroup_cpus", lambda: 3 if limit == "quota" else None
    )
    if limit == "variable":
        monkeypatch.setenv("SOLPROP_THREADS", "3")
    else:
        monkeypatch.delenv("SOLPROP_THREADS", raising=False)
    for name in (
        "SLURM_CPUS_PER_TASK",
        "SOLPROP_SCAN_CPU_OFFSET",
        "SOLPROP_SCAN_CPU_COUNT",
    ):
        monkeypatch.delenv(name, raising=False)
    assert pool._cpus() == (0, 1, 2)


@pytest.mark.parametrize(("budget", "expected"), [(None, (4, 5, 6, 7)), ("2", (4, 5))])
def test_a_slurm_instance_keeps_its_cpus_under_a_smaller_budget(
    monkeypatch, budget, expected
):
    """The second of two instances on eight CPUs: its slice first, then the budget.

    Cutting the mask to the budget before the offset left this instance no CPUs.
    """
    monkeypatch.setattr(
        pool.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False
    )
    monkeypatch.setattr(_process, "cgroup_cpus", lambda: None)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SOLPROP_SCAN_CPU_OFFSET", "4")
    monkeypatch.setenv("SOLPROP_SCAN_CPU_COUNT", "4")
    if budget is None:
        monkeypatch.delenv("SOLPROP_THREADS", raising=False)
    else:
        monkeypatch.setenv("SOLPROP_THREADS", budget)
    assert pool._cpus() == expected


def test_an_invalid_thread_budget_is_a_driver_error(monkeypatch):
    monkeypatch.setenv("SOLPROP_THREADS", "none")
    with pytest.raises(DriverError, match="SOLPROP_THREADS"):
        pool._cpus()


def test_cpus_count_the_processors_where_there_is_no_affinity_call(
    thread_budget_lifted, monkeypatch
):
    """macOS has no `sched_getaffinity`; the count is then `process_cpu_count`.

    Taken out on every platform, so the branch each operating system does not take
    in production is still exercised on it.
    """
    monkeypatch.setattr(_process, "cgroup_cpus", lambda: None)
    monkeypatch.delattr(pool.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(pool.os, "process_cpu_count", lambda: 3)
    for name in (
        "SLURM_CPUS_PER_TASK",
        "SOLPROP_SCAN_CPU_OFFSET",
        "SOLPROP_SCAN_CPU_COUNT",
    ):
        monkeypatch.delenv(name, raising=False)
    assert pool._cpus() == (0, 1, 2)


@pytest.mark.parametrize(
    "options", [{"threads": 0}, {"workers": -1}, {"memory_limit": 0}]
)
def test_pool_refuses_invalid_resource_controls(options):
    with pytest.raises(DriverError, match="positive"):
        pool.Pool(**options)


@pytest.mark.parametrize("options", [{"controller": -1}, {"limit": 0}, {"threads": 0}])
def test_budget_refuses_invalid_inputs(options):
    values = {
        "limit": 1000,
        "controller": 0,
        "per_worker": 100,
        "cores": 4,
        "threads": 1,
    }
    with pytest.raises(DriverError, match="positive"):
        pool.MemoryBudget(**(values | options)).workers()


def test_budget_refuses_nonpositive_requested_count():
    with pytest.raises(DriverError, match="workers must be positive"):
        pool.MemoryBudget(1000, 0, 100, 4, 1).workers(0)


def test_worker_dispatch_probes_generic_callable_and_sets_affinity(monkeypatch):
    parent, child = multiprocessing.Pipe()
    affinity = []
    monkeypatch.setattr(
        pool.os,
        "sched_setaffinity",
        lambda pid, cpus: affinity.append((pid, cpus)),
        raising=False,
    )
    monkeypatch.setattr(pool, "peak_rss", lambda: 1234)
    parent.send(("probe", -3))
    parent.send(("run", -3))
    parent.send(None)
    pool._serve(cast(Connection, child), abs, (2, 4))
    assert parent.recv() == (True, {"baseline_rss": 1234, "rss": 1234, "ledger": 0})
    assert parent.recv() == (True, 3)
    assert affinity == [(0, (2, 4))]
    assert child.closed
    parent.close()


def test_worker_reports_exception_and_closes_pipe():
    parent, child = multiprocessing.Pipe()
    parent.send(("run", "invalid"))
    pool._serve(cast(Connection, child), abs, ())
    success, message = parent.recv()
    assert not success
    assert "TypeError" in message
    assert child.closed
    parent.close()


def test_worker_failure_without_scan_failure_contract_propagates():
    error = pool._WorkerError("lost worker")
    with pytest.raises(DriverError, match="lost worker"):
        pool._failed_work(abs, 1, error)


@pytest.mark.parametrize("broken", [False, True])
def test_stubborn_worker_cleanup_escalates_and_closes_pipe(broken):
    process = Mock(spec=BaseProcess)
    process.is_alive.return_value = True
    connection = Mock(spec=Connection)
    if broken:
        connection.send.side_effect = BrokenPipeError("closed")
    pool._stop(process, connection)
    connection.close.assert_called_once()
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.join.call_count == 3


def test_cleanup_of_a_worker_that_already_exited_only_closes_and_joins():
    """A worker gone before its stop is sent nothing, and never terminated.

    Real scans reach this branch only when a worker dies first, which is timing: a
    coverage lane has missed it, so it is exercised directly.
    """
    process = Mock(spec=BaseProcess)
    process.is_alive.return_value = False
    connection = Mock(spec=Connection)
    pool._stop(process, connection)
    connection.send.assert_not_called()
    connection.close.assert_called_once()
    process.join.assert_called_once()
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_memory_probe_uses_smallest_explicit_and_physical_limit(monkeypatch):
    process = Mock(spec=BaseProcess)
    connection = Mock(spec=Connection)
    monkeypatch.setattr(pool, "_cpus", lambda: (1, 2, 3, 4))
    monkeypatch.setattr(pool, "_spawn", lambda *args: (process, connection))
    monkeypatch.setattr(
        pool,
        "_receive",
        lambda pipe: {
            "baseline_rss": 100,
            "rss": 150,
            "ledger": 100,
            "output_bytes": 50,
        },
    )
    monkeypatch.setattr(pool, "_stop", lambda *args: None)
    monkeypatch.setattr(pool, "_memory_limit", lambda: 1000)
    monkeypatch.setattr(pool, "peak_rss", lambda: 100)
    policy = pool.Pool(memory_limit=2000, pin=False)
    assert policy._admit(abs, 1) == ((1, 2, 3, 4), 2)
    assert policy.budget is not None
    assert policy.budget.per_worker == 300
    assert policy.budget.limit == 1000


def test_darwin_memory_query_spawns_no_process(monkeypatch):
    # macOS answers the physical-memory query through sysconf, as Linux does, so an
    # admission spawns no sysctl process (SEP 25 section 6.3).
    def refuse(*args, **kwargs):
        raise AssertionError("a memory query spawned a process")

    monkeypatch.setattr(pool.sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(
        pool.os,
        "sysconf",
        lambda key: {"SC_PHYS_PAGES": 2, "SC_PAGE_SIZE": 4096}[key],
        raising=False,  # Windows has no sysconf; this test simulates macOS
    )
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    assert pool._memory_limit() == 8192


class EchoWork:
    def __call__(self, point):
        if point < 0:
            raise RuntimeError("point failed after admission")
        return point, os.getpid()

    def failure(self, point, error):
        assert "point failed after admission" in error
        return point, -1


@pytest.mark.parametrize("fresh", [False, True])
def test_pool_reuses_or_replaces_workers_as_requested(monkeypatch, fresh):
    monkeypatch.setattr(pool.Pool, "_admit", lambda *args: ((0,), 1))
    monkeypatch.setenv("SOLPROP_THREADS", "3")
    output = list(pool.Pool(fresh=fresh, pin=False).run([0, 1], EchoWork()))
    assert [point for point, pid in output] == [0, 1]
    assert (output[0][1] != output[1][1]) is fresh
    assert os.environ["SOLPROP_THREADS"] == "3"


def test_pool_replaces_crashed_worker_and_keeps_pending_points(monkeypatch):
    monkeypatch.setattr(pool.Pool, "_admit", lambda *args: ((0,), 1))
    output = list(pool.Pool(pin=False).run([0, -1, 2], EchoWork()))
    assert [point for point, pid in output] == [0, -1, 2]
    assert output[1][1] == -1
    assert output[0][1] != output[2][1]


def test_pool_does_not_spawn_more_workers_than_points(monkeypatch):
    monkeypatch.setattr(pool.Pool, "_admit", lambda *args: ((0, 1, 2), 3))
    assert len(list(pool.Pool(pin=False).run([0], EchoWork()))) == 1


def test_pool_closes_active_worker_when_consumer_stops(monkeypatch):
    monkeypatch.setattr(pool.Pool, "_admit", lambda *args: ((0,), 1))
    original = pool._stop
    stopped = []

    def stop(process, connection):
        original(process, connection)
        stopped.append(process.is_alive())

    monkeypatch.setattr(pool, "_stop", stop)
    results = pool.Pool(pin=False).run([0, 1], EchoWork())
    assert isinstance(results, Generator)
    assert next(results)[0] == 0
    results.close()
    assert stopped == [False]


def test_receive_propagates_worker_traceback():
    parent, child = multiprocessing.Pipe()
    child.send((False, "worker traceback"))
    with pytest.raises(DriverError, match="worker traceback"):
        pool._receive(cast(Connection, parent))
    parent.close()
    child.close()


class _FakeKernel32:
    """Answer the two Windows memory queries as kernel32 would, or refuse them."""

    def __init__(self, *, succeed: bool = True) -> None:
        self.succeed = succeed

    def GetCurrentProcess(self) -> int:  # noqa: N802 - the Win32 name
        return -1

    def K32GetProcessMemoryInfo(self, process, counters, size) -> int:  # noqa: N802
        assert process == -1
        assert size == counters._obj.cb
        counters._obj.PeakWorkingSetSize = 3 * 2**20
        return int(self.succeed)

    def GlobalMemoryStatusEx(self, status) -> int:  # noqa: N802
        assert status._obj.dwLength > 0
        status._obj.ullTotalPhys = 16 * 2**30
        return int(self.succeed)


def test_windows_reads_peak_working_set_and_physical_memory(monkeypatch):
    monkeypatch.setattr(pool, "_kernel32", _FakeKernel32)
    monkeypatch.setattr(pool.sys, "platform", "win32")
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    assert pool.peak_rss() == 3 * 2**20
    assert pool._memory_limit() == 16 * 2**30


def test_windows_memory_queries_report_a_refusal(monkeypatch):
    monkeypatch.setattr(pool, "_kernel32", lambda: _FakeKernel32(succeed=False))
    with pytest.raises(DriverError, match="process memory query"):
        pool._windows_peak_working_set()
    with pytest.raises(DriverError, match="physical memory query"):
        pool._windows_physical_memory()


@pytest.mark.skipif(sys.platform != "win32", reason="the real Windows query")
def test_windows_peak_working_set_is_read_from_the_process():
    # The process handle is pointer-sized; truncated, Windows refuses the query.
    assert pool._windows_peak_working_set() > 2**20
    assert pool._windows_physical_memory() > 2**30


def test_kernel32_is_loaded_only_on_windows():
    if pool.sys.platform == "win32":
        assert pool._kernel32() is not None
    else:
        with pytest.raises(AttributeError):
            pool._kernel32()


def test_kernel32_preserves_pointer_sized_process_handles(monkeypatch):
    kernel32 = Mock(
        spec=ctypes.CDLL, GetCurrentProcess=Mock(), K32GetProcessMemoryInfo=Mock()
    )
    loader = Mock(return_value=kernel32)
    monkeypatch.setattr(pool.ctypes, "WinDLL", loader, raising=False)
    assert pool._kernel32() is kernel32
    loader.assert_called_once_with("kernel32")
    # HANDLE must retain all pointer bits, including the current-process -1 value.
    assert kernel32.GetCurrentProcess.restype is ctypes.c_void_p
    assert kernel32.GetCurrentProcess.argtypes == []
    assert kernel32.K32GetProcessMemoryInfo.restype is ctypes.c_int
    assert kernel32.K32GetProcessMemoryInfo.argtypes == [
        ctypes.c_void_p,
        ctypes.POINTER(pool._ProcessMemoryCounters),
        ctypes.c_uint32,
    ]


def test_windows_pipe_type_accepts_both_connection_classes(monkeypatch):
    class WindowsPipe:
        pass

    # Windows' PipeConnection is a sibling, so accepting only Connection is wrong.
    specification = importlib.util.spec_from_file_location(
        "pool_windows", pool.__file__
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, specification.name, module)
    with monkeypatch.context() as windows:
        windows.setattr(sys, "platform", "win32")
        windows.setattr(connection, "PipeConnection", WindowsPipe, raising=False)
        specification.loader.exec_module(module)
    assert module._Pipe.__value__ == Connection | WindowsPipe


@pytest.mark.parametrize(
    "build", [lambda: Queue("queue"), lambda: Slurm("scan.py")], ids=["queue", "slurm"]
)
def test_the_posix_only_executors_refuse_windows_by_name(monkeypatch, build):
    monkeypatch.setattr("sys.platform", "win32")
    with pytest.raises(DriverError, match="not supported on Windows"):
        build()
