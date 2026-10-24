# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run bounded spawned workers with explicit threads and memory admission control."""

import ctypes
import multiprocessing as mp
import os
import pickle
import sys
import threading
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from itertools import chain
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import cast

from solprop._process import allowed_cpus, thread_budget
from solprop.exceptions import DriverError

__all__ = ["MemoryBudget", "Pool", "peak_rss"]

# A worker's pipe end: Connection on POSIX, but PipeConnection on Windows, which is a
# sibling class rather than a subclass, so the annotations name both there.
if sys.platform == "win32":
    from multiprocessing.connection import PipeConnection

    type _Pipe = Connection | PipeConnection  # pyright: ignore[reportRedeclaration]
else:
    type _Pipe = Connection

_ENVIRONMENT_LOCK = threading.Lock()
_THREAD_VARIABLES = (
    "NUMBA_NUM_THREADS",
    "SOLPROP_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class _WorkerError(DriverError):
    """Report an abnormal child exit separately from an invalid resource budget."""


def peak_rss() -> int:
    """Return this process's peak resident memory in bytes.

    Returns
    -------
    int
        Peak RSS, converting Linux KiB and retaining Darwin bytes; on Windows the
        peak working set.
    """
    if sys.platform == "win32":
        return _windows_peak_working_set()
    import resource  # noqa: PLC0415 - POSIX only; Windows has no such module

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value) if sys.platform == "darwin" else int(value) * 1024


class _ProcessMemoryCounters(ctypes.Structure):
    """Mirror Win32 ``PROCESS_MEMORY_COUNTERS`` for ``K32GetProcessMemoryInfo``."""

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _MemoryStatus(ctypes.Structure):
    """Mirror Win32 ``MEMORYSTATUSEX`` for ``GlobalMemoryStatusEx``."""

    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def _kernel32() -> ctypes.CDLL:
    """Load Windows' kernel32, which carries both memory queries, typed for them.

    Returns
    -------
    ctypes.CDLL
        The library handle.
    """
    kernel32 = ctypes.WinDLL("kernel32")  # pyright: ignore[reportAttributeAccessIssue]
    # The process pseudo-handle is pointer-sized ((HANDLE)-1): ctypes' default int
    # return type truncates it to 32 bits, and Windows refuses the truncated handle.
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.K32GetProcessMemoryInfo.restype = ctypes.c_int
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_uint32,
    ]
    return kernel32


def _windows_peak_working_set() -> int:
    """Read this process's peak working set, Windows' counterpart of peak RSS.

    Returns
    -------
    int
        Peak working set, bytes.

    Raises
    ------
    DriverError
        If Windows refuses the query.
    """
    kernel32 = _kernel32()
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = kernel32.GetCurrentProcess()
    if not kernel32.K32GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    ):
        raise DriverError("Windows refused the process memory query")
    return int(counters.PeakWorkingSetSize)


def _windows_physical_memory() -> int:
    """Read the installed physical memory on Windows.

    Returns
    -------
    int
        Total physical memory, bytes.

    Raises
    ------
    DriverError
        If Windows refuses the query.
    """
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not _kernel32().GlobalMemoryStatusEx(ctypes.byref(status)):
        raise DriverError("Windows refused the physical memory query")
    return int(status.ullTotalPhys)


def _memory_limit() -> int:
    """Read the smallest physical or scheduler memory allocation.

    Returns
    -------
    int
        Smallest available memory limit, bytes.
    """
    limits: list[int] = []
    if sys.platform == "win32":
        limits.append(_windows_physical_memory())
    else:
        limits.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        for path in (
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ):
            file = Path(path)
            if file.exists():
                value = file.read_text(encoding="utf-8").strip()
                if value.isdecimal() and int(value) > 0:
                    limits.append(int(value))
    slurm = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm and slurm.isdecimal():
        limits.append(int(slurm) * 1024**2)
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    if per_cpu and per_cpu.isdecimal():
        limits.append(int(per_cpu) * len(_cpus()) * 1024**2)
    return min(limits)


def _cpus() -> tuple[int, ...]:
    """Restrict CPU choices to affinity, scheduler and cgroup allocations.

    Returns
    -------
    tuple of int
        CPUs available within the allocation.
    """
    try:
        budget = thread_budget()
    except ValueError as error:
        raise DriverError(str(error)) from error
    available = allowed_cpus()
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and allocated.isdecimal():
        available = available[: int(allocated)]
    offset = int(os.environ.get("SOLPROP_SCAN_CPU_OFFSET", "0"))
    count = int(os.environ.get("SOLPROP_SCAN_CPU_COUNT", str(len(available))))
    if offset < 0 or count < 1:
        raise DriverError(
            "scan instance CPU offsets/counts must be non-negative/positive"
        )
    # The instance's own CPUs first, then the controller's budget (ADR-052) over them:
    # cut first, a budget smaller than the allocation would empty a later instance.
    return available[offset : offset + count][: budget.threads]


@dataclass(frozen=True)
class MemoryBudget:
    """Record the admission calculation without treating it as a hard RSS limit.

    Attributes
    ----------
    limit : int
        Effective memory allocation, bytes, positive.
    controller : int
        Controller resident memory and output allowance, bytes, non-negative.
    per_worker : int
        Worker estimate, bytes, positive; backend RSS is counted once.
    cores : int
        CPUs available to this controller, positive.
    threads : int
        Threads per worker, positive.
    """

    limit: int
    controller: int
    per_worker: int
    cores: int
    threads: int

    def workers(self, requested: int | None = None) -> int:
        """Choose a count within both CPU and memory allocations.

        Parameters
        ----------
        requested : int or None, optional
            Optional upper bound on workers, positive.

        Returns
        -------
        int
            Admitted workers after reserving 20 percent of memory.

        Raises
        ------
        DriverError
            If inputs are invalid or one worker cannot fit.
        """
        if (
            min(self.limit, self.per_worker, self.cores, self.threads) < 1
            or self.controller < 0
        ):
            raise DriverError("memory budget requires positive limits and worker sizes")
        if requested is not None and requested < 1:
            raise DriverError("workers must be positive")
        memory_count = (self.limit * 4 // 5 - self.controller) // self.per_worker
        count = min(self.cores // self.threads, memory_count)
        if requested is not None:
            count = min(count, requested)
        if count < 1:
            raise DriverError("the CPU/memory allocation cannot fit one scan worker")
        return count


def _serve(connection: _Pipe, work: Callable, cpus: tuple[int, ...]) -> None:
    """Process bounded commands inside one spawned interpreter.

    Parameters
    ----------
    connection : Connection or PipeConnection
        Worker command/result pipe.
    work : Callable
        Pickleable point operation and optional memory probe.
    cpus : tuple[int, ...]
        Allocated logical CPU identifiers; empty leaves affinity unchanged.
    """
    try:
        if cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        while True:
            message = connection.recv()
            if message is None:
                break
            operation, point = message
            if operation == "probe":
                probe = getattr(work, "probe", None)
                value = (
                    probe(point)
                    if callable(probe)
                    else {"baseline_rss": peak_rss(), "rss": peak_rss(), "ledger": 0}
                )
            else:
                value = work(point)
            connection.send((True, value))
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def _spawn(
    work: Callable, threads: int, cpus: tuple[int, ...]
) -> tuple[BaseProcess, _Pipe]:
    """Set the child environment before interpreter startup.

    Parameters
    ----------
    work : Callable
        Pickleable point operation and optional memory probe.
    threads : int
        Positive threads per worker, established before imports.
    cpus : tuple[int, ...]
        Allocated logical CPU identifiers; empty leaves affinity unchanged.

    Returns
    -------
    tuple
        Worker process and controller pipe.
    """
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_serve, args=(child, work, cpus))
    environment = dict.fromkeys(_THREAD_VARIABLES, str(threads)) | {
        "OMP_WAIT_POLICY": "passive"
    }
    # spawn takes its environment during start(), before replaying __main__ or
    # importing the target. Restore the parent's environment immediately afterwards.
    with _ENVIRONMENT_LOCK:
        previous = {key: os.environ.get(key) for key in environment}
        try:
            os.environ.update(environment)
            process.start()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    child.close()
    return process, cast(_Pipe, parent)


def _receive(connection: _Pipe) -> object:
    """Read a completed result or report abnormal worker termination.

    Parameters
    ----------
    connection : Connection or PipeConnection
        Worker command/result pipe.

    Returns
    -------
    object
        Worker result.
    """
    try:
        success, value = connection.recv()
    except EOFError as error:
        raise _WorkerError("scan worker exited without returning its result") from error
    if not success:
        raise _WorkerError(f"scan worker failed:\n{value}")
    return value


def _failed_work(work: Callable, point: object, error: _WorkerError) -> object:
    """Let a scan record a crashed point while generic executors still raise.

    Parameters
    ----------
    work : callable
        Worker operation with an optional failure-result constructor.
    point : object
        Description assigned to the failed worker.
    error : _WorkerError
        Child failure or unexpected pipe closure.

    Returns
    -------
    object
        Explicit failed result, if the work supports one.

    Raises
    ------
    DriverError
        If the worker operation has no failure-result contract.
    """
    failed = getattr(work, "failure", None)
    if not callable(failed):
        raise error
    return failed(point, str(error))


def _stop(process: BaseProcess, connection: _Pipe) -> None:
    """Close a worker and release its pipe even after a failure.

    Parameters
    ----------
    process : BaseProcess
        Spawned worker process.
    connection : Connection or PipeConnection
        Worker command/result pipe.
    """
    try:
        if process.is_alive():
            connection.send(None)
    except BrokenPipeError, EOFError, OSError:
        pass
    finally:
        connection.close()
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join()


@dataclass
class Pool:
    """Execute independent points in memory-sized spawned processes.

    Attributes
    ----------
    workers : int or None
        Positive maximum workers; None uses the CPU and memory admission rule.
    memory_limit : int or None
        Positive memory cap, bytes; None reads the allocation and physical memory.
    threads : int
        Threads per worker, positive and one by default.
    pin : bool
        Pin workers within Linux CPU affinity; unavailable on macOS.
    fresh : bool
        Start a fresh interpreter for every point instead of persistent workers.
    budget : MemoryBudget or None
        Last measured admission calculation, absent before execution.
    """

    workers: int | None = None
    memory_limit: int | None = None
    threads: int = 1
    pin: bool = True
    fresh: bool = False
    budget: MemoryBudget | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate the explicit resource controls."""
        if self.threads < 1 or (self.workers is not None and self.workers < 1):
            raise DriverError("workers and threads must be positive")
        if self.memory_limit is not None and self.memory_limit < 1:
            raise DriverError("memory_limit must be positive bytes")

    def validate(self, work: Callable) -> None:
        """Reject unpickleable builders before creating output or claiming work.

        Parameters
        ----------
        work : callable
            Worker operation, including its builder and reducers.

        Raises
        ------
        DriverError
            If standard-library pickle cannot transfer the callable.
        """
        try:
            pickle.dumps(work)
        except (pickle.PickleError, AttributeError, TypeError) as error:
            raise DriverError(
                "Pool requires importable pickleable builders and reducers"
            ) from error

    def run[T, R](self, points: Iterable[T], work: Callable[[T], R]) -> Iterator[R]:
        """Probe memory and run bounded work, yielding results as they finish.

        Parameters
        ----------
        points : iterable
            Independent point descriptions with global indices.
        work : callable
            Pickleable worker operation; an optional probe reports RSS and ledger.

        Yields
        ------
        object
            Results in completion order, each retaining its input identity.

        Raises
        ------
        DriverError
            If resource admission fails or a worker exits abnormally.
        """
        self.validate(work)
        iterator = iter(points)
        while True:
            try:
                first = next(iterator)
            except StopIteration:
                return
            try:
                available, count = self._admit(work, first)
                break
            except _WorkerError as error:
                yield cast(R, _failed_work(work, first, error))
        active: dict[_Pipe, tuple[BaseProcess, tuple[int, ...], T]] = {}

        pending = iter(chain((first,), iterator))
        try:
            for slot in range(count):
                try:
                    point = next(pending)
                except StopIteration:
                    break
                cpus = (
                    available[slot * self.threads : (slot + 1) * self.threads]
                    if self.pin
                    else ()
                )
                process, connection = _spawn(work, self.threads, cpus)
                active[connection] = process, cpus, point
                connection.send(("run", point))
            while active:
                for ready in wait(list(active), timeout=0.5):
                    connection = cast(_Pipe, ready)
                    process, cpus, previous_point = active[connection]
                    crashed = False
                    try:
                        result = cast(R, _receive(connection))
                    except _WorkerError as error:
                        result = cast(R, _failed_work(work, previous_point, error))
                        crashed = True
                    yield result
                    try:
                        point = next(pending)
                    except StopIteration:
                        _stop(process, connection)
                        del active[connection]
                        continue
                    if self.fresh or crashed:
                        _stop(process, connection)
                        del active[connection]
                        process, connection = _spawn(work, self.threads, cpus)
                        active[connection] = process, cpus, point
                    active[connection] = process, cpus, point
                    connection.send(("run", point))
        finally:
            for connection, (process, _, _) in active.items():
                _stop(process, connection)

    def _admit(self, work: Callable, first: object) -> tuple[tuple[int, ...], int]:
        """Probe the first point and calculate the worker allocation.

        Parameters
        ----------
        work : Callable
            Pickleable point operation and optional memory probe.
        first : object
            First pending point, used for memory admission.

        Returns
        -------
        tuple
            Allocated CPUs and admitted worker count.
        """
        available = _cpus()
        process, connection = _spawn(
            work, self.threads, available[: self.threads] if self.pin else ()
        )
        try:
            connection.send(("probe", first))
            report = cast(Mapping[str, int], _receive(connection))
        finally:
            _stop(process, connection)
        # Reserve both worker retention and controller transfer buffers per active
        # worker; resident backend pages are already represented by max().
        per_worker = max(
            report["rss"], report["baseline_rss"] + report["ledger"]
        ) + 2 * report.get("output_bytes", 0)
        limit = _memory_limit()
        if self.memory_limit is not None:
            limit = min(limit, self.memory_limit)
        self.budget = MemoryBudget(
            limit, peak_rss(), per_worker, len(available), self.threads
        )
        count = self.budget.workers(self.workers)
        return available, count
