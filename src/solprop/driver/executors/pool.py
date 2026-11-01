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
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import chain
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from solprop._interrupts import StopFlag, StopRequested
from solprop._process import allowed_cpus, thread_budget
from solprop.exceptions import DriverError
from solprop.log import PlaneSummary, ProgressReporter

__all__ = ["ControlledWork", "MemoryBudget", "PlaneScalars", "Pool", "peak_rss"]

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


# The worker pipe's messages. Controller to worker: ``("run", point)`` or
# ``("probe", point)``, the ``STOP`` string while a point runs, and ``None`` to exit.
# Worker to controller: ``(ok, value)`` once per command, and while a watched point runs
# ``(PROGRESS, call, args)``, one of that run's reporter calls (`_WorkerReporter`).
STOP = "stop"
PROGRESS = "progress"


class _WorkerError(DriverError):
    """Report an abnormal child exit separately from an invalid resource budget."""


@runtime_checkable
class ControlledWork(Protocol):
    """Point work a pool can stop and watch (SEP 41, S4 and S17).

    Optional, beside the ``probe`` and ``failure`` hooks: a pool runs any callable, and
    only work with these members is stopped through its worker's pipe and has its
    run's progress sent back. The scan's own work implements it.
    """

    @property
    def stop(self) -> StopFlag | None:
        """Return the caller's stop flag, read in the controlling process.

        Returns
        -------
        StopFlag or None
            The flag; ``None`` never stops.
        """
        ...

    @property
    def watched(self) -> bool:
        """Return whether a worker sends its point's progress to the controller.

        Returns
        -------
        bool
            True when the controller has a reporter for points.
        """
        ...

    def run_point(
        self, point: object, stop: StopFlag, reporter: ProgressReporter | None
    ) -> object:
        """Run one point in a worker under a stop flag and reporter of the worker's.

        Parameters
        ----------
        point : object
            The point description.
        stop : StopFlag
            Set when the controller asks the worker to stop.
        reporter : ProgressReporter or None
            Sends the run's calls to the controller; ``None`` when not watched.

        Returns
        -------
        object
            The point's result.
        """
        ...

    def dispatched(self, point: object) -> None:
        """Record in the controller that a point was sent to a worker.

        Parameters
        ----------
        point : object
            The point description.
        """
        ...

    def relay(self, point: object, call: str, args: tuple[object, ...]) -> None:
        """Make, in the controller, a reporter call a worker sent for a point.

        Parameters
        ----------
        point : object
            The point description the worker is running.
        call : str
            The reporter method: summary, started, plane, finished or interrupted.
        args : tuple
            Its arguments, as `_WorkerReporter` sends them.
        """
        ...


@dataclass(frozen=True)
class PlaneScalars:
    """The four scalars of a plane's summary, as a worker sends them to its controller.

    Satisfies :class:`solprop.log.PlaneSummary`; a frozen copy rather than the backend's
    own summary so that what crosses the pipe is small and always pickles.

    Attributes
    ----------
    n_steps : int
        Steps taken to reach the plane, accepted and rejected together.
    n_rejected : int
        Rejected steps among them.
    max_norm : float
        Largest scaled error norm over the accepted steps.
    wall_time : float
        Seconds the plane took, s.

    Examples
    --------
    >>> from solprop.log import PlaneSummary
    >>> isinstance(PlaneScalars(8, 0, 0.4, 0.1), PlaneSummary)
    True
    """

    n_steps: int
    n_rejected: int
    max_norm: float
    wall_time: float

    @classmethod
    def of(cls, summary: PlaneSummary) -> "PlaneScalars":  # noqa: UP037 - runtime hook evaluates the class body
        """Copy the scalars a reporter reads from a plane's summary.

        Parameters
        ----------
        summary : solprop.log.PlaneSummary
            The summary the run handed its reporter.

        Returns
        -------
        PlaneScalars
            Its picklable copy.
        """
        return cls(
            int(summary.n_steps),
            int(summary.n_rejected),
            float(summary.max_norm),
            float(summary.wall_time),
        )


def _is_stop(message: object) -> bool:
    """Return whether a message from the controller asks the worker to stop.

    Parameters
    ----------
    message : object
        What arrived on the pipe.

    Returns
    -------
    bool
        True for the stop message.
    """
    return isinstance(message, str) and message == STOP


class _Channel:
    """The worker's end of its pipe: commands in, a stop flag, progress out.

    :meth:`is_set` is the stop flag the worker's run polls before every trial step. It
    reads what has arrived without waiting, remembers a stop, and keeps any other
    message for :meth:`command`, so polling never consumes a command.

    Parameters
    ----------
    connection : Connection or PipeConnection
        The worker's end of the pipe.
    """

    def __init__(self, connection: _Pipe) -> None:
        """Hold the pipe, with no stop and nothing held back."""
        self.connection = connection
        self._held: deque[object] = deque()
        self._stopped = False

    def is_set(self) -> bool:
        """Return whether the controller has asked the worker to stop.

        Returns
        -------
        bool
            True once a stop message has arrived; it stays set.
        """
        while not self._stopped and self.connection.poll():
            message = self.connection.recv()
            if _is_stop(message):
                self._stopped = True
            else:
                self._held.append(message)
        return self._stopped

    def command(self) -> object:
        """Return the next command, waiting for it; a stop message is remembered.

        Returns
        -------
        object
            ``("run" or "probe", point)``, or ``None`` to exit.
        """
        while not self._held:
            message = self.connection.recv()
            if _is_stop(message):
                self._stopped = True
            else:
                return message
        return self._held.popleft()

    def send(self, call: str, args: tuple[object, ...]) -> None:
        """Send one reporter call to the controller.

        Parameters
        ----------
        call : str
            The reporter method.
        args : tuple
            Its picklable arguments.
        """
        self.connection.send((PROGRESS, call, args))


class _WorkerReporter:
    """Send a worker's run's reporter calls to the controller, never a field.

    ``step``, on the hot path, is not sent. A plane carries its position and the
    :class:`PlaneScalars` of its summary. An interruption carries its position,
    whether the worker's stop flag caused it, and its text: the exception itself may
    not pickle, and the stop flag that stopped it is the worker's, not the caller's.

    Parameters
    ----------
    channel : _Channel
        The worker's pipe.
    """

    def __init__(self, channel: _Channel) -> None:
        """Hold the pipe."""
        self.channel = channel

    def summary(self, text: str) -> None:
        """Send the run's summary block.

        Parameters
        ----------
        text : str
            See :meth:`solprop.log.SummarisingReporter.summary`.
        """
        self.channel.send("summary", (text,))

    def started(self, z_start: float, z_end: float, n_planes: int) -> None:
        """Send the run's start.

        Parameters
        ----------
        z_start, z_end : float
            See :meth:`solprop.log.ProgressReporter.started`.
        n_planes : int
            See :meth:`solprop.log.ProgressReporter.started`.
        """
        self.channel.send("started", (float(z_start), float(z_end), int(n_planes)))

    def step(self, z: float, dz: float, accepted: bool) -> None:
        """Send nothing: a step is on the hot path.

        Parameters
        ----------
        z, dz : float
            See :meth:`solprop.log.ProgressReporter.step`.
        accepted : bool
            See :meth:`solprop.log.ProgressReporter.step`.
        """

    def plane(self, z: float, summary: PlaneSummary) -> None:
        """Send a plane's position and summary.

        Parameters
        ----------
        z : float
            See :meth:`solprop.log.ProgressReporter.plane`.
        summary : solprop.log.PlaneSummary
            See :meth:`solprop.log.ProgressReporter.plane`.
        """
        self.channel.send("plane", (float(z), PlaneScalars.of(summary)))

    def finished(self, total: float) -> None:
        """Send the run's end.

        Parameters
        ----------
        total : float
            See :meth:`solprop.log.ProgressReporter.finished`.
        """
        self.channel.send("finished", (float(total),))

    def interrupted(self, z: float, error: BaseException) -> None:
        """Send that the run stopped before its end.

        Parameters
        ----------
        z : float
            See :meth:`solprop.log.InterruptibleReporter.interrupted`.
        error : BaseException
            See :meth:`solprop.log.InterruptibleReporter.interrupted`.
        """
        self.channel.send(
            "interrupted",
            (
                float(z),
                isinstance(error, StopRequested),
                f"{type(error).__name__}: {error}",
            ),
        )


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
    channel = _Channel(connection)
    try:
        if cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        while True:
            message = channel.command()
            if message is None:
                break
            operation, point = cast(tuple[str, object], message)
            if operation == "probe":
                probe = getattr(work, "probe", None)
                value = (
                    probe(point)
                    if callable(probe)
                    else {"baseline_rss": peak_rss(), "rss": peak_rss(), "ledger": 0}
                )
            elif isinstance(work, ControlledWork):
                reporter = _WorkerReporter(channel) if work.watched else None
                value = work.run_point(point, channel, reporter)
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


@dataclass(frozen=True)
class _Progress:
    """One reporter call a worker sent while its point runs."""

    call: str
    args: tuple[object, ...]


def _receive(connection: _Pipe) -> object:
    """Read a completed result or report abnormal worker termination.

    Parameters
    ----------
    connection : Connection or PipeConnection
        Worker command/result pipe.

    Returns
    -------
    object
        Worker result, or a `_Progress` call of the run still in progress.
    """
    try:
        message = connection.recv()
    except EOFError as error:
        raise _WorkerError("scan worker exited without returning its result") from error
    if len(message) == 3 and isinstance(message[0], str) and message[0] == PROGRESS:
        return _Progress(message[1], message[2])
    success, value = message
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

        Work that implements :class:`ControlledWork` is stopped and watched through
        each worker's pipe (SEP 41, S4 and S17). Once its caller's flag is set, the
        pool sends every busy worker the stop message, which that worker's run polls
        before every trial step, collects their results, dispatches nothing more and
        shuts the workers down; the abrupt ``terminate`` of a worker that does not
        exit stays the last resort. While a watched point runs, its worker sends the
        run's reporter calls, which the pool hands to :meth:`ControlledWork.relay`.

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
        session = _Session(self, work)
        try:
            session.start(chain((first,), iterator), available, count)
            yield from cast(Iterator[R], session.drain())
        finally:
            session.close()

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


class _Session:
    """The workers of one :meth:`Pool.run`, the points they run, and the stop.

    Parameters
    ----------
    policy : Pool
        The pool's resource controls.
    work : callable
        The point work, :class:`ControlledWork` or not.
    """

    def __init__(self, policy: Pool, work: Callable) -> None:
        """Hold the work, with no worker started and no stop asked for."""
        self.policy = policy
        self.work = work
        self.controlled = work if isinstance(work, ControlledWork) else None
        self.stop = None if self.controlled is None else self.controlled.stop
        self.active: dict[_Pipe, tuple[BaseProcess, tuple[int, ...], object]] = {}
        self.pending: Iterator[object] = iter(())
        self.stopping = False

    def start(
        self, pending: Iterator[object], available: tuple[int, ...], count: int
    ) -> None:
        """Start up to `count` workers, each with its first point.

        Parameters
        ----------
        pending : iterator
            The points, claimed lazily as workers become free.
        available : tuple of int
            The CPUs the workers are pinned to, `threads` each.
        count : int
            The admitted workers.
        """
        self.pending = pending
        threads = self.policy.threads
        for slot in range(count):
            try:
                point = next(pending)
            except StopIteration:
                break
            cpus = (
                available[slot * threads : (slot + 1) * threads]
                if self.policy.pin
                else ()
            )
            process, connection = _spawn(self.work, threads, cpus)
            self._dispatch(process, connection, cpus, point)

    def drain(self) -> Iterator[object]:
        """Collect results as they finish, and keep the free workers busy.

        Yields
        ------
        object
            Results in completion order.
        """
        # A short wait while a flag can stop the pool: it is read between messages.
        timeout = 0.5 if self.stop is None else 0.1
        while self.active:
            self._halt()
            for ready in wait(list(self.active), timeout=timeout):
                connection = cast(_Pipe, ready)
                _, _, point = self.active[connection]
                crashed = False
                try:
                    received = _receive(connection)
                except _WorkerError as error:
                    received = _failed_work(self.work, point, error)
                    crashed = True
                if isinstance(received, _Progress):
                    # Only a controlled work's worker sends progress (`_serve`).
                    controlled = cast(ControlledWork, self.controlled)
                    controlled.relay(point, received.call, received.args)
                    continue
                yield received
                self._next(connection, crashed=crashed)

    def close(self) -> None:
        """Shut every remaining worker down."""
        for connection, (process, _, _) in self.active.items():
            _stop(process, connection)
        self.active.clear()

    def _dispatch(
        self,
        process: BaseProcess,
        connection: _Pipe,
        cpus: tuple[int, ...],
        point: object,
    ) -> None:
        """Send a point to a worker and record it.

        Parameters
        ----------
        process : BaseProcess
            The worker.
        connection : Connection or PipeConnection
            Its pipe.
        cpus : tuple of int
            Its CPUs.
        point : object
            The point it runs.
        """
        self.active[connection] = process, cpus, point
        connection.send(("run", point))
        if self.controlled is not None:
            self.controlled.dispatched(point)

    def _next(self, connection: _Pipe, *, crashed: bool) -> None:
        """Give a worker that returned its result the next point, or shut it down.

        Parameters
        ----------
        connection : Connection or PipeConnection
            The worker's pipe.
        crashed : bool
            Whether the worker died, so that a fresh one replaces it.
        """
        process, cpus, _ = self.active[connection]
        point: object = None
        if not self._halt():
            point = next(self.pending, _EXHAUSTED)
        if self.stopping or point is _EXHAUSTED:
            _stop(process, connection)
            del self.active[connection]
            return
        if self.policy.fresh or crashed:
            _stop(process, connection)
            del self.active[connection]
            process, connection = _spawn(self.work, self.policy.threads, cpus)
        self._dispatch(process, connection, cpus, point)

    def _halt(self) -> bool:
        """Ask every busy worker to stop once the caller's flag is set.

        Returns
        -------
        bool
            Whether the pool is stopping.
        """
        if not self.stopping and self.stop is not None and self.stop.is_set():
            self.stopping = True
            for busy in self.active:
                # A dead worker is collected as a crash.
                with suppress(BrokenPipeError, EOFError, OSError):
                    busy.send(STOP)
        return self.stopping


_EXHAUSTED = object()
