# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The CPUs this process may run on, as the scheduler granted them.

Linux reports affinity per thread. An OpenMP runtime started with ``OMP_PROC_BIND``
set -- GNU libgomp, which pyFFTW's and Kokkos's builds load -- binds the **main**
thread to the first of its places when it starts, so after the first parallel region
``os.sched_getaffinity(0)`` names one CPU although the process was granted many. A
thread pool sized from that answer runs on one core (measured: an 18-core allocation
gave the Numba kernels one thread). The affinity is therefore read once, when
``solprop`` is imported, before any of its runtimes start, and the later answer is
believed only when it is not that collapse.

Threads inherit their creator's affinity, so a pool started from the bound main thread
puts every worker on that one CPU: eighteen Numba workers on one core ran 3.7 times
slower than one thread (measured). :func:`unbound` lifts the binding while a pool is
started and puts it back after.
"""

import contextlib
import functools
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


def _query() -> tuple[int, ...]:
    """Return the calling thread's CPUs: its affinity, or the usable count's range.

    Returns
    -------
    tuple of int
        The CPU indices, sorted.
    """
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return tuple(range(os.process_cpu_count() or 1))


def _at_import() -> tuple[int, ...]:
    """Return the process's CPUs as first seen, before an OpenMP runtime bound it.

    ``solprop_native``, when imported before ``solprop``, starts its OpenMP runtime
    first; it records the process's CPUs just before (``PROCESS_CPUS``), and that wider
    set is the grant when this thread's answer is one of its CPUs.

    Returns
    -------
    tuple of int
        Sorted CPU indices.
    """
    own = _query()
    earlier = getattr(sys.modules.get("solprop_native"), "PROCESS_CPUS", ())
    if len(earlier) > len(own) and set(own) <= set(earlier):
        return tuple(earlier)
    return own


_AT_IMPORT = _at_import()


def _size_numba() -> None:
    """Name the grant's size to Numba when an OpenMP binding would shrink its pool.

    Numba sizes its pool from the importing thread's affinity, once. When an OpenMP
    runtime has already bound this thread to one CPU (``solprop_native`` imported
    first), that pool would hold one thread. ``NUMBA_NUM_THREADS`` is set to the
    grant's size first, unless Numba is already imported or the user set it.
    """
    if (
        len(_AT_IMPORT) > len(_query())
        and "numba" not in sys.modules
        and "NUMBA_NUM_THREADS" not in os.environ
    ):
        os.environ["NUMBA_NUM_THREADS"] = str(len(_AT_IMPORT))


_size_numba()


def allowed_cpus() -> tuple[int, ...]:
    """Return the CPUs this process may run on, undoing an OpenMP main-thread binding.

    The current affinity, unless ``OMP_PROC_BIND`` is set and the current answer is a
    single CPU of the wider set the process had when ``solprop`` was imported: that is
    the runtime's binding of the main thread and not the scheduler's grant, so the
    import-time set is returned. A deliberate narrowing to several CPUs, or any
    narrowing without ``OMP_PROC_BIND``, is believed.

    Returns
    -------
    tuple of int
        Sorted CPU indices, at least one.

    Examples
    --------
    >>> from solprop._process import allowed_cpus
    >>> len(allowed_cpus()) >= 1
    True
    """
    current = _query()
    binding = os.environ.get("OMP_PROC_BIND", "").strip().lower()
    if (
        binding not in {"", "false"}
        and len(current) == 1
        and len(_AT_IMPORT) > 1
        and set(current) <= set(_AT_IMPORT)
    ):
        return _AT_IMPORT
    return current


def _collapsed() -> bool:
    """Return whether the main thread's affinity is an OpenMP binding's single CPU.

    Returns
    -------
    bool
        True when :func:`allowed_cpus` differs from the thread's current affinity.
    """
    return allowed_cpus() != _query()


@dataclass(frozen=True)
class ThreadBudget:
    """The threads a process may run at once, and the limit that set it.

    Attributes
    ----------
    threads : int
        The budget, at least one.
    source : str
        The tightest limit: ``"affinity"``, ``"cgroup quota"`` or the name of the
        environment variable that set it.
    oversubscribe : bool
        Whether ``SOLPROP_OVERSUBSCRIBE`` lets an explicit count exceed the CPUs and
        the quota (see :meth:`explicit`).
    requested : int or None
        ``SOLPROP_THREADS``, which nothing exceeds, or ``None`` when it is unset.
    """

    threads: int
    source: str
    oversubscribe: bool = False
    requested: int | None = None

    def explicit(self, count: int) -> int:
        """Clamp a count someone asked for to what this budget allows.

        Normally the budget itself. With ``SOLPROP_OVERSUBSCRIBE=1`` only
        ``SOLPROP_THREADS`` holds, so a test can run more threads than the machine
        has CPUs on purpose (a thread-count invariance test on a small runner).

        Parameters
        ----------
        count : int
            The positive count asked for.

        Returns
        -------
        int
            The count allowed.

        Examples
        --------
        >>> from solprop._process import ThreadBudget
        >>> ThreadBudget(4, "affinity").explicit(8)
        4
        >>> ThreadBudget(4, "affinity", oversubscribe=True).explicit(8)
        8
        """
        if not self.oversubscribe:
            return min(count, self.threads)
        return count if self.requested is None else min(count, self.requested)


def _quota(directory: Path) -> int | None:
    """Return one control group's CPU quota in whole CPUs, or ``None`` for none.

    Parameters
    ----------
    directory : pathlib.Path
        The group's directory: ``cpu.max`` (version 2) or ``cpu.cfs_quota_us`` and
        ``cpu.cfs_period_us`` (version 1).

    Returns
    -------
    int or None
        The quota divided by its period, rounded down and at least one.
    """
    try:
        limit = directory / "cpu.max"
        if limit.is_file():
            value, period = limit.read_text(encoding="utf-8").split()
            return None if value == "max" else max(1, int(value) // int(period))
        quota, period = directory / "cpu.cfs_quota_us", directory / "cpu.cfs_period_us"
        if quota.is_file() and period.is_file():
            value = int(quota.read_text(encoding="utf-8"))
            if value > 0:
                return max(1, value // int(period.read_text(encoding="utf-8")))
    except OSError, ValueError:
        return None
    return None


def cgroup_cpus(root: Path | None = None) -> int | None:
    """Return the tightest CPU quota on this process's control-group path.

    A quota limits CPU *time*, not which CPUs: a team of threads that exceeds it is
    throttled for the rest of each period, and a team that waits on itself at every
    step stalls whole. The quota is therefore rounded **down** -- 1.5 CPUs is one
    thread. Every group from the process's own up to the hierarchy's root is read,
    for control-group versions 1 and 2, because a parent's quota binds its children.

    This process's own quota (``root`` unset) is read once and kept: every backend
    build and every scan admission asks, and a running process's groups do not
    change under it.

    Parameters
    ----------
    root : pathlib.Path, optional
        A filesystem root to read under instead, for tests; never kept.

    Returns
    -------
    int or None
        Whole CPUs, at least one, or ``None`` where no quota is set or none can be
        read (not Linux, or no control groups).

    Examples
    --------
    >>> from solprop._process import cgroup_cpus
    >>> cgroup_cpus() is None or cgroup_cpus() >= 1
    True
    """
    return _own_cgroup_cpus() if root is None else _read_cgroup_cpus(root)


@functools.cache
def _own_cgroup_cpus() -> int | None:
    """Return this process's CPU quota, read once.

    Returns
    -------
    int or None
        As :func:`cgroup_cpus`.
    """
    return _read_cgroup_cpus(Path("/"))


def _read_cgroup_cpus(root: Path) -> int | None:
    """Read the tightest CPU quota on this process's path under a filesystem root.

    Parameters
    ----------
    root : pathlib.Path
        The filesystem root.

    Returns
    -------
    int or None
        As :func:`cgroup_cpus`.
    """
    try:
        lines = (root / "proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    mounts = root / "sys/fs/cgroup"
    quotas: list[int] = []
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        controllers, path = fields[1], Path(fields[2].lstrip("/"))
        if not controllers:
            base = mounts
        elif "cpu" in controllers.split(","):
            base = mounts / controllers
            base = base if base.is_dir() else mounts / "cpu"
        else:
            continue
        # A container without its own cgroup namespace names the host's path, which
        # its /sys/fs/cgroup does not have; the parents end at the mount, its own group.
        for directory in (path, *path.parents):
            quota = _quota(base / directory)
            if quota is not None:
                quotas.append(quota)
    return min(quotas) if quotas else None


def cores_in_masks(masks: Sequence[int], cpus: Sequence[int]) -> int:
    """Count the cores, given as CPU bitmasks, that hold any of ``cpus``."""
    wanted = sum(1 << cpu for cpu in cpus)
    return sum(1 for mask in masks if mask & wanted)


def _linux_cores(cpus: Sequence[int], root: Path) -> int | None:
    pairs = set()
    for cpu in cpus:
        topology = root / f"cpu{cpu}" / "topology"
        try:
            pairs.add(
                (
                    (topology / "physical_package_id").read_text(encoding="utf-8"),
                    (topology / "core_id").read_text(encoding="utf-8"),
                )
            )
        except OSError:
            return None
    return len(pairs) or None


def _windows_cores(cpus: Sequence[int]) -> int | None:  # pragma: no cover - Windows
    import ctypes  # noqa: PLC0415

    class _Information(ctypes.Structure):
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION: the mask, the relationship and a
        # 16-byte union that is aligned to 8.
        _fields_ = (
            ("mask", ctypes.c_size_t),
            ("relationship", ctypes.c_int),
            ("union", ctypes.c_ulonglong * 2),
        )

    try:
        kernel32 = ctypes.windll.kernel32  # pyright: ignore[reportAttributeAccessIssue]
        query = kernel32.GetLogicalProcessorInformation
        size = ctypes.c_ulong(0)
        query(None, ctypes.byref(size))
        table = (_Information * (size.value // ctypes.sizeof(_Information)))()
        if not query(table, ctypes.byref(size)):
            return None
    except AttributeError, OSError:
        return None
    # Relationship 0 is RelationProcessorCore: one entry per physical core.
    masks = [entry.mask for entry in table if entry.relationship == 0]
    return cores_in_masks(masks, cpus) or None


def physical_cores(
    cpus: Sequence[int], root: Path = Path("/sys/devices/system/cpu")
) -> int | None:
    """Count the physical cores that hold ``cpus``, or ``None`` where unknown.

    Two hardware threads of one core share its arithmetic units, and solprop's kernels
    keep those busy, so a second thread per core adds contention, not speed. Linux
    reads each CPU's package and core from ``root``; Windows asks the kernel. macOS on
    Apple silicon has one thread per core, so the CPUs are the cores.
    """
    if sys.platform.startswith("linux"):
        return _linux_cores(cpus, root)
    if sys.platform == "win32":
        return _windows_cores(cpus)
    return None


def thread_count(name: str) -> int | None:
    """Read a positive thread count from the environment.

    Parameters
    ----------
    name : str
        The variable, for example ``"SOLPROP_THREADS"``.

    Returns
    -------
    int or None
        The count, or ``None`` when the variable is unset or blank.

    Raises
    ------
    ValueError
        If the variable is set to something that is not a positive integer.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        count = int(value)
    except ValueError as error:
        msg = f"{name} must be a thread count, and is {value!r}"
        raise ValueError(msg) from error
    if count < 1:
        msg = f"{name} must be positive, and is {value!r}"
        raise ValueError(msg)
    return count


def thread_budget() -> ThreadBudget:
    """Return how many threads this process may run at once (ADR-052).

    The smallest of the CPUs it may run on (:func:`allowed_cpus`: ``taskset``, a batch
    scheduler's allocation), the physical cores that hold them
    (:func:`physical_cores`, unless ``SOLPROP_THREADS`` is set), its control groups'
    CPU quota (:func:`cgroup_cpus`) and ``SOLPROP_THREADS``. Every pool solprop sizes
    is held within it, and so is any count asked for explicitly unless
    ``SOLPROP_OVERSUBSCRIBE=1``
    (:meth:`ThreadBudget.explicit`); the pools take turns, so each may use the whole
    budget.

    Returns
    -------
    ThreadBudget
        The budget and the limit that set it; on a tie, the earlier in that order.

    Raises
    ------
    ValueError
        If ``SOLPROP_THREADS`` is not a positive thread count, or
        ``SOLPROP_OVERSUBSCRIBE`` is neither ``0`` nor ``1``.

    Examples
    --------
    >>> from solprop._process import thread_budget
    >>> thread_budget().threads >= 1
    True
    """
    cpus = allowed_cpus()
    limits = [(len(cpus), "affinity")]
    requested = thread_count("SOLPROP_THREADS")
    if requested is None and (cores := physical_cores(cpus)) is not None:
        limits.append((cores, "physical cores"))
    quota = cgroup_cpus()
    if quota is not None:
        limits.append((quota, "cgroup quota"))
    if requested is not None:
        limits.append((requested, "SOLPROP_THREADS"))
    threads, source = min(limits, key=lambda limit: limit[0])
    return ThreadBudget(
        threads=max(1, threads),
        source=source,
        oversubscribe=oversubscribing(),
        requested=requested,
    )


def oversubscribing() -> bool:
    """Return whether ``SOLPROP_OVERSUBSCRIBE`` lets explicit counts pass the CPUs.

    Returns
    -------
    bool
        ``True`` for ``1``, ``False`` for ``0`` or unset.

    Raises
    ------
    ValueError
        For any other value.
    """
    value = os.environ.get("SOLPROP_OVERSUBSCRIBE", "").strip()
    if value not in {"", "0", "1"}:
        msg = f"SOLPROP_OVERSUBSCRIBE must be 0 or 1, and is {value!r}"
        raise ValueError(msg)
    return value == "1"


@contextlib.contextmanager
def unbound() -> Iterator[None]:
    """Give the calling thread the process's CPUs while a thread pool is started.

    Workers created inside the block inherit every CPU the process was granted, not
    the one CPU an OpenMP runtime bound the main thread to; the binding is restored on
    leaving, so the runtime's own placement of the main thread is unchanged. Nothing
    happens where the affinity cannot be set or was not collapsed.

    Yields
    ------
    None
        Inside the block the calling thread may run on the whole grant.

    Examples
    --------
    >>> from solprop._process import unbound
    >>> with unbound():
    ...     pass
    """
    if not hasattr(os, "sched_setaffinity") or not _collapsed():
        yield
        return
    bound = _query()
    os.sched_setaffinity(0, _AT_IMPORT)
    try:
        yield
    finally:
        os.sched_setaffinity(0, bound)


def release() -> bool:
    """Give the calling thread the process's CPUs for good, undoing an OpenMP binding.

    For ``numba_cpu``: FFTW's and Numba's threads start whenever a plan first runs or a
    pool first grows, long after construction, and each inherits the thread that starts
    it. When the native backend's OpenMP runtime, loaded first under ``OMP_PROC_BIND``,
    has bound the main thread to one CPU, every such thread would share it: the
    comb-to-continuum window took 45 s against 5.4 s on 64 Zen 5 cores (SEP 28).
    The native backend's own threads keep their placement; only the main thread moves.

    Returns
    -------
    bool
        Whether the affinity was changed.

    Examples
    --------
    >>> from solprop._process import release
    >>> isinstance(release(), bool)
    True
    """
    if not hasattr(os, "sched_setaffinity") or not _collapsed():
        return False
    os.sched_setaffinity(0, _AT_IMPORT)
    return True
