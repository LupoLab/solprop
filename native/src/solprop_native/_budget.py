# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The process's thread budget, read before the native core starts Kokkos (ADR-052).

A copy of solprop's own reader (``solprop._process``): this package never imports
solprop, and Kokkos is sized at import, before solprop could be asked. solprop's
``tests/test_process.py`` runs both copies on the same control groups and environment,
so they cannot drift apart.
"""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ThreadBudget:
    """The threads a process may run at once, and the limit that set it."""

    threads: int
    source: str
    oversubscribe: bool = False
    requested: int | None = None

    def explicit(self, count: int) -> int:
        """Clamp a count asked for: to the budget, or only to ``SOLPROP_THREADS``."""
        if not self.oversubscribe:
            return min(count, self.threads)
        return count if self.requested is None else min(count, self.requested)


def _quota(directory: Path) -> int | None:
    """Return one control group's CPU quota in whole CPUs, rounded down, or ``None``."""
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
    """Return the tightest CPU quota on this process's control-group path, or ``None``.

    Versions 1 and 2, every group from the process's own up to the mount; a quota is
    rounded down, because a team that exceeds it is throttled whole. This process's
    own (``root`` unset) is read once.
    """
    return _own_cgroup_cpus() if root is None else _read_cgroup_cpus(root)


@functools.cache
def _own_cgroup_cpus() -> int | None:
    return _read_cgroup_cpus(Path("/"))


def _read_cgroup_cpus(root: Path) -> int | None:
    """Read the tightest CPU quota under a filesystem root."""
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
        for directory in (path, *path.parents):
            quota = _quota(base / directory)
            if quota is not None:
                quotas.append(quota)
    return min(quotas) if quotas else None


def cores_in_masks(masks: Sequence[int], cpus: Sequence[int]) -> int:
    """Count the cores, given as CPU bitmasks, that hold any of ``cpus``.

    Parameters
    ----------
    masks : sequence of int
        One bitmask per physical core: bit ``n`` set when CPU ``n`` is one of its
        hardware threads.
    cpus : sequence of int
        The CPU indices the process may run on.

    Returns
    -------
    int
        The number of cores holding at least one of ``cpus``.
    """
    wanted = sum(1 << cpu for cpu in cpus)
    return sum(1 for mask in masks if mask & wanted)


def _linux_cores(cpus: Sequence[int], root: Path) -> int | None:
    """Count the distinct (package, core) pairs of ``cpus`` in a sysfs CPU tree.

    Parameters
    ----------
    cpus : sequence of int
        The CPU indices the process may run on.
    root : Path
        The sysfs CPU directory, ``/sys/devices/system/cpu`` on a running system.

    Returns
    -------
    int or None
        The number of physical cores, or ``None`` if any CPU's topology is unreadable.
    """
    pairs: set[tuple[str, str]] = set()
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
    """Count the physical cores holding ``cpus`` from Windows' processor table.

    Parameters
    ----------
    cpus : sequence of int
        The CPU indices the process may run on (processor group 0).

    Returns
    -------
    int or None
        The number of physical cores, or ``None`` if the table cannot be read.
    """
    import ctypes

    class _Information(ctypes.Structure):
        """One ``SYSTEM_LOGICAL_PROCESSOR_INFORMATION`` entry.

        The mask, the relationship and a 16-byte union aligned to 8.
        """

        _fields_ = (
            ("mask", ctypes.c_size_t),
            ("relationship", ctypes.c_int),
            ("union", ctypes.c_ulonglong * 2),
        )

    try:
        kernel32: Any = ctypes.windll.kernel32  # type: ignore[attr-defined]
        query: Any = kernel32.GetLogicalProcessorInformation
        size = ctypes.c_ulong(0)
        query(None, ctypes.byref(size))
        table = (_Information * (size.value // ctypes.sizeof(_Information)))()
        if not query(table, ctypes.byref(size)):
            return None
    except AttributeError, OSError:
        return None
    # Relationship 0 is RelationProcessorCore: one entry per physical core.
    masks = [int(entry.mask) for entry in table if entry.relationship == 0]
    return cores_in_masks(masks, cpus) or None


def physical_cores(
    cpus: Sequence[int], root: Path = Path("/sys/devices/system/cpu")
) -> int | None:
    """Count the physical cores that hold ``cpus``, or ``None`` where unknown.

    Two hardware threads of one core share its arithmetic units, and solprop's kernels
    keep those busy, so a second thread per core adds contention, not speed. Linux
    reads each CPU's package and core from ``root``; Windows asks the kernel. macOS on
    Apple silicon has one thread per core, so the CPUs are the cores.

    Parameters
    ----------
    cpus : sequence of int
        The CPU indices the process may run on.
    root : Path, optional
        The sysfs CPU directory, read on Linux.

    Returns
    -------
    int or None
        The number of physical cores, or ``None`` on other platforms or when the
        topology cannot be read.
    """
    if sys.platform.startswith("linux"):
        return _linux_cores(cpus, root)
    if sys.platform == "win32":
        return _windows_cores(cpus)
    return None


def thread_count(name: str) -> int | None:
    """Read a positive thread count from the environment, ``None`` when unset."""
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


def oversubscribing() -> bool:
    """Return whether ``SOLPROP_OVERSUBSCRIBE=1`` lets explicit counts pass the CPUs."""
    value = os.environ.get("SOLPROP_OVERSUBSCRIBE", "").strip()
    if value not in {"", "0", "1"}:
        msg = f"SOLPROP_OVERSUBSCRIBE must be 0 or 1, and is {value!r}"
        raise ValueError(msg)
    return value == "1"


def thread_budget(cpus: int, cores: int | None = None) -> ThreadBudget:
    """Return the budget: the smallest of the CPUs, their cores, the quota and
    ``SOLPROP_THREADS``.

    ``cpus`` is the process's affinity mask as counted before Kokkos started, and
    ``cores`` the physical cores that hold them (:func:`physical_cores`), which count
    only when ``SOLPROP_THREADS`` is unset. The same budget as solprop's (and so the
    same ``thread_budget`` in every backend's metadata); ``SOLPROP_NATIVE_THREADS`` is
    an explicit count within it (:func:`native_threads`).
    """
    limits = [(max(1, cpus), "affinity")]
    requested = thread_count("SOLPROP_THREADS")
    if requested is None and cores is not None:
        limits.append((cores, "physical cores"))
    quota = cgroup_cpus()
    if quota is not None:
        limits.append((quota, "cgroup quota"))
    if requested is not None:
        limits.append((requested, "SOLPROP_THREADS"))
    threads, source = min(limits, key=lambda limit: limit[0])
    return ThreadBudget(
        threads=threads,
        source=source,
        oversubscribe=oversubscribing(),
        requested=requested,
    )


def native_threads(budget: ThreadBudget) -> int:
    """Return the Kokkos pool: ``SOLPROP_NATIVE_THREADS`` within the budget, else it.

    The native count only narrows the budget, unless ``SOLPROP_OVERSUBSCRIBE=1`` lets
    it exceed the CPUs and the quota (a thread-count invariance test on a small
    runner); never ``SOLPROP_THREADS``.
    """
    native = thread_count("SOLPROP_NATIVE_THREADS")
    return budget.threads if native is None else budget.explicit(native)
