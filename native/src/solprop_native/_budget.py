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
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ThreadBudget:
    """The threads a process may run at once, and the limit that set it."""

    threads: int
    source: str


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


def cgroup_cpus(root: Path = Path("/")) -> int | None:
    """Return the tightest CPU quota on this process's control-group path, or ``None``.

    Versions 1 and 2, every group from the process's own up to the mount; a quota is
    rounded down, because a team that exceeds it is throttled whole.
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
        for directory in (path, *path.parents):
            quota = _quota(base / directory)
            if quota is not None:
                quotas.append(quota)
    return min(quotas) if quotas else None


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


def thread_budget(cpus: int) -> ThreadBudget:
    """Return the budget: the smallest of the CPUs, the quota and the two variables.

    ``cpus`` is the process's affinity mask as counted before Kokkos started.
    ``SOLPROP_THREADS`` is solprop's budget for every CPU backend and
    ``SOLPROP_NATIVE_THREADS`` this backend's own count; both only narrow it, except
    that under ``SOLPROP_OVERSUBSCRIBE=1`` the native count may exceed the CPUs and the
    quota (a thread-count invariance test on a small runner), never ``SOLPROP_THREADS``.
    """
    limits = [(max(1, cpus), "affinity")]
    quota = cgroup_cpus()
    if quota is not None:
        limits.append((quota, "cgroup quota"))
    requested = thread_count("SOLPROP_THREADS")
    if requested is not None:
        limits.append((requested, "SOLPROP_THREADS"))
    native = thread_count("SOLPROP_NATIVE_THREADS")
    oversubscribe = oversubscribing()
    if native is not None and oversubscribe:
        limits = [(native, "SOLPROP_NATIVE_THREADS")]
        if requested is not None:
            limits.append((requested, "SOLPROP_THREADS"))
    elif native is not None:
        limits.append((native, "SOLPROP_NATIVE_THREADS"))
    threads, source = min(limits, key=lambda limit: limit[0])
    return ThreadBudget(threads=threads, source=source)


@functools.cache
def process_budget() -> ThreadBudget:
    """Return this process's budget, read once: first when the package is imported.

    That first read comes before the native core starts an OpenMP runtime, which with
    ``OMP_PROC_BIND`` set would bind the main thread, and so its mask, to one CPU.
    """
    if sys.platform == "linux":
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.process_cpu_count() or 1
    return thread_budget(cpus)
