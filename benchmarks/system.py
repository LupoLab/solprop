# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Record allocations, thread policies and bounded contention observations."""

from __future__ import annotations

import dataclasses
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import psutil

from benchmarks.records import (
    Record,
    SuiteError,
    atomic_record,
    identity,
    read_record,
)

__all__ = [
    "POOL_VARIABLES",
    "Profile",
    "admission",
    "counters",
    "effective_memory",
    "eligible_cpus",
    "git_receipt",
    "guard",
    "interval",
    "profile_for",
    "save_profile",
    "thread_environment",
    "topology",
]

POOL_VARIABLES = (
    "SOLPROP_NATIVE_THREADS",
    "SOLPROP_NATIVE_KERNEL_THREADS",
    "SOLPROP_NATIVE_FFT_THREADS",
    "SOLPROP_NATIVE_RAMAN_FFT_THREADS",
    "SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS",
    "SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES",
    "SOLPROP_NUMBA_THREADS",
    "SOLPROP_FFTW_THREADS",
    "SOLPROP_TRANSVERSE_FFTW_THREADS",
    "SOLPROP_BLAS_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_WAIT_POLICY",
    "KMP_BLOCKTIME",
    "GOMP_SPINCOUNT",
    "SOLPROP_FFTW_PLAN",
    "KOKKOS_TOOLS_LIBS",
)


def eligible_cpus() -> tuple[int, ...]:
    """Read the process allocation, never an OpenMP worker's narrowed placement."""
    try:
        return tuple(psutil.Process().cpu_affinity())
    except AttributeError, psutil.Error:
        return tuple(range(os.process_cpu_count() or os.cpu_count() or 1))


def topology() -> Record:
    """Collect inexpensive portable hardware facts without a hostname."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "allocation": list(eligible_cpus()),
        "memory_bytes": psutil.virtual_memory().total,
        "processor": platform.processor(),
        "core_residency": None,
        "core_residency_reason": "not measured by ordinary collection",
    }


@dataclasses.dataclass(frozen=True)
class Profile:
    """Freeze production width, allocation and policy before measurement."""

    alias: str
    revision: int
    production: int
    allocation: tuple[int, ...]
    topology_hash: str
    policy: str = "budget"
    explicit: bool = True

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.alias):
            raise SuiteError("machine alias must be a public-safe identifier")
        if self.production < 1 or self.production > len(self.allocation):
            raise SuiteError("production width exceeds the declared CPU allocation")
        if self.policy not in ("budget", "all-pools-N", "blas1", "as-installed"):
            raise SuiteError(f"unsupported thread policy {self.policy}")

    @property
    def digest(self) -> str:
        """Identify the full profile revision."""
        return identity(dataclasses.asdict(self))


def profile_for(path: Path | None, *, standard: bool) -> Profile:
    """Load an explicit profile or make a clearly provisional quick profile."""
    current = topology()
    if path is not None:
        values = read_record(path)
        values["allocation"] = tuple(values["allocation"])
        profile = Profile(**values)
        if profile.topology_hash != identity(current):
            raise SuiteError(
                "machine topology/allocation differs from the saved profile"
            )
        if standard and not profile.explicit:
            raise SuiteError(
                "standard collection needs a profile with an explicit production width"
            )
        return profile
    if standard:
        raise SuiteError(
            "standard collection requires --machine-profile with a production width"
        )
    # An unidentified machine is useful for quick diagnostics but cannot become
    # an adoption baseline solely because another host has the same CPU model.
    return Profile(
        os.environ.get("SOLPROP_BENCH_MACHINE", "unidentified"),
        1,
        len(eligible_cpus()),
        eligible_cpus(),
        identity(current),
        explicit=False,
    )


def save_profile(
    path: Path, alias: str, production: int, policy: str = "budget"
) -> Profile:
    """Write an explicit profile without replacing an earlier revision."""
    profile = Profile(
        alias, 1, production, eligible_cpus(), identity(topology()), policy
    )
    atomic_record(path, dataclasses.asdict(profile), immutable=True)
    return profile


def thread_environment(
    width: int, policy: str, incoming: dict[str, str]
) -> dict[str, str]:
    """Set pools before importing their libraries, retaining size-based choices."""
    if width < 1:
        raise SuiteError("thread counts must be positive")
    environment = incoming.copy()
    if policy == "as-installed":
        return environment
    for key in POOL_VARIABLES:
        if key not in ("SOLPROP_FFTW_PLAN", "KOKKOS_TOOLS_LIBS"):
            environment.pop(key, None)
    for key in (
        "SOLPROP_NATIVE_THREADS",
        "SOLPROP_NUMBA_THREADS",
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        environment[key] = str(width)
    for key in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[key] = str(1 if policy == "blas1" else width)
    if width == 1 or policy == "all-pools-N":
        for key in (
            "SOLPROP_NATIVE_KERNEL_THREADS",
            "SOLPROP_NATIVE_FFT_THREADS",
            "SOLPROP_NATIVE_RAMAN_FFT_THREADS",
            "SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS",
            "SOLPROP_FFTW_THREADS",
            "SOLPROP_TRANSVERSE_FFTW_THREADS",
            "SOLPROP_BLAS_THREADS",
        ):
            environment[key] = str(width)
    if policy == "blas1":
        environment["SOLPROP_BLAS_THREADS"] = "1"
    environment.update(
        OMP_PROC_BIND="false",
        OMP_WAIT_POLICY="PASSIVE",
        KMP_BLOCKTIME="0",
        GOMP_SPINCOUNT="0",
    )
    return environment


def effective_memory() -> int:
    """Constrain currently available RAM by a Linux cgroup when observable."""
    available = int(psutil.virtual_memory().available)
    # Unified cgroups: a container's current usage must be subtracted from its
    # limit. The host's free memory alone is not its allocation.
    for parent in (Path("/sys/fs/cgroup"),):
        try:
            limit = (parent / "memory.max").read_text(encoding="utf-8").strip()
            used = int((parent / "memory.current").read_text(encoding="utf-8"))
            if limit != "max":
                available = min(available, max(0, int(limit) - used))
        except OSError, ValueError:
            pass
    return available


def admission(
    estimate: int | None, limit: int | None, *, allow_unestimated: bool = False
) -> Record:
    """Refuse unknown or excessive allocations before expensive construction."""
    available = effective_memory()
    budget = min(int(0.8 * available), limit if limit is not None else available)
    admitted = estimate is not None and estimate <= budget
    if estimate is None and allow_unestimated and limit is not None:
        admitted = limit <= int(0.8 * available)
    return {
        "admitted": admitted,
        "estimate_bytes": estimate,
        "limit_bytes": budget,
        "available_bytes": available,
        "reserve_fraction": 0.2,
        "exploratory": estimate is None,
        "reason": None
        if admitted
        else "unestimated"
        if estimate is None
        else "insufficient memory",
    }


def counters() -> Record:
    """Read counters with explicit unknowns for unsupported OS observations."""
    times = psutil.cpu_times(percpu=True)
    busy = [float(item.user + item.system) for item in times]
    try:
        swap = psutil.swap_memory()
        swapping: int | None = int(swap.sin + swap.sout)
        reason = None
        if os.name == "nt":
            swapping, reason = None, "Windows does not expose swap I/O counters"
    except psutil.Error, OSError:
        swapping, reason = None, "swap I/O counters unavailable"
    return {
        "time": time.monotonic(),
        "busy": busy,
        "swap_bytes": swapping,
        "swap_reason": reason,
        "available_bytes": effective_memory(),
    }


def interval(
    before: Record, after: Record, allocation: tuple[int, ...], own_cpu: float = 0.0
) -> Record:
    """Score external CPU use over an actual elapsed counter interval."""
    elapsed = after["time"] - before["time"]
    valid = elapsed > 0 and all(index < len(after["busy"]) for index in allocation)
    deltas = (
        [after["busy"][index] - before["busy"][index] for index in allocation]
        if valid
        else []
    )
    valid = valid and all(value >= 0 for value in deltas)
    usage = (
        max(0.0, sum(deltas) - own_cpu) / (elapsed * len(allocation)) if valid else None
    )
    before_swap, after_swap = before["swap_bytes"], after["swap_bytes"]
    swap_delta = (
        int(after_swap) - int(before_swap)
        if before_swap is not None and after_swap is not None
        else None
    )
    confidence = (
        "known" if valid and swap_delta is not None and swap_delta >= 0 else "unknown"
    )
    return {
        "external_cpu_fraction": usage,
        "swap_bytes": swap_delta,
        "confidence": confidence,
        "quiet": confidence == "known"
        and usage is not None
        and usage <= 0.1
        and swap_delta == 0,
    }


def guard(
    allocation: tuple[int, ...], *, standard: bool, force: bool = False
) -> Record:
    """Take one quick interval or bounded standard counter intervals."""
    if force:
        return {"quiet": False, "confidence": "unknown", "forced": True, "seconds": 0.0}
    started = time.monotonic()
    observations: list[Record] = []
    before = counters()
    for _ in range(5 if standard else 1):
        time.sleep(1.0)
        after = counters()
        observations.append(interval(before, after, allocation))
        before = after
        if not standard or (
            len(observations) >= 2 and all(row["quiet"] for row in observations[-2:])
        ):
            break
    selected = observations[-2:] if standard else observations
    return {
        "quiet": all(row["quiet"] for row in selected),
        "confidence": "known"
        if all(row["confidence"] == "known" for row in selected)
        else "unknown",
        "seconds": time.monotonic() - started,
        "intervals": observations,
        "threshold_fraction": 0.1,
    }


def git_receipt(directory: Path) -> Record:
    """Record source identity, including a fingerprint of uncommitted contents."""

    def git(*arguments: str) -> bytes:
        return subprocess.check_output(
            ["git", "-C", str(directory), *arguments], stderr=subprocess.DEVNULL
        )

    try:
        revision = git("rev-parse", "HEAD").decode().strip()
        changes = git("diff", "HEAD", "--binary")
        untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        from benchmarks.records import file_hash  # noqa: PLC0415

        fingerprints = {
            name.decode(): file_hash(directory / os.fsdecode(name))
            for name in untracked
            if name and (directory / os.fsdecode(name)).is_file()
        }
        return {
            "revision": revision,
            "dirty": bool(changes or fingerprints),
            "dirty_hash": identity({"diff": changes.hex(), "untracked": fingerprints}),
        }
    except OSError, subprocess.CalledProcessError:
        return {
            "revision": None,
            "dirty": None,
            "reason": "not an accessible Git checkout",
        }
