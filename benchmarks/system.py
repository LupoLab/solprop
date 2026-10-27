# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Identify the hardware, set thread pools and observe contention.

Nothing here records a host name: the hardware is named by its CPU model and the
cores the run may use (SEP 38, section 5.3).
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import time
from pathlib import Path

import psutil

from benchmarks.records import Record, SuiteError, identity
from benchmarks.telemetry import (
    allocation_details,
    gpu_details,
    hardware_details,
    linux_topology,
)

__all__ = [
    "POOL_VARIABLES",
    "THREADING",
    "counters",
    "device_slug",
    "eligible_cpus",
    "git_receipt",
    "guard",
    "hardware",
    "interval",
    "native_changed",
    "numa_domains",
    "physical_cores",
    "slug",
    "thread_environment",
    "widths",
]

POOL_VARIABLES = (
    "SOLPROP_THREADS",
    "SOLPROP_OVERSUBSCRIBE",
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

THREADING: dict[str, dict[str, str]] = {
    # Unpinned, and idle threads sleep at once: robust on a shared machine.
    "passive": {
        "OMP_PROC_BIND": "false",
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_BLOCKTIME": "0",
        "GOMP_SPINCOUNT": "0",
    },
    # Pinned to neighbouring cores, and idle threads sleep at once.
    "passive-bound": {
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_BLOCKTIME": "0",
        "GOMP_SPINCOUNT": "0",
    },
    # Unpinned, with each OpenMP runtime's own waiting (a short spin, then sleep).
    "default": {"OMP_PROC_BIND": "false"},
    # Unpinned, and idle threads spin between parallel regions.
    "active": {"OMP_PROC_BIND": "false", "OMP_WAIT_POLICY": "ACTIVE"},
    # Threads pinned to neighbouring cores, with the runtime's own waiting.
    "bound": {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"},
    # Pinned, and spinning.
    "bound-active": {
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OMP_WAIT_POLICY": "ACTIVE",
    },
}
"""OpenMP placement and waiting policies a run may choose (SEP 38)."""

_MODEL_NOISE = re.compile(
    r"\((?:r|tm)\)|\bcpu\b|\bprocessor\b|\b\d+-core\b|@.*$", re.IGNORECASE
)


def eligible_cpus() -> tuple[int, ...]:
    """Read the process's CPU allocation."""
    try:
        return tuple(psutil.Process().cpu_affinity())
    except AttributeError, psutil.Error:
        return tuple(range(os.process_cpu_count() or os.cpu_count() or 1))


def physical_cores() -> int:
    """Count the physical cores in the allocation.

    On Linux the distinct (socket, core) pairs of the allocated CPUs, so that an
    allocation of one hardware thread per core counts every core; elsewhere the
    allocation scaled by the machine's ratio of cores to hardware threads.
    """
    allocation = eligible_cpus()
    if platform.system() == "Linux":
        rows = linux_topology(Path("/sys/devices/system/cpu"), allocation)
        pairs = {(row["socket"], row["core"]) for row in rows}
        if rows and all(None not in pair for pair in pairs):
            return len(pairs)
    logical = psutil.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    return max(1, len(allocation) * physical // logical)


def widths(value: str | None, production: int) -> tuple[int, ...]:
    """Resolve ``1``, ``production`` and integers, in order and without repeats."""
    if production < 1 or production > len(eligible_cpus()):
        raise SuiteError("the production width must fit the CPU allocation")
    resolved: list[int] = []
    for token in (value or "1,production").split(","):
        width = production if token.strip() == "production" else int(token)
        if width < 1:
            raise SuiteError("thread counts must be positive")
        if width not in resolved:
            resolved.append(width)
    return tuple(resolved)


def _instance_type() -> str | None:
    """Read a cloud instance type from the Linux DMI table, where it names one."""
    root = Path("/sys/class/dmi/id")
    try:
        vendor = (root / "sys_vendor").read_text(encoding="utf-8").strip()
        product = (root / "product_name").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if vendor.startswith("Amazon"):
        return f"aws {product}"
    return None


def hardware(
    production: int, instance_type: str | None = None, *, device: bool = False
) -> Record:
    """Describe the hardware without a host name, and derive its slug.

    A `device` run is named by its GPU, which then sets the pace; the host CPU is
    recorded beside it.
    """
    details = hardware_details(eligible_cpus())
    model = details["cpu_model"] or platform.processor() or platform.machine()
    cores = physical_cores()
    instance = instance_type or _instance_type()
    gpu = gpu_details() if device else {"gpu_model": None}
    if device and not isinstance(gpu["gpu_model"], str):
        raise SuiteError(f"cannot name the GPU: {gpu['gpu_model_reason']}")
    return {
        "numa_domains": numa_domains(details["cpu_topology"]),
        # The cores a run uses name the hardware set: on a cluster the CPU mask can
        # show the whole node while the allocation is a part of it.
        "slug": device_slug(gpu["gpu_model"], instance)
        if isinstance(gpu["gpu_model"], str)
        else slug(model, production, instance),
        "cpu_model": model,
        "gpu_model": gpu["gpu_model"],
        "physical_cores": cores,
        "logical_cpus": len(eligible_cpus()),
        "production": production,
        "memory_bytes": psutil.virtual_memory().total,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "instance_type": instance,
        "allocation": allocation_details(),
    }


def numa_domains(topology: list[Record] | None) -> int:
    """Count the memory (NUMA) domains the allocated CPUs span; 1 where unknown.

    A 16-core allocation inside one domain and one spread over eight share a CPU
    model and core count but not memory bandwidth, so rows compare by this too.
    """
    nodes = {node for row in topology or () for node in row.get("numa_nodes") or ()}
    return max(1, len(nodes))


def native_changed(directory: Path, commit: str, revision: str) -> bool:
    """Whether ``native/`` differs between a native build's commit and `revision`.

    Its documentation and tests do not count; a commit Git cannot find does.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(directory),
                "diff",
                "--quiet",
                commit,
                revision,
                "--",
                "native",
                ":(exclude,glob)native/**/*.md",
                ":(exclude)native/tests",
            ],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return True
    return result.returncode != 0


def slug(model: str, cores: int, instance: str | None = None) -> str:
    """Name a hardware set by CPU model and core count, e.g. ``apple-m5-max-18c``.

    A cloud instance adds its family (``aws m8g.4xlarge`` gives ``-aws-m8g``): one
    core design, such as Arm's Neoverse-V2, is sold in several clouds' instances.
    """
    if instance:
        family = re.sub(r"[^a-z0-9]+", "-", instance.lower().split(".")[0]).strip("-")
        return f"{slug(model, cores)}-{family}"
    words = re.sub(r"[^a-z0-9]+", "-", _MODEL_NOISE.sub(" ", model).lower())
    return f"{words.strip('-') or 'unknown-cpu'}-{cores}c"


def device_slug(model: str, instance: str | None = None) -> str:
    """Name a GPU hardware set by its model (``nvidia-a40``), plus any cloud family."""
    words = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-") or "unknown-gpu"
    if instance:
        family = re.sub(r"[^a-z0-9]+", "-", instance.lower().split(".")[0]).strip("-")
        return f"{words}-{family}"
    return words


def thread_environment(
    width: int, incoming: dict[str, str], threading: str = "passive"
) -> dict[str, str]:
    """Set every pool to `width`, and OpenMP's placement and waiting by `threading`.

    The pools are set before their libraries are imported. `threading` is a key of
    :data:`THREADING`; its OpenMP variables replace any the caller had.
    """
    if width < 1:
        raise SuiteError("thread counts must be positive")
    if threading not in THREADING:
        raise SuiteError(f"unknown threading policy {threading}")
    environment = incoming.copy()
    for key in POOL_VARIABLES:
        # A Kokkos tool left in the shell would profile the timed runs.
        if key != "SOLPROP_FFTW_PLAN":
            environment.pop(key, None)
    for key in (
        "SOLPROP_THREADS",
        "SOLPROP_NATIVE_THREADS",
        "SOLPROP_NUMBA_THREADS",
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[key] = str(width)
    if width == 1:
        for key in (
            "SOLPROP_NATIVE_KERNEL_THREADS",
            "SOLPROP_NATIVE_FFT_THREADS",
            "SOLPROP_NATIVE_RAMAN_FFT_THREADS",
            "SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS",
            "SOLPROP_FFTW_THREADS",
            "SOLPROP_TRANSVERSE_FFTW_THREADS",
            "SOLPROP_BLAS_THREADS",
        ):
            environment[key] = "1"
    environment.update(THREADING[threading])
    return environment


def counters() -> Record:
    """Read CPU and swap counters, with explicit unknowns where unsupported."""
    times = psutil.cpu_times(percpu=True)
    busy = [float(item.user + item.system) for item in times]
    try:
        swap = psutil.swap_memory()
        swapping: int | None = int(swap.sin + swap.sout)
        if os.name == "nt":
            swapping = None
    except psutil.Error, OSError:
        swapping = None
    return {"time": time.monotonic(), "busy": busy, "swap_bytes": swapping}


def interval(before: Record, after: Record, allocation: tuple[int, ...]) -> Record:
    """Score other processes' CPU use over an elapsed counter interval."""
    elapsed = after["time"] - before["time"]
    valid = elapsed > 0 and all(index < len(after["busy"]) for index in allocation)
    deltas = (
        [after["busy"][index] - before["busy"][index] for index in allocation]
        if valid
        else []
    )
    valid = valid and all(value >= 0 for value in deltas)
    usage = sum(deltas) / (elapsed * len(allocation)) if valid else None
    before_swap, after_swap = before["swap_bytes"], after["swap_bytes"]
    swap_delta = (
        after_swap - before_swap
        if before_swap is not None and after_swap is not None
        else None
    )
    # Where swap traffic is not reported (Windows), CPU use alone decides.
    known = usage is not None and (swap_delta is None or swap_delta >= 0)
    return {
        "external_cpu_fraction": usage,
        "swap_bytes": swap_delta,
        "confidence": "known" if known else "unknown",
        "quiet": known and usage is not None and usage <= 0.1 and not swap_delta,
    }


def guard(allocation: tuple[int, ...]) -> Record:
    """Take one one-second reading; it labels a busy machine and never waits."""
    before = counters()
    time.sleep(1.0)
    return interval(before, counters(), allocation)


def git_receipt(directory: Path) -> Record:
    """Record source identity, including a fingerprint of uncommitted contents."""

    def git(*arguments: str) -> bytes:
        return subprocess.check_output(
            ["git", "-C", str(directory), *arguments], stderr=subprocess.DEVNULL
        )

    try:
        revision = git("rev-parse", "HEAD").decode().strip()
        changes = git("diff", "HEAD", "--binary")
        # Untracked files count only where the timed code lives, so an editor's
        # settings directory does not make every run dirty.
        untracked = git(
            "ls-files", "--others", "--exclude-standard", "-z", "--",
            "src", "native", "benchmarks",
        ).split(b"\0")  # fmt: skip
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
