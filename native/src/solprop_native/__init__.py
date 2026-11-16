# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Kokkos execution-space backend for :mod:`solprop`.

Its workspaces support multimode scalar and two-component propagation, Kerr, plasma,
time-domain Raman, weighted control and statistics. Native states remain resident in
their execution space between calls; observations are owning host ``complex128``
arrays.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

# NumPy, and the BLAS it carries, load before the process start below sets the OpenMP
# environment and starts the native runtime.
import numpy  # noqa: F401  # pyright: ignore[reportUnusedImport]

from ._bind import proc_bind
from ._budget import (
    cgroup_cpus,
    native_threads,
    physical_cores,
    thread_budget,
    thread_count,
    top_tier_cores,
)
from ._cpu import check_cpu_baseline, select_variant
from ._execution import ExecutionRecord
from ._execution import builtin as _builtin_execution
from ._profile import Choice, Process, choose
from ._wait import requested_wait, wait_policy

if platform.system() == "Darwin" and "pyfftw" in sys.modules:
    # solprop does not load pyFFTW; this is reached when a script, or a package it
    # imported, did. Its macOS wheels carry a private OpenMP runtime of their own.
    raise RuntimeError(
        "on macOS, import solprop_native before pyfftw; the reverse order is "
        "unsafe with wheels that bundle another OpenMP runtime. pyfftw is already "
        "loaded in this process: restart Python and import solprop_native before "
        "pyfftw or any package that imports it. backend='numpy_ref', the "
        "reference, doesn't load solprop_native, and is unthreaded and far slower"
    )

check_cpu_baseline(Path(__file__).parent)

# The process's CPUs before the native core starts Kokkos: with OMP_PROC_BIND set, its
# OpenMP runtime binds the main thread to one CPU, and solprop, imported after this
# package, would otherwise read that binding as the process's grant (its
# ``_process.allowed_cpus``).
PROCESS_CPUS: tuple[int, ...] = (
    tuple(sorted(os.sched_getaffinity(0))) if sys.platform == "linux" else ()
)

# The process's thread budget (solprop ADR-052): the smallest of those CPUs, their
# physical cores, its control groups' quota and SOLPROP_THREADS, as solprop reads it.
# The Kokkos pool is SOLPROP_NATIVE_THREADS within it, or all of it.
_CPUS = len(PROCESS_CPUS) or os.process_cpu_count() or 1
THREAD_BUDGET = thread_budget(
    _CPUS, physical_cores(PROCESS_CPUS or tuple(range(_CPUS)))
)
_BUDGET_THREADS = native_threads(THREAD_BUDGET)

# A wheel with more than one build registers the chosen one as ``._core`` below.
_STEM = select_variant(Path(__file__).parent)

# This machine's calibration (solprop SEP 46), when one was published for this build
# and allocation: an exact row's process settings apply here, before the runtime
# starts, unless the user set them; each workspace takes its workspace settings.
PROFILE: Choice = choose(
    packages=[Path(entry) for entry in __path__],
    stem=_STEM,
    cpus=PROCESS_CPUS,
    quota=cgroup_cpus(),
)
# A scan's worker is given SOLPROP_THREADS by solprop's Pool, its partition: there the
# row of the workers' allocation may narrow the pool within it.
_PROFILE_THREADS = PROFILE.threads()
_REQUESTED_THREADS = (
    min(_PROFILE_THREADS, THREAD_BUDGET.threads)
    if _PROFILE_THREADS is not None
    and (
        thread_count("SOLPROP_THREADS") is None
        or os.environ.get("SOLPROP_CALIBRATION_ROLE", "").strip() == "worker"
    )
    and thread_count("SOLPROP_NATIVE_THREADS") is None
    else _BUDGET_THREADS
)

# How the OpenMP runtime's idle threads wait: a short spin within native work, unless
# the user or a calibration chose otherwise (ADR-071). The runtime reads this when the
# core loads it, just below.
_WAIT = wait_policy(
    threads=_REQUESTED_THREADS,
    top_tier=top_tier_cores(),
    preferred=PROFILE.wait() or requested_wait(),
)
WAIT_POLICY = _WAIT.description

# Whether its threads are bound: not, unless the user placed them. Kokkos reads this
# when the core starts it, just below, and otherwise prints a notice.
PROC_BIND = proc_bind()

# A wheel with more than one build registers the chosen one as ``._core``, so nothing
# else knows which it is. The loader finds PyInit__core from the module's name, not the
# file's.
if _STEM != "_core":
    _file = next(Path(__file__).parent.glob(f"{_STEM}.*"))
    _spec = importlib.util.spec_from_file_location(f"{__name__}._core", _file)
    if _spec is None or _spec.loader is None:  # pragma: no cover - a file has one
        raise ImportError(f"cannot load {_file}")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)

from . import _core  # noqa: E402  (only after the processor check)

if TYPE_CHECKING:
    from .backend import NativeBackend

__all__ = [
    "BuildInfo",
    "NativeBackend",
    "__version__",
    "available_device_memory",
    "build_info",
    "check_build",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = version("solprop-native")


def _pool_request(name: str, what: str = "thread count") -> int:
    """Read an optional positive process-start setting; zero takes the default."""
    value = os.environ.get(name, "").strip()
    if not value:
        return 0
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive {what}, got {value!r}") from error
    if count < 1:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return count


def _band_request() -> int:
    """Read ``SOLPROP_NATIVE_BAND_TILES``: -1 when unset (the build's band column)."""
    name = "SOLPROP_NATIVE_BAND_TILES"
    value = os.environ.get(name, "").strip()
    if not value:
        return -1
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError(
            f"{name} must be a nonnegative count, got {value!r}"
        ) from error
    if count < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
    return count


# The runtime as one record (solprop SEP 52 section 5.3): the Kokkos pools, the spin,
# and a host OpenBLAS fitted to the Kokkos pool unless its thread count is set
# explicitly. A pthreads OpenBLAS runs its own pool beside Kokkos's and oversubscribes
# the cores (2 to 75 times slower on the modal and wall cases), so it is held to one
# thread; an OpenMP OpenBLAS shares the runtime and is capped at the Kokkos thread
# count, and at the budget where SOLPROP_OVERSUBSCRIBE took Kokkos past it. A compact
# problem's products then run on one OpenBLAS thread (``CpuTeamScope``).
_core.configure_runtime(
    {
        "threads": _REQUESTED_THREADS,
        "kernels": _pool_request("SOLPROP_NATIVE_KERNEL_THREADS"),
        "fft": _pool_request("SOLPROP_NATIVE_FFT_THREADS"),
        "raman_fft": _pool_request("SOLPROP_NATIVE_RAMAN_FFT_THREADS"),
        "raman_tail_fft": _pool_request("SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS"),
        "band_tiles": _band_request(),
        "spin_milliseconds": _WAIT.spin_milliseconds,
        "blas_threads": None
        if any(
            os.environ.get(name, "").strip()
            for name in ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS")
        )
        else min(_REQUESTED_THREADS, THREAD_BUDGET.threads),
        "host_blas": True,
    }
)

# The built-in table's CPU execution record for this process (solprop SEP 46 section
# 6.11): its operating system's values, divided on Apple silicon's spinning top tier,
# with the user's FFT start. A workspace copies a record when it is built.
_FFT_START = (
    _pool_request("SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES", "sample count") or None
)
BUILTIN_EXECUTION: ExecutionRecord = _builtin_execution(
    platform.system(),
    spin_milliseconds=_WAIT.spin_milliseconds,
    width=_REQUESTED_THREADS,
    fft_threshold=_FFT_START,
)

# The process settings in force, and the record a workspace takes by default: the
# profile's workspace settings where they hold under them, the user's FFT start over
# both.
PROCESS = Process(threads=_REQUESTED_THREADS, wait=WAIT_POLICY)
_PROFILED, EXECUTION_SOURCE = PROFILE.workspace(PROCESS, BUILTIN_EXECUTION)
EXECUTION: ExecutionRecord = (
    _PROFILED if _FFT_START is None else _PROFILED.with_fft_start(_FFT_START)
)
# The profile's FFT planning, or None for the plan policy (solprop SEP 46 section 6.7).
PLANNING: dict[str, object] | None = PROFILE.planning(PROCESS)


def refresh_profile() -> str:
    """Read the store again, keeping the process settings in force (solprop SEP 46).

    A first run that calibrated after this package's import applies the new profile's
    workspace settings and FFT planning that hold under the process's waiting policy
    and width; its process settings take effect from the next process. Workspaces
    built afterwards take the new record.

    Returns
    -------
    str
        Where a workspace's record now comes from (``EXECUTION_SOURCE``).
    """
    # The profile's names are this process's state, rebound here and nowhere else;
    # the backend reads them from the package at each use.
    global PROFILE, EXECUTION, EXECUTION_SOURCE, PLANNING
    found = choose(
        packages=[Path(entry) for entry in __path__],
        stem=_STEM,
        cpus=PROCESS_CPUS,
        quota=cgroup_cpus(),
    )
    profiled, source = found.workspace(PROCESS, BUILTIN_EXECUTION)
    PROFILE = found  # pyright: ignore[reportConstantRedefinition]
    EXECUTION_SOURCE = source  # pyright: ignore[reportConstantRedefinition]
    EXECUTION = (  # pyright: ignore[reportConstantRedefinition]
        profiled if _FFT_START is None else profiled.with_fft_start(_FFT_START)
    )
    PLANNING = found.planning(PROCESS)  # pyright: ignore[reportConstantRedefinition]
    return source


# The build record, which reads the process's built-in execution table above.
from ._build import (  # noqa: E402  (after BUILTIN_EXECUTION)
    BuildInfo,
    build_info,
    check_build,
)


def available_device_memory() -> int | None:
    """Return the selected accelerator's free bytes, or None for a host build.

    Returns
    -------
    int or None
        Free bytes from the selected accelerator runtime.
    """
    free = _core.available_device_memory()
    return None if free is None else int(free)


def __getattr__(name: str) -> object:
    """Load the solprop-dependent backend class only when it is requested."""
    if name == "NativeBackend":
        from .backend import NativeBackend

        return NativeBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily loaded class in interactive discovery."""
    return sorted({*globals(), "NativeBackend"})
