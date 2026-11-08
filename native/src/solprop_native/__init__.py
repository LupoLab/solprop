# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Kokkos execution-space backend for :mod:`solprop`.

Singleton and fixed-batch workspaces support multimode scalar and two-component
propagation, Kerr, plasma, time-domain Raman, weighted control and member-owned
statistics. Native states remain resident in their execution space between calls;
observations are owning host ``complex128`` arrays.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import logging
import os
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
_PROFILE_THREADS = PROFILE.threads()
_REQUESTED_THREADS = (
    min(_PROFILE_THREADS, THREAD_BUDGET.threads)
    if _PROFILE_THREADS is not None
    and thread_count("SOLPROP_THREADS") is None
    and thread_count("SOLPROP_NATIVE_THREADS") is None
    else _BUDGET_THREADS
)

# How the OpenMP runtime's idle threads wait: a short spin within native work, unless
# the user or a calibration chose otherwise (ADR-071). The runtime reads this when the
# core loads it, just below.
_WAIT = wait_policy(
    threads=_REQUESTED_THREADS,
    top_tier=top_tier_cores(),
    preferred=requested_wait() or PROFILE.wait(),
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
    from .batch import NativeBatchBackend

__all__ = [
    "ADAMS_MARCH_SURFACE",
    "COUPLED_FLOW_FIELDS",
    "DISPERSIVE_KERR_FIELDS",
    "DRIVER_SURFACE",
    "LINEAR_OPERATOR_FIELDS",
    "MARCH_POLICY_FIELDS",
    "MODAL_TABLES_FIELDS",
    "REGIONAL_RESPONSE_FIELDS",
    "RESPONSE_FIELDS",
    "SEPARABLE_OPERATOR_FIELDS",
    "STAGE_SCALARS_FIELDS",
    "STAGE_TERMS_FIELDS",
    "TRANSFORM_FIELDS",
    "WATCH_POLICY_FIELDS",
    "WATCH_STATE_FIELDS",
    "BuildInfo",
    "NativeBackend",
    "NativeBatchBackend",
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


_core.initialize_runtime(
    _REQUESTED_THREADS,
    _pool_request("SOLPROP_NATIVE_KERNEL_THREADS"),
    _pool_request("SOLPROP_NATIVE_FFT_THREADS"),
    _pool_request("SOLPROP_NATIVE_RAMAN_FFT_THREADS"),
    _pool_request("SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS"),
    _band_request(),
)
_core.configure_native_spin(_WAIT.spin_milliseconds)

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

# Fit a host OpenBLAS to the Kokkos pool unless its thread count is set explicitly. A
# pthreads OpenBLAS runs its own pool beside Kokkos's and oversubscribes the cores (2 to
# 75 times slower on the modal and wall cases), so it is held to one thread; an OpenMP
# OpenBLAS shares the runtime and is capped at the Kokkos thread count, and at the
# budget where SOLPROP_OVERSUBSCRIBE took Kokkos past it.
if not any(
    os.environ.get(name, "").strip()
    for name in ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS")
):
    _core.fit_blas_threads(min(_REQUESTED_THREADS, THREAD_BUDGET.threads))
# A compact problem's products then run on one OpenBLAS thread (``CpuTeamScope``).
_core.register_host_blas()

DRIVER_SURFACE: tuple[str, ...] = (
    "_account_step",
    "_adams_adopt",
    "_adams_block",
    "_adams_engine",
    "_adams_lab_source",
    "_adams_propagator",
    "_adams_restore",
    "_adams_source",
    "_adams_state",
    "_apply_mask",
    "_axis_row_populations",
    "_axis_statistics",
    "_begin_plane",
    "_bind_monitor",
    "_boundary_energies",
    "_collar_changed",
    "_controller_policy",
    "_delta_phi",
    "_dense_record",
    "_dense_stages",
    "_edge_reading",
    "edge_memory",
    "_finish",
    "_hoist",
    "_low_storage_engine",
    "_march_to",
    "_natural_stop",
    "_natural_stop_due",
    "_observe_inside",
    "_observed_field",
    "_on_accepted_step",
    "_on_axis_synthesis",
    "_poll_stop",
    "_record_attempt",
    "_replay_to",
    "_require_grid",
    "_require_state",
    "_stage_bundles",
    "_stage_scalars",
    "_start_up",
    "_statistics_step",
    "_take_step_stats",
    "_trial_step",
    "_unitary_phase",
    "_watched_step",
    "available_device_memory",
    "metadata",
    "set_initial",
    "stats",
)
"""Every ``StepDriver`` method this backend overrides or calls.

solprop freezes the signatures in its ``tests/backends/test_driver.py``, and a change to
one rewrites that table and adds a release note in the same commit. :func:`check_build`
looks each name up, so a solprop that renamed one is refused by name rather than failing
inside a step. What the run layer calls on a backend and this one inherits unchanged
(``propagate_to``, ``observe``, ``set_stop``, ...) is solprop's to keep consistent with
itself, and is not listed.
"""

_DRIVER_CLASS_MEMBERS: tuple[str, ...] = (
    "_collar_quiet",
    "_monitor",
    "_propagator_mode",
    "far_field_statistics",
    "helpers",
    "name",
    "reads_transverse_edge",
)
"""The ``StepDriver`` attributes and properties of the class this backend sets or reads,
which :func:`check_build` looks up as it does :data:`DRIVER_SURFACE`; the attributes set
at construction are frozen beside it in solprop's test."""

MARCH_POLICY_FIELDS: tuple[str, ...] = (
    "order",
    "variable_order",
    "safety",
    "max_growth",
    "min_factor",
    "rejection_growth",
    "hysteresis",
    "restart_after",
    "collar_stability",
    "real_intervals",
    "collar_rate",
    "carrier",
    "carrier_cap",
    "max_calls",
    "equal_approach",
    "interval_cap",
    "dz_min",
    "dz_max",
    "stretch",
    "z_end",
    "save_planes",
    "hard_kinks",
    "breakpoints",
    "landing_ulps",
    "interpolate",
    "quiet_interval",
)
"""``solprop.stepper.multistep.MarchPolicy``'s fields, exactly, since schema 39.

The native Adams march reads every number of its algorithm from this record (solprop
SEP 39 section 4.2), so a field solprop adds is one the native march would not honour:
:func:`check_build` refuses a solprop whose record differs in either direction."""

WATCH_POLICY_FIELDS: tuple[str, ...] = (
    "split",
    "crossing",
    "floor_regions",
    "launch_floor",
    "activity",
    "warn",
    "stop",
    "quiet_share",
    "quiet_checks",
    "quiet_collar",
    "cadence",
    "interval",
    "share",
    "bucket_limits",
)
"""``BoundaryMonitor.watch_policy``'s fields, exactly (solprop SEP 45 section 4.8).

The native march predicts the monitor's checks from these numbers alone, so a field
solprop adds is one it would not honour: the adapter refuses a policy that differs in
either direction."""

WATCH_STATE_FIELDS: tuple[str, ...] = (
    "has_last",
    "last_z",
    "since",
    "busy",
    "quiet_run",
    "quiet",
    "warned",
    "cap",
)
"""``BoundaryMonitor.watch_state``'s fields, exactly: what the predictions read and
change, compared with solprop's monitor after every drain (SEP 45 section 4.8)."""

ADAMS_MARCH_SURFACE: tuple[str, ...] = (
    "_budget_error",
    "_observe",
    "_raise_floor",
    "begin_run",
    "load",
    "march_to",
    "snapshot",
)
"""The ``solprop.backends.multistep.AdamsMarch`` methods the native march's adapter
overrides or calls (schema 39); its state attributes are the snapshot's."""

LINEAR_OPERATOR_FIELDS: tuple[str, ...] = (
    "power_weight_representation",
    "power_weight",
    "power_weight_expansion",
    "power_weight_constant",
    "coupled_flow",
    "modal_metric",
    "phi_representation",
    "polynomial",
    "rate",
)
"""``LinearOperator`` rows this backend relies on.

The four modal power-weight rows the statistics read (schema 11) and the two optional
general-modal leaves added at schema 12: ``modal_metric`` (identity, or the dense
metric of a declared unitary basis) and ``coupled_flow`` (the constant-Schur and
frozen-residual sections; CF4 is refused by name), both executed by this backend.
The general z table (solprop SEP 29 G1): ``phi_representation``, and the table
``polynomial`` and its derivative ``rate`` (cached properties), whose coefficients
this backend holds resident and evaluates natively.
Naming them here makes a solprop that renamed one fail at :func:`check_build` with a
message naming the member, rather than as an ``AttributeError`` raised from inside the
adapter on the first problem it is handed."""

TRANSFORM_FIELDS: tuple[str, ...] = ("modal_tables",)
"""``Transform`` rows this backend relies on.

The general-modal reference of schema 12, whose synthesis and projection maps this
backend executes; the same frozen-name guard as :data:`LINEAR_OPERATOR_FIELDS`, for the
same reason."""

RESPONSE_FIELDS: tuple[str, ...] = (
    "regional_responses",
    "dispersive_kerr",
    "plasma_integral",
    "prefactor_representation",
    "prefactor",
)
"""``CompiledResponses`` rows this backend relies on.

The regional response tables of schema 12, executed by this backend (the instantaneous
cubic in both forms and the silica tensor Raman model), and schema 24's
``dispersive_kerr``, solprop SEP 6's frequency-dependent electronic response, executed
on every route; the prefactor's representation and its z table (solprop SEP 29 G1);
the same frozen-name guard as :data:`LINEAR_OPERATOR_FIELDS`, for the same reason."""

MODAL_TABLES_FIELDS: tuple[str, ...] = (
    "electric",
    "omega_fine",
    "projection",
    "region_index",
    "source_frequency_factor",
    "synthesis",
)
"""``ModalTransformTables`` members the adapter reads itself (``modal_tables``).

The leaves' tuples above guard the leaf names; these guard the members inside them that
this package reads directly when it packs the general modal route. Members read only
inside solprop's own derivations, which the adapter calls (``node_kerr_tensor``,
``node_delayed_maps``, ``metric_at``, ...), are solprop's to keep consistent and are
not listed. For the same reason ``ModalMetricTables`` has no tuple:
the adapter hands ``linear.modal_metric`` to solprop's ``metric_at`` and reads none of
its members. Add a name here whenever the adapter starts reading another member."""

DISPERSIVE_KERR_FIELDS: tuple[str, ...] = (
    "carrier",
    "coefficients",
    "evaluator",
    "filters",
    "n_channels",
    "outputs",
    "positive",
    "products",
    "regions",
    "sector",
)
"""``DispersiveKerrTerms`` members the adapter reads itself.

``responses.dispersive_kerr``, read when the response is laid out for the native
engine (``regions`` on the general modal route); the leaf's grouping (``groups``) is
solprop's own derivation, which the adapter calls. The same guard as
:data:`MODAL_TABLES_FIELDS`; ``n_channels`` is a property, looked up on the class."""

REGIONAL_RESPONSE_FIELDS: tuple[str, ...] = ("n_terms",)
"""``RegionalResponseTables`` members the adapter reads itself.

``responses.regional_responses``; the same guard as :data:`MODAL_TABLES_FIELDS`."""

COUPLED_FLOW_FIELDS: tuple[str, ...] = (
    "connection",
    "generator",
    "inverse_condition_bound",
    "n_deg",
    "n_modes",
    "section_methods",
)
"""``CoupledFlowTables`` members the adapter reads itself (``linear.coupled_flow``).

Read when the host-side Schur factors and residual terms are formed; the same guard as
:data:`MODAL_TABLES_FIELDS`. ``n_deg`` and ``n_modes`` are properties, looked up on the
class."""

SEPARABLE_OPERATOR_FIELDS: tuple[str, ...] = (
    "k_squared",
    "kperp_sq",
    "k_squared_low",
    "kperp_sq_low",
    "frame",
    "spectral_rate",
    "prefactor_numerator",
    "prefactor_clamped",
    "active",
    "evanescent_rate",
    "n_pol",
    "z_span",
    "kperp_rate",
    "user_rate",
)
"""Required shared spatial coefficients, frozen since schema 6.

Schema 31 (solprop SEP 22 phase 2) appends ``kperp_rate``, a transverse collar's power
rate per column."""

STAGE_TERMS_FIELDS: tuple[str, ...] = ("phi", "prefactor", "separable")
"""The ``StageTerms`` fields the native problem is packed from."""

STAGE_SCALARS_FIELDS: tuple[str, ...] = (
    "z0",
    "z1",
    "phi_delta",
    "phi_at",
    "prefactor",
    "density",
    "kerr_density",
    "plasma_densities",
    "plasma_preionisation",
    "raman_densities",
    "dispersive_kerr_densities",
    "taper_scale",
    "raman_rates",
    "flow_section",
    "flow_weights",
    "transform_weights",
)
"""The ``StageScalars`` fields a trial's bundles are packed from.

``flow_section`` and ``flow_weights`` are schema 13's rows for a coupled flow: the
step's frozen section and each stage's weights of the flow's residual terms, packed
when a problem carries ``linear.coupled_flow``. ``transform_weights`` is schema 14's
row: each stage's weights of a moving guide's synthesis and projection maps, which this
backend combines per stage. ``dispersive_kerr_densities`` is schema 24's row: each
dispersive channel's number density at the stage (solprop SEP 6)."""


@dataclass(frozen=True)
class BuildInfo:
    """Immutable build and execution provenance for this extension.

    Attributes
    ----------
    package_version
        Installed ``solprop-native`` distribution version.
    git_commit
        Backend repository commit embedded when CMake configured the extension.
    source_digest
        SHA-256 of what the extension was compiled from: ``CMakeLists.txt`` and the C++
        sources and headers under ``src`` and ``include``
        (``cmake/SourceDigest.cmake``). :func:`check_build` compares it with an editable
        checkout's sources.
    solprop_version_built_against
        The solprop version beside which CMake configured the extension, or
        ``unknown``. Informational: the contract is the seam members
        :func:`check_build` looks for, not a version string.
    kokkos_version, kokkos_fft_version, kokkos_kernels_version, nanobind_version
        Native dependency versions.
    build_profile
        ``"local"`` (the default: vendor BLAS when one is found, the build machine's
        architecture), ``"distribution"`` (``SOLPROP_NATIVE_PROFILE=DISTRIBUTION``: a
        CPU baseline and the platform's BLAS, for wheels that run on other machines)
        or ``"reference"`` (``SOLPROP_NATIVE_PROFILE=REFERENCE``: no vendor BLAS and no
        machine-specific code, for the gate's tidy and sanitizer builds).
    cpu_baseline
        The oldest processor the build runs on (``-march`` on x86-64, ``-mcpu`` on
        Arm), for example ``"x86-64-v3"``; ``"none"`` for the compiler's default or a
        native build.
    vector_math
        The host kernels' transcendental functions: ``"SLEEF 3.9.0 u10, AVX2"`` (its
        1.0 ulp vector functions, in the variant the build selected: ``AVX2``, ``SSE2``,
        ``AVX-512F`` on request or ``AdvSIMD``), the default on x86-64, or
        ``"scalar libm"``, the system's, on Arm and in device builds
        (``SOLPROP_NATIVE_VECTOR_MATH``).
    gemm_provider, tpl_blas_enabled, blas_backend, blas_thread_policy
        Dense-transform implementation and external-library threading policy.
    blas_selection
        How configuration chose the BLAS: ``automatic: ...``, ``explicit: ...`` or
        ``distribution: ...`` or ``disabled: ...``, including the native-order GEMM
        fallback when none was found.
    blas_threads, blas_threading
        Actual OpenBLAS or MKL BLAS thread limit and threading implementation.
        The limit is ``None`` when the provider has no supported runtime query.
    execution_space, host_execution_space, device_backend, device_arch
        Configured execution-space identity.
    device_count
        Visible accelerator count; zero for a CPU build.
    raman_fft_option
        Configured Raman provider; each state reports its actual provider and tasks.
    fft_backend, fft_planner, fft_normalization
        Transform implementation and planning policy.
    fft_library, raman_fft_library, blas_library, openmp_library
        Loaded provider paths resolved through this extension; empty if absent.
    compiler, compiler_family, compiler_flags, cxx_standard, build_type
        Project compilation record.
    coverage_instrumented
        Whether project sources carry Clang source-based coverage with atomic
        counters. These builds are diagnostic and must not supply timing results.
    coverage_source_hash
        SHA-256 of the ordered project C++ source/header hashes in a coverage
        build; empty for ordinary builds. It binds reports to the compiled source.
    openmp_enabled
        Whether the host build enables OpenMP.
    requested_threads, runtime_threads
        Requested process cap and the resolved host-runtime concurrency limit.
    thread_count
        Resolved kernel concurrency within the process cap.
    fftw_threads
        Resolved modal FFT concurrency; zero for device FFTs.
    raman_fft_threads, raman_tail_fft_threads
        Resolved complete-chunk and partial-tail Raman FFT concurrency, zero on
        devices. Each state also reports its actual provider and task limits.
    threaded_fft_min_samples
        Real-side samples per execution from which a modal or spatial CPU FFT plan
        is threaded; shorter plans run on one FFTW thread.

        This field and the work sizes and medium width below are the built-in
        table's for this process (``BUILTIN_EXECUTION``, solprop SEP 46), not
        facts of the build: a workspace may apply a machine calibration's instead,
        and a run's metadata records the values it applied (``calibration``).
    cpu_size_policy_enabled
        Whether this build supports operation-specific CPU widths (OpenMP CPU only).
    explicit_thread_pools
        Explicit pool requests as bits: kernels, FFT, Raman, Raman tail (1, 2, 4, 8).
    stage_thread_min_elements, response_thread_min_elements, dense_thread_min_elements
        Minimum work sizes for automatic stage, response and native dense threading.
    modal_thread_min_elements
        Work enabling full-width general-modal synthesis, projection and map updates.
    flow_thread_min_elements
        Matrix-function work enabling full-width coupled-flow construction.
    copy_thread_min_elements
        Element count enabling full-width automatic packing and temporal copies.
    raman_thread_min_elements
        Minimum real-side sample count for automatic Raman threading.
    stage_full_thread_elements, response_full_thread_elements
        Work sizes enabling full-width stage and response execution.
    response_full_thread_samples
        Fine samples per response column enabling full-width execution.
    response_full_thread_columns
        Active response columns enabling full-width execution after the work threshold.
    dense_full_thread_elements
        Work size enabling full-width native dense execution.
    fft_full_thread_elements, raman_full_thread_elements
        Real-side sample counts enabling full-width automatic FFT plans.
    adams_thread_min_elements, adams_full_thread_elements
        Elements times history rows from which the exponential Adams trial's passes
        are threaded, and from which they take the full pool.
    cpu_medium_threads
        Maximum width between an operation's serial and full-width thresholds.
    band_column, band_tile_threads
        The field plasma column's band form, ``"tiled"`` (a team per tile of each
        row, a device's default) or ``"team"`` (a team per node, the host's
        default), and a tiled column's host team width (zero on a device or for the
        team column); set at start by ``SOLPROP_NATIVE_BAND_TILES``.
    native_arch_enabled, aggressive_vectorization_enabled, ipo_enabled,
    fast_math_enabled, optimization_policy
        Optimization switches and the human-readable compiled policy.
    precision, layout
        Numerical scalar and field layout.
    python_version, numpy_version, platform_machine
        Host Python environment.
    """

    package_version: str
    git_commit: str
    source_digest: str
    solprop_version_built_against: str
    kokkos_version: str
    kokkos_fft_version: str
    kokkos_kernels_version: str
    nanobind_version: str
    build_profile: str
    cpu_baseline: str
    vector_math: str
    gemm_provider: str
    tpl_blas_enabled: bool
    blas_backend: str
    blas_thread_policy: str
    blas_selection: str
    blas_threads: int | None
    blas_threading: str
    execution_space: str
    host_execution_space: str
    device_backend: str
    device_arch: str
    device_count: int
    raman_fft_library: str
    raman_fft_option: str
    fft_backend: str
    fft_planner: str
    fft_normalization: str
    fft_library: str
    blas_library: str
    openmp_library: str
    compiler: str
    compiler_family: str
    compiler_flags: str
    cxx_standard: int
    build_type: str
    coverage_instrumented: bool
    coverage_source_hash: str
    openmp_enabled: bool
    thread_count: int
    requested_threads: int
    runtime_threads: int
    fftw_threads: int
    raman_fft_threads: int
    raman_tail_fft_threads: int
    threaded_fft_min_samples: int
    cpu_size_policy_enabled: bool
    explicit_thread_pools: int
    stage_thread_min_elements: int
    copy_thread_min_elements: int
    flow_thread_min_elements: int
    modal_thread_min_elements: int
    response_thread_min_elements: int
    dense_thread_min_elements: int
    raman_thread_min_elements: int
    stage_full_thread_elements: int
    response_full_thread_elements: int
    response_full_thread_samples: int
    response_full_thread_columns: int
    dense_full_thread_elements: int
    fft_full_thread_elements: int
    raman_full_thread_elements: int
    adams_thread_min_elements: int
    adams_full_thread_elements: int
    cpu_medium_threads: int
    band_column: str
    band_tile_threads: int
    native_arch_enabled: bool
    aggressive_vectorization_enabled: bool
    ipo_enabled: bool
    fast_math_enabled: bool
    optimization_policy: str
    precision: str
    layout: str
    python_version: str
    numpy_version: str
    platform_machine: str


def available_device_memory() -> int | None:
    """Return the selected accelerator's free bytes, or None for a host build.

    A scan batch sizes its groups against them.

    Returns
    -------
    int or None
        Free bytes from the selected accelerator runtime.
    """
    free = _core.available_device_memory()
    return None if free is None else int(free)


def build_info() -> BuildInfo:
    """Return the immutable compiled-build record.

    Returns
    -------
    BuildInfo
        Compiler, dependencies, execution spaces, concurrency, layout, precision,
        optimization policy, and source-version provenance.
    """
    native_record = _core.build_record()
    blas_threads = native_record["blas_threads"]
    record = {key: value for key, value in native_record.items() if value is not None}
    return BuildInfo(
        package_version=__version__,
        git_commit=str(record["backend_commit"]),
        source_digest=str(record["source_digest"]),
        solprop_version_built_against=str(record["solprop_built_against"]),
        kokkos_version=str(record["kokkos_version"]),
        kokkos_fft_version=str(record["kokkos_fft_version"]),
        kokkos_kernels_version=str(record["kokkos_kernels_version"]),
        nanobind_version=str(record["nanobind_version"]),
        build_profile=str(record["build_profile"]),
        cpu_baseline=str(record["cpu_baseline"]),
        vector_math=str(record["vector_math"]),
        gemm_provider=str(record["gemm_provider"]),
        tpl_blas_enabled=bool(record["tpl_blas_enabled"]),
        blas_backend=str(record["blas_backend"]),
        blas_thread_policy=str(record["blas_thread_policy"]),
        blas_selection=str(record["blas_selection"]),
        blas_threads=None if blas_threads is None else int(blas_threads),
        blas_threading=str(record["blas_threading"]),
        execution_space=str(record["execution_space"]),
        host_execution_space=str(record["host_execution_space"]),
        device_backend=str(record["device_backend"]),
        device_arch=str(record["device_arch"]),
        device_count=int(record["device_count"]),
        raman_fft_library=str(record["raman_fft_library"]),
        raman_fft_option=str(record["raman_fft_option"]),
        fft_backend=str(record["fft_backend"]),
        fft_planner=str(record["fft_planner"]),
        fft_normalization=str(record["fft_normalization"]),
        fft_library=str(record["fft_library"]),
        blas_library=str(record["blas_library"]),
        openmp_library=str(record["openmp_library"]),
        compiler=str(record["compiler"]),
        compiler_family=str(record["compiler_family"]),
        compiler_flags=str(record["compiler_flags"]),
        cxx_standard=int(record["cxx_standard"]),
        build_type=str(record["build_type"]),
        coverage_instrumented=bool(record["coverage_instrumented"]),
        coverage_source_hash=str(record["coverage_source_hash"]),
        openmp_enabled=bool(record["openmp_enabled"]),
        thread_count=int(record["thread_count"]),
        requested_threads=int(record["requested_threads"]),
        runtime_threads=int(record["runtime_threads"]),
        fftw_threads=int(record["fftw_threads"]),
        raman_fft_threads=int(record["raman_fft_threads"]),
        raman_tail_fft_threads=int(record["raman_tail_fft_threads"]),
        threaded_fft_min_samples=BUILTIN_EXECUTION.fft_threshold,
        cpu_size_policy_enabled=bool(record["cpu_size_policy_enabled"]),
        explicit_thread_pools=int(record["explicit_thread_pools"]),
        stage_thread_min_elements=BUILTIN_EXECUTION.stage_threshold,
        copy_thread_min_elements=BUILTIN_EXECUTION.copy_threshold,
        flow_thread_min_elements=BUILTIN_EXECUTION.flow_threshold,
        modal_thread_min_elements=BUILTIN_EXECUTION.modal_threshold,
        response_thread_min_elements=BUILTIN_EXECUTION.response_threshold,
        dense_thread_min_elements=BUILTIN_EXECUTION.dense_threshold,
        raman_thread_min_elements=BUILTIN_EXECUTION.raman_threshold,
        stage_full_thread_elements=BUILTIN_EXECUTION.stage_full,
        response_full_thread_elements=BUILTIN_EXECUTION.response_full,
        response_full_thread_samples=BUILTIN_EXECUTION.response_full_samples,
        response_full_thread_columns=BUILTIN_EXECUTION.response_full_columns,
        dense_full_thread_elements=BUILTIN_EXECUTION.dense_full,
        fft_full_thread_elements=BUILTIN_EXECUTION.fft_full,
        raman_full_thread_elements=BUILTIN_EXECUTION.raman_full,
        adams_thread_min_elements=BUILTIN_EXECUTION.adams_threshold,
        adams_full_thread_elements=BUILTIN_EXECUTION.adams_full,
        cpu_medium_threads=BUILTIN_EXECUTION.medium_threads,
        band_column=str(record["band_column"]),
        band_tile_threads=int(record["band_tile_threads"]),
        native_arch_enabled=bool(record["native_arch_enabled"]),
        aggressive_vectorization_enabled=bool(
            record["aggressive_vectorization_enabled"]
        ),
        ipo_enabled=bool(record["ipo_enabled"]),
        fast_math_enabled=bool(record["fast_math_enabled"]),
        optimization_policy=str(record["optimization_policy"]),
        precision=str(record["precision"]),
        layout=str(record["layout"]),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        platform_machine=platform.machine(),
    )


def _source_digest(root: Path) -> str:
    """Return the digest of the sources an extension is compiled from.

    The computation of ``cmake/SourceDigest.cmake``, which embeds it in the build; the
    two must agree. The record is one ``{path}:{sha256 of its bytes}`` line for
    ``CMakeLists.txt`` and each C++ source and header under ``src`` and ``include``, the
    paths relative to `root` and sorted bytewise, and the digest is the SHA-256 of the
    record.

    Parameters
    ----------
    root
        The ``native`` directory of a checkout.

    Returns
    -------
    str
        The digest in lowercase hexadecimal.
    """
    paths = ["CMakeLists.txt"]
    for folder in ("src", "include"):
        paths.extend(
            path.relative_to(root).as_posix()
            for path in (root / folder).rglob("*")
            if path.suffix in {".cpp", ".hpp", ".h"} and path.is_file()
        )
    # Code points sort as their UTF-8 bytes do, so this is CMake's bytewise order.
    record = "".join(
        f"{path}:{hashlib.sha256((root / path).read_bytes()).hexdigest()}\n"
        for path in sorted(paths)
    )
    return hashlib.sha256(record.encode()).hexdigest()


@functools.cache
def _checkout_digest() -> str | None:
    """Return this editable checkout's source digest, computed once per process.

    Returns
    -------
    str or None
        The digest, or ``None`` for an installed wheel, which has no sources beside it
        and which solprop's metadata pins to its own version.
    """
    root = Path(__file__).parents[2]
    return _source_digest(root) if (root / "CMakeLists.txt").is_file() else None


def check_build() -> None:
    """Validate this build against this checkout's sources and the installed solprop.

    Raises
    ------
    solprop.exceptions.BackendError
        If an editable checkout's ``native`` sources are not the ones the extension
        was compiled from, if the installed solprop lacks a seam member this backend
        relies on
        (:data:`DRIVER_SURFACE`, :data:`ADAMS_MARCH_SURFACE`,
        :data:`MARCH_POLICY_FIELDS`, :data:`STAGE_TERMS_FIELDS`,
        :data:`STAGE_SCALARS_FIELDS`, :data:`SEPARABLE_OPERATOR_FIELDS`,
        :data:`LINEAR_OPERATOR_FIELDS`, :data:`TRANSFORM_FIELDS`,
        :data:`RESPONSE_FIELDS`, or the leaves' members in :data:`MODAL_TABLES_FIELDS`,
        :data:`REGIONAL_RESPONSE_FIELDS`, :data:`COUPLED_FLOW_FIELDS` and
        :data:`DISPERSIVE_KERR_FIELDS`),
        execution-space metadata is inconsistent, a reference build carries vendor
        BLAS or machine-specific code, a distribution build carries native code or no
        BLAS, fast-math is enabled, or an accelerator build sees no device. The
        solprop *version* is recorded, never compared here: the two packages are
        released together and pinned to one version, and the members are looked up by
        name.
    """
    from solprop.exceptions import BackendError

    info = build_info()
    digest = _checkout_digest()
    if digest is not None and digest != info.source_digest:
        raise BackendError(
            "the solprop-native extension was built from other sources than this "
            f"checkout's native/ ({Path(__file__).parents[2]}): rebuild it with "
            "`uv sync --reinstall-package solprop-native`"
        )
    _check_seam(info, BackendError)
    if info.fast_math_enabled:
        raise BackendError("solprop-native was built with forbidden fast-math")
    if info.openmp_enabled and info.execution_space != "OpenMP":
        raise BackendError(
            "the build enables OpenMP but its Kokkos execution space is "
            f"{info.execution_space!r}"
        )
    # A CPU build runs OpenMP, or Kokkos Threads where it must add no OpenMP runtime.
    expected_execution = {
        "none": {"OpenMP", "Threads"},
        "cuda": {"Cuda"},
        "hip": {"HIP"},
    }
    if info.execution_space not in expected_execution.get(info.device_backend, set()):
        raise BackendError(
            f"the {info.device_backend!r} device backend is inconsistent with "
            f"execution space {info.execution_space!r}"
        )
    if info.device_backend != "none" and info.device_count < 1:
        raise BackendError(
            f"the {info.device_backend} build has no visible accelerator device"
        )
    providers = {"CUSTOM", "KOKKOSBLAS_NATIVE", "KOKKOSBLAS_TPL", "HYBRID"}
    if info.gemm_provider not in providers:
        raise BackendError(f"unknown compiled GEMM provider {info.gemm_provider!r}")
    if info.gemm_provider == "KOKKOSBLAS_TPL" and not info.tpl_blas_enabled:
        raise BackendError("the TPL GEMM provider was built without a BLAS TPL")
    if info.tpl_blas_enabled and info.blas_backend == "none":
        raise BackendError("the build enables a BLAS TPL but identifies no provider")
    if info.build_profile not in {"local", "distribution", "reference"}:
        raise BackendError(f"unknown build profile {info.build_profile!r}")
    if info.build_profile == "reference" and (
        info.tpl_blas_enabled or info.native_arch_enabled or info.cpu_baseline != "none"
    ):
        raise BackendError(
            "the reference build links a vendor BLAS or machine-specific code"
        )
    if info.build_profile == "distribution" and (
        info.native_arch_enabled or not info.tpl_blas_enabled
    ):
        raise BackendError(
            "the distribution build carries machine-native code or no BLAS"
        )
    device_blas = {"cuda": "cuBLAS", "hip": "rocBLAS"}
    if (
        info.device_backend != "none"
        and info.tpl_blas_enabled
        and info.blas_backend != device_blas.get(info.device_backend)
    ):
        raise BackendError(
            "the selected BLAS provider cannot access this accelerator's memory"
        )


def _check_seam(info: BuildInfo, error: type[Exception]) -> None:
    """Refuse a solprop lacking a driver or bundle member this backend uses."""
    import dataclasses as _dataclasses

    from solprop.backends.driver import StepDriver
    from solprop.backends.multistep import AdamsMarch
    from solprop.backends.protocol import BatchBackend
    from solprop.stepper.multistep import MarchPolicy
    from solprop.translate.coupled import CoupledFlowTables
    from solprop.translate.linop import LinearOperator
    from solprop.translate.masks import Grids
    from solprop.translate.modal import Transform
    from solprop.translate.modal_tables import ModalTransformTables
    from solprop.translate.responses import (
        CompiledResponses,
        DispersiveKerrTerms,
        RegionalResponseTables,
    )
    from solprop.translate.spatial import SeparableOperator
    from solprop.translate.stage import StageScalars, StageTerms

    missing = [
        f"StepDriver.{name}"
        for name in DRIVER_SURFACE
        if not callable(getattr(StepDriver, name, None))
    ]
    missing.extend(
        f"StepDriver.{name}"
        for name in _DRIVER_CLASS_MEMBERS
        if not hasattr(StepDriver, name)
    )
    missing.extend(
        f"AdamsMarch.{name}"
        for name in ADAMS_MARCH_SURFACE
        if not callable(getattr(AdamsMarch, name, None))
    )
    # The policy is exact: a field the native march does not read is not honoured.
    policy = tuple(field.name for field in _dataclasses.fields(MarchPolicy))
    missing.extend(
        f"MarchPolicy.{name}" for name in MARCH_POLICY_FIELDS if name not in policy
    )
    extra = [name for name in policy if name not in MARCH_POLICY_FIELDS]
    missing.extend(
        f"BatchBackend.{name}"
        for name in ("observe_members", "member_metadata", "available_device_memory")
        if not callable(getattr(BatchBackend, name, None))
    )
    for bundle, wanted in (
        (Grids, ("t_window_fine_derivative",)),
        (LinearOperator, LINEAR_OPERATOR_FIELDS),
        (Transform, TRANSFORM_FIELDS),
        (CompiledResponses, RESPONSE_FIELDS),
        (SeparableOperator, SEPARABLE_OPERATOR_FIELDS),
        (StageTerms, STAGE_TERMS_FIELDS),
        (StageScalars, STAGE_SCALARS_FIELDS),
        (ModalTransformTables, MODAL_TABLES_FIELDS),
        (RegionalResponseTables, REGIONAL_RESPONSE_FIELDS),
        (CoupledFlowTables, COUPLED_FLOW_FIELDS),
        (DispersiveKerrTerms, DISPERSIVE_KERR_FIELDS),
    ):
        # A member is a dataclass field or, for the leaves' derived counts and the
        # z table's views, a property or a cached property.
        present = {field.name for field in _dataclasses.fields(bundle)}
        present.update(
            name
            for name in wanted
            if isinstance(
                getattr(bundle, name, None), (property, functools.cached_property)
            )
        )
        missing.extend(
            f"{bundle.__name__}.{name}" for name in wanted if name not in present
        )
    if missing or extra:
        lacks = f"it has no {', '.join(missing)}" if missing else ""
        reads = (
            f"its MarchPolicy has {', '.join(extra)}, which this build does not read"
            if extra
            else ""
        )
        raise error(
            f"solprop-native {__version__} (built beside solprop "
            f"{info.solprop_version_built_against}) does not fit the installed "
            f"solprop {version('solprop')}: {'; '.join(filter(None, (lacks, reads)))};"
            " rebuild the backend against this solprop or install the one it was "
            "built beside"
        )


def __getattr__(name: str) -> object:
    """Load the solprop-dependent backend class only when it is requested."""
    if name == "NativeBackend":
        from .backend import NativeBackend

        return NativeBackend
    if name == "NativeBatchBackend":
        from .batch import NativeBatchBackend

        return NativeBatchBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily loaded class in interactive discovery."""
    return sorted({*globals(), "NativeBackend", "NativeBatchBackend"})
