# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The compiled build's record and its check against this checkout's sources.

:class:`BuildInfo` and :func:`build_info` read the extension's own record; an editable
checkout's sources are digested as ``cmake/SourceDigest.cmake`` digests them, and
:func:`check_build` refuses an extension built from others or with inconsistent
metadata. The package re-exports the public names.
"""

from __future__ import annotations

import functools
import hashlib
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import BUILTIN_EXECUTION, __version__, _core

__all__ = ["BuildInfo", "build_info", "check_build"]


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
        SHA-256 of what the extension was compiled from: ``CMakeLists.txt``, the CMake
        modules in ``cmake`` and the C++ sources and headers under ``src`` and
        ``include`` (``cmake/SourceDigest.cmake``). :func:`check_build` compares it
        with an editable checkout's sources.
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
        Dense-transform implementation (always ``"HYBRID"``: the vendor GEMM for an
        eligible transform, the custom kernels for the rest) and external-library
        threading policy.
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
    layout
        Field layout.
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
    layout: str
    python_version: str
    numpy_version: str
    platform_machine: str


def build_info() -> BuildInfo:
    """Return the immutable compiled-build record.

    Returns
    -------
    BuildInfo
        Compiler, dependencies, execution spaces, concurrency, layout, optimization
        policy, and source-version provenance.
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
        layout=str(record["layout"]),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        platform_machine=platform.machine(),
    )


def _source_digest(root: Path) -> str:
    """Return the digest of the sources an extension is compiled from.

    The computation of ``cmake/SourceDigest.cmake``, which embeds it in the build; the
    two must agree. The record is one ``{path}:{sha256 of its bytes}`` line for
    ``CMakeLists.txt``, each ``*.cmake`` module in ``cmake`` (not below it) and each C++
    source, header and included fragment (``.inc``) under ``src`` and ``include``, the
    paths relative to `root` and sorted bytewise, and the digest is the SHA-256 of the
    record.
    """
    paths = ["CMakeLists.txt"]
    for folder in ("src", "include"):
        paths.extend(
            path.relative_to(root).as_posix()
            for path in (root / folder).rglob("*")
            if path.suffix in {".cpp", ".hpp", ".h", ".inc"} and path.is_file()
        )
    paths.extend(
        path.relative_to(root).as_posix()
        for path in (root / "cmake").glob("*.cmake")
        if path.is_file()
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

    - Returns: the digest, or ``None`` for an installed wheel, which has no sources
      beside it and which solprop's metadata pins to its own version.
    """
    root = Path(__file__).parents[2]
    return _source_digest(root) if (root / "CMakeLists.txt").is_file() else None


def check_build() -> None:
    """Validate this build against this checkout's sources and its own metadata.

    Raises
    ------
    solprop.exceptions.BackendError
        If an editable checkout's ``native`` sources are not the ones the extension
        was compiled from, execution-space metadata is inconsistent, a reference build
        carries vendor BLAS or machine-specific code, a distribution build carries
        native code or no BLAS, fast-math is enabled, or an accelerator build sees no
        device. The solprop *version* is recorded, never compared here: the two
        packages are released together and pinned to one version.
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
    if info.fast_math_enabled:
        raise BackendError("solprop-native was built with forbidden fast-math")
    if info.openmp_enabled and info.execution_space != "OpenMP":
        raise BackendError(
            "the build enables OpenMP but its Kokkos execution space is "
            f"{info.execution_space!r}"
        )
    # A CPU build runs OpenMP.
    expected_execution = {
        "none": {"OpenMP"},
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
    if info.gemm_provider != "HYBRID":
        raise BackendError(f"unknown compiled GEMM provider {info.gemm_provider!r}")
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
