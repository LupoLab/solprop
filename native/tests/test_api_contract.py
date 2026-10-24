# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The public Python/nanobind contract and runtime lifetime."""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest
from bounds import assert_within, step_bound
from conftest import ComplexArray, PhaseACase
from solprop.stepper import FixedSchedule

import solprop_native as native
from solprop_native import _core  # pyright: ignore[reportPrivateUsage]

POOL_VARIABLES = (
    "SOLPROP_NATIVE_KERNEL_THREADS",
    "SOLPROP_NATIVE_FFT_THREADS",
    "SOLPROP_NATIVE_RAMAN_FFT_THREADS",
    "SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS",
)


class ThreadRecord(TypedDict):
    """A thread-count probe's record; its arrays are written beside it."""

    threads: int
    samples: int
    steps: int
    rejected: int


class PhaseBThreadRecord(ThreadRecord):
    """The multicolumn field, periodic-statistics, and provider record."""

    rows: int
    synthesis_provider: str
    projection_provider: str


def test_public_surface_is_exact() -> None:
    assert native.__all__ == [
        "COUPLED_FLOW_FIELDS",
        "DISPERSIVE_KERR_FIELDS",
        "DRIVER_SURFACE",
        "LINEAR_OPERATOR_FIELDS",
        "MODAL_TABLES_FIELDS",
        "REGIONAL_RESPONSE_FIELDS",
        "RESPONSE_FIELDS",
        "SEPARABLE_OPERATOR_FIELDS",
        "STAGE_SCALARS_FIELDS",
        "STAGE_TERMS_FIELDS",
        "TRANSFORM_FIELDS",
        "BuildInfo",
        "NativeBackend",
        "NativeBatchBackend",
        "__version__",
        "available_device_memory",
        "build_info",
        "check_build",
    ]
    # One version for solprop and solprop-native, from the one repository (SEP 29).
    assert native.__version__ == version("solprop")
    assert "_core" not in native.__all__


def test_build_info_is_frozen_and_complete() -> None:
    info = native.build_info()
    assert dataclasses.is_dataclass(info)
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.thread_count = 99  # type: ignore[misc]
    assert info.schema_version == 36
    assert info.response_full_thread_columns > 0
    assert (
        info.response_full_thread_columns
        == _core.build_record()["response_full_thread_columns"]
    )
    assert isinstance(info.coverage_instrumented, bool)
    assert info.coverage_instrumented == _core.build_record()["coverage_instrumented"]
    if info.coverage_instrumented:
        assert len(info.coverage_source_hash) == 64
        assert info.device_backend == "none"
        assert "-fprofile-update=atomic" in info.compiler_flags
        assert "-fcoverage-mapping" in info.compiler_flags
        assert not info.ipo_enabled
    assert info.solprop_version_built_against
    assert not info.solprop_version_built_against.startswith("solprop==")
    assert info.kokkos_version == "5.2.1"
    assert info.kokkos_fft_version == "2.0.0"
    assert info.kokkos_kernels_version == "5.2.0"
    assert info.gemm_provider in {
        "CUSTOM",
        "KOKKOSBLAS_NATIVE",
        "KOKKOSBLAS_TPL",
        "HYBRID",
    }
    if info.tpl_blas_enabled:
        assert info.blas_backend != "none"
    else:
        assert info.blas_backend == "none"
    assert info.build_profile in {"local", "distribution", "reference"}
    if info.build_profile == "reference":
        # The gate's tidy/sanitizer build: no vendor BLAS, no machine-specific code.
        assert not info.tpl_blas_enabled
        assert not info.native_arch_enabled
        assert info.cpu_baseline == "none"
        assert info.blas_selection == "disabled: reference build"
    elif info.build_profile == "distribution":
        # The wheel for other machines: a CPU baseline and the platform's BLAS,
        # whose library is the one the build names and whose thread pool is known.
        assert not info.native_arch_enabled
        assert info.tpl_blas_enabled
        assert info.blas_selection.startswith("distribution: ")
        assert info.blas_library
        if info.blas_backend == "OpenBLAS":
            # Linux shares the extension's OpenMP runtime; Windows keeps the pthreads
            # scipy-openblas32 (or Linux, when selected for comparison).
            expected = (
                "solprop_openblas"
                if info.blas_threading == "OpenMP"
                else "scipy_openblas"
            )
            assert expected in info.blas_library
            assert info.blas_threads is not None
    else:
        assert info.blas_selection.startswith(
            ("automatic: ", "explicit: ", "disabled: ")
        )
        # A local build without BLAS says why: none found, or switched off.
        assert info.tpl_blas_enabled != info.blas_selection.startswith(
            ("disabled: ", "automatic: none found")
        )
    if info.native_arch_enabled:
        assert info.cpu_baseline == "none"
    if info.cpu_baseline != "none":
        assert info.cpu_baseline in info.compiler_flags
        assert info.cpu_baseline in info.optimization_policy
    if info.device_backend == "none":
        if info.openmp_enabled:
            assert info.execution_space == "OpenMP"
            assert info.fft_backend in {"FFTW_OPENMP", "FFTW_PTHREADS", "MKL_FFTW"}
        else:
            # The Kokkos Threads build adds no OpenMP runtime (solprop SEP 29).
            assert info.execution_space == "Threads"
            assert info.fft_backend == "FFTW_THREADS"
        for count in (
            info.thread_count,
            info.fftw_threads,
            info.raman_fft_threads,
            info.raman_tail_fft_threads,
        ):
            assert 1 <= count <= info.runtime_threads
        assert info.threaded_fft_min_samples >= 1
        assert info.band_column in {"team", "tiled"}
        assert (info.band_tile_threads >= 1) == (info.band_column == "tiled")
        assert info.fft_library
        assert bool(info.openmp_library) == info.openmp_enabled
    elif info.device_backend == "cuda":
        assert info.execution_space == "Cuda"
        assert info.fft_backend == "cuFFT"
        assert info.device_count >= 1
        assert info.fftw_threads == 0
        assert info.band_tile_threads == 0
        if info.tpl_blas_enabled:
            assert info.blas_backend == "cuBLAS"
    elif info.device_backend == "hip":
        assert info.execution_space == "HIP"
        assert info.fft_backend == "rocFFT"
        assert info.device_count >= 1
        assert info.fftw_threads == 0
        assert info.band_tile_threads == 0
        if info.tpl_blas_enabled:
            assert info.blas_backend == "rocBLAS"
    else:
        pytest.fail(f"unknown device backend {info.device_backend!r}")
    assert info.fft_planner.startswith("Kokkos-FFT persistent Plan")
    assert info.precision == "complex128"
    assert info.layout == "LayoutRight"
    assert info.fast_math_enabled is False

    native_record = _core.build_record()
    assert native_record["device_team_size"] in {64, 128, 256, 512, 1024}
    assert native_record["device_reduction_size"] in {64, 128, 256, 512, 1024}
    compiler_flags = cast(str, native_record["compiler_flags"])
    assert "__SOLPROP_SPACE__" not in compiler_flags
    assert native_record["blas_thread_policy"] in {
        "disabled",
        "CUDA execution-space stream",
        "HIP execution-space stream",
        "system-managed",
        "OPENBLAS_NUM_THREADS at process start",
        "OMP_NUM_THREADS at process start",
        "MKL_DOMAIN_NUM_THREADS at process start",
        "provider environment at process start",
        "one thread beside the Kokkos pool",
        "the Kokkos thread count",
    }


def test_module_entry_point_prints_one_json_record() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "solprop_native"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "OMP_PROC_BIND": "false"},
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["schema_version"] == 36


@pytest.mark.parametrize("explicit", [None, "3"])
def test_a_host_openblas_is_fitted_to_the_native_pool(explicit: str | None) -> None:
    info = native.build_info()
    if info.device_backend != "none" or info.blas_backend != "OpenBLAS":
        pytest.skip("only a host OpenBLAS keeps a thread pool of its own")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS")
    }
    environment.update(OMP_PROC_BIND="false", SOLPROP_NATIVE_THREADS="2")
    if explicit is not None:
        environment["OPENBLAS_NUM_THREADS"] = explicit
    completed = subprocess.run(
        [sys.executable, "-m", "solprop_native"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    fitted = {
        "pthreads": ("one thread beside the Kokkos pool", 1),
        "OpenMP": ("the Kokkos thread count", 2),
    }
    # A pthreads library is held to one thread, an OpenMP one capped at the Kokkos
    # count; an explicit count always wins.
    if explicit is None and record["blas_threading"] in fitted:
        policy, threads = fitted[record["blas_threading"]]
        assert record["blas_thread_policy"] == policy
        assert record["blas_threads"] == threads
    else:
        assert record["blas_thread_policy"] not in {
            policy for policy, _ in fitted.values()
        }


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many"])
@pytest.mark.parametrize(
    "name",
    [
        "SOLPROP_NATIVE_THREADS",
        "SOLPROP_NATIVE_KERNEL_THREADS",
        "SOLPROP_NATIVE_FFT_THREADS",
        "SOLPROP_NATIVE_RAMAN_FFT_THREADS",
        "SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS",
        "SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES",
    ],
)
def test_invalid_thread_variable_is_rejected_in_a_fresh_process(
    value: str, name: str
) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import solprop_native"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, name: value},
    )
    assert completed.returncode != 0
    assert name in completed.stderr


@pytest.mark.parametrize("value", ["-1", "1.5", "many"])
def test_an_invalid_band_column_request_is_rejected(value: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import solprop_native"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "SOLPROP_NATIVE_BAND_TILES": value},
    )
    assert completed.returncode != 0
    assert "SOLPROP_NATIVE_BAND_TILES" in completed.stderr


@pytest.mark.parametrize(
    ("value", "column", "threads"), [("0", "team", 0), ("3", "tiled", 3)]
)
def test_the_band_column_request_is_reported(
    value: str, column: str, threads: int
) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "solprop_native"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "OMP_PROC_BIND": "false",
            "SOLPROP_NATIVE_BAND_TILES": value,
        },
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    device = record["device_backend"] != "none"
    # A device runs its own team width, and the team column only when asked for it.
    assert record["band_column"] == column
    assert record["band_tile_threads"] == (0 if device else threads)


@pytest.mark.parametrize("limit", [1, 4])
def test_independent_pools_respect_the_process_limit(limit: int) -> None:
    if native.build_info().device_backend != "none":
        pytest.skip("CPU pool policy; GPU execution keeps its device policy")
    completed = subprocess.run(
        [sys.executable, "-m", "solprop_native"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "OMP_PROC_BIND": "false",
            "SOLPROP_NATIVE_THREADS": str(limit),
            "SOLPROP_NATIVE_KERNEL_THREADS": "3",
            "SOLPROP_NATIVE_FFT_THREADS": "1",
            "SOLPROP_NATIVE_RAMAN_FFT_THREADS": "2",
            "SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS": "99",
            "SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES": "4096",
        },
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert record["explicit_thread_pools"] == 15
    assert record["threaded_fft_min_samples"] == 4096
    assert record["runtime_threads"] == limit
    assert record["thread_count"] == min(3, limit)
    assert record["fftw_threads"] == 1
    assert record["raman_fft_threads"] == min(2, limit)
    assert record["raman_tail_fft_threads"] == limit


def test_initial_array_contract(
    phase_a_case: PhaseACase, writable_initial: ComplexArray
) -> None:
    problem, grid = phase_a_case
    backend = native.NativeBackend(problem, grid=grid)
    try:
        with pytest.raises(TypeError, match="NumPy ndarray"):
            backend.set_initial(cast(ComplexArray, writable_initial.tolist()), None)
        with pytest.raises(TypeError, match="complex128"):
            backend.set_initial(
                cast(ComplexArray, writable_initial.astype(np.complex64)), None
            )
        with pytest.raises(ValueError, match="rank 4"):
            backend.set_initial(writable_initial.reshape(-1), None)
        with pytest.raises(ValueError, match="C-contiguous"):
            backend.set_initial(writable_initial[..., ::-1], None)
        read_only = writable_initial.copy()
        read_only.setflags(write=False)
        backend.set_initial(read_only, None)
    finally:
        backend.close()


def test_observation_owns_host_storage(
    phase_a_case: PhaseACase, writable_initial: ComplexArray
) -> None:
    problem, grid = phase_a_case
    backend = native.NativeBackend(problem, grid=grid)
    backend.set_initial(writable_initial, None)
    first = backend.observe().require_field()
    expected = first.copy()
    del writable_initial
    backend.close()
    gc.collect()
    # A lossless host copy: the observation's own storage outlives the backend.
    np.testing.assert_array_equal(first, expected)
    assert first.flags.c_contiguous
    assert not first.flags.writeable


def test_repeated_and_concurrent_instances_are_independent(
    phase_a_case: PhaseACase, writable_initial: ComplexArray
) -> None:
    """Instances share no state: concurrent runs agree with a run alone.

    All three replay one prescribed schedule with the explicit pair, so they perform
    the same discrete computation and differ by rounding alone (solprop SEP 25): four
    steps, each within the transforms' budget of its seven stages
    (:func:`bounds.step_bound`). State leaking between instances would move the
    field by far more.
    """
    original, grid = phase_a_case
    problem = dataclasses.replace(
        original,
        stepper=dataclasses.replace(
            original.stepper, schedule=FixedSchedule.uniform(0.0, 1e-3, 2.5e-4)
        ),
    )

    def run() -> ComplexArray:
        backend = native.NativeBackend(problem, grid=grid)
        try:
            backend.set_initial(writable_initial, None)
            backend.propagate_to(1e-3)
            return backend.observe().require_field()
        finally:
            backend.close()

    expected = run()

    def run_ignored(_: int) -> ComplexArray:
        return run()

    if native.build_info().execution_space == "Threads":
        # Kokkos Threads dispatches only from the thread that started it: another
        # Python thread gets a RuntimeError, never a crash (solprop SEP 29 phase 0).
        with (
            ThreadPoolExecutor(max_workers=1) as executor,
            pytest.raises(RuntimeError, match="master"),
        ):
            executor.submit(run).result()
        return
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_ignored, range(2)))
    bound = step_bound(grid.n_to, 4)
    for result in results:
        assert_within(result, expected, bound, what="concurrent instance")


def test_substantial_native_trial_releases_the_gil(
    phase_a_case: PhaseACase,
) -> None:
    """Another Python thread runs while a native trial does.

    The observer stamps the clock whenever it holds the GIL; a stamp strictly inside a
    trial's own clock interval exists only if the trial released the GIL. The earlier
    form (one trial, an event the observer had to set before the trial returned) lost
    the race whenever the operating system woke the observer later than a 1 024-sample
    trial takes, which one busy Linux host did in about half its runs;
    trials are now repeated, each bracketed, until a stamp lands inside one.
    """
    problem, grid = phase_a_case
    backend = native.NativeBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    active = threading.Event()
    finished = threading.Event()
    stamps: list[float] = []

    def observer() -> None:
        while not finished.is_set():
            if active.is_set():
                stamps.append(time.perf_counter())

    thread = threading.Thread(target=observer)
    thread.start()
    inside = False
    checked = 0
    active.set()
    try:
        for _ in range(200):
            start = time.perf_counter()
            # This contract test deliberately exercises the protected one-call seam.
            backend._trial_step(0.0, 1e-4)  # pyright: ignore[reportPrivateUsage]
            end = time.perf_counter()
            # The observer only appends, in clock order, so the stamps so far are read
            # once each.
            seen = len(stamps)
            inside = any(start < stamp < end for stamp in stamps[checked:seen])
            checked = seen
            if inside:
                break
    finally:
        active.clear()
        finished.set()
        thread.join()
        backend.close()
    assert inside


def test_close_is_idempotent_and_guards_use(phase_a_case: PhaseACase) -> None:
    problem, grid = phase_a_case
    backend = native.NativeBackend(problem, grid=grid)
    backend.close()
    backend.close()
    with pytest.raises(Exception, match="closed"):
        backend.set_initial(problem.initial.field0, None)


def test_import_and_shutdown_in_a_fresh_process() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import solprop_native; print('ok')"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "OMP_PROC_BIND": "false"},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_a_process_that_ran_both_backends_exits_cleanly() -> None:
    """Finalising the runtime does not destroy FFTW state that pyFFTW still uses.

    Kokkos-FFT cleans up FFTW's process-wide planner when Kokkos finalises. Finalised
    from a Python exit handler, that ran before the interpreter released the Numba
    backend's pyFFTW plans, whose destruction then crashed the process. The probe
    also holds a plan on this build's own FFTW library, so the test applies whatever
    pyFFTW was built against.
    """
    script = Path(__file__).with_name("teardown_probe.py")
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "OMP_PROC_BIND": "false"},
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1]) == ["native", "numba_cpu"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS OpenMP-runtime guard")
def test_pyfftw_first_is_refused_before_openmp_initialization() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import pyfftw; import solprop_native"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "OMP_PROC_BIND": "false"},
    )
    assert completed.returncode != 0
    assert "import solprop_native before pyfftw" in completed.stderr


def _thread_probe(
    script: str, threads: int, out: Path, extra: dict[str, str] | None = None
) -> dict[str, object]:
    """Run a probe in a fresh process at `threads`; return its record line."""
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).with_name(script)), str(out)],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "OMP_PROC_BIND": "false",
            "SOLPROP_NATIVE_THREADS": str(threads),
            **{name: str(threads) for name in POOL_VARIABLES},
            **(extra or {}),
        },
    )
    assert completed.returncode == 0, completed.stderr
    return cast("dict[str, object]", json.loads(completed.stdout.splitlines()[-1]))


def test_one_and_eight_threads_agree_to_rounding_in_fresh_processes(
    tmp_path: Path,
) -> None:
    """One and eight threads replay one schedule to within its rounding budget.

    Ten prescribed steps of the explicit pair: the same discrete computation on both
    sides, so the thread count may move the field by rounding only (solprop SEP 25),
    held to the transforms' budget of seventy stages (:func:`bounds.step_bound`). The
    schedule executes its ten steps and rejects none on either side.
    """
    if native.build_info().device_backend != "none":
        pytest.skip("host thread-count invariance is not applicable to a device build")
    records: list[ThreadRecord] = []
    fields: list[ComplexArray] = []
    for threads in (1, 8):
        out = tmp_path / f"field-{threads}.npy"
        record = _thread_probe("thread_probe.py", threads, out)
        records.append(cast(ThreadRecord, record))
        fields.append(cast(ComplexArray, np.load(out)))
    assert [record["threads"] for record in records] == [1, 8]
    assert [record["steps"] for record in records] == [10, 10]
    assert [record["rejected"] for record in records] == [0, 0]
    assert_within(
        fields[1], fields[0], step_bound(records[0]["samples"], 10), what="field"
    )


@pytest.mark.parametrize("threads", [3, 4, 8])
def test_forced_fft_widths_agree_across_threads_to_rounding(
    tmp_path: Path, threads: int
) -> None:
    """Explicit FFT pool widths bypass automatic sizing and can change factorisation.

    FFTW's estimate planner may factorise a threaded plan differently with the thread
    count (FFTW 3.3.10 does at 3 and 18 threads on an x86-64 host), which moves every
    transform by rounding only. Bound, stated
    first: each of the ten fixed steps' seven stages runs one forward and one inverse
    transform, each within Higham's ``7 u log2 n`` of the exact transform, so two plans
    differ by twice that; nothing else in the engine depends on the thread count
    (fixed-order reductions), the linear step is unitary and this soliton's nonlinear
    phase over a millimetre is 0.01 rad, so nothing amplifies it. The bound is
    ``10 * 7 * 2 * 2 * 7 u log2 n``, about 3e-12 relative in the L2 norm at 16 384
    samples.
    """
    if native.build_info().device_backend != "none":
        pytest.skip("host thread-count invariance is not applicable to a device build")
    script = Path(__file__).with_name("fft_threshold_probe.py")
    fields: list[ComplexArray] = []
    samples = 0
    for count in (1, threads):
        out = tmp_path / f"field-{count}.npy"
        completed = subprocess.run(
            [sys.executable, str(script), str(out)],
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "OMP_PROC_BIND": "false",
                "SOLPROP_NATIVE_THREADS": str(count),
                **{name: str(count) for name in POOL_VARIABLES},
            },
        )
        assert completed.returncode == 0, completed.stderr
        record = cast(dict[str, int], json.loads(completed.stdout.splitlines()[-1]))
        assert record["threads"] == count
        assert record["steps"] == 10
        samples = record["samples"]
        fields.append(cast(ComplexArray, np.load(out)))
    assert samples == 16384
    unit = float(np.finfo(np.float64).eps) / 2
    bound = 10 * 7 * 2 * 2 * 7 * unit * np.log2(samples)
    departure = np.linalg.norm(fields[1] - fields[0]) / np.linalg.norm(fields[0])
    assert departure <= bound


def test_phase_b_field_and_statistics_agree_across_threads_to_rounding(
    tmp_path: Path,
) -> None:
    """A two-mode plasma capillary on one and eight threads, to the rounding budget.

    Forty prescribed steps of the explicit pair (the probe), so both sides perform the
    same discrete computation and the thread count moves results by rounding only
    (solprop SEP 25). Bounds, fixed before the comparison:

    - the field, the transforms' budget of 280 stages (:func:`bounds.step_bound`);
    - the step positions, lengths and acceptances exactly: the schedule prescribes them;
    - each step's error norm, one step's budget relative to the field over ``rtol``
      (the embedded difference is formed from the stages, whose rounding is relative
      to the field's, and scored against ``rtol |E|``), plus the field's budget;
    - every statistic, a thousand times the field's budget: a pointwise value read
      from a normwise bound (the square root of the samples, about 45), a squared
      modulus (two) and an ionisation rate's logarithmic sensitivity (at most ten);
      the quadrature error, a small difference of two projections, absolutely.
    """
    if native.build_info().device_backend != "none":
        pytest.skip("host thread-count invariance is not applicable to a device build")
    records: list[PhaseBThreadRecord] = []
    arrays: list[dict[str, np.ndarray]] = []
    for threads in (1, 8):
        out = tmp_path / f"run-{threads}.npz"
        record = _thread_probe(
            "thread_invariance_probe.py",
            threads,
            out,
            {"KMP_BLOCKTIME": "50", "OMP_DYNAMIC": "FALSE"},
        )
        records.append(cast(PhaseBThreadRecord, record))
        with np.load(out) as stored:
            arrays.append({name: stored[name] for name in stored.files})
    assert [record["threads"] for record in records] == [1, 8]
    assert [record["steps"] for record in records] == [40, 40]
    assert [record["rejected"] for record in records] == [0, 0]
    assert [record["rows"] for record in records] == [40, 40]
    assert records[0]["synthesis_provider"] == records[1]["synthesis_provider"]
    assert records[0]["projection_provider"] == records[1]["projection_provider"]
    one, many = arrays
    assert one.keys() == many.keys()
    samples = records[0]["samples"]
    field_bound = step_bound(samples, 40)
    assert_within(many["field"], one["field"], field_bound, what="field")
    for name in ("z", "dz", "accepted"):
        np.testing.assert_array_equal(many[name], one[name], err_msg=name)
    rtol = 1e-6  # the probe's controller tolerance
    np.testing.assert_allclose(
        many["error_norm"],
        one["error_norm"],
        rtol=field_bound,
        atol=step_bound(samples, 1) / rtol,
        err_msg="error_norm",
    )
    statistic_bound = 1e3 * field_bound
    for name in one:
        if not name.startswith(("rows_", "plane_")):
            continue
        absolute = name.endswith("quadrature_error")
        assert_within(
            many[name],
            one[name],
            statistic_bound,
            scale=1.0 if absolute else None,
            what=name,
        )


def test_the_reference_parts_time_what_they_are_asked_to() -> None:
    """solprop SEP 28's reference parts: one time per timed pair or copy, all positive.

    The parts keep their arrays inside the native layer, so the count and the sign of
    the times are what is visible here; a volume that is not rank 3 is refused.
    """
    rng = np.random.default_rng(28)
    volume = rng.standard_normal((3, 4, 5)) + 1j * rng.standard_normal((3, 4, 5))
    pairs = _core.reference_fft3_pairs(np.ascontiguousarray(volume), 1, 3)
    copies = _core.reference_stream_copies(1000, 1, 4)
    assert len(pairs) == 3
    assert all(seconds > 0.0 for seconds in pairs)
    assert len(copies) == 4
    assert all(seconds > 0.0 for seconds in copies)
    with pytest.raises(ValueError, match="three-dimensional"):
        _core.reference_fft3_pairs(np.ascontiguousarray(volume[0]), 0, 1)


def test_operation_sizing_metadata_preserves_process_cap(
    phase_a_case: PhaseACase,
) -> None:
    problem, grid = phase_a_case
    info = native.build_info()
    backend = native.NativeBackend(problem, grid=grid)
    try:
        sizing = cast(dict[str, dict[str, int]], backend.metadata()["execution_sizing"])
        assert sizing
        for choice in sizing.values():
            assert choice["elements"] > 0
            if "fine_samples" in choice:
                assert choice["elements"] == choice["columns"] * choice["fine_samples"]
            assert choice["threads"] >= 1
            if info.cpu_size_policy_enabled:
                assert choice["threads"] <= info.runtime_threads
        assert backend.metadata()["execution_sizing"] == sizing
        assert native.build_info().thread_count == info.thread_count
    finally:
        backend.close()


def test_automatic_and_single_thread_windows_match_numpy(tmp_path: Path) -> None:
    """Automatic and capped serial runs meet the fixed-window rounding budget."""
    if native.build_info().device_backend != "none":
        pytest.skip("CPU operation sizing is not applicable to a device build")
    fields: list[ComplexArray] = []
    samples = 0
    for name, backend, cap in (
        ("reference", "numpy_ref", 1),
        ("single", "native", 1),
        ("automatic", "native", 8),
    ):
        out = tmp_path / f"{name}.npy"
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in (*POOL_VARIABLES, "SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES")
        }
        environment.update(SOLPROP_NATIVE_THREADS=str(cap), OMP_PROC_BIND="false")
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("thread_probe.py")),
                str(out),
                backend,
            ],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr
        record = cast(dict[str, object], json.loads(completed.stdout.splitlines()[-1]))
        assert record["explicit_thread_pools"] == 0
        assert record["steps"] == 10
        assert record["rejected"] == 0
        samples = cast(int, record["samples"])
        fields.append(cast(ComplexArray, np.load(out)))
    # The existing soliton fixture takes 70 matched explicit stages. Its FFT and
    # propagation budget applies independently of each operation's execution width.
    for field in fields[1:]:
        assert_within(field, fields[0], step_bound(samples, 70))
