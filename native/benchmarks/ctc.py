# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure the reduced and full-shape CtC cases in isolated processes."""

# ruff: noqa: I001 -- Kokkos must load before pyFFTW on macOS.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import replace
from typing import Any, cast

from device_record import DeviceRecord, device_record


class Result(DeviceRecord):
    """One isolated CtC measurement."""

    backend: str
    device_before: DeviceRecord
    case: str
    mode: str
    seconds: float
    construction_seconds: float
    planning_seconds: float
    raman_planning_seconds: float
    trials: int
    accepted_steps: int
    rejected_steps: int
    milliseconds_per_trial: float
    milliseconds_per_accepted_step: float | None
    modes: int
    nodes: int
    node_chunk: int
    components: int
    n_omega: int
    n_to: int
    raman_terms: int
    raman_kernel_transforms: int
    raman_memory_ledger: dict[str, int]
    field_sha256: str
    memory_ledger_bytes: int
    kernel_threads: int
    native_threads: int
    fftw_threads: int
    blas_threads: int | str
    gemm_provider: str
    synthesis_provider: str
    projection_provider: str
    blas_backend: str
    propagator_mode: str
    omp_proc_bind: str
    omp_places: str
    omp_wait_policy: str
    load_start: float
    load_end: float


class Measurement(Result):
    """The median run plus spread across isolated processes."""

    minimum_seconds: float
    maximum_seconds: float
    median_absolute_deviation: float


# The two CtC shapes, defined here rather than read from a stored comparison: hydrogen
# at 40 bar in a 16 um capillary, 400 fs and 10 uJ at 515 nm, no loss, the package's
# own materials. The full shape runs 0.3 m on six modes and a 4 ps, 170 nm-4 um grid;
# the reduced one is the fundamental alone over 0.1 m on a 2 ps, 250 nm-2 um grid, five
# times the pulse, which keeps the launch clear of the temporal collar.
CASES = {
    "ctc_reduced": {
        "length": 0.1,
        "modes": 1,
        "wavelength_limits": (250e-9, 2e-6),
        "trange": 2e-12,
        "n_save": 21,
    },
    "ctc_full": {
        "length": 0.3,
        "modes": 6,
        "wavelength_limits": (170e-9, 4e-6),
        "trange": 4e-12,
        "n_save": 201,
    },
}


def build_simulation(case_name: str, backend_name: str) -> Any:
    from solprop.driver import Simulation

    case = CASES[case_name]
    return Simulation.from_capillary(
        16e-6,
        case["length"],
        "H2",
        40.0,
        lambda0=515e-9,
        wavelength_limits=case["wavelength_limits"],
        trange=case["trange"],
        fwhm=400e-15,
        energy=10e-6,
        modes=case["modes"],
        loss=False,
        raman=True,
        n_save=case["n_save"],
        rtol=1e-6,
        atol=1e-10,
        backend=backend_name,
    )


def _metadata_int(metadata: dict[str, object], name: str, default: int) -> int:
    value = metadata.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise TypeError(f"metadata {name!r} is not integer-valued: {value!r}")
    return int(value)


def _metadata_float(metadata: dict[str, object], name: str) -> float:
    value = metadata.get(name, 0.0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"metadata {name!r} is not numeric: {value!r}")
    return float(value)


def _worker(
    backend_name: str,
    case_name: str,
    mode: str,
    trial_count: int,
    node_chunk: int | None,
    progress_seconds: float,
) -> None:
    if backend_name == "native":
        __import__("solprop_native")

    import numpy as np

    from solprop.backends.registry import select
    from solprop.exceptions import MaterialRangeWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = build_simulation(case_name, backend_name)
        problem = simulation.translate()
    if node_chunk is not None:
        problem = replace(
            problem, transform=replace(problem.transform, node_chunk=node_chunk)
        )

    constructed = time.perf_counter()
    backend: Any = select(
        backend_name,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    construction_seconds = time.perf_counter() - constructed
    try:
        backend.set_initial(problem.initial.field0, simulation.noise)
        device_before = device_record(backend.metadata())
        if mode == "trial":
            backend._trial_step(0.0, problem.stepper.dz_init)
            load_start = os.getloadavg()[0]
            started = time.perf_counter()
            for _ in range(trial_count):
                backend._trial_step(0.0, problem.stepper.dz_init)
            seconds = time.perf_counter() - started
            load_end = os.getloadavg()[0]
            _, candidate = backend._trial_step(0.0, problem.stepper.dz_init)
            backend._finish(candidate, problem.stepper.dz_init)
            accepted = 0
            rejected = 0
            trials = trial_count
        else:
            backend.propagate_to(min(float(problem.stepper.save_planes[0]), 1e-8))
            backend.set_initial(problem.initial.field0, simulation.noise)
            load_start = os.getloadavg()[0]
            started = time.perf_counter()
            process_started = time.process_time()
            attempts = 0
            rejected = 0
            planes = len(problem.stepper.save_planes)
            progress_state: dict[str, int | float] = {
                "completed_planes": 0,
                "current_plane": 1,
                "target": float(problem.stepper.save_planes[0]),
                "accepted_steps": 0,
                "rejected_steps": 0,
            }
            progress_stop = threading.Event()

            def emit_progress(kind: str) -> None:
                record: dict[str, bool | float | int | str] = {
                    "progress": True,
                    "kind": kind,
                    "planes": planes,
                    "elapsed_seconds": time.perf_counter() - started,
                    "process_cpu_seconds": time.process_time() - process_started,
                    **progress_state,
                }
                print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)

            def progress_heartbeat() -> None:
                while not progress_stop.wait(progress_seconds):
                    emit_progress("heartbeat")

            progress_thread: threading.Thread | None = None
            if progress_seconds > 0.0:
                progress_thread = threading.Thread(
                    target=progress_heartbeat,
                    name="ctc-progress",
                    daemon=True,
                )
                progress_thread.start()
            try:
                for plane, target in enumerate(problem.stepper.save_planes, start=1):
                    progress_state["current_plane"] = plane
                    progress_state["target"] = float(target)
                    summary = backend.propagate_to(float(target))
                    attempts += summary.n_steps
                    rejected += summary.n_rejected
                    backend.stats(problem.stats_spec)
                    progress_state["completed_planes"] = plane
                    progress_state["accepted_steps"] = attempts - rejected
                    progress_state["rejected_steps"] = rejected
                if progress_seconds > 0.0:
                    emit_progress("complete")
            finally:
                progress_stop.set()
                if progress_thread is not None:
                    progress_thread.join()
            seconds = time.perf_counter() - started
            load_end = os.getloadavg()[0]
            accepted = attempts - rejected
            trials = attempts
        field = backend.observe().field
        if not np.all(np.isfinite(field)):
            raise RuntimeError("CtC benchmark produced a non-finite field")
        metadata: dict[str, object] = backend.metadata()
        device_after = device_record(metadata)
        ledger = backend.memory_ledger()
        raman_transforms = int(getattr(backend, "_raman_transforms", 0))
    finally:
        backend.close()

    threads = _metadata_int(
        metadata, "thread_count", _metadata_int(metadata, "numba_threads", 0)
    )
    fftw_threads = _metadata_int(metadata, "fftw_threads", threads)
    blas_backend = str(metadata.get("blas_backend", "not applicable"))
    if blas_backend == "Accelerate":
        blas_threads: int | str = "system-managed"
    elif blas_backend == "OpenBLAS":
        blas_threads = int(os.environ.get("OPENBLAS_NUM_THREADS", "1"))
    elif blas_backend in {"cuBLAS", "rocBLAS"}:
        blas_threads = "device-stream"
    elif backend_name == "native":
        blas_threads = "disabled"
    else:
        blas_threads = int(os.environ.get("SOLPROP_BLAS_THREADS", "1"))
    result: Result = {
        **device_after,
        "backend": backend_name,
        "device_before": device_before,
        "case": case_name,
        "mode": mode,
        "seconds": seconds,
        "construction_seconds": construction_seconds,
        "planning_seconds": _metadata_float(metadata, "planning_seconds"),
        "raman_planning_seconds": _metadata_float(metadata, "raman_planning_seconds"),
        "trials": trials,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "milliseconds_per_trial": 1000.0 * seconds / trials,
        "milliseconds_per_accepted_step": (
            1000.0 * seconds / accepted if accepted else None
        ),
        "modes": problem.shape[1],
        "nodes": problem.transform.n_nodes,
        "node_chunk": problem.transform.node_chunk,
        "components": problem.transform.n_pol * problem.shape[2],
        "n_omega": problem.shape[-1],
        "n_to": problem.grids.n_to,
        "raman_terms": (
            0
            if problem.responses.raman_coefficients is None
            else problem.responses.raman_coefficients.n_terms
        ),
        "raman_kernel_transforms": raman_transforms,
        "raman_memory_ledger": {
            name: ledger[name]
            for name in (
                "node_raman_drive",
                "node_raman_spectrum",
                "node_raman_convolved",
                "raman_undamped",
                "raman_kernel_time",
                "raman_kernel_spectrum",
                "stage_raman_rates",
            )
            if name in ledger
        },
        "field_sha256": hashlib.sha256(field.tobytes(order="C")).hexdigest(),
        "memory_ledger_bytes": sum(ledger.values()),
        "kernel_threads": threads,
        "native_threads": threads if backend_name == "native" else 0,
        "fftw_threads": fftw_threads,
        "blas_threads": blas_threads,
        "gemm_provider": str(metadata.get("gemm_provider", "not applicable")),
        "synthesis_provider": str(metadata.get("synthesis_provider", "not applicable")),
        "projection_provider": str(
            metadata.get("projection_provider", "not applicable")
        ),
        "blas_backend": blas_backend,
        "propagator_mode": str(metadata.get("propagator_mode", "unknown")),
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND", "<unset>"),
        "omp_places": os.environ.get("OMP_PLACES", "<unset>"),
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", "<unset>"),
        "load_start": load_start,
        "load_end": load_end,
    }
    print(json.dumps(result, sort_keys=True))


def _csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return values


def _run(
    backend: str,
    threads: int,
    blas_threads: int,
    case_name: str,
    mode: str,
    trial_count: int,
    node_chunk: int | None,
    progress_seconds: float,
) -> Result:
    environment = os.environ.copy()
    environment.update(
        {
            "NUMBA_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(blas_threads),
            "SOLPROP_BLAS_THREADS": str(blas_threads),
            "SOLPROP_FFTW_THREADS": str(threads),
            "SOLPROP_NATIVE_THREADS": str(threads),
            "SOLPROP_NUMBA_THREADS": str(threads),
        }
    )
    environment.setdefault("OMP_DYNAMIC", "FALSE")
    environment.setdefault("OMP_PROC_BIND", "spread")
    environment.setdefault("OMP_PLACES", "cores")
    environment.setdefault("OMP_WAIT_POLICY", "active")
    command = [
        sys.executable,
        __file__,
        "--worker",
        backend,
        "--case",
        case_name,
        "--mode",
        mode,
        "--trial-count",
        str(trial_count),
    ]
    if node_chunk is not None:
        command.extend(("--node-chunk", str(node_chunk)))
    if progress_seconds > 0.0:
        command.extend(("--progress-seconds", str(progress_seconds)))
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
    )
    return cast(Result, json.loads(completed.stdout.splitlines()[-1]))


def main() -> None:
    """Run isolated CtC workers and emit JSON samples with robust summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("numba_cpu", "native"))
    parser.add_argument(
        "--case", choices=("ctc_reduced", "ctc_full"), default="ctc_reduced"
    )
    parser.add_argument("--mode", choices=("trial", "propagation"), default="trial")
    parser.add_argument("--trial-count", type=int, default=5)
    parser.add_argument("--node-chunk", type=int)
    parser.add_argument("--progress-seconds", type=float, default=0.0)
    parser.add_argument("--backends", type=_csv, default=["native"])
    parser.add_argument("--threads", type=_csv, default=["8"])
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    arguments = parser.parse_args()
    if min(arguments.trial_count, arguments.blas_threads, arguments.repeats) < 1:
        parser.error("counts must be positive")
    if arguments.node_chunk is not None and arguments.node_chunk < 1:
        parser.error("--node-chunk must be positive")
    if not math.isfinite(arguments.progress_seconds) or arguments.progress_seconds < 0:
        parser.error("--progress-seconds must be finite and non-negative")
    if arguments.worker is not None:
        _worker(
            arguments.worker,
            arguments.case,
            arguments.mode,
            arguments.trial_count,
            arguments.node_chunk,
            arguments.progress_seconds,
        )
        return
    for backend in arguments.backends:
        for thread_text in arguments.threads:
            threads = int(thread_text)
            samples = [
                _run(
                    backend,
                    threads,
                    arguments.blas_threads,
                    arguments.case,
                    arguments.mode,
                    arguments.trial_count,
                    arguments.node_chunk,
                    arguments.progress_seconds,
                )
                for _ in range(arguments.repeats)
            ]
            median = statistics.median(sample["seconds"] for sample in samples)
            chosen = min(samples, key=lambda sample: abs(sample["seconds"] - median))
            measurement: Measurement = {
                **chosen,
                "minimum_seconds": min(sample["seconds"] for sample in samples),
                "maximum_seconds": max(sample["seconds"] for sample in samples),
                "median_absolute_deviation": statistics.median(
                    abs(sample["seconds"] - median) for sample in samples
                ),
            }
            print(json.dumps(measurement, sort_keys=True))


if __name__ == "__main__":
    main()
