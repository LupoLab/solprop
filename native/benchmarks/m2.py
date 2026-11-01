# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure reduced M2 propagation in isolated backend/thread processes."""

# ruff: noqa: I001 -- Kokkos must load before solprop imports pyFFTW on macOS.

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

from device_record import DeviceRecord, device_record


class Result(DeviceRecord):
    """One reduced-M2 propagation measurement."""

    backend: str
    device_before: DeviceRecord
    mode: str
    trials: int
    case: str
    field_family: str
    seconds: float
    construction_seconds: float
    planning_seconds: float
    memory_ledger: dict[str, int]
    memory_ledger_bytes: int
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
    warmed: bool
    native_threads: int
    fftw_threads: int
    blas_threads: int | str
    blas_thread_policy: str
    gemm_provider: str
    synthesis_provider: str
    projection_provider: str
    blas_backend: str
    omp_proc_bind: str
    omp_places: str
    omp_wait_policy: str
    load_start: float
    load_end: float


class Measurement(Result):
    """The median run plus its spread across isolated processes."""

    minimum_seconds: float
    maximum_seconds: float
    median_absolute_deviation: float


class CaseTable(TypedDict):
    """Geometry and sampling values of the M2 benchmark cases (``m2_cases.py``)."""

    radius: float
    length: float
    pressure: float | tuple[float, float]
    modes: int
    lambda_min: float
    lambda_max: float
    trange: float
    n_save: int
    expected_steps: int


class FamilyTable(TypedDict):
    """Pulse and gas values of the M2 benchmark cases (``m2_cases.py``)."""

    gas: str
    wavelength: float
    tau_fwhm: float
    energy: float
    loss: bool


def _case_tables(case_name: str) -> tuple[CaseTable, FamilyTable]:
    from m2_cases import CASES, FAMILIES

    return cast(CaseTable, CASES[case_name]), cast(FamilyTable, FAMILIES[case_name])


def _metadata_int(metadata: Mapping[str, object], name: str, default: int) -> int:
    """Read one integer-valued backend metadata item without a blind cast."""
    value = metadata.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise TypeError(f"metadata {name!r} is not integer-valued: {value!r}")
    return int(value)


def _metadata_float(metadata: Mapping[str, object], name: str) -> float:
    """Read one floating-point backend metadata item without a blind cast."""
    value = metadata.get(name, 0.0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"metadata {name!r} is not numeric: {value!r}")
    return float(value)


def _worker(
    backend_name: str,
    case_name: str,
    field_family: str,
    save_field: Path | None,
    mode: str,
    trial_count: int,
) -> None:
    if backend_name == "native":
        # The extension and pyFFTW use one OpenMP runtime on macOS only when Kokkos
        # loads first. This is also the order documented for local applications.
        __import__("solprop_native")

    import numpy as np

    from solprop.backends.registry import select
    from solprop.driver import Simulation
    from solprop.exceptions import MaterialRangeWarning

    import warnings

    case, family = _case_tables(case_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(
            case["radius"],
            case["length"],
            family["gas"],
            case["pressure"],
            pressure_floor=0.0,
            wavelength=family["wavelength"],
            wavelength_limits=(case["lambda_min"], case["lambda_max"]),
            trange=case["trange"],
            fwhm=family["tau_fwhm"],
            energy=family["energy"],
            modes=case["modes"],
            loss=family["loss"],
            envelope=field_family == "envelope",
            n_save=case["n_save"],
            rtol=1e-6,
            atol=1e-10,
            backend=backend_name,
        )
        problem = simulation.translate()

    construction_started = time.perf_counter()
    backend: Any = select(
        backend_name,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    construction_seconds = time.perf_counter() - construction_started
    try:
        memory_ledger = backend.memory_ledger()
        backend.set_initial(problem.initial.field0, simulation.noise)
        first_plane = float(problem.stepper.save_planes[0])
        backend.propagate_to(min(first_plane, case["length"] * 1e-4))
        backend.set_initial(problem.initial.field0, simulation.noise)
        device_before = device_record(backend.metadata())
        load_start = os.getloadavg()[0]
        started = time.perf_counter()
        if mode == "trial":
            for _ in range(trial_count):
                backend._trial_step(0.0, problem.stepper.dz_init)
            seconds = time.perf_counter() - started
            _, candidate = backend._trial_step(0.0, problem.stepper.dz_init)
            backend._finish(candidate, problem.stepper.dz_init)
            attempts = 0
            rejected = 0
            trials = trial_count
        else:
            backend.observe()
            backend.stats(problem.stats_spec)
            attempts = 0
            rejected = 0
            for target in problem.stepper.save_planes:
                summary = backend.propagate_to(float(target))
                attempts += summary.n_steps
                rejected += summary.n_rejected
                backend.observe()
                backend.stats(problem.stats_spec)
            seconds = time.perf_counter() - started
            trials = attempts
        load_end = os.getloadavg()[0]
        field = backend.observe().field
        if not np.all(np.isfinite(field)):
            raise RuntimeError("reduced M2 produced a non-finite field")
        if save_field is not None:
            np.save(save_field, field)
        metadata = backend.metadata()
        device_after = device_record(metadata)
    finally:
        backend.close()

    accepted = attempts - rejected
    recorded_threads = _metadata_int(
        metadata, "thread_count", _metadata_int(metadata, "numba_threads", 0)
    )
    recorded_fftw_threads = _metadata_int(metadata, "fftw_threads", recorded_threads)
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
        blas_threads = _metadata_int(
            metadata,
            "blas_threads",
            int(os.environ.get("SOLPROP_BLAS_THREADS", "1")),
        )
    result: Result = {
        **device_after,
        "backend": backend_name,
        "device_before": device_before,
        "mode": mode,
        "trials": trials,
        "case": case_name,
        "field_family": field_family,
        "seconds": seconds,
        "construction_seconds": construction_seconds,
        "planning_seconds": _metadata_float(metadata, "planning_seconds"),
        "memory_ledger": memory_ledger,
        "memory_ledger_bytes": sum(memory_ledger.values()),
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
        "warmed": True,
        "native_threads": recorded_threads,
        "fftw_threads": recorded_fftw_threads,
        "blas_threads": blas_threads,
        "blas_thread_policy": str(
            metadata.get("blas_thread_policy", "backend-managed")
        ),
        "gemm_provider": str(metadata.get("gemm_provider", "not applicable")),
        "synthesis_provider": str(metadata.get("synthesis_provider", "not applicable")),
        "projection_provider": str(
            metadata.get("projection_provider", "not applicable")
        ),
        "blas_backend": blas_backend,
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND", "<unset>"),
        "omp_places": os.environ.get("OMP_PLACES", "<unset>"),
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", "<unset>"),
        "load_start": load_start,
        "load_end": load_end,
    }
    print(json.dumps(result, sort_keys=True))


def _parse_csv(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return result


def _run(
    backend: str,
    threads: int,
    blas_threads: int,
    case_name: str,
    field_family: str,
    mode: str,
    trial_count: int,
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
    completed = subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker",
            backend,
            "--case",
            case_name,
            "--field-family",
            field_family,
            "--mode",
            mode,
            "--trial-count",
            str(trial_count),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return cast(Result, json.loads(completed.stdout.splitlines()[-1]))


def main() -> None:
    """Run isolated workers and emit median reduced-M2 JSON records."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("numba_cpu", "native"))
    parser.add_argument(
        "--case", choices=("m2_reduced", "m2_full"), default="m2_reduced"
    )
    parser.add_argument(
        "--field-family", choices=("field_real", "envelope"), default="field_real"
    )
    parser.add_argument(
        "--save-field",
        type=Path,
        help="worker-only owning field output, written after the timed interval",
    )
    parser.add_argument("--backends", type=_parse_csv, default=["numba_cpu", "native"])
    parser.add_argument("--threads", type=_parse_csv, default=["1", "8"])
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--mode", choices=("trial", "propagation"), default="propagation"
    )
    parser.add_argument("--trial-count", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.worker is not None:
        _worker(
            arguments.worker,
            arguments.case,
            arguments.field_family,
            arguments.save_field,
            arguments.mode,
            arguments.trial_count,
        )
        return
    if arguments.save_field is not None:
        parser.error("--save-field requires --worker")
    if min(arguments.repeats, arguments.blas_threads, arguments.trial_count) < 1:
        parser.error("--repeats, --blas-threads and --trial-count must be positive")
    for backend in arguments.backends:
        for thread_text in arguments.threads:
            threads = int(thread_text)
            if threads < 1:
                parser.error("thread counts must be positive")
            samples = [
                _run(
                    backend,
                    threads,
                    arguments.blas_threads,
                    arguments.case,
                    arguments.field_family,
                    arguments.mode,
                    arguments.trial_count,
                )
                for _ in range(arguments.repeats)
            ]
            median = statistics.median(sample["seconds"] for sample in samples)
            chosen = min(samples, key=lambda sample: abs(sample["seconds"] - median))
            deviations = [abs(sample["seconds"] - median) for sample in samples]
            measurement: Measurement = {
                **chosen,
                "minimum_seconds": min(sample["seconds"] for sample in samples),
                "maximum_seconds": max(sample["seconds"] for sample in samples),
                "median_absolute_deviation": statistics.median(deviations),
            }
            print(json.dumps(measurement, sort_keys=True))


if __name__ == "__main__":
    main()
