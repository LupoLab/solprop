# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure the full N=6 soliton run in isolated backend/thread processes."""

# ruff: noqa: I001 -- Kokkos must load before solprop can import pyFFTW on macOS.

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any, cast

from device_record import DeviceRecord, device_record


class Result(DeviceRecord):
    """One validated M1 measurement."""

    backend: str
    device_before: DeviceRecord
    mode: str
    trials: int
    threads: int
    realized_threads: int
    fftw_threads: int
    seconds: float
    accepted_steps: int
    rejected_steps: int
    milliseconds_per_trial: float
    milliseconds_per_accepted_step: float | None
    target_fraction: float
    omp_proc_bind: str
    omp_places: str
    omp_wait_policy: str
    kmp_blocktime: str
    grid_scale: float
    n_omega: int
    n_to: int
    work_elements: int
    warmed: bool
    load_start: float
    load_end: float


class Measurement(Result):
    """The median run plus the observed three-run interval."""

    minimum_seconds: float
    maximum_seconds: float


def _worker(
    backend_name: str,
    target_fraction: float,
    grid_scale: float,
    warmup: bool,
    mode: str,
    trial_count: int,
) -> None:
    if backend_name == "native":
        # macOS must load the extension's OpenMP runtime before pyFFTW's wheel.
        __import__("solprop_native")

    import numpy as np

    from solprop.backends.registry import select
    from solprop.grid import EnvelopeGrid
    from solprop.nonlinear.gnlse import problem_from
    from solprop.pulses import SechPulse
    from solprop.stepper import PIController
    from solprop.translate.problem import assemble

    beta2 = -11.83e-27
    gamma = 0.11
    tau0 = 100e-15
    wavelength = 835e-9
    period = np.pi / 2 * tau0**2 / abs(beta2)
    target = target_fraction * period
    grid = EnvelopeGrid.from_limits(1.0, wavelength, (500e-9, 2e-6), grid_scale * 4e-12)
    model = problem_from(
        [0.0, 0.0, beta2],
        omega_ref=grid.omega0,
        gamma=gamma,
        shock=False,
    )
    peak_power = 36 * abs(beta2) / (gamma * tau0**2)
    field = SechPulse(wavelength=wavelength, width=tau0, peak_power=peak_power).build(
        grid
    )
    problem = assemble(
        grid,
        model.mode,
        [model.kerr],
        field,
        controller=PIController(rtol=1e-10, atol=1e-18),
        save_planes=[target],
        dz_max=period,
        dz_init=1e-4,
        density=1.0,
        shock=False,
        prefactor_index=1.0,
    )
    backend: Any = select(backend_name, problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        if warmup:
            if mode == "trial":
                backend._trial_step(0.0, problem.stepper.dz_init)
            else:
                backend.propagate_to(min(target, 1e-4 * period))
            backend.set_initial(problem.initial.field0, None)
        device_before = device_record(backend.metadata())
        load_start = os.getloadavg()[0]
        started = time.perf_counter()
        if mode == "trial":
            for _ in range(trial_count):
                backend._trial_step(0.0, problem.stepper.dz_init)
            seconds = time.perf_counter() - started
            _, candidate = backend._trial_step(0.0, problem.stepper.dz_init)
            backend._finish(candidate, problem.stepper.dz_init)
            accepted = 0
            rejected = 0
            trials = trial_count
        else:
            summary = backend.propagate_to(target)
            seconds = time.perf_counter() - started
            accepted = summary.n_steps - summary.n_rejected
            rejected = summary.n_rejected
            trials = summary.n_steps + summary.n_rejected
        field_out = backend.observe().field
        load_end = os.getloadavg()[0]
        if not np.all(np.isfinite(field_out)):
            raise RuntimeError("M1 produced a non-finite field")
        metadata = backend.metadata()
        device_after = device_record(metadata)
    finally:
        backend.close()

    recorded_threads = metadata.get("thread_count", metadata.get("numba_threads"))
    realized = (
        recorded_threads
        if isinstance(recorded_threads, int)
        else int(os.environ["BENCH_THREADS"])
    )
    recorded_fftw_threads = metadata.get("fftw_threads")
    realized_fftw_threads = (
        recorded_fftw_threads if isinstance(recorded_fftw_threads, int) else realized
    )
    result: Result = {
        **device_after,
        "backend": backend_name,
        "device_before": device_before,
        "mode": mode,
        "trials": trials,
        "threads": int(os.environ["BENCH_THREADS"]),
        "realized_threads": realized,
        "fftw_threads": realized_fftw_threads,
        "seconds": seconds,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "milliseconds_per_trial": 1000.0 * seconds / trials,
        "milliseconds_per_accepted_step": (
            1000.0 * seconds / accepted if accepted else None
        ),
        "target_fraction": target_fraction,
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND", "<unset>"),
        "omp_places": os.environ.get("OMP_PLACES", "<unset>"),
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", "<unset>"),
        "kmp_blocktime": os.environ.get("KMP_BLOCKTIME", "<unset>"),
        "grid_scale": grid_scale,
        "n_omega": grid.n_omega,
        "n_to": grid.n_to,
        "work_elements": grid.n_to,
        "warmed": warmup,
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
    target_fraction: float,
    grid_scale: float,
    warmup: bool,
    mode: str,
    trial_count: int,
) -> Result:
    environment = os.environ.copy()
    environment.update(
        {
            "BENCH_THREADS": str(threads),
            "NUMBA_NUM_THREADS": str(threads),
            "SOLPROP_NATIVE_THREADS": str(threads),
        }
    )
    environment.setdefault("OMP_DYNAMIC", "FALSE")
    environment.setdefault("OMP_PROC_BIND", "spread")
    environment.setdefault("OMP_PLACES", "cores")
    command = [
        sys.executable,
        __file__,
        "--worker",
        backend,
        "--fraction",
        str(target_fraction),
        "--grid-scale",
        str(grid_scale),
        "--mode",
        mode,
        "--trial-count",
        str(trial_count),
    ]
    if warmup:
        command.append("--warmup")
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return cast(Result, json.loads(completed.stdout.splitlines()[-1]))


def main() -> None:
    """Run workers and print the median end-to-end M1 table as JSON lines."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("numba_cpu", "native"))
    parser.add_argument("--backends", type=_parse_csv, default=["numba_cpu", "native"])
    parser.add_argument("--threads", type=_parse_csv, default=["1", "8"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--mode", choices=("trial", "propagation"), default="propagation"
    )
    parser.add_argument("--trial-count", type=int, default=7)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="fraction of the full soliton period to propagate",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="run one untimed adaptive step before the measured propagation",
    )
    parser.add_argument(
        "--grid-scale",
        type=float,
        default=1.0,
        help="multiple of the M1 time window, and therefore approximately its FFT size",
    )
    arguments = parser.parse_args()
    if not 0.0 < arguments.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1]")
    if arguments.grid_scale <= 0.0:
        parser.error("--grid-scale must be positive")
    if arguments.worker is not None:
        _worker(
            arguments.worker,
            arguments.fraction,
            arguments.grid_scale,
            arguments.warmup,
            arguments.mode,
            arguments.trial_count,
        )
        return
    if arguments.repeats < 1 or arguments.trial_count < 1:
        parser.error("--repeats and --trial-count must be positive")

    for backend in arguments.backends:
        if backend not in {"numba_cpu", "native"}:
            parser.error(f"unknown backend {backend!r}")
        for thread_text in arguments.threads:
            threads = int(thread_text)
            if threads < 1:
                parser.error("thread counts must be positive")
            samples = [
                _run(
                    backend,
                    threads,
                    arguments.fraction,
                    arguments.grid_scale,
                    arguments.warmup,
                    arguments.mode,
                    arguments.trial_count,
                )
                for _ in range(arguments.repeats)
            ]
            middle = statistics.median(sample["seconds"] for sample in samples)
            chosen = min(samples, key=lambda sample: abs(sample["seconds"] - middle))
            measurement: Measurement = {
                **chosen,
                "minimum_seconds": min(sample["seconds"] for sample in samples),
                "maximum_seconds": max(sample["seconds"] for sample in samples),
            }
            print(json.dumps(measurement, sort_keys=True))


if __name__ == "__main__":
    main()
