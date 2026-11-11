# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure reduced M2 propagation in isolated backend/thread processes.

A propagation is the backend's session (solprop SEP 47): its start loads the launch
before the clock starts, and the timed region is the session's advances to its last
plane, with every observation's field exported (the internal convention, as the step
driver observed it) and its statistics, which the session evaluates there. A trial is
the backend's timing probe (``Backend.timing_probe``), which loads the start once and
times single trials from it. ``--planes`` keeps a run's first planes only, for a short
check of the script. The untimed warm-up is the launch plane and one attempt of a
session of the timed run itself, closed there: a backend serves only its own request's
planes, where the step driver's warm-up marched to a short plane of its own.
"""

# ruff: noqa: I001 -- Kokkos loads first, before any other OpenMP runtime on macOS.

from __future__ import annotations

import argparse
import dataclasses
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


def _warm(session: Any) -> None:
    """Take a started session's launch plane and one attempt beyond it, then close it.

    The untimed warm-up: a backend serves only its own request's planes, so it is
    warmed on the timed run's own itinerary, cut short; the timed run loads the
    launch again.
    """
    try:
        for _ in range(2):  # the launch plane's batch, then one attempt
            if session.finished:
                break
            batch = session.advance(work_limit=1)
            terminal = batch.terminal
            if terminal is not None and terminal.error is not None:
                raise terminal.error
            session.acknowledge(batch.receipt)
    finally:
        session.close()


def _march(session: Any) -> tuple[int, int, Any]:
    """Advance a started session to its end, exporting every observation's field.

    Each observation's statistics arrive with it. Returns the attempts, the rejected
    attempts among them and the last plane's field, in the internal convention.
    """
    import numpy as np

    from solprop.backends.events import ObservationReady
    from solprop.backends.export import ExportRecipe

    identity = ExportRecipe(power=False, optics=False)
    attempts = rejected = 0
    field = None
    try:
        while not session.finished:
            batch = session.advance()
            for event in batch.events:
                history = getattr(event, "history", None)
                if history is not None:
                    attempts += history.n_steps
                    rejected += history.n_rejected
                if isinstance(event, ObservationReady):
                    field = np.array(session.export(event.ticket, identity))
            terminal = batch.terminal
            if terminal is not None and terminal.error is not None:
                raise terminal.error
            session.acknowledge(batch.receipt)
    finally:
        session.close()
    return attempts, rejected, field


def _worker(
    backend_name: str,
    case_name: str,
    field_family: str,
    save_field: Path | None,
    mode: str,
    trial_count: int,
    planes: int | None,
) -> None:
    if backend_name == "native":
        # Kokkos loads first, the order documented for local applications.
        __import__("solprop_native")

    import numpy as np

    from solprop.backends.node_chunk import plan_node_chunk
    from solprop.backends.registry import select
    from solprop.backends.session import Start
    from solprop.backends.sessions import start_session
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
        request = simulation._run_request()  # pyright: ignore[reportPrivateUsage]
    if planes is not None:
        request = request.with_stepper(
            dataclasses.replace(
                request.stepper, save_planes=request.stepper.save_planes[:planes]
            )
        )

    def entrance() -> Start:
        return Start.entrance(problem, noise=simulation.noise)

    construction_started = time.perf_counter()
    backend: Any = select(
        backend_name,
        problem,
        request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    construction_seconds = time.perf_counter() - construction_started
    # The chunk a native response pass holds; the reference's is the host plan's.
    node_chunk = (
        int(backend.node_chunk)
        if backend_name == "native"
        else plan_node_chunk(problem).chunk
    )
    dz = float(request.stepper.dz_init)
    try:
        memory_ledger = backend.memory_ledger()
        _warm(start_session(backend, problem, request, entrance(), owns=False))
        if mode == "trial":
            probe = backend.timing_probe(entrance())
            device_before = device_record(backend.metadata())
            load_start = os.getloadavg()[0]
            started = time.perf_counter()
            for _ in range(trial_count):
                probe.trial(0.0, dz)
            seconds = time.perf_counter() - started
            probe.trial(0.0, dz)
            field = probe.field()
            attempts = 0
            rejected = 0
            trials = trial_count
        else:
            session = start_session(backend, problem, request, entrance(), owns=False)
            device_before = device_record(backend.metadata())
            load_start = os.getloadavg()[0]
            started = time.perf_counter()
            attempts, rejected, field = _march(session)
            seconds = time.perf_counter() - started
            trials = attempts
        load_end = os.getloadavg()[0]
        if not np.all(np.isfinite(field)):
            raise RuntimeError("reduced M2 produced a non-finite field")
        if save_field is not None:
            np.save(save_field, field)
        metadata = backend.metadata()
        device_after = device_record(metadata)
    finally:
        backend.close()

    accepted = attempts - rejected
    recorded_threads = _metadata_int(metadata, "thread_count", 0)
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
        "modes": problem.shape[0],
        "nodes": problem.transform.n_nodes,
        "node_chunk": node_chunk,
        "components": problem.transform.n_pol * problem.shape[1],
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
    planes: int | None,
) -> Result:
    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": str(blas_threads),
            "SOLPROP_BLAS_THREADS": str(blas_threads),
            "SOLPROP_NATIVE_THREADS": str(threads),
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
            *(() if planes is None else ("--planes", str(planes))),
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
    parser.add_argument("--worker", choices=("numpy_ref", "native"))
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
    parser.add_argument("--backends", type=_parse_csv, default=["numpy_ref", "native"])
    parser.add_argument("--threads", type=_parse_csv, default=["1", "8"])
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--mode", choices=("trial", "propagation"), default="propagation"
    )
    parser.add_argument("--trial-count", type=int, default=7)
    parser.add_argument(
        "--planes",
        type=int,
        help="run the case's first planes only, to check the script",
    )
    arguments = parser.parse_args()
    if arguments.planes is not None and arguments.planes < 1:
        parser.error("--planes must be positive")
    if arguments.worker is not None:
        _worker(
            arguments.worker,
            arguments.case,
            arguments.field_family,
            arguments.save_field,
            arguments.mode,
            arguments.trial_count,
            arguments.planes,
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
                    arguments.planes,
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
