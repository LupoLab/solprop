# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure the periodic statistics channel inside propagation.

Each case runs three ways through the backend's session, interleaved round by round:
``plain`` (no periodic statistics), ``stashed`` (the electron density's plasma stash
alone) and ``watched`` (every periodic row). The JSON line reports what the channel and
its collection cost per step and per row. ``m2_reduced`` is the two-mode helium RDW
capillary below; ``ctc_reduced`` is the registry's case of that name::

    python -m benchmarks.statistics_cost --case m2_reduced --repeats 5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
import warnings
from typing import TypedDict

import numpy as np
import solprop_native

from benchmarks.builders import capillary, variant
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.stats import StatsSpec
from solprop.translate.run_request import RunRequest

PERIODIC = ("energy", "peak_power", "fwhm_t", "electron_density")


class Arm(TypedDict):
    """One propagation arm."""

    seconds: float
    steps: int
    rows: int


M2_REDUCED = {
    "radius": 125e-6,
    "length": 1.0,
    "gas": "He",
    "pressure": (1.2, 0.0),
    "modes": 2,
    "wavelength": 800e-9,
    "wavelength_limits": (100e-9, 1.5e-6),
    "trange": 200e-15,
    "fwhm": 7.5e-15,
    "energy": 275e-6,
    "n_save": 51,
}
"""The reduced resonant-dispersive-wave capillary: the ``rdw`` case's 7.5 fs, 275 uJ
pulse at 800 nm in helium, over 1 m at 1.2 bar falling to vacuum, two modes."""


def _worker(  # noqa: PLR0915 - one measurement: set up, run the arms, report
    backend_name: str,
    case_name: str,
    target: float,
    repeats: int,
    modes: int | None,
    period: int,
) -> None:
    blas_backend = "not applicable"
    blas_threads: int | str = "not applicable"
    fftw_threads = 1
    native_threads = int(os.environ.get("SOLPROP_NATIVE_THREADS", "1"))
    if backend_name == "native":
        build = solprop_native.build_info()
        blas_backend = build.blas_backend
        fftw_threads = build.fftw_threads
        native_threads = build.thread_count
        if blas_backend == "OpenBLAS":
            blas_threads = int(os.environ.get("OPENBLAS_NUM_THREADS", "1"))
        elif blas_backend == "Accelerate":
            blas_threads = "system-managed"
        else:
            blas_threads = "disabled"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        if case_name == "ctc_reduced":
            simulation = capillary(case_name, variant({"backend": backend_name}))
        else:
            case = M2_REDUCED
            simulation = Simulation.from_capillary(
                case["radius"],
                case["length"],
                case["gas"],
                case["pressure"],
                pressure_floor=0.0,
                wavelength=case["wavelength"],
                wavelength_limits=case["wavelength_limits"],
                trange=case["trange"],
                fwhm=case["fwhm"],
                energy=case["energy"],
                modes=case["modes"] if modes is None else modes,
                loss=True,
                n_save=case["n_save"],
                rtol=1e-6,
                atol=1e-10,
                backend=backend_name,
            )
        problem = simulation.translate()
        plain = simulation._run_request()  # pyright: ignore[reportPrivateUsage]
    stashed = plain.for_run(
        stats=StatsSpec(
            names=PERIODIC,
            periodic=("electron_density",),
            period=1_000_000_000,
        ),
    )
    watched = plain.for_run(
        stats=StatsSpec(names=PERIODIC, periodic=PERIODIC, period=period),
    )

    def run(request: RunRequest) -> Arm:
        # One plane at the target, through the backend's session (SEP 47): the
        # accepted steps and the periodic rows its batches carry.
        request = request.with_stepper(
            dataclasses.replace(request.stepper, save_planes=(target,))
        )
        backend = select(
            backend_name,
            problem,
            request,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        start = Start(
            field=problem.initial.field0, z=problem.initial.z, noise=simulation.noise
        )
        session = start_session(backend, problem, request, start)
        steps = rows = 0
        try:
            started = time.perf_counter()
            while not session.finished:
                batch = session.advance()
                for event in batch.events:
                    history = getattr(event, "history", None)
                    if history is not None:
                        steps += int(np.count_nonzero(history.accepted))
                    periodic = getattr(event, "step_stats", None)
                    if periodic is not None:
                        rows += int(periodic.n_rows)
                session.acknowledge(batch.receipt)
            seconds = time.perf_counter() - started
        finally:
            session.close()
        return {"seconds": seconds, "steps": steps, "rows": rows}

    run(plain)
    run(stashed)
    run(watched)
    requests = {"plain": plain, "stashed": stashed, "watched": watched}
    best: dict[str, Arm] = {
        label: {"seconds": math.inf, "steps": 0, "rows": 0} for label in requests
    }
    samples: dict[str, list[Arm]] = {label: [] for label in requests}
    for repetition in range(repeats):
        labels = list(requests)
        offset = repetition % len(labels)
        order = labels[offset:] + labels[:offset]
        for label in order:
            measured = run(requests[label])
            samples[label].append(measured)
            if measured["seconds"] < best[label]["seconds"]:
                best[label] = measured

    if best["plain"]["steps"] != best["watched"]["steps"]:
        raise RuntimeError("the statistics channel changed the step count")
    if best["plain"]["steps"] != best["stashed"]["steps"]:
        raise RuntimeError("the plasma stash changed the step count")
    if best["watched"]["rows"] != best["watched"]["steps"] // period:
        raise RuntimeError("the statistics did not produce one row per period")
    ratio = best["watched"]["seconds"] / best["plain"]["seconds"] - 1.0
    collection_ratio = best["watched"]["seconds"] / best["stashed"]["seconds"] - 1.0
    step_seconds = best["plain"]["seconds"] / best["plain"]["steps"]
    print(
        json.dumps(
            {
                "backend": backend_name,
                "case": case_name,
                "blas_backend": blas_backend,
                "blas_threads": blas_threads,
                "channel_fraction": ratio,
                "channel_percent": 100.0 * ratio,
                "collection_fraction": collection_ratio,
                "collection_percent": 100.0 * collection_ratio,
                "fftw_threads": fftw_threads,
                "native_threads": native_threads,
                "period": period,
                "load": os.getloadavg()[0],
                "milliseconds_per_row": 1000.0
                * (best["watched"]["seconds"] - best["plain"]["seconds"])
                / best["watched"]["rows"],
                "milliseconds_per_step": 1000.0 * step_seconds,
                "plain": best["plain"],
                "repeats": repeats,
                "samples": samples,
                "stashed": best["stashed"],
                "target": target,
                "watched": best["watched"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    """Run an isolated channel-cost measurement."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numpy_ref", "native"), default="native")
    parser.add_argument(
        "--case", choices=("m2_reduced", "ctc_reduced"), default="m2_reduced"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target", type=float, default=0.2)
    parser.add_argument(
        "--modes", type=int, help="m2_reduced's mode count, in place of its table's"
    )
    parser.add_argument("--period", type=int, default=1, help="steps between rows")
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.target <= 0.0 or arguments.period < 1:
        parser.error("repeats, target and period must be positive")
    if arguments.case == "ctc_reduced" and arguments.target > 0.1:
        parser.error("the reduced CtC target cannot exceed its 0.1 m length")
    if arguments.modes is not None and arguments.case != "m2_reduced":
        parser.error("--modes applies to m2_reduced only")
    _worker(
        arguments.backend,
        arguments.case,
        arguments.target,
        arguments.repeats,
        arguments.modes,
        arguments.period,
    )


if __name__ == "__main__":
    main()
