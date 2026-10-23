# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure the period-one statistics channel inside propagation."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import replace
from typing import Any, TypedDict, cast

PERIODIC = ("energy", "peak_power", "fwhm_t", "electron_density")


class Arm(TypedDict):
    """One propagation arm."""

    seconds: float
    steps: int
    rows: int


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
    lambda0: float
    tau_fwhm: float
    energy: float
    loss: bool


def _case_tables() -> tuple[CaseTable, FamilyTable]:
    from m2_cases import CASES, FAMILIES

    return cast(CaseTable, CASES["m2_reduced"]), cast(
        FamilyTable, FAMILIES["m2_reduced"]
    )


def _worker(backend_name: str, case_name: str, target: float, repeats: int) -> None:
    blas_backend = "not applicable"
    blas_threads: int | str = "not applicable"
    fftw_threads = int(os.environ.get("SOLPROP_FFTW_THREADS", "1"))
    native_threads = int(os.environ.get("SOLPROP_NATIVE_THREADS", "1"))
    if backend_name == "native":
        import solprop_native

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

    from solprop.backends.registry import select
    from solprop.driver import Simulation
    from solprop.exceptions import MaterialRangeWarning
    from solprop.stats import StatsSpec

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        if case_name == "ctc_reduced":
            from ctc import build_simulation

            simulation = build_simulation(case_name, backend_name)
        else:
            case, family = _case_tables()
            simulation = Simulation.from_capillary(
                case["radius"],
                case["length"],
                family["gas"],
                case["pressure"],
                pressure_floor=0.0,
                lambda0=family["lambda0"],
                wavelength_limits=(case["lambda_min"], case["lambda_max"]),
                trange=case["trange"],
                fwhm=family["tau_fwhm"],
                energy=family["energy"],
                modes=case["modes"],
                loss=family["loss"],
                n_save=case["n_save"],
                rtol=1e-6,
                atol=1e-10,
                backend=backend_name,
            )
        plain = simulation.translate()
    stashed = replace(
        plain,
        stats_spec=StatsSpec(
            names=PERIODIC,
            periodic=("electron_density",),
            period=1_000_000_000,
        ),
    )
    watched = replace(
        plain,
        stats_spec=StatsSpec(names=PERIODIC, periodic=PERIODIC, period=1),
    )

    def run(problem: Any) -> Arm:
        backend = select(
            backend_name,
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        try:
            backend.set_initial(problem.initial.field0, simulation.noise)
            started = time.perf_counter()
            summary = backend.propagate_to(target)
            seconds = time.perf_counter() - started
            rows = backend.observe().step_stats.n_rows
            return {"seconds": seconds, "steps": summary.n_steps, "rows": rows}
        finally:
            backend.close()

    run(plain)
    run(stashed)
    run(watched)
    problems = {"plain": plain, "stashed": stashed, "watched": watched}
    best: dict[str, Arm] = {
        label: {"seconds": math.inf, "steps": 0, "rows": 0} for label in problems
    }
    samples: dict[str, list[Arm]] = {label: [] for label in problems}
    for repetition in range(repeats):
        labels = list(problems)
        offset = repetition % len(labels)
        order = labels[offset:] + labels[:offset]
        for label in order:
            measured = run(problems[label])
            samples[label].append(measured)
            if measured["seconds"] < best[label]["seconds"]:
                best[label] = measured

    if best["plain"]["steps"] != best["watched"]["steps"]:
        raise RuntimeError("the statistics channel changed the step count")
    if best["plain"]["steps"] != best["stashed"]["steps"]:
        raise RuntimeError("the plasma stash changed the step count")
    if best["watched"]["rows"] != best["watched"]["steps"]:
        raise RuntimeError("period-one statistics did not produce one row per step")
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
    parser.add_argument("--backend", choices=("numba_cpu", "native"), default="native")
    parser.add_argument(
        "--case", choices=("m2_reduced", "ctc_reduced"), default="m2_reduced"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target", type=float, default=0.2)
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.target <= 0.0:
        parser.error("repeats and target must be positive")
    if arguments.case == "ctc_reduced" and arguments.target > 0.1:
        parser.error("the reduced CtC target cannot exceed its 0.1 m length")
    _worker(arguments.backend, arguments.case, arguments.target, arguments.repeats)


if __name__ == "__main__":
    main()
