# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure the periodic statistics channel inside propagation, against its budget.

The channel's budget is under a quarter of a step at a row per step (``--period 1``)
with energy, peak power, the pulse duration and the electron density, on the reduced
M2 case; the test suite asserts no cost (``docs/dev/testing.md``), so this harness is
where the budget is measured, by hand on an idle machine, and recorded with the
release benchmarks.

Each case runs three ways through the backend's session to ``--target``, interleaved
round by round after one warm-up of each: ``plain`` (no periodic statistics),
``stashed`` (the electron density's plasma stash alone) and ``watched`` (every periodic
row). The step counts of the three must agree and the watched run must give a row per
period, or the harness refuses. The JSON line reports, from the fastest repeat of each,
``channel_fraction`` (watched over plain, minus one: what the channel costs per step),
``collection_fraction`` (watched over stashed) and the per-row and per-step times, with
``budget`` and ``within_budget`` for the channel. ``--stepper runge_kutta`` measures
under the Runge-Kutta march, whose plasma rows reuse the accepted step's last stage;
the default Adams step evaluates at its predicted state, so its rows scan afresh
(``docs/guide/results/statistics.md``). ``m2_reduced`` is the two-mode helium RDW
capillary and ``ctc_reduced`` the one-mode hydrogen comb-to-continuum capillary below,
both at rtol 1e-6 and atol 1e-10::

    python -m benchmarks.statistics_cost --case m2_reduced --repeats 5
    python -m benchmarks.statistics_cost --stepper runge_kutta --target 0.002
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

from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.backends.sessions import start_session
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.stats import StatsSpec
from solprop.translate.run_request import RunRequest

PERIODIC = ("energy", "peak_power", "fwhm_t", "electron_density")

BUDGET = 0.25
"""The channel's budget at a row per step: under a quarter of a step."""


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

CTC_REDUCED = {
    "radius": 16e-6,
    "length": 0.1,
    "gas": "H2",
    "pressure": 40.0,
    "modes": 1,
    "wavelength": 515e-9,
    "wavelength_limits": (250e-9, 2e-6),
    "trange": 2e-12,
    "fwhm": 400e-15,
    "energy": 10e-6,
    "n_save": 21,
}
"""The reduced comb-to-continuum capillary: 400 fs, 10 uJ at 515 nm in 40 bar of
hydrogen, lossless, with Raman, over 0.1 m in one mode."""


def _worker(  # noqa: PLR0915 - one measurement: set up, run the arms, report
    backend_name: str,
    case_name: str,
    *,
    target: float,
    repeats: int,
    modes: int | None,
    period: int,
    stepper: str,
) -> None:
    blas_backend = "not applicable"
    blas_threads: int | str = "not applicable"
    fftw_threads = 1
    native_threads = int(os.environ.get("SOLPROP_NATIVE_THREADS", "1"))
    if backend_name == "native":
        import solprop_native  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

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
            case = CTC_REDUCED
            simulation = Simulation.from_capillary(
                case["radius"],
                case["length"],
                case["gas"],
                case["pressure"],
                wavelength=case["wavelength"],
                wavelength_limits=case["wavelength_limits"],
                trange=case["trange"],
                fwhm=case["fwhm"],
                energy=case["energy"],
                modes=case["modes"],
                loss=False,
                raman=True,
                n_save=case["n_save"],
                rtol=1e-6,
                atol=1e-10,
                backend=backend_name,
            )
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
        # An integrator of None is the Runge-Kutta march.
        integrator = None if stepper == "runge_kutta" else request.stepper.integrator
        request = request.with_stepper(
            dataclasses.replace(
                request.stepper, save_planes=(target,), integrator=integrator
            )
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
                "budget": BUDGET,
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
                "stepper": stepper,
                "target": target,
                "watched": best["watched"],
                "within_budget": ratio < BUDGET,
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
    parser.add_argument(
        "--stepper",
        choices=("default", "runge_kutta"),
        default="default",
        help="the run's own integrator, or the Runge-Kutta march",
    )
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
        target=arguments.target,
        repeats=arguments.repeats,
        modes=arguments.modes,
        period=arguments.period,
        stepper=arguments.stepper,
    )


if __name__ == "__main__":
    main()
