# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure paired ten-step CPU windows; keep generated evidence under .benchmarks."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from threadpoolctl import threadpool_info

from solprop.backends import select
from solprop.boundaries import absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import FieldGrid, FreeGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium, critical_power
from solprop.stats import default_spec
from solprop.stepper import FixedSchedule, PIController, RungeKutta


def simulation(modes: int, backend: str, *, mode_averaged: bool = False) -> Simulation:
    """Build a resolved or mode-averaged capillary on the same measured window."""
    sim = Simulation.from_capillary(
        125e-6,
        0.01,
        "Ar",
        1.0,
        lambda0=800e-9,
        wavelength_limits=(150e-9, 4e-6),
        trange=400e-15,
        fwhm=15e-15,
        energy=1e-6,
        modes=1 if mode_averaged else modes,
        plasma=False,
        raman=False,
        integrator=RungeKutta(),
        backend=backend,
        stats=default_spec("mode_averaged" if mode_averaged else "multimode"),
        schedule=FixedSchedule(np.full(10, 1e-5)),
        save_planes=[1e-4],
        rtol=1e-8,
        atol=1e-14,
    )
    if mode_averaged:
        sim.settings = dataclasses.replace(
            sim.settings, quadrature=None, components=None
        )
    return sim


def radial_simulation(
    nodes: int, backend: str, *, cartesian: bool = False
) -> Simulation:
    """Build a scalar Kerr beam at 0.9 critical power for ten steps."""
    grid = FieldGrid.from_limits(0.01, 800e-9, (300e-9, 2e-6), 200e-15)
    medium = SpatialMedium("Ar", 1.0)
    coefficient = gamma3("Ar", wavelength=800e-9)
    density = float(medium.density(np.array([0.0]))[0])
    power = 0.9 * critical_power(800e-9, coefficient * density)
    constructor = Simulation.from_free if cartesian else Simulation.from_radial
    return constructor(
        grid,
        FreeGrid(500e-6, nodes, 500e-6, nodes)
        if cartesian
        else RadialGrid(1e-3, nodes),
        medium,
        [KerrResponse(coefficient, thg=True)],
        SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, peak_power=power), 100e-6
        ),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[1e-4],
            dz_init=1e-5,
            dz_max=1e-4,
            integrator=RungeKutta(),
            schedule=FixedSchedule(np.full(10, 1e-5)),
            absorber=absorber(grid),
            stats=default_spec("free" if cartesian else "radial"),
            backend=backend,
        ),
    )


def worker(args: argparse.Namespace) -> dict[str, object]:
    """Time construction, first propagation, warm windows and host statistics."""
    start = time.perf_counter()
    sim = (
        radial_simulation(args.modes, args.worker, cartesian=args.cartesian)
        if args.radial or args.cartesian
        else simulation(args.modes, args.worker, mode_averaged=args.mode_averaged)
    )
    problem = sim.translate()
    if args.propagator_mode != "auto":
        problem = dataclasses.replace(
            problem,
            stepper=dataclasses.replace(
                problem.stepper, propagator_mode=args.propagator_mode
            ),
        )
    translation = time.perf_counter() - start
    start = time.perf_counter()
    engine = select(args.worker, problem, grid=sim.grid, modes=sim.modes)
    setup = time.perf_counter() - start
    cpu_before = psutil.cpu_percent(interval=0.2)
    timings, stat_times = [], []
    try:
        for _ in range(args.repeats + 1):
            engine.set_initial(problem.initial.field0, None)
            start = time.perf_counter()
            engine.propagate_to(1e-4)
            timings.append(time.perf_counter() - start)
            start = time.perf_counter()
            engine.observe()
            engine.stats(problem.stats_spec)
            stat_times.append(time.perf_counter() - start)
        np.save(args.output.with_suffix(".npy"), engine.observe().field)
        if args.profile and args.worker == "julia":
            import juliacall  # noqa: PLC0415 - after private runtime initialization

            juliacall.Main.include(str(Path(__file__).with_name("profile.jl")))
            report = juliacall.Main.profile_window(
                engine._engine, problem.initial.field0[0, :, 0, :].T.copy()
            )
            args.output.with_suffix(".profile.txt").write_text(
                str(report), encoding="utf-8"
            )
        return {
            "backend": args.worker,
            "modes": problem.shape[1],
            "transform_kind": problem.transform.kind,
            "threads": args.threads,
            "translation_s": translation,
            "setup_s": setup,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "first_window_s": timings[0],
            "warm_s": timings[1:],
            "observation_stats_s": stat_times[1:],
            "metadata": engine.metadata(),
            "shape": problem.shape,
            "spatial_shape": None
            if problem.transform.geometry is None
            else problem.transform.geometry.shape,
            "fine_samples": problem.grids.n_to,
            "nodes": problem.transform.n_nodes,
            "node_chunk": problem.transform.node_chunk,
            "owned_bytes": sum(engine.memory_ledger().values()),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "numba", "pyfftw", "juliacall")
            },
            "load": os.getloadavg(),
            "system_cpu_percent_before": cpu_before,
            "threadpools": threadpool_info(),
            "numpy_blas": np.show_config(mode="dicts")["Build Dependencies"]["blas"],
            "julia_source_sha256": hashlib.sha256(
                b"".join(
                    path.read_bytes()
                    for path in sorted(Path(__file__).with_name("src").glob("*.jl"))
                )
            ).hexdigest(),
        }
    finally:
        engine.close()


def cpu_name() -> str:
    """Read the host CPU identity without assuming macOS."""
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            capture_output=True,
            check=False,
        )
        return (
            result.stdout.strip() if result.returncode == 0 else "CPU model unavailable"
        )
    if sys.platform == "linux":
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def paired(args: argparse.Namespace) -> dict[str, object]:
    """Run alternating fresh-process pairs and retain each measured sample."""
    args.output.mkdir(parents=True, exist_ok=True)
    widths = args.widths or list(dict.fromkeys([1, 2, 4, psutil.cpu_count(False) or 1]))
    records = []
    counts = (
        (32, 64)
        if args.cartesian
        else (64, 256)
        if args.radial
        else (1,)
        if args.mode_averaged
        else (4, 16)
    )
    for modes in counts:
        for width in widths:
            for round_index in range(args.rounds):
                names = ("numba_cpu", "julia")
                if round_index % 2:
                    names = names[::-1]
                pair = {}
                for name in names:
                    output = (
                        args.output / f"m{modes}-t{width}-r{round_index}-{name}.json"
                    )
                    env = os.environ | {
                        "SOLPROP_JULIA_THREADS": str(width),
                        "SOLPROP_NUMBA_THREADS": str(width),
                        "NUMBA_NUM_THREADS": str(width),
                        "SOLPROP_FFTW_THREADS": str(width),
                        "SOLPROP_BLAS_THREADS": str(width),
                        "SOLPROP_FFTW_PLAN": "estimate",
                        "OPENBLAS_NUM_THREADS": str(width),
                        "OMP_NUM_THREADS": str(width),
                        "VECLIB_MAXIMUM_THREADS": str(width),
                    }
                    command = [
                        sys.executable,
                        __file__,
                        "--worker",
                        name,
                        "--modes",
                        str(modes),
                        "--threads",
                        str(width),
                        "--repeats",
                        str(args.repeats),
                        "--output",
                        str(output),
                    ]
                    if args.mode_averaged:
                        command.append("--mode-averaged")
                    if args.cartesian:
                        command.append("--cartesian")
                    if args.radial:
                        command.append("--radial")
                    subprocess.run(command, env=env, check=True)
                    record = json.loads(output.read_text(encoding="utf-8"))
                    record["round"] = round_index
                    records.append(record)
                    pair[name] = np.load(output.with_suffix(".npy"))
                    sys.stdout.write(
                        f"modes={modes} threads={width} round={round_index + 1} "
                        f"{name}: {statistics.median(record['warm_s']) * 1000:.3f} ms\n"
                    )
                    sys.stdout.flush()
                difference = float(np.linalg.norm(pair["julia"] - pair["numba_cpu"]))
                relative = difference / float(np.linalg.norm(pair["numba_cpu"]))
                records[-1]["pair_relative_difference"] = relative
                # Normwise rounding allowance: 61 sources, four transforms each,
                # at most n_to*n_nodes additions in each contraction. This is an
                # intentionally conservative screening bound; pytest tests accuracy.
                ops = 61 * 4 * record["fine_samples"] * record["nodes"]
                budget = ops * np.finfo(float).eps / (1 - ops * np.finfo(float).eps)
                if relative > budget:
                    raise RuntimeError(f"paired numerical screening failed: {relative}")
    return {
        "machine": platform.platform(),
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(False),
        "memory_bytes": psutil.virtual_memory().total,
        "python": platform.python_version(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
        "window": "10 prescribed 10 um steps; propagation only; FSAL enabled",
        "records": records,
    }


def main() -> None:
    """Parse benchmark controls and write raw JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker", choices=("numpy_ref", "numba_cpu", "native", "julia")
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--propagator-mode", choices=("auto", "tabulate", "recompute"), default="auto"
    )
    parser.add_argument("--modes", type=int, default=4)
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--mode-averaged", action="store_true")
    geometry.add_argument("--radial", action="store_true")
    geometry.add_argument("--cartesian", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--widths", nargs="+", type=int)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path(".benchmarks/julia"))
    args = parser.parse_args()
    record = worker(args) if args.worker else paired(args)
    output = args.output if args.worker else args.output / "results.json"
    output.write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
