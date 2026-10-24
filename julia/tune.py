# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Alternate short windows within one process to reduce drift on a busy machine."""

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from benchmark import radial_simulation, simulation

from solprop.backends import select
from solprop.backends.julia._backend import _configuration


# JuliaCall is dynamic; keep Any at its bridge boundary.
def set_blas_provider(julia: Any, name: str) -> None:
    """Switch a provider outside the timed window; keep BLAS work unnested."""
    if name == "accelerate":
        julia.seval(
            "AppleAccelerate.load_accelerate(;clear=true); "
            "AppleAccelerate.set_num_threads(1)"
        )
    elif name == "mkl":
        julia.seval("MKL.lbt_forward_to_mkl()")
    elif name != "numba_cpu":
        julia.seval("BLAS.lbt_forward(tuning_openblas;clear=true)")
    width = int(os.environ.get("SOLPROP_BLAS_THREADS", "1"))
    julia.seval(f"BLAS.set_num_threads({width})")


def arguments() -> argparse.Namespace:
    """Parse the small experimental benchmark interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cartesian", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--numba-tabulate", action="store_true")
    parser.add_argument("--compare-blas", choices=("accelerate", "mkl"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.compare_blas:
        os.environ["SOLPROP_JULIA_BLAS"] = "openblas"
    return args


def write_result(
    output: Path, result: dict[str, object], timings: dict[str, list[float]]
) -> None:
    """Write raw evidence and display each median with its absolute deviation."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, values in timings.items():
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        print(f"{name}: {median * 1000:.3f} ± {mad * 1000:.3f} ms")  # noqa: T201 - benchmark CLI


# JuliaCall is dynamic; the saved module shares the experimental bridge boundary.
def load_baseline(path: Path) -> Any:
    """Load a saved Julia module with its own companion numerical sources."""
    import juliacall  # noqa: PLC0415 - private runtime initialized by the caller

    code = path.read_text(encoding="utf-8").replace(
        "module SolpropJulia\n", "module SolpropBaseline\n", 1
    )
    code = code.replace(
        'include("free.jl")',
        f"include({json.dumps(str(path.resolve().with_name('free.jl')))})",
    )
    return juliacall.Main.seval(code)


def main() -> None:
    """Compare the current engine with Numba and an optional saved Julia source."""
    args = arguments()
    sim = (
        radial_simulation(args.modes, "julia", cartesian=True)
        if args.cartesian
        else simulation(args.modes, "julia")
    )
    problem = sim.translate()
    engines = {
        name: select(name, problem, grid=sim.grid, modes=sim.modes)
        for name in (
            ("julia", "numba_cpu", "native") if args.native else ("julia", "numba_cpu")
        )
    }
    if args.numba_tabulate:
        tabulated = dataclasses.replace(
            problem,
            stepper=dataclasses.replace(problem.stepper, propagator_mode="tabulate"),
        )
        engines["numba_tabulate"] = select(
            "numba_cpu", tabulated, grid=sim.grid, modes=sim.modes
        )
    if args.baseline:
        module = load_baseline(args.baseline)
        engine = select("julia", problem, grid=sim.grid, modes=sim.modes)
        engine._jl = module
        engine._engine = module.create(_configuration(problem, engine._threads))
        engines["baseline"] = engine
    if args.compare_blas:
        import juliacall  # noqa: PLC0415 - runtime initialized above

        juliacall.Main.seval("using LinearAlgebra")
        juliacall.Main.seval(
            "const tuning_openblas = first(BLAS.get_config().loaded_libs).libname"
        )
        package = {"accelerate": "AppleAccelerate", "mkl": "MKL"}[args.compare_blas]
        juliacall.Main.seval(f"using {package}")
        engines[args.compare_blas] = select(
            "julia", problem, grid=sim.grid, modes=sim.modes
        )
    timings = {name: [] for name in engines}
    metadata, fields = {}, {}
    try:
        for round_index in range(args.repeats + 1):
            order = tuple(engines)
            if round_index % 2:
                order = order[::-1]
            for name in order:
                engine = engines[name]
                if args.compare_blas:
                    set_blas_provider(juliacall.Main, name)
                engine.set_initial(problem.initial.field0, None)
                start = time.perf_counter()
                engine.propagate_to(1e-4)
                elapsed = time.perf_counter() - start
                if round_index:
                    timings[name].append(elapsed)
                fields[name] = engine.observe().field
                metadata[name] = engine.metadata()
        differences = {
            name: float(np.linalg.norm(field - fields["numba_cpu"]))
            / float(np.linalg.norm(fields["numba_cpu"]))
            for name, field in fields.items()
        }
        # Same conservative normwise rounding screen as the separate-process runner.
        ops = 61 * 4 * problem.grids.n_to * problem.transform.n_nodes
        budget = ops * np.finfo(float).eps / (1 - ops * np.finfo(float).eps)
        if any(not np.isfinite(v) or v > budget for v in differences.values()):
            raise RuntimeError(f"numerical screening failed: {differences}")
        result = {
            "modes": args.modes,
            "cartesian": args.cartesian,
            "baseline_sha256": hashlib.sha256(
                args.baseline.read_bytes()
                + args.baseline.with_name("free.jl").read_bytes()
            ).hexdigest()
            if args.baseline
            else None,
            "source_sha256": hashlib.sha256(
                b"".join(
                    path.read_bytes()
                    for path in sorted(Path(__file__).with_name("src").glob("*.jl"))
                )
            ).hexdigest(),
            "thread_environment": {
                key: value
                for key, value in os.environ.items()
                if key.startswith(
                    ("SOLPROP_", "NUMBA_", "OMP_", "OPENBLAS_", "VECLIB_")
                )
            },
            "rounding_screen": budget,
            "timings": timings,
            "metadata": metadata,
            "relative_differences": differences,
            "load": os.getloadavg(),
        }
        write_result(args.output, result, timings)
    finally:
        for engine in engines.values():
            engine.close()


if __name__ == "__main__":
    main()
