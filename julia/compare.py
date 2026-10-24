# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare CPU backends and math providers in independent, alternating processes."""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import psutil
from benchmark import cpu_name


def set_fft(provider: str, root: Path) -> None:
    """Select the checkout-local FFT preference before starting a Julia worker."""
    env = os.environ | {
        "JULIA_DEPOT_PATH": str(root / ".julia-depot"),
        "JULIA_LOAD_PATH": "@:@stdlib",
        "JULIA_CONDAPKG_BACKEND": "Null",
    }
    subprocess.run(
        [
            str(root / ".julia-runtime/julia-1.13.2/bin/julia"),
            "--startup-file=no",
            f"--project={root / 'julia'}",
            "-e",
            f'using FFTW; FFTW.set_provider!("{provider}")',
        ],
        env=env,
        check=True,
    )


def environment(width: int, blas: int) -> dict[str, str]:
    """Give each numerical runtime the same explicit CPU budget and FFT effort."""
    env = os.environ.copy()
    for name in ("OMP_PLACES", "OMP_PROC_BIND", "SOLPROP_FFTW_WISDOM"):
        env.pop(name, None)
    for name in (
        "SOLPROP_JULIA_THREADS",
        "SOLPROP_NUMBA_THREADS",
        "NUMBA_NUM_THREADS",
        "SOLPROP_NATIVE_THREADS",
        "SOLPROP_NATIVE_KERNEL_THREADS",
        "SOLPROP_NATIVE_FFT_THREADS",
        "SOLPROP_FFTW_THREADS",
        "OMP_NUM_THREADS",
    ):
        env[name] = str(width)
    for name in (
        "SOLPROP_BLAS_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = str(blas)
    # A serial BLAS request must not silently cap the MKL FFT provider as well.
    env["MKL_NUM_THREADS"] = str(width)
    env["MKL_DOMAIN_NUM_THREADS"] = f"MKL_DOMAIN_BLAS={blas},MKL_DOMAIN_FFT={width}"
    env["OMP_PROC_BIND"] = "false"
    env["SOLPROP_FFTW_PLAN"] = "estimate"
    return env


def screen(fields: dict[str, np.ndarray], record: dict[str, object]) -> float:
    """Screen each final field against Numba with a predeclared rounding allowance."""
    reference = fields["numba_cpu"]
    # The same conservative 61-source transform/contraction screen as benchmark.py.
    # Focused NumPy-oracle checks, rather than this screen, qualify the numerics.
    ops = 61 * 4 * int(record["fine_samples"]) * int(record["nodes"])
    epsilon = np.finfo(float).eps
    budget = ops * epsilon / (1 - ops * epsilon)
    differences = []
    for name, field in fields.items():
        relative = float(np.linalg.norm(field - reference) / np.linalg.norm(reference))
        if (
            not np.isfinite(field).all()
            or not np.isfinite(relative)
            or relative > budget
        ):
            raise RuntimeError(f"{name} failed numerical screening: {relative}")
        differences.append(relative)
    return max(differences)


def arguments() -> argparse.Namespace:
    """Parse the provider, geometry and planning comparison controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["numba_cpu", "native", "julia-openblas-fftw", "julia-accelerate-fftw"],
        choices=(
            "numba_cpu",
            "native",
            "julia-openblas-fftw",
            "julia-accelerate-fftw",
            "julia-mkl-fftw",
            "julia-mkl-mkl",
        ),
    )
    parser.add_argument("--widths", nargs="+", type=int, default=[1, 4])
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        help="Override mode/node counts or Cartesian grid sides.",
    )
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--mode-averaged", action="store_true")
    geometry.add_argument("--radial", action="store_true")
    geometry.add_argument("--cartesian", action="store_true")
    parser.add_argument("--blas-threads", type=int)
    parser.add_argument(
        "--plan", choices=("estimate", "measure", "patient"), default="estimate"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "numba_cpu" not in args.arms:
        parser.error("include numba_cpu for matched final-field screening")
    return args


def main() -> None:
    """Run the requested provider matrix and preserve raw timings and witnesses."""
    args = arguments()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    records, differences = [], []
    active_fft = None
    try:
        for round_index in range(args.rounds):
            arms = args.arms if round_index % 2 == 0 else args.arms[::-1]
            mode_counts = args.sizes or (
                (32, 64)
                if args.cartesian
                else (64, 256)
                if args.radial
                else (1,)
                if args.mode_averaged
                else (4, 16)
            )
            cases = [(modes, width) for modes in mode_counts for width in args.widths]
            if round_index % 2:
                cases.reverse()
            for modes, width in cases:
                fields = {}
                for arm in arms:
                    env = environment(width, args.blas_threads or width)
                    env["SOLPROP_FFTW_PLAN"] = args.plan
                    backend = arm
                    if arm.startswith("julia-"):
                        backend, blas, fft = arm.split("-")
                        env["SOLPROP_JULIA_BLAS"] = blas
                        if fft != active_fft:
                            set_fft(fft, root)
                            active_fft = fft
                    output = (
                        args.output / f"m{modes}-t{width}-r{round_index}-{arm}.json"
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            str(root / "julia/benchmark.py"),
                            "--worker",
                            backend,
                            "--modes",
                            str(modes),
                            "--threads",
                            str(width),
                            "--repeats",
                            str(args.repeats),
                            "--output",
                            str(output),
                            *(["--mode-averaged"] if args.mode_averaged else []),
                            *(["--radial"] if args.radial else []),
                            *(["--cartesian"] if args.cartesian else []),
                        ],
                        env=env,
                        check=True,
                    )
                    record = json.loads(output.read_text(encoding="utf-8"))
                    record.update(arm=arm, round=round_index)
                    records.append(record)
                    fields[arm] = np.load(output.with_suffix(".npy"))
                    median = statistics.median(record["warm_s"]) * 1000
                    print(  # noqa: T201 - benchmark CLI
                        f"round={round_index + 1} modes={modes} threads={width} "
                        f"{arm}: {median:.3f} ms",
                        flush=True,
                    )
                differences.append(screen(fields, record))
    finally:
        if active_fft and active_fft != "fftw":
            set_fft("fftw", root)
    result = {
        "machine": platform.platform(),
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(False),
        "python": platform.python_version(),
        "affinity": psutil.Process().cpu_affinity()
        if sys.platform == "linux"
        else None,
        "window": "10 prescribed 10 um steps; propagation only; warm compilation/plans",
        "records": records,
        "maximum_relative_field_difference": max(differences),
    }
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
