# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Interleave isolated M2 or CtC runs across prebuilt backend variants."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast


class Variant(TypedDict):
    """One backend label and optional import-package root."""

    label: str
    package_root: str | None


class Result(TypedDict):
    """One worker result augmented with sweep coordinates."""

    variant: str
    repetition: int
    ordinal: int
    seconds: float
    load_start: float
    load_end: float


def _variant(value: str) -> Variant:
    if value == "numba_cpu":
        return {"label": value, "package_root": None}
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError(
            "a Kokkos variant must be LABEL=/path/to/package-root"
        )
    path = Path(raw_path).expanduser().resolve()
    if not (path / "solprop_native").is_dir():
        raise argparse.ArgumentTypeError(
            f"{path} does not contain a solprop_native package directory"
        )
    return {"label": label, "package_root": str(path)}


def _run(
    variant: Variant,
    threads: int,
    blas_threads: int,
    workload: str,
    case_name: str,
    field_family: str,
    mode: str,
    trial_count: int,
    node_chunk: int | None,
) -> dict[str, object]:
    environment = os.environ.copy()
    environment.update(
        {
            "KMP_BLOCKTIME": environment.get("KMP_BLOCKTIME", "50"),
            "NUMBA_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(blas_threads),
            "OMP_DYNAMIC": environment.get("OMP_DYNAMIC", "FALSE"),
            "OMP_PLACES": environment.get("OMP_PLACES", "threads"),
            "OMP_PROC_BIND": environment.get("OMP_PROC_BIND", "close"),
            "SOLPROP_BLAS_THREADS": str(blas_threads),
            "SOLPROP_FFTW_THREADS": str(threads),
            "SOLPROP_NATIVE_THREADS": str(threads),
            "SOLPROP_NUMBA_THREADS": str(threads),
        }
    )
    backend = "numba_cpu"
    package_root = variant["package_root"]
    if package_root is not None:
        backend = "native"
        previous = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            package_root if not previous else package_root + os.pathsep + previous
        )
    command = [sys.executable, str(Path(__file__).with_name(f"{workload}.py"))]
    command.extend(("--worker", backend, "--case", case_name))
    if workload == "m2":
        command.extend(("--field-family", field_family))
    command.extend(("--mode", mode, "--trial-count", str(trial_count)))
    if workload == "ctc":
        if node_chunk is not None:
            command.extend(("--node-chunk", str(node_chunk)))
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, env=environment
    )
    return cast(dict[str, object], json.loads(completed.stdout.splitlines()[-1]))


def main() -> None:
    """Run a round-robin provider sweep and emit samples plus robust summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        action="append",
        type=_variant,
        required=True,
        help="numba_cpu or LABEL=/path/to/package-root; repeat for each arm",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--workload", choices=("m2", "ctc"), default="m2")
    parser.add_argument("--case", default="m2_reduced")
    parser.add_argument(
        "--field-family", choices=("field_real", "envelope"), default="field_real"
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="process-start BLAS pool, independent of the Kokkos/Numba pool",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--mode", choices=("trial", "propagation"), default="trial")
    parser.add_argument("--trial-count", type=int, default=7)
    parser.add_argument("--node-chunk", type=int)
    parser.add_argument(
        "--max-load",
        type=float,
        default=None,
        help="fail if either one-minute load sample exceeds this value",
    )
    arguments = parser.parse_args()
    if (
        arguments.threads < 1
        or arguments.blas_threads < 1
        or arguments.repeats < 1
        or arguments.trial_count < 1
    ):
        parser.error("thread counts and repeats must be positive")
    if not arguments.case.startswith(f"{arguments.workload}_"):
        parser.error("--case must belong to the selected workload")
    if arguments.node_chunk is not None and arguments.node_chunk < 1:
        parser.error("--node-chunk must be positive")

    samples: dict[str, list[Result]] = {
        variant["label"]: [] for variant in arguments.variant
    }
    ordinal = 0
    for repetition in range(arguments.repeats):
        offset = repetition % len(arguments.variant)
        ordered = arguments.variant[offset:] + arguments.variant[:offset]
        for variant in ordered:
            measured = _run(
                variant,
                arguments.threads,
                arguments.blas_threads,
                arguments.workload,
                arguments.case,
                arguments.field_family,
                arguments.mode,
                arguments.trial_count,
                arguments.node_chunk,
            )
            sample: Result = {
                **cast(Result, measured),
                "variant": variant["label"],
                "repetition": repetition,
                "ordinal": ordinal,
            }
            ordinal += 1
            if arguments.max_load is not None and max(
                sample["load_start"], sample["load_end"]
            ) > cast(float, arguments.max_load):
                raise RuntimeError(
                    f"load exceeded {arguments.max_load}: {json.dumps(sample)}"
                )
            samples[variant["label"]].append(sample)
            print(json.dumps({"kind": "sample", **sample}, sort_keys=True))

    for label, values in samples.items():
        times = [value["seconds"] for value in values]
        median = statistics.median(times)
        print(
            json.dumps(
                {
                    "kind": "summary",
                    "variant": label,
                    "repeats": len(times),
                    "median_seconds": median,
                    "median_absolute_deviation": statistics.median(
                        abs(value - median) for value in times
                    ),
                    "minimum_seconds": min(times),
                    "maximum_seconds": max(times),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
