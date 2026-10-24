# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare experimental CPU arms through SEP 30's lightweight A/B worker."""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from benchmarks.ab import launch_worker
from benchmarks.collection import freeze_harness
from benchmarks.records import atomic_record
from benchmarks.registry import CASES
from benchmarks.system import eligible_cpus, thread_environment
from benchmarks.telemetry import allocation_details, hardware_details
from compare import set_fft


def field_difference(left: Path, right: Path) -> float:
    """Compare retained fields in bounded chunks, including the large beam case."""
    reference = np.load(left, mmap_mode="r").reshape(-1)
    candidate = np.load(right, mmap_mode="r").reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("incompatible retained fields")
    delta = scale = 0.0
    for first in range(0, reference.size, 65536):
        block = reference[first : first + 65536]
        difference = candidate[first : first + 65536] - block
        delta += float(np.vdot(difference, difference).real)
        scale += float(np.vdot(block, block).real)
    return (delta / scale) ** 0.5 if scale else delta**0.5


def arguments() -> argparse.Namespace:
    """Read a deliberately small selection and independently prepared CPU arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, help="case:track")
    parser.add_argument("--widths", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compare_fields(args: argparse.Namespace, arms: list, selections: list) -> None:
    """Save numerical differences separately from measured propagation work."""
    differences = []
    for width in args.widths:
        for case, track in selections:
            name = f"{case}--{track}.npy"
            reference = args.output / f"r0-t{width}-{arms[0]['name']}" / "fields" / name
            for arm in arms[1:]:
                candidate = args.output / f"r0-t{width}-{arm['name']}" / "fields" / name
                if reference.exists() and candidate.exists():
                    differences.append(
                        {
                            "id": name,
                            "arm": arm["name"],
                            "width": width,
                            "relative_l2": field_difference(reference, candidate),
                        }
                    )
    atomic_record(
        args.output / "differences.json",
        {"reference_arm": arms[0]["name"], "differences": differences},
    )


def main() -> None:
    """Alternate arms, constructing each case once per worker and retaining work."""
    args = arguments()
    root = Path(__file__).resolve().parents[1]
    arms = json.loads(args.arms.read_text(encoding="utf-8"))
    if args.output.exists():
        raise ValueError("choose a new output directory")
    args.output.mkdir(parents=True)
    harness = (args.output / "harness").resolve()
    frozen = freeze_harness(harness)
    selections = [item.split(":", 1) for item in args.case]
    atomic_record(
        args.output / "plan.json",
        {
            "arms": arms,
            "selections": selections,
            "widths": args.widths,
            "rounds": args.rounds,
            "repeats": args.repeats,
            "harness_files": frozen,
            "hardware": hardware_details(eligible_cpus()),
            "allocation": allocation_details(),
            "purpose": "exploratory short-window A/B; no adoption qualification",
        },
    )
    samples = []
    active_fft = None
    try:
        for round_index in range(args.rounds):
            ordered = arms if round_index % 2 == 0 else arms[::-1]
            for width in args.widths:
                for arm in ordered:
                    label = arm["name"]
                    fft = arm.get("fft")
                    if fft is not None and fft != active_fft:
                        set_fft(fft, root)
                        active_fft = fft
                    environment = thread_environment(width, "budget", dict(os.environ))
                    # Libraries run in separate sequential operations. Each pool
                    # has a cap, rather than nesting BLAS under a Julia task loop.
                    environment["SOLPROP_BLAS_THREADS"] = str(
                        arm.get("blas_threads", width)
                    )
                    environment.update(
                        {
                            key: str(value).format(width=width)
                            for key, value in arm.get("environment", {}).items()
                        }
                    )
                    rows = [
                        {
                            "id": f"{case}--{track}",
                            "configuration": CASES[case].measurement(track)
                            | {
                                "backend": arm["backend"],
                                "threads": width,
                                "policy": "budget",
                            },
                        }
                        for case, track in selections
                    ]
                    directory = (
                        args.output / f"r{round_index}-t{width}-{label}"
                    ).resolve()
                    events = launch_worker(
                        arm.get("python", sys.executable),
                        {
                            "rows": rows,
                            "repeats": args.repeats,
                            "harness_root": str(harness),
                            "fields": str(directory / "fields")
                            if round_index == 0
                            else None,
                        },
                        environment,
                        directory,
                        args.timeout,
                    )
                    atomic_record(directory / "events.json", {"events": events})
                    received = {
                        event["id"]: event
                        for event in events
                        if event["event"] == "row"
                    }
                    for row in rows:
                        event = received.get(
                            row["id"], {"status": "failed", "error": "missing receipt"}
                        )
                        sample = {
                            "arm": label,
                            "width": width,
                            "round": round_index,
                            "id": row["id"],
                            **event,
                        }
                        samples.append(sample)
                        if event["status"] == "ok":
                            median = statistics.median(
                                rep["seconds"] for rep in event["repetitions"]
                            )
                            sys.stdout.write(
                                f"{round_index=} {width=} {label} "
                                f"{row['id']}: {median:.6f} s\n"
                            )
                        else:
                            sys.stdout.write(
                                f"FAILED {label} {row['id']}: {event.get('error')}\n"
                            )
                        sys.stdout.flush()
                    atomic_record(args.output / "samples.json", {"samples": samples})
    finally:
        if active_fft is not None and active_fft != "fftw":
            set_fft("fftw", root)
    compare_fields(args, arms, selections)
    if any(sample["status"] != "ok" or not sample["finite"] for sample in samples):
        raise RuntimeError("one or more A/B rows failed; inspect the preserved logs")


if __name__ == "__main__":
    main()
