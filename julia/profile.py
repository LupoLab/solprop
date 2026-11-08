# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Profile the Julia engine on rows of a benchmark plan, as the harness times them.

``python julia/profile.py PLAN.json ROW...`` builds each matching row's case for the
julia backend and times a complete run (``benchmarks.worker.timed_run``, observations
and the boundary monitor included) after a first, warming one. With
``SOLPROP_JULIA_TIMING=1`` it prints the engine's section timers, synchronised with the
device on a GPU; with ``SOLPROP_JULIA_PROFILE=1`` Julia's sampling profile instead,
which needs ``SOLPROP_JULIA_THREADS=1`` (it crashes when Python owns more threads).
"""

import json
import os
import sys

from benchmarks.builders import build
from benchmarks.worker import timed_run

from solprop.backends import select
from solprop.driver.run import field_retention


def profile(row: dict) -> None:
    """Time one row's run twice and report the second."""
    configuration = row["configuration"] | {"backend": "julia"}
    simulation = build(configuration)
    problem = simulation.translate()
    engine = select(
        "julia",
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    retention = field_retention(
        problem, simulation.settings.fields, simulation.settings.field_planes
    )
    timed_run(simulation, problem, engine, retention, simulation._boundary_monitor())
    jl = engine._jl
    sampling = os.environ.get("SOLPROP_JULIA_PROFILE") == "1"
    if sampling:
        from juliacall import Main  # noqa: PLC0415 - after the runtime starts

        Main.seval("using Profile; Profile.init(n=10^7, delay=0.0005)")
        Main.seval("Profile.clear(); Profile.start_timer()")
    jl.clear_timings()
    jl.clear_counts()
    jl.clear_kernel_timings()
    # SOLPROP_JULIA_ALLOCS=1: the host's allocations by where they are made (sampled;
    # Profile.Allocs uses no signals, which the sampling profile's kill a CUDA process).
    allocs = os.environ.get("SOLPROP_JULIA_ALLOCS") == "1"
    if allocs:
        from juliacall import Main  # noqa: PLC0415 - after the runtime starts

        Main.seval("using Profile; Profile.Allocs.clear(); Profile.Allocs.start(sample_rate=0.002)")
    # SOLPROP_PYTHON_PROFILE=1: the host's Python time too (cProfile, by own time).
    python = os.environ.get("SOLPROP_PYTHON_PROFILE") == "1"
    if python:
        # (This script's directory shadows the standard library's `profile`.)
        here = os.path.dirname(os.path.abspath(__file__))  # noqa: PTH100, PTH120
        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != here]  # noqa: PTH100
        import cProfile  # noqa: PLC0415
        import pstats  # noqa: PLC0415

        profiler = cProfile.Profile()
        profiler.enable()
    record, _ = timed_run(
        simulation, problem, engine, retention, simulation._boundary_monitor()
    )
    if python:
        profiler.disable()
        pstats.Stats(profiler, stream=sys.stdout).sort_stats("cumulative").print_stats("julia/_|integrators|_march", 30)
    if sampling:
        Main.seval("Profile.stop_timer()")
        Main.seval(
            "Profile.print(IOContext(stdout, :displaysize => (200, 220)); "
            "format=:flat, sortedby=:count, mincount=20, noisefloor=1)"
        )
    if allocs:
        Main.seval("Profile.Allocs.stop()")
        sys.stdout.write(str(Main.seval(r"""
        let r = Profile.Allocs.fetch(), sites = Dict{String,Tuple{Int,Int}}()
            for a in r.allocs
                # The first frame in SolpropJulia, and the frame below it.
                i = findfirst(f -> occursin("SolpropJulia", string(f.file)) ||
                                   occursin("/julia/src/", string(f.file)), a.stacktrace)
                i === nothing && continue
                f = a.stacktrace[i]
                below = a.stacktrace[max(i - 1, 1)]
                key = string(f.func, " ", basename(string(f.file)), ":", f.line, " <- ",
                             below.func)
                n, b = get(sites, key, (0, 0))
                sites[key] = (n + 1, b + a.size)
            end
            total = length(r.allocs)
            lines = ["sampled allocations: $total (x500 for the run)"]
            for (k, (n, b)) in sort(collect(sites); by=x -> -x[2][1])[1:min(30, end)]
                push!(lines, lpad(n, 7) * lpad(b, 11) * "  " * k)
            end
            join(lines, "\n")
        end""")) + "\n")
    times = jl.timings()
    total = sum(float(times[name]) for name in times)
    sys.stdout.write(
        f"{row['id']}: {record['seconds']:.3f} s, {record['accepted']} steps, "
        f"sections {total:.3f} s\n"
    )
    for name in sorted(times, key=lambda n: -float(times[n])):
        sys.stdout.write(f"  {name:>18} {float(times[name]):8.4f} s\n")
    # SOLPROP_JULIA_COUNT=1: the device's launches, waits and allocations a step.
    counts = {str(k): int(v) for k, v in jl.counts().items()}
    steps = max(int(record["accepted"]), 1)
    for name in sorted(counts):
        sys.stdout.write(f"  {name:>18} {counts[name] / steps:8.1f} a step ({counts[name]})\n")
    # SOLPROP_JULIA_KERNEL_TIMING=1: each GPU pass alone, by its body.
    kernels = {str(k): list(v) for k, v in jl.kernel_timings().items()}
    for name in sorted(kernels, key=lambda n: -float(kernels[n][1]))[:25]:
        count, total, size = kernels[name]
        sys.stdout.write(
            f"  {float(total) * 1e3:8.2f} ms {int(count):6d} x {int(size):9d}  {name}\n"
        )
    engine.close()


def main() -> None:
    """Profile every row of the plan whose id contains one of the arguments."""
    plan = json.loads(open(sys.argv[1], encoding="utf-8").read())  # noqa: PTH123, SIM115
    for row in plan["rows"]:
        if row["configuration"]["backend"] == "native" and any(
            name in row["id"] for name in sys.argv[2:]
        ):
            profile(row)


if __name__ == "__main__":
    main()
