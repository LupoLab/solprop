# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Profile the Julia engine on one row of a benchmark plan.

``python julia/profile.py PLAN.json ROW [FRACTION]`` builds the row's case for the
julia backend, warms it up, then samples Julia's stacks over a propagation to
``FRACTION`` of the row's stop (default 1) and prints the flat profile. Run it with
``SOLPROP_JULIA_THREADS=1``: Julia's sampling profiler is unreliable when Python
owns the threads.
"""

import json
import sys
import time

from benchmarks.builders import build
from solprop.backends import select


def main() -> None:
    """Profile one plan row."""
    plan = json.loads(open(sys.argv[1], encoding="utf-8").read())  # noqa: PTH123, SIM115
    row = next(r for r in plan["rows"] if sys.argv[2] in r["id"])
    configuration = row["configuration"] | {"backend": "julia"}
    fraction = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0  # noqa: PLR2004
    simulation = build(configuration)
    problem = simulation.translate()
    engine = select(
        "julia",
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    stop = fraction * float(configuration["stop"])
    engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
    engine.propagate_to(0.5 * stop)
    from juliacall import Main as jl  # noqa: PLC0415 - after the runtime starts
    jl.seval("using Profile; Profile.init(n=10^7, delay=0.0005)")
    engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
    jl.seval("Profile.clear(); Profile.start_timer()")
    started = time.perf_counter()
    engine.propagate_to(stop)
    seconds = time.perf_counter() - started
    jl.seval("Profile.stop_timer()")
    jl.seval(
        "Profile.print(IOContext(stdout, :displaysize => (200, 220)); format=:flat, "
        "sortedby=:count, mincount=20, noisefloor=1)"
    )
    sys.stdout.write(f"\n{row['id']}: {seconds:.3f} s to {stop:.3g} m\n")


if __name__ == "__main__":
    main()
