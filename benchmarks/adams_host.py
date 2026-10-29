# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Time an exponential Adams run's host side per trial (SEP 39 section 7.6).

The host time of a trial is the Adams march's wall time less the native ``adams_*``
calls it makes, measured with those two wrappers alone: per-piece wrappers on every
host function inflate the total by about half, so they are for the split between
pieces, not for the sum. The run is one benchmark case, built, warmed up and timed by
the suite's own worker (:func:`benchmarks.worker.timed_run`) in a fresh process under
the suite's thread pools.

Usage::

    uv run python -m benchmarks.adams_host CASE [--width N] [--length F] [--stepper S]
        [--host-stages] [--host-march]

It prints one JSON record: the wall time, the accepted and rejected steps, the source
calls, and the host and native time per trial in microseconds. The native backend is
required; ``--stepper rk`` times a Runge-Kutta run, which has no host figure.
``--host-stages`` has the host form a trial's stage rows and tables, as before SEP 39's
phase 1, and ``--host-march`` steps an admitted run with solprop's Python march
instead of the native one (phase 2), so that each phase is compared with its own
build.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

from benchmarks.system import thread_environment

__all__ = ["HostTotals", "install", "main"]

CHILD = "_SOLPROP_ADAMS_HOST_CHILD"
"""Set in the fresh process the command runs the case in."""


class HostTotals:
    """The march's wall time and its native Adams calls' time, in seconds."""

    def __init__(self) -> None:
        self.march = 0.0
        self.native = 0.0


def install() -> HostTotals:
    """Wrap the Adams march and the resident engine's native calls with two timers.

    Returns
    -------
    HostTotals
        The totals the wrappers add to.
    """
    # Optional, and imported after the pools are set.
    import solprop_native.backend  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    backend = solprop_native.backend

    totals = HostTotals()

    def timed(method: Any, key: str) -> Any:
        def wrapper(*args: Any, **keywords: Any) -> Any:
            started = time.perf_counter()
            try:
                return method(*args, **keywords)
            finally:
                setattr(
                    totals, key, getattr(totals, key) + time.perf_counter() - started
                )

        return wrapper

    # Every native run's march is the native backend's (SEP 39), which runs the
    # Python march itself when it is not admitted; the native march's one call a host
    # event is its native time.
    native_march = backend._NativeAdamsMarch
    native_march.march_to = timed(native_march.march_to, "march")  # pyright: ignore[reportAttributeAccessIssue]
    native_march._native_run = timed(native_march._native_run, "native")  # pyright: ignore[reportAttributeAccessIssue]

    class Native:
        """The native state, its ``adams_*`` calls timed and the rest passed on."""

        def __init__(self, native: Any) -> None:
            object.__setattr__(self, "_native", native)

        def __getattr__(self, name: str) -> Any:
            value = getattr(self._native, name)
            if not name.startswith("adams_"):
                return value

            def timed(*args: Any, **keywords: Any) -> Any:
                started = time.perf_counter()
                try:
                    return value(*args, **keywords)
                finally:
                    totals.native += time.perf_counter() - started

            return timed

    initialise = backend._ResidentAdams.__post_init__

    def post_init(self: Any) -> None:
        initialise(self)
        object.__setattr__(self, "native", Native(self.native))

    backend._ResidentAdams.__post_init__ = post_init
    return totals


def _run(
    case: str,
    stepper: str,
    width: int,
    fraction: float,
    *,
    host_stages: bool,
    host_march: bool,
) -> dict[str, Any]:
    """Run the case in this process and return its record."""
    # Before pyFFTW on macOS.
    import solprop_native  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    import solprop_native.backend  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    from benchmarks.builders import build  # noqa: PLC0415
    from benchmarks.registry import CASES  # noqa: PLC0415
    from benchmarks.worker import timed_run  # noqa: PLC0415
    from solprop.backends import select  # noqa: PLC0415
    from solprop.driver.run import field_retention  # noqa: PLC0415

    totals = install()
    if host_march:
        solprop_native.backend._ADAMS_MARCH = "host"
    if host_stages:
        solprop_native.backend.NativeBackend._stage_tables = lambda _self: None  # pyright: ignore[reportAttributeAccessIssue]
    configuration = CASES[case].configuration(stepper, width, "native", fraction)
    simulation = build(configuration)
    problem = simulation.translate()
    engine = select(
        "native",
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    try:
        retention = field_retention(
            problem, simulation.settings.fields, simulation.settings.field_planes
        )
        engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
        engine.propagate_to(configuration["warmup"] * configuration["stop"])
        totals.march = totals.native = 0.0
        record, _ = timed_run(simulation, problem, engine, retention)
    finally:
        engine.close()
    trials = record["accepted"] + record["rejected"]
    record.update(
        case=case,
        stepper=stepper,
        width=width,
        fraction=fraction,
        host_stages=host_stages,
        host_march=host_march,
    )
    if totals.march > 0.0:
        record["host_us_per_trial"] = 1e6 * (totals.march - totals.native) / trials
        record["native_us_per_trial"] = 1e6 * totals.native / trials
    return record


def main(argv: list[str] | None = None) -> int:
    """Run one case in a fresh process and print its record."""
    parser = argparse.ArgumentParser(description="SEP 39: an Adams run's host time")
    parser.add_argument("case")
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--stepper", default="default")
    parser.add_argument("--host-stages", action="store_true")
    parser.add_argument("--host-march", action="store_true")
    args = parser.parse_args(argv)
    if os.environ.get(CHILD) == "1":
        record = _run(
            args.case,
            args.stepper,
            args.width,
            args.length,
            host_stages=args.host_stages,
            host_march=args.host_march,
        )
        print(json.dumps(record))
        return 0
    environment = thread_environment(args.width, dict(os.environ))
    environment[CHILD] = "1"
    command = [sys.executable, "-m", "benchmarks.adams_host", *sys.argv[1:]]
    if argv is not None:
        command = [sys.executable, "-m", "benchmarks.adams_host", *argv]
    return subprocess.call(command, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
