# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare the native execution session with the driver session, in process.

SEP 47 section 7.4's readiness evidence for the phase 1 prototype
(:class:`solprop_native.session.NativeSession`): a ``sep47`` case is built as the
worker builds it (``benchmarks.builders.build``) and run to its end, alternately, by
the driver session over the native backend
(:class:`~solprop.backends.driver_session.DriverSession`) and by the native session,
each through the same minimal service loop -- advance, keep each observation's
statistics and rows, export a kept plane's field once, acknowledge -- so that the
difference is the sessions' own. The production run loop
(:func:`solprop.driver.run.run_problem`) is not used: it serves the driver session
only, and the native session is an explicit opt-in here and in the tests.

Each run reports its wall time, the time inside the native calls it makes (the march
calls and exports of the driver's backend, the native session's ``advance`` and
``export``) and the Python time outside them, its returns to Python by cause, the
field bytes it exported, the bytes through the state's crossings each way (a device
build's transfers) and its peak traced Python memory. On a device build both sessions
run on the device (SEP 47 phase 3's representative path). Run it pinned, once per
thread setting (threads are read at import)::

    SOLPROP_THREADS=1 taskset -c CPUS python -m benchmarks.native_session \\
        stats_period1_rk --repeats 7

Development evidence only; not part of the tracked suite.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from collections import Counter
from typing import Any

__all__ = ["main", "measure"]

_TIMED = frozenset(
    {
        "adams_march_run",
        "rk_march_run",
        "statistics",
        "spatial_statistics",
        "field_to_host",
        "advance",
        "export",
    }
)
"""The native calls whose time counts as native: the marches, the statistics and the
field crossings of the driver's backend, and the native session's own calls."""


class _Timed:
    """Forward every attribute of a native object, timing the calls in :data:`_TIMED`.

    Parameters
    ----------
    target : object
        The native object.
    """

    def __init__(self, target: object) -> None:
        """Wrap `target`."""
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "seconds", 0.0)
        object.__setattr__(self, "calls", Counter[str]())

    def __getattr__(self, name: str) -> Any:
        """Return the target's attribute, timed when it is a native call of interest."""
        value = getattr(object.__getattribute__(self, "_target"), name)
        if name not in _TIMED or not callable(value):
            return value

        def timed(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return value(*args, **kwargs)
            finally:
                object.__setattr__(
                    self, "seconds", self.seconds + time.perf_counter() - started
                )
                self.calls[name] += 1

        return timed


def _prepare(name: str, fraction: float, *, admitted: bool) -> tuple[Any, Any, Any]:
    """Build and translate a case as the worker does; return it and its request.

    With `admitted` the case runs noiseless and keeps only the statistics the native
    session admits, for both executors alike: a narrower workload than the tracked
    case, for a case whose noise model or statistics the prototype refuses.
    """
    from benchmarks.builders import build  # noqa: PLC0415
    from benchmarks.registry import CASES  # noqa: PLC0415
    from solprop.driver.run import field_retention  # noqa: PLC0415
    from solprop.stats import StatsSpec  # noqa: PLC0415
    from solprop.translate.run_request import describe_run  # noqa: PLC0415

    configuration = CASES[name].configuration(
        "default", 1, fraction=fraction, noise="none" if admitted else "case"
    )
    simulation = build(configuration)
    problem = simulation.translate()
    retention = field_retention(
        problem, simulation.settings.fields, simulation.settings.field_planes
    )
    stats = None
    if admitted:
        from solprop_native.session import (  # noqa: PLC0415
            ADMITTED_SPATIAL_STATISTICS,
            ADMITTED_STATISTICS,
        )

        spec = problem.stats_spec
        allowed = (
            ADMITTED_STATISTICS
            if problem.transform.geometry is None
            else (*ADMITTED_SPATIAL_STATISTICS, "on_axis_spectrum")
            if problem.transform.geometry.kind != "free"
            else ADMITTED_SPATIAL_STATISTICS
        )
        names = tuple(name for name in spec.names if name in allowed)
        periodic = tuple(name for name in spec.periodic if name in names)
        stats = StatsSpec(
            names=names,
            periodic=periodic,
            period=spec.period,
            energy_windows=spec.energy_windows,
        )
    return simulation, problem, describe_run(problem, retention=retention, stats=stats)


def _start(
    executor: str, simulation: Any, problem: Any, request: Any
) -> tuple[Any, Any]:
    """Start one session of `executor`; return it and its timed native object."""
    from solprop.backends.session import Start  # noqa: PLC0415

    start = Start(field=problem.initial.field0[0], z=float(problem.initial.z))
    if executor == "native-session":
        from solprop_native.session import NativeSession  # noqa: PLC0415

        session: Any = NativeSession(
            problem,
            request,
            start,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
            options=simulation.settings.execution_options(),
        )
        timed = _Timed(session._native)
        session._native = timed
        return session, timed
    from solprop.backends.driver_session import DriverSession  # noqa: PLC0415
    from solprop.backends.registry import select  # noqa: PLC0415

    backend: Any = select(
        "native",
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
        options=simulation.settings.execution_options(),
    )
    timed = _Timed(backend._native_state)
    backend._native_state = timed
    return DriverSession(backend, problem, request, start), timed


def _serve(session: Any, request: Any) -> tuple[int, Counter[str], int]:
    """Drive a session to its end as a minimal service loop; count what it returned."""
    from solprop.backends.events import (  # noqa: PLC0415
        ObservationReady,
        Progress,
        ServiceRequired,
        Terminal,
    )
    from solprop.backends.export import ExportRecipe  # noqa: PLC0415

    consumer = ExportRecipe(power=True, optics=False)
    retention = request.observation.retention
    kept: list[object] = []
    kinds: Counter[str] = Counter()
    saved = 0
    batches = 0
    while not session.finished:
        batch = session.advance()
        batches += 1
        for event in batch.events:
            if isinstance(event, ObservationReady):
                kinds[f"observation:{event.kind.value}"] += 1
                field = (
                    session.export(event.ticket, consumer)
                    if retention.keeps(saved)
                    else None
                )
                kept.append((event.stats, event.step_stats, event.history, field))
                saved += 1
            elif isinstance(event, Progress):
                kinds["progress"] += 1
            elif isinstance(event, ServiceRequired):
                kinds["service"] += 1
            elif isinstance(event, Terminal):
                kinds[f"terminal:{event.outcome.value}"] += 1
        session.acknowledge(batch.receipt)
    return batches, kinds, saved


def _once(
    executor: str, simulation: Any, problem: Any, request: Any, *, count: bool = False
) -> dict[str, Any]:
    """Run one session to its end; return its timings and, with `count`, its counts.

    The driver's structural counters hook every attempt, so a counted run is not one
    of the timed ones; both sessions are counted the same way
    (:mod:`solprop_native._counters`).
    """
    from solprop_native import _counters  # noqa: PLC0415

    session, timed = _start(executor, simulation, problem, request)
    counters = _counters.enable() if count else None
    try:
        started = time.perf_counter()
        batches, kinds, saved = _serve(session, request)
        seconds = time.perf_counter() - started
    finally:
        if counters is not None:
            _counters.disable()
    # The field bytes through the state's crossings, each way (SEP 44), on either
    # session: a device's plane exports and loads, or a host build's copies.
    state = session._backend._native_state if executor == "native-session" else timed
    to_host, from_host = state.crossed_bytes()
    record: dict[str, Any] = {
        "crossed_to_host": int(to_host),
        "crossed_from_host": int(from_host),
        "seconds": seconds,
        "native_seconds": timed.seconds,
        "python_seconds": seconds - timed.seconds,
        "batches": batches,
        "events": dict(kinds),
        "planes": saved,
        "attempts": session.status().attempts,
        "native_calls": dict(timed.calls),
    }
    if counters is not None:
        counted = counters.record()
        record["native_returns"] = counted["native_returns"]
        record["march_calls"] = counted["march_calls"]
        record["field_bytes"] = counted["field_bytes"]
        record["statistics_rows"] = counted["statistics_rows"]
    if count and executor == "native-session":
        record["buffer_bytes"] = session.native.status().buffer_bytes
    session.close()
    return record


def _traced(executor: str, simulation: Any, problem: Any, request: Any) -> int:
    """Run once under tracemalloc; return the peak traced Python memory, bytes."""
    session, _ = _start(executor, simulation, problem, request)
    tracemalloc.start()
    try:
        _serve(session, request)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        session.close()
    return peak


def measure(
    name: str, repeats: int, fraction: float = 1.0, *, admitted: bool = False
) -> dict[str, Any]:
    """Time `repeats` interleaved pairs of a case; return medians and the last counts.

    Parameters
    ----------
    name : str
        A ``sep47`` case.
    repeats : int
        Interleaved pairs after one warm-up of each, alternating which runs first.
    fraction : float, optional
        The fraction of the case's length to run.
    admitted : bool, optional
        Run the case noiseless with only the statistics the native session admits.

    Returns
    -------
    dict
        Per executor: the median, minimum and spread of the wall, native and Python
        seconds, and the last run's counts.
    """
    simulation, problem, request = _prepare(name, fraction, admitted=admitted)
    executors = ("driver", "native-session")
    for executor in executors:
        _once(executor, simulation, problem, request)
    runs: dict[str, list[dict[str, Any]]] = {executor: [] for executor in executors}
    for index in range(repeats):
        order = executors if index % 2 == 0 else executors[::-1]
        for executor in order:
            runs[executor].append(_once(executor, simulation, problem, request))
    from solprop_native import build_info  # noqa: PLC0415

    result: dict[str, Any] = {
        "case": name,
        "device_backend": build_info().device_backend,
        "fraction": fraction,
        "repeats": repeats,
        "admitted": admitted,
        "shape": list(problem.shape),
        "statistics": list(request.observation.stats.names),
    }
    for executor, records in runs.items():
        summary: dict[str, Any] = {}
        for key in ("seconds", "native_seconds", "python_seconds"):
            values = [record[key] for record in records]
            summary[key] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        last = _once(executor, simulation, problem, request, count=True)
        summary.update(
            {key: value for key, value in last.items() if not key.endswith("seconds")}
        )
        summary["peak_traced_bytes"] = _traced(executor, simulation, problem, request)
        result[executor] = summary
    return result


def main(argv: list[str] | None = None) -> int:
    """Measure the named cases and print one JSON record each."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument(
        "--admitted",
        action="store_true",
        help="run noiseless with only the statistics the native session admits",
    )
    args = parser.parse_args(argv)
    for name in args.cases:
        record = measure(name, args.repeats, args.fraction, admitted=args.admitted)
        print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
