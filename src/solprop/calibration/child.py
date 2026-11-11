# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A calibration child: one candidate's process settings, timed on request (SEP 46).

The controller starts one child per candidate process setting (SEP 46 section 6.2),
with the setting in its environment, and drives it over JSON lines on its standard
input and output. The child pins itself to the allocation being calibrated before the
native runtime starts, builds the calibration problems, and then times their windows
as the controller asks, under the workspace record it is given or its process's own.
It never calibrates and never reads a profile (``SOLPROP_CALIBRATION=off``).

Run as ``python -m solprop.calibration.child CPUS``, ``CPUS`` a comma-separated list
of logical CPUs (empty for no pinning). Requests::

    {"op": "prepare", "problems": {"gnlse": 1.0, ...}}
    {"op": "time", "problem": "gnlse", "repeats": 40, "record": {...} | null,
     "planning": {...} | null}
    {"op": "sizing", "record": {...} | null}
    {"op": "capture", "effort": "measure", "limit": 2.0, "budget": 30.0,
     "record": {...} | null}
    {"op": "regions", "elements": 1000, "repeats": 200}
    {"op": "exit"}

Each request has one reply line, ``{"ok": ...}`` or ``{"error": "..."}``.
"""

from __future__ import annotations

import io
import json
import os
import statistics
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from solprop_native._execution import (  # pyright: ignore[reportMissingImports]
        ExecutionRecord,
    )
    from solprop_native.backend import (  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

__all__ = ["main", "pin", "serve"]


def pin(cpus: Sequence[int]) -> None:
    """Pin this process to `cpus` before the native runtime starts.

    Linux sets its affinity mask; Windows its process affinity mask (CPUs of the first
    processor group); macOS has no affinity, and is left alone.

    Parameters
    ----------
    cpus : sequence of int
        Logical CPUs; empty leaves the process as it is.
    """
    if not cpus:
        return
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    elif sys.platform == "win32":  # pragma: no cover - Windows
        import ctypes  # noqa: PLC0415 - Windows only

        mask = sum(1 << cpu for cpu in cpus if cpu < 64)
        # A private handle on the DLL, so that declaring the types touches no one
        # else's; a HANDLE and a DWORD_PTR are pointer wide.
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask)


_PLANS_PER_PROBLEM = 4
"""About how many transforms a calibration problem plans, for sharing a budget."""

_SHORTEST_LIMIT = 0.05
"""The shortest per-plan time limit worth a problem's capture, in seconds."""


class _Session:
    """The child's problems, and the backends built for each record."""

    def __init__(self) -> None:
        self.problems: dict[str, tuple[Any, Any]] = {}
        self.backends: dict[
            tuple[str, str], tuple[NativeBackend, Callable[[], object]]
        ] = {}

    def backend(
        self,
        name: str,
        record: Mapping[str, object] | None,
        planning: Mapping[str, object] | None = None,
    ) -> Callable[[], object]:
        """Return the trial step of problem `name` under `record`, built once.

        Parameters
        ----------
        name : str
            A prepared problem.
        record : Mapping[str, object] or None
            A native execution record, or ``None`` for the process's own.
        planning : Mapping[str, object] or None, optional
            An FFT planning (``fft_planning``), or ``None`` for the plan policy.

        Returns
        -------
        collections.abc.Callable
            One trial step from the problem's initial state.
        """
        key = self._key(name, record, planning)
        if key not in self.backends:
            from solprop_native._execution import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
                ExecutionRecord,
            )
            from solprop_native.backend import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
                NativeBackend,
            )

            from solprop.backends.session import Start  # noqa: PLC0415

            simulation, problem = self.problems[name]
            execution: ExecutionRecord | None = (
                None if record is None else ExecutionRecord.from_native(record)
            )
            backend = NativeBackend(
                problem,
                simulation._run_request(),
                grid=simulation.grid,
                modes=simulation.modes,
                profiles=simulation.profiles,
                cpu_execution=execution,
                fft_planning=planning,
                options=simulation.settings.execution_options(),
            )
            step = backend.timing_probe(Start.entrance(problem)).trial
            step()
            self.backends[key] = (backend, step)
        return self.backends[key][1]

    @staticmethod
    def _key(
        name: str,
        record: Mapping[str, object] | None,
        planning: Mapping[str, object] | None = None,
    ) -> tuple[str, str]:
        """Return the cache key of problem `name`'s backend under a setup.

        Parameters
        ----------
        name : str
            A prepared problem.
        record, planning : Mapping[str, object] or None
            The setup.

        Returns
        -------
        tuple of str
            The problem and its setup as JSON.
        """
        return name, json.dumps([record, planning], sort_keys=True)

    def prepare(self, problems: Mapping[str, float]) -> dict[str, object]:
        """Build each problem at its scale and time one step of each.

        Parameters
        ----------
        problems : Mapping[str, float]
            Each calibration problem's scale.

        Returns
        -------
        dict
            Each problem's fastest step and shape, the process settings in force, its
            CPUs, and its execution and built-in records.
        """
        import solprop_native  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        from solprop.calibration.problems import build  # noqa: PLC0415

        report: dict[str, object] = {}
        for name, scale in problems.items():
            simulation = build(name, float(scale))
            problem = simulation.translate()
            self.problems[name] = (simulation, problem)
            step = self.backend(name, None)
            # The fastest of a few steps: the first ones still fault pages in.
            seconds = []
            for _ in range(3):
                begun = time.perf_counter()
                step()
                seconds.append(time.perf_counter() - begun)
            report[name] = {"step": min(seconds), "shape": list(problem.shape)}
        return {
            "problems": report,
            "process": solprop_native.PROCESS.record(),
            "cpus": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else [],
            "execution": solprop_native.EXECUTION.native(),
            "builtin": solprop_native.BUILTIN_EXECUTION.native(),
        }

    def time(
        self,
        name: str,
        repeats: int,
        record: Mapping[str, object] | None,
        planning: Mapping[str, object] | None = None,
    ) -> float:
        """Return the seconds a trial step of `name` takes under `record`.

        Two untimed steps first bring the problem's arrays back into the caches,
        which another child's window has just used.

        Parameters
        ----------
        name : str
            A prepared problem.
        repeats : int
            The trial steps timed.
        record : Mapping[str, object] or None
            A native execution record, or ``None`` for the process's own.
        planning : Mapping[str, object] or None, optional
            An FFT planning, or ``None`` for the plan policy.

        Returns
        -------
        float
            Seconds per step.
        """
        step = self.backend(name, record, planning)
        for _ in range(2):
            step()
        begun = time.perf_counter()
        for _ in range(repeats):
            step()
        return (time.perf_counter() - begun) / repeats

    def capture(
        self,
        effort: str,
        limit: float,
        record: Mapping[str, object] | None,
        budget: float | None = None,
    ) -> dict[str, object]:
        """Plan every problem's transforms alone at `effort`; return their wisdom.

        Each transform is seeded from no wisdom, within `limit` seconds, and its
        wisdom kept by its plan key (SEP 46 section 6.7). The workspaces built are
        closed: only the wisdom is wanted. Within a `budget`, each problem's plans
        share what is left of it, at most `limit` each, and the problems it leaves no
        time for are skipped and listed.

        Parameters
        ----------
        effort : str
            ``"measure"`` or ``"patient"``.
        limit : float
            FFTW's planning time limit per transform, in seconds.
        record : Mapping[str, object] or None
            The execution record the problems run under, whose widths the plans'
            threads follow.
        budget : float or None, optional
            The seconds the whole capture may take; none bounds it but `limit`.

        Returns
        -------
        dict
            ``wisdom`` by plan key, each key's ``groups`` and the problems
            containing it (``problems``), the problems skipped for want of time
            (``skipped``), the seconds spent (``seconds``), and each plan's execution
            alone (``alone``: the seconds of the captured plan and of the same
            transform at estimate, ``_core.time_fft_plan``; solprop issue #595).
        """
        from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            _core,  # pyright: ignore[reportUnknownVariableType]
        )
        from solprop_native._execution import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            ExecutionRecord,
        )
        from solprop_native.backend import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            NativeBackend,
        )

        wisdom: dict[str, str] = {}
        groups: dict[str, str] = {}
        contains: dict[str, list[str]] = {}
        skipped: list[str] = []
        begun = time.perf_counter()
        names = list(self.problems)
        for index, name in enumerate(names):
            simulation, problem = self.problems[name]
            each = limit
            if budget is not None:
                left = budget - (time.perf_counter() - begun)
                # A problem plans a few transforms: share what is left among them.
                each = min(limit, left / (_PLANS_PER_PROBLEM * (len(names) - index)))
                if each < _SHORTEST_LIMIT:
                    skipped.extend(names[index:])
                    break
            backend = NativeBackend(
                problem,
                simulation._run_request(),
                grid=simulation.grid,
                modes=simulation.modes,
                profiles=simulation.profiles,
                cpu_execution=None
                if record is None
                else ExecutionRecord.from_native(record),
                fft_planning={
                    "mode": "capture",
                    "capture_effort": effort,
                    "time_limit": each,
                },
                options=simulation.settings.execution_options(),
            )
            try:
                found = backend._native_state.fft_planning  # pyright: ignore[reportPrivateUsage]
            finally:
                backend.close()
            wisdom |= cast(dict[str, str], found["captured"])
            for key, group in cast(dict[str, str], found["groups"]).items():
                groups[key] = group
                contains.setdefault(name, []).append(key)
        # Each plan alone, as its capture chose it and at estimate, to tell a plan that
        # loses in the step from one that is slower alone (issue #595).
        alone = {
            key: cast(dict[str, float], _core.time_fft_plan(key, text, effort))
            for key, text in wisdom.items()
        }
        return {
            "wisdom": wisdom,
            "groups": groups,
            "problems": contains,
            "skipped": skipped,
            "seconds": time.perf_counter() - begun,
            "alone": alone,
        }

    def sizing(self, record: Mapping[str, object] | None) -> dict[str, object]:
        """Return each problem's operations as its workspace sizes them.

        Parameters
        ----------
        record : Mapping[str, object] or None
            A native execution record, or ``None`` for the process's own.

        Returns
        -------
        dict
            Each problem's ``execution_sizing``: work and width by operation.
        """
        found: dict[str, object] = {}
        for name in self.problems:
            self.backend(name, record)
            backend = self.backends[self._key(name, record)][0]
            found[name] = backend.metadata()["execution_sizing"]
        return found

    def close(self) -> None:
        """Close every backend."""
        for backend, _ in self.backends.values():
            backend.close()
        self.backends.clear()


def _regions(elements: int, repeats: int) -> dict[str, float]:
    """Return the cost of a parallel region at the process's width, in seconds.

    The median of `repeats` timed scaled copies of `elements` complex values on the
    kernels' execution space (``reference_stream_copies``), and of single-element
    copies, whose cost is the region's own.

    Parameters
    ----------
    elements : int
        The complex values of each copy.
    repeats : int
        The timed copies of each size.

    Returns
    -------
    dict of str to float
        ``copy`` and ``empty``: the median seconds of each.
    """
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _core,
    )

    copies = _core.reference_stream_copies(elements, 20, repeats)
    empty = _core.reference_stream_copies(1, 20, repeats)
    return {"copy": statistics.median(copies), "empty": statistics.median(empty)}


def serve(requests: Iterable[str], replies: io.TextIOBase) -> None:
    """Answer requests until ``exit`` or the end of input.

    Parameters
    ----------
    requests : Iterable[str]
        JSON lines, one request each.
    replies : io.TextIOBase
        One JSON line written per request.
    """
    session = _Session()
    try:
        for line in requests:
            if not line.strip():
                continue
            request = json.loads(line)
            op = request.get("op")
            if op == "exit":
                replies.write(json.dumps({"ok": True}) + "\n")
                replies.flush()
                return
            try:
                if op == "prepare":
                    answer: object = session.prepare(request["problems"])
                elif op == "time":
                    answer = session.time(
                        request["problem"],
                        int(request["repeats"]),
                        request.get("record"),
                        request.get("planning"),
                    )
                elif op == "capture":
                    answer = session.capture(
                        request["effort"],
                        float(request["limit"]),
                        request.get("record"),
                        None
                        if request.get("budget") is None
                        else float(request["budget"]),
                    )
                elif op == "sizing":
                    answer = session.sizing(request.get("record"))
                elif op == "regions":
                    answer = _regions(int(request["elements"]), int(request["repeats"]))
                else:
                    raise ValueError(f"unknown request {op!r}")
                reply: dict[str, object] = {"ok": answer}
            except Exception as error:
                reply = {"error": f"{type(error).__name__}: {error}"}
            replies.write(json.dumps(reply) + "\n")
            replies.flush()
    finally:
        session.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Pin, then serve the controller on the process's standard streams.

    The protocol keeps a private copy of standard output; anything else the process
    or the native runtime prints goes to standard error.

    Parameters
    ----------
    argv : sequence of str or None, optional
        ``[CPUS]``; the process's own arguments when ``None``.

    Returns
    -------
    int
        Zero.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    cpus = [int(cpu) for cpu in arguments[0].split(",") if cpu] if arguments else []
    pin(cpus)
    # The synthetic problems' own warnings (an absorber on a small ladder size, say)
    # concern nothing the user built.
    warnings.simplefilter("ignore")
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    serve(sys.stdin, protocol)
    return 0


if __name__ == "__main__":  # pragma: no cover - run as a module by the controller
    raise SystemExit(main())
