# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Opt-in wall-clock timers for the passes of a free-space evaluation (SEP 28 phase 0).

A measurement aid, not a profiler: the backend reads :data:`ENVIRONMENT` once, at
construction, and holds a :class:`PassTimers` only when it is set to ``1``. Otherwise it
holds ``None``, and each instrumented pass costs one comparison with ``None`` -- no
clock read, no call, no allocation -- so a run without the switch takes the same path
as before it existed.

Each instrumented function keeps its own mark, a clock reading, and laps it after every
pass::

    mark = timers.now()
    ...  # the pass
    mark = timers.lap("free.pack", mark)

so that timed functions nest: a caller's lap around a timed callee includes the
callee's own laps. Every pass the backend times is synchronous on the host (FFTW's
threaded execution and Numba's parallel loops return when their work is done), so a
lap is that pass's wall time. The names the backend records are listed in the SEP 28
phase 0 report.
"""

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from solprop.exceptions import BackendError

__all__ = ["ENVIRONMENT", "PassTimers", "PassTotal", "timers_from_environment"]

ENVIRONMENT = "SOLPROP_PASS_TIMERS"
"""The environment variable that enables the timers: ``1`` on, unset or ``0`` off."""


@dataclass(frozen=True)
class PassTotal:
    """The accumulated wall time of one named pass.

    Attributes
    ----------
    calls : int
        How many times the pass ran.
    seconds : float
        Its total wall time, s.
    """

    calls: int
    seconds: float


class PassTimers:
    """Accumulate wall time and call counts per named pass.

    Parameters
    ----------
    clock : callable, optional
        A monotonic clock in seconds; :func:`time.perf_counter` by default.

    Examples
    --------
    >>> ticks = iter([0.0, 0.5, 2.0])
    >>> timers = PassTimers(clock=lambda: next(ticks))
    >>> mark = timers.now()
    >>> mark = timers.lap("pack", mark)
    >>> mark = timers.lap("to_time", mark)
    >>> timers.snapshot()["to_time"]
    PassTotal(calls=1, seconds=1.5)
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}

    def now(self) -> float:
        """Return the clock's reading, s, to start a sequence of laps.

        Returns
        -------
        float
            The clock's reading, s.
        """
        return self._clock()

    def lap(self, name: str, mark: float) -> float:
        """Charge the time since `mark` to `name`, and return the new mark.

        Parameters
        ----------
        name : str
            The pass, for example ``"free.to_time"``.
        mark : float
            The clock's reading when the pass began, s.

        Returns
        -------
        float
            The clock's reading now, s: the next pass's mark.
        """
        now = self._clock()
        self._seconds[name] = self._seconds.get(name, 0.0) + (now - mark)
        self._calls[name] = self._calls.get(name, 0) + 1
        return now

    def reset(self) -> None:
        """Forget every pass recorded so far, to start a timed window."""
        self._seconds.clear()
        self._calls.clear()

    def snapshot(self) -> dict[str, PassTotal]:
        """Return the passes recorded since the last :meth:`reset`.

        Returns
        -------
        dict of str to PassTotal
            Each pass's calls and total wall time, sorted by name.
        """
        return {
            name: PassTotal(self._calls[name], self._seconds[name])
            for name in sorted(self._seconds)
        }


def timers_from_environment(
    environ: Mapping[str, str] = os.environ,
) -> PassTimers | None:
    """Return a :class:`PassTimers` if :data:`ENVIRONMENT` asks for one.

    Parameters
    ----------
    environ : mapping of str to str, optional
        The environment, :data:`os.environ` by default.

    Returns
    -------
    PassTimers or None
        Timers when the variable is ``1``; ``None`` when it is unset, empty or ``0``.

    Raises
    ------
    BackendError
        If the variable holds anything else.

    Examples
    --------
    >>> timers_from_environment({}) is None
    True
    >>> isinstance(timers_from_environment({ENVIRONMENT: "1"}), PassTimers)
    True
    """
    value = environ.get(ENVIRONMENT, "").strip()
    if value in {"", "0"}:
        return None
    if value == "1":
        return PassTimers()
    msg = f"{ENVIRONMENT} must be 0 or 1, and is {value!r}"
    raise BackendError(msg)
