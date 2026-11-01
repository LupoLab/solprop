# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Logging and progress reporting.

Library modules log through ``logging.getLogger(__name__)`` and never configure
handlers; scripts call :func:`configure_logging` once. Progress during a propagation is
reported by the driver through a :class:`ProgressReporter`, never from inside compiled
kernels (``docs/dev/architecture.md`` section 14).

This module is deliberately free of every other part of the package. ``import solprop``
imports it eagerly, and reaching the seam from here -- for the type of the summary a
save plane hands a reporter -- would pull NumPy and jaxtyping into that import and take
it from 0.06 s to 0.6 s, past the budget ``tests/test_package.py`` holds. Hence
:class:`PlaneSummary`, which is the structural half of
:class:`solprop.backends.protocol.StepSummary` that a reporter actually reads.
"""

import builtins
import importlib.util
import io
import logging
import math
import sys
import time
import warnings
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from statistics import median
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "BAR_FORMAT",
    "DEFAULT_BAR_PERIOD",
    "DEFAULT_PROGRESS_PERIOD",
    "NOT_ESTIMATED",
    "BarReporter",
    "InterruptibleReporter",
    "LogReporter",
    "NullReporter",
    "NullScanReporter",
    "PlaneSummary",
    "ProgressChoice",
    "ProgressReporter",
    "ProgressStatus",
    "ScanReporter",
    "SummarisingReporter",
    "WatchingScanReporter",
    "configure_logging",
    "default_progress",
    "default_reporter",
    "format_duration",
    "is_interactive",
    "resolve_reporter",
    "set_default_progress",
]

PACKAGE_LOGGER = "solprop"
DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

DEFAULT_PROGRESS_PERIOD = 10.0
"""Seconds between periodic progress lines when nothing says otherwise."""

DEFAULT_BAR_PERIOD = 0.5
"""Seconds between redraws of a progress bar."""

NOT_ESTIMATED = "not estimated (replayed schedule)"
"""What a line prints in place of a tolerance ratio the march did not estimate.

The low-storage march records 0.0 in the norm column, meaning "not estimated" (SEP 24
section 6.3); printed as a ratio it would read as a perfect step.
"""

ProgressChoice = Literal["auto", "bar", "log", "none"]
"""The names ``progress=`` accepts beside a reporter of its own.

``"auto"`` draws a bar when the process is interactive (:func:`is_interactive`) and
logs lines otherwise; ``"bar"`` and ``"log"`` force one or the other; ``"none"``
reports nothing.
"""

_PROGRESS_CHOICES: tuple[str, ...] = ("auto", "bar", "log", "none")
_DEFAULTS: dict[str, ProgressChoice] = {"progress": "auto"}

BAR_FORMAT = (
    "{desc}{percentage:5.1f}%|{bar}| {n:.4g}/{total:.4g} {unit} [{elapsed}{postfix}]"
)
"""What a bar shows: no ``{remaining}`` and no ``{rate}``.

tqdm's own remaining-time estimate is the whole-run one, which was measured to be 1.6 to
23 times worse than the windowed estimate of :class:`LogReporter`; the windowed one
rides in the postfix, and a bar must not show two ETAs.
"""

# A run says nothing about its ETA until it is this far along *and* this many periods in
# (whichever is later). Both guards are needed: a fraction alone lets a run that raced
# through a cheap first millimetre extrapolate from one sample, and a count of periods
# alone lets a run that has covered nothing divide by a rate near zero.
ETA_MIN_FRACTION = 0.02
ETA_MIN_PERIODS = 2

# The trailing window the reported rate is taken over: five periods, or a tenth of the
# elapsed time once that is longer, so that the window grows with the run instead of
# staying at a fixed number of samples. Chosen, not derived: a run only a few periods
# long has a window that is most of the run.
ETA_WINDOW_PERIODS = 5
ETA_WINDOW_FRACTION = 0.10


def configure_logging(
    level: int | str = logging.INFO,
    file: Path | None = None,
    fmt: str = DEFAULT_FORMAT,
    *,
    progress: ProgressChoice | None = None,
) -> logging.Logger:
    """Attach a handler to the package logger and set its level.

    Parameters
    ----------
    level : int or str, optional
        Logging level for the package logger, e.g. ``logging.DEBUG`` or ``"DEBUG"``.
    file : Path or None, optional
        If given, log to this file instead of standard error.
    fmt : str, optional
        Log record format.
    progress : {'auto', 'bar', 'log', 'none'} or None, optional
        The process-wide default for how a run reports progress when a call does not
        say (:func:`set_default_progress`). ``None`` leaves the current default.

    Returns
    -------
    logging.Logger
        The package logger, configured. Calling this function again replaces the handler
        it installed before, so repeated calls do not duplicate output.

    Examples
    --------
    >>> import logging
    >>> logger = configure_logging("WARNING")
    >>> logger.level == logging.WARNING
    True
    >>> logger.setLevel(logging.NOTSET)  # back to inheriting the root logger's level
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        if getattr(handler, "_solprop_installed", False):
            logger.removeHandler(handler)
            handler.close()
    handler: logging.Handler
    handler = (
        logging.FileHandler(file)
        if file is not None
        else logging.StreamHandler(sys.stderr)
    )
    handler.setFormatter(logging.Formatter(fmt))
    handler._solprop_installed = True  # type: ignore[attr-defined]  # marks our handler
    logger.addHandler(handler)
    logger.setLevel(level)
    if progress is not None:
        set_default_progress(progress)
    return logger


def set_default_progress(choice: ProgressChoice) -> None:
    """Set how a run reports progress when its call does not say.

    Parameters
    ----------
    choice : {'auto', 'bar', 'log', 'none'}
        See :data:`ProgressChoice`. A per-call ``progress=`` keyword beats this
        default, which beats ``"auto"``.

    Raises
    ------
    ValueError
        If the choice is not one of the four names.

    Examples
    --------
    >>> from solprop.log import default_progress, set_default_progress
    >>> set_default_progress("log")
    >>> default_progress()
    'log'
    >>> set_default_progress("auto")
    """
    if choice not in _PROGRESS_CHOICES:
        raise ValueError(f"progress must be one of {_PROGRESS_CHOICES}, got {choice!r}")
    _DEFAULTS["progress"] = choice


def default_progress() -> ProgressChoice:
    """Return the process-wide progress default.

    Returns
    -------
    {'auto', 'bar', 'log', 'none'}
        What :func:`set_default_progress` last set, ``"auto"`` until then.
    """
    return _DEFAULTS["progress"]


def is_interactive(stream: io.TextIOBase | None = None) -> bool:
    """Return whether a progress bar would be seen by a person.

    True inside a Jupyter kernel (the shell ``get_ipython()`` returns is a
    ``ZMQInteractiveShell``, the test ``tqdm.autonotebook`` itself makes) and when the
    stream is a terminal. A redirect, a pipe, a batch scheduler and pytest's capture
    are none of these, so they get log lines and never carriage returns.

    Parameters
    ----------
    stream : io.TextIOBase or None, optional
        The stream a bar would draw on; ``None`` is standard error.

    Returns
    -------
    bool
        Whether to draw a bar under ``progress="auto"``.

    Examples
    --------
    >>> import io
    >>> from solprop.log import is_interactive
    >>> is_interactive(io.StringIO())
    False
    """
    get_ipython = getattr(builtins, "get_ipython", None)
    if get_ipython is not None:
        shell = get_ipython()
        if shell is not None and type(shell).__name__ == "ZMQInteractiveShell":
            return True
    target: Any = sys.stderr if stream is None else stream
    isatty = getattr(target, "isatty", None)
    try:
        return bool(isatty()) if isatty is not None else False
    except ValueError, OSError:
        # A closed or detached stream is not a terminal anyone is watching.
        return False


def format_duration(seconds: float) -> str:
    """Render a duration as ``H:MM:SS``, or ``N days, H:MM:SS`` beyond a day.

    Whole seconds: a progress line that resolves an ETA finer than that is claiming an
    accuracy no estimator here has. A duration that is negative or not finite -- an ETA
    formed from a rate of zero, say -- reads ``unknown`` rather than raising, because a
    reporter must not be able to end a run.

    Parameters
    ----------
    seconds : float
        The duration, s.

    Returns
    -------
    str
        The duration, or ``"unknown"``.

    Examples
    --------
    >>> format_duration(751.4)
    '0:12:31'
    >>> format_duration(3600.0)
    '1:00:00'
    >>> format_duration(90061.0)
    '1 day, 1:01:01'
    >>> format_duration(float("inf"))
    'unknown'
    """
    if not math.isfinite(seconds) or seconds < 0.0:
        return "unknown"
    return str(timedelta(seconds=round(seconds)))


@runtime_checkable
class PlaneSummary(Protocol):
    """The scalars a reporter reads from a save plane.

    Structural rather than imported, so that this module stays free of the seam and
    ``import solprop`` keeps its budget (see the module docstring).
    :class:`solprop.backends.protocol.StepSummary` satisfies it. The four are properties
    rather than attributes because the class that satisfies it is a frozen dataclass,
    and a writable protocol member would not accept one.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> isinstance(SimpleNamespace(n_steps=8, n_rejected=0, max_norm=0.4,
    ...                            wall_time=0.1), PlaneSummary)
    True
    """

    @property
    def n_steps(self) -> int:
        """Return the steps taken to reach the plane, accepted and rejected together.

        Returns
        -------
        int
            The attempted steps.
        """
        ...

    @property
    def n_rejected(self) -> int:
        """Return the rejected steps among them.

        Returns
        -------
        int
            The rejected steps.
        """
        ...

    @property
    def max_norm(self) -> float:
        """Return the largest scaled error norm over the accepted steps.

        Returns
        -------
        float
            The norm; 1 is the tolerance the run asked for.
        """
        ...

    @property
    def wall_time(self) -> float:
        """Return the seconds the plane took.

        Returns
        -------
        float
            Wall time, s.
        """
        ...


@runtime_checkable
class ProgressReporter(Protocol):
    """Receive progress events from the propagation driver.

    Four calls, in this order: :meth:`started` once, :meth:`step` once per **attempted**
    step including the rejected ones, :meth:`plane` once per save plane, and
    :meth:`finished` once at the end. A reporter may also implement the fifth call
    of :class:`solprop.log.InterruptibleReporter`, which the driver makes instead
    of :meth:`finished` when the loop raises, ``KeyboardInterrupt`` included, so
    that a reporter holding a terminal can give it back; a reporter without it is
    still complete. The driver holds no state on a reporter's behalf and never
    inspects what it returns.

    :meth:`step` is on the hot path -- one attribute lookup and one call inside the step
    loop -- so a reporter that does real work there must do it on a wall-clock period,
    not on every call. Nothing a reporter does may change the propagation: a run with a
    reporter is bitwise the run without one.

    Subclass :class:`solprop.log.NullReporter` to implement only the calls you care
    about.

    Examples
    --------
    >>> isinstance(NullReporter(), ProgressReporter)
    True
    """

    def started(self, z_start: float, z_end: float, n_planes: int) -> None:
        """Record that a run has begun.

        Parameters
        ----------
        z_start : float
            Position the run starts from, m; non-zero on a resume.
        z_end : float
            Position the run ends at, m.
        n_planes : int
            Save planes the run will write.
        """

    def step(self, z: float, dz: float, accepted: bool) -> None:
        """Record one attempted step.

        Parameters
        ----------
        z : float
            Position at the end of the attempt, m. On a **rejected** step this is where
            the attempt would have landed, which is ahead of the run's true position.
        dz : float
            Length of the attempt, m.
        accepted : bool
            Whether the step was kept.
        """

    def plane(self, z: float, summary: PlaneSummary) -> None:
        """Record that a save plane was reached.

        Parameters
        ----------
        z : float
            Position of the plane, m.
        summary : PlaneSummary
            What the call to the plane cost and achieved.
        """

    def finished(self, total: float) -> None:
        """Record that the run ended.

        Parameters
        ----------
        total : float
            Wall time of the whole run, s.
        """


@runtime_checkable
class InterruptibleReporter(ProgressReporter, Protocol):
    """A reporter that also wants to hear that the run stopped before its end.

    The driver calls :meth:`interrupted` from its ``finally`` when the loop raised,
    and re-raises. :class:`NullReporter` implements it as a no-op, so a subclass has
    it; a reporter written to the four calls alone is still a
    :class:`ProgressReporter` and is simply not told.

    Examples
    --------
    >>> isinstance(NullReporter(), InterruptibleReporter)
    True
    """

    def interrupted(self, z: float, error: BaseException) -> None:
        """Record that the run stopped before its end.

        Parameters
        ----------
        z : float
            The last save plane the run reached, m.
        error : BaseException
            What stopped it; the driver re-raises it after this call.
        """


@runtime_checkable
class SummarisingReporter(ProgressReporter, Protocol):
    """A reporter that also shows the summary a run gives before its first step.

    The driver calls :meth:`summary` once, before the run's first step, with the block
    of :func:`solprop.driver.summary.run_summary`: the backend, the grid, the
    representation, the medium, the responses, the shot noise, the integrator and its
    tolerances, the save planes, the output and whether the run is a resume.
    :class:`NullReporter` implements it as a no-op; a reporter written to the four
    basic calls alone is not shown it.

    Examples
    --------
    >>> isinstance(NullReporter(), SummarisingReporter)
    True
    """

    def summary(self, text: str) -> None:
        """Show what the run is about to compute.

        Parameters
        ----------
        text : str
            The summary block, one line per row, without a trailing newline.
        """


@runtime_checkable
class ScanReporter(Protocol):
    """Receive the progress of a parameter scan, point by point (SEP 41, S4).

    :func:`solprop.driver.scan.run_scan` takes one as ``progress=`` in place of its
    named choices and then draws no display of its own. Four calls:
    :meth:`scan_started` once, :meth:`point_started` and :meth:`point_finished` once
    for each point dispatched or collected, and :meth:`scan_finished` once when the
    scan returns, stopped by its flag or not; an exception from the scan propagates
    without it. The calls are made in the controlling process, in the thread that
    called ``run_scan``. When a point starts depends on the executor: ``Local`` at its
    build, ``Pool`` when the point is sent to a worker, and ``Batch``, ``Queue`` and
    ``Slurm`` only when its result is collected, just before :meth:`point_finished`. A
    reporter that also wants each point's own run implements
    :class:`solprop.log.WatchingScanReporter`.

    Subclass :class:`solprop.log.NullScanReporter` to implement only the calls you care
    about.

    Examples
    --------
    >>> isinstance(NullScanReporter(), ScanReporter)
    True
    """

    def scan_started(self, n_points: int) -> None:
        """Record that a scan has begun.

        Parameters
        ----------
        n_points : int
            The selected points, those a rerun skips as complete included.
        """

    def point_started(self, index: int, point: object) -> None:
        """Record that a point has been dispatched.

        Parameters
        ----------
        index : int
            The point's global index.
        point : solprop.driver.scan.ScanPoint
            Its values and seed.
        """

    def point_finished(self, index: int, status: str, wall: float) -> None:
        """Record that a point's result has been collected.

        Parameters
        ----------
        index : int
            The point's global index.
        status : str
            ``"done"``; ``"failed"``; ``"skipped"`` for a point complete in the file;
            ``"conflict"`` for a complete point whose stored record agrees but for
            rounding, neutral values or provenance, which is kept; or ``"stopped"``
            for a point the stop flag stopped, which is not committed and a rerun
            computes again.
        wall : float
            The point's build and run time as its worker measured it, s; for a batch
            member, its group's.
        """

    def scan_finished(self) -> None:
        """Record that the scan has returned."""


@runtime_checkable
class WatchingScanReporter(ScanReporter, Protocol):
    """A scan reporter that also watches each point's own run (SEP 41, S4).

    Before a point runs, the scan asks for a :class:`ProgressReporter` for it, which
    then receives that run's calls. ``Local`` runs the point with it in this process.
    A ``Pool`` worker sends the point's ``summary``, ``started``, ``plane``,
    ``finished`` and ``interrupted`` calls to the controller, which makes them on the
    reporter returned here as the messages arrive; each plane carries its position and
    the four scalars of its :class:`PlaneSummary`, never a field, and ``step``, on the
    hot path, is not sent. ``Batch``, ``Queue`` and ``Slurm`` ask for none.

    Examples
    --------
    >>> isinstance(NullScanReporter(), WatchingScanReporter)
    True
    """

    def point_reporter(self, index: int) -> ProgressReporter | None:
        """Return the reporter for one point's run, or ``None`` for none.

        Parameters
        ----------
        index : int
            The point's global index.

        Returns
        -------
        ProgressReporter or None
            Where the point's run reports; ``None`` runs it silently.
        """
        ...


class NullScanReporter:
    """A scan reporter that does nothing, and the base for one that does something.

    Every method is a no-op and :meth:`point_reporter` returns ``None``, so a subclass
    overrides only what it wants:

    Examples
    --------
    >>> class Count(NullScanReporter):
    ...     def point_finished(self, index, status, wall):
    ...         print(index, status)
    >>> isinstance(Count(), ScanReporter)
    True
    >>> Count().point_reporter(0) is None
    True
    """

    def scan_started(self, n_points: int) -> None:
        """Ignore the event.

        Parameters
        ----------
        n_points : int
            See :meth:`ScanReporter.scan_started`.
        """

    def point_started(self, index: int, point: object) -> None:
        """Ignore the event.

        Parameters
        ----------
        index : int
            See :meth:`ScanReporter.point_started`.
        point : object
            See :meth:`ScanReporter.point_started`.
        """

    def point_finished(self, index: int, status: str, wall: float) -> None:
        """Ignore the event.

        Parameters
        ----------
        index : int
            See :meth:`ScanReporter.point_finished`.
        status : str
            See :meth:`ScanReporter.point_finished`.
        wall : float
            See :meth:`ScanReporter.point_finished`.
        """

    def scan_finished(self) -> None:
        """Ignore the event."""

    def point_reporter(self, index: int) -> ProgressReporter | None:
        """Watch no point.

        Parameters
        ----------
        index : int
            See :meth:`WatchingScanReporter.point_reporter`.

        Returns
        -------
        None
            The point runs silently.
        """
        del index
        return None


class NullReporter:
    """A reporter that does nothing, and the base class for one that does something.

    Every method is a no-op, so a subclass overrides only what it wants:

    Examples
    --------
    >>> class Bar(NullReporter):
    ...     def plane(self, z, summary):
    ...         print(f"reached {z} m")
    >>> isinstance(Bar(), ProgressReporter)
    True
    """

    def summary(self, text: str) -> None:
        """Ignore the event.

        Parameters
        ----------
        text : str
            See :meth:`solprop.log.SummarisingReporter.summary`.
        """

    def started(self, z_start: float, z_end: float, n_planes: int) -> None:
        """Ignore the event.

        Parameters
        ----------
        z_start, z_end : float
            See :meth:`ProgressReporter.started`.
        n_planes : int
            See :meth:`ProgressReporter.started`.
        """

    def step(self, z: float, dz: float, accepted: bool) -> None:
        """Ignore the event.

        Parameters
        ----------
        z, dz : float
            See :meth:`ProgressReporter.step`.
        accepted : bool
            See :meth:`ProgressReporter.step`.
        """

    def plane(self, z: float, summary: PlaneSummary) -> None:
        """Ignore the event.

        Parameters
        ----------
        z : float
            See :meth:`ProgressReporter.plane`.
        summary : PlaneSummary
            See :meth:`ProgressReporter.plane`.
        """

    def finished(self, total: float) -> None:
        """Ignore the event.

        Parameters
        ----------
        total : float
            See :meth:`ProgressReporter.finished`.
        """

    def interrupted(self, z: float, error: BaseException) -> None:
        """Ignore the event.

        Parameters
        ----------
        z : float
            See :meth:`solprop.log.InterruptibleReporter.interrupted`.
        error : BaseException
            See :meth:`solprop.log.InterruptibleReporter.interrupted`.
        """


class LogReporter:
    """Report progress as log records: a periodic line, one per plane, one at the end.

    Three kinds of line, all at ``level`` (``INFO`` by default) on this module's logger,
    after the run's **summary** (:meth:`summary`), one record logged at the same level.
    A **periodic** line, at most one per ``period`` seconds however fast the steps
    come, says where the run is and when it expects to finish. A **plane** line says
    what the last save plane cost. The **closing** line says what the run achieved
    against what it asked for, and how far off its own ETAs turned out to be. Nothing
    is printed; call :func:`solprop.log.configure_logging` to see the lines.

    **The ETA is from the recent rate, and it is measured.** Two estimates are formed
    at every periodic line: the whole-run one, ``remaining z / (z done / elapsed)``,
    and one from the rate of z over a trailing window of
    ``max(5 periods, 10 % of the elapsed time)``. The windowed one
    is reported, because the whole-run one is linear in z from the first step and most
    runs are not: the steps are short while a pulse compresses and long after it, a
    gradient fill changes the operator and the plasma cost along z, and a plane-dense
    run pays its planes unevenly. When a fixed schedule is replayed the total step count
    is known, so the estimate is the remaining steps at the recent cost per step. Both
    estimates' median absolute relative errors are kept, and :meth:`finished` logs the
    reported one, so the estimator is measured on every run rather than assumed.

    Parameters
    ----------
    period : float, optional
        Seconds between periodic lines, s. ``0`` disables them and leaves the plane and
        closing lines.
    logger : logging.Logger or None, optional
        Logger to use; defaults to this module's.
    clock : callable, optional
        Returns a monotonic time in seconds. Injected so that the period and the ETA can
        be tested without waiting.
    n_steps : int or None, optional
        Total steps of a replayed fixed schedule, which turns the ETA into a
        remaining-steps estimate. ``None`` for an adaptive run.
    level : int, optional
        Level the three lines are logged at.
    estimated : bool, optional
        Whether the steps carry an error estimate. ``False`` for the low-storage
        march, whose 0.0 norms mean "not estimated": the closing line then prints
        :data:`solprop.log.NOT_ESTIMATED` in place of the tolerance ratio.

    Examples
    --------
    >>> import logging
    >>> from types import SimpleNamespace
    >>> reporter = LogReporter(level=logging.DEBUG)
    >>> reporter.plane(0.5, SimpleNamespace(n_steps=40, n_rejected=1, max_norm=0.7,
    ...                                     wall_time=0.03))
    >>> reporter.finished(0.06)
    >>> reporter.n_steps, reporter.tolerance_ratio
    (40, 0.7)
    """

    def __init__(
        self,
        *,
        period: float = DEFAULT_PROGRESS_PERIOD,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] = time.perf_counter,
        n_steps: int | None = None,
        level: int = logging.INFO,
        estimated: bool = True,
    ) -> None:
        self._period = float(period)
        self._estimated = estimated
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._clock = clock
        self._n_total_steps = n_steps
        self._level = level
        # The plane line's own level: a bar demotes it to DEBUG, because a screen
        # showing a bar does not want one line per plane as well.
        self._plane_level = level
        self._planes_done = 0
        self._t0: float | None = None
        self._emitted = 0
        self._z_start = 0.0
        self._z_end = 0.0
        # What `started` was told, kept whole: a subclass reading this reporter's state
        # should see the run's shape, not only the parts these three lines use.
        self._n_planes = 0
        self._z = 0.0
        self._z_plane = 0.0
        self._n_attempts = 0
        self._n_accepted = 0
        self._n_rejected = 0
        self._plane_steps = 0
        self._plane_rejected = 0
        self._dz = 0.0
        self._max_norm = 0.0
        self._window: deque[tuple[float, float, int]] = deque()
        # (elapsed at the line, windowed ETA, whole-run ETA) for every line that
        # published one; `finished` turns them into the two error medians.
        self._predictions: list[tuple[float, float, float]] = []
        self._elapsed_final: float | None = None

    @property
    def n_steps(self) -> int:
        """Return the attempted steps of the run so far.

        Returns
        -------
        int
            The larger of the two channels that count: the per-step calls when a driver
            is attached, and the plane summaries otherwise. Both count every attempted
            step, so when both are live they agree.
        """
        return max(self._n_attempts, self._plane_steps)

    @property
    def n_rejected(self) -> int:
        """Return the rejected steps of the run so far.

        Returns
        -------
        int
            Counted from whichever channel is live; see :attr:`n_steps`.
        """
        return max(self._n_rejected, self._plane_rejected)

    @property
    def tolerance_ratio(self) -> float:
        """Return what the run achieved against what it asked for.

        Returns
        -------
        float
            The largest scaled error norm any accepted step scored
            (:attr:`solprop.stepper.controller.ControlRecord.tolerance_ratio`); 0 before
            the first plane.
        """
        return self._max_norm

    @property
    def eta_error(self) -> float:
        """Return how far off the reported ETAs were.

        Returns
        -------
        float
            Median absolute relative error of the windowed estimate over every periodic
            line that published one; NaN before :meth:`finished`, or when no line did.
        """
        return self._median_error(0)

    @property
    def eta_error_whole_run(self) -> float:
        """Return how far off the whole-run estimator would have been.

        Formed on the same lines as :attr:`eta_error` but never reported, so that the
        two can be compared on every run.

        Returns
        -------
        float
            Median absolute relative error of the whole-run estimate; NaN as for
            :attr:`eta_error`.
        """
        return self._median_error(1)

    def summary(self, text: str) -> None:
        """Log the run's summary as one record.

        Parameters
        ----------
        text : str
            See :meth:`solprop.log.SummarisingReporter.summary`.
        """
        self._say(self._level, "%s", text)

    def started(self, z_start: float, z_end: float, n_planes: int) -> None:
        """Take the run's extent and start the clock.

        Emits nothing: the promise is one periodic line per period, and a run shorter
        than its period says nothing until its first plane.

        Parameters
        ----------
        z_start, z_end : float
            See :meth:`ProgressReporter.started`.
        n_planes : int
            See :meth:`ProgressReporter.started`.
        """
        self._t0 = self._clock()
        self._emitted = 0
        self._z_start = float(z_start)
        self._z_end = float(z_end)
        self._n_planes = int(n_planes)
        self._planes_done = 0
        self._z = float(z_start)
        self._z_plane = float(z_start)
        self._window.clear()
        self._predictions.clear()
        self._elapsed_final = None

    def step(self, z: float, dz: float, accepted: bool) -> None:
        """Count one attempted step and emit a periodic line if one is due.

        Parameters
        ----------
        z, dz : float
            See :meth:`ProgressReporter.step`.
        accepted : bool
            See :meth:`ProgressReporter.step`.
        """
        self._n_attempts += 1
        self._dz = dz
        if accepted:
            self._n_accepted += 1
            # A rejected attempt's `z` is where it *would* have landed, so only an
            # accepted one moves the position a progress line reports.
            self._z = z
        else:
            self._n_rejected += 1
        if self._period <= 0.0 or self._t0 is None:
            return
        now = self._clock()
        # Count whole periods since the start rather than measuring from the last line:
        # `now - last >= period` drifts by up to one step's duration every line, and on
        # a clock whose ticks are tenths it loses lines outright to rounding.
        elapsed_periods = int((now - self._t0) / self._period)
        if elapsed_periods <= self._emitted:
            return
        self._emitted = elapsed_periods
        self._emit(now)

    def plane(self, z: float, summary: PlaneSummary) -> None:
        """Log one line for a save plane.

        Parameters
        ----------
        z : float
            See :meth:`ProgressReporter.plane`.
        summary : PlaneSummary
            See :meth:`ProgressReporter.plane`.
        """
        span = z - self._z_plane
        self._z_plane = z
        self._z = max(self._z, z)
        self._planes_done += 1
        self._plane_steps += summary.n_steps
        self._plane_rejected += summary.n_rejected
        self._max_norm = max(self._max_norm, summary.max_norm)
        self._logger.log(
            self._plane_level,
            "z = %.6e m: %d steps (%d rejected), mean step %.3e m, %.2f s",
            z,
            summary.n_steps,
            summary.n_rejected,
            span / summary.n_steps if summary.n_steps else span,
            summary.wall_time,
        )

    def finished(self, total: float) -> None:
        """Log the closing line, with the ETA error the run's own estimates made.

        Parameters
        ----------
        total : float
            See :meth:`ProgressReporter.finished`.
        """
        # The reporter's own clock, not `total`: the driver starts timing before the
        # entrance plane is written, so the two zeros are not the same instant and an
        # error measured against `total` would carry that offset.
        if self._t0 is not None:
            self._elapsed_final = self._clock() - self._t0
        error = self.eta_error
        lines = len(self._predictions)
        tail = (
            ""
            if math.isnan(error)
            else f", ETA error {100.0 * error:.1f} % over {lines} lines"
        )
        self._say(
            self._level,
            "finished at z = %.6e m: %d steps (%d rejected) in %.2f s, achieved/"
            "requested tolerance ratio %s%s",
            self._z,
            self.n_steps,
            self.n_rejected,
            total,
            f"{self._max_norm:.3e}" if self._estimated else NOT_ESTIMATED,
            tail,
        )

    def interrupted(self, z: float, error: BaseException) -> None:
        """Log one warning saying where the run stopped and what stopped it.

        Parameters
        ----------
        z : float
            See :meth:`solprop.log.InterruptibleReporter.interrupted`.
        error : BaseException
            See :meth:`solprop.log.InterruptibleReporter.interrupted`.
        """
        if self._t0 is not None:
            self._elapsed_final = self._clock() - self._t0
        self._z = max(self._z, float(z))
        self._say(
            logging.WARNING,
            "stopped at z = %.6e m by %s after %d steps (%d rejected), %s elapsed: %s",
            self._z,
            type(error).__name__,
            self.n_steps,
            self.n_rejected,
            format_duration(self._elapsed_final or 0.0),
            error,
        )

    def _say(self, level: int, message: str, *args: object) -> None:
        """Log one of the lines a person reads whole: the summary and the closing one.

        Parameters
        ----------
        level : int
            The record's level.
        message : str
            The ``%``-format of the record.
        *args : object
            Its arguments.
        """
        self._logger.log(level, message, *args)

    def _median_error(self, index: int) -> float:
        """Return the median absolute relative error of one of the two estimators.

        Parameters
        ----------
        index : int
            ``0`` for the windowed estimate, ``1`` for the whole-run one.

        Returns
        -------
        float
            The median over every line that published an estimate, or NaN.
        """
        if self._elapsed_final is None:
            return math.nan
        errors = [
            abs(sample[1 + index] - remaining) / remaining
            for sample in self._predictions
            if (remaining := self._elapsed_final - sample[0]) > 0.0
            and math.isfinite(sample[1 + index])
        ]
        return median(errors) if errors else math.nan

    def _status(self, now: float) -> "ProgressStatus":  # noqa: UP037 - bound after the class body
        """Advance the trailing window, form both ETAs and record them.

        The one place the estimator's bookkeeping happens, so every reporter built on
        this one -- the line and the bar -- is measured the same way.

        Parameters
        ----------
        now : float
            The clock reading the status is formed at, s.

        Returns
        -------
        ProgressStatus
            Where the run is and what it expects.
        """
        elapsed = now - self._t0 if self._t0 is not None else 0.0
        self._window.append((now, self._z, self._n_accepted))
        width = max(ETA_WINDOW_PERIODS * self._period, ETA_WINDOW_FRACTION * elapsed)
        while len(self._window) > 1 and now - self._window[0][0] > width:
            self._window.popleft()
        span = self._z_end - self._z_start
        done = self._z - self._z_start
        fraction = done / span if span > 0.0 else 1.0
        windowed, whole, window = self._estimates(now, elapsed)
        known = (
            fraction >= ETA_MIN_FRACTION
            and self._emitted >= ETA_MIN_PERIODS
            and math.isfinite(windowed)
        )
        if known:
            self._predictions.append((elapsed, windowed, whole))
        return ProgressStatus(elapsed, fraction, windowed, window, known)

    @staticmethod
    def _tail(status: "ProgressStatus") -> str:  # noqa: UP037 - bound after the class body
        """Return the ETA clause of a line or a bar.

        Parameters
        ----------
        status : ProgressStatus
            What :meth:`_status` returned.

        Returns
        -------
        str
            ``ETA H:MM:SS (from the last H:MM:SS)`` or ``ETA unknown``.
        """
        if not status.known:
            return "ETA unknown"
        return (
            f"ETA {format_duration(status.eta)} "
            f"(from the last {format_duration(status.window)})"
        )

    def _emit(self, now: float) -> None:
        """Form both ETAs, record them, and log one periodic line.

        Parameters
        ----------
        now : float
            The clock reading this line is being emitted at, s.
        """
        status = self._status(now)
        if not self._logger.isEnabledFor(self._level):
            return
        self._logger.log(
            self._level,
            "progress %.1f %%: z = %.4f of %.4f m, %d steps (%d rejected), dz %.1e m, "
            "%s elapsed, %s",
            100.0 * status.fraction,
            self._z,
            self._z_end,
            self._n_attempts,
            self._n_rejected,
            self._dz,
            format_duration(status.elapsed),
            self._tail(status),
        )

    def _estimates(self, now: float, elapsed: float) -> tuple[float, float, float]:
        """Return the windowed ETA, the whole-run ETA and the window's own length.

        Parameters
        ----------
        now : float
            The clock reading the estimates are formed at, s.
        elapsed : float
            Time since :meth:`started`, s.

        Returns
        -------
        tuple of float
            The windowed estimate, the whole-run estimate and the length of the window
            the first was taken over, all in seconds. An estimate that cannot be formed
            from a positive rate is infinite, which formats as ``unknown``.
        """
        remaining = max(self._z_end - self._z, 0.0)
        done = self._z - self._z_start
        whole = remaining * elapsed / done if done > 0.0 < elapsed else math.inf
        opened, z_opened, n_opened = self._window[0]
        window = now - opened
        if window <= 0.0:
            return math.inf, whole, window
        if self._n_total_steps is not None:
            # A replayed schedule knows how many steps are left, so the ETA is those
            # steps at the cost per step the window measured -- which is right even
            # where the step length is not, because a fixed schedule's steps are not
            # uniform in z.
            taken = self._n_accepted - n_opened
            left = max(self._n_total_steps - self._n_accepted, 0)
            return (left * window / taken if taken > 0 else math.inf), whole, window
        covered = self._z - z_opened
        return (
            remaining * window / covered if covered > 0.0 else math.inf,
            whole,
            window,
        )


@dataclass(frozen=True)
class ProgressStatus:
    """Where a run is and what it expects, as the line reporter forms it each period.

    Attributes
    ----------
    elapsed : float
        Seconds since :meth:`ProgressReporter.started`.
    fraction : float
        The fraction of the run's extent covered, in ``[0, 1]``.
    eta : float
        The windowed estimate of the seconds remaining; infinite when none can be
        formed.
    window : float
        The length of the trailing window the estimate was taken over, s.
    known : bool
        Whether the estimate is published: past both thresholds and finite.
    """

    elapsed: float
    fraction: float
    eta: float
    window: float
    known: bool


TQDM_MISSING = (
    "the progress bar needs tqdm, a dependency of solprop that this "
    "environment does not have: run `uv sync` (or install tqdm) in the environment "
    "the kernel or script runs in"
)


def _tqdm_available() -> bool:
    """Return whether tqdm can be imported, without importing it.

    Returns
    -------
    bool
        Whether a bar can be drawn in this environment.
    """
    return importlib.util.find_spec("tqdm") is not None


def _tqdm_class() -> Any:
    """Import tqdm's automatic front end, on first use only.

    ``import solprop`` must stay inside its budget (``tests/test_package.py``), so the
    import happens when a bar is first drawn. ``tqdm.auto`` warns when a notebook
    lacks ``ipywidgets`` and falls back to the text bar, which is the documented
    fallback, so the warning is not one the suite should turn into an error.

    Returns
    -------
    type
        The ``tqdm`` class to draw with.

    Raises
    ------
    ImportError
        If tqdm is not installed in this environment, saying what to do.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from tqdm.auto import tqdm  # noqa: PLC0415 - deferred: the import budget
        except ModuleNotFoundError as error:
            raise ImportError(TQDM_MISSING) from error
    return tqdm


class _BarHandler(logging.Handler):
    """A console handler that writes above a bar instead of through it.

    Takes the place of a :class:`logging.StreamHandler` for a bar's lifetime and
    writes each record with ``tqdm.write``, which clears the bars drawn on that stream,
    writes the line and redraws them.

    Parameters
    ----------
    tqdm : type
        The class whose ``write`` to use.
    original : logging.StreamHandler
        The handler being stood in for, whose stream, level, formatter and filters
        this one takes.
    """

    def __init__(self, tqdm: Any, original: logging.StreamHandler) -> None:  # type: ignore[type-arg] - a text stream
        super().__init__(original.level)
        self.setFormatter(original.formatter)
        for record_filter in original.filters:
            self.addFilter(record_filter)
        self._tqdm = tqdm
        self._stream = original.stream

    def emit(self, record: logging.LogRecord) -> None:
        """Write one record above the bars on this handler's stream.

        Parameters
        ----------
        record : logging.LogRecord
            The record to write.
        """
        try:
            self._tqdm.write(self.format(record), file=self._stream)
        except Exception:  # logging must not end a run: logging.Handler's contract
            self.handleError(record)


class BarReporter(LogReporter):
    """Draw one line that redraws in place: a bar over z with the counts and the ETA.

    A :class:`LogReporter` whose periodic event redraws a ``tqdm`` bar instead of
    logging a line, so the counters, the clock, the windowed ETA and its measured
    error are the same code as the line's. What the bar shows is the periodic line's
    content: the fraction of the run, the position against the extent (metres, or
    steps when a fixed schedule is replayed and the count is known), the elapsed time,
    and in the postfix the attempted and rejected steps, the current step and this
    package's ETA with the window it came from. tqdm's own remaining-time estimate is
    not shown (``BAR_FORMAT``). The plane line is demoted to ``DEBUG`` and the
    bar's description says which plane the run is on; the closing line is logged as
    on the line reporter, after the bar has been closed. Log records written to the
    bar's console while it is up go above it (``_BarHandler``), and the handlers are
    put back when the bar closes -- also when the run is interrupted.

    **The summary and the closing line are shown, not only logged.** A bar is drawn
    where a person is watching, and the package installs no handler, so a record alone
    would reach nobody: each of the two is written to the bar's stream as well, unless a
    console handler on the package logger would print it there already
    (:func:`configure_logging`), in which case it is only logged and appears once.

    ``tqdm.auto`` draws an ``ipywidgets`` widget in a notebook that has the package
    and a text bar otherwise; ``ipywidgets`` is optional and not a dependency.
    ``step`` never calls into tqdm: the redraw happens on the period, so the
    per-step cost is the line reporter's.

    Parameters
    ----------
    period : float, optional
        Seconds between redraws, positive and finite. Default
        :data:`DEFAULT_BAR_PERIOD`. It is also the period of the estimator's window
        rule (``max(5 periods, 10 % of the elapsed time)``).
    file : io.TextIOBase or None, optional
        The stream to draw on; ``None`` is standard error. Tests inject an
        ``io.StringIO``.
    position : int or None, optional
        The bar's line offset, for a parent drawing several (tqdm's ``position``).
    leave : bool, optional
        Whether the finished bar stays on the screen. Default ``True``.
    ncols : int or None, optional
        A fixed width; ``None`` follows the terminal. Tests fix it.
    ascii : bool, str or None, optional
        The ``ascii`` option of tqdm: ``True`` for ``#`` characters, ``None`` for its
        default.
    logger : logging.Logger or None, optional
        See :class:`LogReporter`.
    clock : callable, optional
        See :class:`LogReporter`.
    n_steps : int or None, optional
        See :class:`LogReporter`.
    level : int, optional
        Level of the closing line; the plane line is ``DEBUG`` regardless.
    estimated : bool, optional
        See :class:`LogReporter`.
    echo : bool, optional
        Whether the summary and the closing line are written to the stream when no
        console handler shows them. A parent drawing a bar over something other than
        one run -- a scan's points -- turns it off.

    Raises
    ------
    ValueError
        If the period is not positive and finite.

    Examples
    --------
    >>> import io
    >>> reporter = BarReporter(file=io.StringIO(), ncols=60, ascii=True)
    >>> reporter.started(0.0, 0.1, 2)
    >>> reporter.bar is not None
    True
    >>> reporter.finished(0.0)
    >>> reporter.bar is None
    True
    """

    def __init__(  # noqa: PLR0913 - the bar's own switches, one each, all keyword
        self,
        *,
        period: float = DEFAULT_BAR_PERIOD,
        file: io.TextIOBase | None = None,
        position: int | None = None,
        leave: bool = True,
        ncols: int | None = None,
        ascii: bool | str | None = None,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] = time.perf_counter,
        n_steps: int | None = None,
        level: int = logging.INFO,
        estimated: bool = True,
        echo: bool = True,
    ) -> None:
        if not (math.isfinite(period) and period > 0.0):
            raise ValueError(
                "a progress bar redraws every `period` seconds, which must be "
                f"positive and finite; got {period!r}"
            )
        super().__init__(
            period=period,
            logger=logger,
            clock=clock,
            n_steps=n_steps,
            level=level,
            estimated=estimated,
        )
        self._plane_level = logging.DEBUG
        self._echo = echo
        self._file = file
        self._position = position
        self._leave = leave
        self._ncols = ncols
        self._ascii = ascii
        self._bar: Any = None
        self._swapped: list[tuple[logging.Logger, list[logging.Handler]]] = []

    @property
    def bar(self) -> Any:
        """Return the tqdm instance while the bar is up, else ``None``.

        Returns
        -------
        tqdm.tqdm or None
            The bar, for a parent placing its own beside it.
        """
        return self._bar

    def started(self, z_start: float, z_end: float, n_planes: int) -> None:
        """Open the bar and route console logging above it.

        Parameters
        ----------
        z_start, z_end : float
            See :meth:`ProgressReporter.started`.
        n_planes : int
            See :meth:`ProgressReporter.started`.
        """
        super().started(z_start, z_end, n_planes)
        self._close_bar()
        tqdm = _tqdm_class()
        replay = self._n_total_steps is not None
        total = (
            float(self._n_total_steps)
            if self._n_total_steps is not None
            else max(self._z_end - self._z_start, 0.0)
        )
        self._bar = tqdm(
            total=total,
            unit="steps" if replay else "m",
            file=self._file,
            position=self._position,
            leave=self._leave,
            ncols=self._ncols,
            dynamic_ncols=self._ncols is None,
            ascii=self._ascii,
            mininterval=0.0,
            miniters=1,
            bar_format=BAR_FORMAT,
        )
        self._bar.set_postfix_str("ETA unknown", refresh=False)
        self._bar.refresh()
        self._redirect_logging(tqdm)

    def plane(self, z: float, summary: PlaneSummary) -> None:
        """Count the plane, log it at ``DEBUG`` and name it on the bar.

        Parameters
        ----------
        z : float
            See :meth:`ProgressReporter.plane`.
        summary : PlaneSummary
            See :meth:`ProgressReporter.plane`.
        """
        super().plane(z, summary)
        if self._bar is not None:
            self._bar.set_description_str(
                f"plane {self._planes_done}/{self._n_planes} ", refresh=False
            )

    def finished(self, total: float) -> None:
        """Fill the bar, close it, put the handlers back and log the closing line.

        Parameters
        ----------
        total : float
            See :meth:`ProgressReporter.finished`.
        """
        if self._bar is not None:
            self._bar.n = self._bar.total
            self._bar.set_postfix_str(
                f"{self._n_attempts} steps ({self._n_rejected} rejected), done",
                refresh=False,
            )
            self._bar.refresh()
        self._close_bar()
        super().finished(total)

    def interrupted(self, z: float, error: BaseException) -> None:
        """Leave the bar where it stopped, put the handlers back and log the warning.

        Parameters
        ----------
        z : float
            See :meth:`solprop.log.InterruptibleReporter.interrupted`.
        error : BaseException
            See :meth:`solprop.log.InterruptibleReporter.interrupted`.
        """
        if self._bar is not None:
            self._bar.refresh()
        self._close_bar()
        super().interrupted(z, error)

    def _emit(self, now: float) -> None:
        """Advance the estimator and redraw the bar.

        Parameters
        ----------
        now : float
            The clock reading this redraw happens at, s.
        """
        status = self._status(now)
        if self._bar is None:
            return
        self._bar.n = (
            float(self._n_accepted)
            if self._n_total_steps is not None
            else self._z - self._z_start
        )
        self._bar.set_postfix_str(
            f"{self._n_attempts} steps ({self._n_rejected} rejected), "
            f"dz {self._dz:.1e} m, {self._tail(status)}",
            refresh=False,
        )
        self._bar.refresh()

    def _say(self, level: int, message: str, *args: object) -> None:
        """Log a line a person reads whole, and write it out when nothing else would.

        Parameters
        ----------
        level : int
            The record's level.
        message : str
            The ``%``-format of the record.
        *args : object
            Its arguments.
        """
        super()._say(level, message, *args)
        if self._echo and not self._console_shows(level):
            stream: Any = sys.stderr if self._file is None else self._file
            stream.write(f"{message % args}\n")
            stream.flush()

    def _console_shows(self, level: int) -> bool:
        """Return whether a console handler already prints a record of this level.

        Parameters
        ----------
        level : int
            The record's level.

        Returns
        -------
        bool
            Whether the logger is enabled for it and a handler on its path, at or
            below that level, writes to standard output or error or to the bar's own
            stream.
        """
        if not self._logger.isEnabledFor(level):
            return False
        consoles = [sys.stdout, sys.stderr, self._file]
        logger: logging.Logger | None = self._logger
        while logger is not None:
            for handler in logger.handlers:
                stream = getattr(handler, "stream", getattr(handler, "_stream", None))
                if (
                    handler.level <= level
                    and not isinstance(handler, logging.FileHandler)
                    and stream is not None
                    and any(stream is console for console in consoles)
                ):
                    return True
            logger = logger.parent if logger.propagate else None
        return False

    def _redirect_logging(self, tqdm: Any) -> None:
        """Stand a bar-aware handler in for each console handler, for the bar's life.

        Parameters
        ----------
        tqdm : type
            The class whose ``write`` the stand-ins use.
        """
        consoles = [sys.stdout, sys.stderr]
        if self._file is not None:
            consoles.append(self._file)
        for logger in (logging.getLogger(PACKAGE_LOGGER), logging.getLogger()):
            originals = list(logger.handlers)
            replaced: list[logging.Handler] = []
            changed = False
            for handler in originals:
                if (
                    isinstance(handler, logging.StreamHandler)
                    and not isinstance(handler, logging.FileHandler)
                    and any(handler.stream is console for console in consoles)
                ):
                    replaced.append(_BarHandler(tqdm, handler))
                    changed = True
                else:
                    replaced.append(handler)
            if changed:
                self._swapped.append((logger, originals))
                logger.handlers = replaced

    def _close_bar(self) -> None:
        """Close the bar if one is up and restore the handlers it displaced."""
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        for logger, originals in reversed(self._swapped):
            logger.handlers = originals
        self._swapped.clear()


def default_reporter(
    *,
    choice: ProgressChoice | None = None,
    period: float = DEFAULT_PROGRESS_PERIOD,
    n_steps: int | None = None,
    stream: io.TextIOBase | None = None,
    estimated: bool = True,
) -> ProgressReporter:
    """Build the reporter a progress choice means.

    Parameters
    ----------
    choice : {'auto', 'bar', 'log', 'none'} or None, optional
        The choice; ``None`` takes :func:`default_progress`.
    period : float, optional
        The line reporter's period, s; a bar has its own redraw period.
    n_steps : int or None, optional
        The step count of a replayed schedule, for the ETA.
    stream : io.TextIOBase or None, optional
        Where a bar would draw, and what ``"auto"`` tests; ``None`` is standard error.
    estimated : bool, optional
        Whether the steps carry an error estimate (:class:`LogReporter`).

    Returns
    -------
    ProgressReporter
        A :class:`BarReporter`, a :class:`LogReporter` or a :class:`NullReporter`.

    Raises
    ------
    ValueError
        If the choice is not one of the four names.

    Examples
    --------
    >>> import io
    >>> from solprop.log import LogReporter, default_reporter
    >>> isinstance(default_reporter(choice="auto", stream=io.StringIO()), LogReporter)
    True
    """
    selected = default_progress() if choice is None else choice
    if selected not in _PROGRESS_CHOICES:
        raise ValueError(
            f"progress must be one of {_PROGRESS_CHOICES} or a reporter, got "
            f"{selected!r}"
        )
    if selected == "auto":
        selected = "bar" if is_interactive(stream) else "log"
        if selected == "bar" and not _tqdm_available():
            # An environment that has solprop but not its newest dependency: say so
            # once and log lines rather than fail the run. An explicit "bar" still
            # raises, because it was asked for.
            warnings.warn(
                f"{TQDM_MISSING}; logging progress lines instead",
                RuntimeWarning,
                stacklevel=2,
            )
            selected = "log"
    if selected == "none":
        return NullReporter()
    if selected == "log":
        return LogReporter(period=period, n_steps=n_steps, estimated=estimated)
    return BarReporter(n_steps=n_steps, file=stream, estimated=estimated)


def resolve_reporter(
    progress: ProgressReporter | ProgressChoice | None,
    *,
    period: float = DEFAULT_PROGRESS_PERIOD,
    n_steps: int | None = None,
    estimated: bool = True,
) -> ProgressReporter:
    """Turn a ``progress=`` argument into a reporter.

    Parameters
    ----------
    progress : ProgressReporter, {'auto', 'bar', 'log', 'none'} or None
        A reporter, which is returned as given, or a choice; ``None`` takes the
        process default.
    period : float, optional
        See :func:`default_reporter`.
    n_steps : int or None, optional
        See :func:`default_reporter`.
    estimated : bool, optional
        See :func:`default_reporter`.

    Returns
    -------
    ProgressReporter
        The reporter to attach to the backend.
    """
    if progress is None or isinstance(progress, str):
        return default_reporter(
            choice=progress, period=period, n_steps=n_steps, estimated=estimated
        )
    return progress
