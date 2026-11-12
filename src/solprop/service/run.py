# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run's job: build, run, continue or scan, observe, and record (design 13.4).

Every run is a job of its own (:mod:`solprop.service.job`), on this computer or
another, and streams to nothing but its record (:mod:`solprop.service.record`): the
window follows the record. The march runs in the main thread, and its plane observer
(:class:`LiveObserver`) only reduces each plane and hands the row to the record, whose
own thread writes it. The run's stop file sets its stop token
(:class:`solprop.backends.session.StopToken`), which reaches the session.

Whether a run *stopped* comes from the worker's own flag and the run's clean return,
never from solprop's outcome, which records a stop as a failure.

This module imports solprop and runs only in a worker process.
"""

import logging
import threading
import time
import traceback
import warnings
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np

from solprop.backends.session import StopToken
from solprop.driver.executors import Pool
from solprop.driver.scan import run_scan
from solprop.exceptions import RunTerminated
from solprop.grid import TimeGrid
from solprop.log import NullReporter, ScanReporter
from solprop.output import open_file
from solprop.recipe import Recipe, resume_file
from solprop.service.beam import beam_reading, profile_row
from solprop.service.messages import Event
from solprop.service.record import (
    EventLog,
    Job,
    clear_stop,
    set_aside,
    watch_stop,
    write_job,
)
from solprop.service.reduce import (
    SPECTRAL_AXES,
    TIME_OVERSAMPLING,
    Display,
    Reducer,
    stats_row,
)
from solprop.service.scan_reductions import reductions
from solprop.service.worker import LOG, OUTPUT, RECIPE, SCAN_OUTPUT, RunSpec

__all__ = ["PROGRESS_PERIOD", "EventHandler", "LiveObserver", "ScanObserver", "execute"]

PROGRESS_PERIOD = 0.25
"""The least time between two progress events, s: at most four a second."""


class LiveObserver(NullReporter):
    """The run's reporter: progress on a period, and each plane reduced to a row.

    Parameters
    ----------
    run_id : str
        The run.
    reducer : solprop.service.reduce.Reducer
        The display's bins.
    emit : Callable
        Where its events go: the run's record.

    Attributes
    ----------
    axes : Event or None
        The axes event, once the first plane has fixed the bins.
    """

    def __init__(
        self, run_id: str, reducer: Reducer, emit: Callable[[Event], None]
    ) -> None:
        """Start with no planes seen."""
        self.run_id = run_id
        self.reducer = reducer
        self.emit = emit
        self.axes: Event | None = None
        self._steps = 0
        self._rejected = 0
        self._z_end = float("nan")
        self._last = 0.0
        self.space: tuple[TimeGrid, object, object] | None = None

    def measure(self, simulation: object) -> None:
        """Measure a free-space run's planes with its grids and medium.

        Parameters
        ----------
        simulation : solprop.driver.Simulation
            The run; one without a transverse grid is measured by its modes.
        """
        space = getattr(simulation, "spatial_grid", None)
        if space is not None:
            grid = simulation.grid  # pyright: ignore[reportAttributeAccessIssue]
            self.space = (grid, space, simulation.medium)  # pyright: ignore[reportAttributeAccessIssue]

    def _put(self, kind: str, **data: object) -> None:
        """Hand one event to the record."""
        self.emit(Event(self.run_id, kind, data))

    def started(self, z_start: float, z_end: float, n_planes: int) -> None:
        """Report the run's span.

        Parameters
        ----------
        z_start : float
            Where it starts, m.
        z_end : float
            Where it ends, m.
        n_planes : int
            The planes it will save.
        """
        self._z_end = z_end
        self._put("started", z_start=z_start, z_end=z_end, n_planes=n_planes)

    def step(self, z: float, dz: float, accepted: bool) -> None:
        """Count a step, and report progress at most every :data:`PROGRESS_PERIOD`.

        Parameters
        ----------
        z : float
            Where the attempt ended, m.
        dz : float
            Its length, m.
        accepted : bool
            Whether it was kept.
        """
        self._steps += 1
        if not accepted:
            self._rejected += 1
            return
        now = time.monotonic()
        if now - self._last >= PROGRESS_PERIOD:
            self._last = now
            self._put(
                "progress",
                z=z,
                dz=dz,
                z_end=self._z_end,
                steps=self._steps,
                rejected=self._rejected,
            )

    def observe_plane(self, z: float, plane: object) -> None:
        """Reduce one saved plane to a row and hand it over.

        Parameters
        ----------
        z : float
            Its position, m.
        plane : solprop.driver.plane_view.PlaneView
            The lazy view of it, valid only during the call.
        """
        bandpass = self.reducer.display.bandpass
        profile = None
        if plane.basis == "spatial":  # pyright: ignore[reportAttributeAccessIssue]
            if self.space is None:
                return  # a continued run: its file is read when it ends
            reading = beam_reading(
                *self.space,
                z,
                plane.stored(),  # pyright: ignore[reportAttributeAccessIssue]
                np.asarray(plane.field_plane().spectral_energy()),  # pyright: ignore[reportAttributeAccessIssue]
                bandpass=bandpass,
            )
            spectra = reading["spectra"]
            t, power = reading["t"], reading["power"]
            profile = profile_row(reading)
        else:
            spectra = {
                axis: plane.spectral_density(axis=axis, modes="sum")  # pyright: ignore[reportAttributeAccessIssue]
                for axis in SPECTRAL_AXES
            }
            t, power = plane.power(  # pyright: ignore[reportAttributeAccessIssue]
                modes="sum", bandpass=bandpass, oversampling=TIME_OVERSAMPLING
            )  # pyright: ignore[reportAttributeAccessIssue]
        reduced_density, reduced_power = self.reducer.reduce(spectra, t, power)  # pyright: ignore[reportArgumentType]
        if self.axes is None:
            self.axes = Event(self.run_id, "axes", dict(self.reducer.axes()))
            self.emit(self.axes)
        self.emit(
            Event(
                self.run_id,
                "row",
                {
                    "index": int(plane.index),  # pyright: ignore[reportAttributeAccessIssue]
                    "z": float(z),
                    "density": reduced_density,
                    "power": reduced_power,
                    "stats": stats_row(plane.stats),  # pyright: ignore[reportAttributeAccessIssue]
                    "steps": self._steps,
                    "rejected": self._rejected,
                    **({} if profile is None else {"profile": profile}),
                },
            )
        )


class ScanObserver(ScanReporter):
    """Turn a scan's progress into events: its start, and each point's start and end.

    Parameters
    ----------
    run_id : str
        The scan's identity.
    emit : Callable
        Where its events go: the run's record.
    """

    def __init__(self, run_id: str, emit: Callable[[Event], None]) -> None:
        """Hold where the events go."""
        self.run_id = run_id
        self.emit = emit
        self.points = 0
        self.done = 0
        self.shape: tuple[int, ...] = ()
        self.names: list[str] = []
        self.values: list[list[str]] = []

    def scan_started(self, n_points: int) -> None:
        """Announce the scan and its number of points.

        Parameters
        ----------
        n_points : int
            The points to run.
        """
        self.points = n_points
        self.emit(
            Event(
                self.run_id,
                "scan_started",
                {
                    "points": n_points,
                    "shape": list(self.shape),
                    "names": self.names,
                    "values": self.values,
                },
            )
        )

    def point_started(self, index: int, point: object) -> None:
        """Announce a point, with its values.

        Parameters
        ----------
        index : int
            Its global index.
        point : object
            The scan's point (``ScanPoint``), whose ``values`` are given in SI.
        """
        values = getattr(point, "values", {})
        # A keyword's value is read in SI; a nested path's is the quantity as written.
        plain = {str(k): _plain(v) for k, v in dict(values).items()}  # pyright: ignore[reportUnknownArgumentType]
        place = [int(i) for i in getattr(point, "multi_index", ())]
        self.emit(
            Event(
                self.run_id,
                "point",
                {"index": index, "values": plain, "status": None, "place": place},
            )
        )

    def point_finished(self, index: int, status: str, wall: float) -> None:
        """Announce a point's end, and the scan's share done.

        Parameters
        ----------
        index : int
            Its global index.
        status : str
            ``"done"`` or ``"failed"``.
        wall : float
            Its wall time, s.
        """
        self.done += 1
        self.emit(
            Event(
                self.run_id,
                "point",
                {
                    "index": index,
                    "status": status,
                    "wall": wall,
                    "done": self.done,
                    "points": self.points,
                },
            )
        )

    def scan_finished(self) -> None:
        """Do nothing: the scan's terminal event is the worker's."""


class EventHandler(logging.Handler):
    """Forward log records to the run's record as ``log`` events.

    Parameters
    ----------
    run_id : str
        The run.
    send : Callable
        Where the events go: the run's record.
    """

    def __init__(self, run_id: str, send: Callable[[Event], None]) -> None:
        """Forward records of every level from here on."""
        super().__init__()
        self.run_id = run_id
        self.send = send

    def emit(self, record: logging.LogRecord) -> None:
        """Send one record.

        Parameters
        ----------
        record : logging.LogRecord
            The record.
        """
        self.send(
            Event(
                self.run_id,
                "log",
                {
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                },
            )
        )


def _ending(output: Path, stopped: bool) -> tuple[str, dict[str, object]]:
    """Return the terminal event of a run that returned.

    - Returns: the kind (``"stopped"``, ``"finished"`` or ``"failed"``) and its data:
      the outcome's ``status``, ``z`` and ``message``, and the file's ``path``.
    """
    with open_file(output) as handler:
        outcome = handler.outcome
    data: dict[str, object] = {
        "path": str(output),
        "status": None if outcome is None else outcome.status,
        "z": None if outcome is None else outcome.z,
        "message": "" if outcome is None else outcome.message,
    }
    if outcome is not None and outcome.status == "complete":
        # A stop that came as the run reached its end found nothing left to stop.
        return "finished", data
    if stopped:
        return "stopped", data
    return "failed", data


def _plain(value: object) -> float | str:
    """Return a point's value as an event carries it.

    - ``value``: a number, or a quantity as a recipe writes it (``"0.8 bar"``).
    """
    try:
        return float(value)  # pyright: ignore[reportArgumentType]
    except TypeError, ValueError:
        return str(value)


def _written(recipe: Recipe, variables: Mapping[str, object]) -> list[list[str]]:
    """Return each scan variable's values as the recipe's ``[scan]`` table writes them.

    - ``variables``: the scan's variables and their values, in its axis order.
    """
    table: Mapping[str, object] = recipe.to_dict().get("scan", {})
    written: list[list[str]] = []
    for name, values in variables.items():
        given = table.get(name)
        if isinstance(given, list):
            written.append([str(value) for value in given])
        else:
            written.append([f"{float(v):.4g}" for v in np.asarray(values).ravel()])
    return written


def _scan(
    recipe: Recipe,
    folder: Path,
    spec: RunSpec,
    emit: Callable[[Event], None],
    stop: StopToken,
) -> tuple[str, dict[str, object]]:
    """Run a recipe's ``[scan]`` (``Recipe.scan`` and ``run_scan``) into its folder.

    Its points run in a :class:`~solprop.driver.executors.Pool` that sizes itself to
    the CPUs and memory the run was given: this computer's, a machine's or a Slurm
    allocation's (design 13.4).

    - Returns: the terminal event: ``"stopped"`` or ``"finished"``, with the scan file's
      ``path`` and the number of points ``failed``.
    """
    scan = recipe.scan()
    output = folder / SCAN_OUTPUT
    observer = ScanObserver(spec.run_id, emit)
    observer.shape = tuple(int(n) for n in scan.scan.shape)
    observer.names = list(scan.scan.variables)
    observer.values = _written(recipe, scan.scan.variables)
    result = run_scan(
        scan.scan,
        scan.build,
        output=output,
        progress=observer,
        stop=stop,
        reductions=reductions(spec.reductions, recipe),
        executor=Pool(),
        **scan.keywords,  # pyright: ignore[reportArgumentType]
    )
    failed = int(np.count_nonzero(np.asarray(result.status) == "failed"))
    kind = "stopped" if stop.is_set() else "finished"
    return kind, {"path": str(output), "scan": True, "failed": failed, "message": ""}


def _display(spec: RunSpec, output: Path) -> Display:
    """Return the display bins of a run: the spec's, in the recipe's window.

    - Returns: the display; the spec's own where the recipe gives no wavelength window
      or cannot be read.
    """
    try:
        if spec.recipe is not None:
            base = Path(spec.base) if spec.base else None
            recipe = Recipe.loads(spec.recipe, base=base)
        else:
            recipe = Recipe.from_output(output, resolved=True)
        return spec.display.within(recipe.keywords())
    except Exception:
        return spec.display


def execute(spec: RunSpec) -> None:
    """Run, or continue, one run, recording its events until its terminal one.

    The run appends its events to its record and obeys its stop file
    (:mod:`solprop.service.record`). A signal that ends the run (``SIGTERM``, as Slurm
    sends at a time limit or ``scancel``, and as Kill sends) leaves the file resumable,
    so it ends the run as stopped. A run tied to a window (``spec.owner``) stops, in
    the same way as Stop, when that window's process has gone.

    Parameters
    ----------
    spec : solprop.service.worker.RunSpec
        What to run.
    """
    folder = Path(spec.folder)
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / OUTPUT
    stop = StopToken()
    done = threading.Event()
    if spec.resume is not None:
        set_aside(folder)  # as the launcher did, for a run continued by hand
    clear_stop(folder)
    owner = None if spec.owner is None else (int(spec.owner[0]), spec.owner[1])
    watch_stop(folder, stop, done, owner=owner)
    record = EventLog(
        folder,
        spec.run_id,
        job=lambda: Job.here(spec.run_id, spec.launcher, spec.threads),
    )
    observer = LiveObserver(spec.run_id, Reducer(_display(spec, output)), record.put)
    detach = _attach_log(folder, spec.run_id, record.put)

    kind = "failed"
    data: dict[str, object] = {"path": str(output)}
    try:
        kind, data = _march(spec, folder, record.put, observer, stop)
    except RunTerminated as error:
        # solprop wrote the state a resume needs before raising.
        kind, data = "stopped", {"path": str(output), "terminated": True}
        data["message"] = str(error) or "ended by a signal"
    except BaseException as error:
        data = {
            "path": str(output),
            "message": str(error) or type(error).__name__,
            "detail": traceback.format_exc(),
        }
    finally:
        record.put(Event(spec.run_id, kind, data))
        done.set()
        record.close()
        if record.job is not None:
            write_job(folder, replace(record.job, ended=time.time()))
        detach()


def _attach_log(
    folder: Path, run_id: str, emit: Callable[[Event], None]
) -> Callable[[], None]:
    """Send solprop's log to the run's ``run.log``, and its warnings as events."""
    log_file = logging.FileHandler(folder / LOG, encoding="utf-8")
    log_file.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    forward = EventHandler(run_id, emit)
    forward.setLevel(logging.WARNING)
    solprop_logger = logging.getLogger("solprop")
    level = solprop_logger.level
    # Warnings go to logging while the run goes; whoever had them before gets them back.
    before = warnings.showwarning
    logging.captureWarnings(True)
    captured = warnings.showwarning is not before
    loggers = [solprop_logger, logging.getLogger("py.warnings")]
    for logger in loggers:
        logger.addHandler(log_file)
        logger.addHandler(forward)
    solprop_logger.setLevel(logging.INFO)

    def detach() -> None:
        """Take the handlers off, close the log, and leave logging as it was."""
        for logger in loggers:
            logger.removeHandler(log_file)
            logger.removeHandler(forward)
        log_file.close()
        solprop_logger.setLevel(level)
        if captured:
            logging.captureWarnings(False)

    return detach


def _march(
    spec: RunSpec,
    folder: Path,
    emit: Callable[[Event], None],
    observer: LiveObserver,
    stop: StopToken,
) -> tuple[str, dict[str, object]]:
    """Run, continue or scan, and return the terminal event.

    - Returns: the terminal event's kind and data.
    """
    output = folder / OUTPUT
    if spec.resume is None:
        if spec.recipe is None:
            msg = "a new run needs its recipe"
            raise ValueError(msg)
        recipe = Recipe.loads(spec.recipe, base=Path(spec.base) if spec.base else None)
        (folder / RECIPE).write_text(spec.recipe, encoding="utf-8")
        if "scan" in recipe.to_dict():
            return _scan(recipe, folder, spec, emit, stop)
        # Built here and kept: a free-space run's planes are measured with its
        # grids and medium; run as Recipe.run would.
        simulation = recipe.build()
        observer.measure(simulation)
        simulation.run(output, progress=observer, stop=stop)
    else:
        resume_file(output, spec.resume, progress=observer, stop=stop)  # pyright: ignore[reportArgumentType]
    return _ending(output, stop.is_set())
