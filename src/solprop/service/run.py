# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The worker's run: build, run or resume, observe, and stream (design 13.3).

Three threads share the worker. The march runs in the main thread, and its plane
observer (:class:`LiveObserver`) only reduces each plane and hands the row to an
:class:`~solprop.service.outbox.Outbox`. A sender thread writes the outbox to the
events pipe; a broken pipe marks the window disconnected and never touches the run. A
control thread sets solprop's stop flag on ``stop``, and replays the rows kept in
memory on ``subscribe``.

Whether a run *stopped* comes from the worker's own flag and the run's clean return,
never from solprop's outcome, which records a stop as a failure.

This module imports solprop and runs only in a worker process.
"""

import logging
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np

from solprop.driver.scan import run_scan
from solprop.exceptions import RunTerminated
from solprop.grid import TimeGrid
from solprop.log import NullReporter, ScanReporter
from solprop.output import open_file
from solprop.recipe import Recipe, resume_file
from solprop.service.beam import beam_reading, profile_row
from solprop.service.messages import Control, Event
from solprop.service.outbox import Outbox
from solprop.service.record import EventLog, Job, clear_stop, watch_stop, write_job
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
    outbox : solprop.service.outbox.Outbox
        Where its events go.

    Attributes
    ----------
    rows : list of Event
        Every row event made, in plane order, kept for replay.
    axes : Event or None
        The axes event, once the first plane has fixed the bins.
    """

    def __init__(self, run_id: str, reducer: Reducer, outbox: Outbox) -> None:
        """Start with no planes seen."""
        self.run_id = run_id
        self.reducer = reducer
        self.outbox = outbox
        self.rows: list[Event] = []
        self.axes: Event | None = None
        self._lock = threading.Lock()
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
        """Hand one event to the outbox.

        Parameters
        ----------
        kind : str
            Its kind.
        **data : object
            Its content.
        """
        self.outbox.put(Event(self.run_id, kind, data))

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
        with self._lock:
            if self.axes is None:
                self.axes = Event(self.run_id, "axes", dict(self.reducer.axes()))
                self.outbox.put(self.axes)
            row = Event(
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
            self.rows.append(row)
        self.outbox.put(row)

    def replay(self, from_plane: int) -> None:
        """Hand over the axes and every row kept from a plane on.

        Parameters
        ----------
        from_plane : int
            The first plane index to replay.
        """
        with self._lock:
            axes = self.axes
            rows = [row for row in self.rows if int(row.data["index"]) >= from_plane]  # pyright: ignore[reportArgumentType]
        if axes is not None:
            self.outbox.put(axes, copy=False)
        for row in rows:
            self.outbox.put(row, copy=False)


class ScanObserver(ScanReporter):
    """Turn a scan's progress into events: its start, and each point's start and end.

    Parameters
    ----------
    run_id : str
        The scan's identity.
    outbox : solprop.service.outbox.Outbox
        Where its events go.
    """

    def __init__(self, run_id: str, outbox: Outbox) -> None:
        """Hold where the events go."""
        self.run_id = run_id
        self.outbox = outbox
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
        self.outbox.put(
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
        self.outbox.put(
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
        self.outbox.put(
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
    """Forward log records to the window as ``log`` events.

    Parameters
    ----------
    run_id : str
        The run.
    outbox : solprop.service.outbox.Outbox
        Where the events go.
    """

    def __init__(self, run_id: str, outbox: Outbox) -> None:
        """Forward records of every level from here on."""
        super().__init__()
        self.run_id = run_id
        self.outbox = outbox

    def emit(self, record: logging.LogRecord) -> None:
        """Send one record.

        Parameters
        ----------
        record : logging.LogRecord
            The record.
        """
        self.outbox.put(
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


def _send(outbox: Outbox, events: Connection) -> None:
    """Write the outbox to the pipe until it is closed and empty.

    A broken pipe means the window has gone: the events are then taken and
    discarded, so the run never waits on it.

    Parameters
    ----------
    outbox : solprop.service.outbox.Outbox
        The run's events.
    events : multiprocessing.connection.Connection
        The pipe to the window.
    """
    connected = True
    while True:
        event = outbox.get(timeout=0.5)
        if event is None:
            if outbox.closed and not len(outbox):
                return
            continue
        if not connected:
            continue
        try:
            events.send(event)
        except BrokenPipeError, EOFError, OSError:
            connected = False


def _listen(control: Connection, stop: threading.Event, observer: LiveObserver) -> None:
    """Act on the window's control requests until it closes the pipe.

    Parameters
    ----------
    control : multiprocessing.connection.Connection
        The pipe from the window.
    stop : threading.Event
        The run's stop flag, solprop's.
    observer : LiveObserver
        The run's observer, whose rows a subscription replays.
    """
    while True:
        try:
            request: Control = control.recv()
        except EOFError, OSError:
            return
        if request.kind == "stop":
            stop.set()
        else:  # subscribe
            observer.replay(request.from_plane)


def _ending(output: Path, stopped: bool) -> tuple[str, dict[str, object]]:
    """Return the terminal event of a run that returned.

    Parameters
    ----------
    output : pathlib.Path
        Its results file.
    stopped : bool
        Whether its stop flag was set.

    Returns
    -------
    tuple
        The kind (``"stopped"``, ``"finished"`` or ``"failed"``) and its data: the
        outcome's ``status``, ``z`` and ``message``, and the file's ``path``.
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

    Parameters
    ----------
    value : object
        A number, or a quantity as a recipe writes it (``"0.8 bar"``).

    Returns
    -------
    float or str
        The number, or the text.

    Examples
    --------
    >>> _plain(np.float64(0.5)), _plain("0.8 bar")
    (0.5, '0.8 bar')
    """
    try:
        return float(value)  # pyright: ignore[reportArgumentType]
    except TypeError, ValueError:
        return str(value)


def _written(recipe: Recipe, variables: Mapping[str, object]) -> list[list[str]]:
    """Return each scan variable's values as the recipe's ``[scan]`` table writes them.

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        The recipe.
    variables : Mapping
        The scan's variables and their values, in its axis order.

    Returns
    -------
    list of list of str
        A list given as quantities keeps its units; a range's values are numbers in
        its keyword's unit.
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
    run_id: str,
    outbox: Outbox,
    stop: threading.Event,
    specs: Sequence[Mapping[str, object]] = (),
) -> tuple[str, dict[str, object]]:
    """Run a recipe's ``[scan]`` (``Recipe.scan`` and ``run_scan``) into its folder.

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        The recipe, with its ``[scan]`` table.
    folder : pathlib.Path
        The scan's folder.
    run_id : str
        Its identity, for its events.
    outbox : solprop.service.outbox.Outbox
        Where its events go.
    stop : threading.Event
        Its stop flag: the points not yet started are not run.
    specs : Sequence of Mapping, optional
        What it records at each point beyond its statistics.

    Returns
    -------
    tuple
        The terminal event: ``"stopped"`` or ``"finished"``, with the scan file's
        ``path`` and the number of points ``failed``.
    """
    scan = recipe.scan()
    output = folder / SCAN_OUTPUT
    observer = ScanObserver(run_id, outbox)
    observer.shape = tuple(int(n) for n in scan.scan.shape)
    observer.names = list(scan.scan.variables)
    observer.values = _written(recipe, scan.scan.variables)
    result = run_scan(
        scan.scan,
        scan.build,
        output=output,
        progress=observer,
        stop=stop,
        reductions=reductions(specs, recipe),
        **scan.keywords,  # pyright: ignore[reportArgumentType]
    )
    failed = int(np.count_nonzero(np.asarray(result.status) == "failed"))
    kind = "stopped" if stop.is_set() else "finished"
    return kind, {"path": str(output), "scan": True, "failed": failed, "message": ""}


def _display(spec: RunSpec, output: Path) -> Display:
    """Return the display bins of a run: the spec's, in the recipe's window.

    Parameters
    ----------
    spec : solprop.service.worker.RunSpec
        The run.
    output : pathlib.Path
        Its results file, whose recipe a continued run reads.

    Returns
    -------
    solprop.service.reduce.Display
        The display; the spec's own where the recipe gives no wavelength window or
        cannot be read.
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


def execute(
    spec: RunSpec, events: Connection | None = None, control: Connection | None = None
) -> None:
    """Run, or continue, one run and stream its events until its terminal one.

    A run with a record (``spec.record``) also appends its events to its folder and
    obeys its stop file; a run started with no window (:mod:`solprop.service.job`)
    has no pipes, and its record is all it streams to. A signal that ends the run
    (``SIGTERM``, as Slurm sends at a time limit or ``scancel``) leaves the file
    resumable, so it ends the run as stopped.

    Parameters
    ----------
    spec : solprop.service.worker.RunSpec
        What to run.
    events : multiprocessing.connection.Connection or None, optional
        Where the events go; ``None`` for none.
    control : multiprocessing.connection.Connection or None, optional
        Where stop and subscribe requests come from; ``None`` for none.
    """
    folder = Path(spec.folder)
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / OUTPUT
    stop = threading.Event()
    done = threading.Event()
    record, job = _open_record(spec, folder, stop, done)
    outbox = Outbox(copy_to=None if record is None else record.put)
    observer = LiveObserver(spec.run_id, Reducer(_display(spec, output)), outbox)
    sender: threading.Thread | None = None
    if events is not None:
        sender = threading.Thread(target=_send, args=(outbox, events), daemon=True)
        sender.start()
    if control is not None:
        threading.Thread(
            target=_listen, args=(control, stop, observer), daemon=True
        ).start()
    detach = _attach_log(folder, spec.run_id, outbox)

    kind = "failed"
    data: dict[str, object] = {"path": str(output)}
    try:
        kind, data = _march(spec, folder, outbox, observer, stop)
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
        outbox.put(Event(spec.run_id, kind, data))
        outbox.close()
        done.set()
        if record is not None and job is not None:
            record.close()
            write_job(folder, replace(job, ended=time.time()))
        if sender is not None:
            sender.join(timeout=10.0)
        detach()


def _open_record(
    spec: RunSpec, folder: Path, stop: threading.Event, done: threading.Event
) -> tuple[EventLog | None, Job | None]:
    """Start a run's record, when its spec asks for one.

    Parameters
    ----------
    spec : solprop.service.worker.RunSpec
        The run.
    folder : pathlib.Path
        Its folder.
    stop : threading.Event
        Its stop flag, which its stop file sets.
    done : threading.Event
        Set when it ends.

    Returns
    -------
    tuple
        Its record and its ``job.json``; both ``None`` for a run without a record.
    """
    if not spec.record:
        return None, None
    clear_stop(folder)
    job = Job.here(spec.run_id, spec.launcher, spec.threads)
    write_job(folder, job)
    watch_stop(folder, stop, done)
    return EventLog(folder, spec.run_id), job


def _attach_log(folder: Path, run_id: str, outbox: Outbox) -> Callable[[], None]:
    """Send solprop's log to the run's ``run.log``, and its warnings as events.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder.
    run_id : str
        The run.
    outbox : solprop.service.outbox.Outbox
        Where its events go.

    Returns
    -------
    Callable
        What takes the handlers off again and closes the log.
    """
    log_file = logging.FileHandler(folder / LOG, encoding="utf-8")
    log_file.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    forward = EventHandler(run_id, outbox)
    forward.setLevel(logging.WARNING)
    logging.captureWarnings(True)
    loggers = [logging.getLogger("solprop"), logging.getLogger("py.warnings")]
    for logger in loggers:
        logger.addHandler(log_file)
        logger.addHandler(forward)
    logging.getLogger("solprop").setLevel(logging.INFO)

    def detach() -> None:
        for logger in loggers:
            logger.removeHandler(log_file)
            logger.removeHandler(forward)
        log_file.close()

    return detach


def _march(
    spec: RunSpec,
    folder: Path,
    outbox: Outbox,
    observer: LiveObserver,
    stop: threading.Event,
) -> tuple[str, dict[str, object]]:
    """Run, continue or scan, and return the terminal event.

    Parameters
    ----------
    spec : solprop.service.worker.RunSpec
        The run.
    folder : pathlib.Path
        Its folder.
    outbox : solprop.service.outbox.Outbox
        Where a scan's events go.
    observer : LiveObserver
        The run's reporter.
    stop : threading.Event
        Its stop flag.

    Returns
    -------
    tuple
        The terminal event's kind and data.

    Raises
    ------
    ValueError
        For a new run without its recipe.
    """
    output = folder / OUTPUT
    if spec.resume is None:
        if spec.recipe is None:
            msg = "a new run needs its recipe"
            raise ValueError(msg)
        recipe = Recipe.loads(spec.recipe, base=Path(spec.base) if spec.base else None)
        (folder / RECIPE).write_text(spec.recipe, encoding="utf-8")
        if "scan" in recipe.to_dict():
            return _scan(recipe, folder, spec.run_id, outbox, stop, spec.reductions)
        # Built here and kept: a free-space run's planes are measured with its
        # grids and medium; run as Recipe.run would.
        simulation = recipe.build()
        observer.measure(simulation)
        simulation.run(output, progress=observer, stop=stop)
    else:
        resume_file(output, spec.resume, progress=observer, stop=stop)  # pyright: ignore[reportArgumentType]
    return _ending(output, stop.is_set())
