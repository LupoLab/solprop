# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The worker's run: build, run or resume, observe, and stream (design 13.3).

Three threads share the worker. The march runs in the main thread, and its plane
observer (:class:`LiveObserver`) only reduces each plane and hands the row to an
:class:`~solprop_gui.service.outbox.Outbox`. A sender thread writes the outbox to the
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
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
from solprop.log import NullReporter
from solprop.output import open_file
from solprop.recipe import Recipe, resume_file

from solprop_gui.service.messages import Control, Event
from solprop_gui.service.outbox import Outbox
from solprop_gui.service.reduce import Display, Reducer, stats_row
from solprop_gui.service.worker import LOG, OUTPUT, RECIPE, RunSpec

__all__ = ["PROGRESS_PERIOD", "EventHandler", "LiveObserver", "execute"]

PROGRESS_PERIOD = 0.25
"""The least time between two progress events, s: at most four a second."""


class LiveObserver(NullReporter):
    """The run's reporter: progress on a period, and each plane reduced to a row.

    Parameters
    ----------
    run_id : str
        The run.
    reducer : solprop_gui.service.reduce.Reducer
        The display's bins.
    outbox : solprop_gui.service.outbox.Outbox
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
        lam, density = plane.spectral_density(axis="wavelength", modes="sum")  # pyright: ignore[reportAttributeAccessIssue]
        t, power = plane.power(modes="sum")  # pyright: ignore[reportAttributeAccessIssue]
        reduced_density, reduced_power = self.reducer.reduce(
            np.asarray(lam), np.asarray(density), np.asarray(t), np.asarray(power)
        )
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
            self.outbox.put(axes)
        for row in rows:
            self.outbox.put(row)


class EventHandler(logging.Handler):
    """Forward log records to the window as ``log`` events.

    Parameters
    ----------
    run_id : str
        The run.
    outbox : solprop_gui.service.outbox.Outbox
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
    outbox : solprop_gui.service.outbox.Outbox
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


def _display(spec: RunSpec, output: Path) -> Display:
    """Return the display bins of a run: the spec's, in the recipe's window.

    Parameters
    ----------
    spec : solprop_gui.service.worker.RunSpec
        The run.
    output : pathlib.Path
        Its results file, whose recipe a continued run reads.

    Returns
    -------
    solprop_gui.service.reduce.Display
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


def execute(spec: RunSpec, events: Connection, control: Connection) -> None:
    """Run, or continue, one run and stream its events until its terminal one.

    Parameters
    ----------
    spec : solprop_gui.service.worker.RunSpec
        What to run.
    events : multiprocessing.connection.Connection
        Where the events go.
    control : multiprocessing.connection.Connection
        Where stop and subscribe requests come from.
    """
    folder = Path(spec.folder)
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / OUTPUT
    outbox = Outbox()
    stop = threading.Event()
    observer = LiveObserver(spec.run_id, Reducer(_display(spec, output)), outbox)
    sender = threading.Thread(target=_send, args=(outbox, events), daemon=True)
    sender.start()
    threading.Thread(
        target=_listen, args=(control, stop, observer), daemon=True
    ).start()

    log_file = logging.FileHandler(folder / LOG, encoding="utf-8")
    log_file.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    forward = EventHandler(spec.run_id, outbox)
    forward.setLevel(logging.WARNING)
    logging.captureWarnings(True)
    loggers = [logging.getLogger("solprop"), logging.getLogger("py.warnings")]
    for logger in loggers:
        logger.addHandler(log_file)
        logger.addHandler(forward)
    logging.getLogger("solprop").setLevel(logging.INFO)

    kind = "failed"
    data: dict[str, object] = {"path": str(output)}
    try:
        if spec.resume is None:
            if spec.recipe is None:
                msg = "a new run needs its recipe"
                raise ValueError(msg)
            recipe = Recipe.loads(
                spec.recipe, base=Path(spec.base) if spec.base else None
            )
            (folder / RECIPE).write_text(spec.recipe, encoding="utf-8")
            recipe.run(output, progress=observer, stop=stop)
        else:
            resume_file(output, spec.resume, progress=observer, stop=stop)  # pyright: ignore[reportArgumentType]
        kind, data = _ending(output, stop.is_set())
    except BaseException as error:
        data = {
            "path": str(output),
            "message": str(error) or type(error).__name__,
            "detail": traceback.format_exc(),
        }
    finally:
        outbox.put(Event(spec.run_id, kind, data))
        outbox.close()
        sender.join(timeout=10.0)
        for logger in loggers:
            logger.removeHandler(log_file)
            logger.removeHandler(forward)
        log_file.close()
