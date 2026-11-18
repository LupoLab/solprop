# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run's record in its folder: what a run says about itself (design 13.4).

Beside ``recipe.toml``, ``run.h5`` and ``run.log`` a run writes ``job.json`` (how it
was started: :class:`Job`, :func:`write_job`, :func:`read_job`), ``events.bin`` (every
event it streams, appended as frames by :class:`EventLog` and read back by
:func:`read_events`) and ``status.json`` (a small summary, :func:`read_status`), and it
watches a ``stop`` file (:func:`request_stop`, :func:`stop_requested`,
:func:`watch_stop`, :func:`clear_stop`). So a run needs no connection: whoever can read
its folder can follow it, whoever can write there can stop it, and its state is read
from these files, never kept by a daemon. :func:`set_aside` keeps a continued run's
earlier record apart, :func:`write_json` replaces a file whole, and :func:`alive` says
whether a run's process still lives.

**The ending is recorded last**: the terminal event, then ``job.json``'s end, and only
then ``status.json``, after which the run writes nothing more, so a reader acting on a
run's end goes by the summary. The files, the periods and the ending's order are in
``docs/dev/gui/design.md`` (13.4, "The record").
"""

from __future__ import annotations

import contextlib
import os
import queue
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from solprop._durable import write_atomic
from solprop._process import process_stamp
from solprop.service.frames import encode, split_frames
from solprop.service.messages import TERMINAL, Event, ServiceError

if TYPE_CHECKING:
    # Only named here: this module stays light (no propagation imports), and the
    # run's process, which runs the propagation, makes the token.
    from solprop.backends.session import StopToken

__all__ = [
    "EVENTS",
    "HEARTBEAT",
    "JOB",
    "OWNER_PERIOD",
    "RECORD_PROGRESS_PERIOD",
    "STATUS",
    "STOP",
    "EventLog",
    "Job",
    "alive",
    "clear_stop",
    "process_stamp",
    "read_events",
    "read_job",
    "read_status",
    "request_stop",
    "set_aside",
    "stop_requested",
    "watch_stop",
    "write_job",
    "write_json",
]

JOB: Final = "job.json"
"""How a run was started, in its folder."""

EVENTS: Final = "events.bin"
"""A run's events, appended as frames, in its folder."""

STATUS: Final = "status.json"
"""A run's summary, rewritten as it goes, in its folder."""

STOP: Final = "stop"
"""The file whose appearance in a run's folder asks the run to stop."""

RECORD_PROGRESS_PERIOD: Final = 2.0
"""The least time between two progress events kept in the record, s."""

HEARTBEAT: Final = 30.0
"""The longest time between two rewrites of a living run's summary, s."""

STOP_PERIOD: Final = 0.25
"""How often a run looks for its stop file, s."""

OWNER_PERIOD: Final = 2.0
"""How often a run tied to a window checks that the window's process lives, s."""


def write_json(path: Path, value: Mapping[str, object]) -> None:
    """Replace a small JSON file whole, so a reader never sees half of it.

    Parameters
    ----------
    path : pathlib.Path
        The file.
    value : Mapping
        Its content, JSON-serialisable.
    """
    import json  # noqa: PLC0415 - light, and only here

    write_atomic(path, json.dumps(value, indent=1, sort_keys=True))


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return a small JSON file's object, or ``None`` when it is missing or torn."""
    import json  # noqa: PLC0415 - light, and only here

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return value if isinstance(value, dict) else None


def alive(pid: int, stamp: str | None) -> bool:
    """Return whether a process of this machine still runs, and is the one meant.

    Parameters
    ----------
    pid : int
        The process.
    stamp : str or None
        Its start (:func:`process_stamp`) when it was recorded; ``None`` trusts the
        number alone.

    Returns
    -------
    bool
        True while that process runs.

    Examples
    --------
    >>> alive(os.getpid(), process_stamp(os.getpid()))
    True
    """
    if pid < 1:
        return False
    now = process_stamp(pid)
    # No start time: the process has gone (every system here gives one for a
    # process that exists).
    return now is not None and (stamp is None or stamp == now)


@dataclass(frozen=True)
class Job:
    """How a run was started, as its ``job.json`` records it.

    Attributes
    ----------
    run_id : str
        The run, its folder's name.
    launcher : str
        ``"direct"`` (a detached process) or ``"slurm"``.
    host : str
        The machine the run's process runs on.
    pid : int
        Its process.
    stamp : str or None
        When that process started (:func:`process_stamp`).
    slurm_job : str or None
        The Slurm job solprop submitted it as (launcher ``"slurm"``); ``None`` for a
        run started any other way, even one inside a Slurm allocation.
    started : float
        When it started, s since the epoch.
    ended : float or None
        When it ended; ``None`` while it runs.
    solprop : str
        The solprop version it runs.
    threads : int or None
        Its thread budget, where one was given.

    Examples
    --------
    >>> Job.here("r1", "direct").host == socket.gethostname()
    True
    """

    run_id: str
    launcher: str
    host: str
    pid: int
    stamp: str | None
    slurm_job: str | None
    started: float
    ended: float | None = None
    solprop: str = ""
    threads: int | None = None

    @classmethod
    def here(cls, run_id: str, launcher: str, threads: int | None = None) -> "Job":
        """Return the record of a run starting in this process.

        Parameters
        ----------
        run_id : str
            The run.
        launcher : str
            How it was started; ``"slurm"`` records this process's Slurm job as the
            run's own.
        threads : int or None, optional
            Its thread budget.

        Returns
        -------
        Job
            Its record, with this machine, process and version.
        """
        from solprop import __version__  # noqa: PLC0415 - the package's own

        return cls(
            run_id=run_id,
            launcher=launcher,
            host=socket.gethostname(),
            pid=os.getpid(),
            stamp=process_stamp(os.getpid()),
            slurm_job=os.environ.get("SLURM_JOB_ID") if launcher == "slurm" else None,
            started=time.time(),
            solprop=__version__,
            threads=threads,
        )


def write_job(folder: Path, job: Job) -> None:
    """Write a run's ``job.json``.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder.
    job : Job
        Its record.
    """
    write_json(folder / JOB, asdict(job))


def read_job(folder: Path) -> Job | None:
    """Return a run's ``job.json``, or ``None`` for a run without one.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder.

    Returns
    -------
    Job or None
        Its record.
    """
    value = _read_json(folder / JOB)
    if value is None:
        return None
    try:
        return Job(**value)
    except TypeError:
        return None


def read_status(folder: Path) -> dict[str, Any] | None:
    """Return a run's ``status.json``, or ``None`` for a run without one.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder.

    Returns
    -------
    dict or None
        Its summary: ``updated`` (s since the epoch), the last ``progress``, a scan's
        ``points`` and ``done``, and its ``ending`` (the terminal event's kind and
        data) once it has one.
    """
    return _read_json(folder / STATUS)


def request_stop(folder: Path) -> None:
    """Ask a run to stop, by its stop file.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder.
    """
    (folder / STOP).touch()


def stop_requested(folder: Path) -> bool:
    """Return whether a run has been asked to stop.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder.

    Returns
    -------
    bool
        True while its stop file is there.
    """
    return (folder / STOP).exists()


def set_aside(folder: Path) -> None:
    """Keep a continued run's earlier record apart, so its new one starts empty.

    A window following the continued run reads its new events from the start, not the
    ending of the part before (``events.bin`` becomes ``events-1.bin``, then
    ``events-2.bin``, ...), and the earlier part's ``job.json`` goes beside them
    (``job-1.json``, ...), so its process is never taken for the new part's; its
    summary and stop file start afresh too. A folder with no record is left as it is.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder.

    Examples
    --------
    >>> import tempfile
    >>> folder = Path(tempfile.mkdtemp())
    >>> _ = (folder / EVENTS).write_bytes(b"")
    >>> _ = (folder / JOB).write_text("{}", encoding="utf-8")
    >>> set_aside(folder)
    >>> sorted(path.name for path in folder.iterdir())
    ['events-1.bin', 'job-1.json']
    """
    count = 1
    while (folder / f"events-{count}.bin").exists() or (
        folder / f"job-{count}.json"
    ).exists():
        count += 1
    for name, kept in ((EVENTS, f"events-{count}.bin"), (JOB, f"job-{count}.json")):
        if (folder / name).exists():
            (folder / name).replace(folder / kept)
    (folder / STATUS).unlink(missing_ok=True)
    clear_stop(folder)


def clear_stop(folder: Path) -> None:
    """Remove a run's stop file, as a launcher does before a run starts.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder.
    """
    (folder / STOP).unlink(missing_ok=True)


def watch_stop(
    folder: Path,
    stop: StopToken,
    done: threading.Event,
    *,
    owner: tuple[int, str | None] | None = None,
) -> threading.Thread:
    """Set a run's stop flag when its stop file appears, from a thread of its own.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder.
    stop : solprop.backends.session.StopToken
        The run's stop token (solprop's ``stop=``). Setting it from this thread
        reaches the run's session through the token's listeners at its next safe
        boundary.
    done : threading.Event
        Set when the run ends, which ends the watch.
    owner : tuple or None, optional
        A window's process (its number and start) the run is tied to: when it has
        gone, the run stops as it does on its stop file. Checked every
        :data:`OWNER_PERIOD` seconds.

    Returns
    -------
    threading.Thread
        The watching thread, started.
    """

    def watch() -> None:
        """Look for the stop file, and the owner, until the run ends."""
        checked = time.monotonic()
        while not done.wait(STOP_PERIOD):
            if stop_requested(folder):
                stop.set()
                return
            if owner is not None and time.monotonic() - checked >= OWNER_PERIOD:
                checked = time.monotonic()
                if not alive(*owner):
                    request_stop(folder)
                    stop.set()
                    return

    thread = threading.Thread(target=watch, name="solprop-stop-file", daemon=True)
    thread.start()
    return thread


def _value(event: Event) -> dict[str, object]:
    """Return an event as a frame carries it.

    - Returns: its run, kind and data.
    """
    return {"run_id": event.run_id, "kind": event.kind, "data": event.data}


def read_events(folder: Path, offset: int = 0) -> tuple[list[Event], int]:
    """Return the events a run has recorded from a place in its record on.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder.
    offset : int, optional
        Where to start, bytes: ``0``, or what the last read returned.

    Returns
    -------
    tuple
        The events, and where the next read starts; a frame still being written is
        left for that read.

    Raises
    ------
    solprop.service.messages.ServiceError
        If the record is not a run's events.
    """
    try:
        with (folder / EVENTS).open("rb") as stream:
            stream.seek(offset)
            data = stream.read()
    except FileNotFoundError:
        return [], offset
    values, used = split_frames(data)
    events: list[Event] = []
    for value in values:
        if not isinstance(value, dict) or "kind" not in value:
            msg = f"{folder / EVENTS} holds something other than a run's events"
            raise ServiceError(msg)
        events.append(
            Event(str(value["run_id"]), str(value["kind"]), dict(value["data"]))  # pyright: ignore[reportUnknownArgumentType]
        )
    return events, offset + used


class EventLog:
    """Append a run's events to its record, from a thread of its own.

    A plane's observer only hands its row over; the writing (and a slow shared
    filesystem) never holds up the march. Rows, a scan's points, the log and the ending
    are all kept; progress is kept at most every :data:`RECORD_PROGRESS_PERIOD`
    seconds. The summary (``status.json``) is rewritten after each kept event at most
    that often, and at least every :data:`HEARTBEAT` seconds while the run lives.

    Parameters
    ----------
    folder : pathlib.Path
        The run's folder, which exists.
    run_id : str
        The run.
    job : Callable or None, optional
        Makes the run's ``job.json`` record, written first by the writer, so that
        stamping the process (a ``ps`` off Linux) and syncing the file stay off the
        run's own path; the record is :attr:`job` once written.

    Attributes
    ----------
    job : Job or None
        The run's ``job.json``, once the writer has written it, with its end once
        the ending is recorded.

    Examples
    --------
    >>> import tempfile
    >>> folder = Path(tempfile.mkdtemp())
    >>> log = EventLog(folder, "r1")
    >>> log.put(Event("r1", "row", {"index": 0}))
    >>> log.put(Event("r1", "finished", {"path": "run.h5"}))
    >>> log.close()
    >>> [event.kind for event in read_events(folder)[0]]
    ['row', 'finished']
    >>> read_status(folder)["ending"]["kind"]
    'finished'
    """

    def __init__(
        self, folder: Path, run_id: str, job: Callable[[], Job] | None = None
    ) -> None:
        """Open the record for appending and start its writer."""
        self.folder = folder
        self.run_id = run_id
        self.job: Job | None = None
        self._make_job = job
        self._queue: queue.SimpleQueue[Event | None] = queue.SimpleQueue()
        self._last_progress = -float("inf")
        self._status: dict[str, Any] = {"run_id": run_id, "updated": time.time()}
        self._written = -float("inf")
        self._stream = (folder / EVENTS).open("ab")
        self._length = self._stream.tell()  # appending starts at the end
        self._intact = True
        self._thread = threading.Thread(
            target=self._write, name="solprop-record", daemon=True
        )
        self._thread.start()

    def put(self, event: Event) -> None:
        """Hand over one event, without waiting.

        Parameters
        ----------
        event : Event
            The event.
        """
        self._queue.put(event)

    def _kept(self, event: Event) -> bool:
        """Return whether an event goes into the record, or only into the summary."""
        if event.kind != "progress":
            return True
        now = time.monotonic()
        if now - self._last_progress < RECORD_PROGRESS_PERIOD:
            return False
        self._last_progress = now
        return True

    def _summarise(self, event: Event) -> None:
        """Fold one event into the summary."""
        if event.kind == "progress":
            self._status["progress"] = dict(event.data)
        elif event.kind == "started":
            self._status["started"] = dict(event.data)
        elif event.kind == "scan_started":
            self._status["points"] = int(event.data.get("points", 0))
            self._status["done"] = 0
        elif event.kind == "point" and event.data.get("status") is not None:
            self._status["done"] = int(event.data.get("done", 0))
        elif event.kind == "row":
            self._status["rows"] = int(self._status.get("rows", 0)) + 1
        elif event.kind in TERMINAL:
            data = {k: v for k, v in event.data.items() if k != "detail"}
            self._status["ending"] = {"kind": event.kind, "data": data}

    def _rewrite(self, *, force: bool = False) -> None:
        """Rewrite the summary, at most every period unless forced."""
        now = time.monotonic()
        if not force and now - self._written < RECORD_PROGRESS_PERIOD:
            return
        self._written = now
        self._status["updated"] = time.time()
        with contextlib.suppress(OSError, TypeError, ValueError):
            write_json(self.folder / STATUS, self._status)

    def _write(self) -> None:
        """Write the run's ``job.json``, then its events until the record is closed."""
        if self._make_job is not None:
            # A job.json that cannot be made or written costs the record its job, never
            # its events.
            with contextlib.suppress(OSError, ValueError):
                self.job = self._make_job()
                write_job(self.folder, self.job)
        while True:
            try:
                event = self._queue.get(timeout=HEARTBEAT)
            except queue.Empty:
                self._rewrite(force=True)
                continue
            if event is None:
                self._rewrite(force=True)
                return
            if event.terminal:
                self._end(event)
                return
            if self._kept(event) and not self._append(event):
                continue
            self._summarise(event)
            self._rewrite()

    def _append(self, event: Event) -> bool:
        """Append one event to ``events.bin``; return whether it was written."""
        if not self._intact:
            return False
        try:
            frame = encode(_value(event))
            self._stream.write(frame)
            self._stream.flush()
        except ServiceError:
            return False  # an unwritable value costs that event, not the run
        except OSError:
            # A full disk costs the event, not the run. Part of the frame may have
            # been written, and a reader would take the frames after it for its rest.
            self._intact = self._cut_back()
            return False
        self._length += len(frame)
        return True

    def _cut_back(self) -> bool:
        """Cut ``events.bin`` back to its whole frames and reopen it; return success.

        Closing drops what the failed write left buffered; a record that cannot be cut
        back keeps no more events, so none follows a part of a frame.
        """
        with contextlib.suppress(OSError):
            self._stream.close()
        try:
            os.truncate(self.folder / EVENTS, self._length)
            self._stream = (self.folder / EVENTS).open("ab")
        except OSError:
            return False
        return True

    def _end(self, event: Event) -> None:
        """Record the run's ending, in the order the module describes, and stop.

        The summary says the ending only once ``events.bin`` is closed and
        ``job.json`` has its end, and nothing is rewritten after it: a run continued
        the moment its ending is recorded sets aside a record no process holds, and
        its fresh summary is never overwritten by this one.
        """
        self._append(event)
        with contextlib.suppress(OSError):
            self._stream.close()
        if self.job is not None:
            with contextlib.suppress(OSError, ValueError):
                self.job = replace(self.job, ended=time.time())
                write_job(self.folder, self.job)
        self._summarise(event)
        self._rewrite(force=True)

    def close(self) -> None:
        """Write what is waiting, then close the record.

        The record ends at its terminal event: an event handed over after it is not
        kept.
        """
        self._queue.put(None)
        self._thread.join(timeout=30.0)
        self._stream.close()
