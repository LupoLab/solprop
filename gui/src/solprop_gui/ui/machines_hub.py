# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The window's machines, as Qt signals: one connection each (design 13.4).

The hub holds a :class:`~solprop.service.machines.Connection` to each machine the
window uses, re-emits what they say on the window's thread, and runs their requests on
threads of its own, so the window never waits on a network. It remembers where it has
read each followed run's record to, and when a machine's connection comes back it
follows those runs again from there: a dropped network costs the view nothing, and the
runs nothing at all.

The machines themselves are kept in the window's settings (:class:`MachineStore`).
"""

import contextlib
import json
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QSettings, Signal
from solprop.service.machines import CHUNK, LOCAL, Connection, Machine
from solprop.service.messages import Event, Reply, Request, ServiceError

__all__ = ["MachineHub", "MachineRun", "MachineStore"]

_KEY = "machines"


def _read_marker(path: Path) -> dict[str, object] | None:
    """Return the source a partial copy was taken from, as its marker records it.

    Parameters
    ----------
    path : pathlib.Path
        The marker beside the partial copy.

    Returns
    -------
    dict or None
        The source's ``size`` and ``mtime``; ``None`` with no readable marker.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return value if isinstance(value, dict) else None


def _copy_to(connection: Connection, local: Path, remote: str) -> None:
    """Copy one file to a machine through its helper, piece by piece.

    Parameters
    ----------
    connection : solprop.service.machines.Connection
        The machine's connection.
    local : pathlib.Path
        The file here.
    remote : str
        Where it goes there.
    """
    with local.open("rb") as stream:
        append = False
        while True:
            data = stream.read(CHUNK)
            piece = np.frombuffer(data, dtype=np.uint8)
            connection.call("put", {"path": remote, "data": piece, "append": append})
            append = True
            if len(data) < CHUNK:
                return


class MachineStore:
    """The machines the window remembers, in its settings.

    Parameters
    ----------
    settings : PySide6.QtCore.QSettings
        The window's settings.
    """

    def __init__(self, settings: QSettings) -> None:
        """Hold the settings."""
        self._settings = settings

    def machines(self) -> list[Machine]:
        """Return the remembered machines, this computer first.

        Returns
        -------
        list of Machine
            This computer, then the others in the order they were added.
        """
        found: list[Machine] = [Machine(LOCAL, workspace="")]
        text = self._settings.value(_KEY, "[]")
        try:
            values = json.loads(str(text))
        except ValueError:
            values = []
        for value in values if isinstance(values, list) else []:
            if isinstance(value, dict) and value.get("name") not in {None, LOCAL}:
                found.append(Machine.from_value(value))
        return found

    def get(self, name: str) -> Machine | None:
        """Return a machine by its name.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        Machine or None
            The machine, if remembered.
        """
        return next((m for m in self.machines() if m.name == name), None)

    def save(self, machines: list[Machine]) -> None:
        """Remember these machines (this computer is always there).

        Parameters
        ----------
        machines : list of Machine
            The machines.
        """
        values = [m.value() for m in machines if not m.local]
        self._settings.setValue(_KEY, json.dumps(values))


class MachineRun:
    """A run on a machine, as a run panel holds a worker's handle.

    Stop writes the run's stop file; Kill ends its job (``scancel``, or ``SIGTERM``),
    which still leaves a resumable file.

    Parameters
    ----------
    hub : MachineHub
        The hub.
    machine : str
        The machine's name.
    folder : str
        The run's folder on it.
    """

    def __init__(self, hub: MachineHub, machine: str, folder: str) -> None:
        """Hold where the run is."""
        self.hub = hub
        self.machine = machine
        self.folder = folder
        self.ended = threading.Event()

    def stop(self) -> None:
        """Ask the run to stop at its next step."""
        self.hub.call(self.machine, "stop", {"folder": self.folder})

    def subscribe(self, from_plane: int = 0) -> None:
        """Do nothing: a run's record drops no rows, so none need replaying.

        Parameters
        ----------
        from_plane : int, optional
            The first plane wanted.
        """
        del from_plane

    def cancel(self) -> None:
        """End the run's job (``SIGTERM``, ``scancel``); solprop saves where it was."""
        self.hub.call(self.machine, "cancel", {"folder": self.folder})

    def kill(self) -> None:
        """Kill the run's process outright, as a last resort; or cancel a Slurm job."""
        self.hub.call(self.machine, "cancel", {"folder": self.folder, "force": True})

    def end_now(self, *, kill: bool = False) -> None:
        """Stop, or kill, the run now, waiting for the machine to take the request.

        For a window closing: what it asks must reach the machine before the window's
        connections close.

        Parameters
        ----------
        kill : bool, optional
            Kill the run's process outright, rather than stop it at its next step.
        """
        connection = self.hub.connections.get(self.machine)
        if connection is None:
            return
        kind, body = (
            ("cancel", {"folder": self.folder, "force": True})
            if kill
            else (
                "stop",
                {"folder": self.folder},
            )
        )
        with contextlib.suppress(ServiceError):
            connection.call(kind, body, timeout=10.0)

    def join(self, timeout: float | None = None) -> None:
        """Wait for the run's ending to arrive.

        Parameters
        ----------
        timeout : float or None, optional
            The most seconds to wait.
        """
        self.ended.wait(timeout)


class MachineHub(QObject):
    """The connections to the window's machines.

    Parameters
    ----------
    parent : PySide6.QtCore.QObject or None, optional
        The owner.
    askpass : str or None, optional
        The program ``ssh`` asks a password or code of.

    Attributes
    ----------
    state_changed : Signal
        Emitted with a machine's name, its connection's state and the detail.
    run_event : Signal
        Emitted with each :class:`~solprop.service.messages.Event` of a followed run.
    replied : Signal
        Emitted with each :class:`~solprop.service.messages.Reply` of a machine's
        kernel.
    answered : Signal
        Emitted with a call's token and its value, or the error it raised.
    """

    state_changed = Signal(str, str, object)
    run_event = Signal(object)
    replied = Signal(object)
    answered = Signal(int, object, object)

    def __init__(
        self, parent: QObject | None = None, *, askpass: str | None = None
    ) -> None:
        """Hold no connection until a machine is used."""
        super().__init__(parent)
        self.askpass = askpass
        self.connections: dict[str, Connection] = {}
        self.states: dict[str, str] = {}
        self._offsets: dict[tuple[str, str], int] = {}
        self._periods: dict[tuple[str, str], float] = {}
        self._runs: dict[tuple[str, str], MachineRun] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="solprop-machine"
        )
        self._tokens = iter(range(1, 1 << 62))
        self._callbacks: dict[int, Callable[[object, object], None]] = {}
        self._closed = False
        self.answered.connect(self._answer)

    # -------------------------------------------------------------- connections

    def ensure(self, machine: Machine) -> Connection:
        """Return a machine's connection, opening it if need be.

        Parameters
        ----------
        machine : Machine
            The machine.

        Returns
        -------
        solprop.service.machines.Connection
            Its connection.
        """
        connection = self.connections.get(machine.name)
        if connection is not None and connection.machine != machine:
            # Retired on a thread: closing waits for the helper, which a stuck SSH
            # session can make slow, and the window must not wait.
            self._pool.submit(connection.close)
            connection = None
        if connection is None:
            name = machine.name
            connection = Connection(
                machine,
                lambda frame: self._frame(name, frame),
                lambda state, detail: self._state(name, state, detail),
                askpass=self.askpass,
            )
            self.connections[name] = connection
            connection.open()
        return connection

    def forget(self, name: str) -> None:
        """Close a machine's connection; its runs go on.

        Parameters
        ----------
        name : str
            The machine.
        """
        connection = self.connections.pop(name, None)
        if connection is not None:
            self._pool.submit(connection.close)

    def _state(self, name: str, state: str, detail: object) -> None:
        """Take a connection's new state, from its thread.

        Parameters
        ----------
        name : str
            The machine.
        state : str
            The state.
        detail : object
            What goes with it.
        """
        if self._closed:
            return
        self.states[name] = state
        if state == "up":
            # Back: follow again, from where each run's record was read to.
            for (machine, folder), offset in list(self._offsets.items()):
                if machine == name:
                    period = self._periods.get((machine, folder), 0.5)
                    body = {"folder": folder, "offset": offset, "period": period}
                    self.call(name, "follow", body)
        self.state_changed.emit(name, state, detail)

    def _frame(self, name: str, frame: Mapping[str, Any]) -> None:
        """Take a frame a machine sent unasked, from its connection's thread.

        Parameters
        ----------
        name : str
            The machine.
        frame : Mapping
            The frame.
        """
        if self._closed:
            return
        kind = frame.get("kind")
        if kind == "events":
            folder = str(frame["folder"])
            self._offsets[(name, folder)] = int(frame["offset"])
            for value in frame["events"]:
                event = Event(
                    str(value["run_id"]), str(value["kind"]), dict(value["data"])
                )
                if event.terminal:
                    self._offsets.pop((name, folder), None)
                    run = self._runs.pop((name, folder), None)
                    if run is not None:
                        run.ended.set()
                self.run_event.emit(event)
        elif kind == "kernel_reply":
            value = dict(frame["reply"])
            self.replied.emit(Reply(**value))

    # ------------------------------------------------------------------ requests

    def call(
        self,
        name: str,
        kind: str,
        body: Mapping[str, object] | None = None,
        done: Callable[[object, object], None] | None = None,
    ) -> None:
        """Send a request on a thread of the hub's; the answer comes to ``done``.

        Parameters
        ----------
        name : str
            The machine.
        kind : str
            The request.
        body : Mapping or None, optional
            Its body.
        done : Callable or None, optional
            Called on the window's thread with the value and ``None``, or ``None``
            and the error's text.
        """
        token = next(self._tokens)
        if done is not None:
            self._callbacks[token] = done

        def ask() -> None:
            """Send the request and hand its answer to the window's thread."""
            connection = self.connections.get(name)
            try:
                if connection is None:
                    msg = f"{name} is not a machine the window uses"
                    raise ServiceError(msg)
                value = connection.call(kind, body)
            except ServiceError as error:
                self._answered(token, None, str(error))
                return
            self._answered(token, value, None)

        self._pool.submit(ask)

    def call_after_copies(
        self,
        name: str,
        copies: list[tuple[Path, str]],
        kind: str,
        body: Mapping[str, object],
        done: Callable[[object, object], None],
    ) -> None:
        """Copy files to a machine, piece by piece, then send a request.

        Parameters
        ----------
        name : str
            The machine.
        copies : list of tuple
            Each file here, and where it goes there.
        kind : str
            The request sent once every file is there.
        body : Mapping
            Its body.
        done : Callable
            Called on the window's thread with the request's value and ``None``, or
            ``None`` and the error's text (a copy's or the request's).
        """
        token = next(self._tokens)
        self._callbacks[token] = done

        def work() -> None:
            """Copy each file, then ask."""
            connection = self.connections.get(name)
            try:
                if connection is None:
                    msg = f"{name} is not a machine the window uses"
                    raise ServiceError(msg)
                for local, remote in copies:
                    _copy_to(connection, local, remote)
                value = connection.call(kind, body)
            except (ServiceError, OSError) as error:
                self._answered(token, None, str(error))
                return
            self._answered(token, value, None)

        self._pool.submit(work)

    def _answered(self, token: int, value: object, error: object) -> None:
        """Hand an answer to the window's thread, unless the hub has closed.

        Parameters
        ----------
        token : int
            The call.
        value : object
            Its value.
        error : object
            Its error's text, or ``None``.
        """
        if not self._closed:
            self.answered.emit(token, value, error)

    def _answer(self, token: int, value: object, error: object) -> None:
        """Hand a call's answer to whoever asked, on the window's thread.

        Parameters
        ----------
        token : int
            The call.
        value : object
            Its value.
        error : object
            Its error's text, or ``None``.
        """
        done = self._callbacks.pop(token, None)
        if done is not None:
            done(value, error)

    def kernel(self, name: str, request: Request) -> None:
        """Send a request to a machine's kernel; its reply comes as :attr:`replied`.

        Parameters
        ----------
        name : str
            The machine.
        request : solprop.service.messages.Request
            The request.
        """
        body = {
            "request": {
                "kind": request.kind,
                "payload": request.payload,
                "request_id": request.request_id,
                "revision": request.revision,
                "key": request.key,
            }
        }

        def failed(value: object, error: object) -> None:
            """Answer a kernel request that never reached the machine with its error.

            Parameters
            ----------
            value : object
                The call's value, unused.
            error : object
                Why it failed, or ``None``.
            """
            if error is not None:
                self.replied.emit(
                    Reply(
                        request.kind,
                        request.request_id,
                        request.revision,
                        error=str(error),
                    )
                )
            del value

        self.call(name, "kernel", body, failed)

    def follow(
        self,
        name: str,
        folder: str,
        offset: int = 0,
        *,
        period: float = 0.5,
        send: bool = True,
    ) -> MachineRun:
        """Follow a run's events from a place in its record on.

        Parameters
        ----------
        name : str
            The machine.
        folder : str
            The run's folder there.
        offset : int, optional
            Where to start, bytes; 0 for its whole history.
        period : float, optional
            How often the machine reads the record, s.
        send : bool, optional
            Ask the machine now; False leaves it to :meth:`begin`, as for a run not
            started yet.

        Returns
        -------
        MachineRun
            The run's handle.
        """
        self._offsets[(name, folder)] = offset
        self._periods[(name, folder)] = period
        run = MachineRun(self, name, folder)
        self._runs[(name, folder)] = run
        if send:
            self.begin(name, folder)
        return run

    def begin(self, name: str, folder: str) -> None:
        """Ask a machine to send a run's events, from where the hub has read to.

        Parameters
        ----------
        name : str
            The machine.
        folder : str
            The run's folder there.
        """
        offset = self._offsets.get((name, folder), 0)
        period = self._periods.get((name, folder), 0.5)
        body = {"folder": folder, "offset": offset, "period": period}
        self.call(name, "follow", body)

    def unfollow(self, name: str, folder: str) -> None:
        """Stop following a run.

        Parameters
        ----------
        name : str
            The machine.
        folder : str
            The run's folder there.
        """
        self._offsets.pop((name, folder), None)
        self._runs.pop((name, folder), None)
        if name in self.connections:
            self.call(name, "unfollow", {"folder": folder})

    def fetch(
        self,
        name: str,
        remote: str,
        local: Path,
        progress: Callable[[int, int], None],
        done: Callable[[object, object], None],
    ) -> threading.Event:
        """Copy a machine's file here, piece by piece, on a thread of the hub's.

        A copy cut short resumes from what it has when asked again.

        Parameters
        ----------
        name : str
            The machine.
        remote : str
            The file there.
        local : pathlib.Path
            Where the copy goes.
        progress : Callable
            Called, from the copying thread, with the bytes copied and the total.
        done : Callable
            Called on the window's thread with the copy's path and ``None``, or
            ``None`` and the error's text.

        Returns
        -------
        threading.Event
            Set to give up.
        """
        token = next(self._tokens)
        self._callbacks[token] = done
        cancelled = threading.Event()

        def copy() -> None:
            """Copy the file piece by piece, resuming a partial copy."""
            connection = self.connections.get(name)
            partial = local.with_name(local.name + ".part")
            try:
                if connection is None:
                    msg = f"{name} is not a machine the window uses"
                    raise ServiceError(msg)
                found = connection.call("stat", {"path": remote})
                if found is None:
                    msg = f"{remote} is not on {name}"
                    raise ServiceError(msg)
                total = int(found["size"])
                local.parent.mkdir(parents=True, exist_ok=True)
                # A partial copy is resumed only from the same file: one changed
                # since (a run continued, a run still writing) starts over.
                source = {"size": total, "mtime": found["mtime"]}
                marker = local.with_name(local.name + ".part.json")
                if _read_marker(marker) != source:
                    partial.unlink(missing_ok=True)
                    marker.write_text(json.dumps(source), encoding="utf-8")
                have = partial.stat().st_size if partial.exists() else 0
                with partial.open("ab") as stream:
                    while have < total and not cancelled.is_set():
                        piece = connection.call(
                            "fetch", {"path": remote, "offset": have, "size": CHUNK}
                        )
                        data = bytes(piece)
                        if not data:
                            break
                        stream.write(data)
                        have += len(data)
                        progress(have, total)
                if cancelled.is_set():
                    msg = "the copy was cancelled"
                    raise ServiceError(msg)
                after = connection.call("stat", {"path": remote})
                unchanged = (
                    after is not None
                    and {
                        "size": int(after["size"]),
                        "mtime": after["mtime"],
                    }
                    == source
                )
                if not unchanged or partial.stat().st_size != total:
                    partial.unlink(missing_ok=True)
                    marker.unlink(missing_ok=True)
                    msg = f"{remote} changed while it was copied: copy it again"
                    raise ServiceError(msg)
                partial.replace(local)
                marker.unlink(missing_ok=True)
            except (ServiceError, OSError) as error:
                self._answered(token, None, str(error))
                return
            self._answered(token, str(local), None)

        self._pool.submit(copy)
        return cancelled

    def copy_here(
        self, source: Path, target: Path, done: Callable[[object, object], None]
    ) -> None:
        """Copy a file of this computer's on a thread of the hub's.

        Parameters
        ----------
        source : pathlib.Path
            The file.
        target : pathlib.Path
            Where the copy goes.
        done : Callable
            Called on the window's thread with the copy's path and ``None``, or
            ``None`` and the error's text.
        """
        import shutil  # noqa: PLC0415 - only here

        token = next(self._tokens)
        self._callbacks[token] = done

        def copy() -> None:
            """Copy the file."""
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            except OSError as error:
                self._answered(token, None, str(error))
                return
            self._answered(token, str(target), None)

        self._pool.submit(copy)

    def close(self) -> None:
        """Close every connection, now; the machines' runs go on."""
        self._closed = True
        for connection in self.connections.values():
            connection.close()
        self.connections.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)
