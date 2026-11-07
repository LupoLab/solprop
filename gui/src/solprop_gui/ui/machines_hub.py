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

import json
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSettings, Signal
from solprop.service.machines import CHUNK, LOCAL, Connection, Machine
from solprop.service.messages import Event, Reply, Request, ServiceError

__all__ = ["MachineHub", "MachineRun", "MachineStore"]

_KEY = "machines"


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

    def kill(self) -> None:
        """End the run's job; solprop still writes where it stood."""
        self.hub.call(self.machine, "cancel", {"folder": self.folder})

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
            connection.close()
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
                    self.call(name, "follow", {"folder": folder, "offset": offset})
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

    def follow(self, name: str, folder: str, offset: int = 0) -> MachineRun:
        """Follow a run's events from a place in its record on.

        Parameters
        ----------
        name : str
            The machine.
        folder : str
            The run's folder there.
        offset : int, optional
            Where to start, bytes; 0 for its whole history.

        Returns
        -------
        MachineRun
            The run's handle.
        """
        self._offsets[(name, folder)] = offset
        run = MachineRun(self, name, folder)
        self._runs[(name, folder)] = run
        self.call(name, "follow", {"folder": folder, "offset": offset})
        return run

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
                partial.replace(local)
            except (ServiceError, OSError) as error:
                self._answered(token, None, str(error))
                return
            self._answered(token, str(local), None)

        self._pool.submit(copy)
        return cancelled

    def close(self) -> None:
        """Close every connection, now; the machines' runs go on."""
        self._closed = True
        for connection in self.connections.values():
            connection.close()
        self.connections.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)
