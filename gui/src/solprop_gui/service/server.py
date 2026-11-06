# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The solprop service: one process that owns the kernel and the runs (design 13.2, C).

The window's service interface (requests and replies, runs and their events) is the
same as in the process tree of option B; here it runs in a process of its own, which a
window reaches over a local socket, so that runs outlive the window that started them
and a window re-attaches to them later. One warm kernel answers every window's
requests; each run is a worker process, as before.

The socket is the user's alone: a Unix socket (a named pipe on Windows) whose address
and random token are in a file only the user can read
(:func:`state_file`), and every connection must prove the token
(``multiprocessing.connection``'s HMAC challenge) before a message is read. The
service ends itself when no window is connected and no run is going, after
:data:`IDLE_SECONDS`.

Messages are :class:`Envelope` pairs of a kind and a body:

- from a window: ``request`` (a :class:`~.messages.Request`), ``start_run`` (a
  :class:`~.worker.RunSpec`), ``control`` (a run's id and a
  :class:`~.messages.Control`, or ``"kill"``), ``attach`` (a run's id: its events,
  replayed from its first plane), ``list_runs``;
- to a window: ``reply``, ``event``, ``runs`` (each run's id, folder and last event).
"""

import contextlib
import json
import os
import secrets
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection, Listener
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.connection import PipeConnection  # Windows only
from pathlib import Path

from solprop_gui.service.client import KernelClient, ReplyFilter, RunClient
from solprop_gui.service.messages import Control, Event, Reply, Request

__all__ = [
    "IDLE_SECONDS",
    "Envelope",
    "Service",
    "main",
    "read_state",
    "state_file",
]

IDLE_SECONDS = 600.0
"""How long the service waits with no window and no run before it ends, s."""


@dataclass(frozen=True)
class Envelope:
    """One message between a window and the service.

    Attributes
    ----------
    kind : str
        What it is (see the module's description).
    body : object
        Its content.
    """

    kind: str
    body: object = None


def state_file() -> Path:
    """Return the file that says where this user's service listens.

    Returns
    -------
    pathlib.Path
        ``~/.solprop-gui/service.json`` (``SOLPROP_GUI_STATE`` overrides the
        folder).
    """
    folder = os.environ.get("SOLPROP_GUI_STATE")
    return Path(folder or Path.home() / ".solprop-gui") / "service.json"


def read_state(path: Path | None = None) -> dict[str, object] | None:
    """Return the running service's address, token and process, if one is recorded.

    Parameters
    ----------
    path : pathlib.Path or None, optional
        The state file; :func:`state_file` by default.

    Returns
    -------
    dict or None
        ``address``, ``authkey`` (hex), ``pid`` and ``family``; ``None`` when no
        service is recorded or the file does not read.
    """
    target = path or state_file()
    try:
        state = json.loads(target.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return state if isinstance(state, dict) else None


def _address() -> tuple[str, str]:
    """Return a fresh address for the service's socket, and its family.

    Returns
    -------
    tuple of str
        The address and ``"AF_UNIX"`` (a socket file in a private folder) or
        ``"AF_PIPE"`` (a named pipe, on Windows).
    """
    if sys.platform == "win32":  # pragma: no cover - Windows only
        return rf"\\.\pipe\solprop-gui-{secrets.token_hex(8)}", "AF_PIPE"
    # A short path: a Unix socket's path is limited to about a hundred bytes.
    folder = Path(tempfile.mkdtemp(prefix="solprop-gui-"))
    folder.chmod(0o700)
    return str(folder / "service.sock"), "AF_UNIX"


class _Everything(ReplyFilter):
    """A reply filter that keeps every reply: the windows filter their own."""

    def accepts(self, reply: Reply) -> bool:
        """Keep a reply.

        Parameters
        ----------
        reply : Reply
            The reply.

        Returns
        -------
        bool
            Always True.
        """
        del reply
        return True


class _Session:
    """One connected window.

    Parameters
    ----------
    connection : multiprocessing.connection.Connection
        Its socket (a ``PipeConnection`` on Windows).
    """

    def __init__(self, connection: Connection | PipeConnection) -> None:
        """Hold the socket and a lock for sending on it."""
        self.connection = connection
        self.lock = threading.Lock()
        self.open = True
        self.runs: set[str] = set()

    def send(self, envelope: Envelope) -> None:
        """Send a message, marking the session closed if the window has gone.

        Parameters
        ----------
        envelope : Envelope
            The message.
        """
        if not self.open:
            return
        try:
            with self.lock:
                self.connection.send(envelope)
        except BrokenPipeError, EOFError, OSError:
            self.open = False


class Service:
    """The service's state: the kernel, the runs and the connected windows.

    Parameters
    ----------
    kernel : KernelClient or None, optional
        The kernel's client; one is started on the first request by default.
    """

    def __init__(self, kernel: KernelClient | None = None) -> None:
        """Start with no window and no run."""
        self._kernel = kernel
        self._lock = threading.Lock()
        self._asked: dict[int, tuple[_Session, int]] = {}
        self.sessions: list[_Session] = []
        self.runs: dict[str, RunClient] = {}
        self.last: dict[str, str] = {}
        self.idle_since = time.monotonic()

    # ------------------------------------------------------------------ the kernel

    def _client(self) -> KernelClient:
        """Return the kernel's client, starting the kernel if it is not running.

        Returns
        -------
        KernelClient
            The client.
        """
        if self._kernel is None or not self._kernel.alive:
            self._kernel = KernelClient(self._replied)
            # Every window filters its own replies by its revision; the service
            # passes on all of them.
            self._kernel.filter = _Everything()
        return self._kernel

    def _replied(self, reply: Reply) -> None:
        """Send a kernel reply to the window that asked, with the window's own id.

        Parameters
        ----------
        reply : Reply
            The reply, under the service's id.
        """
        with self._lock:
            asked = self._asked.pop(reply.request_id, None)
        if asked is None:
            return
        session, own = asked
        session.send(
            Envelope(
                "reply",
                Reply(
                    reply.kind,
                    own,
                    reply.revision,
                    value=reply.value,
                    error=reply.error,
                    detail=reply.detail,
                ),
            )
        )

    def request(self, session: _Session, request: Request) -> None:
        """Pass a window's request to the kernel under an id of the service's.

        Parameters
        ----------
        session : _Session
            The window.
        request : Request
            Its request, with its own id.
        """
        client = self._client()
        # The lock is held until the id is recorded, so that the reply, which the
        # kernel's reader hands to _replied under the same lock, finds it.
        with self._lock:
            own = client.request(
                request.kind,
                request.payload,
                revision=request.revision,
                key=f"{id(session)}:{request.key}",  # latest wins within a window
            )
            self._asked[own] = (session, request.request_id)

    # ------------------------------------------------------------------ the runs

    def start_run(self, session: _Session, spec: object) -> None:
        """Start a run's worker, its events going to the window that started it.

        Parameters
        ----------
        session : _Session
            The window.
        spec : object
            The run's :class:`~.worker.RunSpec`.
        """
        run_id = str(spec.run_id)  # pyright: ignore[reportAttributeAccessIssue]
        session.runs.add(run_id)
        self.last[run_id] = "starting"
        self.runs[run_id] = RunClient(spec, self._event)  # pyright: ignore[reportArgumentType]

    def _event(self, event: Event) -> None:
        """Send a run's event to every window attached to it.

        Parameters
        ----------
        event : Event
            The event.
        """
        self.last[event.run_id] = event.kind
        for session in list(self.sessions):
            if event.run_id in session.runs:
                session.send(Envelope("event", event))

    def control(self, run_id: str, request: object) -> None:
        """Stop, kill or replay a run.

        Parameters
        ----------
        run_id : str
            The run.
        request : object
            A :class:`~.messages.Control`, or ``"kill"``.
        """
        client = self.runs.get(run_id)
        if client is None:
            return
        if request == "kill":
            client.kill()
        elif isinstance(request, Control) and request.kind == "stop":
            client.stop()
        elif isinstance(request, Control):
            client.subscribe(request.from_plane)

    def attach(self, session: _Session, run_id: str) -> None:
        """Send a run's events to a window from now on, its rows replayed.

        Parameters
        ----------
        session : _Session
            The window.
        run_id : str
            The run.
        """
        if run_id in self.runs:
            session.runs.add(run_id)
            self.runs[run_id].subscribe(0)

    def list_runs(self) -> list[dict[str, object]]:
        """Return every run the service holds.

        Returns
        -------
        list of dict
            Each run's ``run_id``, ``folder``, ``last`` event and whether it is
            ``alive``.
        """
        return [
            {
                "run_id": run_id,
                "folder": str(client.spec.folder),
                "last": self.last.get(run_id),
                "alive": client.alive,
            }
            for run_id, client in self.runs.items()
        ]

    # ------------------------------------------------------------------ windows

    def serve(self, session: _Session) -> None:
        """Answer one window's messages until it disconnects.

        Parameters
        ----------
        session : _Session
            The window.
        """
        self.sessions.append(session)
        try:
            while session.open:
                try:
                    envelope = session.connection.recv()
                except EOFError, OSError:
                    break
                self.handle(session, envelope)
        finally:
            session.open = False
            self.sessions.remove(session)
            self.idle_since = time.monotonic()

    def handle(self, session: _Session, envelope: object) -> None:
        """Act on one message of a window.

        Parameters
        ----------
        session : _Session
            The window.
        envelope : object
            The message; one of another shape is ignored.
        """
        if not isinstance(envelope, Envelope):
            return
        if envelope.kind == "request" and isinstance(envelope.body, Request):
            self.request(session, envelope.body)
        elif envelope.kind == "start_run":
            self.start_run(session, envelope.body)
        elif envelope.kind == "control" and isinstance(envelope.body, tuple):
            run_id, request = envelope.body
            self.control(str(run_id), request)
        elif envelope.kind == "attach":
            self.attach(session, str(envelope.body))
        elif envelope.kind == "list_runs":
            session.send(Envelope("runs", self.list_runs()))

    @property
    def idle(self) -> bool:
        """Return whether the service has had no window and no run for long enough.

        Returns
        -------
        bool
            True after :data:`IDLE_SECONDS` with neither.
        """
        busy = self.sessions or any(client.alive for client in self.runs.values())
        if busy:
            self.idle_since = time.monotonic()
            return False
        return time.monotonic() - self.idle_since > IDLE_SECONDS

    def close(self) -> None:
        """End the kernel; runs still going are left to finish."""
        if self._kernel is not None:
            self._kernel.close(timeout=2.0)


def hang_up(connection: object) -> None:
    """Close a connection, waking a thread blocked reading it.

    On Linux, closing a socket does not wake a thread blocked reading it, and the
    socket stays open until that read returns: the other end never sees it close.
    Shutting the socket down first wakes the reader at both ends. A Windows pipe
    wakes on its own.

    Parameters
    ----------
    connection : object
        A :class:`multiprocessing.connection.Connection`, or a Windows pipe's.
    """
    if isinstance(connection, Connection):
        with (
            contextlib.suppress(OSError),
            socket.socket(fileno=os.dup(connection.fileno())) as duplicate,
        ):
            duplicate.shutdown(socket.SHUT_RDWR)
    with contextlib.suppress(OSError):
        connection.close()  # pyright: ignore[reportAttributeAccessIssue]


def _accept(listener: Listener, service: Service, closing: threading.Event) -> None:
    """Accept windows, each served on a thread of its own, until the service closes.

    Parameters
    ----------
    listener : multiprocessing.connection.Listener
        The socket.
    service : Service
        The service.
    closing : threading.Event
        Set as the service closes; :func:`main` then connects once more to wake this.
    """
    while True:
        try:
            connection = listener.accept()
        except Exception:  # a stranger without the token, or one gone
            if closing.is_set():  # the listener closed under it
                return
            continue
        if closing.is_set():
            hang_up(connection)
            return
        threading.Thread(
            target=service.serve, args=(_Session(connection),), daemon=True
        ).start()


def main(state: Path | None = None, *, poll: float = 1.0) -> None:
    """Run the service until it has been idle for :data:`IDLE_SECONDS`.

    Parameters
    ----------
    state : pathlib.Path or None, optional
        Where to record the address and token; :func:`state_file` by default.
    poll : float, optional
        How often to check for idleness, s.
    """
    target = state or state_file()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    address, family = _address()
    authkey = secrets.token_bytes(32)
    listener = Listener(address, family=family, authkey=authkey)
    record = {
        "address": address,
        "family": family,
        "authkey": authkey.hex(),
        "pid": os.getpid(),
    }
    # Written private, then moved into place: no moment where another user reads it.
    temporary = target.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(record, stream)
    temporary.replace(target)
    service = Service()
    closing = threading.Event()
    acceptor = threading.Thread(
        target=_accept, args=(listener, service, closing), daemon=True
    )
    acceptor.start()
    try:
        while not service.idle:
            time.sleep(poll)
    finally:
        closing.set()
        # A last connection wakes the acceptor; on Linux, closing the listener does not.
        with contextlib.suppress(Exception):
            hang_up(Client(address, family=family, authkey=authkey))
        acceptor.join(2.0)
        if read_state(target) == record:
            target.unlink(missing_ok=True)
        listener.close()
        service.close()


if __name__ == "__main__":  # pragma: no cover - the service's own process
    main()
