# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A window's connection to the solprop service (design 13.2, option C).

:class:`ServiceConnection` finds this user's service (:func:`.server.read_state`), or
starts one in a process of its own that outlives the window, and speaks the
service's :class:`~.messages.Envelope` messages over its socket. It offers what the
in-window clients of option B offer, the same calls with the same callbacks, so the
bridge chooses one or the other and the window does not know which: requests with
latest-wins and revision filtering, runs started and controlled through
:class:`RemoteRun` handles, and in addition the service's runs listed and attached to.
"""

import contextlib
import itertools
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from multiprocessing.connection import Client
from pathlib import Path

from solprop.service.client import ReplyFilter
from solprop.service.messages import Control, Envelope, Event, Reply, Request
from solprop.service.server import hang_up, read_state, state_file
from solprop.service.worker import RunSpec

__all__ = ["START_SECONDS", "RemoteRun", "ServiceConnection", "start_service"]

START_SECONDS = 20.0
"""How long a window waits for a service it started to say where it listens, s."""


def start_service(state: Path | None = None) -> None:
    """Start the service in a process of its own, which outlives the window.

    Parameters
    ----------
    state : pathlib.Path or None, optional
        Where it records its address; :func:`.server.state_file` by default.
    """
    command = [sys.executable, "-m", "solprop.service.server"]
    environment = None
    if state is not None:
        import os  # noqa: PLC0415 - only here

        environment = {**os.environ, "SOLPROP_GUI_STATE": str(state.parent)}
    flags = {}
    if sys.platform == "win32":  # pragma: no cover - Windows only
        flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008
    else:
        flags["start_new_session"] = True
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=environment,
        **flags,  # pyright: ignore[reportArgumentType]
    )


class RemoteRun:
    """A run the service holds, as the window's handle on it.

    Parameters
    ----------
    connection : ServiceConnection
        The window's connection.
    spec : RunSpec
        What it runs.
    """

    def __init__(self, connection: ServiceConnection, spec: RunSpec) -> None:
        """Hold the connection and the run's spec."""
        self._connection = connection
        self.spec = spec
        self.ended = threading.Event()

    @property
    def alive(self) -> bool:
        """Return whether the run has not yet sent its terminal event.

        Returns
        -------
        bool
            True until it has.
        """
        return not self.ended.is_set()

    def stop(self) -> bool:
        """Ask the run to stop through solprop's stop flag.

        Returns
        -------
        bool
            Whether the request was sent.
        """
        return self._connection.control(self.spec.run_id, Control("stop"))

    def subscribe(self, from_plane: int = 0) -> bool:
        """Ask the run's worker to replay its rows from a plane on.

        Parameters
        ----------
        from_plane : int, optional
            The first plane.

        Returns
        -------
        bool
            Whether the request was sent.
        """
        return self._connection.control(
            self.spec.run_id, Control("subscribe", from_plane=from_plane)
        )

    def kill(self) -> None:
        """Ask the service to end the run's worker at once."""
        self._connection.control(self.spec.run_id, "kill")

    def join(self, timeout: float | None = None) -> None:
        """Wait for the run's terminal event.

        Parameters
        ----------
        timeout : float or None, optional
            The most seconds to wait.
        """
        self.ended.wait(timeout)


class ServiceConnection:
    """A window's connection to the service, started if none is running.

    It connects on its reader thread, never the window's: finding the service, or
    starting one and waiting up to :data:`START_SECONDS` for it, would freeze the
    window. What the window sends meanwhile is held, and sent once connected; if no
    service can be reached, ``on_failed`` is given back what was held, and then any
    request or run sent after the failure.

    Parameters
    ----------
    on_reply : callable
        Called with each wanted :class:`Reply`, from the reader thread.
    on_event : callable
        Called with each :class:`Event` of the runs the window started or attached.
    on_runs : callable or None, optional
        Called with the service's list of runs, after :meth:`list_runs`.
    on_lost : callable or None, optional
        Called once if the connection drops while the window is open.
    state : pathlib.Path or None, optional
        The service's state file; :func:`.server.state_file` by default.
    on_failed : callable or None, optional
        Called from the reader thread if no service can be reached: with the reason,
        the requests held and the runs started while connecting, none sent. Called
        again, from the sender's thread, with each request or run sent after that.
    ids : iterator of int or None, optional
        Where request ids come from; a count of its own by default.
    """

    def __init__(  # noqa: PLR0913 - the window's callbacks, its state file and ids
        self,
        on_reply: Callable[[Reply], None],
        on_event: Callable[[Event], None],
        *,
        on_runs: Callable[[list[dict[str, object]]], None] | None = None,
        on_lost: Callable[[], None] | None = None,
        state: Path | None = None,
        on_failed: Callable[[str, list[Request], list[RunSpec]], None] | None = None,
        ids: Iterator[int] | None = None,
    ) -> None:
        """Start connecting, on the reader thread."""
        self._state = state or state_file()
        self._connection: object | None = None
        self._held: list[Envelope] = []
        self._failed = False
        self._reason = ""
        # Set once the reader has told the window of the failure and given back what was
        # held: anything sent later is given back only after that, so that the window,
        # which may close this connection on hearing of it, cannot drop the held batch.
        self._told = threading.Event()
        self._on_reply = on_reply
        self._on_event = on_event
        self._on_runs = on_runs
        self._on_lost = on_lost
        self._on_failed = on_failed
        self._ids = itertools.count(1) if ids is None else ids
        self._lock = threading.Lock()
        self._closing = False
        self.filter = ReplyFilter()
        self.runs: dict[str, RemoteRun] = {}
        self._reader = threading.Thread(target=self._run, daemon=True)
        self._reader.start()

    def _run(self) -> None:
        """Connect, send what was held, and deliver the service's messages."""
        try:
            connection = self._connect()
        except ConnectionError as error:
            with self._lock:
                self._failed = True
                self._reason = str(error)
                held, self._held = self._held, []
            try:
                self._give_back(held, news=True)
            finally:
                self._told.set()
            return
        with self._lock:
            if self._closing:
                hang_up(connection)
                return
            self._connection = connection
            held, self._held = self._held, []
            for envelope in held:
                with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                    connection.send(envelope)  # pyright: ignore[reportAttributeAccessIssue]
        self._read()

    def _try(self) -> object | None:
        """Connect to the recorded service, if it answers.

        Returns
        -------
        object or None
            The connection; ``None`` when none is recorded or it does not answer.
        """
        recorded = read_state(self._state)
        if recorded is None:
            return None
        try:
            return Client(
                str(recorded["address"]),
                family=str(recorded["family"]),
                authkey=bytes.fromhex(str(recorded["authkey"])),
            )
        except OSError, EOFError, ValueError:
            return None

    def _connect(self) -> object:
        """Connect to the service, starting one if none answers.

        Returns
        -------
        object
            The connection.

        Raises
        ------
        ConnectionError
            If none answers within :data:`START_SECONDS` of starting one.
        """
        connection = self._try()
        if connection is not None:
            return connection
        start_service(self._state)
        deadline = time.monotonic() + START_SECONDS
        while time.monotonic() < deadline:
            time.sleep(0.1)
            connection = self._try()
            if connection is not None:
                return connection
        msg = "the solprop service did not start"
        raise ConnectionError(msg)

    @property
    def revision(self) -> int:
        """Return the recipe revision whose replies are wanted.

        Returns
        -------
        int
            The revision.
        """
        return self.filter.revision

    @revision.setter
    def revision(self, value: int) -> None:
        """Set the recipe revision whose replies are wanted.

        Parameters
        ----------
        value : int
            The revision.
        """
        self.filter.revision = value

    @property
    def failed(self) -> bool:
        """Return whether no service could be reached.

        Returns
        -------
        bool
            True once connecting has failed; ``on_failed`` has then been given what
            was held.
        """
        return self._failed

    @property
    def alive(self) -> bool:
        """Return whether the connection is open.

        Returns
        -------
        bool
            True until it drops or is closed.
        """
        return not self._closing and self._reader.is_alive()

    def _send(self, envelope: Envelope) -> bool:
        """Send a message to the service.

        Parameters
        ----------
        envelope : Envelope
            The message.

        Returns
        -------
        bool
            Whether it was sent.
        """
        try:
            with self._lock:
                if self._connection is None and not self._failed:
                    # Still connecting: held, and sent once connected.
                    self._held.append(envelope)
                    return True
                if self._connection is not None:
                    self._connection.send(envelope)  # pyright: ignore[reportAttributeAccessIssue]
                    return True
        except BrokenPipeError, EOFError, OSError:
            return False
        # Connecting failed before this was sent, perhaps between the window's look at
        # :attr:`failed` and now: given back like what was held, or it would be lost,
        # once the reader has given back its own (unless this is the reader, telling).
        if self._reader is not threading.current_thread():
            self._told.wait()
        self._give_back([envelope], news=False)
        return False

    def _give_back(self, envelopes: list[Envelope], *, news: bool) -> None:
        """Hand the window the requests and runs the service never took.

        Parameters
        ----------
        envelopes : list of Envelope
            What was held, or sent after connecting failed.
        news : bool
            Whether this is the failure itself, told even with nothing to give back.
        """
        requests = [e.body for e in envelopes if isinstance(e.body, Request)]
        runs = [e.body for e in envelopes if isinstance(e.body, RunSpec)]
        if self._closing or self._on_failed is None:
            return
        if news or requests or runs:
            self._on_failed(self._reason, requests, runs)

    def request(
        self,
        kind: str,
        payload: Mapping[str, object] | None = None,
        *,
        revision: int = 0,
        key: str = "",
    ) -> int:
        """Send one request to the service's kernel.

        Parameters
        ----------
        kind : str
            The handler.
        payload : Mapping or None, optional
            Its arguments.
        revision : int, optional
            The recipe revision it is for; 0 for one about no recipe.
        key : str, optional
            Its latest-wins key.

        Returns
        -------
        int
            The request's id.
        """
        request = Request(
            kind,
            dict(payload or {}),
            request_id=next(self._ids),
            revision=revision,
            key=key,
        )
        self.filter.sent(request)
        self._send(Envelope("request", request))
        return request.request_id

    def start_run(self, spec: RunSpec) -> RemoteRun:
        """Start a run in the service.

        Parameters
        ----------
        spec : RunSpec
            What to run.

        Returns
        -------
        RemoteRun
            Its handle; its events arrive through ``on_event``.
        """
        run = RemoteRun(self, spec)
        self.runs[spec.run_id] = run
        self._send(Envelope("start_run", spec))
        return run

    def attach(self, spec: RunSpec) -> RemoteRun:
        """Take a running run's events, replayed from its first plane.

        Parameters
        ----------
        spec : RunSpec
            The run's identity and folder (as :meth:`list_runs` gives them).

        Returns
        -------
        RemoteRun
            Its handle.
        """
        run = RemoteRun(self, spec)
        self.runs[spec.run_id] = run
        self._send(Envelope("attach", spec.run_id))
        return run

    def control(self, run_id: str, request: object) -> bool:
        """Stop, replay or kill a run.

        Parameters
        ----------
        run_id : str
            The run.
        request : object
            A :class:`Control`, or ``"kill"``.

        Returns
        -------
        bool
            Whether it was sent.
        """
        return self._send(Envelope("control", (run_id, request)))

    def list_runs(self) -> bool:
        """Ask for the service's runs; they arrive through ``on_runs``.

        Returns
        -------
        bool
            Whether the request was sent.
        """
        return self._send(Envelope("list_runs"))

    def _read(self) -> None:
        """Deliver the service's messages until the connection closes."""
        while True:
            try:
                envelope = self._connection.recv()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
            except EOFError, OSError, TypeError:
                break
            if self._closing:
                break
            self._deliver(envelope)
        if not self._closing and self._on_lost is not None:
            self._on_lost()

    def _deliver(self, envelope: object) -> None:
        """Hand one message of the service to its callback.

        Parameters
        ----------
        envelope : object
            The message; one of another shape is ignored.
        """
        if not isinstance(envelope, Envelope):
            return
        body = envelope.body
        if envelope.kind == "reply" and isinstance(body, Reply):
            if self.filter.accepts(body):
                self._on_reply(body)
        elif envelope.kind == "event" and isinstance(body, Event):
            run = self.runs.get(body.run_id)
            if run is not None and body.terminal:
                run.ended.set()
            self._on_event(body)
        elif envelope.kind == "runs" and self._on_runs is not None:
            self._on_runs(list(body))  # pyright: ignore[reportArgumentType]

    def close(self, timeout: float = 2.0) -> None:
        """Close the connection; the service and its runs go on.

        Parameters
        ----------
        timeout : float, optional
            Seconds to wait for the reader to finish.
        """
        self._closing = True
        with self._lock:
            connection = self._connection
        if connection is not None:
            hang_up(connection)
        if self._reader is not threading.current_thread():
            self._reader.join(timeout)
