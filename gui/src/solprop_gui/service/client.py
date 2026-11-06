# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The window's side of the service: start the kernel and the workers, talk to them.

:class:`KernelClient` and :class:`RunClient` own a child process each and a reader
thread that hands every message to a callback; the window's Qt adapter turns those
calls into signals (design 13.3). They import no Qt and no solprop.

Children are started with ``spawn`` on every platform, so a child imports what it needs
and nothing of the window: no Qt, no open files, no threads.
"""

import itertools
import multiprocessing
import threading
from collections.abc import Callable, Iterator, Mapping
from multiprocessing.connection import Connection
from typing import Any

from solprop_gui.service import kernel, worker
from solprop_gui.service.messages import Control, Event, Reply, Request
from solprop_gui.service.worker import RunSpec

__all__ = ["POLL_PERIOD", "KernelClient", "ReplyFilter", "RunClient"]

POLL_PERIOD = 0.5
"""How often a run's reader looks whether its worker is still alive, s."""


class ReplyFilter:
    """Decide which kernel replies are still wanted.

    A reply is dropped when it was superseded, when a later request of the same
    kind and key has been sent since (latest wins), or when it answers a recipe
    revision that is no longer the current one. A request about no recipe
    (revision 0) is never dropped for its revision.

    Examples
    --------
    >>> from solprop_gui.service.messages import Reply, Request
    >>> wanted = ReplyFilter()
    >>> wanted.sent(Request("check", {}, 1, revision=1)); wanted.revision = 1
    >>> wanted.sent(Request("check", {}, 2, revision=1))
    >>> wanted.accepts(Reply("check", 1, 1)), wanted.accepts(Reply("check", 2, 1))
    (False, True)
    """

    def __init__(self) -> None:
        """Start with no requests and revision 0."""
        self.revision = 0
        self._latest: dict[tuple[str, str], int] = {}
        self._keys: dict[int, tuple[str, str]] = {}
        self._lock = threading.Lock()

    def sent(self, request: Request) -> None:
        """Record a request as the latest of its kind and key.

        Parameters
        ----------
        request : Request
            The request.
        """
        with self._lock:
            key = (request.kind, request.key)
            self._latest[key] = request.request_id
            self._keys[request.request_id] = key

    def accepts(self, reply: Reply) -> bool:
        """Return whether a reply is still wanted, forgetting its request.

        Parameters
        ----------
        reply : Reply
            The reply.

        Returns
        -------
        bool
            Whether to deliver it.
        """
        with self._lock:
            key = self._keys.pop(reply.request_id, None)
            if reply.superseded or key is None:
                return False
            if self._latest.get(key) != reply.request_id:
                return False
            return reply.revision in (0, self.revision)


class KernelClient:
    """Start the kernel process and exchange requests and replies with it.

    Parameters
    ----------
    on_reply : callable
        Called with each wanted :class:`Reply`, from the reader thread.
    on_lost : callable or None, optional
        Called once if the kernel process ends while the client is open.
    target : callable, optional
        The child's entry point, called with its end of the pipe; the kernel's
        :func:`~solprop_gui.service.kernel.serve` by default.
    ids : iterator of int or None, optional
        Where request ids come from; a count of its own by default. A window that may
        change transport shares one, so a request keeps its id across the change.
    """

    def __init__(
        self,
        on_reply: Callable[[Reply], None],
        *,
        on_lost: Callable[[], None] | None = None,
        target: Callable[[Connection], None] = kernel.serve,
        ids: Iterator[int] | None = None,
    ) -> None:
        """Start the kernel and its reader thread."""
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(
            target=target, args=(child,), daemon=True, name="solprop-kernel"
        )
        self._process.start()
        child.close()
        self._on_reply = on_reply
        self._on_lost = on_lost
        self._ids = itertools.count(1) if ids is None else ids
        self._send_lock = threading.Lock()
        self._closing = False
        self.filter = ReplyFilter()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    @property
    def revision(self) -> int:
        """Return the current recipe revision.

        Returns
        -------
        int
            Replies to requests made for another revision are dropped.
        """
        return self.filter.revision

    @revision.setter
    def revision(self, value: int) -> None:
        """Set the current recipe revision.

        Parameters
        ----------
        value : int
            The revision.
        """
        self.filter.revision = value

    @property
    def alive(self) -> bool:
        """Return whether the kernel process is running.

        Returns
        -------
        bool
            True while it is.
        """
        return self._process.is_alive()

    def request(
        self,
        kind: str,
        payload: Mapping[str, object] | None = None,
        *,
        revision: int = 0,
        key: str = "",
    ) -> int:
        """Send one request.

        Parameters
        ----------
        kind : str
            The handler.
        payload : Mapping or None, optional
            Its arguments.
        revision : int, optional
            The recipe revision it is made for; 0 for one about no recipe.
        key : str, optional
            What distinguishes it from other requests of its kind for latest-wins.

        Returns
        -------
        int
            The request's id.
        """
        return self.submit(
            Request(
                kind,
                dict(payload or {}),
                request_id=next(self._ids),
                revision=revision,
                key=key,
            )
        )

    def submit(self, request: Request) -> int:
        """Send a request already made, keeping its id.

        Parameters
        ----------
        request : Request
            The request; one another transport took and could not send.

        Returns
        -------
        int
            Its id.
        """
        self.filter.sent(request)
        with self._send_lock:
            self._connection.send(request)
        return request.request_id

    def _read(self) -> None:
        """Deliver replies until the pipe closes."""
        while True:
            try:
                reply: Reply = self._connection.recv()
            except EOFError, OSError, TypeError:
                # TypeError: close() released the handle while recv() waited on it.
                break
            if self._closing:
                break  # the window is going: it takes no more replies
            if self.filter.accepts(reply):
                try:
                    self._on_reply(reply)
                except RuntimeError:
                    # Qt deleted the receiver before its window closed this client.
                    self._closing = True
                    break
        if not self._closing and self._on_lost is not None:
            self._on_lost()

    def close(self, timeout: float = 5.0) -> None:
        """Close the pipe, and end the kernel.

        Parameters
        ----------
        timeout : float, optional
            Seconds to wait for it to end before terminating it.
        """
        self._closing = True
        self._connection.close()
        self._process.join(timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout)
        # The reader may be handing over a reply that arrived as the pipe closed: it
        # is done before the window, which receives it, can go.
        if self._reader is not threading.current_thread():
            self._reader.join(timeout)


class RunClient:
    """Start one run's worker and receive its events.

    Parameters
    ----------
    spec : RunSpec
        What to run.
    on_event : callable
        Called with each :class:`Event`, from the reader thread. If the worker ends
        without a terminal event, a ``lost`` event with its exit code is made here.
    target : callable, optional
        The child's entry point, called with the spec and the two pipe ends;
        :func:`solprop_gui.service.worker.main` by default.
    """

    def __init__(
        self,
        spec: RunSpec,
        on_event: Callable[[Event], None],
        *,
        target: Callable[[RunSpec, Connection, Connection], Any] = worker.main,
    ) -> None:
        """Start the worker and its reader thread."""
        context = multiprocessing.get_context("spawn")
        self._events, events_child = context.Pipe(duplex=False)
        control_child, self._control = context.Pipe(duplex=False)
        self.spec = spec
        self._process = context.Process(
            target=target,
            args=(spec, events_child, control_child),
            daemon=False,
            name=f"solprop-run-{spec.run_id}",
        )
        self._process.start()
        events_child.close()
        control_child.close()
        self._on_event = on_event
        self.ended = threading.Event()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    @property
    def alive(self) -> bool:
        """Return whether the worker process is running.

        Returns
        -------
        bool
            True while it is.
        """
        return self._process.is_alive()

    def _read(self) -> None:
        """Deliver events until the terminal one, or until the worker is gone."""
        terminal = False
        while not terminal:
            try:
                if not self._events.poll(POLL_PERIOD):
                    # A worker killed before it took its end of the pipe never
                    # closes it (Windows): its death is the end of the stream.
                    if self._process.is_alive():
                        continue
                    break
                event: Event = self._events.recv()
            except EOFError, OSError:
                break
            terminal = event.terminal
            self._on_event(event)
        if not terminal:
            self._process.join(5.0)
            self._on_event(
                Event(
                    self.spec.run_id,
                    "lost",
                    {"exitcode": self._process.exitcode},
                )
            )
        self.ended.set()

    def _control_send(self, request: Control) -> bool:
        """Send a control request, if the worker can still hear it.

        Parameters
        ----------
        request : Control
            The request.

        Returns
        -------
        bool
            Whether it was sent.
        """
        try:
            self._control.send(request)
        except BrokenPipeError, EOFError, OSError:
            return False
        return True

    def stop(self) -> bool:
        """Ask the run to stop through solprop's stop flag.

        Returns
        -------
        bool
            Whether the request reached the worker.
        """
        return self._control_send(Control("stop"))

    def subscribe(self, from_plane: int = 0) -> bool:
        """Ask the worker to replay its rows from a plane on.

        Parameters
        ----------
        from_plane : int, optional
            The first plane.

        Returns
        -------
        bool
            Whether the request reached the worker.
        """
        return self._control_send(Control("subscribe", from_plane=from_plane))

    def kill(self) -> None:
        """End the worker at once; its file may hold no restart state."""
        self._process.kill()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker to end and its events to be delivered.

        Parameters
        ----------
        timeout : float or None, optional
            The most seconds to wait.
        """
        self.ended.wait(timeout)
        self._process.join(timeout)
        self._control.close()
