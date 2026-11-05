# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel process: a warm solprop that answers the window's requests.

The kernel validates, previews, estimates and reads results for the window, one
request at a time, with one or two threads (design 13.3). Requests are *latest wins*:
before it starts on one, the kernel takes every request already waiting and, of those
with the same kind and key, answers only the newest; the others get a
``superseded`` reply, so the window knows they will not be answered.

This module is imported in the window's process (it is the child's entry point), so it
imports nothing heavy; :func:`serve` imports the handlers in the child.
"""

import os
import traceback
from collections.abc import Callable, Iterable, Mapping
from multiprocessing.connection import Connection

from solprop_gui.service.messages import PROTOCOL, Reply, Request

__all__ = ["KERNEL_THREADS", "RUN_THREADS", "answer", "latest", "serve"]

KERNEL_THREADS = "2"
"""The thread budget of the kernel process (``SOLPROP_THREADS``)."""

RUN_THREADS = "SOLPROP_GUI_RUN_THREADS"
"""Where the kernel keeps the ``SOLPROP_THREADS`` it started with, empty for unset."""

Handler = Callable[[Mapping[str, object]], object]


def latest(requests: Iterable[Request]) -> tuple[list[Request], list[Request]]:
    """Split waiting requests into those to answer and those superseded.

    Parameters
    ----------
    requests : Iterable of Request
        The requests, in the order they arrived.

    Returns
    -------
    tuple of list of Request
        The requests to answer, in arrival order, and those a later request of the
        same kind and key supersedes.

    Examples
    --------
    >>> from solprop_gui.service.messages import Request
    >>> keep, dropped = latest([Request("check", {}, 1), Request("check", {}, 2)])
    >>> [r.request_id for r in keep], [r.request_id for r in dropped]
    ([2], [1])
    """
    ordered = list(requests)
    newest: dict[tuple[str, str], Request] = {}
    for request in ordered:
        newest[request.kind, request.key] = request
    keep_ids = {request.request_id for request in newest.values()}
    keep = [r for r in ordered if r.request_id in keep_ids]
    dropped = [r for r in ordered if r.request_id not in keep_ids]
    return keep, dropped


def answer(request: Request, handlers: Mapping[str, Handler]) -> Reply:
    """Answer one request.

    A refusal from solprop, or any other exception, becomes the reply's error, with
    the traceback as its detail: the kernel never dies of a request.

    Parameters
    ----------
    request : Request
        The request.
    handlers : Mapping
        The handlers, by kind.

    Returns
    -------
    Reply
        The reply.

    Examples
    --------
    >>> from solprop_gui.service.messages import Request
    >>> answer(Request("echo", {"x": 1}, 7), {"echo": lambda p: p["x"]}).value
    1
    """
    if request.protocol != PROTOCOL:
        return Reply(
            request.kind,
            request.request_id,
            request.revision,
            error=f"protocol {request.protocol} is not this kernel's {PROTOCOL}",
        )
    handler = handlers.get(request.kind)
    if handler is None:
        return Reply(
            request.kind,
            request.request_id,
            request.revision,
            error=f"no handler for {request.kind!r}",
        )
    try:
        value = handler(request.payload)
    except Exception as error:
        return Reply(
            request.kind,
            request.request_id,
            request.revision,
            error=str(error) or type(error).__name__,
            detail=traceback.format_exc(),
        )
    return Reply(request.kind, request.request_id, request.revision, value=value)


def _waiting(connection: Connection) -> list[Request]:
    """Take one request, waiting for it, and every other already waiting.

    Parameters
    ----------
    connection : multiprocessing.connection.Connection
        The kernel's end of the pipe.

    Returns
    -------
    list of Request
        The requests, in arrival order.

    Raises
    ------
    EOFError
        When the window has closed its end.
    """
    requests = [connection.recv()]
    while connection.poll(0):
        requests.append(connection.recv())
    return requests


def serve(
    connection: Connection, handlers: Mapping[str, Handler] | None = None
) -> None:
    """Answer requests until the window closes its end of the pipe.

    Parameters
    ----------
    connection : multiprocessing.connection.Connection
        The kernel's end of the pipe.
    handlers : Mapping or None, optional
        The handlers, by kind; ``None`` imports solprop's
        (:data:`solprop_gui.service.handlers.HANDLERS`).
    """
    # What a run would get: a run's worker starts from the window's environment.
    os.environ[RUN_THREADS] = os.environ.get("SOLPROP_THREADS", "")
    os.environ.setdefault("SOLPROP_THREADS", KERNEL_THREADS)
    if handlers is None:
        from solprop_gui.service.handlers import HANDLERS  # noqa: PLC0415 - child only

        handlers = HANDLERS
    while True:
        try:
            requests = _waiting(connection)
        except EOFError, OSError:
            return
        keep, dropped = latest(requests)
        try:
            for request in dropped:
                connection.send(
                    Reply(
                        request.kind,
                        request.request_id,
                        request.revision,
                        superseded=True,
                    )
                )
            for request in keep:
                connection.send(answer(request, handlers))
        except BrokenPipeError, OSError:
            return
