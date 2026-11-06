# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The messages between the window and the service's processes.

Every message is a frozen dataclass of plain data (numbers, strings, lists, dicts and
NumPy arrays), so it pickles across a pipe today and can be encoded for a socket later
(design 13.3). The window sends a :class:`Request` to the kernel and receives a
:class:`Reply`; it sends a :class:`Control` to a run's worker and receives its
:class:`Event` stream.

Each request carries an id and the recipe revision it was made for, and each reply
carries both back, so the window can drop a reply that a later edit made stale.
"""

from dataclasses import dataclass, field
from typing import Any, Final, Literal

__all__ = [
    "PROTOCOL",
    "REPLACEABLE",
    "TERMINAL",
    "Control",
    "Envelope",
    "Event",
    "Reply",
    "Request",
]

PROTOCOL: Final = 1
"""The version of these messages; a peer of another version is refused."""

TERMINAL: Final = frozenset({"finished", "stopped", "failed", "lost"})
"""The event kinds that end a run's stream: never coalesced, never dropped."""

REPLACEABLE: Final = frozenset({"progress"})
"""The event kinds a later one of the same kind replaces while undelivered."""


@dataclass(frozen=True)
class Request:
    """A request to the kernel.

    Attributes
    ----------
    kind : str
        The handler, one of the kernel's (``"check"``, ``"preview_input"``, ...).
    payload : dict
        Its arguments, by name.
    request_id : int
        Unique within the client that sent it.
    revision : int
        The recipe revision it was made for; 0 for a request about no recipe.
    key : str
        What makes two requests of one kind the same question: a later request with
        the same kind and key supersedes an earlier one not yet answered.
    protocol : int
        The protocol version of the sender.

    Examples
    --------
    >>> Request("check", {"text": "..."}, request_id=1, revision=3).key
    ''
    """

    kind: str
    payload: dict[str, object]
    request_id: int
    revision: int = 0
    key: str = ""
    protocol: int = PROTOCOL


@dataclass(frozen=True)
class Reply:
    """The kernel's answer to one request.

    Attributes
    ----------
    kind : str
        The request's kind.
    request_id : int
        The request's id.
    revision : int
        The request's revision.
    value : Any
        The result, as plain data; ``None`` when the request failed.
    error : str or None
        Why it failed, in solprop's words; ``None`` when it succeeded.
    detail : str or None
        The traceback of an unexpected failure, for the log.
    superseded : bool
        Whether the kernel dropped the request unanswered because a later one with
        the same kind and key arrived first.

    Examples
    --------
    >>> Reply("check", 1, 3, value=[]).ok
    True
    """

    kind: str
    request_id: int
    revision: int
    value: Any = None  # plain data, by contract
    error: str | None = None
    detail: str | None = None
    superseded: bool = False

    @property
    def ok(self) -> bool:
        """Return whether the request succeeded.

        Returns
        -------
        bool
            True when it was answered without an error.
        """
        return self.error is None and not self.superseded


@dataclass(frozen=True)
class Event:
    """One event of a run's stream.

    Attributes
    ----------
    run_id : str
        The run.
    kind : str
        ``"started"``, ``"progress"``, ``"axes"``, ``"row"``, ``"warning"``,
        ``"log"``, or one of :data:`TERMINAL`.
    data : dict
        Its content, by name.

    Examples
    --------
    >>> Event("run-1", "finished").terminal
    True
    """

    run_id: str
    kind: str
    data: dict[str, Any] = field(default_factory=dict)  # plain data, by contract

    @property
    def terminal(self) -> bool:
        """Return whether the event ends the run's stream.

        Returns
        -------
        bool
            True for :data:`TERMINAL` kinds.
        """
        return self.kind in TERMINAL

    @property
    def replaceable(self) -> bool:
        """Return whether a later event of its kind replaces it while undelivered.

        Returns
        -------
        bool
            True for :data:`REPLACEABLE` kinds.
        """
        return self.kind in REPLACEABLE


@dataclass(frozen=True)
class Control:
    """A request to a run's worker.

    Attributes
    ----------
    kind : {'stop', 'subscribe'}
        Stop the run through solprop's stop flag, or replay the reduced rows from a
        plane on.
    from_plane : int
        For ``"subscribe"``, the first plane to replay.

    Examples
    --------
    >>> Control("subscribe", from_plane=3).from_plane
    3
    """

    kind: Literal["stop", "subscribe"]
    from_plane: int = 0


@dataclass(frozen=True)
class Envelope:
    """One message between a window and the solprop service, over its socket.

    It lives here, in a module only ever imported, and not in the service's own
    module: the service runs as ``python -m solprop_gui.service.server``, where that
    module is ``__main__``, and a class defined there would not be the class a window
    pickles, so every message would fail the service's ``isinstance`` check.

    Attributes
    ----------
    kind : str
        What it is (:mod:`solprop_gui.service.server` lists the kinds).
    body : object
        Its content.
    """

    kind: str
    body: object = None
