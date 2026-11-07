# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The messages between a window, its kernel and its runs.

Every message is a frozen dataclass of plain data (numbers, strings, lists, dicts and
NumPy arrays), so it pickles across the kernel's pipe and encodes as a frame
(:mod:`solprop.service.frames`) for a run's record and a machine's helper (design 13.3,
13.4). The window sends a :class:`Request` to a kernel and receives a :class:`Reply`;
it follows a run's :class:`Event` stream from the run's record.

Each request carries an id and the recipe revision it was made for, and each reply
carries both back, so the window can drop a reply that a later edit made stale.
"""

from dataclasses import dataclass, field
from typing import Any, Final

from solprop.exceptions import SolpropError

__all__ = [
    "PROTOCOL",
    "TERMINAL",
    "Event",
    "Reply",
    "Request",
    "ServiceError",
]


class ServiceError(SolpropError, RuntimeError):
    """The front end's service refused a message, a run's record or a machine."""


PROTOCOL: Final = 1
"""The version of these messages; a peer of another version is refused."""

TERMINAL: Final = frozenset({"finished", "stopped", "failed", "lost"})
"""The event kinds that end a run's stream: never coalesced, never dropped."""


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
