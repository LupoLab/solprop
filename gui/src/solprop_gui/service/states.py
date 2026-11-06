# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run's execution state, the window's own (design 4 and 10.2).

The execution state says what the window and the worker are doing; solprop's outcome
says what the run achieved, and whether a file can be continued is always asked of
solprop (``resumability``), never read from either.

::

    queued ─▶ starting ─▶ running ─▶ complete
                 │           │  └──▶ failed
                 │           └─▶ stopping ─▶ stopped
                 └─▶ failed            └──▶ (complete, failed: the run ended first)
    any active state ─▶ worker lost
"""

from enum import StrEnum
from typing import Final

__all__ = ["ACTIVE", "TRANSITIONS", "ExecutionState", "advance", "failure_advice"]


class ExecutionState(StrEnum):
    """What a run is doing, as the window sees it."""

    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETE = "complete"
    FAILED = "failed"
    LOST = "worker lost"


ACTIVE: Final = frozenset(
    {
        ExecutionState.QUEUED,
        ExecutionState.STARTING,
        ExecutionState.RUNNING,
        ExecutionState.STOPPING,
    }
)
"""The states in which a run may still change."""

_ENDINGS: Final = {
    "finished": ExecutionState.COMPLETE,
    "failed": ExecutionState.FAILED,
    "lost": ExecutionState.LOST,
}

TRANSITIONS: Final[dict[ExecutionState, dict[str, ExecutionState]]] = {
    ExecutionState.QUEUED: {"launch": ExecutionState.STARTING, **_ENDINGS},
    ExecutionState.STARTING: {
        "started": ExecutionState.RUNNING,
        "stop": ExecutionState.STOPPING,
        **_ENDINGS,
    },
    ExecutionState.RUNNING: {
        "stop": ExecutionState.STOPPING,
        "stopped": ExecutionState.STOPPED,
        **_ENDINGS,
    },
    ExecutionState.STOPPING: {
        "stopped": ExecutionState.STOPPED,
        **_ENDINGS,
    },
}
"""For each active state, the event names that move it, and where to."""


def advance(state: ExecutionState, event: str) -> ExecutionState:
    """Return the state an event moves a run to.

    Parameters
    ----------
    state : ExecutionState
        The run's state.
    event : str
        A worker event's kind (``"started"``, ``"stopped"``, ``"finished"``,
        ``"failed"``, ``"lost"``) or a window action (``"launch"``, ``"stop"``).
        Any other event (progress, rows, logs) leaves the state as it is.

    Returns
    -------
    ExecutionState
        The new state; the same state when the event does not move it, and always
        the same for a run that has ended.

    Examples
    --------
    >>> advance(ExecutionState.RUNNING, "stop")
    <ExecutionState.STOPPING: 'stopping'>
    >>> advance(ExecutionState.STOPPING, "finished")
    <ExecutionState.COMPLETE: 'complete'>
    """
    return TRANSITIONS.get(state, {}).get(event, state)


_MEMORY_HINT = (
    "A large run may have run out of memory. RungeKutta() keeps no Adams history and "
    "LowStorageRungeKutta() holds the least; fewer modes or a narrower window help too."
)


def failure_advice(kind: str, message: str, exitcode: object = None) -> str:
    """Say why a run stopped short, and what to try.

    solprop admits every run and a run that does not fit fails where it fails: with a
    ``MemoryError``, or with its worker killed and no Python error (ADR-065).

    Parameters
    ----------
    kind : {'failed', 'lost'}
        How it ended.
    message : str
        The message solprop gave, if any.
    exitcode : object, optional
        The worker's exit status, for one that was lost.

    Returns
    -------
    str
        What to tell the user.

    Examples
    --------
    >>> failure_advice("failed", "The grid is too small.")
    'The grid is too small.'
    >>> failure_advice("failed", "MemoryError").endswith("help too.")
    True
    >>> failure_advice("lost", "", -9).startswith("The run's worker was killed")
    True
    >>> failure_advice("lost", "", 1)
    "The run's worker ended without saying why (exit status 1)."
    """
    if kind == "failed":
        if "MemoryError" in message or "out of memory" in message.lower():
            return f"{message} {_MEMORY_HINT}"
        return message
    if exitcode in (-9, 137):
        return f"The run's worker was killed, with no error from Python. {_MEMORY_HINT}"
    return f"The run's worker ended without saying why (exit status {exitcode})."
