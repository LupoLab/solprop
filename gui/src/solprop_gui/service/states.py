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

__all__ = ["ACTIVE", "TRANSITIONS", "ExecutionState", "advance"]


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
