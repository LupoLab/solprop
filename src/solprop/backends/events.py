# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The records an execution session hands its service loop (SEP 47 section 5.3).

A session (:class:`solprop.backends.session.RunSession`) owns the numerical state
of one trajectory. It runs until an external service is needed or a bounded yield
falls due, and reports what happened as a short batch of immutable events. Events are
completed work or explicit service requests; none of them is a second live
controller. Four kinds exist, and the set is closed:

- :class:`ObservationReady`: a requested plane, its statistics and, while the batch
  is pending, an export ticket for its field;
- :class:`ServiceRequired`: an operation the caller must complete before the session
  advances further, such as delivering a warning or persisting a snapshot;
- :class:`Progress`: a bounded yield's accounting, with no field;
- :class:`Terminal`: the outcome of the run, its committed position and whether that
  position can be resumed.

A batch (:class:`EventBatch`) carries a :class:`Receipt`. Acknowledging the receipt
releases the batch's resources, invalidates its tickets and lets the session advance
across the barrier the batch ended at. A receipt is good once: acknowledging it twice,
or acknowledging an older one, is refused, so no barrier is crossed twice.

**Attempts are delivered exactly once, in chunks.** Every event carries the attempts
(:class:`~solprop.backends.protocol.BackendHistory`) and periodic statistics rows
(:class:`~solprop.backends.protocol.StepStats`, without a batch axis) made since the
*previous event*, whatever its kind -- not since the previous observation. Each
attempt and each row is in exactly one event's chunk, so a session buffers at most
the attempts of one bounded advance, and a full buffer yields a :class:`Progress`
rather than discarding records (SEP 47 section 5.3). A row's ``step`` indexes the
attempts of its own chunk. Wall times count from the start of their chunk.

The consumer assembles what spans several events: a saved plane's history is the
chunks since the previous observation, its own last (:func:`join_attempts`); a running
checkpoint's is the chunks since the previous observation up to and including the
snapshot's, which it includes without consuming, so the next plane still receives
them; a terminal checkpoint's is the chunks since the previous observation up to and
including the terminal event's (SEP 47 section 5.5).
"""

from __future__ import annotations

import enum
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from solprop.backends.protocol import BackendHistory, StepStats, StepSummary

__all__ = [
    "Eligibility",
    "Event",
    "EventBatch",
    "ExportTicket",
    "GroupBatch",
    "ObservationKind",
    "ObservationReady",
    "Outcome",
    "Progress",
    "Receipt",
    "ServiceKind",
    "ServiceRequired",
    "Terminal",
    "emit_warning",
    "join_attempts",
]


class ObservationKind(enum.Enum):
    """Why an observation was made.

    Attributes
    ----------
    LAUNCH : str
        The entrance plane, before the first step.
    PLANE : str
        A requested save plane, reached exactly or interpolated (dense output).
    STEP : str
        The endpoint of every ``step_planes``-th accepted step.
    NATURAL : str
        A natural stop short of the requested plane (SEP 14), saved as a plane.
    """

    LAUNCH = "launch"
    PLANE = "plane"
    STEP = "step"
    NATURAL = "natural"


class ServiceKind(enum.Enum):
    """What a :class:`ServiceRequired` event asks of the caller.

    Attributes
    ----------
    WARNING : str
        Emit a warning through Python's warning machinery. Under the caller's warning
        filters it may become an exception, so it is a barrier before propagation
        continues (SEP 47 section 5.5).
    SNAPSHOT : str
        Persist a running checkpoint at this accepted endpoint and copy the snapshot
        the caller requested (SEP 47 section 5.5).
    """

    WARNING = "warning"
    SNAPSHOT = "snapshot"


class Outcome(enum.Enum):
    """How a session ended.

    Attributes
    ----------
    COMPLETED : str
        Every requested plane was observed and the final checks passed.
    STOPPED : str
        A stop request was honoured at a safe boundary.
    FAILED : str
        Propagation failed; :attr:`Terminal.error` carries the diagnostic.
    """

    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class Eligibility(enum.Enum):
    """Whether a terminal position can be persisted as a resumable checkpoint.

    Attributes
    ----------
    ELIGIBLE : str
        The committed state is valid; a checkpoint there resumes the run.
    POISONED : str
        An in-place step failed mid-flight and left no valid state at the last accepted
        position (the low-storage march, SEP 24); resume only from an earlier persisted
        state.
    """

    ELIGIBLE = "eligible"
    POISONED = "poisoned"


@dataclass(frozen=True)
class Receipt:
    """Identifies one pending batch of events.

    Attributes
    ----------
    session : int
        The issuing session's identity, so that a receipt cannot be acknowledged on
        another session.
    sequence : int
        The batch's position in its session, from 1, strictly increasing.
    """

    session: int
    sequence: int


@dataclass(frozen=True)
class ExportTicket:
    """Names one field that a session can export while its batch is pending.

    A ticket keys an export by committed state generation, so that every consumer of
    the same payload -- the output, a plane observer, a checkpoint -- shares one
    detached result (SEP 47 section 5.4). It is valid until its batch is acknowledged.

    Attributes
    ----------
    receipt : Receipt
        The batch the ticket belongs to.
    generation : int
        The committed state generation (or dense observation) the field is taken from.
    z : float
        Position of that state, m.
    member : int or None
        The group member whose field it names (a
        :class:`~solprop.backends.session.GroupSession`'s ticket, whose members share
        one batch and generation); ``None`` for a lone session's.
    """

    receipt: Receipt
    generation: int
    z: float
    member: int | None = None


@dataclass(frozen=True, eq=False)
class ObservationReady:
    """A requested plane, observed.

    Attributes
    ----------
    z : float
        Position of the plane, m.
    kind : ObservationKind
        Why it was observed.
    plane_index : int
        The index of the save plane this observation answers, into the request's save
        planes; ``-1`` for the launch plane.
    history : solprop.backends.protocol.BackendHistory
        Every attempt since the previous event, accepted and rejected: the last chunk
        of the plane's history (module docstring).
    step_stats : solprop.backends.protocol.StepStats
        The periodic statistics rows sampled since the previous event, likewise.
    stats : collections.abc.Mapping
        The plane's statistics by catalogue name, without a row or batch axis.
    summary : solprop.backends.protocol.StepSummary
        The cost and tolerance of the propagation that reached this plane.
    ticket : ExportTicket
        The plane's field, exportable while this batch is pending.
    """

    z: float
    kind: ObservationKind
    plane_index: int
    history: BackendHistory
    step_stats: StepStats
    stats: Mapping[str, np.ndarray]
    summary: StepSummary
    ticket: ExportTicket


@dataclass(frozen=True, eq=False)
class ServiceRequired:
    """An external operation that must complete before the session advances.

    Attributes
    ----------
    kind : ServiceKind
        What is asked.
    z : float
        The committed position the request arises at, m.
    message : str
        For a warning, its text; otherwise a short description.
    category : type[Warning] or None
        For a warning, its category; ``None`` otherwise.
    ticket : ExportTicket or None
        For a snapshot, the committed field to persist; for a warning, the committed
        field a failure checkpoint persists should the warning be raised; ``None``
        otherwise.
    history : solprop.backends.protocol.BackendHistory or None
        For a snapshot or a warning, the attempts since the previous event: the last
        chunk of a checkpoint's history there, which includes the chunks since the
        previous observation without consuming them (SEP 47 section 5.5). ``None``
        otherwise.
    step_stats : solprop.backends.protocol.StepStats or None
        For a snapshot or a warning, the periodic rows since the previous event,
        likewise.
    warning : Warning or None
        For a warning, the instance to emit, with every attribute its issuer gives it
        (:func:`emit_warning`); ``None`` to emit `category` with `message`.
    """

    kind: ServiceKind
    z: float
    message: str
    category: type[Warning] | None = None
    ticket: ExportTicket | None = None
    history: BackendHistory | None = None
    step_stats: StepStats | None = None
    warning: Warning | None = None


def emit_warning(event: ServiceRequired, *, stacklevel: int = 2) -> None:
    """Emit a warning barrier's warning through Python's warning machinery.

    The caller's filters apply: one that raises it raises it here, at the barrier, and
    the run ends at that committed endpoint (SEP 47 section 5.5). The warning is the
    event's own instance where it carries one, so that it reaches the filters and the
    caller exactly as its issuer would have warned.

    Parameters
    ----------
    event : ServiceRequired
        A :attr:`ServiceKind.WARNING` request.
    stacklevel : int, optional
        As for :func:`warnings.warn`, counted from the caller of this function.

    Raises
    ------
    ValueError
        If `event` is not a warning.

    Examples
    --------
    >>> import warnings
    >>> from solprop.backends.events import ServiceKind, ServiceRequired, emit_warning
    >>> event = ServiceRequired(ServiceKind.WARNING, 0.0, "edge", UserWarning)
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     emit_warning(event)
    >>> str(caught[0].message)
    'edge'
    """
    if event.kind is not ServiceKind.WARNING:
        msg = f"a {event.kind.value} request is not a warning"
        raise ValueError(msg)
    if event.warning is not None:
        warnings.warn(event.warning, stacklevel=stacklevel + 1)
        return
    warnings.warn(
        event.message, event.category or UserWarning, stacklevel=stacklevel + 1
    )


@dataclass(frozen=True, eq=False)
class Progress:
    """A bounded yield: the attempts since the previous event, with no field.

    Attributes
    ----------
    z : float
        Committed position, m.
    history : solprop.backends.protocol.BackendHistory
        The attempts made since the previous event, delivered here once; the consumer
        keeps them for the next plane (module docstring). Delivery changes nothing
        numerical.
    step_stats : solprop.backends.protocol.StepStats
        The periodic rows sampled since the previous event, likewise.
    """

    z: float
    history: BackendHistory
    step_stats: StepStats = field(default_factory=StepStats.empty)


@dataclass(frozen=True, eq=False)
class Terminal:
    """The end of a session.

    Attributes
    ----------
    outcome : Outcome
        How it ended.
    z : float
        The last committed position, m.
    eligibility : Eligibility
        Whether a checkpoint at `z` would resume the run.
    error : BaseException or None
        The primary failure, for :attr:`Outcome.FAILED`; ``None`` otherwise.
    ticket : ExportTicket or None
        The committed field at `z`, exportable for a terminal checkpoint while the batch
        is pending; ``None`` when the state is poisoned.
    history : solprop.backends.protocol.BackendHistory or None
        The attempts since the previous event: the last chunk of a terminal
        checkpoint's history, which carries every chunk since the previous observation
        (module docstring). Delivered also for a poisoned state, whose attempts before
        the abandoned one were made all the same.
    step_stats : solprop.backends.protocol.StepStats or None
        The periodic rows since the previous event, likewise.
    """

    outcome: Outcome
    z: float
    eligibility: Eligibility
    error: BaseException | None = None
    ticket: ExportTicket | None = None
    history: BackendHistory | None = None
    step_stats: StepStats | None = None


Event = ObservationReady | ServiceRequired | Progress | Terminal
"""One of the four event kinds of SEP 47 section 5.3."""


@dataclass(frozen=True, eq=False)
class EventBatch:
    """The events one call to ``advance`` produced, ending at a barrier.

    Attributes
    ----------
    receipt : Receipt
        Acknowledge it to release the batch and cross its barrier.
    events : tuple
        The events, in committed order. At most one carries a field-bearing
        observation or service request; the rest are field-free.
    """

    receipt: Receipt
    events: tuple[Event, ...] = field(default_factory=tuple)

    @property
    def terminal(self) -> Terminal | None:  # numpydoc ignore=RT01
        """The batch's terminal event, or ``None`` if it did not end the session."""
        if self.events and isinstance(self.events[-1], Terminal):
            return self.events[-1]
        return None


@dataclass(frozen=True, eq=False)
class GroupBatch:
    """What one call to a group's ``advance`` produced: one common schedule window.

    A group (:class:`solprop.backends.session.GroupSession`) advances its members
    through one shared fixed schedule. Each member keeps its own events: its
    observations, with its own attempts and statistics and a ticket for its field,
    and its own terminal outcome (SEP 47 section 6.7).

    Attributes
    ----------
    receipt : Receipt
        Acknowledge it to release the window and advance to the next.
    members : tuple of tuple
        Each member's events in member order, in committed order; empty for a
        member that ended in an earlier window.
    outcome : Outcome or None
        The group's own outcome when the batch ends it: ``COMPLETED`` once the
        schedule is run, whatever each member's outcome; ``FAILED`` or ``STOPPED``
        when a group-wide failure or stop ended every member still running.
        ``None`` otherwise.
    error : BaseException or None
        A group-wide failure's primary error, carried also by each such member's
        :class:`Terminal`; ``None`` otherwise.
    """

    receipt: Receipt
    members: tuple[tuple[Event, ...], ...]
    outcome: Outcome | None = None
    error: BaseException | None = None


def _column(
    histories: list[BackendHistory], name: str, dtype: type = np.float64
) -> np.ndarray:
    """Return one column of consecutive histories, joined in order.

    At module level rather than nested in :func:`join_attempts`, so that no function
    object is made per call (each one costs the coverage lane's import hook a compile).

    - ``dtype``: the column's dtype, which an empty join keeps.
    """
    return np.concatenate(
        [np.empty(0, dtype=dtype), *(getattr(part, name) for part in histories)]
    )


def join_attempts(
    chunks: Iterable[tuple[BackendHistory | None, StepStats | None]],
) -> tuple[BackendHistory, StepStats]:
    """Assemble consecutive chunks of attempts and rows into one record.

    The consumer's half of exactly-once chunked delivery (module docstring): a plane's
    or a checkpoint's history is its chunks in delivery order. Each row's ``step`` is
    offset by the attempts of the chunks before its own, and each chunk's wall times by
    the last wall time assembled before it, so the record reads as one span.

    Parameters
    ----------
    chunks : iterable
        ``(history, rows)`` per event, in delivery order; ``None`` for an event that
        carries none.

    Returns
    -------
    tuple
        The assembled history and rows (rows without a batch axis).

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.events import join_attempts
    >>> from solprop.backends.protocol import BackendHistory, StepStats
    >>> def chunk(z, wall):
    ...     n = len(z)
    ...     history = BackendHistory(np.array(z), np.full(n, 0.1), np.zeros(n),
    ...                              np.ones(n, bool), np.array(wall))
    ...     rows = StepStats(np.array(z[-1:]), np.array([n - 1]),
    ...                      {"energy": np.array([1.0])})
    ...     return history, rows
    >>> first, second = chunk([0.1, 0.2], [1.0, 2.0]), chunk([0.3], [1.0])
    >>> history, rows = join_attempts([first, second])
    >>> history.wall_time.tolist(), rows.step.tolist()
    ([1.0, 2.0, 3.0], [1, 2])
    """
    histories: list[BackendHistory] = []
    rows: list[StepStats] = []
    attempts = 0
    wall = 0.0
    for history, chunk_rows in chunks:
        if chunk_rows is not None and chunk_rows.n_rows:
            rows.append(
                StepStats(chunk_rows.z, chunk_rows.step + attempts, chunk_rows.values)
            )
        if history is not None and history.n_steps:
            histories.append(
                BackendHistory(
                    z=history.z,
                    dz=history.dz,
                    error_norm=history.error_norm,
                    accepted=history.accepted,
                    wall_time=history.wall_time + wall,
                )
            )
            attempts += history.n_steps
            wall += float(history.wall_time[-1])

    joined = BackendHistory(
        z=_column(histories, "z"),
        dz=_column(histories, "dz"),
        error_norm=_column(histories, "error_norm"),
        accepted=_column(histories, "accepted", np.bool_),
        wall_time=_column(histories, "wall_time"),
    )
    if not rows:
        return joined, StepStats.empty()
    if len(rows) == 1:
        return joined, rows[0]
    return joined, StepStats(
        np.concatenate([part.z for part in rows]),
        np.concatenate([part.step for part in rows]),
        {
            name: np.concatenate([part.values[name] for part in rows])
            for name in rows[0].values
        },
    )
