# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The records an execution session hands its service loop (SEP 47 section 5.3).

A session (:class:`solprop.backends.session.RunSession`) owns the numerical state
of one trajectory. It runs until an external service is needed or a bounded yield
falls due, and reports what happened as a short batch of immutable events. Events are
completed work or explicit service requests; none of them is a second live
controller. Four kinds exist, and the set is closed:

- :class:`ObservationReady`: a requested plane, its statistics, the attempts since
  the previous observation and, while the batch is pending, an export ticket for its
  field;
- :class:`ServiceRequired`: an operation the caller must complete before the session
  advances further, such as delivering a warning or persisting a snapshot;
- :class:`Progress`: attempts and accounting since the previous event, with no field;
- :class:`Terminal`: the outcome of the run, its committed position and whether that
  position can be resumed.

A batch (:class:`EventBatch`) carries a :class:`Receipt`. Acknowledging the receipt
releases the batch's resources, invalidates its tickets and lets the session advance
across the barrier the batch ended at. A receipt is good once: acknowledging it twice,
or acknowledging an older one, is refused, so no barrier is crossed twice.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from solprop.backends.protocol import BackendHistory, StepStats, StepSummary

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

__all__ = [
    "Eligibility",
    "Event",
    "EventBatch",
    "ExportTicket",
    "ObservationKind",
    "ObservationReady",
    "Outcome",
    "Progress",
    "Receipt",
    "ServiceKind",
    "ServiceRequired",
    "Terminal",
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
    """

    receipt: Receipt
    generation: int
    z: float


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
        Every attempt since the previous observation, accepted and rejected.
    step_stats : solprop.backends.protocol.StepStats
        The periodic statistics rows sampled since the previous observation.
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
        For a snapshot, the committed field to persist; ``None`` otherwise.
    history : solprop.backends.protocol.BackendHistory or None
        For a snapshot, the attempts since the previous observation, included in the
        running checkpoint without being consumed (SEP 47 section 5.5).
    step_stats : solprop.backends.protocol.StepStats or None
        For a snapshot, the periodic rows since the previous observation, likewise.
    """

    kind: ServiceKind
    z: float
    message: str
    category: type[Warning] | None = None
    ticket: ExportTicket | None = None
    history: BackendHistory | None = None
    step_stats: StepStats | None = None


@dataclass(frozen=True, eq=False)
class Progress:
    """Attempts since the previous event, with no field.

    Attributes
    ----------
    z : float
        Committed position, m.
    history : solprop.backends.protocol.BackendHistory
        The attempts made since the previous event, for a progress reporter. Delivery
        changes nothing numerical; the next observation's history still holds them.
    """

    z: float
    history: BackendHistory


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
        The attempts since the last observation, which a terminal checkpoint carries;
        ``None`` when there are none to report.
    step_stats : solprop.backends.protocol.StepStats or None
        The periodic rows since the last observation, likewise.
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
    def terminal(self) -> Terminal | None:
        """Return the batch's terminal event, if it ended the session.

        Returns
        -------
        Terminal or None
            The last event when it is terminal; ``None`` otherwise.
        """
        if self.events and isinstance(self.events[-1], Terminal):
            return self.events[-1]
        return None
