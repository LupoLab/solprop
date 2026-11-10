# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The execution-session contract between solprop and a backend (SEP 47 section 5).

Python constructs and validates the scientific problem
(:class:`solprop.translate.problem.PropagationProblem`) and resolves every run policy
into an immutable :class:`solprop.translate.run_request.RunRequest`. A backend prepares
them once (:meth:`SessionBackend.prepare`, which every backend offers as
``Backend.prepare``), and the prepared execution starts one
:class:`RunSession` per trajectory. The session is the only owner of that trajectory's
mutable numerical state: the field, the method's history, the controller and monitor,
the counters and cursors, and the pending events. Python's service loop
(:mod:`solprop.driver.service`) advances it, services the events it yields -- files,
callbacks, warnings, progress -- and acknowledges them::

    prepared = backend.prepare(problem, request)
    with prepared.start(start, stop=token) as session:
        while not session.finished:
            batch = session.advance(work_limit=256, deadline=0.1)
            service(batch)
            session.acknowledge(batch.receipt)

Yields are exact transport no-ops (SEP 47 section 5.1): the attempt bound, the
deadline, the event capacity and the timing of acknowledgments change when control
returns, never the numerical state or the control trace.

Event storage is bounded (SEP 47 section 5.3): every event delivers, exactly once,
the attempts and periodic rows made since the *previous event* -- a yield's
:class:`~solprop.backends.events.Progress` as much as an observation, a snapshot or
the terminal event -- and a row's ``step`` indexes its own chunk's attempts. A session
therefore buffers no more than one advance's attempts, and the service loop, not the
session, assembles a plane's or a checkpoint's history from the chunks
(:func:`solprop.backends.events.join_attempts`). The status counts every attempt once,
as it is drained.

Control requests travel separately from the immutable request: a thread-safe
:class:`StopToken` asks for a stop at the next safe boundary, and
:meth:`RunSession.request_snapshot` latches a snapshot for the next eligible accepted
endpoint (SEP 47 section 5.5).

A :class:`GroupSession` advances compatible trajectories through one common fixed
schedule, a window at a time, and reports each member's own events and outcome (SEP 47
section 6.7).

:class:`SessionLedger` is the lifecycle bookkeeping every Python session shares:
receipts, the pending barrier, closing and poisoning. It holds no numerical state.
"""

from __future__ import annotations

import enum
import itertools
import operator
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Protocol, SupportsIndex, cast, runtime_checkable

import numpy as np
from jaxtyping import Complex

from solprop.backends.events import (
    EventBatch,
    ExportTicket,
    GroupBatch,
    Outcome,
    Receipt,
)
from solprop.backends.export import ExportRecipe
from solprop.backends.protocol import BackendHistory, StepStats
from solprop.exceptions import BackendError
from solprop.log import StopFlag
from solprop.noise import NoiseRealisation
from solprop.output.schema import MarchSnapshot
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

__all__ = [
    "DEFAULT_DEADLINE",
    "DEFAULT_WORK_LIMIT",
    "ExecutionOptions",
    "GroupSession",
    "PreparedExecution",
    "RunSession",
    "SessionBackend",
    "SessionError",
    "SessionLedger",
    "SessionState",
    "SessionStatus",
    "SnapshotReceipt",
    "Start",
    "StopToken",
    "check_group",
    "check_member_ticket",
    "positive_count",
]

DEFAULT_WORK_LIMIT = 256
"""Attempts one ``advance`` makes at most before yielding, as the native marches do.

Python signal handlers run only between native calls, so a bounded return keeps a
long run interruptible (SEP 47 section 5.5).
"""

DEFAULT_DEADLINE = 0.1
"""Host seconds after which ``advance`` yields at its next safe boundary, s."""


class SessionError(BackendError):
    """A session was used outside its lifecycle.

    Raised for a stale or foreign receipt, an advance across an unacknowledged barrier,
    use after close or after a terminal outcome, or an export with an expired ticket.
    """


class StopToken:
    """A thread-safe request to stop a run at its next safe boundary.

    Setting it from any thread -- a GUI's stop button, a signal handler on the main
    thread, a scan's supervisor -- asks every session attached to it to finish the
    attempt in hand, complete that endpoint's mandatory accounting, and end with
    :attr:`~solprop.backends.events.Outcome.STOPPED` (SEP 47 section 5.5). It
    implements :class:`solprop.log.StopFlag`.

    Examples
    --------
    >>> from solprop.backends.session import StopToken
    >>> token = StopToken()
    >>> token.is_set()
    False
    >>> token.set()
    >>> token.is_set()
    True
    """

    def __init__(self) -> None:
        """Create a token that is not set."""
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._listeners: list[Callable[[], None]] = []

    def set(self) -> None:
        """Request a stop, and forward it to every attached session."""
        with self._lock:
            self._event.set()
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener()

    def clear(self) -> None:
        """Withdraw the request, for a token reused between runs."""
        self._event.clear()

    def is_set(self) -> bool:
        """Return whether a stop was requested.

        Returns
        -------
        bool
            True once :meth:`set` was called and not cleared.
        """
        return self._event.is_set()

    def attach(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call `listener` whenever the token is set, and now if it already is.

        Native sessions attach the store of their own atomic flag, so that a stop set
        from another thread reaches a march that holds no Python lock.

        Parameters
        ----------
        listener : callable
            Called with no arguments. It must not raise and must not block.

        Returns
        -------
        callable
            Detaches the listener.
        """
        with self._lock:
            self._listeners.append(listener)
            already = self._event.is_set()

        def detach() -> None:
            """Remove the listener; a second call does nothing."""
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        if already:
            listener()
        return detach


@dataclass(frozen=True)
class ExecutionOptions:
    """Execution choices that are not part of the numerical problem.

    Two runs may share their scientific description while differing in these (SEP 47
    section 4.3). They are recorded as execution metadata, never compared by a restart.

    Attributes
    ----------
    node_budget : int or None
        An explicit workspace budget for node chunks, bytes: any integral value
        (``operator.index``, a NumPy integer too), held as an ``int``. ``None`` leaves
        the backend's automatic choice. An explicit budget is authoritative: the backend
        plans its chunk from it and no machine rule overrides it
        (:func:`solprop.backends.node_chunk.plan_node_chunk`).
    node_chunk : int or None
        An exact node chunk, integral and held as an ``int``, which overrides both the
        budget and the automatic choice: a measurement's or a test's control, not a user
        setting. A backend takes at most every node, and on the planar kind whole
        transverse rows.
    backend_options : collections.abc.Mapping
        Backend-specific execution choices, copied into an owned mapping. A backend
        validates its keys and values before allocation. Outside scientific and
        restart identities; recorded as execution metadata.

    Raises
    ------
    solprop.exceptions.BackendError
        If `node_budget` or `node_chunk` is given and is not a positive integer,
        or a backend option key is not a string.

    Examples
    --------
    >>> from solprop.backends.session import ExecutionOptions
    >>> ExecutionOptions(node_budget=2**26).node_budget
    67108864
    """

    node_budget: int | SupportsIndex | None = None
    node_chunk: int | SupportsIndex | None = None
    backend_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Hold an integral budget or chunk as an ``int``; refuse any other value.

        Raises
        ------
        solprop.exceptions.BackendError
            Naming the offending option.
        """
        for name in ("node_budget", "node_chunk"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, positive_count(name, value))
        if any(not isinstance(key, str) for key in self.backend_options):
            raise BackendError("backend_options keys must be strings")
        object.__setattr__(self, "backend_options", dict(self.backend_options))

    def check_backend_options(
        self, backend: str, supported: Sequence[str] | frozenset[str] = ()
    ) -> None:
        """Refuse unrecognized backend options before constructing a workspace.

        Parameters
        ----------
        backend : str
            The concrete backend name, for the diagnostic.
        supported : sequence of str or frozenset of str, optional
            Supported keys. The backend separately validates their values.

        Raises
        ------
        BackendError
            Naming every unsupported key.
        """
        unsupported = set(self.backend_options) - set(supported)
        if unsupported:
            raise BackendError(
                f"{backend} does not support backend_options {sorted(unsupported)}"
            )

    @property
    def budget(self) -> int | None:
        """Return the explicit node budget as an ``int``, or ``None``.

        Returns
        -------
        int or None
            :attr:`node_budget`, which construction already made an ``int``.
        """
        return None if self.node_budget is None else operator.index(self.node_budget)

    @property
    def chunk(self) -> int | None:
        """Return the exact node chunk as an ``int``, or ``None``.

        Returns
        -------
        int or None
            :attr:`node_chunk`, which construction already made an ``int``.
        """
        return None if self.node_chunk is None else operator.index(self.node_chunk)


def check_group(
    count: int,
    problems: Sequence[PropagationProblem],
    requests: Sequence[RunRequest],
    starts: Sequence[Start],
) -> None:
    """Refuse a group's inputs that no group implementation starts from.

    A group (:class:`GroupSession`) takes one problem, request and start per member,
    and starts every member at its entrance, with no method history or monitor record
    to continue (SEP 47 section 6.7).

    Parameters
    ----------
    count : int
        The members given.
    problems : collections.abc.Sequence of solprop.translate.problem.PropagationProblem
        Each member's problem.
    requests : collections.abc.Sequence of solprop.translate.run_request.RunRequest
        Each member's run request.
    starts : collections.abc.Sequence of Start
        Each member's start.

    Raises
    ------
    SessionError
        If the sequences differ in length or are empty, entrances alias, or a start
        has no field or is not a fresh entrance whose launch plane is owed.
    """
    if count == 0 or not count == len(problems) == len(requests) == len(starts):
        msg = (
            f"a group needs one problem, request and start per member, and got "
            f"{count} members, {len(problems)} problems, {len(requests)} "
            f"requests and {len(starts)} starts"
        )
        raise SessionError(msg)
    if len({id(start) for start in starts}) != count:
        raise SessionError("a group requires a distinct entrance for each member")
    for start in starts:
        if start.field is None:
            raise SessionError("a group cannot reuse an entrance whose field was taken")
        fresh = start.march is None and start.monitor is None
        if not fresh or start.plane_index or not start.plane_pending:
            msg = (
                "a group starts its members at their entrance, its launch plane owed, "
                "with no method history or monitor record to continue"
            )
            raise SessionError(msg)


def check_member_ticket(ticket: ExportTicket, member: int) -> None:
    """Refuse a group ticket presented for another member than the one it names.

    Parameters
    ----------
    ticket : solprop.backends.events.ExportTicket
        A ticket of a group's pending batch.
    member : int
        The member it is presented for.

    Raises
    ------
    SessionError
        If the ticket names another member, or none.
    """
    if ticket.member != member:
        msg = (
            f"the export ticket for z = {ticket.z!r} m is member {ticket.member}'s, "
            f"not member {member}'s"
        )
        raise SessionError(msg)


def positive_count(name: str, value: object) -> int:
    """Return an integral, positive budget or chunk as an ``int``.

    Parameters
    ----------
    name : str
        ``"node_budget"`` or ``"node_chunk"``, for the message.
    value : object
        Anything ``operator.index`` accepts, such as an ``int`` or a NumPy integer.

    Returns
    -------
    int
        The value.

    Raises
    ------
    solprop.exceptions.BackendError
        If `value` is a ``bool``, not integral, or not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.session import positive_count
    >>> positive_count("node_budget", np.int64(2**26))
    67108864
    """
    unit = " of bytes" if name == "node_budget" else ""
    msg = f"{name} must be a positive number{unit}, got {value!r}"
    if isinstance(value, bool | np.bool_):
        raise BackendError(msg)
    try:
        count = operator.index(cast("SupportsIndex", value))
    except TypeError:
        raise BackendError(msg) from None
    if count < 1:
        raise BackendError(msg)
    return count


@dataclass(eq=False)
class Start:
    """Where a session starts: an initial state, or a checkpoint being resumed.

    A start is handed to one session, which takes its field (:meth:`take_field`): the
    session loads a copy made once at start, and the start forgets the array, so an
    input that nothing else holds -- a runner's caller's array -- is released before
    the first step rather than held through the run by every frame that passed the
    start along (SEP 47 section 5.4).

    Attributes
    ----------
    field : numpy.ndarray or None
        The state in internal amplitude units and the FFT convention, complex,
        shape ``(n_modes, n_pol, n_omega)``; ``None`` once a session took it.
        Mutating the caller's array after the start changes nothing.
    z : float
        Its position, m.
    noise : solprop.noise.NoiseRealisation or None
        The run's noise realisation, when a noise model is active.
    plane_index : int
        The first save plane strictly beyond `z`.
    march : solprop.output.schema.MarchSnapshot or None
        A method history to restore, when resuming one that was retained.
    ledgers : collections.abc.Mapping or None
        Persisted call ledgers to restore (TDSE and budgeted sources).
    monitor : collections.abc.Mapping or None
        When resuming a monitored run, the monitor's continuation: ``"record"``, its
        persisted record; ``"plane"``, whether the resumed position is a saved plane;
        ``"unwatched"``, the spans the earlier executions left unwatched.
    plane_pending : bool
        Whether the plane at `z` is owed (the launch plane of a fresh run, or a resumed
        position whose plane was not saved).
    resume : bool
        Whether `field` was loaded from a plane the run already saved, so that an
        ``input`` noise realisation is not added a second time.
    """

    field: Complex[np.ndarray, "n_modes n_pol n_omega"] | None
    z: float
    noise: NoiseRealisation | None = None
    plane_index: int = 0
    march: MarchSnapshot | None = None
    ledgers: Mapping[str, object] | None = None
    monitor: Mapping[str, object] | None = None
    plane_pending: bool = True
    resume: bool = False

    @classmethod
    def entrance(
        cls, problem: PropagationProblem, *, noise: NoiseRealisation | None = None
    ) -> "Start":  # noqa: UP037 - runtime type hook
        """Return the fresh start of a problem: its launch state, its launch plane owed.

        Parameters
        ----------
        problem : solprop.translate.problem.PropagationProblem
            The translated problem.
        noise : solprop.noise.NoiseRealisation or None, optional
            A realisation to carry instead of the one the problem was assembled with.

        Returns
        -------
        Start
            The start at ``problem.initial.z``, before the first save plane.
        """
        return cls(
            field=problem.initial.field0, z=float(problem.initial.z), noise=noise
        )

    def take_field(self) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Hand the field to the session starting here, and forget it.

        Returns
        -------
        numpy.ndarray
            The field, shape ``(n_modes, n_pol, n_omega)``; the caller copies it.

        Raises
        ------
        SessionError
            If a session already took it: a start is used once.
        """
        field = self.field
        if field is None:
            msg = "this start was already used: a start is handed to one session"
            raise SessionError(msg)
        self.field = None
        return field


class SessionState(enum.Enum):
    """Where a session is in its lifecycle.

    Attributes
    ----------
    READY : str
        It can advance.
    PENDING : str
        A batch is out; it advances again once that batch is acknowledged.
    FINISHED : str
        It yielded a terminal event; nothing more can happen but close.
    CLOSED : str
        Its resources are released.
    """

    READY = "ready"
    PENDING = "pending"
    FINISHED = "finished"
    CLOSED = "closed"


@dataclass(frozen=True)
class SessionStatus:
    """A compact, read-only view of a session, for reporting.

    Attributes
    ----------
    z : float
        The committed position, m.
    plane_index : int
        The next save plane to observe.
    attempts : int
        Attempts made, accepted and rejected.
    rejected : int
        Rejected attempts among them.
    state : SessionState
        Lifecycle state.
    outcome : solprop.backends.events.Outcome or None
        The terminal outcome, once finished.
    """

    z: float
    plane_index: int
    attempts: int
    rejected: int
    state: SessionState
    outcome: Outcome | None = None


@dataclass(frozen=True)
class SnapshotReceipt:
    """Identifies a snapshot request until it is fulfilled or refused.

    Attributes
    ----------
    request : int
        Sequence number of the request in its session, from 1.
    coalesced : bool
        Whether it joined an equivalent request still outstanding, so that one
        snapshot fulfils both (SEP 47 section 5.5 makes coalescing explicit).
    """

    request: int
    coalesced: bool


@runtime_checkable
class RunSession(Protocol):
    """One trajectory's numerical execution, owned by its backend.

    See :mod:`solprop.backends.session` for the loop that drives it.
    """

    @property
    def finished(self) -> bool:
        """Return whether a terminal event was yielded."""
        ...

    @property
    def closed(self) -> bool:
        """Return whether the session was closed."""
        ...

    def status(self) -> SessionStatus:
        """Return a read-only summary of the session."""
        ...

    def advance(
        self,
        *,
        work_limit: int = DEFAULT_WORK_LIMIT,
        deadline: float = DEFAULT_DEADLINE,
    ) -> EventBatch:
        """Run until a barrier, the attempt bound or the deadline, and report it.

        Parameters
        ----------
        work_limit : int, optional
            Attempts before a yield.
        deadline : float, optional
            Host seconds before a yield, s.
        """
        ...

    def acknowledge(self, receipt: Receipt) -> None:
        """Release a batch and allow advancing across its barrier.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The pending batch's receipt.
        """
        ...

    def export(
        self,
        ticket: ExportTicket,
        recipe: ExportRecipe,
        out: Complex[np.ndarray, "n_modes n_pol n_omega"] | None = None,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Fill a destination with a pending observation's converted field.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The pending observation's ticket.
        recipe : solprop.backends.export.ExportRecipe
            The conversion.
        out : numpy.ndarray or None, optional
            The destination; a new array by default.
        """
        ...

    def noise_phase(self, ticket: ExportTicket) -> np.ndarray | None:
        """Return the noise phase at a pending observation, when a noise model runs.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The pending observation's ticket.
        """
        ...

    def request_snapshot(self) -> SnapshotReceipt:
        """Latch a snapshot for the next eligible accepted endpoint."""
        ...

    def march_snapshot(self, ticket: ExportTicket) -> MarchSnapshot | None:
        """Return the method history at a pending observation, when retained.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The pending observation's ticket.
        """
        ...

    def march_snapshot_nbytes(self) -> int | None:
        """Return the size a method-history snapshot would have, without making it."""
        ...

    def execution_record(self) -> Mapping[str, object]:
        """Return the session's execution facts for the run's execution record."""
        ...

    def monitor_record(self) -> Mapping[str, object] | None:
        """Return the boundary monitor's record, or ``None`` without a monitor."""
        ...

    def continue_monitor(self, continuation: Mapping[str, object]) -> None:
        """Continue a resumed run's boundary-monitor record, before it watches the run.

        As :attr:`Start.monitor`, for a service loop that reads the earlier
        execution's record from its output after the session started.

        Parameters
        ----------
        continuation : collections.abc.Mapping
            The continuation, as :attr:`Start.monitor` holds it.
        """
        ...

    def undelivered(self, ticket: ExportTicket) -> tuple[BackendHistory, StepStats]:
        """Deliver the attempts and rows made since the previous event, once.

        For a terminal checkpoint after a failure outside the session, whose attempts
        no event delivered; empty when every attempt was delivered.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The checkpoint's ticket.

        Returns
        -------
        tuple
            The attempts and the periodic rows.
        """
        ...

    def pending_ticket(self) -> ExportTicket | None:
        """Return a ticket for the committed state after a failure outside the session.

        ``None`` when nothing can be exported: the session is closed or poisoned, or
        no batch is pending to bind the ticket to.
        """
        ...

    def close(self) -> None:
        """Release the session's resources; idempotent."""
        ...

    def __enter__(self) -> "RunSession":  # noqa: UP037 - runtime type hook
        """Return the session, for scoped ownership."""
        ...

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Close the session.

        Parameters
        ----------
        kind : type or None
            The type of the exception leaving the scope, if any.
        error : BaseException or None
            The exception, if any.
        trace : types.TracebackType or None
            Its traceback, if any.
        """
        ...


@runtime_checkable
class GroupSession(Protocol):
    """Compatible trajectories advanced together through one common fixed schedule.

    A group request is a sequence of independently translated member problems that
    pass the compatibility predicate a batched scan groups its points by; the
    common schedule's maximum error norm is a diagnostic only (SEP 47 section 6.7).
    Each :meth:`advance` runs every member still running through the next common
    schedule window -- to the next save plane -- and reports each member's own
    events (:class:`~solprop.backends.events.GroupBatch`)::

        with group:
            while not group.finished:
                batch = group.advance()
                service(batch)                 # each member's planes and outcome
                group.acknowledge(batch.receipt)

    **Failure.** A boundary error, or a boundary warning raised as an error, ends
    that member alone with a :class:`~solprop.backends.events.Terminal`, and the
    others go on. A non-finite error norm, a step floor, a trial exception or an
    in-flight failure ends the group: every member still running ends with the same
    error. Members are stepped in turn, so earlier members may already have advanced
    the failed scheduled step: each terminal reports that member's actual committed
    position and checkpoint eligibility, and no common cursor or rollback is
    invented. A group-wide stop request ends every running member at a common
    accepted endpoint.
    """

    @property
    def n_members(self) -> int:
        """Return the number of members."""
        ...

    @property
    def finished(self) -> bool:
        """Return whether every member has ended."""
        ...

    @property
    def closed(self) -> bool:
        """Return whether the group was closed."""
        ...

    def advance(self) -> GroupBatch:
        """Run every running member through the next common window, and report it."""
        ...

    def acknowledge(self, receipt: Receipt) -> None:
        """Release a window's batch and allow advancing past it.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The batch's receipt.
        """
        ...

    def export(
        self,
        member: int,
        ticket: ExportTicket,
        recipe: ExportRecipe,
        out: Complex[np.ndarray, "n_modes n_pol n_omega"] | None = None,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Fill a destination with one member's converted field at a pending ticket.

        Parameters
        ----------
        member : int
            The member's index, from zero.
        ticket : solprop.backends.events.ExportTicket
            The member's pending ticket.
        recipe : solprop.backends.export.ExportRecipe
            The conversion.
        out : numpy.ndarray or None, optional
            The destination; a new array by default.

        Returns
        -------
        numpy.ndarray
            The filled destination.
        """
        ...

    def noise_phase(self, member: int, ticket: ExportTicket) -> np.ndarray | None:
        """Return one member's noise phase at a pending ticket.

        Parameters
        ----------
        member : int
            The member's index, from zero.
        ticket : solprop.backends.events.ExportTicket
            The member's pending ticket.

        Returns
        -------
        numpy.ndarray or None
            The phase, or ``None`` without noise.
        """
        ...

    def member_status(self, member: int) -> SessionStatus:
        """Return one member's committed position, counters and outcome.

        Parameters
        ----------
        member : int
            The member's index, from zero.

        Returns
        -------
        SessionStatus
            The member's status.
        """
        ...

    def execution_record(self, member: int) -> Mapping[str, object]:
        """Return one member's execution facts.

        Parameters
        ----------
        member : int
            The member's index, from zero.

        Returns
        -------
        collections.abc.Mapping
            The member's execution record.
        """
        ...

    def monitor_record(self, member: int) -> Mapping[str, object] | None:
        """Return one member's boundary-monitor record, ``None`` when unwatched.

        Parameters
        ----------
        member : int
            The member's index, from zero.

        Returns
        -------
        collections.abc.Mapping or None
            The record.
        """
        ...

    def close(self) -> None:
        """Release every member's resources; idempotent."""
        ...

    def __enter__(self) -> "GroupSession":  # noqa: UP037 - runtime type hook
        """Return the group, for scoped ownership."""
        ...

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Close the group.

        Parameters
        ----------
        kind : type or None
            The type of the exception leaving the scope, if any.
        error : BaseException or None
            The exception, if any.
        trace : types.TracebackType or None
            Its traceback, if any.
        """
        ...


@runtime_checkable
class PreparedExecution(Protocol):
    """A backend's lowering of one problem and request, reusable across sessions."""

    @property
    def backend(self) -> str:
        """Return the registered name of the backend that prepared it."""
        ...

    def start(
        self,
        start: Start,
        *,
        listener: Callable[[float, float, bool], None] | None = None,
        started: Callable[[float, float, int], None] | None = None,
        stop: StopFlag | None = None,
        owns: bool = True,
    ) -> RunSession:
        """Start a session from an initial state or a checkpoint.

        Parameters
        ----------
        start : Start
            The initial state or checkpoint the session starts from.
        listener : callable or None, optional
            Called with each attempt's end, length and acceptance as it is made, for a
            progress reporter; it observes only.
        started : callable or None, optional
            Called with the start, the end and the plane count when propagation
            begins.
        stop : solprop.log.StopFlag or None, optional
            The run's stop token.
        owns : bool, optional
            Whether closing the session closes the backend.
        """
        ...

    def metadata(self) -> Mapping[str, object]:
        """Return the execution metadata recorded with the run."""
        ...

    def memory_ledger(self) -> Mapping[str, int]:
        """Return every retained buffer by name, bytes."""
        ...

    def close(self) -> None:
        """Release the prepared resources; idempotent."""
        ...


class SessionBackend(Protocol):
    """A backend that prepares executions (SEP 47 section 6.1).

    Every :class:`~solprop.backends.protocol.ExecutionBackend` is one: it is built for
    a problem and a request (:func:`solprop.backends.registry.select`). Its ``prepare``
    binds the problem and the request one run serves.
    """

    name: str

    def prepare(
        self,
        problem: PropagationProblem,
        request: RunRequest,
    ) -> PreparedExecution:
        """Lower a problem and request; refuse an unsupported one before allocating.

        Parameters
        ----------
        problem : solprop.translate.problem.PropagationProblem
            The translated problem.
        request : solprop.translate.run_request.RunRequest
            What the run asks of the backend.
        """
        ...


_SESSION_IDS = itertools.count(1)


class SessionLedger:
    """The lifecycle bookkeeping of a Python session; no numerical state.

    It issues receipts, records the pending barrier, refuses stale or foreign
    receipts and advancing across an unacknowledged barrier, and remembers whether the
    session finished, failed or closed.

    Examples
    --------
    >>> from solprop.backends.session import SessionLedger
    >>> ledger = SessionLedger()
    >>> receipt = ledger.issue()
    >>> ledger.ticket_valid(receipt)
    True
    >>> ledger.acknowledge(receipt)
    >>> ledger.ticket_valid(receipt)
    False
    """

    def __init__(self) -> None:
        """Start a ready session's ledger."""
        self.session = next(_SESSION_IDS)
        self._sequence = 0
        self._pending: Receipt | None = None
        self.state = SessionState.READY
        self.outcome: Outcome | None = None

    def check_advance(self) -> None:
        """Refuse to advance a session that is pending, finished or closed.

        Raises
        ------
        SessionError
            Naming the state that forbids it.
        """
        if self.state is SessionState.PENDING:
            msg = "acknowledge the pending batch before advancing the session again"
            raise SessionError(msg)
        if self.state is SessionState.FINISHED:
            msg = "the session has ended and cannot advance"
            raise SessionError(msg)
        self.check_open()

    def check_open(self) -> None:
        """Refuse a closed session.

        Raises
        ------
        SessionError
            If the session was closed.
        """
        if self.state is SessionState.CLOSED:
            msg = "the session is closed"
            raise SessionError(msg)

    def issue(self, *, outcome: Outcome | None = None) -> Receipt:
        """Issue the receipt of a new pending batch.

        Parameters
        ----------
        outcome : solprop.backends.events.Outcome or None, optional
            The batch's terminal outcome, when it ends the session.

        Returns
        -------
        solprop.backends.events.Receipt
            The batch's receipt.
        """
        self._sequence += 1
        receipt = Receipt(self.session, self._sequence)
        self._pending = receipt
        self.state = SessionState.PENDING
        if outcome is not None:
            self.outcome = outcome
        return receipt

    def acknowledge(self, receipt: Receipt) -> None:
        """Accept the pending batch's receipt.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The receipt to acknowledge.

        Raises
        ------
        SessionError
            If the receipt is not the pending one: foreign, stale or repeated.
        """
        self.check_open()
        if receipt != self._pending:
            msg = (
                f"receipt {receipt} is not the pending batch's "
                f"({self._pending}): a foreign, stale or repeated acknowledgment"
            )
            raise SessionError(msg)
        self._pending = None
        self.state = (
            SessionState.FINISHED if self.outcome is not None else SessionState.READY
        )

    def ticket_valid(self, receipt: Receipt) -> bool:
        """Return whether tickets of a batch can still be exported.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The batch's receipt.

        Returns
        -------
        bool
            True while that batch is the pending one.
        """
        return self.state is not SessionState.CLOSED and receipt == self._pending

    def check_ticket(self, ticket: ExportTicket) -> None:
        """Refuse an export with an expired or foreign ticket.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The ticket presented.

        Raises
        ------
        SessionError
            If its batch is no longer pending.
        """
        if not self.ticket_valid(ticket.receipt):
            msg = (
                f"the export ticket for z = {ticket.z!r} m has expired: its batch was "
                "acknowledged, or it belongs to another session"
            )
            raise SessionError(msg)

    def close(self) -> bool:
        """Mark the session closed.

        Returns
        -------
        bool
            True the first time, False when it was already closed.
        """
        if self.state is SessionState.CLOSED:
            return False
        self.state = SessionState.CLOSED
        self._pending = None
        return True

    @property
    def pending(self) -> Receipt | None:
        """Return the pending batch's receipt.

        Returns
        -------
        solprop.backends.events.Receipt or None
            The receipt, or ``None`` when nothing is pending.
        """
        return self._pending

    @property
    def finished(self) -> bool:
        """Return whether a terminal batch was issued.

        Returns
        -------
        bool
            True once the session yielded its terminal event.
        """
        return self.outcome is not None
