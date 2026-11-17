# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The execution-session contract between solprop and a backend (SEP 47 section 5).

Python constructs and validates the scientific problem
(:class:`solprop.translate.problem.PropagationProblem`) and resolves every run policy
into an immutable :class:`solprop.translate.run_request.RunRequest`. A backend prepares
them once (:meth:`solprop.backends.protocol.Backend.prepare`), and the prepared
execution starts one
:class:`RunSession` per trajectory. The session is the only owner of that trajectory's
mutable numerical state: the field, the method's history, the controller and monitor,
the counters and cursors, and the pending events. Python's session loop
(:mod:`solprop.driver.session_loop`) advances it, services the events it yields --
files, callbacks, warnings, progress -- and acknowledges them::

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

:func:`foreign_request` and :func:`check_preparation` are the refusals a preparation
makes before any session starts (section 5.2), on both backends.
"""

from __future__ import annotations

import enum
import operator
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, SupportsIndex, cast, runtime_checkable

import numpy as np
from jaxtyping import Complex

from solprop.backends.events import (
    EventBatch,
    ExportTicket,
    Outcome,
    Receipt,
)
from solprop.backends.export import ExportRecipe
from solprop.backends.protocol import BackendHistory, StepStats
from solprop.exceptions import BackendError
from solprop.log import StopFlag
from solprop.noise import NoiseRealisation
from solprop.output.records import MarchSnapshot
from solprop.stats import CATALOGUE, PLASMA_INTEGRATION_STATISTICS
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

__all__ = [
    "DEFAULT_DEADLINE",
    "DEFAULT_WORK_LIMIT",
    "AttemptListener",
    "ExecutionOptions",
    "PreparedExecution",
    "RunSession",
    "SessionError",
    "SessionState",
    "SessionStatus",
    "SnapshotReceipt",
    "Start",
    "StopToken",
    "check_preparation",
    "foreign_request",
    "positive_count",
    "statistics_unavailable",
]

DEFAULT_WORK_LIMIT = 256
"""Attempts one ``advance`` makes at most before yielding, as the native marches do.

Python signal handlers run only between native calls, so a bounded return keeps a
long run interruptible (SEP 47 section 5.5).
"""

DEFAULT_DEADLINE = 0.1
"""Host seconds after which ``advance`` yields at its next safe boundary, s."""


AttemptListener = Callable[[float, float, bool], None]
"""Called with each attempt's end, length and acceptance, for a progress reporter."""


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

    Raises
    ------
    solprop.exceptions.BackendError
        If `node_budget` or `node_chunk` is given and is not a positive integer.

    Examples
    --------
    >>> from solprop.backends.session import ExecutionOptions
    >>> ExecutionOptions(node_budget=2**26).node_budget
    67108864
    """

    node_budget: int | SupportsIndex | None = None
    node_chunk: int | SupportsIndex | None = None

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

    @property
    def budget(self) -> int | None:  # numpydoc ignore=RT01
        """Explicit node budget as an ``int``, or ``None``."""
        return None if self.node_budget is None else operator.index(self.node_budget)

    @property
    def chunk(self) -> int | None:  # numpydoc ignore=RT01
        """Exact node chunk as an ``int``, or ``None``."""
        return None if self.node_chunk is None else operator.index(self.node_chunk)


_MODAL_ONLY = (
    "omega0",
    "energy_window",
    "peak_power",
    "peak_power_window",
    "peak_power_total",
    "fwhm_t",
    "fwhm_t_outer",
    "zdw",
)
"""The statistics of a modal state that a spatial route's columns do not carry: its
temporal Poynting statistics and the modes' zero-dispersion wavelength
(:func:`solprop.backends.spatial.collect_spatial` leaves each of them)."""


def statistics_unavailable(
    problem: PropagationProblem, names: Sequence[str]
) -> str | None:
    """Return why no executor computes some of a problem's statistics, or ``None``.

    What the problem does not carry, refused by name before any session starts, on
    every executor (SEP 47 section 6.3), rather than at the first plane: a modal
    statistic of a spatial route, a spatial statistic of a modal one, the plasma
    integration diagnostics of a problem without the field-grid plasma column
    (SEP 18), and the electron density or the peak ionisation rate of a problem with
    no compiled plasma response. The messages are the reference's at its first plane.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    names : collections.abc.Sequence of str
        The statistics asked for, at planes or between them.

    Returns
    -------
    str or None
        The refusal, or ``None``.

    Examples
    --------
    >>> from solprop.backends.session import statistics_unavailable
    >>> from solprop.driver import Simulation
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech",
    ... )
    >>> problem = simulation.translate()
    >>> statistics_unavailable(problem, ["energy"]) is None
    True
    >>> statistics_unavailable(problem, ["beam_radius"])
    "the 'beam_radius' statistic requires spatial geometry"
    """
    asked = list(dict.fromkeys(names))
    if problem.transform.geometry is not None:
        modal = sorted(set(asked) & set(_MODAL_ONLY))
        if modal:
            return f"statistics {modal} are not spatial saved-field statistics"
    elif problem.transform.modal_tables is None:
        # A general modal guide refuses these by its accessor whitelist, first.
        for name in asked:
            entry = CATALOGUE.get(name)
            if entry is not None and "spatial_geometry" in entry.requires:
                return f"the {name!r} statistic requires spatial geometry"
    responses = problem.responses
    if set(asked) & set(PLASMA_INTEGRATION_STATISTICS) and (
        not responses.plasma_channels or responses.envelope_plasma is not None
    ):
        return (
            "the plasma integration diagnostics read the field-grid plasma column, "
            "and this problem carries none"
        )
    if (
        set(asked) & {"electron_density", "peak_ionisation_rate"}
        and not responses.population_channels
    ):
        return (
            "the electron density and the peak ionisation rate need a compiled "
            "plasma response, and this problem declares none"
        )
    return None


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
    march : solprop.output.records.MarchSnapshot or None
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
    ) -> "Start":
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
    def finished(self) -> bool:  # numpydoc ignore=RT01
        """Whether a terminal event was yielded."""
        ...

    @property
    def closed(self) -> bool:  # numpydoc ignore=RT01
        """Whether the session was closed."""
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

    @property
    def march_refusal(self) -> str | None:  # numpydoc ignore=RT01
        """Why the start's method history could not be installed.

        The reason, or ``None`` when it was installed or the start carried none.
        """
        ...

    @property
    def in_flight(self) -> tuple[float, float] | None:  # numpydoc ignore=RT01
        """In-place step a failure left in flight, or ``None``.

        Its start and length, m, while the state is neither of the step's ends: a
        low-storage step overwrites the state in place (SEP 47 section 5.5).
        """
        ...

    def failure_ticket(self) -> ExportTicket | None:
        """Return a ticket for the committed state after a failure outside the session.

        A service failure -- an output error, a signal during a commit -- leaves the
        session's batch pending, and the ticket is bound to it; when none is pending,
        the session issues a failed terminal batch at its committed state, so that the
        state can be exported for a terminal checkpoint and nothing advances again.
        ``None`` once the session closed, its state was poisoned or left in flight, or
        before it issued a batch it could bind the ticket to.
        """
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

    def __enter__(self) -> "RunSession":
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
class PreparedExecution(Protocol):
    """A backend's lowering of one problem and request, reusable across sessions."""

    @property
    def backend(self) -> str:  # numpydoc ignore=RT01
        """Registered name of the backend that prepared it."""
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


def foreign_request(own: RunRequest, request: RunRequest) -> str | None:
    """Return why `request` is not a run of `own`, or ``None`` when it is.

    A backend lowers its method, error control, replay and itinerary where it is
    built (SEP 47 section 5.2), so a run it serves may differ from its own request
    only where :meth:`~solprop.translate.run_request.RunRequest.for_run` changes it:
    the start, retention, statistics, monitor, history and snapshots. ``for_run``
    keeps the other sections by identity, which is what is checked.

    Parameters
    ----------
    own : solprop.translate.run_request.RunRequest
        The request the backend was built for.
    request : solprop.translate.run_request.RunRequest
        The request one run asks it to serve.

    Returns
    -------
    str or None
        The first section that differs, named, or ``None``.

    Examples
    --------
    >>> from solprop.stepper import DOPRI5, PIController
    >>> from solprop.translate.problem import StepperSettings
    >>> from solprop.translate.run_request import describe_run
    >>> settings = StepperSettings(DOPRI5, PIController(rtol=1e-8, atol=1e-14),
    ...                            1e-9, 0.1, 1e-3, 1.1, None, [0.5, 1.0], "auto")
    >>> own = describe_run(settings)
    >>> foreign_request(own, own.for_run(start_plane=1)) is None
    True
    >>> foreign_request(own, describe_run(settings))
    "the run's integration is not the backend's own"
    """
    sections = (
        ("integration", own.integration, request.integration),
        ("error control", own.error_control, request.error_control),
        ("replay", own.replay, request.replay),
        ("planes", own.observation.save_planes, request.observation.save_planes),
    )
    for name, mine, theirs in sections:
        if mine is not theirs:
            return f"the run's {name} is not the backend's own"
    return None


def check_preparation(
    name: str, own: RunRequest, problem: PropagationProblem, request: RunRequest
) -> None:
    """Refuse a run a constructed backend cannot prepare, by name (SEP 47 5.2).

    Parameters
    ----------
    name : str
        The backend's registered name, for the message.
    own : solprop.translate.run_request.RunRequest
        The request the backend was built for.
    problem : solprop.translate.problem.PropagationProblem
        The run's problem.
    request : solprop.translate.run_request.RunRequest
        The run's request.

    Raises
    ------
    solprop.backends.session.SessionError
        If the request is not one run of the backend's own (:func:`foreign_request`).
    solprop.exceptions.BackendError
        If the request asks for statistics the problem does not carry
        (:func:`~solprop.backends.session.statistics_unavailable`).
    """
    foreign = foreign_request(own, request)
    if foreign is not None:
        msg = f"{name} cannot prepare this run: {foreign}"
        raise SessionError(msg)
    stats = request.observation.stats
    unavailable = statistics_unavailable(problem, (*stats.names, *stats.periodic))
    if unavailable is not None:
        # Refused by name before any session starts, on every executor.
        msg = f"{name} cannot prepare this run: {unavailable}"
        raise BackendError(msg)
