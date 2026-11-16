# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference's run session: the schedule of a run (SEP 47, SEP 52).

:class:`HostSession` is the :class:`~solprop.backends.session.RunSession` that serves
every ``numpy_ref`` run, and the Python counterpart of native's ``session.cpp``, whose
scheduling decisions are this module's transcribed. It owns the run's schedule: the
save planes and step planes, the natural stops, the bounded yields, the landing
targets, the halts and the snapshot latch, the dense observation of a plane a step
crossed (through the backend's :class:`Dense`), the periodic statistics rows, the
boundary monitor's and the transverse edge's checks with the split collar's sums, the
quadrature alarm's report, the load of the start with its noise, the events and the
exports. It drives the run's march directly (:class:`HostMethod`), as the march's
:class:`~solprop.backends.host.marches.MarchHost`; the state and every state-sized
operation are the backend's workspace's
(:class:`~solprop.backends.numpy_ref.workspace.NumpyWorkspace`).

**Boundary warnings are barriers.** The monitor's and the edge's warnings go into a
sink (:attr:`solprop.monitor.BoundaryMonitor.sink`) rather than being issued where they
arise, and each is reported, in order, as a ``WARNING`` barrier event at the committed
endpoint it arose at, whose consumer emits it
(:func:`~solprop.backends.events.emit_warning`): its filters may raise it, which
fails the run there. Once allowed, the next advance does the rest of that point's work,
once, as native's session does.

:class:`HostPrepared` is the prepared execution (SEP 47 section 5.2) a reference backend
returns from ``prepare``; each of its starts opens one session. The session composes the
parts every Python session shares (:mod:`solprop.backends.session_core`) with the
lifecycle ledger (:class:`~solprop.backends.session.SessionLedger`). It imports nothing
of the native backend's.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np

from solprop._interrupts import poll
from solprop._ownership import hold
from solprop.backends.engines import AcceptRecord
from solprop.backends.events import (
    Eligibility,
    Event,
    EventBatch,
    ExportTicket,
    ObservationKind,
    ObservationReady,
    Outcome,
    Progress,
    Receipt,
    ServiceKind,
    ServiceRequired,
    Terminal,
    join_attempts,
)
from solprop.backends.export import ExportRecipe
from solprop.backends.host.adams import HostAdamsMarch
from solprop.backends.host.low_storage import HostLowStorageMarch
from solprop.backends.host.marches import MarchState, in_flight_refusal
from solprop.backends.host.runge_kutta import RungeKuttaMarch
from solprop.backends.protocol import (
    BackendHistory,
    Observation,
    StepStats,
    StepSummary,
)
from solprop.backends.records import collar_rests, plane_plan
from solprop.backends.session import (
    DEFAULT_DEADLINE,
    DEFAULT_WORK_LIMIT,
    RunSession,
    SessionError,
    SessionLedger,
    SessionState,
    SessionStatus,
    SnapshotReceipt,
    Start,
)
from solprop.backends.session_core import (
    EMPTY_SUMMARY,
    AttemptCounter,
    AttemptListener,
    ObservedField,
    SnapshotLatch,
    YieldDue,
    bind,
    check_preparation,
    merge_summaries,
)
from solprop.exceptions import BackendError, BoundaryError, StopRequested
from solprop.log import NOT_ESTIMATED, StopFlag
from solprop.monitor import BoundaryEnergies, BoundaryMonitor
from solprop.output.records import MarchSnapshot
from solprop.stats import StatsSpec
from solprop.stepper.controller import ControlRecord
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.schedule import reached, replay_tolerance
from solprop.translate.modal import QUADRATURE_ALARM, warn_quadrature
from solprop.translate.problem import PropagationProblem, StepperSettings
from solprop.translate.routes import transverse_of
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    # Under TYPE_CHECKING, so that the runtime type checks of the test suite do not
    # check a refused export destination's shape before the session refuses it.
    from jaxtyping import Complex

    from solprop.backends.numpy_ref.workspace import NumpyWorkspace
    from solprop.backends.protocol import Backend

__all__ = [
    "INITIAL_STEP_STATS_CAPACITY",
    "Dense",
    "HistoryRows",
    "HostBackend",
    "HostMethod",
    "HostPrepared",
    "HostSession",
    "natural_stop",
    "start_position",
    "step_cap",
]

logger = logging.getLogger(__name__)

INITIAL_STEP_STATS_CAPACITY = 64
"""Rows a plane's statistics buffer holds before it first has to grow."""


@runtime_checkable
class Dense(Protocol):
    """A march's dense output: what a session interpolates a crossed plane from.

    Native's ``session::Dense``; the reference's are
    :mod:`solprop.backends.numpy_ref.dense`'s.
    """

    def crosses(self, position: float, target: float) -> bool:
        """Return whether a march that returned at `position` stepped across `target`.

        Parameters
        ----------
        position : float
            The committed position, m.
        target : float
            The plane, m.

        Returns
        -------
        bool
            Whether the plane is observed inside the last step.
        """
        ...

    def observe(self, position: float) -> None:
        """Hold the committed state aside; make the state the field at `position`.

        Parameters
        ----------
        position : float
            The plane, m, inside the last accepted step.
        """
        ...

    def restore(self) -> None:
        """Make the held committed state current again."""
        ...


@runtime_checkable
class HistoryRows(Protocol):
    """An Adams history's rows where the state lives, which a snapshot carries."""

    def nbytes(self) -> int:
        """Return the bytes :meth:`export` would copy.

        Returns
        -------
        int
            The size of every row.
        """
        ...

    def export(self) -> dict[str, np.ndarray]:
        """Return copies of the rows, by name.

        Returns
        -------
        dict
            The rows.
        """
        ...

    def install(self, rows: Mapping[str, np.ndarray], shape: tuple[int, ...]) -> None:
        """Install rows :meth:`export` returned, for a state of `shape`.

        Parameters
        ----------
        rows : collections.abc.Mapping
            The rows, by name.
        shape : tuple of int
            The state's shape.
        """
        ...


@dataclass(eq=False)
class HostMethod:
    """The march one session runs, built for it by the backend (native's ``Method``).

    Attributes
    ----------
    march : object
        The march: a :class:`~solprop.backends.host.runge_kutta.RungeKuttaMarch`,
        :class:`~solprop.backends.host.adams.HostAdamsMarch` or
        :class:`~solprop.backends.host.low_storage.HostLowStorageMarch`, its session
        the march's host.
    dense : Dense or None
        Its dense output when the run interpolates its planes; ``None`` otherwise.
    rows : HistoryRows or None
        The Adams history's rows a snapshot carries; ``None`` for the other marches.
    """

    march: RungeKuttaMarch | HostAdamsMarch | HostLowStorageMarch
    dense: Dense | None = None
    rows: HistoryRows | None = None


@runtime_checkable
class HostBackend(Protocol):
    """What the reference's session reads of its backend.

    Attributes
    ----------
    name : str
        The registry name.
    reads_transverse_edge : bool
        Whether a free-space run's transverse edge is read (SEP 40 section 5.3).
    """

    name: str
    reads_transverse_edge: bool

    @property
    def request(self) -> RunRequest:  # numpydoc ignore=RT01
        """Request the backend was built for."""
        ...

    @property
    def workspace(self) -> NumpyWorkspace:  # numpydoc ignore=RT01
        """Workspace that holds the state."""
        ...

    @property
    def closed(self) -> bool:  # numpydoc ignore=RT01
        """Whether the backend has released its buffers."""
        ...

    def method(self, host: object) -> HostMethod:
        """Return a fresh march, and its dense output and rows, reporting to `host`.

        Parameters
        ----------
        host : object
            The session, as the march's
            :class:`~solprop.backends.host.marches.MarchHost`.

        Returns
        -------
        HostMethod
            The method.
        """
        ...

    def metadata(self) -> dict[str, object]:
        """Return the execution's facts.

        Returns
        -------
        dict
            The record.
        """
        ...

    def memory_ledger(self) -> dict[str, int]:
        """Return every retained buffer by name, bytes.

        Returns
        -------
        dict
            The ledger.
        """
        ...

    def close(self) -> None:
        """Release the buffers."""
        ...


def start_position(
    problem: PropagationProblem, stepper: StepperSettings, z: float
) -> tuple[float, int]:
    """Return where a state may be loaded, and a replay's cursor there.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem, whose operator table bounds the position.
    stepper : solprop.translate.problem.StepperSettings
        The run's stepper settings, whose schedule a replay follows.
    z : float
        The requested position, m.

    Returns
    -------
    tuple
        The position, m, and how many steps of a replayed schedule it has consumed
        (zero for an adaptive run).

    Raises
    ------
    BackendError
        If `z` lies outside the operator table, or a schedule is replayed and `z` is
        not one of its edges: every save plane is one (``validate`` checks it), so this
        fires only for a position the run never stopped at.
    """
    low, high = problem.linear.z_span
    if not (math.isfinite(z) and low <= z <= high):
        msg = (
            f"a state can only be loaded inside the operator table, which covers "
            f"[{low:.6g}, {high:.6g}] m, and z = {z!r} does not"
        )
        raise BackendError(msg)
    schedule = stepper.schedule
    if schedule is None:
        return z, 0
    edges = schedule.edges
    index = int(np.argmin(np.abs(edges - z)))
    if abs(float(edges[index]) - z) > replay_tolerance(schedule, z):
        msg = (
            f"a replayed schedule can only be resumed at one of its edges, and "
            f"z = {z:.6g} m is not one; the nearest is {float(edges[index]):.6g} m"
        )
        raise BackendError(msg)
    return z, index


def step_cap(monitor: object, quiet: bool, proposal: float, z: float) -> float:
    """Return a proposal held to the boundary monitor's limits on a step.

    The session's rule (:meth:`HostSession.cap`), which a decision-trace harness drives
    the real march with.

    Parameters
    ----------
    monitor : solprop.monitor.BoundaryMonitor or None
        The attached monitor, or ``None`` before one watches (no cap); a harness's
        stand-in needs only its ``interval``, ``step_cap`` and ``transverse``.
    quiet : bool
        Whether the stages leave a quiet in-stage temporal collar out.
    proposal : float
        The controller's proposal, m.
    z : float
        Where the step starts, m.

    Returns
    -------
    float
        The proposal, or the quiet collar's interval, the transverse edge's read
        spacing or the split collar's crossing limit when shorter, m.
    """
    if monitor is None:
        return proposal
    monitor = cast("BoundaryMonitor", monitor)
    # A left-out collar absorbs nothing, and the monitor sees only committed states:
    # while the term is out no step is longer than the interval it checks a quiet
    # window at (`BoundaryMonitor.interval`), so light has to walk about twenty
    # windows over the run to cross the collar between two checks.
    if quiet:
        proposal = min(proposal, monitor.interval)
    # A low-source run commits long steps, and light could cross the transverse edge
    # unseen within one: its read spacing is a ceiling, the step not made to land on
    # a read (SEP 40 section 5.3).
    edge = monitor.transverse
    if edge is not None:
        proposal = min(proposal, edge.ceiling(z))
    # The split collar's crossing limit, read from the occupancy at every check: light
    # spreading into faster-walking bins shortens the steps until the next check.
    return min(proposal, monitor.step_cap)


def natural_stop(every: int, count: int, z: float, target: float) -> bool | None:
    """Return whether the `count`-th accepted step since the load is a step plane.

    With ``step_planes = k`` the end of every ``k``-th accepted step is a plane (SEP 14
    phase 3): the march returns there, short of its target. A step that lands on the
    target is saved as the target; one that crosses a densely observed target is saved
    after it. The session's rule (:meth:`HostSession.natural_stop`), which a
    decision-trace harness drives the real march with.

    Parameters
    ----------
    every : int
        The step planes' period ``k``; zero for none.
    count : int
        Accepted steps since the load, this one included.
    z : float
        Its end, m.
    target : float
        The position the march is stepping to, m.

    Returns
    -------
    bool or None
        ``None`` when the step is no ``k``-th; otherwise whether the march stops at its
        end -- ``False`` at the target, whose plane observes it.
    """
    if every == 0 or count % every:
        return None
    return not reached(z, target)


class _StepStatsBuffer:
    """The rows a periodic statistics subset produces between two events.

    Grown by doubling from a preallocated block and reset without releasing it, so
    that after the first plane a run allocates nothing per step: the capacity a plane
    needed is the capacity the next starts with. Growing by concatenation on every step
    would be quadratic. The shape of each statistic is learnt from the first row.
    """

    def __init__(self) -> None:
        self._capacity = 0
        self._n_rows = 0
        self._z = np.zeros(0)
        self._step = np.zeros(0, dtype=np.int64)
        self._values: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        """Forget the rows held, keeping the allocation."""
        self._n_rows = 0

    def append(self, z: float, step: int, values: Mapping[str, np.ndarray]) -> None:
        """Add one row taken at `z`, after the chunk's attempt `step`."""
        if self._capacity == 0:
            self._allocate(values)
        elif self._n_rows == self._capacity:
            self._grow()
        index = self._n_rows
        self._z[index] = z
        self._step[index] = step
        for name, value in values.items():
            row = self._values.get(name)
            if row is None or np.shape(value) != row.shape[1:]:
                expected = None if row is None else row.shape[1:]
                msg = (
                    f"the {name!r} statistic came back with shape "
                    f"{np.shape(value)} and this plane's rows are {expected}"
                )
                raise BackendError(msg)
            row[index] = value
        self._n_rows = index + 1

    def drain(self) -> StepStats:
        """Return the rows as a frozen record of held copies and forget them."""
        rows = self._n_rows
        drained = StepStats(
            z=hold(self._z[:rows].copy()),
            step=hold(self._step[:rows].copy()),
            values={
                name: hold(value[:rows].copy()) for name, value in self._values.items()
            },
        )
        self.reset()
        return drained

    def _allocate(self, values: Mapping[str, np.ndarray]) -> None:
        self._capacity = INITIAL_STEP_STATS_CAPACITY
        self._z = np.zeros(self._capacity)
        self._step = np.zeros(self._capacity, dtype=np.int64)
        self._values = {
            name: np.zeros((self._capacity, *np.shape(value)), dtype=np.float64)
            for name, value in values.items()
        }

    def _grow(self) -> None:
        self._capacity *= 2
        self._z = np.resize(self._z, self._capacity)
        self._step = np.resize(self._step, self._capacity)
        self._values = {
            name: np.concatenate([value, np.zeros_like(value)])
            for name, value in self._values.items()
        }


_Work = Generator[Warning, None, object]
"""A point's work: it yields each boundary warning as it arises, in order, and returns
what follows it."""


class HostSession:
    """The reference's :class:`~solprop.backends.session.RunSession`.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The reference backend (a :class:`HostBackend`), constructed for `problem`. A
        backend reused from an earlier run is loaded again from `start`; the session
        builds its own march (:meth:`HostBackend.method`), and closes the backend with
        itself when `owns` is set.
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    start : solprop.backends.session.Start
        Where the run starts. The session loads it -- the field, the noise, the call
        ledgers and any method history -- and takes its field.
    listener : callable or None, optional
        Called with each attempt as it is made, for a progress reporter. It observes
        only; it cannot change the run.
    started : callable or None, optional
        Called with the start, the end and the plane count when propagation begins.
    stop : solprop.log.StopFlag or None, optional
        The run's stop token, polled before every trial.
    owns : bool, optional
        Whether closing the session closes the backend. A caller that reuses its
        backend for later runs (a runner, a scan) keeps it open.

    Attributes
    ----------
    reads_transverse_edge : bool
        The backend's (SEP 40 section 5.3), which the service loop reads.

    Raises
    ------
    solprop.exceptions.BackendError
        If `backend` is not a reference backend, is closed, or `start` does not fit
        the problem: its field's shape, or its position off the operator table or off
        a replayed schedule's edges.
    """

    def __init__(  # noqa: PLR0915 - the session's whole state, set once
        self,
        backend: Backend,
        problem: PropagationProblem,
        request: RunRequest,
        start: Start,
        *,
        listener: AttemptListener | None = None,
        started: Callable[[float, float, int], None] | None = None,
        stop: StopFlag | None = None,
        owns: bool = True,
    ) -> None:
        """Refuse a backend that is not the reference's, and load the starting state."""
        if not isinstance(backend, HostBackend):
            msg = (
                f"the reference session serves a reference backend, not "
                f"{type(backend).__name__}"
            )
            raise BackendError(msg)
        if backend.closed:
            raise BackendError("this backend is closed; its buffers were released")
        self._backend = backend
        self._workspace = backend.workspace
        self._problem = problem
        self._request = request
        self._stepper = request.stepper
        self._name = backend.name
        self.reads_transverse_edge = bool(backend.reads_transverse_edge)
        self._ledger = SessionLedger()
        # The run's plane plan (SEP 52 section 5.1): its planes, the first beyond the
        # start, whether the start's own plane is owed, and the step planes.
        plan = plane_plan(request, start)
        self._planes = plan.planes
        self._step_planes = plan.step_planes
        self._owns = owns
        self._index = plan.next_plane
        self._launch_pending = plan.launch_pending
        # The statistics channel between planes: the periodic subset promoted to
        # `names`, so a row goes through the very same evaluation a save plane uses.
        spec = request.observation.stats
        self._row_spec: StatsSpec | None = (
            None
            if not spec.periodic
            else dataclasses.replace(spec, names=spec.periodic, periodic=(), period=1)
        )
        self._row_period = spec.period
        self._rows = _StepStatsBuffer()
        # The session takes over the request's fresh monitor and feeds it; its and its
        # edge's warnings come here, each reported as a barrier.
        self._monitor: BoundaryMonitor | None = request.boundary.monitor
        self._sink: list[Warning] = []
        self._monitor_resume = start.monitor
        self._attached = False
        self._collar_quiet = False
        self._split_removed = 0.0
        self._split_found: np.ndarray | None = None
        # A point's work suspended at a warning barrier, and what follows it.
        self._continuation: tuple[_Work, Callable[[object], EventBatch | None]] | None
        self._continuation = None
        self._barrier: Warning | None = None
        self._generation = 0
        self._attempts = 0
        self._rejected = 0
        self._stepped = False
        self._summary: StepSummary | None = None
        self._latch = SnapshotLatch()
        self._fields = ObservedField(problem, self._drain)
        self._counter = AttemptCounter(listener, started)
        self._stop = stop
        # The schedule's cursors (`session.cpp`): accepted steps since the load, the
        # natural count, a step plane owed past an interpolated target, the halt latch
        # and the yield check.
        self._accepted_steps = 0
        self._natural_count = 0
        self._natural_due = False
        self._halt_requested = False
        self._halt_check: YieldDue | None = None
        self._halted = False
        self._midplane = False
        self._stop_here = False
        self._stopped_short = False
        self._target = math.nan
        self._progress_started = False
        self._alarm_reported = False
        self._history: list[tuple[float, float, float, bool, float]] = []
        self._clock = time.perf_counter()
        # The dense observation (SEP 47 section 5.3): whether the state holds an
        # interpolated plane, and where.
        self._observed = False
        self._observed_z = math.nan
        self._method = backend.method(self)
        self._march = self._method.march
        self._load(start)
        if start.ledgers is not None:
            self._restore_ledgers(
                start.ledgers.get("logical", {}),  # pyright: ignore[reportArgumentType]
                start.ledgers.get("atomic", {}),  # pyright: ignore[reportArgumentType]
            )
        # After the field, which starts the method afresh, as a resume always did.
        self._march_refusal = (
            None if start.march is None else self._restore_march(start.march)
        )

    # -- protocol -------------------------------------------------------------------

    @property
    def finished(self) -> bool:  # numpydoc ignore=RT01
        """Whether the session yielded its terminal event."""
        return self._ledger.finished

    @property
    def closed(self) -> bool:  # numpydoc ignore=RT01
        """Whether the session was closed."""
        return self._ledger.state is SessionState.CLOSED

    @property
    def in_flight(self) -> tuple[float, float] | None:  # numpydoc ignore=RT01
        """Low-storage step a failure abandoned in flight, or ``None``.

        Its start and length, m, while the state is neither of its ends.
        """
        return self._march.in_flight

    def status(self) -> SessionStatus:
        """Return a read-only summary of the session.

        Returns
        -------
        solprop.backends.session.SessionStatus
            Position, plane, attempts and lifecycle state.
        """
        return SessionStatus(
            z=self._here(),
            plane_index=self._index,
            attempts=self._attempts,
            rejected=self._rejected,
            state=self._ledger.state,
            outcome=self._ledger.outcome,
        )

    def advance(
        self,
        *,
        work_limit: int = DEFAULT_WORK_LIMIT,
        deadline: float = DEFAULT_DEADLINE,
    ) -> EventBatch:
        """Run to the next barrier and report it.

        A batch ends at the next observation, snapshot, warning or terminal outcome,
        or at a bounded yield: once `work_limit` attempts are made or `deadline` has
        passed, the march returns at its next accepted endpoint, exactly as at a
        natural stop, and the batch reports only progress. Nothing numerical depends
        on where a yield falls.

        Parameters
        ----------
        work_limit : int, optional
            Attempts before a yield.
        deadline : float, optional
            Host seconds before a yield, s.

        Returns
        -------
        solprop.backends.events.EventBatch
            The events, ending at a barrier.
        """
        self._ledger.check_advance()
        # `session.cpp`'s order: the owed launch plane, a warning's continuation, the
        # monitor's attachment, a latched snapshot at an endpoint, completion, then
        # the march to the next plane.
        if self._launch_pending:
            self._launch_pending = False
            return self._observation(ObservationKind.LAUNCH, -1, EMPTY_SUMMARY)
        started = time.perf_counter()
        limit = self._counter.count + max(int(work_limit), 1)
        self._halt_check = YieldDue(self._counter, limit, started, deadline)
        try:
            return self._advance()
        except StopRequested as error:
            return self._terminal(Outcome.STOPPED, error)
        except Exception as error:
            return self._terminal(Outcome.FAILED, error)
        finally:
            self._halt_check = None

    def acknowledge(self, receipt: Receipt) -> None:
        """Release the pending batch and allow advancing across its barrier.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The pending batch's receipt.
        """
        self._ledger.acknowledge(receipt)
        # One observation lives at a time: a consumer that kept it holds its own.
        self._fields.release()

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

        Returns
        -------
        numpy.ndarray
            The converted field.
        """
        self._ledger.check_ticket(ticket)
        return self._fields.export(ticket, recipe, out)

    def noise_phase(self, ticket: ExportTicket) -> np.ndarray | None:
        """Return the noise phase at a pending observation.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The pending observation's ticket.

        Returns
        -------
        numpy.ndarray or None
            The phase, shape ``(n_modes, n_pol, n_omega)``, or ``None`` without noise.
        """
        self._ledger.check_ticket(ticket)
        return self._fields.noise_phase(ticket)

    def pending_ticket(self) -> ExportTicket | None:
        """Return a ticket for the committed state after a failure outside the session.

        A service failure, or a signal raised while the session held no uncommitted
        state, leaves the field where the last commit put it. When a batch is pending,
        the ticket is bound to it; otherwise a failed terminal batch is issued, so the
        state can be exported for a terminal checkpoint and nothing advances again.

        Returns
        -------
        solprop.backends.events.ExportTicket or None
            The ticket, or ``None`` when the session is closed or its state poisoned.
        """
        if self.closed or self.in_flight is not None:
            return None
        if self._ledger.state is not SessionState.PENDING:
            self._generation += 1
            self._ledger.issue(outcome=Outcome.FAILED)
        receipt = self._ledger.pending
        assert receipt is not None
        return ExportTicket(receipt, self._generation, self._here())

    def request_snapshot(self) -> SnapshotReceipt:
        """Latch a snapshot for the next eligible accepted endpoint.

        Returns
        -------
        solprop.backends.session.SnapshotReceipt
            The request's receipt; a request while one is outstanding joins it.
        """
        self._ledger.check_open()
        receipt = self._latch.request()
        if not receipt.coalesced:
            # The halt latch the march's endpoints read (`_halted_short`).
            self._halt_requested = True
        return receipt

    def continue_monitor(self, continuation: Mapping[str, object]) -> None:
        """Continue a resumed run's boundary-monitor record, before it is attached.

        The record is read from the run's output, which the service loop opens only
        once the session has loaded its start, so that the opened record states the
        loaded backend's execution facts; it reaches the session before its first
        advance past the launch plane. A continuation given in the
        :class:`~solprop.backends.session.Start` is kept.

        Parameters
        ----------
        continuation : collections.abc.Mapping
            As :attr:`solprop.backends.session.Start.monitor`.

        Raises
        ------
        solprop.backends.session.SessionError
            If the monitor is already watching.
        """
        if self._attached:
            msg = "a boundary monitor is continued before it watches the run"
            raise SessionError(msg)
        if self._monitor_resume is None:
            self._monitor_resume = continuation

    def undelivered(self, ticket: ExportTicket) -> tuple[BackendHistory, StepStats]:
        """Deliver the attempts and rows made since the previous event, once.

        For a terminal checkpoint written after a failure outside the session (a
        service failure, or a signal out of a propagation), whose attempts no event
        delivered. The caller joins them to the chunks it holds since the last saved
        plane (:func:`solprop.backends.events.join_attempts`).

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The pending ticket the checkpoint is written for.

        Returns
        -------
        tuple
            The attempts and the rows (without a batch axis); empty when every attempt
            was delivered.
        """
        self._ledger.check_ticket(ticket)
        return self._chunk()

    def march_snapshot(self, ticket: ExportTicket) -> MarchSnapshot | None:
        """Return the method history at a pending observation, when the march has one.

        The march's part of the snapshot, the committed state and the history's rows,
        at an accepted endpoint: ``None`` while a dense observation holds the state
        between endpoints, for a march that keeps no history, and before the first
        sample.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The pending observation's ticket.

        Returns
        -------
        solprop.output.records.MarchSnapshot or None
            The history at the ticket's position, or ``None``.

        Raises
        ------
        solprop.exceptions.BackendError
            If a low-storage step was abandoned in flight.
        """
        self._ledger.check_ticket(ticket)
        self._refuse_in_flight()
        march, rows = self._march, self._method.rows
        if self._observed or not isinstance(march, HostAdamsMarch) or rows is None:
            return None
        fields = march.snapshot_fields()
        if fields is None:
            return None
        snapshot = MarchSnapshot(
            state=np.array(self._workspace.committed(), dtype=np.complex128, copy=True),
            rows=rows.export(),
            **fields,
        )
        return None if snapshot.z != ticket.z else snapshot

    def march_snapshot_nbytes(self) -> int | None:
        """Return the size a method-history snapshot would have, copying nothing.

        Returns
        -------
        int or None
            The state and every row, bytes, or ``None`` where a snapshot would be
            ``None``.

        Raises
        ------
        solprop.exceptions.BackendError
            If a low-storage step was abandoned in flight.
        """
        self._refuse_in_flight()
        march, rows = self._march, self._method.rows
        if (
            self._observed
            or not isinstance(march, HostAdamsMarch)
            or rows is None
            or not march.has_history
        ):
            return None
        state = math.prod(self._problem.shape) * np.dtype(np.complex128).itemsize
        return state + rows.nbytes()

    def march_export_refusal(self) -> str | None:
        """Return ``None``: every march the reference admits carries its history.

        Returns
        -------
        None
            The Adams history exports its rows; the other marches keep none.
        """
        return None

    @property
    def march_refusal(self) -> str | None:  # numpydoc ignore=RT01
        """Why the start's method history could not be installed.

        The reason, or ``None`` when it was installed or the start carried none.
        """
        return self._march_refusal

    def execution_record(self) -> Mapping[str, object]:
        """Return the session's execution facts.

        Returns
        -------
        collections.abc.Mapping
            The backend's metadata, with this run's call or evaluation counts, and the
            boundary monitor's record once it watches.
        """
        record = self._backend.metadata()
        if self._attached and self._monitor is not None:
            record["boundary_monitor"] = self._monitor.summary()
        return record

    def monitor_record(self) -> Mapping[str, object] | None:
        """Return the boundary monitor's record.

        Returns
        -------
        collections.abc.Mapping or None
            Its summary, or ``None`` without a monitor.
        """
        return None if self._monitor is None else self._monitor.summary()

    def close(self) -> None:
        """Release the backend when the session owns it; idempotent."""
        if self._ledger.close():
            self._stop = None
            if self._owns:
                self._backend.close()

    def __enter__(self) -> HostSession:
        """Return the session.

        Returns
        -------
        HostSession
            This session.
        """
        return self

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
            The exception type, if one is propagating.
        error : BaseException or None
            The exception, if one is propagating.
        trace : types.TracebackType or None
            Its traceback.
        """
        del kind, error, trace
        self.close()

    # -- the march's host (`MarchHost`) -----------------------------------------------

    @property
    def collar_quiet(self) -> bool:  # numpydoc ignore=RT01
        """Whether the stages leave a quiet in-stage temporal collar out (SEP 21)."""
        return self._collar_quiet

    def poll_stop(self) -> None:
        """Raise when the run's stop flag is set; called before every trial."""
        poll(self._stop)

    def cap(self, proposal: float, z: float) -> float:
        """Return a proposal held to the quiet collar's, the edge's and split's limits.

        Parameters
        ----------
        proposal : float
            The controller's proposal, m.
        z : float
            Where the step starts, m.

        Returns
        -------
        float
            The capped proposal, m.
        """
        monitor = self._monitor if self._attached else None
        return step_cap(monitor, self._collar_quiet, proposal, z)

    def record(self, end: float, dz: float, norm: float, accepted: bool) -> None:
        """Append one attempt to the history since the last event.

        A diverged attempt's norm (overflowed or NaN) is recorded as infinity, which
        the controller never accepts. An accepted attempt's quadrature score is read
        then, and one above :data:`~solprop.translate.modal.QUADRATURE_ALARM` reported
        once a plane: a statistic and a log line, never a controller input.

        Parameters
        ----------
        end : float
            The attempt's end, m.
        dz : float
            Its length, m.
        norm : float
            The scaled error norm as executed.
        accepted : bool
            Whether it was accepted.
        """
        if not math.isfinite(norm):
            norm = math.inf
        self._history.append(
            (end, dz, norm, accepted, time.perf_counter() - self._clock)
        )
        self._counter.step(end, dz, accepted)
        if accepted:
            value, z = self._workspace.score_quadrature()
            if value > QUADRATURE_ALARM and not self._alarm_reported:
                self._alarm_reported = True
                warn_quadrature(value, z)

    def absorb(self, record: AcceptRecord) -> None:
        """Accumulate an accepted step's split collar's sums (SEP 52 section 5.2).

        The removal adds over every accepted step since the previous check, which the
        next reading drains; the found occupancies are the latest accepted step's. A
        step that took no split sums leaves both as they were.

        Parameters
        ----------
        record : solprop.backends.engines.AcceptRecord
            What the step's finish reported.
        """
        if record.found is not None:
            self._split_removed += record.removed
            self._split_found = record.found

    def stepped(self) -> None:
        """Account the step just committed: the monitor, the edge, the row and stop.

        A warning on the way suspends the rest of the point's work, which the next
        advance does once the warning's barrier is acknowledged.
        """
        work = self._endpoint(self._target)
        try:
            warning = next(work)
        except StopIteration as done:
            self._stop_here = bool(done.value)
            return
        self._barrier = warning
        self._continuation = (work, self._endpoint_done)
        self._stop_here = True

    def natural_stop(self, target: float) -> bool:
        """Return whether the march stops at the endpoint just accounted.

        Parameters
        ----------
        target : float
            The position the march is stepping to, m.

        Returns
        -------
        bool
            Whether a step plane, a halt or a warning barrier stops it there.
        """
        del target  # the session's own, read in `stepped`
        stop, self._stop_here = self._stop_here, False
        self._stopped_short = self._stopped_short or stop
        return stop

    # -- the schedule ---------------------------------------------------------------

    def _advance(self) -> EventBatch:  # noqa: PLR0911 - one return per barrier kind
        """Run to the next barrier; failures propagate to :meth:`advance`."""
        continued = self._continuation is not None
        if continued:
            batch = self._continue()
            if batch is not None:
                return batch
        if not continued or not self._midplane:
            if not self._attached:
                batch = self._serve(self._attach(), _nothing)
                if batch is not None:
                    return batch
            if self._latch.pending is not None and self._at_endpoint():
                return self._snapshot()
            if self._index >= self._planes.size:
                return self._complete()
        target = float(self._planes[self._index])
        # Checked from the observed position, before a held plane is restored.
        self._require_state()
        self._check_target(target)
        restored = self._observed
        if restored:
            # The march is ahead of the last interpolated plane: resume it there.
            dense = self._method.dense
            assert dense is not None
            dense.restore()
            self._observed = False
        z = self._march.state.z
        if self._natural_due and not reached(z, target):
            # The step that crossed the last plane ends before this one, and is
            # observed first (`_natural_stop_due`).
            self._natural_due = False
            self._account_plane()
            return self._at_plane(ObservationKind.STEP, self._index)
        if (
            restored
            and self._latch.pending is not None
            and self._at_endpoint()
            and not reached(z, target)
        ):
            # A snapshot latched while a plane was held is taken at the endpoint ahead
            # of it, every plane its step crossed observed (`halt_ahead`).
            return self._snapshot()
        return self._march_to(target)

    def _march_to(self, target: float) -> EventBatch:
        """March towards `target` and report what stopped it."""
        step = self._propagate(target)
        self._account(step)
        summary = merge_summaries(self._summary, step)
        self._summary = summary
        if self._barrier is not None:
            # A warning at an endpoint the march returned at: the rest of that point's
            # work, then the march, follows the barrier.
            return self._warning()
        return self._returned(target)

    def _returned(self, target: float) -> EventBatch:
        """Report a march that returned without a barrier, its plane checked."""
        if self._latch.pending is None and not self._halted:
            # A yield latched that the target came first.
            self._halt_requested = False
        if self._halted:
            # A halt short of the target is no plane (SEP 47): no plane check there.
            return self._classify(target)
        return self._serve(self._plane_checks(), lambda _: self._classify(target))  # pyright: ignore[reportReturnType]

    def _endpoint_done(self, stop: object) -> EventBatch | None:
        """Follow an endpoint's work a barrier suspended: stop there, or march on."""
        if not stop:
            return None
        target = self._target
        self._stopped_short = True
        # A step plane begins the next plane afresh; a halt goes on with this one.
        self._midplane = self._halted
        self._after_march(target)
        return self._returned(target)

    def _propagate(self, target: float) -> StepSummary:
        """Step the march towards `target`, landing on it exactly.

        The plane's per-step latches start afresh unless a halt or a barrier returned
        the last march short of it, and a plane the march stepped across is observed
        inside its last step.
        """
        if not self._progress_started:
            # Announced at the first march, the first moment the run costs anything.
            self._progress_started = True
            self._counter.started(
                self._here(), self._stepper.z_end, len(self._stepper.save_planes)
            )
        started = time.perf_counter()
        first = len(self._history)
        if not self._midplane:
            self._rows.reset()
            # Each plane reports its own quadrature alarm.
            self._workspace.reset_quadrature()
            self._alarm_reported = False
        self._midplane = False
        self._halted = False
        self._stopped_short = False
        self._target = target
        self._march.advance(target)
        if self._halted or self._barrier is not None:
            self._midplane = True
        if self._barrier is None:
            self._after_march(target)
        taken = self._history[first:]
        record = ControlRecord.from_norms(
            self._stepper.controller,
            [entry[2] for entry in taken if entry[3]],
            n_rejected=sum(1 for entry in taken if not entry[3]),
        )
        if not record.met_tolerance:
            logger.warning(
                "%s: a step was too coarse for the tolerance it is scaled by on the "
                "way to %.6e m; the largest scaled error norm was %.3e against the "
                "requested 1.0 (rtol %.1e)",
                self._name,
                target,
                record.max_norm,
                record.requested_rtol,
            )
        logger.debug(
            "%s: reached %.6e m in %d steps (%d rejected), achieved/requested "
            "tolerance ratio %s",
            self._name,
            self._march.state.z,
            record.n_accepted + record.n_rejected,
            record.n_rejected,
            # The run's integrator decides: the low-storage march estimates no error.
            NOT_ESTIMATED
            if isinstance(self._stepper.integrator, LowStorageRungeKutta)
            else f"{record.tolerance_ratio:.3e}",
        )
        return StepSummary(
            z=self._here(),
            n_steps=len(taken),
            n_rejected=record.n_rejected,
            max_norm=record.max_norm,
            mean_norm=record.mean_norm,
            wall_time=time.perf_counter() - started,
            met_tolerance=record.met_tolerance,
        )

    def _after_march(self, target: float) -> None:
        """Observe a plane a march stepped across; settle a step plane owed there."""
        z = self._march.state.z
        dense = self._method.dense
        if not self._stopped_short and dense is not None and dense.crosses(z, target):
            # `_observe_inside` and the Adams march's observation: the state becomes
            # the crossing step's interpolated field at the plane, the committed state
            # held aside, and the plane is observed there (SEP 47 section 5.3).
            dense.observe(target)
            self._observed = True
            self._observed_z = target
            if isinstance(self._march, HostAdamsMarch):
                self._march.ledger.dense_observations += 1
        if self._natural_due and reached(self._here(), target) and not self._observed:
            # The step owed as a step plane ends on the target itself, whose plane
            # observes it: it is not observed a second time there.
            self._natural_due = False

    def _classify(self, target: float) -> EventBatch:
        """Report what the march stopped at: a plane, a halt or a step plane."""
        summary = self._summary
        assert summary is not None
        if bool(np.isclose(self._here(), target, rtol=1e-12, atol=0.0)):
            plane = self._index
            self._index += 1
            self._summary = None
            return self._observation(ObservationKind.PLANE, plane, summary)
        if self._halted:
            # A halt short of the target is never a step plane: a step plane is counted
            # first and stops the march unhalted, so a coincident plane is observed
            # and the latch, still set, is serviced at the next advance (SEP 47 section
            # 5.3: a coincident plane's transaction precedes the snapshot).
            if self._latch.pending is not None:
                return self._snapshot()
            # A bounded yield: no barrier, only progress; the plane goes on later.
            self._halt_requested = False
            history, rows = self._chunk()
            return self._issue([Progress(self._here(), history, rows)])
        self._summary = None
        return self._observation(
            ObservationKind.STEP if self._step_planes else ObservationKind.NATURAL,
            self._index,
            summary,
        )

    def _at_plane(self, kind: ObservationKind, plane: int) -> EventBatch:
        """Check the boundaries at a plane short of the next, then observe it."""
        summary = merge_summaries(
            self._summary, dataclasses.replace(EMPTY_SUMMARY, z=self._here())
        )

        def _observe(_: object) -> EventBatch:
            self._summary = None
            return self._observation(kind, plane, summary)

        batch = self._serve(self._plane_checks(), _observe)
        assert batch is not None
        return batch

    def _account_plane(self) -> None:
        """Begin the observation of a step plane owed at an endpoint already reached."""
        self._rows.reset()
        self._workspace.reset_quadrature()
        self._alarm_reported = False
        self._generation += 1

    def _natural_stop(self, target: float) -> bool:
        """Count an accepted step; return whether to stop at its end to save it.

        With ``step_planes = k`` the end of every ``k``-th accepted step since the load
        is a plane (SEP 14 phase 3): the march returns there, short of its target. A
        step that lands on the target is saved as the target; one that crosses a
        densely observed target is saved after it, at the next advance.
        """
        self._natural_count += 1
        z = self._march.state.z
        stop = natural_stop(self._stepper.step_planes, self._natural_count, z, target)
        if stop is None:
            return self._halted_short(z, target)
        if not stop:
            # A step that crossed a densely observed target is saved after it.
            self._natural_due = not reached(target, z)
        return stop

    def _halted_short(self, z: float, target: float) -> bool:
        """Return whether a halt -- a snapshot or a yield -- stops the march at `z`."""
        if reached(z, target):
            return False
        check = self._halt_check
        if check is not None and not self._halt_requested and check():
            self._halt_requested = True
        if self._halt_requested:
            self._halted = True
        return self._halt_requested

    def _at_endpoint(self) -> bool:
        """Return whether the state stands at an accepted endpoint of this session."""
        # The starting position is no accepted endpoint of this session: an Adams
        # march holds no history there to snapshot.
        return self._stepped and not self._observed

    def _here(self) -> float:
        """Return the observed position: a held dense plane's, else the march's."""
        return self._observed_z if self._observed else self._march.state.z

    # -- the boundary monitor and the transverse edge --------------------------------

    def _serve(
        self, work: _Work, then: Callable[[object], EventBatch | None]
    ) -> EventBatch | None:
        """Run a point's work to its next warning, or to its end and what follows."""
        try:
            warning = next(work)
        except StopIteration as done:
            return then(done.value)
        self._barrier = warning
        self._continuation = (work, then)
        return self._warning()

    def _continue(self) -> EventBatch | None:
        """Resume the point's work a warning barrier suspended."""
        assert self._continuation is not None
        work, then = self._continuation
        self._continuation = None
        return self._serve(work, then)

    def _drained(self) -> Generator[Warning]:
        """Yield the warnings the monitor and the edge put in the sink, in order."""
        sink = self._sink
        while sink:
            yield sink.pop(0)

    def _attach(self) -> _Work:
        """Restore and attach the boundary monitor after the launch plane's commit.

        The launch is recorded and judged now; every committed step is then accounted
        after the state is finished and the position advanced (:meth:`_endpoint`).
        """
        self._attached = True
        monitor = self._monitor
        if monitor is None:
            return None
        resume = self._monitor_resume
        if resume is not None:
            monitor.resume(
                resume.get("record"),  # pyright: ignore[reportArgumentType]
                plane=bool(resume.get("plane", False)),
                unwatched=resume.get("unwatched", ()),  # pyright: ignore[reportArgumentType]
            )
        n_t = self._problem.grids.n_t
        n_modes, _, _ = self._problem.shape
        columns = monitor.columns
        given = None if columns is None else columns.shape
        if monitor.weights.shape != (4, n_t) or given not in {None, (n_modes,)}:
            msg = (
                f"{self._name}: the boundary monitor's weights {monitor.weights.shape} "
                f"or columns {given} do not match the grid's {n_t} samples and the "
                f"state's {n_modes} modes"
            )
            raise BackendError(msg)
        if monitor.split:
            # The split collar's crossing limit is read again at every check.
            monitor.crossing = self._problem.split_crossing
        monitor.sink = self._sink
        if monitor.transverse is not None:
            monitor.transverse.sink = self._sink
        self._bind_monitor(monitor)
        workspace = self._workspace
        monitor.launch(self._energies(), self._here(), floor=workspace.input_floor)
        self._set_collar_quiet(monitor.collar_quiet)
        yield from self._drained()
        # A resumed fixed schedule is held to the state it resumes from at once.
        self._refuse_scheduled_crossing(monitor)
        edge = monitor.transverse
        if edge is not None:
            floor = workspace.input_floor
            schedule = self._stepper.schedule
            self._require_state()
            edge.launch(
                workspace.edge_reading(
                    edge,
                    # A resumed launch reads no share, but continues the current.
                    full=not edge.resumed,
                    current=edge.every_step and edge.resumed,
                ),
                self._here(),
                noise=None if floor is None else np.asarray(floor),
                clean_columns=workspace.clean_columns(edge),
                steps=None if schedule is None else np.asarray(schedule.steps),
                reads=self.reads_transverse_edge,
                declared=_declared_discard(self._problem),
            )
            yield from self._drained()
        return None

    def _endpoint(self, target: float) -> _Work:
        """Account the step just committed (:meth:`_account_step`), then its stop."""
        yield from self._account_step()
        return self._natural_stop(target)

    def _account_step(self) -> _Work:
        """Account the step just committed, after the engine's accept and the move.

        The boundary monitor's check or skip, as its cadence asks, with the transverse
        edge's read where due, then the periodic statistics row when the step is the
        period-th. It is the only point at which the state and the position both
        describe the step just committed.
        """
        monitor = self._monitor if self._attached else None
        if monitor is not None:
            if monitor.due(self._march.state.z):
                yield from self._check(monitor)
            else:
                monitor.skip(self._march.state.z)
            yield from self._read_transverse(monitor, plane=False)
        if self._row_spec is not None:
            self._accepted_steps += 1
            if self._accepted_steps % self._row_period == 0:
                # The attempt this row belongs to is the one just recorded, counted in
                # the chunk; -1 when a warning barrier delivered it, as native's.
                z = self._march.state.z
                self._rows.append(
                    z,
                    len(self._history) - 1,
                    self._workspace.stats(self._row_spec, z),
                )
        return None

    def _plane_checks(self) -> _Work:
        """Close the boundary monitor's row for the plane just reached (SEP 21, D5).

        A plane's state is checked even when its last step was not.
        """
        monitor = self._monitor if self._attached else None
        if monitor is None:
            return None
        if monitor.stale:
            yield from self._check(monitor, plane=True)
        yield from self._read_transverse(monitor, plane=True)
        monitor.plane(self._here())
        return None

    def _finish(self) -> _Work:
        """Run the monitor's final check (`BoundaryMonitor.finish`)."""
        monitor = self._monitor
        if monitor is not None and self._attached:
            monitor.finish()
            yield from self._drained()
        return None

    def _check(self, monitor: BoundaryMonitor, *, plane: bool = False) -> _Work:
        """Book a reading of the committed state with the monitor (SEP 45 4.10).

        - ``plane``: true for a saved plane's check of a step the monitor skipped.
        """
        self._require_state()
        z = self._here()
        monitor.step(self._energies(), z, plane=plane)
        self._set_collar_quiet(monitor.collar_quiet)
        yield from self._drained()
        self._refuse_scheduled_crossing(monitor, z=z)

    def _read_transverse(self, monitor: BoundaryMonitor, *, plane: bool) -> _Work:
        """Read the transverse edge where its spacing or a saved plane asks (SEP 40).

        - ``plane``: whether a saved plane has just been reached, which is always read.
        """
        edge = monitor.transverse
        if edge is None:
            return None
        z = self._here()
        # Read before the next step would carry past the read (`TransverseEdge.due`).
        due = edge.due(z, self._next_step())
        # A plane the step that reached it already read has nothing to add.
        read = (plane or due) and edge.wants_read(z)
        # A collar's inward current accumulates at every committed step.
        accumulate = edge.every_step and not plane
        if not (read or accumulate):
            return None
        self._require_state()
        reading = self._workspace.edge_reading(edge, faces=read, current=accumulate)
        if accumulate:
            assert reading.current is not None
            edge.step(reading.current, z)
            yield from self._drained()
        if read:
            edge.read(reading, z)
            yield from self._drained()
        return None

    def _next_step(self) -> float:
        """Return the step the march takes next, m.

        A replay takes its schedule's and never updates the proposal, which stays
        ``dz_init``: read from that, an edge read comes early or late by the
        difference. Zero past the schedule's end.
        """
        schedule = self._stepper.schedule
        state: MarchState = self._march.state
        if schedule is None:
            return state.proposal
        if state.cursor >= schedule.n_steps:
            return 0.0
        return float(schedule.steps[state.cursor])

    def _bind_monitor(self, monitor: BoundaryMonitor | None) -> None:
        """Read a monitor's energies from now on, or stop; either empties the sums.

        The workspace takes the split's sums at each accepted step's finish while a
        monitor of that scheme is bound (SEP 52 section 5.2).
        """
        self._split_removed = 0.0
        self._split_found = None
        self._workspace.bind_monitor(monitor)

    def _energies(self) -> BoundaryEnergies:
        """Return the monitor's reading of the state, draining the split's removal.

        The workspace's reading of the committed state (its energy, its four weighted
        energies and its occupancy), with what the split collar took at the accepted
        steps since the previous reading and the energies of the power the last step's
        split found (SEP 21, ADR-057).
        """
        removed, self._split_removed = self._split_removed, 0.0
        reading = self._workspace.reading()
        return BoundaryEnergies(
            total=reading.total,
            regions=reading.regions,
            removed=removed,
            found=self._split_found,
            occupancy=reading.occupancy,
        )

    def _refuse_scheduled_crossing(
        self, monitor: BoundaryMonitor, *, z: float | None = None
    ) -> None:
        """Stop a fixed schedule whose steps before the next check pass the cap.

        Translation admitted the schedule on the launch's occupancy; light that has
        since spread into faster-walking bins lowers the cap, and a schedule cannot
        shorten its steps (:class:`solprop.boundaries.SplitCrossing`). The cap holds
        until the next check, due within
        :meth:`~solprop.monitor.BoundaryMonitor.steps_to_check` steps at most, so the
        longest of those is held to it. Read at the launch too, so that a resumed
        schedule is held to the state it resumes from.

        - ``z``: the step's end, m; the session's position by default.
        """
        schedule = self._stepper.schedule
        first = self._march.state.cursor
        position = self._here() if z is None else z
        if schedule is None or first >= schedule.n_steps:
            return
        ahead = schedule.steps[first : first + monitor.steps_to_check()]
        step = float(np.max(ahead))
        cap = monitor.step_cap
        if step > cap:
            msg = (
                f"at z = {position:.6e} m the split temporal collar's crossing limit "
                f"fell to {cap:.3e} m, below the fixed schedule's step {step:.3e} m "
                "before the next check: light has spread into bins that walk faster "
                "than the launch's, and a step that long could carry it through the "
                "collar and round the window undamped. Take shorter steps, step "
                'adaptively, or use absorber_scheme="nonlinear"'
            )
            raise BoundaryError(msg)

    def _set_collar_quiet(self, quiet: bool) -> None:
        """Say whether the collar is quiet, as the monitor last found it.

        The term is left out only where the route can rest it
        (:func:`solprop.backends.records.collar_rests`, SEP 21 D15), and the monitor is
        told what the stages do, so that it adds no removal while the term is out.
        """
        quiet = quiet and collar_rests(self._problem, self._stepper)
        if self._monitor is not None and self._attached:
            self._monitor.note_collar(quiet)
        if quiet != self._collar_quiet:
            self._collar_quiet = quiet
            self._workspace.collar_quiet = quiet

    # -- loading ----------------------------------------------------------------------

    def _load(self, start: Start) -> None:
        """Load the start's state and noise, and start the march afresh at it.

        The state is masked to the active band once (the operator and the prefactor
        are exactly zero outside it). The ``input`` noise model adds its realisation at
        the run's launch and never again; a resume, which loads a plane the run saved,
        never takes it (:meth:`~solprop.backends.numpy_ref.workspace.NumpyWorkspace.
        load`). The controller starts afresh from ``dz_init``
        (``docs/dev/stepper.md`` section 3), and a replay from the edge the state sits
        on. Everything a march reads starts as a fresh backend's does (SEP 22 4.7):
        the session's march is its own, the monitor unbound, the collar back in the
        stages and the quadrature alarm forgotten.
        """
        values = np.ascontiguousarray(start.take_field(), dtype=np.complex128)
        if values.shape != self._problem.shape:
            msg = (
                f"the initial state has shape {values.shape}, expected "
                f"{self._problem.shape} from the contract"
            )
            raise BackendError(msg)
        position, cursor = start_position(self._problem, self._stepper, float(start.z))
        workspace = self._workspace
        workspace.load(values, start.noise, z=position, resume=start.resume)
        self._march.load(position, self._stepper.dz_init, cursor)
        self._bind_monitor(None)
        workspace.collar_quiet = False

    def _restore_ledgers(
        self, logical: Mapping[str, object], atomic: Mapping[str, object]
    ) -> None:
        """Continue a resumed run's cumulative call counts, its history afresh.

        A Runge-Kutta or low-storage march keeps no logical ledger, so nothing is
        installed for it; atomic counts go to the workspace's bound source, which
        refuses them without one.
        """
        if logical and isinstance(self._march, HostAdamsMarch):
            self._march.restore_ledger(logical)
        if atomic:
            self._workspace.restore_atomic_ledger(atomic)

    def _restore_march(self, snapshot: MarchSnapshot) -> str | None:
        """Continue the Adams march from a snapshot at the loaded position.

        The state is replaced by the snapshot's own, exactly as the interrupted run held
        it, and the rows installed. Nothing is changed when the snapshot cannot be used.

        - Returns: ``None`` when the history was continued, otherwise why the march
          starts afresh, as a resume did before snapshots were kept.
        """
        march, rows = self._march, self._method.rows
        if not isinstance(march, HostAdamsMarch) or rows is None:
            kind = (
                "low-storage"
                if isinstance(march, HostLowStorageMarch)
                else "Runge-Kutta"
            )
            return f"the run steps with the {kind} march, which keeps no history"
        reason = march.refusal(snapshot)
        if reason is not None:
            return reason
        try:
            rows.install(snapshot.rows, self._problem.shape)
        except BackendError as error:
            return str(error)
        self._workspace.install(np.array(snapshot.state, copy=True))
        march.continue_from(snapshot)
        return None

    # -- guards -----------------------------------------------------------------------

    def _require_state(self) -> None:
        """Refuse a closed or unloaded backend, or a state left in flight."""
        if self._backend.closed:
            raise BackendError("this backend is closed; its buffers were released")
        self._workspace.committed()
        self._refuse_in_flight()

    def _refuse_in_flight(self) -> None:
        """Refuse every read or advance of a state left between two steps (SEP 24)."""
        pending = self.in_flight
        if pending is not None:
            raise BackendError(in_flight_refusal(self._name, *pending))

    def _check_target(self, target: float) -> None:
        """Refuse a plane behind the observed position: a start's wrong plane index.

        The planes themselves are the validated request's: finite, increasing and
        inside the operator table (`validate_run`).
        """
        z = self._here()
        if target < z:
            msg = (
                f"cannot propagate backwards: asked for z = {target:.6g} m from "
                f"{z:.6g} m; the start's plane index names a plane behind it"
            )
            raise BackendError(msg)

    # -- events -----------------------------------------------------------------------

    def _account(self, summary: StepSummary) -> None:
        """Advance the committed position's generation after a march."""
        self._stepped = self._stepped or summary.n_steps > summary.n_rejected
        self._generation += 1

    def _observe(self, *, field: bool) -> Observation:
        """Return the state at the observed position and the attempts since the last.

        The field is the workspace's state itself, frozen and held so that an
        observation takes a view of it without a copy: the next step replaces it.
        With ``field=False`` nothing crosses (SEP 22 section 5.1) and nothing of the
        state is read, so a failure's terminal event drains its attempts even when the
        failure was the backend's closing; a state left in flight is refused either way.
        """
        if field:
            self._require_state()
        else:
            self._refuse_in_flight()
        history = _history_record(self._history)
        whole = self._workspace.state if field else None
        if whole is not None:
            hold(whole)
        z = self._here()
        plane = Observation(
            z=z,
            field=whole,
            noise_phase=self._workspace.noise_phase(z) if field else None,
            step_history=history,
            n_steps=history.n_steps,
            n_rejected=history.n_rejected,
            step_stats=self._rows.drain(),
        )
        self._history = []
        self._clock = time.perf_counter()
        return plane

    def _drain(self, *, field: bool) -> Observation:
        """Observe the state, draining the attempts and rows; count them once."""
        return self._counted(self._observe(field=field))

    def _counted(self, observed: Observation) -> Observation:
        """Count a drained observation's attempts into the status."""
        self._attempts += observed.n_steps
        self._rejected += observed.n_rejected
        return observed

    def _drain_attempts(self) -> Observation:
        """Drain the attempts and rows of a state left in flight, with no field."""
        entries, self._history = self._history, []
        self._clock = time.perf_counter()
        history = _history_record(entries)
        return Observation(
            z=self._march.state.z,
            field=None,
            noise_phase=None,
            step_history=history,
            n_steps=history.n_steps,
            n_rejected=history.n_rejected,
            step_stats=self._rows.drain(),
        )

    def _chunk(self) -> tuple[BackendHistory, StepStats]:
        """Drain the attempts and rows since the previous event, for the next one.

        - Returns: the attempts and the rows (without a batch axis), delivered by
          exactly one event (:mod:`solprop.backends.events`).
        """
        drained = (
            self._counted(self._drain_attempts())
            if self.in_flight is not None
            else self._drain(field=False)
        )
        chunks = self._fields.undelivered()
        chunks.append((drained.step_history, drained.step_stats))
        return join_attempts(chunks)

    def _issue(
        self, events: list[Event], *, outcome: Outcome | None = None
    ) -> EventBatch:
        """Issue a batch of events under a new receipt.

        - ``outcome``: the terminal outcome, when the batch ends the session.
        """
        receipt = self._ledger.issue(outcome=outcome)
        return EventBatch(receipt, tuple(bind(event, receipt) for event in events))

    def _ticket(self) -> ExportTicket:
        """Return an unbound ticket for the current committed state."""
        return ExportTicket(
            Receipt(self._ledger.session, 0), self._generation, self._here()
        )

    def _observation(
        self, kind: ObservationKind, plane: int, summary: StepSummary
    ) -> EventBatch:
        """Observe the state field-free, evaluate the plane's statistics, report it."""
        history, rows = self._chunk()
        self._fields.release()
        try:
            stats = dict(
                self._workspace.stats(self._request.observation.stats, self._here())
            )
        except Exception as error:
            return self._terminal(Outcome.FAILED, error, (history, rows))
        event = ObservationReady(
            z=self._here(),
            kind=kind,
            plane_index=plane,
            history=history,
            step_stats=rows,
            stats=stats,
            summary=summary,
            ticket=self._ticket(),
        )
        return self._issue([event])

    def _warning(self) -> EventBatch:
        """Report a boundary warning as a barrier at the committed endpoint.

        Its field can be exported for a checkpoint while the caller emits the warning,
        which its filters may raise; it delivers the attempts and rows since the
        previous event (SEP 47 section 5.5).
        """
        warning, self._barrier = self._barrier, None
        assert warning is not None
        history, rows = self._chunk()
        return self._issue(
            [
                ServiceRequired(
                    kind=ServiceKind.WARNING,
                    z=self._here(),
                    message=str(warning),
                    category=type(warning),
                    ticket=self._ticket(),
                    history=history,
                    step_stats=rows,
                    warning=warning,
                )
            ]
        )

    def _snapshot(self) -> EventBatch:
        """Report a snapshot service request at the current accepted endpoint.

        It delivers the attempts and rows since the previous event; the consumer joins
        them to those it holds since the previous observation for the running
        checkpoint, and keeps them for the next plane (SEP 47 section 5.5).
        """
        history, rows = self._chunk()
        self._halt_requested = False
        self._latch.clear()
        return self._issue(
            [
                ServiceRequired(
                    kind=ServiceKind.SNAPSHOT,
                    z=self._here(),
                    message="snapshot at an accepted endpoint",
                    ticket=self._ticket(),
                    history=history,
                    step_stats=rows,
                )
            ]
        )

    def _complete(self) -> EventBatch:
        """Run the final checks and report completion."""

        def _completed(_: object) -> EventBatch:
            return self._issue(
                [
                    Terminal(
                        Outcome.COMPLETED,
                        self._here(),
                        Eligibility.ELIGIBLE,
                        None,
                        None,
                        *self._chunk(),
                    )
                ],
                outcome=Outcome.COMPLETED,
            )

        batch = self._serve(self._finish(), _completed)
        assert batch is not None
        return batch

    def _terminal(
        self,
        outcome: Outcome,
        error: BaseException,
        chunk: tuple[BackendHistory, StepStats] | None = None,
    ) -> EventBatch:
        """Report a stop or failure at the last committed position.

        - ``chunk``: the attempts and rows since the previous event, when already
          drained for an event the failure replaced; drained here otherwise.
        - Returns: the terminal event, with a ticket when the state is valid.
        """
        self._continuation = None
        self._barrier = None
        poisoned = self.in_flight is not None
        history, rows = self._chunk() if chunk is None else chunk
        self._generation += 1
        event = Terminal(
            outcome,
            self._here(),
            Eligibility.POISONED if poisoned else Eligibility.ELIGIBLE,
            error,
            None if poisoned else self._ticket(),
            history,
            rows,
        )
        return self._issue([event], outcome=outcome)


def _nothing(_: object) -> None:
    """Follow a point's work with nothing: the advance goes on."""


class HostPrepared:
    """A prepared run of a reference backend (SEP 47 section 5.2).

    The backend was built for the problem and a request -- it lowered the method, the
    error control, the itinerary, the planes and the statistics then -- so preparing
    binds the problem and the request one run serves, a request that differs from the
    backend's own only in its start, retention, monitor and history
    (:meth:`~solprop.translate.run_request.RunRequest.for_run`). Each :meth:`start`
    opens one :class:`HostSession` over the backend.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The constructed reference backend; it stays its caller's unless a session owns
        it.
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's request.

    Raises
    ------
    solprop.backends.session.SessionError
        If the request is not one run of the backend's own
        (:func:`~solprop.backends.session_core.foreign_request`).
    solprop.exceptions.BackendError
        If the request asks for statistics the problem does not carry
        (:func:`~solprop.backends.session.statistics_unavailable`).
    """

    def __init__(
        self, backend: Backend, problem: PropagationProblem, request: RunRequest
    ) -> None:
        """Bind the problem and the request to the backend."""
        check_preparation(backend.name, backend.request, problem, request)
        self._backend = backend
        self._problem = problem
        self._request = request
        self._closed = False

    @property
    def backend(self) -> str:  # numpydoc ignore=RT01
        """Registered name of the backend that prepared it."""
        return self._backend.name

    def start(
        self,
        start: Start,
        *,
        listener: AttemptListener | None = None,
        started: Callable[[float, float, int], None] | None = None,
        stop: StopFlag | None = None,
        owns: bool = True,
    ) -> RunSession:
        """Start a session from an initial state or a checkpoint.

        Parameters
        ----------
        start : solprop.backends.session.Start
            Where the run starts; the session takes its field.
        listener : callable or None, optional
            Called with each attempt as it is made, for a progress reporter.
        started : callable or None, optional
            Called with the start, the end and the plane count when propagation
            begins.
        stop : solprop.log.StopFlag or None, optional
            The run's stop token.
        owns : bool, optional
            Whether closing the session closes the backend.

        Returns
        -------
        solprop.backends.session.RunSession
            The started :class:`HostSession`.

        Raises
        ------
        solprop.backends.session.SessionError
            If this preparation is closed.
        """
        if self._closed:
            msg = "the prepared execution is closed"
            raise SessionError(msg)
        return HostSession(
            self._backend,
            self._problem,
            self._request,
            start,
            listener=listener,
            started=started,
            stop=stop,
            owns=owns,
        )

    def metadata(self) -> Mapping[str, object]:
        """Return the execution metadata recorded with the run.

        Returns
        -------
        collections.abc.Mapping
            The backend's own (:meth:`solprop.backends.protocol.Backend.metadata`).
        """
        return self._backend.metadata()

    def memory_ledger(self) -> Mapping[str, int]:
        """Return every retained buffer by name, bytes.

        Returns
        -------
        collections.abc.Mapping
            The backend's ledger.
        """
        return self._backend.memory_ledger()

    def close(self) -> None:
        """Refuse further starts and close the backend it was prepared on; idempotent.

        A caller that keeps the backend for later runs leaves the preparation open: a
        session it started with ``owns=False`` never closes the backend.
        """
        if self._closed:
            return
        self._closed = True
        self._backend.close()


def _declared_discard(problem: PropagationProblem) -> bool:
    """Return whether a run declared content above ``discard_above`` (SEP 40 5.3)."""
    design = transverse_of(problem.route)
    return design is not None and design.specification.discard_above is not None


def _history_record(
    entries: Sequence[tuple[float, float, float, bool, float]],
) -> BackendHistory:
    """Return the attempts since the last observation as a held history record.

    - ``entries``: one ``(z, dz, error_norm, accepted, wall_time)`` per attempt.
    """
    columns = list(zip(*entries, strict=True)) if entries else [()] * 5
    return BackendHistory(
        z=hold(np.array(columns[0], dtype=np.float64)),
        dz=hold(np.array(columns[1], dtype=np.float64)),
        error_norm=hold(np.array(columns[2], dtype=np.float64)),
        accepted=hold(np.array(columns[3], dtype=np.bool_)),
        wall_time=hold(np.array(columns[4], dtype=np.float64)),
    )
