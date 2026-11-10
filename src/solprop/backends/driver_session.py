# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A transitional execution session over the inherited driver interface (SEP 47).

Until the native session (SEP 47 phase 3) owns its run, the native backend still
subclasses :class:`solprop.backends.driver.StepDriver`. :class:`DriverSession` presents
any such driver through the session protocol of :mod:`solprop.backends.session`, so the
Python service loop (:mod:`solprop.driver.service`) has one contract to serve. It
absorbs the plane schedule the old run loop kept -- the launch plane, each save plane,
natural stops and step planes -- the launch-before-monitor order, the snapshot latch and
the terminal outcome with its checkpoint eligibility.

:class:`DriverGroupSession` does the same for a group (SEP 47 section 6.7): it
presents the lockstep runners a batched scan steps compatible members with through
the group protocol, :class:`~solprop.backends.session.GroupSession`.

Both are temporary internal adapters, named in SEP 47's Part II with their removal
milestone; nothing outside the session machinery constructs them directly.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, cast

import numpy as np

from solprop.backends.events import (
    Eligibility,
    Event,
    EventBatch,
    ExportTicket,
    GroupBatch,
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
from solprop.backends.export import ExportRecipe, apply_recipe
from solprop.backends.protocol import (
    BackendHistory,
    Observation,
    StepStats,
    StepSummary,
)
from solprop.backends.session import (
    DEFAULT_DEADLINE,
    DEFAULT_WORK_LIMIT,
    SessionError,
    SessionLedger,
    SessionState,
    SessionStatus,
    SnapshotReceipt,
    Start,
    check_group,
)
from solprop.exceptions import StopRequested
from solprop.log import NullReporter, PlaneSummary, StopFlag
from solprop.monitor import BoundaryMonitor
from solprop.output.schema import MarchSnapshot
from solprop.translate.linop import power_weight_at
from solprop.translate.problem import KernelProblem
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    # Under TYPE_CHECKING, so that the runtime type checks of the test suite do not
    # hold the scripted drivers to the full protocol or check a refused export
    # destination's shape before the session refuses it.
    from jaxtyping import Complex

    from solprop.backends.driver import StepDriver
    from solprop.backends.protocol import Backend

__all__ = ["AttemptListener", "DriverGroupSession", "DriverSession"]

AttemptListener = Callable[[float, float, bool], None]
"""Called with each attempt's end, length and acceptance, for a progress reporter."""


class _Attempts(NullReporter):
    """Forwards a driver's per-attempt progress to a listener, and nothing else.

    The service loop reports planes itself; a driver's own ``plane`` calls are
    dropped here.

    Parameters
    ----------
    listener : callable or None
        Called with each attempt; ``None`` drops them.
    started : callable or None, optional
        Called once, when the driver first propagates; ``None`` drops it.
    """

    def __init__(
        self,
        listener: AttemptListener | None,
        started: Callable[[float, float, int], None] | None = None,
    ) -> None:
        """Hold the listeners."""
        self._listener = listener
        self._started = started

    def started(self, z_start: float, z_end: float, n_planes: int) -> None:
        """Forward the driver's start, made when it first propagates.

        Parameters
        ----------
        z_start : float
            Where propagation starts, m.
        z_end : float
            Where it ends, m.
        n_planes : int
            The number of save planes.
        """
        if self._started is not None:
            self._started(z_start, z_end, n_planes)

    def step(self, z: float, dz: float, accepted: bool) -> None:
        """Forward one attempt.

        Parameters
        ----------
        z : float
            Where the attempt ended, m.
        dz : float
            Its length, m.
        accepted : bool
            Whether it was accepted.
        """
        if self._listener is not None:
            self._listener(z, dz, accepted)

    def plane(self, z: float, summary: PlaneSummary) -> None:
        """Drop a driver's plane report: the service loop reports planes.

        Parameters
        ----------
        z : float
            The plane, m.
        summary : solprop.log.PlaneSummary
            Its summary.
        """
        del z, summary


def _hook(driver: object, name: str, *arguments: object) -> None:
    """Call a driver's halt hook when it has one.

    The halt latch and the yield check belong to the inherited
    :class:`~solprop.backends.driver.StepDriver`; a driver written to the bare
    protocol (a test double) has neither, and then runs whole planes.

    Parameters
    ----------
    driver : object
        The driver.
    name : str
        The hook's name.
    *arguments : object
        Its arguments.
    """
    hook = getattr(driver, name, None)
    if hook is not None:
        hook(*arguments)


def _strip(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop the leading batch axis a driver's plane statistics carry.

    Parameters
    ----------
    values : collections.abc.Mapping
        Statistics with a leading batch axis of one.

    Returns
    -------
    dict
        The same statistics without it.
    """
    return {name: np.asarray(value[0]) for name, value in values.items()}


def _rows(rows: StepStats) -> StepStats:
    """Drop the batch axis a driver's periodic rows carry after their row axis.

    Parameters
    ----------
    rows : solprop.backends.protocol.StepStats
        Rows shaped ``(n_rows, 1, ...)``.

    Returns
    -------
    solprop.backends.protocol.StepStats
        The same rows shaped ``(n_rows, ...)``.
    """
    return StepStats(
        rows.z, rows.step, {name: value[:, 0] for name, value in rows.values.items()}
    )


def _recorded(driver: Backend, z: float) -> Observation:
    """Return the attempts a driver recorded since its last observation, field-free.

    For a driver whose state was left in flight, which refuses
    :meth:`~solprop.backends.protocol.Backend.observe`. The record is the step driver's
    own history and periodic rows (:class:`solprop.backends.driver.StepDriver`),
    drained as an observation drains them.

    Parameters
    ----------
    driver : solprop.backends.protocol.Backend
        The driver.
    z : float
        The last committed position, m.

    Returns
    -------
    solprop.backends.protocol.Observation
        The attempts and rows, with no field.
    """
    stepping = cast("StepDriver[object]", driver)
    entries, stepping._history = stepping._history, []  # pyright: ignore[reportPrivateUsage]
    history = BackendHistory(
        z=np.array([entry[0] for entry in entries], dtype=np.float64),
        dz=np.array([entry[1] for entry in entries], dtype=np.float64),
        error_norm=np.array([entry[2] for entry in entries], dtype=np.float64),
        accepted=np.array([entry[3] for entry in entries], dtype=np.bool_),
        wall_time=np.array([entry[4] for entry in entries], dtype=np.float64),
    )
    rows = stepping._drain_step_stats()  # pyright: ignore[reportPrivateUsage]
    return Observation(
        z=z,
        field=None,
        noise_phase=None,
        step_history=history,
        n_steps=history.n_steps,
        n_rejected=history.n_rejected,
        step_stats=rows,
    )


_EMPTY_SUMMARY = StepSummary(
    z=0.0,
    n_steps=0,
    n_rejected=0,
    max_norm=0.0,
    mean_norm=0.0,
    wall_time=0.0,
    met_tolerance=True,
)


class _YieldDue:
    """Says whether a bounded advance has reached its attempt bound or its deadline.

    A class rather than a closure in :meth:`DriverSession.advance`, so that no function
    object is made per call (each one costs the coverage lane's import hook a compile).

    Parameters
    ----------
    session : DriverSession
        The session whose attempts are counted.
    limit : int
        The attempt count at which the march halts.
    started : float
        The advance's start, in :func:`time.perf_counter` seconds.
    deadline : float
        The seconds after `started` at which the march halts.
    """

    def __init__(
        self, session: DriverSession, limit: int, started: float, deadline: float
    ) -> None:
        self._session = session
        self._limit = limit
        self._started = started
        self._deadline = deadline

    def __call__(self) -> bool:
        """Return whether the attempt bound or the deadline has been reached.

        Returns
        -------
        bool
            True once the march should halt at its next accepted endpoint.
        """
        return (
            self._session._listened >= self._limit
            or time.perf_counter() - self._started >= self._deadline
        )


class DriverSession:
    """A :class:`~solprop.backends.session.RunSession` over an inherited driver.

    Parameters
    ----------
    driver : solprop.backends.protocol.Backend
        The driver, constructed for `problem`. A driver reused from an earlier run is
        loaded again from `start`, which restores everything its march reads. The
        session closes it with itself when `owns` is set.
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    start : solprop.backends.session.Start
        Where the run starts. The session loads it into the driver -- the field, the
        noise, the call ledgers and any method history -- and takes its field.
    listener : callable or None, optional
        Called with each attempt as it is made, for a progress reporter. It observes
        only; it cannot change the run.
    started : callable or None, optional
        Called with the start, the end and the plane count when propagation begins.
    stop : solprop.log.StopFlag or None, optional
        The run's stop token, polled before every trial (or native call).
    owns : bool, optional
        Whether closing the session closes the driver. A caller that reuses its driver
        for later runs (a runner, a scan) keeps it open.
    """

    def __init__(  # noqa: PLR0913 - the session context
        self,
        driver: Backend,
        problem: KernelProblem,
        request: RunRequest,
        start: Start,
        *,
        listener: AttemptListener | None = None,
        started: Callable[[float, float, int], None] | None = None,
        stop: StopFlag | None = None,
        owns: bool = True,
    ) -> None:
        """Load the starting state into the driver."""
        self._driver = driver
        self._problem = problem
        self._request = request
        self._ledger = SessionLedger()
        self._planes = np.asarray(request.observation.save_planes, dtype=np.float64)
        self._owns = owns
        self._index = start.plane_index
        self._launch_pending = start.plane_pending
        # The session takes over the request's fresh monitor and feeds it.
        self._monitor: BoundaryMonitor | None = request.boundary.monitor
        self._monitor_resume = start.monitor
        self._attached = False
        self._generation = 0
        self._z = float(start.z)
        self._attempts = 0
        self._rejected = 0
        self._stepped = False
        self._summary: StepSummary | None = None
        # Attempts a field fetch drained that no event has delivered yet.
        self._undelivered: list[tuple[BackendHistory, StepStats]] = []
        self._snapshot_requests = 0
        self._snapshot_pending: int | None = None
        self._observed: Observation | None = None
        self._observed_generation = -1
        self._converted: ExportRecipe | None = None
        self._listened = 0
        # Whether the observation's own array was handed out unconverted (export).
        self._lent = False

        def counted(z: float, dz: float, accepted: bool) -> None:
            """Count an attempt and pass it on to the listener.

            Parameters
            ----------
            z : float
                The position the attempt started from, m.
            dz : float
                The attempted step, m.
            accepted : bool
                Whether the step was accepted.
            """
            self._listened += 1
            if listener is not None:
                listener(z, dz, accepted)

        driver.set_progress(_Attempts(counted, started))
        driver.set_stop(stop)
        driver.set_initial(
            np.asarray(start.take_field())[np.newaxis],
            start.noise,
            z=float(start.z),
            resume=start.resume,
        )
        if start.ledgers is not None:
            logical = start.ledgers.get("logical", {})
            atomic = start.ledgers.get("atomic", {})
            driver.restore_call_ledgers(logical, atomic)  # pyright: ignore[reportArgumentType]
        # After the field, which resets the method's history, as a resume always did.
        self._march_refusal = (
            None if start.march is None else driver.restore_march(start.march)
        )

    # -- protocol -------------------------------------------------------------------

    @property
    def driver(self) -> Backend:
        """Return the wrapped driver, for the transitional service paths.

        Returns
        -------
        solprop.backends.protocol.Backend
            The driver.
        """
        return self._driver

    @property
    def finished(self) -> bool:
        """Return whether the session yielded its terminal event.

        Returns
        -------
        bool
            True once finished.
        """
        return self._ledger.finished

    @property
    def closed(self) -> bool:
        """Return whether the session was closed.

        Returns
        -------
        bool
            True once closed.
        """
        return self._ledger.state is SessionState.CLOSED

    def status(self) -> SessionStatus:
        """Return a read-only summary of the session.

        Returns
        -------
        solprop.backends.session.SessionStatus
            Position, plane, attempts and lifecycle state.
        """
        return SessionStatus(
            z=self._z,
            plane_index=self._index,
            attempts=self._attempts,
            rejected=self._rejected,
            state=self._ledger.state,
            outcome=self._ledger.outcome,
        )

    def advance(  # noqa: PLR0911, PLR0912 - one return per barrier kind
        self,
        *,
        work_limit: int = DEFAULT_WORK_LIMIT,
        deadline: float = DEFAULT_DEADLINE,
    ) -> EventBatch:
        """Run to the next barrier and report it.

        A batch ends at the next observation, snapshot or terminal outcome, or at a
        bounded yield: once `work_limit` attempts are made or `deadline` has passed,
        the march returns at its next accepted endpoint (a native march after its next
        call and one step), exactly as at a natural stop, and the batch reports only
        progress. Nothing numerical depends on where a yield falls.

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
        if self._launch_pending:
            self._launch_pending = False
            return self._observation(ObservationKind.LAUNCH, -1, _EMPTY_SUMMARY)
        if not self._attached:
            self._attach_monitor()
        if self._snapshot_pending is not None and self._at_endpoint():
            return self._snapshot()
        if self._index >= self._planes.size:
            return self._complete()
        target = float(self._planes[self._index])
        if self._snapshot_pending is not None and self._holding_plane():
            # The march restores the endpoint ahead of the plane it holds, and halts
            # there if the target lies beyond: planes closer than the steps would
            # otherwise put off the snapshot with every step (SEP 47 section 5.5).
            _hook(self._driver, "halt_ahead")
        started = time.perf_counter()
        limit = self._listened + max(int(work_limit), 1)

        due = _YieldDue(self, limit, started, deadline)

        _hook(self._driver, "set_halt_check", due)
        try:
            step = self._driver.propagate_to(target)
        except StopRequested as error:
            return self._terminal(Outcome.STOPPED, error)
        except Exception as error:
            return self._terminal(Outcome.FAILED, error)
        finally:
            _hook(self._driver, "set_halt_check", None)
        self._account(step)
        summary = self._summary_so_far(step)
        if self._snapshot_pending is None and not self._driver_halted():
            # A yield latched between native calls that the target came first.
            _hook(self._driver, "clear_halt")
        reached_target = bool(np.isclose(step.z, target, rtol=1e-12, atol=0.0))
        if reached_target:
            plane = self._index
            self._index += 1
            self._summary = None
            return self._observation(ObservationKind.PLANE, plane, summary)
        if self._driver_halted():
            if self._snapshot_pending is not None:
                if not self._step_plane_here():
                    return self._snapshot()
            else:
                # A bounded yield: no barrier, only progress; the plane goes on later.
                _hook(self._driver, "clear_halt")
                history, rows = self._chunk()
                return self._issue([Progress(self._z, history, rows)])
        self._summary = None
        return self._observation(
            ObservationKind.STEP
            if self._request.observation.step_planes
            else ObservationKind.NATURAL,
            self._index,
            summary,
        )

    def acknowledge(self, receipt: Receipt) -> None:
        """Release the pending batch and allow advancing across its barrier.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The pending batch's receipt.
        """
        self._ledger.acknowledge(receipt)
        # One observation lives at a time: a consumer that kept it holds its own.
        self._observed = None
        self._converted = None
        self._lent = False

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
        spatial = self._problem.transform.geometry is not None
        if recipe.power and spatial:
            # A spatial route's field measure is already the stored quantity.
            recipe = dataclasses.replace(recipe, power=False)
        weight = power_weight_at(self._problem, ticket.z) if recipe.power else None
        observed = self._fielded(ticket)
        converted = self._converted
        if converted is not None and (converted != recipe or out is not None):
            # Converted in place for another recipe, and handed out so: observe the raw
            # field again rather than write to an array a consumer holds.
            self._observed = None
            self._converted = None
            observed = self._fielded(ticket)
        state = self._field(ticket)
        if self._converted is not None:
            # Converted in place for this recipe already: the same detached result.
            return state
        identity = not recipe.optics and (weight is None or bool(np.all(weight == 1.0)))
        if identity and out is None and (observed.field_owned or spatial):
            # An observation's arrays are read-only and nothing writes to them again
            # (Backend.observe), so an export with nothing to convert is that array.
            # A modal field the backend may still hold is detached by a copy, as the
            # true-power conversion always detached it. Once lent, the array is never
            # converted in place.
            self._lent = True
            return state
        if out is None and observed.field_owned and not self._lent:
            # The driver made this array for this session alone, which hands out only
            # the export: convert it where it is (SEP 47 section 5.4).
            self._converted = recipe
            # The flag is the observation's own array's; `state` is its member view.
            whole = observed.require_field()
            whole.setflags(write=True)
            try:
                member = whole[0]
                apply_recipe(member, recipe, weight=weight, out=member)
            finally:
                whole.setflags(write=False)
            member.setflags(write=False)
            return member
        return apply_recipe(state, recipe, weight=weight, out=out)

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
        observed = self._fielded(ticket)
        phase = observed.noise_phase
        return None if phase is None else np.array(phase[0])

    def pending_ticket(self) -> ExportTicket | None:
        """Return a ticket for the committed state after a failure outside the session.

        A service failure, or a signal raised while the driver held no uncommitted
        state, leaves the field where the last commit put it. When a batch is pending,
        the ticket is bound to it; otherwise a failed terminal batch is issued, so the
        state can be exported for a terminal checkpoint and nothing advances again.

        Returns
        -------
        solprop.backends.events.ExportTicket or None
            The ticket, or ``None`` when the session is closed or its state poisoned.
        """
        if self.closed or self._driver.in_flight is not None:
            return None
        if self._ledger.state is not SessionState.PENDING:
            self._generation += 1
            self._z = float(getattr(self._driver, "_z", self._z))
            self._ledger.issue(outcome=Outcome.FAILED)
        receipt = self._ledger.pending
        assert receipt is not None
        return ExportTicket(receipt, self._generation, self._z)

    def request_snapshot(self) -> SnapshotReceipt:
        """Latch a snapshot for the next eligible accepted endpoint.

        Returns
        -------
        solprop.backends.session.SnapshotReceipt
            The request's receipt; a request while one is outstanding joins it.
        """
        self._ledger.check_open()
        if self._snapshot_pending is not None:
            return SnapshotReceipt(self._snapshot_pending, coalesced=True)
        self._snapshot_requests += 1
        self._snapshot_pending = self._snapshot_requests
        _hook(self._driver, "request_halt")
        return SnapshotReceipt(self._snapshot_pending, coalesced=False)

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

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The pending observation's ticket.

        Returns
        -------
        solprop.output.schema.MarchSnapshot or None
            The history at the ticket's position, or ``None``.
        """
        self._ledger.check_ticket(ticket)
        snapshot = self._driver.march_snapshot()
        if snapshot is None or snapshot.z != ticket.z:
            return None
        return snapshot

    def march_snapshot_nbytes(self) -> int | None:
        """Return the size a method-history snapshot would have.

        Returns
        -------
        int or None
            Bytes, or ``None`` when the march keeps no history.
        """
        return self._driver.march_snapshot_nbytes(self._z)

    def march_export_refusal(self) -> str | None:
        """Return why the march cannot export its history, or ``None``.

        Returns
        -------
        str or None
            The reason, or ``None`` when it can.
        """
        return self._driver.march_export_refusal()

    @property
    def march_refusal(self) -> str | None:
        """Return why the start's method history could not be installed.

        Returns
        -------
        str or None
            The reason, or ``None`` when it was installed or the start carried none.
        """
        return self._march_refusal

    def execution_record(self) -> Mapping[str, object]:
        """Return the session's execution facts.

        Returns
        -------
        collections.abc.Mapping
            The driver's metadata.
        """
        return self._driver.metadata()

    def monitor_record(self) -> Mapping[str, object] | None:
        """Return the boundary monitor's record.

        Returns
        -------
        collections.abc.Mapping or None
            Its summary, or ``None`` without a monitor.
        """
        return None if self._monitor is None else self._monitor.summary()

    @property
    def monitor(self) -> BoundaryMonitor | None:
        """Return the session's own boundary monitor, for transitional reporting.

        Returns
        -------
        solprop.monitor.BoundaryMonitor or None
            The monitor, or ``None``.
        """
        return self._monitor

    def close(self) -> None:
        """Release the driver; idempotent."""
        if self._ledger.close():
            self._driver.set_stop(None)
            if self._owns:
                self._driver.close()

    def __enter__(self) -> "DriverSession":  # noqa: UP037 - runtime type hook
        """Return the session.

        Returns
        -------
        DriverSession
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

    # -- internals ------------------------------------------------------------------

    def _attach_monitor(self) -> None:
        """Restore and attach the boundary monitor after the launch plane's commit."""
        self._attached = True
        if self._monitor is None:
            return
        resume = self._monitor_resume
        if resume is not None:
            self._monitor.resume(
                resume.get("record"),  # pyright: ignore[reportArgumentType]
                plane=bool(resume.get("plane", False)),
                unwatched=resume.get("unwatched", ()),  # pyright: ignore[reportArgumentType]
            )
        self._driver.attach_monitor(self._monitor)

    def _account(self, summary: StepSummary) -> None:
        """Advance the committed position and the state generation.

        Parameters
        ----------
        summary : solprop.backends.protocol.StepSummary
            What the propagation did.
        """
        self._stepped = self._stepped or summary.n_steps > summary.n_rejected
        self._z = float(summary.z)
        self._generation += 1

    def _drain(self, *, field: bool) -> Observation:
        """Observe the driver, draining its attempts and rows since the last drain.

        Every attempt is drained exactly once, so the status counts them here.

        Parameters
        ----------
        field : bool
            Whether the field crosses too.

        Returns
        -------
        solprop.backends.protocol.Observation
            The observation.
        """
        return self._counted(self._driver.observe(field=field))

    def _counted(self, observed: Observation) -> Observation:
        """Count a drained observation's attempts into the status.

        Parameters
        ----------
        observed : solprop.backends.protocol.Observation
            Attempts drained from the driver, once.

        Returns
        -------
        solprop.backends.protocol.Observation
            The same observation.
        """
        self._attempts += observed.n_steps
        self._rejected += observed.n_rejected
        return observed

    def _chunk(self) -> tuple[BackendHistory, StepStats]:
        """Drain the attempts and rows since the previous event, for the next one.

        A state left in flight cannot be observed; the attempts recorded before the
        abandoned step are drained from the driver's record instead (:func:`_recorded`).

        Returns
        -------
        tuple
            The attempts and the rows (without a batch axis), delivered by exactly one
            event (:mod:`solprop.backends.events`).
        """
        drained = (
            self._counted(_recorded(self._driver, self._z))
            if self._driver.in_flight is not None
            else self._drain(field=False)
        )
        chunks = [*self._undelivered, (drained.step_history, _rows(drained.step_stats))]
        self._undelivered = []
        return join_attempts(chunks)

    def _summary_so_far(self, step: StepSummary) -> StepSummary:
        """Merge a propagation's summary with the yields before it on the same plane.

        Parameters
        ----------
        step : solprop.backends.protocol.StepSummary
            The latest propagation.

        Returns
        -------
        solprop.backends.protocol.StepSummary
            The summary of the plane so far.
        """
        before = self._summary
        if before is None:
            merged = step
        else:
            accepted_before = before.n_steps - before.n_rejected
            accepted_now = step.n_steps - step.n_rejected
            accepted = accepted_before + accepted_now
            merged = StepSummary(
                z=step.z,
                n_steps=before.n_steps + step.n_steps,
                n_rejected=before.n_rejected + step.n_rejected,
                max_norm=max(before.max_norm, step.max_norm),
                mean_norm=0.0
                if accepted == 0
                else (
                    before.mean_norm * accepted_before + step.mean_norm * accepted_now
                )
                / accepted,
                wall_time=before.wall_time + step.wall_time,
                met_tolerance=before.met_tolerance and step.met_tolerance,
            )
        self._summary = merged
        return merged

    def _at_endpoint(self) -> bool:
        """Return whether the driver stands at an accepted endpoint.

        Returns
        -------
        bool
            False while a dense observation holds the state between endpoints.
        """
        # The starting position is no accepted endpoint of this session: an Adams
        # march holds no history there to snapshot.
        return self._stepped and not self._holding_plane()

    def _holding_plane(self) -> bool:
        """Return whether a dense observation holds the driver's state.

        Returns
        -------
        bool
            True while the driver's own (a Runge-Kutta march's) or its Adams march's
            observation holds an interpolated plane, the march ahead of it.
        """
        driver = self._driver
        adams = getattr(driver, "_adams", None)
        return (
            getattr(driver, "_dense_ahead", None) is not None
            or getattr(adams, "_ahead", None) is not None
        )

    def _driver_halted(self) -> bool:
        """Return whether the driver stopped at a requested halt.

        Returns
        -------
        bool
            True while the latch is set and the driver did not reach its target.
        """
        return bool(getattr(self._driver, "_halted", False))

    def _step_plane_here(self) -> bool:
        """Return whether the driver stopped at a step plane, latch or not.

        A halt requested for a snapshot and the ``step_planes``-th accepted step can
        end at the same endpoint. The step plane is observed first and the latch, still
        set, is serviced at the next advance (SEP 47 section 5.3: a coincident plane's
        transaction precedes the snapshot).

        Returns
        -------
        bool
            True when the endpoint ends a ``step_planes``-th accepted step.
        """
        every = self._request.observation.step_planes
        count = int(getattr(self._driver, "_natural_count", 0))
        return bool(every) and count % every == 0

    def _issue(
        self, events: list[Event], *, outcome: Outcome | None = None
    ) -> EventBatch:
        """Issue a batch of events under a new receipt.

        Parameters
        ----------
        events : list
            The events; tickets are bound to the receipt here.
        outcome : solprop.backends.events.Outcome or None, optional
            The terminal outcome, when the batch ends the session.

        Returns
        -------
        solprop.backends.events.EventBatch
            The batch.
        """
        receipt = self._ledger.issue(outcome=outcome)
        bound: list[Event] = []
        for event in events:
            ticket = getattr(event, "ticket", None)
            if ticket is not None:
                event = dataclasses.replace(  # noqa: PLW2901
                    event, ticket=dataclasses.replace(ticket, receipt=receipt)
                )
            bound.append(event)
        return EventBatch(receipt, tuple(bound))

    def _ticket(self) -> ExportTicket:
        """Return an unbound ticket for the current committed state.

        Returns
        -------
        solprop.backends.events.ExportTicket
            The ticket, its receipt filled in when the batch is issued.
        """
        return ExportTicket(Receipt(self._ledger.session, 0), self._generation, self._z)

    def _observation(
        self, kind: ObservationKind, plane: int, summary: StepSummary
    ) -> EventBatch:
        """Observe the driver field-free and report it.

        Parameters
        ----------
        kind : solprop.backends.events.ObservationKind
            Why it was observed.
        plane : int
            The save plane it answers.
        summary : solprop.backends.protocol.StepSummary
            The propagation that reached it.

        Returns
        -------
        solprop.backends.events.EventBatch
            One observation.
        """
        history, rows = self._chunk()
        self._observed = None
        try:
            stats = _strip(self._driver.stats(self._request.observation.stats))
        except Exception as error:
            return self._terminal(Outcome.FAILED, error, (history, rows))
        event = ObservationReady(
            z=self._z,
            kind=kind,
            plane_index=plane,
            history=history,
            step_stats=rows,
            stats=stats,
            summary=summary,
            ticket=self._ticket(),
        )
        return self._issue([event])

    def _snapshot(self) -> EventBatch:
        """Report a snapshot service request at the current accepted endpoint.

        It delivers the attempts and rows since the previous event; the consumer joins
        them to those it holds since the previous observation for the running
        checkpoint, and keeps them for the next plane (SEP 47 section 5.5).

        Returns
        -------
        solprop.backends.events.EventBatch
            The snapshot request.
        """
        history, rows = self._chunk()
        _hook(self._driver, "clear_halt")
        self._snapshot_pending = None
        return self._issue(
            [
                ServiceRequired(
                    kind=ServiceKind.SNAPSHOT,
                    z=self._z,
                    message="snapshot at an accepted endpoint",
                    ticket=self._ticket(),
                    history=history,
                    step_stats=rows,
                )
            ]
        )

    def _complete(self) -> EventBatch:
        """Run the final checks and report completion.

        Returns
        -------
        solprop.backends.events.EventBatch
            The terminal event.
        """
        chunk = self._chunk()
        try:
            if self._monitor is not None and self._attached:
                self._monitor.finish()
        except Exception as error:
            return self._terminal(Outcome.FAILED, error, chunk)
        return self._issue(
            [
                Terminal(
                    Outcome.COMPLETED,
                    self._z,
                    Eligibility.ELIGIBLE,
                    None,
                    None,
                    *chunk,
                )
            ],
            outcome=Outcome.COMPLETED,
        )

    def _terminal(
        self,
        outcome: Outcome,
        error: BaseException,
        chunk: tuple[BackendHistory, StepStats] | None = None,
    ) -> EventBatch:
        """Report a stop or failure at the last committed position.

        Parameters
        ----------
        outcome : solprop.backends.events.Outcome
            Stopped or failed.
        error : BaseException
            The primary failure or the stop request.
        chunk : tuple or None, optional
            The attempts and rows since the previous event, when already drained for
            an event the failure replaced; drained here otherwise.

        Returns
        -------
        solprop.backends.events.EventBatch
            The terminal event, with a ticket when the state is valid.
        """
        poisoned = self._driver.in_flight is not None
        history, rows = self._chunk() if chunk is None else chunk
        self._generation += 1
        self._z = float(getattr(self._driver, "_z", self._z))
        event = Terminal(
            outcome,
            self._z,
            Eligibility.POISONED if poisoned else Eligibility.ELIGIBLE,
            error,
            None if poisoned else self._ticket(),
            history,
            rows,
        )
        return self._issue([event], outcome=outcome)

    def _fielded(self, ticket: ExportTicket) -> Observation:
        """Observe the field for a ticket once, sharing it among its consumers.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            A valid ticket.

        Returns
        -------
        solprop.backends.protocol.Observation
            The driver's observation with its field.
        """
        if self._observed is None or self._observed_generation != ticket.generation:
            observed = self._drain(field=True)
            if observed.n_steps or observed.step_stats.n_rows:
                # Attempts a field fetch drained belong to the next event's chunk.
                self._undelivered.append(
                    (observed.step_history, _rows(observed.step_stats))
                )
            self._observed = observed
            self._observed_generation = ticket.generation
            self._converted = None
            self._lent = False
        return self._observed

    def _field(self, ticket: ExportTicket) -> np.ndarray:
        """Return the committed field for a ticket, without a batch axis.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            A valid ticket.

        Returns
        -------
        numpy.ndarray
            The field, internal units and convention, read-only.
        """
        field = self._fielded(ticket).field
        assert field is not None  # observed with its field
        return field[0]


def _bound(event: Event, receipt: Receipt) -> Event:
    """Return an event with its ticket bound to the batch that delivers it.

    Parameters
    ----------
    event : solprop.backends.events.Event
        An event, with an unbound ticket or none.
    receipt : solprop.backends.events.Receipt
        The batch's receipt.

    Returns
    -------
    solprop.backends.events.Event
        The event, its ticket bound; the same event when it carries none.
    """
    ticket = getattr(event, "ticket", None)
    if ticket is None:
        return event
    return dataclasses.replace(
        event,  # pyright: ignore[reportArgumentType]
        ticket=dataclasses.replace(ticket, receipt=receipt),
    )


class DriverGroupSession:
    """A :class:`~solprop.backends.session.GroupSession` over the lockstep runners.

    It loads each member from its start, then hands the loaded members to the
    backend's lockstep runner (:func:`solprop.backends.registry.select_batch`: the
    reference's :class:`~solprop.backends.host.group.HostGroup`, the native bridge's
    :class:`~solprop.backends.batch.LockstepRunner`), and presents the runner's
    common fixed-schedule march through the group protocol: one window per save
    plane, each member's own events, tickets and outcome, and the failure distinction
    of SEP 47 section 6.7. It is a temporary internal adapter, like
    :class:`DriverSession`.

    As the batched scan always has, the first advance gives every member its boundary
    monitor before the launch plane is reported, so that a launch a monitor refuses
    ends that member at once.

    Parameters
    ----------
    members : collections.abc.Sequence of solprop.backends.protocol.Backend
        One constructed backend per member, all of one backend. Once the group is
        constructed it owns them and closes them with itself; a construction that
        fails leaves them to the caller.
    problems : collections.abc.Sequence of solprop.translate.problem.KernelProblem
        Each member's independently translated problem.
    requests : collections.abc.Sequence of solprop.translate.run_request.RunRequest
        Each member's run request, with its own fresh boundary monitor; their save
        planes and statistics are the group's common schedule.
    starts : collections.abc.Sequence of solprop.backends.session.Start
        Each member's start: its entrance, with no method history or monitor to
        continue. The group takes each start's field.
    stop : solprop.log.StopFlag or None, optional
        A group-wide stop token, polled before every common step.

    Raises
    ------
    solprop.backends.session.SessionError
        If the sequences differ in length or are empty, or if a start is not a fresh
        entrance.
    solprop.exceptions.BackendError
        If the runner refuses the members (``select_batch``): another backend, or
        another schedule, planes or periodic statistics.
    """

    def __init__(
        self,
        members: Sequence[Backend],
        problems: Sequence[KernelProblem],
        requests: Sequence[RunRequest],
        starts: Sequence[Start],
        *,
        stop: StopFlag | None = None,
    ) -> None:
        """Load every member from its start and build the lockstep runner."""
        from solprop.backends.registry import select_batch  # noqa: PLC0415 - cycle

        count = len(members)
        check_group(count, problems, requests, starts)
        observation = requests[0].observation
        for member, start in zip(members, starts, strict=True):
            member.set_initial(
                np.asarray(start.take_field())[np.newaxis],
                start.noise,
                z=float(start.z),
                resume=start.resume,
            )
        self._members = tuple(members)
        self._problems = tuple(problems)
        self._requests = tuple(requests)
        self._planes = np.asarray(observation.save_planes, dtype=np.float64)
        self._runner = select_batch(self._members)
        self._runner.set_stop(stop)
        self._ledger = SessionLedger()
        self._launched = False
        self._index = 0
        self._generation = 0
        self._z = float(starts[0].z)
        self._outcomes: list[Outcome | None] = [None] * count
        self._attempts = [0] * count
        self._rejected = [0] * count
        self._observed: dict[int, Observation] = {}
        self._observed_generation = -1
        # Set once a group-wide stop or failure ended every running member.
        self._broken = False
        # Each member's attempts drained since its previous event and not yet delivered.
        self._drained: dict[int, tuple[BackendHistory, StepStats]] = {}

    # -- protocol -------------------------------------------------------------------

    @property
    def n_members(self) -> int:
        """Return the number of members.

        Returns
        -------
        int
            The members.
        """
        return len(self._members)

    @property
    def finished(self) -> bool:
        """Return whether the group's terminal batch was issued.

        Returns
        -------
        bool
            True once every member has ended.
        """
        return self._ledger.finished

    @property
    def closed(self) -> bool:
        """Return whether the group was closed.

        Returns
        -------
        bool
            True once closed.
        """
        return self._ledger.state is SessionState.CLOSED

    def advance(self) -> GroupBatch:
        """Run every running member through the next common window, and report it.

        Returns
        -------
        solprop.backends.events.GroupBatch
            Each member's events in the window.
        """
        self._ledger.check_advance()
        if not self._launched:
            self._launched = True
            return self._launch()
        if self._index >= self._planes.size:
            return self._complete()
        target = float(self._planes[self._index])
        try:
            summary = self._runner.propagate_to(target)
        except StopRequested as error:
            return self._ended(Outcome.STOPPED, error)
        except Exception as error:
            return self._ended(Outcome.FAILED, error)
        self._z = float(summary.z)
        self._generation += 1
        plane = self._index
        self._index += 1
        return self._window(ObservationKind.PLANE, plane, summary)

    def acknowledge(self, receipt: Receipt) -> None:
        """Release the pending batch and allow advancing past it.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The pending batch's receipt.
        """
        self._ledger.acknowledge(receipt)
        self._observed = {}

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
            The member's position.
        ticket : solprop.backends.events.ExportTicket
            A ticket of the pending batch, for this member: its position is the
            member's own.
        recipe : solprop.backends.export.ExportRecipe
            The conversion.
        out : numpy.ndarray or None, optional
            The destination; a new array by default.

        Returns
        -------
        numpy.ndarray
            The member's converted field.
        """
        self._ledger.check_ticket(ticket)
        problem = self._problems[self._member(member)]
        if recipe.power and problem.transform.geometry is not None:
            # A spatial route's field measure is already the stored quantity.
            recipe = dataclasses.replace(recipe, power=False)
        weight = power_weight_at(problem, ticket.z) if recipe.power else None
        field = self._fielded(ticket, member).field
        assert field is not None  # observed with its field
        return apply_recipe(field[0], recipe, weight=weight, out=out)

    def noise_phase(self, member: int, ticket: ExportTicket) -> np.ndarray | None:
        """Return one member's noise phase at a pending ticket.

        Parameters
        ----------
        member : int
            The member's position.
        ticket : solprop.backends.events.ExportTicket
            A ticket of the pending batch.

        Returns
        -------
        numpy.ndarray or None
            The phase, shape ``(n_modes, n_pol, n_omega)``, or ``None`` without noise.
        """
        self._ledger.check_ticket(ticket)
        phase = self._fielded(ticket, self._member(member)).noise_phase
        return None if phase is None else np.array(phase[0])

    def member_status(self, member: int) -> SessionStatus:
        """Return one member's committed position, counters and outcome.

        Parameters
        ----------
        member : int
            The member's position.

        Returns
        -------
        solprop.backends.session.SessionStatus
            Its status; a member that ended reports its own outcome while the group
            goes on.
        """
        index = self._member(member)
        outcome = self._outcomes[index]
        state = self._ledger.state
        if outcome is not None and state is not SessionState.CLOSED:
            state = SessionState.FINISHED
        return SessionStatus(
            z=self._cursor(index),
            plane_index=self._index,
            attempts=self._attempts[index],
            rejected=self._rejected[index],
            state=state,
            outcome=outcome,
        )

    def execution_record(self, member: int) -> Mapping[str, object]:
        """Return one member's execution facts, with the group's strategy.

        Parameters
        ----------
        member : int
            The member's position.

        Returns
        -------
        collections.abc.Mapping
            Its backend provenance.
        """
        return self._runner.member_metadata()[self._member(member)]

    def monitor_record(self, member: int) -> Mapping[str, object] | None:
        """Return one member's boundary-monitor record.

        Parameters
        ----------
        member : int
            The member's position.

        Returns
        -------
        collections.abc.Mapping or None
            Its record as it closed or as it stands, or ``None`` when unwatched.
        """
        return self._runner.member_records[self._member(member)]

    def close(self) -> None:
        """Release every member and the runner; idempotent."""
        if self._ledger.close():
            self._runner.set_stop(None)
            self._runner.close()

    def __enter__(self) -> "DriverGroupSession":  # noqa: UP037 - runtime type hook
        """Return the group.

        Returns
        -------
        DriverGroupSession
            This group.
        """
        return self

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
            The exception type, if one is propagating.
        error : BaseException or None
            The exception, if one is propagating.
        trace : types.TracebackType or None
            Its traceback.
        """
        del kind, error, trace
        self.close()

    # -- internals ------------------------------------------------------------------

    def _member(self, member: int) -> int:
        """Return a valid member position.

        Parameters
        ----------
        member : int
            The position asked for.

        Returns
        -------
        int
            The same position.

        Raises
        ------
        solprop.backends.session.SessionError
            If the group has no such member.
        """
        if not 0 <= member < len(self._members):
            msg = f"the group has {len(self._members)} members, not a member {member}"
            raise SessionError(msg)
        return member

    def _cursor(self, index: int) -> float:
        """Return one member's committed position.

        Members are stepped in turn, so after a failure partway through a common step
        a member ahead of the failing one stands at the step's end, its step neither
        validated nor recorded -- in flight, and so poisoned -- and the others at the
        step's start.

        Parameters
        ----------
        index : int
            The member's position.

        Returns
        -------
        float
            The position, m.
        """
        return float(getattr(self._members[index], "_z", self._z))

    def _running(self) -> list[int]:
        """Return the members that have not ended.

        Returns
        -------
        list of int
            Their positions.
        """
        return [index for index, ended in enumerate(self._outcomes) if ended is None]

    def _chunks(self) -> list[tuple[BackendHistory, StepStats]]:
        """Drain every member's attempts and rows since the previous window.

        Returns
        -------
        list of tuple
            Each member's attempts and rows (without a batch axis), in member order.
        """
        chunks: list[tuple[BackendHistory, StepStats]] = []
        for index in range(len(self._members)):
            if index not in self._drained:
                # One member at a time, each kept until an event delivers it: a later
                # member's failure, or the statistics', ends the group with the
                # attempts already drained delivered by the terminals.
                observation = self._members[index].observe(field=False)
                self._attempts[index] += observation.n_steps
                self._rejected[index] += observation.n_rejected
                self._drained[index] = (
                    observation.step_history,
                    _rows(observation.step_stats),
                )
            chunks.append(self._drained[index])
        return chunks

    def _final_chunk(self, index: int) -> tuple[BackendHistory, StepStats] | None:
        """Drain one member's attempts after a group-wide stop or failure.

        A member left in flight cannot be observed; the attempts it recorded before
        the abandoned step are drained from its record instead (:func:`_recorded`).

        Parameters
        ----------
        index : int
            The member's position.

        Returns
        -------
        tuple or None
            Its attempts and rows, or ``None`` when its state cannot be read at all.
        """
        member = self._members[index]
        try:
            observed = (
                _recorded(member, self._cursor(index))
                if member.in_flight is not None
                else member.observe(field=False)
            )
        except Exception:
            return None
        self._attempts[index] += observed.n_steps
        self._rejected[index] += observed.n_rejected
        return observed.step_history, _rows(observed.step_stats)

    def _ticket(self, z: float | None = None) -> ExportTicket:
        """Return an unbound ticket for a committed state of the current generation.

        Parameters
        ----------
        z : float or None, optional
            The member's committed position, m; the window's end by default.

        Returns
        -------
        solprop.backends.events.ExportTicket
            The ticket, its receipt filled in when the batch is issued.
        """
        return ExportTicket(
            Receipt(self._ledger.session, 0),
            self._generation,
            self._z if z is None else z,
        )

    def _member_stopped(
        self, index: int, chunk: tuple[BackendHistory, StepStats]
    ) -> Terminal | None:
        """Return a stopped member's terminal event, or ``None`` if it runs on.

        Parameters
        ----------
        index : int
            The member's position.
        chunk : tuple
            Its attempts and rows since its previous event.

        Returns
        -------
        solprop.backends.events.Terminal or None
            The boundary stop that ended it, which ends it alone.
        """
        stop = self._runner.member_stops[index]
        if stop is None:
            return None
        self._outcomes[index] = Outcome.FAILED
        cursor = self._cursor(index)
        return Terminal(
            Outcome.FAILED,
            cursor,
            Eligibility.ELIGIBLE,
            stop,
            self._ticket(cursor),
            *chunk,
        )

    def _launch(self) -> GroupBatch:
        """Give each member its monitor and report the launch planes.

        Returns
        -------
        solprop.backends.events.GroupBatch
            Each member's launch plane, or the boundary stop that refused it.
        """
        try:
            self._runner.attach_monitors(
                [request.boundary.monitor for request in self._requests]
            )
        except Exception as error:
            return self._ended(Outcome.FAILED, error)
        return self._window(ObservationKind.LAUNCH, -1, _EMPTY_SUMMARY)

    def _window(
        self, kind: ObservationKind, plane: int, summary: StepSummary
    ) -> GroupBatch:
        """Report a window's end: each running member's plane, or its boundary stop.

        Parameters
        ----------
        kind : solprop.backends.events.ObservationKind
            The launch or a save plane.
        plane : int
            The save plane, ``-1`` for the launch.
        summary : solprop.backends.protocol.StepSummary
            The common march to it; its maximum norm is a group diagnostic.

        Returns
        -------
        solprop.backends.events.GroupBatch
            The window's events.
        """
        try:
            chunks = self._chunks()
            stats = self._runner.stats(self._requests[0].observation.stats)
        except Exception as error:
            return self._ended(Outcome.FAILED, error)
        events: list[tuple[Event, ...]] = []
        for index, chunk in enumerate(chunks):
            if self._outcomes[index] is not None:
                events.append(())
                continue
            stopped = self._member_stopped(index, chunk)
            if stopped is not None:
                events.append((stopped,))
                continue
            events.append(
                (
                    ObservationReady(
                        z=self._z,
                        kind=kind,
                        plane_index=plane,
                        history=chunk[0],
                        step_stats=chunk[1],
                        stats={
                            name: np.asarray(value[index])
                            for name, value in stats.items()
                        },
                        summary=summary,
                        ticket=self._ticket(),
                    ),
                )
            )
        if not self._running():
            # Every member ended at its own boundary: nothing is left to advance.
            return self._issue(events, outcome=Outcome.COMPLETED)
        return self._issue(events)

    def _complete(self) -> GroupBatch:
        """Run each watched member's final check and report every outcome.

        Returns
        -------
        solprop.backends.events.GroupBatch
            Each running member's completion, or the final warning that failed it.
        """
        try:
            chunks = self._chunks()
            self._runner.finish_monitors()
        except Exception as error:
            return self._ended(Outcome.FAILED, error)
        events: list[tuple[Event, ...]] = []
        for index, chunk in enumerate(chunks):
            if self._outcomes[index] is not None:
                events.append(())
                continue
            stopped = self._member_stopped(index, chunk)
            if stopped is not None:
                events.append((stopped,))
                continue
            self._outcomes[index] = Outcome.COMPLETED
            events.append(
                (
                    Terminal(
                        Outcome.COMPLETED,
                        self._cursor(index),
                        Eligibility.ELIGIBLE,
                        None,
                        None,
                        *chunk,
                    ),
                )
            )
        return self._issue(events, outcome=Outcome.COMPLETED)

    def _ended(self, outcome: Outcome, error: BaseException) -> GroupBatch:
        """End every running member with a group-wide stop or failure.

        Parameters
        ----------
        outcome : solprop.backends.events.Outcome
            Stopped or failed.
        error : BaseException
            The primary failure or the stop request.

        Returns
        -------
        solprop.backends.events.GroupBatch
            Each running member's terminal event at its own committed position.
        """
        self._generation += 1
        self._broken = True
        chunks = [
            join_attempts(
                [
                    self._drained.get(index, (None, None)),
                    self._final_chunk(index) or (None, None),
                ]
            )
            for index in range(len(self._members))
        ]
        events: list[tuple[Event, ...]] = []
        for index, chunk in enumerate(chunks):
            if self._outcomes[index] is not None:
                events.append(())
                continue
            self._outcomes[index] = outcome
            poisoned = self._members[index].in_flight is not None
            history, rows = chunk
            cursor = self._cursor(index)
            events.append(
                (
                    Terminal(
                        outcome,
                        cursor,
                        Eligibility.POISONED if poisoned else Eligibility.ELIGIBLE,
                        error,
                        None if poisoned else self._ticket(cursor),
                        history,
                        rows,
                    ),
                )
            )
        return self._issue(events, outcome=outcome, error=error)

    def _issue(
        self,
        members: list[tuple[Event, ...]],
        *,
        outcome: Outcome | None = None,
        error: BaseException | None = None,
    ) -> GroupBatch:
        """Issue a window's events under a new receipt.

        Parameters
        ----------
        members : list
            Each member's events; tickets are bound to the receipt here.
        outcome : solprop.backends.events.Outcome or None, optional
            The group's outcome, when the batch ends it.
        error : BaseException or None, optional
            A group-wide failure's primary error.

        Returns
        -------
        solprop.backends.events.GroupBatch
            The batch.
        """
        receipt = self._ledger.issue(outcome=outcome)
        # Every drained attempt is delivered by this batch.
        self._drained = {}
        bound = tuple(
            tuple(_bound(event, receipt) for event in events) for events in members
        )
        return GroupBatch(receipt, bound, outcome, error)

    def _fielded(self, ticket: ExportTicket, member: int) -> Observation:
        """Observe a member's field for a ticket once, shared by its exports.

        A window's fields are observed together, once; after a group-wide failure each
        member is observed alone, since a member left in flight refuses observation.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            A valid ticket.
        member : int
            The member's position.

        Returns
        -------
        solprop.backends.protocol.Observation
            The member's observation with its field.
        """
        if self._observed_generation != ticket.generation:
            self._observed = {}
            self._observed_generation = ticket.generation
        if member not in self._observed:
            # The window's attempts were drained when it ended; nothing ran since.
            if self._broken:
                self._observed[member] = self._members[member].observe(field=True)
            else:
                self._observed.update(
                    enumerate(self._runner.observe_members(field=True))
                )
        return self._observed[member]
