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

It is a temporary internal adapter, named in SEP 47's Part II with its removal
milestone; nothing outside the session machinery constructs it directly.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import TYPE_CHECKING, cast

import numpy as np

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

__all__ = ["AttemptListener", "DriverSession"]

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
        The driver, freshly constructed for `problem`, holding no state yet. The session
        owns it from here and closes it with itself.
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    start : solprop.backends.session.Start or None
        Where the run starts; ``None`` when the caller already loaded the driver and
        the run starts at its first plane with the launch plane owed (transitional).
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
        start: Start | None,
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
        self._index = (
            request.observation.start_plane if start is None else start.plane_index
        )
        self._launch_pending = True if start is None else start.plane_pending
        # The session takes over the request's fresh monitor and feeds it.
        self._monitor: BoundaryMonitor | None = request.boundary.monitor
        self._monitor_resume = None if start is None else start.monitor
        self._attached = False
        self._generation = 0
        self._z = float(getattr(driver, "_z", 0.0) if start is None else start.z)
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
        if start is None:
            return
        driver.set_initial(
            np.asarray(start.field)[np.newaxis],
            start.noise,
            z=float(start.z),
            resume=start.resume,
        )
        if start.ledgers is not None:
            logical = start.ledgers.get("logical", {})
            atomic = start.ledgers.get("atomic", {})
            driver.restore_call_ledgers(logical, atomic)  # pyright: ignore[reportArgumentType]

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

        For a caller that loaded the driver itself (``start=None``) and so could not
        pass the continuation in :class:`~solprop.backends.session.Start`; one given
        there is kept.

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

    def restore_march(self, snapshot: MarchSnapshot) -> str | None:
        """Install a persisted method history before the first advance.

        Parameters
        ----------
        snapshot : solprop.output.schema.MarchSnapshot
            The persisted history.

        Returns
        -------
        str or None
            Why it could not be installed, or ``None`` when it was.
        """
        return self._driver.restore_march(snapshot)

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
