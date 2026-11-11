# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference's run session (SEP 47 sections 5 and 6.6).

:class:`HostSession` is the :class:`~solprop.backends.session.RunSession` that serves
every ``numpy_ref`` run, in both precisions: the launch-before-monitor order, the plane
schedule, the snapshot latch, bounded yields, chunked attempts, exports and the terminal
outcome with its checkpoint eligibility, over the reference's own driver
(:class:`solprop.backends.host.driver.HostDriver`), through its halt and yield hooks
(``request_halt``, ``clear_halt``, ``halt_ahead``, ``set_halt_check``) and its read-only
march state (``halted``, ``holding_plane``, ``position``, ``drain_attempts``).
:class:`HostPrepared` is the prepared execution (SEP 47 section 5.2) a reference backend
returns from ``prepare``; each of its starts opens one.

The session composes the parts every Python session shares
(:mod:`solprop.backends.session_core`: the attempt counter, the yield check, the
snapshot latch, the shared observation and its exports, and the preparation's
refusals) with the lifecycle ledger (:class:`~solprop.backends.session.SessionLedger`).
It imports nothing of the native backend's.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Any

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
from solprop.backends.export import ExportRecipe
from solprop.backends.host.driver import HostDriver
from solprop.backends.protocol import (
    BackendHistory,
    Observation,
    StepStats,
    StepSummary,
)
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
from solprop.exceptions import BackendError, StopRequested
from solprop.log import StopFlag
from solprop.monitor import BoundaryMonitor
from solprop.output.schema import MarchSnapshot
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    # Under TYPE_CHECKING, so that the runtime type checks of the test suite do not
    # hold the scripted drivers to the full protocol or check a refused export
    # destination's shape before the session refuses it.
    from jaxtyping import Complex

    from solprop.backends.protocol import Backend, HostStepping

__all__ = ["HostPrepared", "HostSession"]


class HostSession:
    """The reference's :class:`~solprop.backends.session.RunSession`.

    Parameters
    ----------
    driver : solprop.backends.protocol.Backend
        The reference backend (a :class:`~solprop.backends.host.driver.HostDriver`),
        constructed for `problem`. A driver reused from an earlier run is loaded again
        from `start`, which restores everything its march reads. The session closes it
        with itself when `owns` is set.
    problem : solprop.translate.problem.PropagationProblem
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
        The run's stop token, polled before every trial.
    owns : bool, optional
        Whether closing the session closes the driver. A caller that reuses its driver
        for later runs (a runner, a scan) keeps it open.

    Raises
    ------
    solprop.exceptions.BackendError
        If `driver` is not a reference driver.
    """

    def __init__(  # noqa: PLR0913 - the session context
        self,
        driver: Backend,
        problem: PropagationProblem,
        request: RunRequest,
        start: Start,
        *,
        listener: AttemptListener | None = None,
        started: Callable[[float, float, int], None] | None = None,
        stop: StopFlag | None = None,
        owns: bool = True,
    ) -> None:
        """Refuse a driver that is not the reference's, and load the starting state."""
        if not isinstance(driver, HostDriver):
            msg = (
                f"the reference session serves a host driver, not "
                f"{type(driver).__name__}"
            )
            raise BackendError(msg)
        self._driver = driver
        # The same object, as the march whose hooks and state the session reads.
        self._host: HostDriver[Any] = driver
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
        self._latch = SnapshotLatch()
        self._fields = ObservedField(problem, self._drain)
        self._counter = AttemptCounter(listener, started)
        driver.set_progress(self._counter)
        driver.set_stop(stop)
        driver.set_initial(
            np.asarray(start.take_field()),
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
    def driver(self) -> HostStepping:
        """Return the reference backend, whose facts the service loop records.

        Returns
        -------
        solprop.backends.protocol.HostStepping
            The backend, through the host stepping seam.
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

    def advance(  # noqa: PLR0911 - one return per barrier kind
        self,
        *,
        work_limit: int = DEFAULT_WORK_LIMIT,
        deadline: float = DEFAULT_DEADLINE,
    ) -> EventBatch:
        """Run to the next barrier and report it.

        A batch ends at the next observation, snapshot or terminal outcome, or at a
        bounded yield: once `work_limit` attempts are made or `deadline` has passed,
        the march returns at its next accepted endpoint, exactly as at a natural stop,
        and the batch reports only progress. Nothing numerical depends on where a
        yield falls.

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
            return self._observation(ObservationKind.LAUNCH, -1, EMPTY_SUMMARY)
        if not self._attached:
            self._attach_monitor()
        host = self._host
        if self._latch.pending is not None and self._at_endpoint():
            return self._snapshot()
        if self._index >= self._planes.size:
            return self._complete()
        target = float(self._planes[self._index])
        if self._latch.pending is not None and host.holding_plane:
            # The march restores the endpoint ahead of the plane it holds, and halts
            # there if the target lies beyond: planes closer than the steps would
            # otherwise put off the snapshot with every step (SEP 47 section 5.5).
            host.halt_ahead()
        started = time.perf_counter()
        limit = self._counter.count + max(int(work_limit), 1)
        host.set_halt_check(YieldDue(self._counter, limit, started, deadline))
        try:
            step = self._driver.propagate_to(target)
        except StopRequested as error:
            return self._terminal(Outcome.STOPPED, error)
        except Exception as error:
            return self._terminal(Outcome.FAILED, error)
        finally:
            host.set_halt_check(None)
        self._account(step)
        summary = merge_summaries(self._summary, step)
        self._summary = summary
        if self._latch.pending is None and not host.halted:
            # A yield latched that the target came first.
            host.clear_halt()
        if bool(np.isclose(step.z, target, rtol=1e-12, atol=0.0)):
            plane = self._index
            self._index += 1
            self._summary = None
            return self._observation(ObservationKind.PLANE, plane, summary)
        if host.halted:
            # A halt short of the target is never a step plane: the driver counts a
            # step plane first and stops there unhalted, so a coincident plane is
            # observed below and the latch, still set, is serviced at the next advance
            # (SEP 47 section 5.3: a coincident plane's transaction precedes the
            # snapshot).
            if self._latch.pending is not None:
                return self._snapshot()
            # A bounded yield: no barrier, only progress; the plane goes on later.
            host.clear_halt()
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

        A service failure, or a signal raised while the driver held no uncommitted
        state, leaves the field where the last commit put it. When a batch is pending,
        the ticket is bound to it; otherwise a failed terminal batch is issued, so the
        state can be exported for a terminal checkpoint and nothing advances again.

        Returns
        -------
        solprop.backends.events.ExportTicket or None
            The ticket, or ``None`` when the session is closed or its state poisoned.
        """
        if self.closed or self._host.in_flight is not None:
            return None
        if self._ledger.state is not SessionState.PENDING:
            self._generation += 1
            self._z = self._host.position
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
        receipt = self._latch.request()
        if not receipt.coalesced:
            self._host.request_halt()
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

    def close(self) -> None:
        """Release the driver; idempotent."""
        if self._ledger.close():
            self._driver.set_stop(None)
            if self._owns:
                self._driver.close()

    def __enter__(self) -> "HostSession":  # noqa: UP037 - runtime type hook
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
        abandoned step are drained from the driver's record instead
        (:meth:`~solprop.backends.host.driver.HostDriver.drain_attempts`).

        Returns
        -------
        tuple
            The attempts and the rows (without a batch axis), delivered by exactly one
            event (:mod:`solprop.backends.events`).
        """
        drained = (
            self._counted(self._host.drain_attempts())
            if self._host.in_flight is not None
            else self._drain(field=False)
        )
        chunks = self._fields.undelivered()
        chunks.append((drained.step_history, drained.step_stats))
        return join_attempts(chunks)

    def _at_endpoint(self) -> bool:
        """Return whether the driver stands at an accepted endpoint.

        Returns
        -------
        bool
            False while a dense observation holds the state between endpoints.
        """
        # The starting position is no accepted endpoint of this session: an Adams
        # march holds no history there to snapshot.
        return self._stepped and not self._host.holding_plane

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
        return EventBatch(receipt, tuple(bind(event, receipt) for event in events))

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
        self._fields.release()
        try:
            stats = dict(self._driver.stats(self._request.observation.stats))
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
        self._host.clear_halt()
        self._latch.clear()
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
        poisoned = self._host.in_flight is not None
        history, rows = self._chunk() if chunk is None else chunk
        self._generation += 1
        self._z = self._host.position
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
    driver : solprop.backends.protocol.Backend
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
        self, driver: Backend, problem: PropagationProblem, request: RunRequest
    ) -> None:
        """Bind the problem and the request to the backend."""
        check_preparation(driver.name, driver.request, problem, request)
        self._driver = driver
        self._problem = problem
        self._request = request
        self._closed = False

    @property
    def backend(self) -> str:
        """Return the registered name of the backend that prepared it.

        Returns
        -------
        str
            The backend's registered name.
        """
        return self._driver.name

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
            self._driver,
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
        return self._driver.metadata()

    def memory_ledger(self) -> Mapping[str, int]:
        """Return every retained buffer by name, bytes.

        Returns
        -------
        collections.abc.Mapping
            The backend's ledger.
        """
        return self._driver.memory_ledger()

    def close(self) -> None:
        """Refuse further starts and close the backend it was prepared on; idempotent.

        A caller that keeps the backend for later runs leaves the preparation open: a
        session it started with ``owns=False`` never closes the backend.
        """
        if self._closed:
            return
        self._closed = True
        self._driver.close()
