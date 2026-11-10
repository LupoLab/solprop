# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The service loop: Python's side of an execution session (SEP 47 section 5).

A session (:mod:`solprop.backends.session`) owns the run's numerical state and its
schedule. This loop advances it and services what it reports -- saved planes and their
output transactions, plane observers and progress, snapshot persistence, warnings and
the terminal record -- then acknowledges each batch. It holds no numerical state of its
own: it never rediscovers the next plane, rebuilds a statistic or replays the attempts
to reconstruct a controller.

Ordering follows SEP 47 section 5.3: a plane is committed with its method history and
its execution record in one transaction (``solprop._interrupts.committing``); a
snapshot is serviced after that commit; the plane observer and the reporter follow.
A service failure leaves the batch unacknowledged, so the field it names can still be
exported for the terminal checkpoint, and the primary failure is preserved.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from solprop._host import RunClock, usage
from solprop._interrupts import committing, poll
from solprop.backends.driver_session import DriverSession
from solprop.backends.events import (
    EventBatch,
    ExportTicket,
    ObservationKind,
    ObservationReady,
    Outcome,
    Progress,
    ServiceKind,
    ServiceRequired,
    Terminal,
    emit_warning,
    join_attempts,
)
from solprop.backends.export import ExportRecipe
from solprop.backends.protocol import Backend, BackendHistory
from solprop.backends.protocol import StepStats as BackendRows
from solprop.backends.session import RunSession
from solprop.driver.plane_view import PlaneWatch
from solprop.driver.run import (
    SessionFacts,
    _announce,
    _earlier_execution,
    _FailedObservation,
    _history_of,
    _MarchKeeper,
    _method_history,
    _open_run,
    _record_abandoned_step,
    _record_execution,
    _refuse_unread_collar,
    _resume_position,
    _retain,
    _Status,
    retention_at,
)
from solprop.driver.summary import resources
from solprop.exceptions import ResponseBudgetError, RunTerminated, StopRequested
from solprop.fourier import GridKind, spectrum_to_internal
from solprop.log import (
    InterruptibleReporter,
    ProgressReporter,
    StopFlag,
    SummarisingReporter,
)
from solprop.output.hdf5 import HDF5Output
from solprop.output.schema import (
    CheckpointOutput,
    FieldRetention,
    Output,
    PlaneStore,
    RecipeRecord,
    RunMetadata,
    SavedPlane,
    StepHistory,
    StepStats,
    StoredField,
    TerminalCheckpoint,
    _adopted,
    stored_spectrum,
)
from solprop.output.snapshots import SnapshotController
from solprop.output.tdse_archive import TDSEArchiveWriter
from solprop.stepper.multistep import Multistep
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest
from solprop.translate.tdse import TDSEBinding

__all__ = ["MONITOR_RECONSTRUCTED", "ServiceContext", "serve"]

logger = logging.getLogger("solprop.driver.run")

_CONSUMER = ExportRecipe(power=True, optics=False)
"""The field a plane observer, a checkpoint record and an output that is not a plane
store hold: output amplitudes in the internal FFT convention (SEP 47 section 5.4)."""

_STORED = ExportRecipe(power=True, optics=True)
"""The field a plane store keeps: :data:`_CONSUMER` in the optics convention, which the
store's own conversion makes (``PlaneStore.save``)."""


@dataclass(frozen=True)
class ServiceContext:
    """What the service loop needs besides the session and the output.

    Attributes
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    metadata : solprop.output.schema.RunMetadata
        The run's record.
    reporter : solprop.log.ProgressReporter
        Where progress goes.
    retention : solprop.output.schema.FieldRetention
        Which planes keep their field.
    resume_history : bool or None
        The public method-history retention choice.
    snapshots : solprop.output.snapshots.SnapshotController or None
        Snapshot requests, when enabled.
    tdse_binding : solprop.translate.tdse.TDSEBinding or None
        A live atomic runtime.
    tdse_archive : solprop.output.tdse_archive.TDSEArchiveWriter or None
        A live archive to suspend beside the optical writer.
    recipe : solprop.output.schema.RecipeRecord or None
        The recipe the run was built from.
    start_plane : int
        The first save plane beyond the starting state.
    stop : solprop.log.StopFlag or None
        The run's stop flag, polled between batches as well as inside the session.
    restoration : collections.abc.Mapping or None, optional
        What a resumed execution restored of the method history
        (``Simulation._continue_march``), for its
        execution record; ``None`` for a fresh run or a caller that says nothing.
    """

    problem: PropagationProblem
    request: RunRequest
    metadata: RunMetadata
    reporter: ProgressReporter
    retention: FieldRetention
    resume_history: bool | None
    snapshots: SnapshotController | None
    tdse_binding: TDSEBinding | None
    tdse_archive: TDSEArchiveWriter | None
    recipe: RecipeRecord | None
    start_plane: int
    stop: StopFlag | None = None
    restoration: Mapping[str, object] | None = None


def _rows(rows: BackendRows) -> StepStats:
    """Return a session's periodic rows in the form an output handler stores.

    Parameters
    ----------
    rows : solprop.backends.protocol.StepStats
        The rows, with a leading row axis and no batch axis.

    Returns
    -------
    solprop.output.schema.StepStats
        The same rows.
    """
    return StepStats(z=rows.z, step=rows.step, values=dict(rows.values))


MONITOR_RECONSTRUCTED = {
    "state": "reconstructed",
    "reason": "persisted_records",
    "restored": ["cumulative_totals", "quiet_state", "transverse_record"],
    "restarted": ["check_cadence", "transverse_cap"],
}
"""How a resumed execution's boundary monitor was restored (SEP 47 section 5.5).

Today's reconstruction from the persisted records: the cumulative totals, the quiet
state and the transverse record are restored, and the check cadence and the transverse
cap restart from that restored state; persisting them is not claimed."""


def _restart(base: Mapping[str, object], keeper: _MarchKeeper) -> dict[str, object]:
    """Return an execution's restart record with the latest retention decision.

    Parameters
    ----------
    base : collections.abc.Mapping
        What the execution restored, fixed when it started.
    keeper : solprop.driver.run._MarchKeeper
        The method-history keeper, holding the latest retention decision.

    Returns
    -------
    dict
        The record (:data:`solprop.driver.run.RESTART_RECORD`).
    """
    retention = keeper.record()
    return dict(base) if retention is None else {"retention": retention, **base}


class _Exports:
    """One export per pending ticket, shared by every consumer.

    The output, a plane observer and a terminal checkpoint asking for the same field
    share one export (SEP 47 section 5.4); a field-free plane fetches nothing unless a
    consumer asks. A plane store's field is exported straight into its stored form,
    into the store's own room where it has one; a consumer that wants the internal
    convention after that is given the exact inverse of the optics conversion
    (:func:`solprop.fourier.spectrum_to_internal`), with no second export. An output
    that is not a plane store is given the consumer field itself.

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        The session whose tickets are exported.
    kind : {'field', 'envelope'}
        The grid's spectral layout.
    """

    def __init__(self, session: RunSession, kind: GridKind) -> None:
        """Start with nothing exported."""
        self._session = session
        self._kind: GridKind = kind
        self._ticket: ExportTicket | None = None
        self._field: np.ndarray | None = None
        self._stored: np.ndarray | None = None
        self._phase: np.ndarray | None = None
        self._phased = False

    def _hold(self, ticket: ExportTicket) -> None:
        """Forget what another ticket exported.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The ticket now asked for.
        """
        if self._ticket != ticket:
            self._ticket = ticket
            self._field = self._stored = self._phase = None
            self._phased = False

    def field(self, ticket: ExportTicket) -> np.ndarray:
        """Return the consumer field for a ticket, exporting it once.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            A pending ticket.

        Returns
        -------
        numpy.ndarray
            Read-only output amplitudes in the internal convention.
        """
        self._hold(ticket)
        if self._field is None:
            if self._stored is not None:
                field = spectrum_to_internal(self._stored, kind=self._kind)
            else:
                field = self._session.export(ticket, _CONSUMER)
            field.setflags(write=False)
            self._field = field
        return self._field

    def stored(self, ticket: ExportTicket, out: np.ndarray | None) -> np.ndarray:
        """Return the stored form of a ticket's field, exported once into `out`.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            A pending ticket.
        out : numpy.ndarray or None
            The plane store's room (``PlaneStore.field_destination``), or ``None``
            for a new array.

        Returns
        -------
        numpy.ndarray
            Read-only output amplitudes in the optics convention: `out` when given.
        """
        self._hold(ticket)
        if self._stored is None:
            if self._field is not None:
                # Exported already for another consumer: the store's own conversion.
                stored = stored_spectrum(self._field, self._kind, out)
            else:
                stored = self._session.export(ticket, _STORED, out=out)
            stored.setflags(write=False)
            self._stored = stored
        return self._stored

    def release(self) -> None:
        """Drop the exports once their batch is acknowledged."""
        self._ticket = self._field = self._stored = self._phase = None
        self._phased = False

    def phase(self, ticket: ExportTicket) -> np.ndarray | None:
        """Return the noise phase for a ticket, read once.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            A pending ticket.

        Returns
        -------
        numpy.ndarray or None
            The phase, read-only, or ``None`` without a noise model.
        """
        self._hold(ticket)
        if not self._phased:
            phase = self._session.noise_phase(ticket)
            if phase is not None:
                phase.setflags(write=False)
            self._phase, self._phased = phase, True
        return self._phase


_Chunk = tuple[BackendHistory | None, BackendRows | None]
"""One event's attempts and periodic rows, as it delivered them."""


class _Chunks:
    """The attempts and rows delivered since the last saved plane, in order.

    A session delivers each attempt once, with the event after it
    (:mod:`solprop.backends.events`); this is where they are assembled. An observation
    takes them all as its plane's history and starts afresh; a running checkpoint
    includes them without consuming them; a terminal checkpoint includes them.

    Parameters
    ----------
    step : callable
        The progress reporter's per-attempt call, made with each attempt's end,
        length and acceptance as its chunk arrives.
    """

    def __init__(self, step: Callable[[float, float, bool], None]) -> None:
        """Start with nothing held."""
        self._chunks: list[_Chunk] = []
        self._step = step

    def add(
        self,
        history: BackendHistory | None,
        rows: BackendRows | None,
        *,
        report: bool = True,
    ) -> None:
        """Hold one event's chunk, and report its attempts.

        Every attempt arrives exactly once, so the reporter hears each once, in order,
        at the cadence of the session's yields rather than from inside its march.

        Parameters
        ----------
        history : solprop.backends.protocol.BackendHistory or None
            Its attempts.
        rows : solprop.backends.protocol.StepStats or None
            Its periodic rows, without a batch axis.
        report : bool, optional
            Whether the reporter hears the attempts; ``False`` for the attempts no
            event delivered, which a failure's checkpoint holds and no reporter call
            may lose.
        """
        self._chunks.append((history, rows))
        if report and history is not None:
            for z, dz, accepted in zip(
                history.z.tolist(),
                history.dz.tolist(),
                history.accepted.tolist(),
                strict=True,
            ):
                self._step(z, dz, accepted)

    def joined(self) -> tuple[BackendHistory, BackendRows]:
        """Return every chunk held, assembled, without consuming them.

        Returns
        -------
        tuple
            The attempts and rows since the last saved plane.
        """
        return join_attempts(self._chunks)

    def take(self) -> tuple[BackendHistory, BackendRows]:
        """Return every chunk held, assembled, and start afresh.

        Returns
        -------
        tuple
            The attempts and rows since the last saved plane.
        """
        joined = self.joined()
        self._chunks = []
        return joined


def _plane(
    event: ObservationReady,
    attempts: tuple[BackendHistory, BackendRows],
    field: np.ndarray | None,
    phase: np.ndarray | None,
) -> SavedPlane:
    """Build the plane an output stores from an observation.

    Parameters
    ----------
    event : solprop.backends.events.ObservationReady
        The observation.
    attempts : tuple
        Its history and rows, assembled from the chunks since the previous plane.
    field : numpy.ndarray or None
        Its field in the internal convention, or ``None``: a field-free plane, or one
        whose field is exported straight into a plane store.
    phase : numpy.ndarray or None
        Its noise phase, where it keeps its field and a noise model runs.

    Returns
    -------
    solprop.output.schema.SavedPlane
        The plane.
    """
    return SavedPlane(
        z=event.z,
        field=None if field is None else _adopted(field),
        stats=dict(event.stats),
        step_history=_history_of(attempts[0]),
        step_stats=_rows(attempts[1]),
        noise_phase=None if phase is None else _adopted(phase),
    )


def _prepare_plane(
    event: ObservationReady,
    output: Output,
    exports: _Exports,
    attempts: tuple[BackendHistory, BackendRows],
    shape: tuple[int, ...],
    *,
    keep: bool,
) -> tuple[SavedPlane, StoredField | None]:
    """Build an observation's plane, its kept field exported once for the output.

    A plane store that :attr:`~solprop.output.schema.PlaneStore.takes_exports` is
    given the field exported straight into its stored form, into its own room where it
    has one (SEP 47 section 5.4), and the plane carries none; any other output is
    given the consumer field on the plane (:func:`_save_plane`).

    Parameters
    ----------
    event : solprop.backends.events.ObservationReady
        The observation.
    output : solprop.output.schema.Output
        The run's output.
    exports : _Exports
        The pending ticket's exports.
    attempts : tuple
        Its history and rows.
    shape : tuple of int
        ``(n_modes, n_pol, n_omega)`` of the field.
    keep : bool
        Whether the plane keeps its field.

    Returns
    -------
    tuple
        The plane and, where its field was exported into a plane store, the stored
        field (``None`` otherwise).
    """
    ticket = event.ticket
    phase = exports.phase(ticket) if keep else None
    if keep and isinstance(output, PlaneStore) and output.takes_exports:
        plane = _plane(event, attempts, None, None)
        room = output.field_destination(plane, shape)
        stored = StoredField(
            exports.stored(ticket, room), None if phase is None else _adopted(phase)
        )
        return plane, stored
    return _plane(event, attempts, exports.field(ticket) if keep else None, phase), None


def _save_plane(output: Output, plane: SavedPlane, stored: StoredField | None) -> None:
    """Hand a prepared plane to the output (:func:`_prepare_plane`).

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's output.
    plane : solprop.output.schema.SavedPlane
        The plane.
    stored : solprop.output.schema.StoredField or None
        Its field exported into the plane store's stored form, or ``None``.
    """
    if stored is None:
        output.save(plane)
    else:
        assert isinstance(output, PlaneStore)
        output.save_exported(plane, stored)


def serve(  # noqa: PLR0912, PLR0915 - the commit sequence of SEP 47 section 5.3
    session: RunSession,
    output: Output,
    context: ServiceContext,
) -> None:
    """Advance a session to its end, servicing every event it reports.

    Any :class:`~solprop.backends.session.RunSession` is served: the transitional
    :class:`~solprop.backends.driver_session.DriverSession`, whose driver gives the
    execution facts and method history, or a backend's own session, through its
    protocol alone (:class:`~solprop.driver.run.SessionFacts`).

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        The started session.
    output : solprop.output.schema.Output
        Where the planes go; closed when the loop ends, whether or not it succeeded.
    context : ServiceContext
        Everything else the service needs.

    Raises
    ------
    BaseException
        The run's primary failure, a stop request or a signal, after its terminal
        record and any checkpoint are written.
    """
    problem, reporter = context.problem, context.reporter
    backend = _facts(session)
    planes = context.request.observation.save_planes
    started = time.perf_counter()
    clock = RunClock.start()
    checkpoint_output = _open_run(
        output, context.metadata, backend, context.tdse_binding
    )
    if context.recipe is not None and isinstance(output, PlaneStore):
        with committing():
            output.record_recipe(context.recipe)
    reserve = getattr(output, "reserve", None)
    if reserve is not None:
        reserve(planes.size + 1)
    _retain(output, context.retention, problem.shape)
    reached, missing_current_save = _resume_position(output, checkpoint_output, problem)
    _refuse_unread_collar(problem, backend)
    # The request's own monitor: its thresholds; the session owns its state.
    monitor = context.request.boundary.monitor
    # Read before this execution's first record replaces the earlier one's.
    earlier = None if checkpoint_output is None else checkpoint_output.outcome
    resumed, monitored, unwatched = _earlier_execution(output, checkpoint_output)
    base: dict[str, object] = {}
    if context.restoration is not None:
        base["restoration"] = dict(context.restoration)
    if resumed and monitor is not None:
        # The record continues across the resume (SEP 21, ADR-057).
        session.continue_monitor(
            {
                "record": monitored,
                "plane": missing_current_save
                or output.n_saved == 0
                or float(output["z"][-1]) == reached,
                "unwatched": unwatched,
            }
        )
        base["monitor"] = dict(MONITOR_RECONSTRUCTED)
    _announce(
        problem,
        context.metadata,
        backend,
        output,
        reporter,
        (context.start_plane, reached),
        boundary_check=None if monitor is None else monitor.check,
    )
    keeper = _MarchKeeper(
        output,
        backend,
        context.resume_history,
        method=_method_history(problem),
        retention=None if earlier is None else retention_at(earlier.execution, reached),
    )
    exports = _Exports(session, context.metadata.grid.kind)
    shape = tuple(context.problem.shape)
    current: list[ObservationReady | None] = [None]

    def fetch() -> np.ndarray:
        """Return the pending observation's field, for the plane observer.

        Returns
        -------
        numpy.ndarray
            The field of the observation the observer is shown.
        """
        event = current[0]
        assert event is not None  # the observer is shown a pending plane only
        return exports.field(event.ticket)

    watch = PlaneWatch(reporter, context.metadata, fetch, clock.observer)
    pending: SavedPlane | None = None
    pending_save_count = output.n_saved
    pending_ticket: ExportTicket | None = None
    snapshots = context.snapshots
    snapshots_started = False
    terminal_rows: tuple[object, object] = (None, None)
    chunks = _Chunks(reporter.step)
    failure: BaseException | None = None
    adams = isinstance(context.request.integration.method, Multistep)
    try:
        if snapshots is not None:
            snapshots._start()
            snapshots_started = True
        _record_execution(
            checkpoint_output,
            backend,
            context.tdse_binding,
            _Status("running", reached),
            clock=clock,
            restart=_restart(base, keeper),
        )
        batch: EventBatch | None = None
        while True:
            if batch is not None:
                # Between batches, before the next propagation: a stop set during the
                # service is honoured here, while the batch's field can still be
                # exported for the checkpoint (SEP 41, S17).
                poll(context.stop)
                session.acknowledge(batch.receipt)
                exports.release()
                current[0] = None
                batch = None
            if session.finished:
                break
            if snapshots is not None and snapshots._due():
                session.request_snapshot()
            pending = None
            pending_save_count = output.n_saved
            pending_ticket = None
            terminal_rows = (None, None)
            batch = session.advance()
            for event in batch.events:
                if isinstance(event, Progress):
                    chunks.add(event.history, event.step_stats)
                elif isinstance(event, ObservationReady):
                    chunks.add(event.history, event.step_stats)
                    attempts = chunks.take()
                    if event.kind is ObservationKind.LAUNCH and not (
                        output.n_saved == 0 or missing_current_save
                    ):
                        # A resumed file already holds the plane at its position.
                        continue
                    current[0] = event
                    pending_ticket = event.ticket
                    # Its attempts, should the plane fail before it holds them.
                    terminal_rows = attempts
                    if event.kind is not ObservationKind.LAUNCH:
                        _warn_coarse(event, planes)
                    keep = context.retention.keeps(output.n_saved)
                    pending, stored = _prepare_plane(
                        event, output, exports, attempts, shape, keep=keep
                    )
                    pending_save_count = output.n_saved
                    with committing():
                        _save_plane(output, pending, stored)
                        if event.kind is not ObservationKind.LAUNCH:
                            keeper.keep(pending.z)
                            reached = event.z
                        _record_execution(
                            checkpoint_output,
                            backend,
                            context.tdse_binding,
                            _Status("running", reached),
                            clock=clock,
                            restart=_restart(base, keeper),
                        )
                    watch.show(pending, output.n_saved - 1, stored)
                    if event.kind is not ObservationKind.LAUNCH:
                        reporter.plane(event.z, event.summary)
                elif isinstance(event, ServiceRequired):
                    chunks.add(event.history, event.step_stats)
                    if event.kind is ServiceKind.WARNING:
                        # A warning barrier (SEP 47 section 5.5): the filters may
                        # raise it, which fails the run at this committed endpoint,
                        # its field and the attempts since the last plane there for
                        # the checkpoint.
                        pending_ticket = event.ticket
                        terminal_rows = chunks.joined()
                        emit_warning(event)
                    elif event.kind is ServiceKind.SNAPSHOT and snapshots is not None:
                        assert isinstance(output, HDF5Output)
                        _service_snapshot(
                            event,
                            output,
                            checkpoint_output,
                            backend,
                            keeper,
                            exports,
                            context,
                            chunks.joined(),
                            clock=clock,
                            adams=adams,
                            base=base,
                        )
                elif isinstance(event, Terminal):
                    chunks.add(event.history, event.step_stats)
                    if event.outcome is Outcome.COMPLETED:
                        with committing():
                            _record_execution(
                                checkpoint_output,
                                backend,
                                context.tdse_binding,
                                _Status("complete", reached),
                                clock=clock,
                                restart=_restart(base, keeper),
                            )
                    else:
                        assert event.error is not None
                        pending_ticket = event.ticket
                        terminal_rows = chunks.joined()
                        raise event.error
        reporter.finished(time.perf_counter() - started)
        if isinstance(reporter, SummarisingReporter):
            reporter.summary(resources(usage(clock)))
    except BaseException as error:
        failure = error
        failed_z = _persist_failure(
            session,
            checkpoint_output,
            context,
            keeper,
            exports,
            pending_ticket,
            terminal_rows,
            chunks,
            base,
            _FailedObservation(
                pending,
                output.n_saved > pending_save_count,
                error,
                reached,
                clock,
                output.n_saved > 0 and context.retention.keeps(output.n_saved - 1),
            ),
        )
        if failed_z is not None:
            reached = failed_z
        if isinstance(error, RunTerminated | StopRequested):
            logger.log(
                logging.INFO if isinstance(error, StopRequested) else logging.WARNING,
                "%s at z = %.6e m; %s",
                error,
                reached,
                "the run's state there is written, and resume=True continues it"
                if failed_z is not None
                else "no state was written there",
            )
        if isinstance(reporter, InterruptibleReporter):
            reporter.interrupted(reached, error)
        raise
    finally:
        try:
            output.close()
        except BaseException as error:
            if failure is None:
                raise
            failure.add_note(f"the optical output could not be closed: {error}")
        finally:
            if snapshots is not None and snapshots_started:
                snapshots._finish()


def _warn_coarse(event: ObservationReady, planes: np.ndarray) -> None:
    """Log a plane whose propagation did not meet its tolerance.

    Parameters
    ----------
    event : solprop.backends.events.ObservationReady
        The observation.
    planes : numpy.ndarray
        The requested save planes, m.
    """
    summary = event.summary
    if summary.met_tolerance:
        return
    index = min(event.plane_index, planes.size - 1)
    logger.warning(
        "a step was too coarse for the tolerance it is scaled by on the way to the "
        "save plane at z = %.6e m (plane %d of %d): the largest scaled error norm was "
        "%.3e against the requested 1.0",
        float(planes[index]),
        index + 1,
        planes.size,
        summary.max_norm,
    )


def _service_snapshot(  # noqa: PLR0913, PLR0917 - the snapshot transaction's context
    event: ServiceRequired,
    output: HDF5Output,
    checkpoint_output: CheckpointOutput | None,
    backend: object,
    keeper: _MarchKeeper,
    exports: _Exports,
    context: ServiceContext,
    attempts: tuple[BackendHistory, BackendRows],
    *,
    clock: RunClock,
    adams: bool,
    base: Mapping[str, object],
) -> None:
    """Persist a running checkpoint at an accepted endpoint and copy the snapshot.

    At a saved plane the plane is the checkpoint, as before; elsewhere the endpoint's
    field and the attempts and rows since the last plane are written as a running
    checkpoint without being consumed (SEP 47 section 5.5).

    Parameters
    ----------
    event : solprop.backends.events.ServiceRequired
        The snapshot request.
    output : solprop.output.hdf5.HDF5Output
        The live writer.
    checkpoint_output : solprop.output.schema.CheckpointOutput or None
        The writer as a checkpoint store.
    backend : object
        The driver, for the execution record.
    keeper : _MarchKeeper
        Keeps the method history beside the checkpoint.
    exports : _Exports
        The endpoint's field.
    context : ServiceContext
        The run's context.
    attempts : tuple
        The attempts and rows since the last plane, the snapshot's own chunk included:
        the running checkpoint's, which does not consume them.
    clock : solprop._host.RunClock
        When the run started.
    adams : bool
        Whether the run's method keeps an Adams history, which a snapshot requires.
    base : collections.abc.Mapping
        What the execution restored, for its restart record.
    """
    snapshots = context.snapshots
    assert snapshots is not None
    z = event.z
    at_plane = output.n_saved > 0 and float(output["z"][-1]) == z
    checkpoint = None

    def persist() -> None:
        """Keep the method history and record the execution beside the snapshot."""
        keeper.keep(z, required=adams)
        _record_execution(
            checkpoint_output,
            backend,  # pyright: ignore[reportArgumentType]
            context.tdse_binding,
            _Status("running", z),
            clock=clock,
            checkpoint=checkpoint,
            restart=_restart(base, keeper),
        )

    # A snapshot at the plane just saved needs no running checkpoint beside it.
    if not at_plane:
        assert event.ticket is not None
        history, rows = attempts
        phase = exports.phase(event.ticket)
        checkpoint = TerminalCheckpoint(
            plane=SavedPlane(
                z=z,
                field=_adopted(exports.field(event.ticket)),
                stats={},
                step_history=_history_of(history),
                step_stats=_rows(rows),
                noise_phase=None if phase is None else _adopted(phase),
            ),
            next_save_index=int(
                np.searchsorted(context.request.observation.save_planes, z, "right")
            ),
            landed=True,
        )

    snapshots._service(
        output,
        context.tdse_archive,
        next_save_index=int(
            np.searchsorted(context.request.observation.save_planes, z, "right")
        ),
        require_history=adams,
        persist=persist,
        provenance=context.metadata,
        z=z,
    )


def _persist_failure(  # noqa: PLR0913, PLR0917 - the failure's whole context
    session: RunSession,
    output: CheckpointOutput | None,
    context: ServiceContext,
    keeper: _MarchKeeper,
    exports: _Exports,
    ticket: ExportTicket | None,
    rows: tuple[object, object],
    chunks: _Chunks,
    base: Mapping[str, object],
    failure: _FailedObservation,
) -> float | None:
    """Checkpoint the accepted state without replacing the original exception.

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        The session, its last batch still pending.
    output : solprop.output.schema.CheckpointOutput or None
        Handler supporting terminal checkpoints, when available.
    context : ServiceContext
        The run's context.
    keeper : _MarchKeeper
        Keeps the method history beside the checkpoint.
    exports : _Exports
        Shares a field already exported for the pending plane.
    ticket : solprop.backends.events.ExportTicket or None
        The committed field to checkpoint, or ``None`` when the state is poisoned or
        the failure arose outside the session.
    rows : tuple
        The attempts and periodic rows since the last saved plane, assembled for the
        terminal event or the failed plane, or ``None`` for each.
    chunks : _Chunks
        The chunks delivered since the last saved plane, for a failure outside the
        session, whose own undelivered attempts are joined to them.
    base : collections.abc.Mapping
        What the execution restored, for its restart record.
    failure : solprop.driver.run._FailedObservation
        The primary failure and the plane in hand.

    Returns
    -------
    float or None
        The checkpointed position, m, when one was written.
    """
    if output is None:
        return None
    backend = _facts(session)
    in_flight = backend.in_flight
    if failure.pending is None and in_flight is not None:
        # A driver's step in flight, or a poisoned session's: the latest running
        # checkpoint, when one follows the last plane, is the state to resume from
        # (SEP 47 section 5.5).
        return _record_abandoned_step(
            backend,
            output,
            context.tdse_binding,
            failure,
            in_flight,
            restart=_restart(base, keeper),
        )
    if ticket is None:
        # The loop's own session is open, and a state left in flight is recorded
        # above: the committed state always has a ticket.
        ticket = _fallback_ticket(session)
        assert ticket is not None
    problem = context.problem
    reached = None
    try:
        pending = failure.pending
        field = _adopted(exports.field(ticket))
        phase = exports.phase(ticket)
        if pending is None or pending.z != ticket.z:
            history, step_rows = rows
            if history is None and step_rows is None:
                # A failure outside the session: the chunks since the last plane, a
                # running checkpoint's included, and what no event delivered, which
                # the run's reporter does not hear: it is ending, and a reporter that
                # raised here would lose the checkpoint.
                chunks.add(*session.undelivered(ticket), report=False)
                history, step_rows = chunks.joined()
            pending = SavedPlane(
                z=ticket.z,
                field=field,
                stats={},
                step_history=StepHistory.empty()
                if history is None
                else _history_of(history),  # pyright: ignore[reportArgumentType]
                step_stats=StepStats.empty() if step_rows is None else _rows(step_rows),  # pyright: ignore[reportArgumentType]
                noise_phase=None if phase is None else _adopted(phase),
            )
        elif pending.field is None:
            pending = dataclasses.replace(
                pending,
                field=field,
                noise_phase=None if phase is None else _adopted(phase),
            )
        history = pending.step_history
        ends = (history.z + history.h)[history.accepted]
        landed = (
            not context.request.observation.dense
            or pending.z == problem.initial.z
            or bool(np.any(np.isclose(ends, pending.z, rtol=1e-12, atol=0.0)))
        )
        if failure.already_saved:
            pending = dataclasses.replace(
                pending, step_history=StepHistory.empty(), step_stats=StepStats.empty()
            )
        reached = pending.z
        checkpoint = TerminalCheckpoint(
            plane=pending,
            next_save_index=int(
                np.searchsorted(
                    context.request.observation.save_planes, pending.z, side="right"
                )
            ),
            landed=landed,
        )
        with committing(deliver=False):
            # The history first, so that the record states what was kept beside the
            # checkpoint; a failure to keep it still leaves the checkpoint recorded.
            try:
                keeper.keep(reached)
            except Exception as unkept:
                failure.error.add_note(
                    f"the method history could not be kept beside the terminal "
                    f"checkpoint: {unkept}"
                )
                logger.exception("could not keep the method history")
            _record_execution(
                output,
                backend,
                context.tdse_binding,
                _Status(
                    "budget_exhausted"
                    if isinstance(failure.error, ResponseBudgetError)
                    else "failed",
                    reached,
                    f"{type(failure.error).__name__}: {failure.error}",
                ),
                clock=failure.clock,
                checkpoint=checkpoint,
                restart=_restart(base, keeper),
            )
    except Exception as checkpoint_error:
        failure.error.add_note(
            f"terminal checkpoint could not be persisted: {checkpoint_error}"
        )
        logger.exception("could not persist the terminal checkpoint")
    return reached


def _fallback_ticket(session: RunSession) -> ExportTicket | None:
    """Return a ticket for the committed state after a failure outside the session.

    A service failure -- an output error, a signal during a commit -- leaves the
    session's batch pending, so its state is still where the batch left it. When
    nothing is pending, the session issues a failed terminal batch whose ticket exports
    the committed state (``DriverSession.pending_ticket``).

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        The session.

    Returns
    -------
    solprop.backends.events.ExportTicket or None
        A ticket for the committed state, or ``None`` when the session is closed or
        its state was left in flight.
    """
    # A session that can end itself at its committed state for the checkpoint says
    # so (`NativeSession.failure_ticket`); the transitional session's pending ticket
    # already does.
    issue = getattr(session, "failure_ticket", None)
    if callable(issue):
        return cast("ExportTicket | None", issue())
    return session.pending_ticket()


def _facts(session: RunSession) -> Backend | SessionFacts:
    """Return what the service loop's records read as the session's backend.

    The transitional session's driver, or a session's own facts
    (:class:`~solprop.driver.run.SessionFacts`).

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        The session.

    Returns
    -------
    Any
        The driver, or the facts.
    """
    # The transitional session's own module imports this loop's helpers.

    if isinstance(session, DriverSession):
        return session.driver
    return SessionFacts(session)
