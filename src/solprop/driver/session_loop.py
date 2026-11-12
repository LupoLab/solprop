# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The session loop: Python's side of an execution session (SEP 47 section 5).

Not to be confused with :mod:`solprop.service`, the front end's service.

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
from typing import Literal, cast

import numpy as np

from solprop._host import RunClock, machine, usage
from solprop._interrupts import committing, poll
from solprop.backends.events import (
    EventBatch,
    ExportTicket,
    ObservationKind,
    ObservationReady,
    Outcome,
    Progress,
    ServiceKind,
    ServiceRequired,
    emit_warning,
    join_attempts,
)
from solprop.backends.export import ExportRecipe
from solprop.backends.host.session import HostSession
from solprop.backends.protocol import Backend, BackendHistory, HostStepping
from solprop.backends.protocol import StepStats as BackendRows
from solprop.backends.session import RunSession
from solprop.driver.plane_view import PlaneWatch
from solprop.driver.run import (
    RESTART_RECORD,
    SessionFacts,
    history_of,
    open_run,
    retention_at,
)
from solprop.driver.summary import resources, run_summary
from solprop.exceptions import (
    BackendError,
    OutputError,
    ResponseBudgetError,
    RunTerminated,
    StopRequested,
)
from solprop.fourier import GridKind, spectrum_to_internal
from solprop.log import (
    InterruptibleReporter,
    ProgressReporter,
    StopFlag,
    SummarisingReporter,
)
from solprop.monitor import BoundaryCheck
from solprop.output.hdf5 import HDF5Output
from solprop.output.schema import (
    RUN_SUMMARY_ANNOTATION,
    CheckpointOutput,
    FieldRetention,
    Output,
    PlaneStore,
    RecipeRecord,
    RetainingOutput,
    RunMetadata,
    RunOutcome,
    SavedPlane,
    StepHistory,
    StepStats,
    StoredField,
    TerminalCheckpoint,
    _adopted,
)
from solprop.output.snapshots import SnapshotController
from solprop.output.tdse_archive import TDSEArchiveWriter
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import transverse_of
from solprop.translate.run_request import RunRequest
from solprop.translate.tdse import TDSEBinding

__all__ = [
    "MARCH_SNAPSHOT_AUTO_BYTES",
    "MONITOR_RECONSTRUCTED",
    "ServiceContext",
    "serve",
]

logger = logging.getLogger("solprop.driver.run")

MARCH_SNAPSHOT_AUTO_BYTES = 256 * 2**20
"""The largest Adams march snapshot a run keeps when ``resume_history`` is ``None``.

The snapshot -- the state and the history's rows, a few to sixty copies of the state --
is rewritten at every save plane; a state too large for that to be cheap keeps none
unless asked to, and its resume starts the history afresh.
"""

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
        (``solprop.driver.resume._continue_march``), for its
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

    - ``rows``: the rows, with a leading row axis and no batch axis.
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

    - ``base``: what the execution restored, fixed when it started.
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

    - ``kind``: the grid's spectral layout.
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
        """Forget what another ticket exported."""
        if self._ticket != ticket:
            self._ticket = ticket
            self._field = self._stored = self._phase = None
            self._phased = False

    def field(self, ticket: ExportTicket) -> np.ndarray:
        """Return the consumer field for a ticket, exporting it once.

        - Returns: Read-only output amplitudes in the internal convention.
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
        """Export the stored form of a ticket's field into `out`.

        The plane store is a plane's first consumer (:func:`_prepare_plane`), so its
        export is the ticket's first; any consumer after it is given the consumer
        field from it (:meth:`field`).

        - ``out``: the plane store's room (``PlaneStore.field_destination``), or
          ``None`` for a new array.
        - Returns: Read-only output amplitudes in the optics convention: `out` when
          given.
        """
        self._hold(ticket)
        assert self._field is None and self._stored is None, "a plane's first export"
        stored = self._session.export(ticket, _STORED, out=out)
        stored.setflags(write=False)
        self._stored = stored
        return stored

    def release(self) -> None:
        """Drop the exports once their batch is acknowledged."""
        self._ticket = self._field = self._stored = self._phase = None
        self._phased = False

    def phase(self, ticket: ExportTicket) -> np.ndarray | None:
        """Return the noise phase for a ticket, read once.

        - Returns: the phase, read-only, or ``None`` without a noise model.
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

        - ``rows``: its periodic rows, without a batch axis.
        - ``report``: ``False`` for the attempts no event delivered, which a failure's
          checkpoint holds and no reporter call may lose.
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

        - Returns: the attempts and rows since the last saved plane.
        """
        return join_attempts(self._chunks)

    def take(self) -> tuple[BackendHistory, BackendRows]:
        """Return every chunk held, assembled, and start afresh.

        - Returns: the attempts and rows since the last saved plane.
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

    - ``attempts``: its history and rows, assembled from the chunks since the previous
      plane.
    - ``field``: its field in the internal convention, or ``None``: a field-free plane,
      or one whose field is exported straight into a plane store.
    - ``phase``: its noise phase, where it keeps its field and a noise model runs.
    """
    return SavedPlane(
        z=event.z,
        field=None if field is None else _adopted(field),
        stats=dict(event.stats),
        step_history=history_of(attempts[0]),
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

    - ``attempts``: its history and rows.
    - ``shape``: ``(n_modes, n_pol, n_omega)`` of the field.
    - Returns: the plane and, where its field was exported into a plane store, the
      stored field (``None`` otherwise).
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

    - ``stored``: its field exported into the plane store's stored form, or ``None``.
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

    Any :class:`~solprop.backends.session.RunSession` is served: the reference's
    :class:`~solprop.backends.host.session.HostSession`, whose driver gives the
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
    checkpoint_output = open_run(
        output, context.metadata, backend, context.tdse_binding
    )
    if context.recipe is not None and isinstance(output, PlaneStore):
        with committing():
            output.record_recipe(context.recipe)
    reserve = getattr(output, "reserve", None)
    if reserve is not None:
        reserve(planes.size + 1)
    _retain(output, context.retention, problem.shape)
    reached, missing_current_save = _resume_position(
        output, checkpoint_output, problem, context.request
    )
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
        request=context.request,
        boundary_check=None if monitor is None else monitor.check,
    )
    keeper = _MarchKeeper(
        output,
        backend,
        context.resume_history,
        method=_method_history(context.request),
        retention=None if earlier is None else retention_at(earlier.execution, reached),
    )
    exports = _Exports(session, context.metadata.grid.kind)
    shape = tuple(context.problem.shape)
    current: list[ObservationReady | None] = [None]

    def _fetch() -> np.ndarray:
        """Return the pending observation's field, for the plane observer."""
        event = current[0]
        assert event is not None  # the observer is shown a pending plane only
        return exports.field(event.ticket)

    watch = PlaneWatch(reporter, context.metadata, _fetch, clock.observer)
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
                    else:
                        # A snapshot, which a session yields only at the request above.
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
                else:
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

    - ``planes``: the requested save planes, m.
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


def _service_snapshot(  # noqa: PLR0917 - the snapshot transaction's context
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

    - ``attempts``: the attempts and rows since the last plane, the snapshot's own chunk
      included: the running checkpoint's, which does not consume them.
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
                step_history=history_of(history),
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


def _persist_failure(  # noqa: PLR0917 - the failure's whole context
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

    - ``output``: handler supporting terminal checkpoints, when available.
    - ``ticket``: the committed field to checkpoint, or ``None`` when the state is
      poisoned or the failure arose outside the session.
    - ``rows``: the attempts and periodic rows since the last saved plane, assembled for
      the terminal event or the failed plane, or ``None`` for each.
    - Returns: the checkpointed position, m, when one was written.
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
                else history_of(history),  # pyright: ignore[reportArgumentType]
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
    the committed state (``HostSession.pending_ticket``).

    - Returns: a ticket for the committed state, or ``None`` when the session is closed
      or its state was left in flight.
    """
    # A session that can end itself at its committed state for the checkpoint says
    # so (`NativeSession.failure_ticket`); the reference session's pending ticket
    # already does.
    issue = getattr(session, "failure_ticket", None)
    if callable(issue):
        return cast("ExportTicket | None", issue())
    return session.pending_ticket()


def _facts(session: RunSession) -> HostStepping | SessionFacts:
    """Return what the service loop's records read as the session's backend.

    The reference session's driver, or a session's own facts
    (:class:`~solprop.driver.run.SessionFacts`).
    """
    if isinstance(session, HostSession):
        return session.driver
    return SessionFacts(session)


def _announce(  # noqa: PLR0917 - called positionally
    problem: PropagationProblem,
    metadata: RunMetadata,
    backend: Backend | SessionFacts,
    output: Output,
    reporter: ProgressReporter,
    start: tuple[int, float],
    *,
    request: RunRequest,
    boundary_check: BoundaryCheck | None = None,
) -> None:
    """Show the run's summary through the reporter and record it on a plane store.

    - ``reporter``: where progress goes; only a :class:`solprop.log.SummarisingReporter`
      is shown the summary.
    - ``start``: the first save plane beyond the loaded state, and the position the run
      starts from, m.
    - ``boundary_check``: the thresholds the window's edges are watched with, or
      ``None``.
    """
    start_plane, z_start = start
    summary = run_summary(
        problem,
        request,
        metadata,
        backend=backend.name,
        execution=backend.metadata(),
        output=output,
        start_plane=start_plane,
        z_start=z_start,
        boundary_check=boundary_check,
    )
    if isinstance(reporter, SummarisingReporter):
        reporter.summary(summary.text())
    if isinstance(output, PlaneStore):
        # An annotation stays outside the record and its hash, so a summary needs no
        # schema of its own; a resume adds its continuation's to the list.
        earlier = output.annotations.get(RUN_SUMMARY_ANNOTATION, [])
        output.annotate(
            RUN_SUMMARY_ANNOTATION,
            [*cast("list[object]", earlier), summary.record()],
        )


def _retain(
    output: Output, retention: FieldRetention, field_shape: tuple[int, ...]
) -> None:
    """Tell an output that can hold it which planes keep their field.

    Called once the output is open, so that a file being continued compares the rule
    with the one it holds.

    - ``field_shape``: ``(n_modes, n_pol, n_omega)`` of the run's state.
    """
    if isinstance(output, RetainingOutput):
        output.retain(retention, field_shape)


def _resume_position(
    output: Output,
    checkpoint_output: CheckpointOutput | None,
    problem: PropagationProblem,
    request: RunRequest,
) -> tuple[float, bool]:
    """Resolve the actual checkpoint position and any scheduled observation owed.

    - ``checkpoint_output``: Checkpoint-capable handler, if available, holding any
      terminal state.
    - ``problem``: translated problem with the absolute launch position, m.
    - ``request``: the run's request, with the scheduled save positions, m.
    - Returns: actual restart position in m and whether its scheduled observation still
      needs saving because failure occurred before the regular save completed.
    """
    reached = float(output["z"][-1]) if output.n_saved else problem.initial.z
    if (
        checkpoint_output is not None
        and checkpoint_output.terminal_checkpoint is not None
    ):
        reached = checkpoint_output.terminal_checkpoint.plane.z
    missing_current_save = (
        checkpoint_output is not None
        and checkpoint_output.terminal_checkpoint is not None
        and bool(np.any(request.observation.save_planes == reached))
        and (output.n_saved == 0 or float(output["z"][-1]) < reached)
    )
    return reached, missing_current_save


def _refuse_unread_collar(
    problem: PropagationProblem, backend: Backend | SessionFacts
) -> None:
    """Refuse a requested collar on a backend that does not read its return (SEP 40).

    A requested real-space collar is judged by what it returns, which the monitor's
    transverse edge reads; a backend that does not read it refuses the collar by name
    before the first step (SEP 40 section 6). Its in-stage rate is not the reason:
    the backends compute it alike.
    """
    design = transverse_of(problem.route)
    if design is None or not design.has_collar:
        return
    if backend.reads_transverse_edge:
        return
    msg = (
        f"the {backend.name} backend does not carry a requested real-space collar "
        "(SpatialAbsorber with real_space=True): run it on one of solprop's own "
        "backends, or ask for the safety ramp alone"
    )
    raise BackendError(msg)


def _earlier_execution(
    output: Output, checkpoint_output: CheckpointOutput | None
) -> tuple[bool, Mapping[str, object] | None, tuple[tuple[float, float], ...]]:
    """Return whether this execution resumes another, and the monitor record to go on.

    - ``checkpoint_output``: Checkpoint-capable handler, if available, holding the
      earlier outcome.
    - Returns ``record``: the latest earlier execution's boundary-monitor record,
      searching back past executions that kept none, or ``None`` when none did.
    - Returns ``unwatched``: the stretches, ``(start, end)`` in m, that executions after
      that record ran without a monitor; from the launch when no execution kept one.
    """
    outcome = None if checkpoint_output is None else checkpoint_output.outcome
    resumed = output.n_saved > 0 or outcome is not None
    if outcome is None:
        return resumed, None, ()
    # Earliest first: each earlier execution's closing record, then the latest.
    chain: list[tuple[float, Mapping[str, object]]] = []
    for previous in outcome.previous_executions:
        execution = previous.get("execution")
        reached = previous.get("z")
        chain.append(
            (
                float(reached) if isinstance(reached, int | float) else 0.0,
                cast("Mapping[str, object]", execution)
                if isinstance(execution, Mapping)
                else {},
            )
        )
    chain.append((float(outcome.z), outcome.execution))
    # The latest record that holds a monitor; executions after it ran unwatched,
    # each from where the one before it stopped to where it stopped itself, which is
    # where the next resumed. A hard kill before the monitor is attached
    # leaves a record without one and an empty stretch, and is passed over.
    found = -1
    for index in range(len(chain) - 1, -1, -1):
        if isinstance(chain[index][1].get("boundary_monitor"), Mapping):
            found = index
            break
    unwatched: list[tuple[float, float]] = []
    start = 0.0 if found < 0 else chain[found][0]
    for reached, _ in chain[found + 1 :]:
        if reached > start:
            unwatched.append((start, reached))
        start = max(start, reached)
    if found < 0:
        return resumed, None, tuple(unwatched)
    record = cast("Mapping[str, object]", chain[found][1]["boundary_monitor"])
    return resumed, record, tuple(unwatched)


@dataclasses.dataclass(frozen=True)
class _FailedObservation:
    """Retain drained history and writer progress when a scheduled save fails."""

    pending: SavedPlane | None
    already_saved: bool
    error: BaseException
    reached: float
    clock: RunClock
    last_kept_field: bool


def _method_history(
    request: RunRequest,
) -> Literal["adams", "rk", "low_storage"]:
    """Return which kind of method history a run's march keeps.

    An Adams history a resume can continue; Runge-Kutta controller memory (the proposal
    and the norm memory), which a resume resets; or none at all.
    """
    integrator = request.integration.method
    if isinstance(integrator, Multistep):
        return "adams"
    if isinstance(integrator, LowStorageRungeKutta):
        return "low_storage"
    return "rk"


class _MarchKeeper:
    """Keep the Adams march's state beside each save, and record what was kept.

    Every persisted plane and checkpoint states whether its method history was
    retained, omitted or is not applicable, and why (SEP 47 section 5.5); the run's
    execution record carries the latest decision (:meth:`record`).

    - ``output``: the run's handler; only one with ``save_march`` keeps a snapshot.
    - ``retention``: the decision already recorded at the position the run starts
      from, carried into this execution's record until it persists a position of its
      own.
    """

    def __init__(
        self,
        output: Output,
        backend: HostStepping | SessionFacts,
        policy: bool | None,
        *,
        method: Literal["adams", "rk", "low_storage"] = "adams",
        retention: Mapping[str, object] | None = None,
    ) -> None:
        """Hold what keeping a snapshot needs."""
        self._save = getattr(output, "save_march", None)
        self._snapshot = backend.march_snapshot
        self._snapshot_nbytes = backend.march_snapshot_nbytes
        self._refusal = backend.march_export_refusal
        self._policy = policy
        self._method = method
        # Measured once against the automatic limit: kept from then on, or not.
        self._over_limit = False
        self._within_limit = False
        self._kept_z: float | None = None
        self._decision: dict[str, object] | None = (
            None if retention is None else dict(retention)
        )

    def record(self) -> dict[str, object] | None:
        """Return the latest retention decision, for the execution record.

        - Returns: ``"state"`` (``"retained"``, ``"omitted"`` or ``"not_applicable"``),
          ``"reason"`` and ``"z"``, m, with ``"nbytes"`` where the size was measured;
          ``None`` before anything was persisted.
        """
        return None if self._decision is None else dict(self._decision)

    def _decide(self, z: float, state: str, reason: str, **sizes: int) -> None:
        """Record a retention decision at `z`.

        - ``z``: the persisted position, m.
        - ``state``: ``"retained"``, ``"omitted"`` or ``"not_applicable"``.
        - ``**sizes``: ``nbytes``, the history's size where it was measured, and
          ``limit``, the automatic limit it was held to, bytes.
        """
        self._decision = {"state": state, "reason": reason, "z": z, **sizes}

    def keep(self, z: float, *, required: bool = False) -> None:  # noqa: PLR0911, PLR0912 - one return per decision
        """Keep the march's state if it stands at `z`, the position just saved.

        - ``z``: the saved plane's or the checkpoint's position, m.
        - ``required``: an immutable snapshot needs the full Adams state, so the
          automatic retention-size cutoff is ignored and a history that cannot be
          persisted here is refused.
        """
        if required and self._kept_z == z:
            return
        if self._method == "low_storage":
            self._decide(z, "not_applicable", "no_method_history")
            return
        if self._method == "rk":
            # The proposal and the norm memory are not persisted (issue #596).
            self._decide(z, "omitted", "unsupported_method")
            return
        if self._save is None:
            self._decide(z, "omitted", "output_cannot_store")
            if required:
                raise OutputError(
                    "snapshot requires a backend and output with Adams history"
                )
            return
        if self._policy is False and not required:
            self._decide(z, "omitted", "explicit_policy")
            return
        if self._over_limit and not required:
            self._decide(
                z, "omitted", "automatic_size_limit", limit=MARCH_SNAPSHOT_AUTO_BYTES
            )
            return
        if self._policy is None and not required and not self._within_limit:
            # Before any copy: the snapshot over the limit is the state and the whole
            # history, on the host.
            nbytes = self._snapshot_nbytes(z)
            if nbytes is None:
                self._omit_unavailable(z)
                return
            if nbytes > MARCH_SNAPSHOT_AUTO_BYTES:
                logger.info(
                    "the Adams march's state is not kept for a resume: %d bytes is "
                    "over the %d kept by default (resume_history=True keeps it)",
                    nbytes,
                    MARCH_SNAPSHOT_AUTO_BYTES,
                )
                self._over_limit = True
                self._decide(
                    z,
                    "omitted",
                    "automatic_size_limit",
                    nbytes=nbytes,
                    limit=MARCH_SNAPSHOT_AUTO_BYTES,
                )
                return
            self._within_limit = True
        snapshot = self._snapshot()
        if snapshot is None or snapshot.z != z:
            self._omit_unavailable(z)
            if required:
                raise OutputError(
                    "snapshot requires Adams history at the accepted endpoint"
                )
            return
        try:
            self._save(snapshot)
        except BaseException:
            # A torn write reads as no snapshot (`_write_march`): expected, and lost.
            self._decide(z, "omitted", "write_failed")
            raise
        self._kept_z = z
        self._decide(
            z,
            "retained",
            "snapshot_required"
            if required
            else "explicit_policy"
            if self._policy
            else "automatic_policy",
            nbytes=snapshot.nbytes,
        )

    def _omit_unavailable(self, z: float) -> None:
        """Record history that could not be had at `z`, naming an engine's refusal.

        - ``z``: the persisted position, m.
        """
        refused = self._refusal() is not None
        self._decide(
            z,
            "omitted",
            "incompatible_representation" if refused else "unavailable_history",
        )


def _record_abandoned_step(
    backend: Backend | SessionFacts,
    output: CheckpointOutput,
    binding: TDSEBinding | None,
    failure: _FailedObservation,
    step: tuple[float, float],
    *,
    restart: Mapping[str, object] | None = None,
) -> float | None:
    """Record a failure that left a low-storage step in flight, with no new checkpoint.

    The low-storage march accepts each step in place (SEP 24 section 6.2), so a step
    abandoned between its start and its commit leaves a state that is neither of the
    step's ends, which the backend refuses to observe, so nothing honest can be written
    as a terminal checkpoint. The file falls back to the latest state it holds beyond
    its last saved plane, a running checkpoint (SEP 47 section 5.5) or the checkpoint
    this execution resumed from, which is kept; without one it ends at its last saved
    plane: a resume continues from one that kept its field, to rounding, with
    ``step_planes`` bounding what is lost, and refuses one that did not.

    - ``step``: where the abandoned step started and its length, m.
    - Returns: the position of the checkpoint fallen back to, m, or ``None`` without
      one.
    """
    z, dz = step
    fallback = output.terminal_checkpoint
    if fallback is not None:
        ending = (
            f"the file falls back to its latest checkpoint, z = "
            f"{fallback.plane.z:.6g} m, "
            + (
                "from which resume=True continues"
                if fallback.landed
                else "an interpolated field, from which only "
                "resume='interpolated' continues"
            )
        )
    else:
        kept = (
            "from which resume=True continues"
            if failure.last_kept_field
            else "which kept no field, so this file cannot be resumed; rerun it, "
            "keeping a plane's field"
        )
        ending = (
            f"the file ends at its last saved plane, z = {failure.reached:.6g} m, "
            f"{kept}"
        )
    reason = (
        f"no terminal checkpoint is written: the low-storage step of {dz:.6g} m from "
        f"z = {z:.6g} m was abandoned in flight, so the state holds neither end of "
        f"it; {ending}"
    )
    failure.error.add_note(reason)
    logger.warning("%s", reason)
    try:
        with committing(deliver=False):
            _record_execution(
                output,
                backend,
                binding,
                _Status(
                    "failed",
                    failure.reached if fallback is None else fallback.plane.z,
                    f"{type(failure.error).__name__}: {failure.error}; {reason}",
                ),
                clock=failure.clock,
                checkpoint=fallback,
                restart=restart,
            )
    except Exception as record_error:
        failure.error.add_note(f"the failure could not be recorded: {record_error}")
        logger.exception("could not record the failure")
        return None
    return None if fallback is None else fallback.plane.z


@dataclasses.dataclass(frozen=True)
class _Status:
    """What an execution record says of the run, before its accounting is added.

    Attributes
    ----------
    status : {'running', 'complete', 'budget_exhausted', 'failed'}
        The run's status (:class:`~solprop.output.schema.RunOutcome`).
    z : float
        Last accepted or saved position, m.
    message : str
        Explanation of a stop or failure; empty on ordinary progress.
    """

    status: Literal["running", "complete", "budget_exhausted", "failed"]
    z: float
    message: str = ""


def _record_execution(
    output: CheckpointOutput | None,
    backend: Backend | SessionFacts,
    binding: TDSEBinding | None,
    status: _Status,
    *,
    clock: RunClock,
    checkpoint: TerminalCheckpoint | None = None,
    restart: Mapping[str, object] | None = None,
) -> None:
    """Snapshot call accounting without requesting any diagnostic evaluation.

    The record also names the machine (:func:`solprop._host.machine`) and what the run
    has spent so far (:func:`solprop._host.usage`): written at every save plane, it
    keeps the wall time and peak memory of a run that is killed between them.

    - ``status``: requested status, absolute position in m and any error message.
    - ``restart``: the execution's restart record (:data:`RESTART_RECORD`); ``None`` or
      an empty one records none.
    """
    if output is None:
        return
    record = {**backend.metadata(), "host": machine(), "resources": usage(clock)}
    if restart:
        record[RESTART_RECORD] = dict(restart)
    monitor = record.get("boundary_monitor")
    if status.status == "running" and isinstance(monitor, Mapping):
        # The monitor's per-plane rows go out with the closing record alone: rewritten
        # at every plane, the record grew with them and the file with their square.
        record["boundary_monitor"] = {
            name: value for name, value in monitor.items() if name != "planes"
        }
    if binding is not None:
        record = {**record, "tdse": dict(binding.metadata())}
    calls = record.get("response_calls", {})
    logical = cast("Mapping[str, object]", calls) if isinstance(calls, Mapping) else {}
    atomic = (
        record.get("atomic_calls", {})
        if binding is None
        else dataclasses.asdict(binding.ledger)
    )
    # Built once, so that a saved plane validates its record once (RunOutcome).
    output.record_outcome(
        RunOutcome(
            status.status,
            status.z,
            status.message,
            logical_calls=logical,
            atomic_calls=cast("Mapping[str, object]", atomic)
            if isinstance(atomic, Mapping)
            else {},
            execution=record,
        ),
        checkpoint,
    )
