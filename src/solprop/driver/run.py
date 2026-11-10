# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The outer loop of a run: save planes in, saved planes out.

The inner loop -- many adaptive or replayed steps between two save planes -- lives
inside a backend and is compiled; this is the Python loop around it, which runs a few
hundred times at most (``docs/dev/architecture.md`` section 5). It asks the backend to
land on each save plane in turn, takes the observation and the statistics there, and
hands them to an output handler.

**What this module does not do.** It never conjugates a spectrum and never touches the
state between planes: the backend owns the step loop and the output handler owns the
sign convention on disk (``docs/dev/conventions.md`` section 2, boundary place 2). A
plane is carried across as a :class:`solprop.output.schema.SavedPlane` in the internal
FFT convention, with modal amplitudes converted to true-power units.

Progress is reported through :class:`solprop.log.ProgressReporter` -- the run's summary
(:mod:`solprop.driver.summary`) before the first step, one call at the start, one per
attempted step from inside the backend, one per save plane and one at the end. The
protocol and the default :class:`solprop.log.LogReporter` live in :mod:`solprop.log`,
and are re-exported here because this is where a run reaches them. A bar shows the
summary and the closing line on its stream; the line reporter logs them, and
:func:`solprop.log.configure_logging` is how a script sees those records
(``docs/dev/architecture.md`` section 14).

The manual page is ``docs/guide/setup/running.md``.
"""

import dataclasses
import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import Literal, cast

import numpy as np

from solprop._host import RunClock, machine, usage
from solprop._interrupts import committing, handled
from solprop.backends.populations import axis_refusal
from solprop.backends.protocol import Backend, BackendHistory, Observation
from solprop.backends.registry import statistics_refusal
from solprop.backends.session import RunSession, Start
from solprop.driver.summary import run_summary
from solprop.exceptions import (
    BackendError,
    DriverError,
    OutputError,
    SolpropWarning,
)
from solprop.log import (
    DEFAULT_PROGRESS_PERIOD,
    LogReporter,
    ProgressChoice,
    ProgressReporter,
    StopFlag,
    SummarisingReporter,
    resolve_reporter,
)
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.output.capability import (
    capability_of,
    declared_capability,
    refusal_for,
    withheld_accessors,
)
from solprop.output.hdf5 import HDF5Output
from solprop.output.schema import (
    RUN_SUMMARY_ANNOTATION,
    BackendRecord,
    CheckpointOutput,
    FieldRetention,
    FieldsChoice,
    MarchSnapshot,
    Output,
    PlaneStore,
    RecipeRecord,
    RetainingOutput,
    RunMetadata,
    RunOutcome,
    SavedPlane,
    StepHistory,
    StepStats,
    TerminalCheckpoint,
    field_retention_refusal,
)
from solprop.output.snapshots import SnapshotController
from solprop.output.tdse_archive import TDSEArchiveWriter
from solprop.stats import StatsSpec
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import tdse_descriptor, transverse_of
from solprop.translate.run_request import HistoryPolicy, RunRequest
from solprop.translate.tdse import TDSEBinding

__all__ = [
    "MARCH_SNAPSHOT_AUTO_BYTES",
    "RESTART_RECORD",
    "LogReporter",
    "ProgressReporter",
    "SessionFacts",
    "check_snapshot_request",
    "field_retention",
    "open_run_record",
    "plane_from_observation",
    "retention_at",
    "run_problem",
]

logger = logging.getLogger(__name__)

MARCH_SNAPSHOT_AUTO_BYTES = 256 * 2**20
"""The largest Adams march snapshot a run keeps when ``resume_history`` is ``None``.

The snapshot -- the state and the history's rows, a few to sixty copies of the state --
is rewritten at every save plane; a state too large for that to be cheap keeps none
unless asked to, and its resume starts the history afresh.
"""


def _history_of(history: BackendHistory) -> StepHistory:
    """Return a backend's step history in the form an output handler stores.

    The two records carry the same five columns, and two of them mean different things
    on the two sides of the seam: the backend's ``z`` is the position at the *end* of
    each attempt and its ``wall_time`` is the clock reading since the previous
    observation, which is what a stepper measures cheaply, while
    :class:`solprop.output.schema.StepHistory` stores the position at the *start* of
    each attempt and the time that one attempt took, which is what a reader wants per
    row. The conversion is made here: the start is the end less the length, to
    round-off, and the per-attempt times are the differences of the clock, the first
    from zero.

    Parameters
    ----------
    history : solprop.backends.protocol.BackendHistory
        What the stepper did since the previous observation.

    Returns
    -------
    solprop.output.schema.StepHistory
        The same attempts, in the columns the output layer stores.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.protocol import BackendHistory
    >>> from solprop.driver.run import _history_of
    >>> history = BackendHistory(np.array([0.5, 1.0]), np.array([0.5, 0.5]),
    ...                           np.array([0.4, 0.5]), np.array([True, True]),
    ...                           np.array([0.25, 0.75]))
    >>> stored = _history_of(history)
    >>> stored.z.tolist(), stored.wall_time.tolist()
    ([0.0, 0.5], [0.25, 0.5])
    """
    return StepHistory(
        z=history.z - history.dz,
        h=history.dz,
        norm=history.error_norm,
        accepted=history.accepted,
        wall_time=np.diff(history.wall_time, prepend=0.0),
    )


def plane_from_observation(
    observation: Observation,
    stats: Mapping[str, np.ndarray] | None = None,
    *,
    field: bool = True,
) -> SavedPlane:
    """Turn one observation and its statistics into the plane an output handler stores.

    The seam's arrays are one trajectory's, as a stored plane is, so no shape changes
    on the way (``docs/dev/conventions.md`` section 6). Nothing is conjugated --
    the field stays in the internal FFT convention, and the handler converts it on the
    way to storage (``docs/dev/conventions.md`` section 2). Nothing is rescaled either:
    the field is whatever amplitude the observation carries, the backend's
    vacuum-normalised one unless the caller has applied the interface conversion to
    true-power amplitudes (a session's export with
    ``ExportRecipe(power=True, ...)``; see ``docs/dev/decisions.md``).

    Parameters
    ----------
    observation : solprop.backends.protocol.Observation
        The state at a save plane, from
        :meth:`solprop.backends.protocol.Backend.observe`.
    stats : collections.abc.Mapping, optional
        What :meth:`solprop.backends.protocol.Backend.stats` returned there, keyed by
        statistic name. Empty when omitted.
    field : bool, optional
        Whether the plane keeps the field and its noise phase; ``False``, or an
        observation that carries no field, builds a field-free plane (SEP 22), which
        copies neither.

    Returns
    -------
    solprop.output.schema.SavedPlane
        The position, the field, the statistics and the step history.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.protocol import BackendHistory, Observation
    >>> from solprop.driver.run import plane_from_observation
    >>> history = BackendHistory(np.array([0.5]), np.array([0.5]), np.array([0.3]),
    ...                       np.array([True]), np.array([0.02]))
    >>> observation = Observation(0.5, np.ones((1, 1, 4), complex), None, history, 1, 0)
    >>> plane = plane_from_observation(observation, {"energy": np.ones((1, 1))})
    >>> plane.z, plane.shape, plane.stats["energy"].shape
    (0.5, (1, 1, 4), (1, 1))
    """
    values = {} if stats is None else stats
    rows = observation.step_stats
    return SavedPlane(
        z=observation.z,
        field=observation.field if field else None,
        stats={name: np.asarray(value) for name, value in values.items()},
        step_history=_history_of(observation.step_history),
        step_stats=StepStats(z=rows.z, step=rows.step, values=dict(rows.values)),
        noise_phase=observation.noise_phase if field else None,
    )


class SessionFacts:
    """A session's execution facts, as the service loop's records read a backend's.

    The transitional output and record helpers (:mod:`solprop.driver.run`) read a
    backend: its name, metadata, method history and in-flight state. A session that
    owns its backend answers them through its protocol: its execution record, its
    method-history size, and the in-place step a failure left in flight, which a
    session that steps in place reports as ``in_flight`` (the native session's
    low-storage replay, SEP 47 section 5.5); any other leaves none.

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        The session.
    """

    def __init__(self, session: RunSession) -> None:
        """Read the session's facts through its protocol."""
        self._session = session
        record = session.execution_record()
        self.name = str(record.get("backend", type(session).__name__))
        self.reads_transverse_edge = bool(
            getattr(session, "reads_transverse_edge", False)
        )

    @property
    def in_flight(self) -> tuple[float, float] | None:
        """Return the in-place step a failure left in flight, or ``None``.

        Returns
        -------
        tuple or None
            Where the step started and its length, m, as the session reports it.
        """
        return getattr(self._session, "in_flight", None)

    def metadata(self) -> dict[str, object]:
        """Return the session's execution record.

        Returns
        -------
        dict
            The record, with the monitor's under ``boundary_monitor``.
        """
        return dict(self._session.execution_record())

    def march_snapshot(self) -> MarchSnapshot | None:
        """Return the session's method history at its pending committed state.

        Returns
        -------
        solprop.output.schema.MarchSnapshot or None
            The session's snapshot
            (:meth:`~solprop.backends.session.RunSession.march_snapshot`), or
            ``None`` without one or with no batch pending, for which the session's
            :meth:`~solprop.backends.session.RunSession.pending_ticket` gives none.
        """
        session = self._session
        ticket = session.pending_ticket()
        return None if ticket is None else session.march_snapshot(ticket)

    def march_snapshot_nbytes(self, z: float) -> int | None:
        """Return the size of the session's method-history snapshot, or ``None``.

        Parameters
        ----------
        z : float
            The position, m.

        Returns
        -------
        int or None
            Bytes, or ``None`` without one.
        """
        del z
        return self._session.march_snapshot_nbytes()

    def march_export_refusal(self) -> str | None:
        """Return why the session's method history is not exported here, or ``None``.

        Returns
        -------
        str or None
            The session's own reason
            (:meth:`~solprop.backends.session.RunSession.march_export_refusal`).
        """
        return self._session.march_export_refusal()


@handled()  # SIGTERM and Ctrl-C stop a run between its commits
def run_problem(  # noqa: PLR0913 - the loop's inputs and commit sequence
    problem: PropagationProblem,
    backend: Backend,
    output: Output,
    *,
    start: Start,
    metadata: RunMetadata,
    stats_spec: StatsSpec | None = None,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
    tdse_binding: TDSEBinding | None = None,
    resume_history: bool | None = None,
    fields: FieldsChoice = "all",
    field_planes: Sequence[int] = (),
    snapshots: SnapshotController | None = None,
    tdse_archive: TDSEArchiveWriter | None = None,
    boundary_monitor: BoundaryMonitor | None = None,
    stop: StopFlag | None = None,
    recipe: RecipeRecord | None = None,
    restoration: Mapping[str, object] | None = None,
) -> None:
    """Run one translated problem to its save planes, writing each one out.

    The loop of ``docs/dev/architecture.md`` section 5, usable without a
    :class:`solprop.driver.simulation.Simulation`. The run starts a session on the
    constructed backend from `start` (SEP 47); the handler is opened here, saved to
    plane by plane, and closed in a ``finally`` so that an interrupted run leaves a
    readable file. The backend stays open for the caller, who may start another run
    on it.

    The plane at ``problem.initial.z`` is saved first, with an empty step history, and
    then each save plane in turn: propagate, observe, collect statistics, save. Steps
    land on the planes exactly, so a saved field is an exact step endpoint and there
    is no interpolation (``docs/dev/stepper.md`` section 3), unless the request asks
    for dense output. With ``stepper.step_planes`` the march also stops at the end of
    every ``step_planes``-th accepted step, which is saved too before the same plane
    is asked for again. A plane whose propagation did not meet its tolerance is logged
    as a warning naming that plane (``docs/dev/stepper.md`` section 8).

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    backend : solprop.backends.protocol.Backend
        The backend stepping it, constructed for `problem` and its run's request
        (:attr:`~solprop.backends.protocol.Backend.request`), whose save planes are
        the itinerary; the run's own request is that one with this run's start,
        retention, statistics, monitor and history
        (:meth:`~solprop.translate.run_request.RunRequest.for_run`). The run loads
        `start` into it and leaves it open.
    output : solprop.output.schema.Output
        Where the planes go. Closed when the loop ends, whether or not it succeeded.
    start : solprop.backends.session.Start
        Where the run starts: :meth:`~solprop.backends.session.Start.entrance` for a
        fresh run, or a resumed state with its ``plane_index``, the first save plane
        strictly beyond it. A resume loads either the last regular plane or a separate
        terminal checkpoint at its actual absolute position; the launch plane is
        written when the handler holds no regular observations, so a resumed run never
        stores it twice. The session takes the start's field.
    metadata : solprop.output.schema.RunMetadata
        The record of the run, written when the handler opens.
        :func:`solprop.driver.simulation.metadata_for` builds one.
    stats_spec : solprop.stats.StatsSpec, optional
        Which statistics to collect at each plane. Defaults to the backend's request's
        own (section 8 of ``docs/dev/kernel-contract.md``).
    progress : ProgressReporter, {'auto', 'bar', 'log', 'none'} or None, optional
        Where progress goes. ``None`` takes the process default
        (:func:`solprop.log.default_progress`, ``"auto"`` unless
        :func:`solprop.log.configure_logging` was told otherwise): a
        :class:`solprop.log.BarReporter` drawing one redrawing line on standard error
        when that is a terminal or the process is a Jupyter kernel, and a
        :class:`solprop.log.LogReporter` -- a periodic line, one line per plane and one
        at the end, all at ``INFO`` -- otherwise, so redirected, piped and captured
        output gets lines and never a bar. The four names force a choice; a reporter
        is used as given.
    progress_period : float, optional
        Seconds between the line reporter's periodic lines, s; ``0`` leaves only the
        plane and closing lines. A bar has its own redraw period and a given reporter
        its own business, so it is ignored for both.
    tdse_binding : solprop.translate.tdse.TDSEBinding or None, optional
        Explicit live atomic runtime owned by the caller. Its ledgers and execution
        facts are captured before the caller closes it. TDSE requires an output
        supporting the separate checkpoint capability.
    resume_history : bool or None, optional
        Whether an output that can keep it (``save_march``) keeps the Adams march's
        state at every save plane and terminal checkpoint, so that a resume continues
        the history instead of starting it afresh. ``None`` keeps it while it is at
        most :data:`MARCH_SNAPSHOT_AUTO_BYTES`.
    fields : {'all', 'planes', 'none'}, optional
        Which saved planes keep their field (SEP 22): every one (the default), those
        of `field_planes`, or none. A plane that keeps none stores its position, its
        statistics and its step history, and is not copied, converted or stored;
        the march is the same either way. A terminal checkpoint, written when the
        run stops, carries the field whatever this says, so a stopped run resumes.
    field_planes : sequence of int, optional
        Saved-plane indices for ``fields='planes'``, the entrance as 0 and negative
        from the end.
    snapshots : solprop.output.SnapshotController or None, optional
        Requests immutable disk copies at committed accepted steps, each serviced at
        the next accepted endpoint, after every interpolated plane its step crossed.
        Requires HDF5 and all fields.
    tdse_archive : solprop.output.tdse_archive.TDSEArchiveWriter or None, optional
        Live archive to suspend and copy beside the optical writer.
    boundary_monitor : solprop.monitor.BoundaryMonitor or None, optional
        Watches the time window's edges (SEP 21): attached to the backend once its
        launch is recorded and fed every committed step. Its record is written into
        the execution record at every save plane without the per-plane rows, which
        the closing record (complete, failed or out of budget) carries, and its removed
        fraction is reported at the end. ``None`` watches nothing.
    stop : solprop.log.StopFlag or None, optional
        The run's own stop flag, any object with an ``is_set()`` method (SEP 41,
        S17). It is polled before the first step, before every trial step (on a native
        ``Multistep()`` march, before every native call) and between save planes; once
        it is set the run writes its state as for SIGTERM, tells the reporter through
        ``interrupted``, and raises
        :class:`~solprop.exceptions.StopRequested`, which
        :meth:`~solprop.driver.simulation.Simulation.run` turns into a return.
    recipe : solprop.output.schema.RecipeRecord or None, optional
        The recipe the run was built from (SEP 41, S2), recorded in a plane store at
        the first run start and never rewritten; ``None`` records none.
    restoration : collections.abc.Mapping or None, optional
        On a resume, what the caller restored of the method history, written into
        every execution record under ``outcome.execution["restart"]`` beside what this
        execution keeps; ``None`` records no restoration. When the start carries a
        method history (``start.march``), this is the record of its restoration, and
        a history the session cannot install is warned about and recorded as reset
        instead.

    Raises
    ------
    DriverError
        If the statistics given here disagree with the request's own about the
        between-planes channel; if the record and the problem disagree about
        whether this is a general modal run; if the start's ``plane_index`` is not a
        plane of this problem; or if the field retention is not valid, names a plane
        outside the run, asks for ``fields='planes'`` beside ``step_planes`` (whose
        plane count is not known in advance), or is not ``'all'`` for an output that
        cannot store field-free planes.

    Examples
    --------
    >>> from solprop.backends.registry import select
    >>> from solprop.backends.session import Start
    >>> from solprop.driver.run import run_problem
    >>> from solprop.driver.simulation import metadata_for
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.output import MemoryOutput
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> grid = EnvelopeGrid.from_limits(0.01, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8).build(grid)
    >>> problem, request = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-6, atol=1e-16),
    ...                    save_planes=[0.005, 0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> backend = select("numpy_ref", problem, request, grid=grid, modes=[gnlse.mode])
    >>> handler = MemoryOutput()
    >>> run_problem(problem, backend, handler, start=Start.entrance(problem),
    ...             metadata=metadata_for(problem, request, grid, modes=[gnlse.mode]))
    >>> backend.close()
    >>> handler.n_saved, handler["z"].tolist()
    (3, [0.0, 0.005, 0.01])
    """
    prepared = backend.request
    spec = prepared.observation.stats if stats_spec is None else stats_spec
    planes = np.asarray(prepared.observation.save_planes, dtype=np.float64)
    start_plane = start.plane_index
    _refuse_unrunnable(
        problem, prepared, spec, start_plane, planes.size, metadata=metadata
    )
    # The backend's door for a spec handed here rather than the problem's own, which
    # `select` already checked (SEP 23: a backend without a statistic refuses it).
    refusal = statistics_refusal(backend.name, spec.names)
    if refusal is not None:
        raise DriverError(refusal)
    retention = field_retention(prepared, fields, field_planes)
    _check_retainable(output, retention)
    check_snapshot_request(prepared, output, snapshots, resume_history, fields)
    if tdse_descriptor(problem.route) is not None and not isinstance(
        output, CheckpointOutput
    ):
        raise DriverError(
            "TDSE output must support terminal checkpoints and call ledgers"
        )
    reporter = resolve_reporter(
        progress,
        period=progress_period,
        n_steps=_replayed_steps(prepared, start_plane),
        # The low-storage march records 0.0, "not estimated" (SEP 24 section 6.3).
        estimated=not isinstance(prepared.integration.method, LowStorageRungeKutta),
    )
    # The session and its service loop (SEP 47): imported here, since the service
    # loop builds on this module's output transactions.
    from solprop.backends.sessions import start_session  # noqa: PLC0415
    from solprop.driver.service import ServiceContext, serve  # noqa: PLC0415

    request = prepared.for_run(
        stats=spec,
        retention=retention,
        start_plane=start_plane,
        monitor=boundary_monitor,
        history=HistoryPolicy.from_choice(resume_history),
        snapshots=snapshots is not None,
    )
    logger.debug(
        "running %d save planes from index %d, collecting %d statistics",
        planes.size,
        start_plane,
        len(spec.names),
    )
    march = start.march
    session = start_session(
        backend,
        problem,
        request,
        start,
        started=reporter.started,
        stop=stop,
        owns=False,
    )
    try:
        if march is not None and restoration is not None:
            restoration = _restored(
                march, restoration, getattr(session, "march_refusal", None)
            )
        serve(
            session,
            output,
            ServiceContext(
                problem=problem,
                request=request,
                metadata=metadata,
                reporter=reporter,
                retention=retention,
                resume_history=resume_history,
                snapshots=snapshots,
                tdse_binding=tdse_binding,
                tdse_archive=tdse_archive,
                recipe=recipe,
                start_plane=start_plane,
                stop=stop,
                restoration=restoration,
            ),
        )
    finally:
        session.close()


def _restored(
    march: MarchSnapshot,
    restoration: Mapping[str, object],
    refusal: str | None,
) -> Mapping[str, object]:
    """Return a resume's restoration record once its session tried the history.

    The caller hands the session the method history it decided to continue, with the
    record of a successful restoration; a history the session could not install
    (another engine's rows, another build's shapes) starts afresh, which is warned
    about once and recorded as a reset (SEP 47 section 5.5).

    Parameters
    ----------
    march : solprop.output.schema.MarchSnapshot
        The history the start carried.
    restoration : collections.abc.Mapping
        The record of its restoration, as the caller expects it.
    refusal : str or None
        Why the session could not install it, or ``None`` when it did.

    Returns
    -------
    collections.abc.Mapping
        The record of what happened.
    """
    if refusal is None:
        logger.info(
            "continuing the Adams history at order %d, %d samples",
            march.order,
            len(march.identities),
        )
        return restoration
    # Asked to continue a history this backend cannot install: the run is still
    # correct to the integrator's tolerance, but it does less than was asked, so say so.
    warnings.warn(
        f"the Adams history starts afresh: {refusal}", SolpropWarning, stacklevel=4
    )
    return {**restoration, "state": "reset", "reason": "incompatible_representation"}


def _announce(  # noqa: PLR0913 - the run's parts, each a distinct object
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

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    metadata : solprop.output.schema.RunMetadata
        The run's record.
    backend : solprop.backends.protocol.Backend
        The backend about to step it.
    output : solprop.output.schema.Output
        The open handler.
    reporter : solprop.log.ProgressReporter
        Where progress goes; only a :class:`solprop.log.SummarisingReporter` is shown
        the summary.
    start : tuple of int and float
        The first save plane beyond the loaded state, and the position the run starts
        from, m.
    request : solprop.translate.run_request.RunRequest
        The run's request.
    boundary_check : solprop.monitor.BoundaryCheck or None, optional
        The thresholds the window's edges are watched with, or ``None``.
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


def check_snapshot_request(
    request: RunRequest,
    output: Output,
    snapshots: SnapshotController | None,
    resume_history: bool | None,
    fields: FieldsChoice,
) -> None:
    """Refuse unsupported snapshot policies before allocating native resources.

    No backend is refused by name: every backend's fused and exponential Adams
    engines, the native backend's resident one included, export their history
    (SEP 29 section 6.6).

    Parameters
    ----------
    request : solprop.translate.run_request.RunRequest
        The run's request, whose method decides what a snapshot needs.
    output : Output
        Requested optical writer.
    snapshots : SnapshotController or None
        Optional snapshot request controller.
    resume_history : bool or None
        Whether the caller explicitly disabled Adams history retention.
    fields : FieldsChoice
        Saved-field retention policy.

    Raises
    ------
    DriverError
        If snapshots are requested outside their supported scope.
    """
    if snapshots is None:
        return
    if not isinstance(snapshots, SnapshotController):
        raise DriverError("snapshots must be a SnapshotController or None")
    if not isinstance(output, HDF5Output):
        raise DriverError("snapshots require HDF5Output")
    output._refuse_readonly()
    # A snapshot is taken at the next accepted endpoint, persisted as a running
    # checkpoint when no plane is saved there, so it needs no step planes (SEP 47
    # section 5.5).
    # Interpolated output is admitted: every session takes a snapshot at an accepted
    # endpoint only, once the planes its step crossed are observed, so a checkpoint
    # never needs the step's extension to reconstruct a plane (SEP 47 section 5.5).
    if fields != "all":
        raise DriverError("snapshots require fields='all'")
    if isinstance(request.integration.method, Multistep) and resume_history is False:
        raise DriverError(
            "snapshots require Adams history; resume_history=False is incompatible"
        )


def field_retention(
    request: RunRequest, fields: object, field_planes: Sequence[int]
) -> FieldRetention:
    """Return a run's field retention, resolved against its saved planes.

    The driver's door to :func:`solprop.output.schema.field_retention_refusal`, the
    validator :class:`~solprop.driver.simulation.SimulationSettings` and
    :func:`~solprop.driver.scan.run_scan` share.

    Parameters
    ----------
    request : solprop.translate.run_request.RunRequest
        The run's request: its save planes, and the entrance, are the planes.
    fields : object
        ``"all"``, ``"planes"`` or ``"none"``.
    field_planes : collections.abc.Sequence of int
        Saved-plane indices for ``"planes"``, negative from the end.

    Returns
    -------
    solprop.output.schema.FieldRetention
        The rule, its indices counted from the entrance.

    Raises
    ------
    DriverError
        If the request is not valid, names a plane outside the run, asks for
        ``fields='planes'`` while ``step_planes`` saves an unknown number of planes.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.driver.run import field_retention
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech", n_save=3,
    ... )
    >>> field_retention(simulation._run_request(), "planes", [-1]).field_planes
    (2,)
    """
    refusal = field_retention_refusal(fields, field_planes)
    if refusal is not None:
        raise DriverError(refusal)
    observation = request.observation
    if fields == "planes" and observation.step_planes:
        msg = (
            "fields='planes' names saved planes by index, and step_planes saves the "
            "march's own steps as well, so the plane count is not known in advance; "
            "keep fields='all' or 'none' beside step_planes"
        )
        raise DriverError(msg)
    try:
        return FieldRetention(
            cast("FieldsChoice", fields), tuple(int(index) for index in field_planes)
        ).resolved(observation.save_planes.size + 1)
    except OutputError as error:
        raise DriverError(str(error)) from error


def _retain(
    output: Output, retention: FieldRetention, field_shape: tuple[int, ...]
) -> None:
    """Tell an output that can hold it which planes keep their field.

    Called once the output is open, so that a file being continued compares the rule
    with the one it holds.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's handler.
    retention : solprop.output.schema.FieldRetention
        The run's resolved rule.
    field_shape : tuple of int
        ``(n_modes, n_pol, n_omega)`` of the run's state.
    """
    if isinstance(output, RetainingOutput):
        output.retain(retention, field_shape)


def _check_retainable(output: Output, retention: FieldRetention) -> None:
    """Refuse a field-free rule for an output that cannot hold one, before opening it.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's handler.
    retention : solprop.output.schema.FieldRetention
        The run's resolved rule.

    Raises
    ------
    DriverError
        If the rule keeps fewer than every field and the output has no ``retain``
        (:class:`~solprop.output.schema.RetainingOutput`).
    """
    if retention.fields != "all" and not isinstance(output, RetainingOutput):
        msg = (
            f"this run asks for {retention.describe()} and its output "
            f"{type(output).__name__} cannot store field-free planes: give it a "
            "retain method (RetainingOutput), or use a MemoryOutput or an HDF5Output, "
            "or fields='all'"
        )
        raise DriverError(msg)


def _resume_position(
    output: Output,
    checkpoint_output: CheckpointOutput | None,
    problem: PropagationProblem,
    request: RunRequest,
) -> tuple[float, bool]:
    """Resolve the actual checkpoint position and any scheduled observation owed.

    Parameters
    ----------
    output : Output
        Open handler containing the regular saved observations.
    checkpoint_output : CheckpointOutput or None
        Checkpoint-capable handler, if available, holding any terminal state.
    problem : PropagationProblem
        Translated problem with the absolute launch position, m.
    request : solprop.translate.run_request.RunRequest
        The run's request, with the scheduled save positions, m.

    Returns
    -------
    tuple of float and bool
        Actual restart position in m and whether its scheduled observation still
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

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The run's problem.
    backend : solprop.backends.protocol.Backend
        The selected backend.

    Raises
    ------
    BackendError
        If the problem has a requested collar and the backend does not read the edge.
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

    Parameters
    ----------
    output : Output
        Open handler, before this execution saves or records anything.
    checkpoint_output : CheckpointOutput or None
        Checkpoint-capable handler, if available, holding the earlier outcome.

    Returns
    -------
    resumed : bool
        Whether the handler holds an earlier execution's planes or outcome.
    record : collections.abc.Mapping or None
        The latest earlier execution's boundary-monitor record, searching back past
        executions that kept none, or ``None`` when none did.
    unwatched : tuple of tuple of float
        The stretches, ``(start, end)`` in m, that executions after that record ran
        without a monitor; from the launch when no execution kept one.
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


def _open_run(
    output: Output,
    metadata: RunMetadata,
    backend: Backend | SessionFacts,
    binding: TDSEBinding | None,
) -> CheckpointOutput | None:
    """Attach actual runtime facts outside the scientific resume identity.

    Parameters
    ----------
    output : Output
        Handler to open without changing its scientific identity.
    metadata : RunMetadata
        Scientific description of this run.
    backend : Backend
        Prepared optical backend supplying actual execution facts.
    binding : TDSEBinding or None
        Optional atomic service supplying execution provenance.

    Returns
    -------
    CheckpointOutput or None
        The handler's checkpoint interface, or ``None`` for a handler without one.
    """
    executed = backend.metadata()
    if binding is not None:
        executed = {**executed, "tdse": dict(binding.metadata())}
    output.open(metadata.with_backend(BackendRecord.from_metadata(executed)))
    return output if isinstance(output, CheckpointOutput) else None


def open_run_record(
    output: Output,
    metadata: RunMetadata,
    backend: Backend | SessionFacts,
    binding: TDSEBinding | None,
) -> None:
    """Open an output with the run's record and execution facts, and save nothing.

    Used when a run is refused after its resources were prepared and before its
    first step, such as a failed haCC consistency check (SEP 20 section 5.4): the
    record, with the binding's provenance, is written without a plane or a
    terminal checkpoint, so it cannot be mistaken for a propagation state.

    Parameters
    ----------
    output : Output
        Handler to open.
    metadata : RunMetadata
        Scientific description of the run.
    backend : Backend
        Prepared optical backend.
    binding : TDSEBinding or None
        Optional atomic service supplying execution provenance.
    """
    _open_run(output, metadata, backend, binding)


@dataclasses.dataclass(frozen=True)
class _FailedObservation:
    """Retain drained history and writer progress when a scheduled save fails."""

    pending: SavedPlane | None
    already_saved: bool
    error: BaseException
    reached: float
    clock: RunClock
    last_kept_field: bool


RESTART_RECORD = "restart"
"""The execution record's entry for what a restart keeps and restores (SEP 47 5.5).

Optional, under the output compatibility rules: ``"retention"``, what the run kept of
its method history at the record's position (``_MarchKeeper``), and, in a resumed
execution, ``"restoration"`` (what it restored,
``Simulation._continue_march``) and ``"monitor"`` (how
the boundary monitor was restored). Each decision is a mapping with a ``"state"`` and
a ``"reason"``. A file written before this record has none, which a reader takes as
unknown, never as a reason (``retention_at``).
"""


def _method_history(
    request: RunRequest,
) -> Literal["adams", "rk", "low_storage"]:
    """Return which kind of method history a run's march keeps.

    Parameters
    ----------
    request : solprop.translate.run_request.RunRequest
        The run's request.

    Returns
    -------
    {'adams', 'rk', 'low_storage'}
        An Adams history a resume can continue; Runge-Kutta controller memory (the
        proposal and the norm memory), which a resume resets; or no history at all.
    """
    integrator = request.integration.method
    if isinstance(integrator, Multistep):
        return "adams"
    if isinstance(integrator, LowStorageRungeKutta):
        return "low_storage"
    return "rk"


def retention_at(
    execution: Mapping[str, object] | None, z: float
) -> Mapping[str, object] | None:
    """Return the method-history retention an execution record states at `z`.

    Parameters
    ----------
    execution : collections.abc.Mapping or None
        An execution record (:attr:`solprop.output.schema.RunOutcome.execution`).
    z : float
        The position a resume starts from, m.

    Returns
    -------
    collections.abc.Mapping or None
        The retention decision recorded there, or ``None`` when the record states
        none at `z`: a file written before the restart record, or a position where no
        method history was persisted.
    """
    if execution is None:
        return None
    restart = execution.get(RESTART_RECORD)
    if not isinstance(restart, Mapping):
        return None
    retention = cast("Mapping[str, object]", restart).get("retention")
    if not isinstance(retention, Mapping):
        return None
    retention = cast("Mapping[str, object]", retention)
    return retention if retention.get("z") == z else None


class _MarchKeeper:
    """Keep the Adams march's state beside each save, and record what was kept.

    Every persisted plane and checkpoint states whether its method history was
    retained, omitted or is not applicable, and why (SEP 47 section 5.5); the run's
    execution record carries the latest decision (:meth:`record`).

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's handler; only one with ``save_march`` keeps a snapshot.
    backend : solprop.backends.protocol.Backend
        The run's backend, whose march has a snapshot to keep or none.
    policy : bool or None
        The run's ``resume_history``.
    method : {'adams', 'rk', 'low_storage'}, optional
        The march's history (:func:`_method_history`).
    retention : collections.abc.Mapping or None, optional
        The decision already recorded at the position the run starts from, carried
        into this execution's record until it persists a position of its own.
    """

    def __init__(
        self,
        output: Output,
        backend: Backend | SessionFacts,
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

        Returns
        -------
        dict or None
            ``"state"`` (``"retained"``, ``"omitted"`` or ``"not_applicable"``),
            ``"reason"`` and ``"z"``, m, with ``"nbytes"`` where the size was measured;
            ``None`` before anything was persisted.
        """
        return None if self._decision is None else dict(self._decision)

    def _decide(self, z: float, state: str, reason: str, **sizes: int) -> None:
        """Record a retention decision at `z`.

        Parameters
        ----------
        z : float
            The persisted position, m.
        state : str
            ``"retained"``, ``"omitted"`` or ``"not_applicable"``.
        reason : str
            Why.
        **sizes : int
            ``nbytes``, the history's size where it was measured, and ``limit``, the
            automatic limit it was held to, bytes.
        """
        self._decision = {"state": state, "reason": reason, "z": z, **sizes}

    def keep(self, z: float, *, required: bool = False) -> None:  # noqa: PLR0911, PLR0912 - one return per decision
        """Keep the march's state if it stands at `z`, the position just saved.

        Parameters
        ----------
        z : float
            The saved plane's or the checkpoint's position, m.
        required : bool, optional
            Require the full Adams state for an immutable snapshot, ignoring the
            automatic retention-size cutoff.

        Raises
        ------
        OutputError
            If required history cannot be persisted at this endpoint.
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

        Parameters
        ----------
        z : float
            The persisted position, m.
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

    Parameters
    ----------
    backend : Backend or SessionFacts
        The backend, refusing its state, or a poisoned session's facts.
    output : CheckpointOutput
        Handler recording the outcome.
    binding : TDSEBinding or None
        Optional atomic runtime, for the execution record.
    failure : _FailedObservation
        The original exception and the last saved position.
    step : tuple of float
        Where the abandoned step started and its length, m.
    restart : collections.abc.Mapping or None, optional
        The execution's restart record (:func:`_record_execution`).

    Returns
    -------
    float or None
        The position of the checkpoint fallen back to, m, or ``None`` without one.
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


def _record_execution(  # noqa: PLR0913 - the record's parts
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

    Parameters
    ----------
    output : CheckpointOutput or None
        Optional handler for completion and failure records.
    backend : Backend
        Optical backend supplying cumulative logical calls and execution facts.
    binding : TDSEBinding or None
        Optional atomic runtime supplying attempted native calls.
    status : _Status
        Requested status, absolute position in m and any error message.
    clock : solprop._host.RunClock
        When the run started.
    checkpoint : TerminalCheckpoint or None, optional
        Accepted field to preserve separately from scheduled observations.
    restart : collections.abc.Mapping or None, optional
        The execution's restart record (:data:`RESTART_RECORD`); ``None`` or an empty
        one records none.
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


def _replayed_steps(request: RunRequest, start_plane: int) -> int | None:
    """Return the step count a replayed schedule will take, when it is knowable.

    A fixed schedule knows how many steps are left, which is a better ETA than any rate
    of z: the steps of a schedule are not uniform in z, so a run replaying one covers
    the last tenth of a metre at a different price from the first. A **resumed** replay
    is deliberately excluded -- the schedule's cursor belongs to the backend, not to
    this loop, and a total that is wrong by the steps already taken is worse than
    falling back to the rate of z.

    Parameters
    ----------
    request : solprop.translate.run_request.RunRequest
        The run's request, whose ``replay`` is the schedule if there is one.
    start_plane : int
        Index of the first save plane to propagate to; above zero the run is a resume.

    Returns
    -------
    int or None
        The schedule's whole step count, or ``None`` for an adaptive or resumed run.
    """
    if request.replay is None or start_plane > 0:
        return None
    return request.replay.schedule.n_steps


def _refuse_a_record_that_misnames_the_basis(
    problem: PropagationProblem, metadata: RunMetadata
) -> None:
    """Refuse a run whose record and problem disagree about the basis, by name.

    The lowest of the whitelist's doors, and the one a caller reaches by building the
    pieces itself: ``run_problem(general_problem, backend, handler, metadata=...)``
    with a record that carries no ``modal`` identity would write a file that
    :func:`solprop.output.capability.capability_of` reads as an **ordinary modal**
    run, and every legacy accessor would then be admitted on a three-component basis
    -- the one thing the whitelist exists to prevent. The two are required to agree
    in both directions: a record claiming a general modal guide for a problem that
    carries no tables would promise a reconstruction no guide can be rebuilt for.

    What is compared is the record's **basis**, not whether it carries an identity at
    all, and it is read exactly as :func:`solprop.output.capability.capability_of` reads
    it. A capillary carrying a coherent wall block writes an identity too, but its basis
    is ``"capillary HE_1m"``: the block is a coupled flow through the same orthogonal
    ``HE_1m`` set the run always had, it builds no ``transform.modal_tables``, and the
    ordinary modal accessors stay admitted on it. Keying on the identity's presence
    alone would refuse that run here, for a basis disagreement that does not exist.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem about to be run.
    metadata : solprop.output.schema.RunMetadata
        The record about to be written.

    Raises
    ------
    DriverError
        Naming which of the two says what.
    """
    general = problem.transform.modal_tables is not None
    if general == (capability_of(metadata) == "general_modal"):
        return
    if general:
        msg = (
            "this problem carries transform.modal_tables and its run record carries "
            "no modal identity: the file would be read back as an ordinary modal "
            "run, on which energy, peak_power and peak_intensity are admitted, and "
            "they are not defined for a three-component basis. Build the record "
            "with metadata_for(problem, grid, modes=..., modal=modal_identity(...))"
        )
    else:
        msg = (
            "this run record carries a general modal identity and the problem "
            "carries no transform.modal_tables: the file would promise a "
            "reconstruction from a guide nothing in this run was built from. Pass "
            "the record of the problem being run"
        )
    raise DriverError(msg)


def _refuse_unrunnable(
    problem: PropagationProblem,
    request: RunRequest,
    spec: StatsSpec,
    start_plane: int,
    n_planes: int,
    *,
    metadata: RunMetadata,
) -> None:
    """Refuse, by name, every run this loop cannot execute.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem about to be run.
    request : solprop.translate.run_request.RunRequest
        The request the backend was built for.
    spec : solprop.stats.StatsSpec
        The statistics asked for. A ``periodic`` subset is accepted -- the seam carries
        it, and a name no backend can evaluate between planes was refused by
        :class:`~solprop.stats.StatsSpec` when the spec was built -- but it must agree
        with the request's own, because the channel is armed when the backend is built
        and reads the request's spec, not this one.
    start_plane : int
        Index of the first save plane to propagate to.
    n_planes : int
        How many save planes the problem has.
    metadata : solprop.output.schema.RunMetadata
        The record about to be written, which must say the same thing about the
        basis as the problem does.

    Raises
    ------
    DriverError
        Naming what cannot be done.
    """
    _refuse_a_record_that_misnames_the_basis(problem, metadata)
    channel = request.observation.stats
    if (spec.periodic, spec.period) != (channel.periodic, channel.period):
        msg = (
            f"the statistics given to this loop ask for {list(spec.periodic)} every "
            f"{spec.period} accepted steps and the translated problem asks for "
            f"{list(channel.periodic)} every {channel.period}; the between-planes "
            "channel is armed when the backend is built, from the problem's own spec, "
            "so the two would disagree -- pass the problem's spec, or translate again"
        )
        raise DriverError(msg)
    if not 0 <= start_plane <= n_planes:
        msg = (
            f"start_plane must be a save plane of this problem, between 0 and "
            f"{n_planes}, got {start_plane}"
        )
        raise DriverError(msg)
    if problem.transform.modal_tables is not None and "quadrature_error" in spec.names:
        # A silent `nan` at every plane would read as "no alarm raised" rather than
        # "no alarm exists": a provider's node rule carries no embedded coarse pair
        # to difference, so there is nothing to score.
        msg = (
            "quadrature_error is unavailable for this general modal guide; its "
            "node rule carries no embedded coarse rule to difference, so the "
            "transverse-resolution alarm has no error estimate and the statistic "
            "would be nan at every plane. The rule's certified tail bound and its "
            "sampled Gram defect are measured once, when the guide is built"
        )
        raise DriverError(msg)
    # The accessor whitelist, at the first of its two doors: never redefine
    # an old name on a basis it is not defined for, and never answer a general name
    # from a file that carries neither the certified metric nor the E/H reference it
    # promises. The second door is the processing accessors, and the reference
    # backend checks again for a caller who builds it and asks it directly.
    refusal = refusal_for(
        declared_capability(
            general_modal=problem.transform.modal_tables is not None,
            spatial=problem.transform.geometry is not None,
        ),
        [*spec.names, *spec.periodic],
        withheld=withheld_accessors(
            general_modal=problem.transform.modal_tables is not None,
            coupled_flow=problem.linear.coupled_flow is not None,
        ),
    )
    if refusal is not None:
        raise DriverError(refusal)
    refusal = axis_refusal(problem, [*spec.names, *spec.periodic])
    if refusal is not None:
        raise DriverError(refusal)
