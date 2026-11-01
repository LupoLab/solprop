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
import math
import time
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

from solprop._host import RunClock, machine, usage
from solprop._interrupts import committing, handled
from solprop.backends.populations import axis_refusal
from solprop.backends.protocol import Backend, Observation
from solprop.backends.protocol import StepHistory as BackendHistory
from solprop.backends.registry import statistics_refusal
from solprop.driver.memory import MemoryAdmission
from solprop.driver.summary import RUN_SUMMARY_ANNOTATION, resources, run_summary
from solprop.exceptions import (
    DriverError,
    OutputError,
    ResponseBudgetError,
    RunTerminated,
)
from solprop.fourier import power_amplitude
from solprop.log import (
    DEFAULT_PROGRESS_PERIOD,
    InterruptibleReporter,
    LogReporter,
    ProgressChoice,
    ProgressReporter,
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
    BackendRecord,
    CheckpointOutput,
    FieldRetention,
    FieldsChoice,
    Output,
    PlaneStore,
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
from solprop.translate.linop import power_weight_at
from solprop.translate.problem import KernelProblem
from solprop.translate.tdse import TDSEBinding

__all__ = [
    "MARCH_SNAPSHOT_AUTO_BYTES",
    "LogReporter",
    "ProgressReporter",
    "check_snapshot_request",
    "field_retention",
    "open_run_record",
    "plane_from_observation",
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
    history : solprop.backends.protocol.StepHistory
        What the stepper did since the previous observation.

    Returns
    -------
    solprop.output.schema.StepHistory
        The same attempts, in the columns the output layer stores.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.protocol import StepHistory
    >>> from solprop.driver.run import _history_of
    >>> stored = _history_of(StepHistory(np.array([0.5, 1.0]), np.array([0.5, 0.5]),
    ...                                  np.array([0.4, 0.5]), np.array([True, True]),
    ...                                  np.array([0.25, 0.75])))
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
    member: int = 0,
    field: bool = True,
) -> SavedPlane:
    """Turn one observation and its statistics into the plane an output handler stores.

    The only shape change on the way: the seam's arrays carry a leading batch axis and a
    stored plane does not, so one member is taken from the field, the noise phase and
    every statistic (``docs/dev/conventions.md`` section 6). Nothing is conjugated --
    the field stays in the internal FFT convention, and the handler converts it on the
    way to storage (``docs/dev/conventions.md`` section 2). Nothing is rescaled either:
    the field is whatever amplitude the observation carries, the backend's
    vacuum-normalised one unless the caller has applied the interface conversion to
    true-power amplitudes (``_power_observation``; see ``docs/dev/decisions.md``).

    Parameters
    ----------
    observation : solprop.backends.protocol.Observation
        The state at a save plane, from
        :meth:`solprop.backends.protocol.Backend.observe`.
    stats : collections.abc.Mapping, optional
        What :meth:`solprop.backends.protocol.Backend.stats` returned there, keyed by
        statistic name, each with a leading batch axis. Empty when omitted.

        The observation's between-planes rows are taken across the same way: their
        batch axis is the one *after* the row axis, so the member is taken from axis
        one rather than axis zero.
    member : int, optional
        Which batch member to store; 0, the only member of an unbatched run, by
        default. :func:`solprop.driver.batch.split_observation` stores each member of
        a lockstep batch in turn.
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
    >>> from solprop.backends.protocol import Observation, StepHistory
    >>> from solprop.driver.run import plane_from_observation
    >>> history = StepHistory(np.array([0.5]), np.array([0.5]), np.array([0.3]),
    ...                       np.array([True]), np.array([0.02]))
    >>> observation = Observation(0.5, np.ones((1, 1, 1, 4), complex), None, history,
    ...                           1, 0)
    >>> plane = plane_from_observation(observation, {"energy": np.ones((1, 1, 1))})
    >>> plane.z, plane.shape, plane.stats["energy"].shape
    (0.5, (1, 1, 4), (1, 1))
    """
    values = {} if stats is None else stats
    rows = observation.step_stats
    return SavedPlane(
        z=observation.z,
        field=observation.field[member]
        if field and observation.field is not None
        else None,
        stats={name: np.asarray(value[member]) for name, value in values.items()},
        step_history=_history_of(observation.step_history),
        step_stats=StepStats(
            z=rows.z,
            step=rows.step,
            values={
                name: np.asarray(value[:, member])
                for name, value in rows.values.items()
            },
        ),
        noise_phase=None
        if observation.noise_phase is None or not field
        else observation.noise_phase[member],
    )


def _power_observation(problem: KernelProblem, observation: Observation) -> Observation:
    """Convert modal output units once, before the handler changes Fourier sign.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        Translated problem supplying modal weights or spatial geometry.
    observation : solprop.backends.protocol.Observation
        Internal state and diagnostics at a saved plane.

    Returns
    -------
    solprop.backends.protocol.Observation
        Observation carrying true-power amplitudes and unchanged diagnostics.
    """
    if problem.transform.geometry is not None or observation.field is None:
        return observation
    return dataclasses.replace(
        observation,
        field=power_amplitude(
            observation.field, power_weight_at(problem, observation.z)
        ),
    )


@handled()  # SIGTERM and Ctrl-C stop a run between its commits
def run_problem(  # noqa: PLR0912, PLR0913, PLR0915 - the loop's inputs and commit sequence
    problem: KernelProblem,
    backend: Backend,
    output: Output,
    *,
    metadata: RunMetadata,
    stats_spec: StatsSpec | None = None,
    progress: ProgressReporter | ProgressChoice | None = None,
    progress_period: float = DEFAULT_PROGRESS_PERIOD,
    start_plane: int = 0,
    memory: MemoryAdmission | None = None,
    tdse_binding: TDSEBinding | None = None,
    resume_history: bool | None = None,
    fields: FieldsChoice = "all",
    field_planes: Sequence[int] = (),
    snapshots: SnapshotController | None = None,
    tdse_archive: TDSEArchiveWriter | None = None,
    boundary_monitor: BoundaryMonitor | None = None,
) -> None:
    """Run one translated problem to its save planes, writing each one out.

    The loop of ``docs/dev/architecture.md`` section 5, usable without a
    :class:`solprop.driver.simulation.Simulation`. The backend must already hold its
    initial state (:meth:`solprop.backends.protocol.Backend.set_initial`); the handler
    is opened here, saved to plane by plane, and closed in a ``finally`` so that an
    interrupted run leaves a readable file.

    The plane at ``problem.initial.z`` is saved first, with an empty step history, and
    then each save plane in turn: propagate, observe, collect statistics, save. Steps
    land on the planes exactly, so a saved field is an exact step endpoint and there
    is no interpolation (``docs/dev/stepper.md`` section 3), unless the problem asks
    for dense output. With ``stepper.step_planes`` the march also stops at the end of
    every ``step_planes``-th accepted step, which is saved too before the same plane
    is asked for again. A plane whose propagation did not meet its tolerance is logged
    as a warning naming that plane (``docs/dev/stepper.md`` section 8).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem, whose ``stepper.save_planes`` is the itinerary.
    backend : solprop.backends.protocol.Backend
        The backend stepping it, already loaded with its initial state.
    output : solprop.output.schema.Output
        Where the planes go. Closed when the loop ends, whether or not it succeeded.
    metadata : solprop.output.schema.RunMetadata
        The record of the run, written when the handler opens.
        :func:`solprop.driver.simulation.metadata_for` builds one.
    stats_spec : solprop.stats.StatsSpec, optional
        Which statistics to collect at each plane. Defaults to the contract's own
        ``problem.stats_spec`` (section 8 of ``docs/dev/kernel-contract.md``).
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
    start_plane : int, optional
        Index of the first save plane strictly beyond the loaded state. A resume
        loads either the last regular plane or a separate terminal checkpoint at
        its actual absolute position. The launch plane is written when the handler
        holds no regular observations, so a resumed run never stores it twice.
    memory : solprop.driver.memory.MemoryAdmission or None, optional
        The run's memory admission (:func:`solprop.driver.memory.admit_memory`),
        recorded as ``memory`` in the backend record's settings; ``None`` records none.
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
        Requests immutable disk copies at committed accepted steps. Requires HDF5,
        ``step_planes=1``, all fields and non-interpolated observations.
    tdse_archive : solprop.output.tdse_archive.TDSEArchiveWriter or None, optional
        Live archive to suspend and copy beside the optical writer.
    boundary_monitor : solprop.monitor.BoundaryMonitor or None, optional
        Watches the time window's edges (SEP 21): attached to the backend once its
        launch is recorded and fed every committed step. Its record is written into
        the execution record at every save plane without the per-plane rows, which
        the closing record (complete, failed or out of budget) carries, and its removed
        fraction is reported at the end. ``None`` watches nothing.

    Raises
    ------
    DriverError
        If the statistics given here disagree with the problem's own about the
        between-planes channel; if the problem carries a batch axis, which the output
        schema has no room for yet; if the record and the problem disagree about
        whether this is a general modal run; if `start_plane` is not a plane of
        this problem; or if the field retention is not valid, names a plane outside
        the run, asks for ``fields='planes'`` beside ``step_planes`` (whose plane
        count is not known in advance), or is not ``'all'`` for an output that cannot
        store field-free planes.

    Examples
    --------
    >>> from solprop.backends.registry import select
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
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-6, atol=1e-16),
    ...                    save_planes=[0.005, 0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> backend = select("numpy_ref", problem, grid=grid, modes=[gnlse.mode])
    >>> backend.set_initial(problem.initial.field0, None)
    >>> handler = MemoryOutput()
    >>> run_problem(problem, backend, handler,
    ...             metadata=metadata_for(problem, grid, modes=[gnlse.mode]))
    >>> backend.close()
    >>> handler.n_saved, handler["z"].tolist()
    (3, [0.0, 0.005, 0.01])
    """
    spec = problem.stats_spec if stats_spec is None else stats_spec
    planes = np.asarray(problem.stepper.save_planes, dtype=np.float64)
    _refuse_unrunnable(problem, spec, start_plane, planes.size, metadata=metadata)
    # The backend's door for a spec handed here rather than the problem's own, which
    # `select` already checked (SEP 23: a backend without a statistic refuses it).
    refusal = statistics_refusal(backend.name, spec.names)
    if refusal is not None:
        raise DriverError(refusal)
    retention = field_retention(problem, fields, field_planes)
    _check_retainable(output, retention)
    check_snapshot_request(problem, output, snapshots, resume_history, fields)
    if problem.tdse is not None and not isinstance(output, CheckpointOutput):
        raise DriverError(
            "TDSE output must support terminal checkpoints and call ledgers"
        )
    reporter = resolve_reporter(
        progress,
        period=progress_period,
        n_steps=_replayed_steps(problem, start_plane),
        # The low-storage march records 0.0, "not estimated" (SEP 24 section 6.3).
        estimated=not isinstance(problem.stepper.integrator, LowStorageRungeKutta),
    )
    backend.set_progress(reporter)
    started = time.perf_counter()
    clock = RunClock.start()
    logger.debug(
        "running %d save planes from index %d, collecting %d statistics",
        planes.size,
        start_plane,
        len(spec.names),
    )
    checkpoint_output = _open_run(
        output,
        metadata,
        backend,
        tdse_binding,
        memory,
    )
    # A store that preallocates (`PlaneStore.reserve`) sizes itself for the entrance
    # and every save plane; a resumed one keeps what it holds and grows to the rest.
    reserve = getattr(output, "reserve", None)
    if reserve is not None:
        reserve(planes.size + 1)
    _retain(output, retention, problem.shape[1:])
    reached, missing_current_save = _resume_position(output, checkpoint_output, problem)
    # Read before this execution's first record replaces the earlier one's.
    resumed, monitored = _earlier_execution(output, checkpoint_output)
    _announce(
        problem,
        metadata,
        backend,
        output,
        reporter,
        (start_plane, reached),
        boundary_check=None if boundary_monitor is None else boundary_monitor.check,
    )
    keeper = _MarchKeeper(output, backend, resume_history)
    pending: SavedPlane | None = None
    pending_save_count = output.n_saved
    snapshots_started = False
    failure: BaseException | None = None
    try:
        if snapshots is not None:
            snapshots._start()
            snapshots_started = True
        _record_execution(
            checkpoint_output,
            backend,
            tdse_binding,
            RunOutcome("running", reached),
            clock=clock,
        )
        # The entrance is saved when the handler holds nothing, rather than whenever the
        # run starts at plane zero: a run interrupted before it reached its first save
        # plane leaves a file holding the entrance and nothing else, and resuming it
        # starts at plane zero again without wanting a second copy of it.
        if output.n_saved == 0 or missing_current_save:
            keep = retention.keeps(output.n_saved)
            pending = plane_from_observation(
                _observed(problem, backend, keep=keep), field=keep
            )
            pending = dataclasses.replace(
                pending,
                stats={
                    name: np.asarray(value[0])
                    for name, value in backend.stats(spec).items()
                },
            )
            with committing():
                output.save(pending)
                _record_execution(
                    checkpoint_output,
                    backend,
                    tdse_binding,
                    RunOutcome("running", reached),
                    clock=clock,
                )
        attach = getattr(backend, "attach_monitor", None)
        if boundary_monitor is not None and attach is not None:
            if resumed:
                # The record continues across the resume (SEP 21, ADR-057).
                boundary_monitor.resume(
                    monitored,
                    plane=output.n_saved > 0 and float(output["z"][-1]) == reached,
                )
            attach(boundary_monitor)
        index = start_plane
        while index < planes.size:
            target = float(planes[index])
            pending = None
            pending_save_count = output.n_saved
            summary = backend.propagate_to(target)
            if not summary.met_tolerance:
                logger.warning(
                    "a step was too coarse for the tolerance it is scaled by on the "
                    "way to the save plane at z = %.6e m (plane %d of %d): the largest "
                    "scaled error norm was %.3e against the requested 1.0",
                    target,
                    index + 1,
                    planes.size,
                    summary.max_norm,
                )
            keep = retention.keeps(output.n_saved)
            pending = plane_from_observation(
                _observed(problem, backend, keep=keep), field=keep
            )
            pending_save_count = output.n_saved
            pending = dataclasses.replace(
                pending,
                stats={
                    name: np.asarray(value[0])
                    for name, value in backend.stats(spec).items()
                },
            )
            # The plane, the march's snapshot beside it and the execution record are
            # one commit: a signal waits for the three (`solprop._interrupts`).
            with committing():
                output.save(pending)
                keeper.keep(pending.z)
                reached = summary.z
                _record_execution(
                    checkpoint_output,
                    backend,
                    tdse_binding,
                    RunOutcome("running", reached),
                    clock=clock,
                )
            if snapshots is not None:
                assert isinstance(output, HDF5Output)

                def persist(z: float = reached) -> None:
                    keeper.keep(
                        z,
                        required=isinstance(problem.stepper.integrator, Multistep),
                    )
                    _record_execution(
                        checkpoint_output,
                        backend,
                        tdse_binding,
                        RunOutcome("running", z),
                        clock=clock,
                    )

                snapshots._service(
                    output,
                    tdse_archive,
                    next_save_index=int(np.searchsorted(planes, reached, side="right")),
                    require_history=isinstance(problem.stepper.integrator, Multistep),
                    persist=persist,
                    provenance=metadata,
                )
            reporter.plane(summary.z, summary)
            # A natural step can stop short of the requested plane (SEP 14).
            if math.isclose(summary.z, target, rel_tol=1e-12, abs_tol=0.0):
                index += 1
        if boundary_monitor is not None and attach is not None:
            boundary_monitor.finish()
        with committing():
            _record_execution(
                checkpoint_output,
                backend,
                tdse_binding,
                RunOutcome("complete", reached),
                clock=clock,
            )
        reporter.finished(time.perf_counter() - started)
        if isinstance(reporter, SummarisingReporter):
            reporter.summary(resources(usage(clock)))
    except BaseException as error:
        failure = error
        failed_z = _persist_failure(
            problem,
            backend,
            checkpoint_output,
            tdse_binding,
            keeper,
            _FailedObservation(
                pending, output.n_saved > pending_save_count, error, reached, clock
            ),
        )
        if failed_z is not None:
            reached = failed_z
        if isinstance(error, RunTerminated):
            logger.warning(
                "%s at z = %.6e m; the run's state there is written, and "
                "resume=True continues it",
                error,
                reached,
            )
        # A reporter holding a terminal (a bar) gives it back before the error
        # propagates; a reporter implementing only the four basic calls is not told,
        # and it is not this loop's business to demand more of it.
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


def _announce(  # noqa: PLR0913 - the run's parts, each a distinct object
    problem: KernelProblem,
    metadata: RunMetadata,
    backend: Backend,
    output: Output,
    reporter: ProgressReporter,
    start: tuple[int, float],
    *,
    boundary_check: BoundaryCheck | None = None,
) -> None:
    """Show the run's summary through the reporter and record it on a plane store.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
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
    boundary_check : solprop.monitor.BoundaryCheck or None, optional
        The thresholds the window's edges are watched with, or ``None``.
    """
    start_plane, z_start = start
    summary = run_summary(
        problem,
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
    problem: KernelProblem,
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
    problem : KernelProblem
        Translated run and observation policy.
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
    if problem.shape[0] != 1 or problem.stepper.step_planes != 1:
        raise DriverError("snapshots require an unbatched run with step_planes=1")
    if problem.stepper.interpolate_planes or fields != "all":
        raise DriverError("snapshots require fields='all' and no interpolated output")
    if isinstance(problem.stepper.integrator, Multistep) and resume_history is False:
        raise DriverError(
            "snapshots require Adams history; resume_history=False is incompatible"
        )


def _observed(problem: KernelProblem, backend: Backend, *, keep: bool) -> Observation:
    """Observe the backend at a save plane, in output units where the field is kept.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    backend : solprop.backends.protocol.Backend
        The backend at the plane.
    keep : bool
        Whether the plane keeps its field. A field-free plane asks the backend for no
        field at all (SEP 22 section 5.1), so nothing crosses to the host.

    Returns
    -------
    solprop.backends.protocol.Observation
        The observation, its field in true-power units when it is kept and ``None``
        otherwise.
    """
    return _power_observation(problem, backend.observe(field=keep))


def field_retention(
    problem: KernelProblem, fields: object, field_planes: Sequence[int]
) -> FieldRetention:
    """Return a run's field retention, resolved against its saved planes.

    The driver's door to :func:`solprop.output.schema.field_retention_refusal`, the
    validator :class:`~solprop.driver.simulation.SimulationSettings` and
    :func:`~solprop.driver.scan.run_scan` share.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem: its save planes, and the entrance, are the planes.
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
    >>> field_retention(simulation.translate(), "planes", [-1]).field_planes
    (2,)
    """
    refusal = field_retention_refusal(fields, field_planes)
    if refusal is not None:
        raise DriverError(refusal)
    if fields == "planes" and problem.stepper.step_planes:
        msg = (
            "fields='planes' names saved planes by index, and step_planes saves the "
            "march's own steps as well, so the plane count is not known in advance; "
            "keep fields='all' or 'none' beside step_planes"
        )
        raise DriverError(msg)
    try:
        return FieldRetention(
            cast("FieldsChoice", fields), tuple(int(index) for index in field_planes)
        ).resolved(len(problem.stepper.save_planes) + 1)
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
    retain = getattr(output, "retain", None)
    if retain is not None:
        retain(retention, field_shape)


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
        If the rule keeps fewer than every field and the output has no ``retain``.
    """
    if retention.fields != "all" and getattr(output, "retain", None) is None:
        msg = (
            f"this run asks for {retention.describe()} and its output "
            f"{type(output).__name__} cannot store field-free planes (it has no "
            "retain); use a MemoryOutput or an HDF5Output, or fields='all'"
        )
        raise DriverError(msg)


def _resume_position(
    output: Output, checkpoint_output: CheckpointOutput | None, problem: KernelProblem
) -> tuple[float, bool]:
    """Resolve the actual checkpoint position and any scheduled observation owed.

    Parameters
    ----------
    output : Output
        Open handler containing the regular saved observations.
    checkpoint_output : CheckpointOutput or None
        Checkpoint-capable handler, if available, holding any terminal state.
    problem : KernelProblem
        Translated problem with absolute launch and scheduled save positions, m.

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
        and bool(np.any(problem.stepper.save_planes == reached))
        and (output.n_saved == 0 or float(output["z"][-1]) < reached)
    )
    return reached, missing_current_save


def _earlier_execution(
    output: Output, checkpoint_output: CheckpointOutput | None
) -> tuple[bool, Mapping[str, object] | None]:
    """Return whether this execution resumes another, and that one's monitor record.

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
        That execution's boundary-monitor record, or ``None`` without one.
    """
    outcome = None if checkpoint_output is None else checkpoint_output.outcome
    resumed = output.n_saved > 0 or outcome is not None
    execution = None if outcome is None else outcome.execution
    record = None if execution is None else execution.get("boundary_monitor")
    if not isinstance(record, Mapping):
        return resumed, None
    return resumed, cast("Mapping[str, object]", record)


def _open_run(
    output: Output,
    metadata: RunMetadata,
    backend: Backend,
    binding: TDSEBinding | None,
    memory: MemoryAdmission | None,
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
    memory : MemoryAdmission or None
        Optional run admission record in bytes.

    Returns
    -------
    CheckpointOutput or None
        The handler's checkpoint interface, or ``None`` for a handler without one.
    """
    executed = backend.metadata()
    if binding is not None:
        executed = {**executed, "tdse": dict(binding.metadata())}
    if memory is not None:
        executed = {**executed, "memory": memory.record()}
    output.open(
        dataclasses.replace(metadata, backend=BackendRecord.from_metadata(executed))
    )
    return output if isinstance(output, CheckpointOutput) else None


def open_run_record(
    output: Output,
    metadata: RunMetadata,
    backend: Backend,
    binding: TDSEBinding | None,
    memory: MemoryAdmission | None,
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
    memory : MemoryAdmission or None
        Optional run admission record.
    """
    _open_run(output, metadata, backend, binding, memory)


@dataclasses.dataclass(frozen=True)
class _FailedObservation:
    """Retain drained history and writer progress when a scheduled save fails."""

    pending: SavedPlane | None
    already_saved: bool
    error: BaseException
    reached: float
    clock: RunClock


class _MarchKeeper:
    """Keep the Adams march's state beside each save, when the output can hold it.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's handler; only one with ``save_march`` keeps a snapshot.
    backend : solprop.backends.protocol.Backend
        The run's backend; only one with ``march_snapshot`` has one to keep.
    policy : bool or None
        The run's ``resume_history``.
    """

    def __init__(self, output: Output, backend: Backend, policy: bool | None) -> None:
        """Hold what keeping a snapshot needs."""
        self._save = getattr(output, "save_march", None)
        self._snapshot = getattr(backend, "march_snapshot", None)
        self._policy = policy
        self._kept_z: float | None = None

    def keep(self, z: float, *, required: bool = False) -> None:
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
        if self._save is None or self._snapshot is None:
            if required:
                raise OutputError(
                    "snapshot requires a backend and output with Adams history"
                )
            return
        if self._policy is False and not required:
            return
        snapshot = self._snapshot()
        if snapshot is None or snapshot.z != z:
            if required:
                raise OutputError(
                    "snapshot requires Adams history at the accepted endpoint"
                )
            return
        if self._policy is None and not required:
            if snapshot.nbytes > MARCH_SNAPSHOT_AUTO_BYTES:
                logger.info(
                    "the Adams march's state is not kept for a resume: %d bytes is "
                    "over the %d kept by default (resume_history=True keeps it)",
                    snapshot.nbytes,
                    MARCH_SNAPSHOT_AUTO_BYTES,
                )
                self._policy = False
                return
            self._policy = True
        self._save(snapshot)
        self._kept_z = z


def _persist_failure(
    problem: KernelProblem,
    backend: Backend,
    output: CheckpointOutput | None,
    binding: TDSEBinding | None,
    keeper: _MarchKeeper,
    failure: _FailedObservation,
) -> float | None:
    """Checkpoint the accepted state without replacing the original exception.

    Parameters
    ----------
    problem : KernelProblem
        Translated problem, including absolute save positions in m.
    backend : Backend
        Backend retaining its last accepted field.
    output : CheckpointOutput or None
        Handler supporting separate terminal state persistence, when available.
    binding : TDSEBinding or None
        Optional live atomic service providing accounting snapshots.
    keeper : _MarchKeeper
        Keeps the Adams march's state beside the checkpoint.
    failure : _FailedObservation
        Original exception and any observation already drained before failure.

    Returns
    -------
    float or None
        Last accepted position in m, if it could be observed.
    """
    if output is None:
        return None
    pending = failure.pending
    in_flight = getattr(backend, "_in_flight", None)
    held = None
    if pending is None and in_flight is not None:
        holder = getattr(backend, "held_observation", None)
        try:
            held = None if holder is None else holder()
        except Exception as held_error:
            # A held state that cannot be observed (one not finite outside the norm
            # mask, say) must not replace the original exception: the failure is
            # recorded as an abandoned step, with the reason beside it.
            failure.error.add_note(
                f"the held plane could not be observed: {held_error}"
            )
            logger.exception("could not observe the held plane")
            held = None
        if held is None:
            _record_abandoned_step(backend, output, binding, failure, in_flight)
            return None
    reached = None
    try:
        if held is not None:
            # The low-storage step in flight holds neither end, and the last plane
            # kept no field: the state held there is checkpointed (SEP 22, Q14).
            # Its history went out with the plane, so the checkpoint carries none.
            pending = plane_from_observation(_power_observation(problem, held))
        elif pending is None:
            pending = plane_from_observation(
                _power_observation(problem, backend.observe())
            )
        elif pending.field is None:
            # A field-free plane keeps no field, and a checkpoint always does (SEP
            # 22, Q1): the state still stands where the plane was observed, so it is
            # observed again for its field alone, the history being the plane's.
            again = plane_from_observation(
                _power_observation(problem, backend.observe())
            )
            pending = dataclasses.replace(
                pending, field=again.field, noise_phase=again.noise_phase
            )
        history = pending.step_history
        ends = (history.z + history.h)[history.accepted]
        landed = (
            not problem.stepper.interpolate_planes
            or pending.z == problem.initial.z
            or bool(np.any(np.isclose(ends, pending.z, rtol=1e-12, atol=0.0)))
        )
        if failure.already_saved:
            # Appending may succeed before flushing fails. The regular plane then
            # already owns this history; replay must not append it twice.
            pending = dataclasses.replace(
                pending,
                step_history=StepHistory.empty(),
                step_stats=StepStats.empty(),
            )
        reached = pending.z
        checkpoint = TerminalCheckpoint(
            plane=pending,
            next_save_index=int(
                np.searchsorted(problem.stepper.save_planes, pending.z, side="right")
            ),
            landed=landed,
        )
        # Written while the run is already stopping: a second signal is dropped.
        with committing(deliver=False):
            _record_execution(
                output,
                backend,
                binding,
                RunOutcome(
                    "budget_exhausted"
                    if isinstance(failure.error, ResponseBudgetError)
                    else "failed",
                    reached,
                    message=f"{type(failure.error).__name__}: {failure.error}",
                ),
                clock=failure.clock,
                checkpoint=checkpoint,
            )
            keeper.keep(reached)
    except Exception as checkpoint_error:
        failure.error.add_note(
            f"terminal checkpoint could not be persisted: {checkpoint_error}"
        )
        logger.exception("could not persist the terminal checkpoint")
    return reached


def _record_abandoned_step(
    backend: Backend,
    output: CheckpointOutput,
    binding: TDSEBinding | None,
    failure: _FailedObservation,
    step: tuple[float, float],
) -> None:
    """Record a failure that left a low-storage step in flight, with no checkpoint.

    The low-storage march accepts each step in place (SEP 24 section 6.2), so a step
    abandoned between its start and its commit leaves a state that is neither of the
    step's ends, which the backend refuses to observe. When the last plane kept its
    field nothing is held either (a field-free one's state is held and checkpointed
    instead, SEP 22, Q14), and nothing honest can be written as a terminal
    checkpoint: the file ends at its last saved plane, from which a resume continues
    to rounding, and ``step_planes`` bounds what is lost.

    Parameters
    ----------
    backend : Backend
        The backend, refusing its state.
    output : CheckpointOutput
        Handler recording the outcome.
    binding : TDSEBinding or None
        Optional atomic runtime, for the execution record.
    failure : _FailedObservation
        The original exception and the last saved position.
    step : tuple of float
        Where the abandoned step started and its length, m.
    """
    z, dz = step
    reason = (
        f"no terminal checkpoint is written: the low-storage step of {dz:.6g} m from "
        f"z = {z:.6g} m was abandoned in flight, so the state holds neither end of "
        f"it; the file ends at its last saved plane, z = {failure.reached:.6g} m, "
        "from which resume=True continues"
    )
    failure.error.add_note(reason)
    logger.warning("%s", reason)
    try:
        with committing(deliver=False):
            _record_execution(
                output,
                backend,
                binding,
                RunOutcome(
                    "failed",
                    failure.reached,
                    message=f"{type(failure.error).__name__}: {failure.error}; "
                    f"{reason}",
                ),
                clock=failure.clock,
            )
    except Exception as record_error:
        failure.error.add_note(f"the failure could not be recorded: {record_error}")
        logger.exception("could not record the failure")


def _record_execution(
    output: CheckpointOutput | None,
    backend: Backend,
    binding: TDSEBinding | None,
    outcome: RunOutcome,
    *,
    clock: RunClock,
    checkpoint: TerminalCheckpoint | None = None,
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
    outcome : RunOutcome
        Requested status, absolute position in m and any error message.
    clock : solprop._host.RunClock
        When the run started.
    checkpoint : TerminalCheckpoint or None, optional
        Accepted field to preserve separately from scheduled observations.
    """
    if output is None:
        return
    record = {**backend.metadata(), "host": machine(), "resources": usage(clock)}
    monitor = record.get("boundary_monitor")
    if outcome.status == "running" and isinstance(monitor, Mapping):
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
    output.record_outcome(
        dataclasses.replace(
            outcome,
            logical_calls=logical,
            atomic_calls=cast("Mapping[str, object]", atomic)
            if isinstance(atomic, Mapping)
            else {},
            execution=record,
        ),
        checkpoint,
    )


def _replayed_steps(problem: KernelProblem, start_plane: int) -> int | None:
    """Return the step count a replayed schedule will take, when it is knowable.

    A fixed schedule knows how many steps are left, which is a better ETA than any rate
    of z: the steps of a schedule are not uniform in z, so a run replaying one covers
    the last tenth of a metre at a different price from the first. A **resumed** replay
    is deliberately excluded -- the schedule's cursor belongs to the backend, not to
    this loop, and a total that is wrong by the steps already taken is worse than
    falling back to the rate of z.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem, whose ``stepper.schedule`` is the schedule if there is
        one.
    start_plane : int
        Index of the first save plane to propagate to; above zero the run is a resume.

    Returns
    -------
    int or None
        The schedule's whole step count, or ``None`` for an adaptive or resumed run.
    """
    schedule = problem.stepper.schedule
    if schedule is None or start_plane > 0:
        return None
    return schedule.n_steps


def _refuse_a_record_that_misnames_the_basis(
    problem: KernelProblem, metadata: RunMetadata
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
    problem : solprop.translate.problem.KernelProblem
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
    problem: KernelProblem,
    spec: StatsSpec,
    start_plane: int,
    n_planes: int,
    *,
    metadata: RunMetadata,
) -> None:
    """Refuse, by name, every run this loop cannot execute.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem about to be run.
    spec : solprop.stats.StatsSpec
        The statistics asked for. A ``periodic`` subset is accepted -- the seam carries
        it, and a name no backend can evaluate between planes was refused by
        :class:`~solprop.stats.StatsSpec` when the spec was built -- but it must agree
        with the problem's own, because the channel is armed when the backend is built
        and reads the problem's spec, not this one.
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
    channel = problem.stats_spec
    if (spec.periodic, spec.period) != (channel.periodic, channel.period):
        msg = (
            f"the statistics given to this loop ask for {list(spec.periodic)} every "
            f"{spec.period} accepted steps and the translated problem asks for "
            f"{list(channel.periodic)} every {channel.period}; the between-planes "
            "channel is armed when the backend is built, from the problem's own spec, "
            "so the two would disagree -- pass the problem's spec, or translate again"
        )
        raise DriverError(msg)
    if problem.shape[0] != 1:
        msg = (
            f"this problem carries a batch axis of {problem.shape[0]} and a stored "
            "plane has no batch axis; run a batch through "
            "solprop.driver.scan.run_scan with executor=solprop.driver.executors."
            "Batch(...), which stores each member separately"
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
