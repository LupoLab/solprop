# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run a constructed backend through its own session, for the tests outside the suite.

SEP 47 phase 5 retired the native backend's step-driver calls (``set_initial``,
``propagate_to``, ``observe``, ``stats``): every run starts through
:func:`solprop.backends.sessions.start_session`, which opens the native session for a
native backend and the reference's own
:class:`~solprop.backends.host.session.HostSession` for ``numpy_ref``, so one call
serves both sides of a parity check. These helpers start that session on a backend the
test built, drive it with the conformance suite's driver
(:func:`sessions.executors.drive`) and hand back its
:class:`sessions.executors.Run`, whose fields are exported as the session holds the
state (:data:`sessions.executors.IDENTITY`), which is what the retired
``observe().require_field()`` returned.

- :func:`run` drives a backend's session from a start to the end of its itinerary,
  and :func:`final_field` returns the state it ended at; a failed run raises its
  error by default, as a service loop does;
- :func:`launched`, :func:`launch` and :func:`launch_statistics` start a session with
  its launch plane owed and read that plane: the statistics of a loaded state, which
  the retired direct ``stats`` call read;
- :func:`bare`, :func:`to_planes`, :func:`saved`, :class:`Snapshots`,
  :class:`Records`, :func:`run_error`, :func:`summary`, :func:`adams_ledger` and
  :func:`watched_run` shape a request or read a run;
- :func:`load_native`, :func:`native_trial`, :func:`native_source`,
  :func:`bind_native_monitor` and :func:`native_reading` reach a native backend's
  state (``solprop_native._core.State``) for the kernel-level checks, as the native
  session's march does; :func:`native_stage_rows` and :func:`native_rate_weights`
  pack a trial's host rows, and :func:`native_adams_start`, :func:`native_adams_rows`
  and :func:`native_exponential_trial` reach its resident Adams history, through the
  private test module (``solprop_native._core._testing``).

``native/tests/session_runs.py`` is the native test root's copy of the same names.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, cast

import numpy as np

from sessions.executors import IDENTITY, Plane, Run, Started, drive
from solprop.backends.events import EventBatch, ObservationReady, Terminal
from solprop.backends.export import ExportRecipe
from solprop.backends.ledger import CallLedger
from solprop.backends.protocol import StepSummary
from solprop.backends.registry import select
from solprop.backends.session import ExecutionOptions, RunSession, Start
from solprop.backends.sessions import start_session
from solprop.boundaries import SPLIT_RATE_BUCKETS
from solprop.driver import Simulation
from solprop.monitor import BoundaryEnergies, BoundaryMonitor
from solprop.stats import StatsSpec
from solprop.translate.linop import rate_weights
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

__all__ = [
    "IDENTITY",
    "Records",
    "Snapshots",
    "adams_ledger",
    "bare",
    "bind_native_monitor",
    "final_field",
    "launch",
    "launch_statistics",
    "launched",
    "load_native",
    "native_adams_rows",
    "native_adams_start",
    "native_exponential_trial",
    "native_rate_weights",
    "native_reading",
    "native_source",
    "native_stage_rows",
    "native_trial",
    "run",
    "run_error",
    "saved",
    "summary",
    "to_planes",
    "watched_run",
]


def bare(request: RunRequest) -> RunRequest:
    """Return a request that observes no statistics and watches no boundary.

    What the step driver's direct calls ran: a march between planes, with no
    statistics evaluated and no monitor attached.

    Parameters
    ----------
    request : solprop.translate.run_request.RunRequest
        The run's request.

    Returns
    -------
    solprop.translate.run_request.RunRequest
        The same method, control, itinerary and planes, without statistics or monitor.
    """
    return request.for_run(stats=StatsSpec())


def to_planes(request: RunRequest, planes: Sequence[float]) -> RunRequest:
    """Return a request whose run observes `planes` and ends at the last of them.

    Parameters
    ----------
    request : solprop.translate.run_request.RunRequest
        The run's request.
    planes : sequence of float
        The save planes, m, increasing.

    Returns
    -------
    solprop.translate.run_request.RunRequest
        The request with these planes, its other policies kept.
    """
    stepper = dataclasses.replace(
        request.stepper, save_planes=np.asarray(planes, dtype=np.float64)
    )
    return request.with_stepper(stepper)


def run(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest | None = None,
    start: Start | None = None,
    *,
    owns: bool = False,
    record: dict[str, object] | None = None,
    monitor: dict[str, object] | None = None,
    raises: bool = True,
    export: ExportRecipe | None = IDENTITY,
    stop: object | None = None,
    observations: int | None = None,
    until: float | None = None,
    between: Callable[[RunSession, EventBatch], None] | None = None,
) -> Run:
    """Drive a backend's session from a start to the end of its itinerary.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The constructed backend; it stays the caller's unless `owns`.
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest or None, optional
        The run's request; the backend's own by default.
    start : solprop.backends.session.Start or None, optional
        Where the run starts; the problem's entrance by default.
    owns : bool, optional
        Whether the session closes the backend when it closes.
    record : dict or None, optional
        Updated with the session's execution record
        (:meth:`~solprop.backends.session.RunSession.execution_record`) before it
        closes.
    monitor : dict or None, optional
        Updated with the session's boundary monitor record
        (:meth:`~solprop.backends.session.RunSession.monitor_record`) before it
        closes; a run without a monitor adds nothing.
    raises : bool, optional
        Raise a failed run's terminal error, as a service loop does; ``False`` returns
        the run with its terminal failure.
    export : solprop.backends.export.ExportRecipe or None, optional
        The recipe every observation's field is exported with: the state as the
        session holds it by default, as the step driver's observation read it;
        ``None`` exports nothing.
    stop : object or None, optional
        The run's stop token.
    observations : int or None, optional
        Close the session once this many observations were reported, the launch
        plane's included, instead of driving it to its end.
    until : float or None, optional
        Close the session after the first observation at this position, m.
    between : callable or None, optional
        Called with the session and each batch before it is acknowledged.

    Returns
    -------
    sessions.executors.Run
        What the session reported; the session is closed.
    """
    session = start_session(
        backend,
        problem,
        backend.request if request is None else request,
        Start.entrance(problem) if start is None else start,
        owns=owns,
        stop=stop,  # pyright: ignore[reportArgumentType]
    )
    try:
        result = drive(
            Started(session),
            export=export,
            between=between,
            observations=observations,
            until=until,
            raises=False,
        )
        if record is not None:
            record.update(session.execution_record())
        if monitor is not None:
            monitor.update(session.monitor_record() or {})
    finally:
        session.close()
    if raises and result.batches and result.batches[-1].terminal is not None:
        error = result.terminal.error
        if error is not None:
            raise error
    return result


def final_field(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest | None = None,
    start: Start | None = None,
    *,
    z: float | None = None,
    ended: bool = False,
    owns: bool = False,
    **options: Any,
) -> np.ndarray:
    """Return the state at the last observation of a run, as a host copy.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The constructed backend.
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest or None, optional
        The run's request; the backend's own by default.
    start : solprop.backends.session.Start or None, optional
        Where the run starts; the problem's entrance by default.
    z : float or None, optional
        Where the last observation must be, exactly, m; unchecked by default.
    ended : bool, optional
        Check that the last observation is the request's end, to within the
        rounding of a replayed schedule's sum (relative ``1e-12``).
    owns : bool, optional
        Whether the session closes the backend when it closes.
    **options
        Passed to :func:`run`.

    Returns
    -------
    numpy.ndarray
        The field, ``(n_modes, n_pol, n_omega)``, in the FFT convention.
    """
    own: RunRequest = backend.request if request is None else request
    result = run(backend, problem, own, start, owns=owns, **options)
    last = result.observations[-1]
    if z is not None:
        assert last.z == z, (last.z, z)
    if ended:
        # A replay lands on its schedule's last edge, the steps' sum, which may differ
        # from the end by the sum's rounding.
        end = float(own.stepper.z_end)
        assert math.isclose(last.z, end, rel_tol=1e-12), (last.z, end)
    return result.fields[-1]


def _launch_start(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest | None,
    field: np.ndarray | None,
    stats: StatsSpec | None,
    z: float | None,
) -> tuple[RunRequest, Start]:
    """Return the request and the start of a session whose launch plane is owed.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The constructed backend.
    problem : solprop.translate.problem.PropagationProblem
        Its problem.
    request : solprop.translate.run_request.RunRequest or None
        The run's request; the backend's own by default.
    field : numpy.ndarray or None
        The state to start from; the problem's launch by default.
    stats : solprop.stats.StatsSpec or None
        The statistics to observe in place of the request's: the session then starts
        at `z` at the first save plane strictly beyond it, which the request starts
        at too.
    z : float or None
        The state's position with `stats`, m; the problem's entrance by default.

    Returns
    -------
    tuple
        The request and the start.
    """
    own: RunRequest = backend.request if request is None else request
    launch = problem.initial.field0 if field is None else field
    if stats is None:
        return own, Start(field=np.array(launch), z=float(problem.initial.z))
    position = float(problem.initial.z) if z is None else z
    planes = np.asarray(own.stepper.save_planes, dtype=float)
    index = int(np.searchsorted(planes, position, side="right"))
    return (
        own.for_run(stats=stats, start_plane=index),
        Start(field=launch, z=position, plane_index=index),
    )


def _launch_observation(session: RunSession) -> ObservationReady:
    """Advance `session` to its first observation, the launch plane, unacknowledged.

    Parameters
    ----------
    session : solprop.backends.session.RunSession
        A session started with its launch plane owed.

    Returns
    -------
    solprop.backends.events.ObservationReady
        The launch plane's observation, its batch still pending.

    Raises
    ------
    BaseException
        The session's own failure, when it fails before observing.
    """
    while not session.finished:
        batch = session.advance(work_limit=1)
        for event in batch.events:
            if isinstance(event, Terminal) and event.error is not None:
                raise event.error
            if isinstance(event, ObservationReady):
                return event
        session.acknowledge(batch.receipt)
    msg = "the session observed no launch plane"
    raise AssertionError(msg)


@contextmanager
def launched(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest | None = None,
    start: Start | None = None,
    *,
    field: np.ndarray | None = None,
    stats: StatsSpec | None = None,
    z: float | None = None,
    owns: bool = False,
) -> Generator[tuple[RunSession, ObservationReady]]:
    """Yield `backend`'s own session and the observation of its launch plane.

    The session starts with the launch plane owed and is closed on exit; its
    observation is pending, so its ticket exports the loaded state as the session
    holds it.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The constructed backend; it stays the caller's.
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest or None, optional
        The run's request, whose statistics the launch plane carries; the backend's
        own by default.
    start : solprop.backends.session.Start or None, optional
        The start itself, in place of `field`, `stats` and `z`: a resumed plane's, or
        one with noise.
    field : numpy.ndarray or None, optional
        The state to load at the problem's entrance; the problem's launch by default.
    stats : solprop.stats.StatsSpec or None, optional
        The statistics to observe in place of the request's, of the state at `z`
        (:func:`_launch_start`).
    z : float or None, optional
        The state's position with `stats`, m.
    owns : bool, optional
        Whether the session closes the backend when it closes.

    Yields
    ------
    tuple
        The session and its launch observation.
    """
    if start is None:
        own, start = _launch_start(backend, problem, request, field, stats, z)
    else:
        own = backend.request if request is None else request
    session = start_session(backend, problem, own, start, owns=owns)
    try:
        yield session, _launch_observation(session)
    finally:
        session.close()


def launch(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest | None = None,
    start: Start | None = None,
    *,
    field: np.ndarray | None = None,
    stats: StatsSpec | None = None,
    z: float | None = None,
    owns: bool = False,
) -> Plane:
    """Return the launch plane `backend`'s own session observes, its field exported.

    The load crosses the state into the session, and the export of the launch plane
    back out, in the state's own convention: what the retired load and
    ``observe().require_field()`` did. Nothing is stepped.

    Parameters
    ----------
    backend, problem, request, start, field, stats, z, owns
        As :func:`launched` takes them.

    Returns
    -------
    sessions.executors.Plane
        The launch plane: its observation, its field copied, and the attempts before
        it (none).
    """
    with launched(
        backend, problem, request, start, field=field, stats=stats, z=z, owns=owns
    ) as (
        session,
        observation,
    ):
        held = np.array(session.export(observation.ticket, IDENTITY))
        return Plane(observation, held, observation.history, observation.step_stats)


def launch_statistics(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest | None = None,
    start: Start | None = None,
    *,
    field: np.ndarray | None = None,
    stats: StatsSpec | None = None,
    z: float | None = None,
    owns: bool = False,
) -> dict[str, np.ndarray]:
    """Return the statistics `backend`'s own session observes at its launch plane.

    Nothing is exported and nothing is stepped.

    Parameters
    ----------
    backend, problem, request, start, field, stats, z, owns
        As :func:`launched` takes them.

    Returns
    -------
    dict
        The statistics of the loaded state, copied.
    """
    with launched(
        backend, problem, request, start, field=field, stats=stats, z=z, owns=owns
    ) as (
        _,
        observation,
    ):
        return {name: np.array(value) for name, value in observation.stats.items()}


def saved(result: Run) -> list[tuple[ObservationReady, np.ndarray]]:
    """Return a run's save-plane observations with their fields, the launch left out.

    Parameters
    ----------
    result : sessions.executors.Run
        The run, driven with an export recipe.

    Returns
    -------
    list
        ``(observation, field)`` for every observation of a save plane, in order.
    """
    return [
        (event, field)
        for event, field in zip(result.observations, result.fields, strict=True)
        if event.plane_index >= 0
    ]


class Snapshots:
    """Take the method history at every observation a session makes.

    Passed as :func:`run`'s ``between``; :attr:`taken` holds
    ``(observation, snapshot)`` in order, the snapshot ``None`` where the march holds no
    history there (a Runge-Kutta march, the launch, a dense plane).
    """

    def __init__(self) -> None:
        self.taken: list[tuple[ObservationReady, Any]] = []

    def __call__(self, session: RunSession, batch: EventBatch) -> None:
        """Take each observation's snapshot while its batch is pending.

        Parameters
        ----------
        session : solprop.backends.session.RunSession
            The session.
        batch : solprop.backends.events.EventBatch
            Its pending batch.
        """
        for event in batch.events:
            if isinstance(event, ObservationReady):
                self.taken.append((event, session.march_snapshot(event.ticket)))

    def at_plane(self, index: int) -> Any:
        """Return the snapshot at save plane `index`.

        Parameters
        ----------
        index : int
            The save plane's index into the request's planes.

        Returns
        -------
        solprop.output.records.MarchSnapshot or None
            Its snapshot.
        """
        [found] = [
            snapshot for event, snapshot in self.taken if event.plane_index == index
        ]
        return found


class Records:
    """Take the session's execution record at every observation it makes.

    Passed as :func:`run`'s ``between``; :attr:`taken` holds
    ``(observation, record)`` in order, the launch plane's first when it was owed.
    """

    def __init__(self) -> None:
        self.taken: list[tuple[ObservationReady, dict[str, Any]]] = []

    def __call__(self, session: RunSession, batch: EventBatch) -> None:
        """Take the record at each observation while its batch is pending.

        Parameters
        ----------
        session : solprop.backends.session.RunSession
            The session.
        batch : solprop.backends.events.EventBatch
            Its pending batch.
        """
        for event in batch.events:
            if isinstance(event, ObservationReady):
                self.taken.append((event, dict(session.execution_record())))


def run_error(
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest,
    start: Start | None = None,
    *,
    owns: bool = False,
) -> BaseException | None:
    """Return why a run of `backend` failed: refused at its preparation or start, or
    its session's terminal error; ``None`` for a run that completed.

    The native backend refuses some runs by name before any session starts, where the
    reference's session starts and fails at its first evaluation; either is the run's
    failure.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The backend, constructed for `problem`.
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    request : solprop.translate.run_request.RunRequest
        The run's request.
    start : solprop.backends.session.Start or None, optional
        Where the run starts; the problem's entrance by default.
    owns : bool, optional
        Whether the session closes the backend when it closes.

    Returns
    -------
    BaseException or None
        The failure.
    """
    try:
        result = run(
            backend, problem, request, start, export=None, owns=owns, raises=False
        )
    except Exception as error:
        return error
    return result.terminal.error


def summary(result: Run) -> StepSummary:
    """Return a run's summary from its launch: every plane's attempts together.

    What the step driver's one march to the last plane summarised.

    Parameters
    ----------
    result : sessions.executors.Run
        The run.

    Returns
    -------
    solprop.backends.protocol.StepSummary
        Its attempts, rejections, norms and tolerance, at its last position.
    """
    attempts = result.joined[0]
    norms = np.asarray(attempts.error_norm)[np.asarray(attempts.accepted, dtype=bool)]
    planes = [event.summary for event in result.observations]
    return StepSummary(
        z=float(result.status.z),
        n_steps=attempts.n_steps,
        n_rejected=attempts.n_rejected,
        max_norm=float(np.max(norms, initial=0.0)),
        mean_norm=float(np.mean(norms)) if norms.size else 0.0,
        wall_time=sum(float(one.wall_time) for one in planes),
        met_tolerance=all(one.met_tolerance for one in planes),
    )


def adams_ledger(backend: Any, record: Mapping[str, object]) -> CallLedger:
    """Return a run's Adams call ledger.

    The reference's is its own march's, which its session steps; the native
    session's is in its execution record (``response_calls``).

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The backend the run was driven on, still open.
    record : collections.abc.Mapping
        The run's execution record (:func:`run`'s `record`).

    Returns
    -------
    solprop.backends.numpy_ref.multistep.CallLedger
        The ledger.
    """
    if backend.name == "numpy_ref":
        return backend._adams.ledger
    return CallLedger.from_record(
        cast("Mapping[str, object]", record["response_calls"])
    )


def watched_run(
    simulation: Simulation,
    name: str,
    problem: PropagationProblem | None = None,
    options: ExecutionOptions | None = None,
    run_request: RunRequest | None = None,
    *,
    launch: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object], dict[str, int]]:
    """Run a case's session watched by its boundary monitor; return what it left.

    The session form of ``monitor_cases.watched`` and a march to the end: the backend
    is selected as a simulation selects it, and the run's request carries the case's
    own monitor (:meth:`~solprop.translate.run_request.RunRequest.for_run`).

    Parameters
    ----------
    simulation : solprop.driver.Simulation
        The case.
    name : str
        The backend's registry name.
    problem : solprop.translate.problem.PropagationProblem or None, optional
        The problem; the simulation's own translation by default.
    options : solprop.backends.session.ExecutionOptions or None, optional
        The execution options; the simulation's own by default.
    run_request : solprop.translate.run_request.RunRequest or None, optional
        The request; the simulation's own by default.
    launch : numpy.ndarray or None, optional
        The launch state; the problem's own by default.

    Returns
    -------
    tuple
        The final state (internal convention), the monitor's record
        (:meth:`solprop.monitor.BoundaryMonitor.summary`'s layout) and the backend's
        memory ledger after the run.
    """
    problem = simulation.translate() if problem is None else problem
    run_request = simulation._run_request() if run_request is None else run_request
    backend = select(
        name,
        problem,
        run_request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
        options=simulation.settings.execution_options() if options is None else options,
    )
    try:
        boundary = simulation._boundary_monitor()
        assert boundary is not None
        field = problem.initial.field0 if launch is None else launch
        record: dict[str, object] = {}
        result = run(
            backend,
            problem,
            run_request.for_run(monitor=boundary),
            Start(field=field, z=float(problem.initial.z)),
            monitor=record,
        )
        return result.fields[-1], record, backend.memory_ledger()
    finally:
        backend.close()


def load_native(
    backend: Any,
    field: np.ndarray,
    noise: np.ndarray | None = None,
) -> Any:
    """Load a host state into a native backend's workspace, as a run's start does.

    The kernel-level tests call the workspace's state (``solprop_native._core.
    _testing``) directly, as the session's march does: the field and noise go in, the
    collar back into the stages, the recycled stage dropped and no monitor bound
    (``Workspace::load``).

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend.
    field : numpy.ndarray
        The state, internal amplitude units and FFT convention, shape
        ``(n_modes, n_pol, n_omega)``.
    noise : numpy.ndarray or None, optional
        The ``modified`` noise model's spectrum the stages add, in the state's layout
        (:func:`solprop.translate.masks.noise_spectrum`), or ``None``.

    Returns
    -------
    solprop_native._core.Workspace
        The backend's workspace, loaded.
    """
    from solprop_native._core import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
        _testing,
    )

    workspace = backend._workspace
    values = np.ascontiguousarray(field, dtype=np.complex128).reshape(-1)
    added = (
        np.empty(0, dtype=np.complex128)
        if noise is None
        else np.ascontiguousarray(noise, dtype=np.complex128).reshape(-1)
    )
    _testing.load(workspace, values, added)
    return workspace


def native_trial(
    backend: Any, z: float, dz: float, *, field: np.ndarray | None = None
) -> np.ndarray:
    """Take one Runge-Kutta trial on a native backend's stage tables; its partials.

    The trial is the workspace's own (``_testing.rk_trial``), as the native march
    takes it, at the nodes of the backend's own request's tableau; its stages stay in
    the state for ``stage_to_host``.

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend, its workspace loaded (:func:`load_native`).
    z : float
        Where the trial starts, m.
    dz : float
        Its length, m.
    field : numpy.ndarray or None, optional
        A state to load first (:func:`load_native`); the state as it is by default.

    Returns
    -------
    numpy.ndarray
        The trial's scored partials, float64.
    """
    from solprop_native._core import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
        _testing,
    )
    from solprop_native.backend import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _envelope_errors,
    )

    if field is not None:
        load_native(backend, field)
    with _envelope_errors():
        partials = _testing.rk_trial(backend._workspace, z, dz)
    return np.array(partials, dtype=np.float64)


def native_source(backend: Any, field: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Evaluate a native backend's source once at a field: its whole stage source.

    A zero-length Runge-Kutta trial at `z` (:func:`native_trial`), whose every stage
    reads the loaded field itself where the flow over no distance is exactly one: its
    first stage is the route's source untransported, the sample the reference's
    ``_adams_lab_source`` evaluates. The native state keeps the evaluation's projected
    spectra too (``projected_to_host``).

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend.
    field : numpy.ndarray
        The lab-frame field, shape ``(n_modes, n_pol, n_omega)``.
    z : float, optional
        The evaluation position, m.

    Returns
    -------
    numpy.ndarray
        The source, shape ``(n_modes, n_pol, n_omega)`` as the reference's.
    """
    from solprop_native._core import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
        _testing,
    )

    workspace = load_native(backend, field)
    native_trial(backend, z, 0.0)
    return np.array(_testing.stage_to_host(workspace, 0)).reshape(np.shape(field))


def bind_native_monitor(
    backend: Any, problem: PropagationProblem, monitor: BoundaryMonitor
) -> BoundaryMonitor:
    """Hold a boundary monitor's weights and walk-off buckets in a native workspace.

    What a native run's start binds for its monitor (the backend's
    ``_monitor_binding``, the split collar's crossing the problem's), through
    ``_testing.bind_monitor``, for the kernel-level readings of :func:`native_reading`.

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend.
    problem : solprop.translate.problem.PropagationProblem
        Its problem.
    monitor : solprop.monitor.BoundaryMonitor
        A fresh monitor.

    Returns
    -------
    solprop.monitor.BoundaryMonitor
        The monitor as bound, for :func:`native_reading`.
    """
    from solprop_native._core import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
        _testing,
    )

    monitor = dataclasses.replace(
        monitor, crossing=problem.split_crossing if monitor.split else None
    )
    occupancy = (
        None
        if monitor.crossing is None
        else {
            "buckets": np.ascontiguousarray(
                monitor.crossing.buckets, dtype=np.uint8
            ).reshape(-1),
            "count": SPLIT_RATE_BUCKETS,
        }
    )
    _testing.bind_monitor(
        backend._workspace, backend._monitor_binding(monitor), None, occupancy
    )
    return monitor


def native_reading(
    backend: Any, monitor: BoundaryMonitor, field: np.ndarray | None = None
) -> BoundaryEnergies:
    """Return a native state's boundary reading, as the native session reduces it.

    The five numbers, the split's removal since the last reading, what the last
    step's split found and the walk-off occupancy, from the state's own reduction
    (``monitor_reading``), laid out as the backend's reading lays them out.

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The backend, its monitor bound (:func:`bind_native_monitor`).
    monitor : solprop.monitor.BoundaryMonitor
        The monitor as bound (:func:`bind_native_monitor`).
    field : numpy.ndarray or None, optional
        A state to load first (:func:`load_native`, which unbinds the monitor:
        bind it again after); the state as it stands by default.

    Returns
    -------
    solprop.monitor.BoundaryEnergies
        The reading.
    """
    from solprop_native._core import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
        _testing,
    )

    if field is not None:
        load_native(backend, field)
    reading = np.asarray(_testing.monitor_reading(backend._workspace), dtype=np.float64)
    reading = reading.reshape(-1)
    found = reading[6:10].copy() if monitor.split else None
    occupancy = None
    if monitor.crossing is not None:
        occupancy = reading[10 : 10 + SPLIT_RATE_BUCKETS].copy()
    return BoundaryEnergies(
        float(reading[0]),
        reading[1:5].copy(),
        removed=float(reading[5]),
        found=found,
        occupancy=occupancy,
    )


def native_stage_rows(
    backend: Any, anchor: float, position: float
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Return the host-packed rows of a trial whose every stage reads `position`.

    The arrays a resident Adams trial took before the native march formed its rows
    from the stage tables (SEP 39): the bundle at `position`, anchored at `anchor`,
    repeated over the tableau's stages, packed as the backend packed them, with a
    coupled flow's section and a frozen-residual flow's weights. The tests hold the
    native evaluation to them. A z table's rows are its native slots, which no host
    bundle holds: such a problem is refused here.

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend.
    anchor : float
        The step's anchor, m.
    position : float
        Where every stage reads its scalars, m.

    Returns
    -------
    tuple
        The seven row arrays (``phi_delta`` to ``raman_rates``) and the material
        record.
    """
    tables = backend._z_tables
    if tables is not None and (tables.phi or tables.prefactor):
        msg = "a z table's stage rows are its native slots, not a host bundle's"
        raise NotImplementedError(msg)
    bundle = backend._stage_scalars(anchor, position)
    stages = backend.request.stepper.tableau.n_stages
    bundles = [bundle] * stages
    if backend._terms.separable is not None:
        phi_delta = np.array([[one.z1 - one.z0] for one in [*bundles, bundle]])
        phi_at = np.array([[one.z1] for one in bundles])
        prefactor = np.ones((stages, 1))
    else:
        phi_delta = np.stack([one.phi_delta for one in [*bundles, bundle]])
        phi_at = np.stack([one.phi_at for one in bundles])
        prefactor = np.stack([one.prefactor for one in bundles])
    arrays = tuple(
        np.ascontiguousarray(values, dtype=np.float64)
        for values in (
            phi_delta,
            phi_at,
            prefactor,
            [one.density for one in bundles],
            [one.taper_scale for one in bundles],
            [one.preionisation for one in bundles],
            [one.raman_rates for one in bundles],
        )
    )
    material: dict[str, Any] = {
        "kerr_density": np.asarray([one.kerr_density for one in bundles]),
        "plasma_densities": np.asarray([one.plasma_densities for one in bundles]),
        "plasma_preionisation": np.asarray(
            [one.plasma_preionisation for one in bundles]
        ),
        "raman_densities": np.asarray([one.raman_densities for one in bundles]),
    }
    if backend._problem.responses.dispersive_kerr is not None:
        material["dispersive_densities"] = np.ascontiguousarray(
            [one.dispersive_kerr_densities for one in bundles], dtype=np.float64
        )
    if backend._moving_maps:
        material["transform_weights"] = np.ascontiguousarray(
            np.stack([one.transform_weights for one in bundles]), dtype=np.float64
        )
    flow = backend._flow
    if flow is not None:
        material |= {
            "flow_section": bundle.flow_section,
            "flow_weights": np.empty(0, dtype=np.float64)
            if flow.residual_terms is None
            else np.ascontiguousarray(
                np.tile(bundle.flow_weights, stages), dtype=np.float64
            ),
        }
    return arrays, material


def native_rate_weights(backend: Any, z: float, *, before: bool = False) -> np.ndarray:
    """Return ``D(z)``'s weights over a native backend's ``Phi`` terms (SEP 16).

    solprop's :func:`~solprop.translate.linop.rate_weights` for a constant or an
    expansion; for a z table, the rate table evaluated natively into its slot
    (``table_rate_at``, SEP 29 G1), one at that slot.

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend.
    z : float
        Position, m.
    before : bool, optional
        Read an interior breakpoint from the interval ending there.

    Returns
    -------
    numpy.ndarray
        The weights, float64.
    """
    from solprop_native._core import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
        _testing,
    )

    tables = backend._z_tables
    if tables is not None and tables.phi:
        _testing.table_rate_at(backend._workspace, float(z), before)
        rate = np.zeros(tables.n_phi)
        rate[tables.rate_slot] = 1.0
        return rate
    linear = backend._problem.linear
    weights = rate_weights(linear, z, before=before, helpers=backend.helpers)
    return np.ascontiguousarray(weights, dtype=np.float64)


def _native_compensated(backend: Any) -> bool:
    """Whether a native backend's resident Adams history carries a compensation."""
    method = backend.request.stepper.integrator
    return bool(backend._compensated(method, "transported"))


def native_adams_start(backend: Any, z: float) -> None:
    """Evaluate a native backend's first Adams sample at its loaded state, into row 0.

    The native march's start-up (``engines::ResidentEngine``) taken alone, for the
    kernel-level checks that read one source sample: every stage row at `z`
    (:func:`native_stage_rows`) and, for a compensated history, the state and rate
    read on the far side of a kink (:func:`native_rate_weights`).

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend, under an Adams method, its workspace loaded
        (:func:`load_native`).
    z : float
        The state's position, m.
    """
    from solprop_native._core import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
        _testing,
    )
    from solprop_native.backend import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _envelope_errors,
    )

    workspace = backend._workspace
    backend._stages.hoist(z)
    arrays, material = native_stage_rows(backend, z, z)
    with _envelope_errors():
        _testing.adams_start(workspace, 0, *arrays, material=material)
    if _native_compensated(backend):
        _testing.adams_compensation_start(workspace, 0, native_rate_weights(backend, z))


def native_adams_rows(backend: Any) -> dict[str, np.ndarray]:
    """Return host copies of a native backend's Adams history rows, losslessly.

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend, under an Adams method.

    Returns
    -------
    dict of str to numpy.ndarray
        Each row the engine keeps, ``(n_slots, n_modes, n_pol, n_omega)``, as a
        snapshot holds it.
    """
    from solprop_native._core import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
        _testing,
    )

    workspace = backend._workspace
    expected = (backend._adams.n_slots, *backend._problem.shape)
    return {
        name: _testing.adams_rows_to_host(workspace, name).reshape(expected)
        for name in backend._diagnostics()["history_rows"]
    }


def native_exponential_trial(
    backend: Any,
    z: float,
    dz: float,
    nodes: Sequence[float],
    slots: np.ndarray,
    order: int,
    orders: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Take one exponential Adams trial on a native backend's resident history.

    The native march's trial (``engines::ResidentEngine``) taken alone: the tables, the
    stage scalars and the rate formed natively from the nodes, under the run's carrier
    frame, whose packed shift and rotations are formed here as the march forms them.

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend, under the exponential formulation.
    z : float
        The step's anchor, m.
    dz : float
        The step, m.
    nodes : sequence of float
        The history's nodes ``theta``, newest first.
    slots : numpy.ndarray
        The history rows the nodes name.
    order : int
        The corrector's order.
    orders : sequence of int
        The orders whose defects the trial scores.

    Returns
    -------
    tuple of numpy.ndarray
        The scored norms, and the packed carrier frame, empty without one.
    """
    from solprop_native._core import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
        _testing,
    )
    from solprop_native.backend import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _envelope_errors,
    )

    rows = max(orders)
    carrier = np.empty(0, dtype=np.float64)
    reference = backend._carrier
    if reference is not None and reference.active:
        # (Im sigma, Re f_0, Im f_0, ...), f_t = exp(sigma (1 - theta_t)), over the
        # rows read; sigma is the reference element's chord over the step (SEP 35).
        shift = float(np.imag(reference.chord.exponent(z, z + dz)))
        spans = 1.0 - np.asarray(nodes[:rows], dtype=np.float64)
        carrier = np.empty(1 + 2 * rows, dtype=np.float64)
        carrier[0] = shift
        carrier[1::2] = np.cos(shift * spans)
        carrier[2::2] = np.sin(shift * spans)
    with _envelope_errors():
        norms = _testing.adams_exponential_trial_at(
            backend._workspace,
            z,
            dz,
            np.asarray(nodes, dtype=np.float64),
            np.ascontiguousarray(slots[: max(rows, order)], dtype=np.uint64),
            order,
            list(orders),
            _native_compensated(backend),
            carrier,
        )
    return np.asarray(norms, dtype=np.float64).reshape(-1), carrier
