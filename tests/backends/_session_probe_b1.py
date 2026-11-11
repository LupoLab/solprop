# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run a constructed backend through its session, for the backend tests (SEP 47).

The backend tests once loaded a backend and marched it plane by plane through the
step driver's own calls. A native run is a session now (SEP 47 phase 5), so these
start the backend's session (:func:`~solprop.backends.sessions.start_session`) on a
backend the test keeps, drive it to its end as the conformance suite's driver does
(:func:`sessions.executors.drive`) and hand back what the planes reported, as its
:class:`sessions.executors.Run`: the same
helpers serve the reference, whose session is a
:class:`~solprop.backends.host.session.HostSession`.

- :func:`run` drives a backend's session from a start to the end of its itinerary;
- :func:`last_field`, :func:`plane_fields`, :func:`rows`, :func:`history` and
  :func:`summary` read a run's final state, its planes, its periodic rows, its
  attempts and its summary, and :func:`adams_ledger` its Adams call ledger;
- :func:`state_stats` evaluates statistics of a given state, as the session's
  launch plane reports them;
- :func:`watched_run` runs a case watched by its boundary monitor, and
  :func:`bind_native_monitor` and :func:`native_reading` read a native state's
  boundary as the native session does;
- :func:`launched_state`, :func:`rk_trial` and :func:`native_source` load a native
  backend's native state, take one trial on it and evaluate its source once, for the
  kernel-level checks.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np

from sessions.executors import IDENTITY, Run
from solprop.backends.events import (
    EventBatch,
    ObservationKind,
    ObservationReady,
    ServiceKind,
    ServiceRequired,
    emit_warning,
    join_attempts,
)
from solprop.backends.export import ExportRecipe
from solprop.backends.multistep import CallLedger
from solprop.backends.protocol import BackendHistory, StepStats, StepSummary
from solprop.backends.registry import select
from solprop.backends.session import ExecutionOptions, RunSession, Start
from solprop.backends.sessions import start_session
from solprop.boundaries import SPLIT_RATE_BUCKETS
from solprop.driver import Simulation
from solprop.monitor import BoundaryEnergies, BoundaryMonitor
from solprop.stats import StatsSpec
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest


def run(  # noqa: PLR0913 - the session's context and the drive's options
    backend: Any,
    problem: PropagationProblem,
    request: RunRequest | None = None,
    start: Start | None = None,
    *,
    export: ExportRecipe | None = IDENTITY,
    owns: bool = False,
    stop: object | None = None,
    observations: int | None = None,
    record: dict[str, object] | None = None,
    monitor: dict[str, object] | None = None,
    between: Callable[[RunSession, EventBatch], None] | None = None,
    raises: bool = True,
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
    export : solprop.backends.export.ExportRecipe or None, optional
        The recipe every observation's field is exported with: the state as the
        session holds it by default, as the step driver's observation read it.
    owns : bool, optional
        Whether the session closes the backend.
    stop : object or None, optional
        The run's stop token.
    observations : int or None, optional
        Close the session once this many observations were reported, the launch
        plane's included, instead of driving it to its end.
    record : dict or None, optional
        Updated with the session's execution record
        (:meth:`~solprop.backends.session.RunSession.execution_record`) before it
        closes.
    monitor : dict or None, optional
        Updated with the session's boundary monitor record
        (:meth:`~solprop.backends.session.RunSession.monitor_record`) before it
        closes; a run without a monitor adds nothing.
    between : callable or None, optional
        Called with the session and each batch before it is acknowledged.
    raises : bool, optional
        Raise a failed run's error, as a service loop does; ``False`` returns the run
        with its terminal failure.

    Returns
    -------
    sessions.executors.Run
        What the session reported.
    """
    session = start_session(
        backend,
        problem,
        backend.request if request is None else request,
        Start.entrance(problem) if start is None else start,
        owns=owns,
        stop=stop,  # pyright: ignore[reportArgumentType]
    )
    batches: list[EventBatch] = []
    fields: list[np.ndarray] = []
    seen = 0
    try:
        while not session.finished:
            batch = session.advance()
            batches.append(batch)
            for event in batch.events:
                if isinstance(event, ServiceRequired) and (
                    event.kind is ServiceKind.WARNING
                ):
                    # A warning barrier: emitted as a service loop emits it.
                    emit_warning(event)
                if isinstance(event, ObservationReady):
                    seen += 1
                    if export is not None:
                        fields.append(np.array(session.export(event.ticket, export)))
            if between is not None:
                between(session, batch)
            session.acknowledge(batch.receipt)
            if observations is not None and seen >= observations:
                break
        if record is not None:
            record.update(session.execution_record())
        if monitor is not None:
            monitor.update(session.monitor_record() or {})
        result = Run(batches, fields, session.status())
        if raises and result.batches[-1].terminal is not None:
            error = result.terminal.error
            if error is not None:
                raise error
        return result
    finally:
        session.close()


def last_field(result: Run) -> np.ndarray:
    """Return the field of a run's last observation.

    Parameters
    ----------
    result : sessions.executors.Run
        A run driven with an export recipe.

    Returns
    -------
    numpy.ndarray
        The field, shape ``(n_modes, n_pol, n_omega)``.
    """
    return result.fields[-1]


def history(result: Run) -> BackendHistory:
    """Return every attempt of a run, assembled in order.

    Parameters
    ----------
    result : sessions.executors.Run
        The run.

    Returns
    -------
    solprop.backends.protocol.BackendHistory
        The attempts.
    """
    return join_attempts(result.chunks)[0]


def rows(result: Run) -> StepStats:
    """Return every periodic statistics row of a run, assembled in order.

    Parameters
    ----------
    result : sessions.executors.Run
        The run.

    Returns
    -------
    solprop.backends.protocol.StepStats
        The rows, without a batch axis.
    """
    return join_attempts(result.chunks)[1]


def state_stats(
    backend: Any,
    problem: PropagationProblem,
    field: np.ndarray,
    spec: StatsSpec,
    *,
    z: float | None = None,
    plane_index: int | None = None,
) -> dict[str, np.ndarray]:
    """Return the statistics of a state, as a session's launch plane reports them.

    The session starts at the state with its plane owed, and stops after that plane's
    observation: nothing is propagated.

    Parameters
    ----------
    backend : solprop.backends.protocol.Backend
        The constructed backend; it stays the caller's.
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    field : numpy.ndarray
        The state, internal amplitude units and FFT convention.
    spec : solprop.stats.StatsSpec
        The statistics.
    z : float or None, optional
        Its position, m; the problem's entrance by default.
    plane_index : int or None, optional
        The first save plane strictly beyond `z`; by default the backend's request's.

    Returns
    -------
    dict
        Each statistic's value by name.
    """
    position = float(problem.initial.z) if z is None else z
    if plane_index is None:
        planes = np.asarray(backend.request.stepper.save_planes, dtype=float)
        plane_index = int(np.searchsorted(planes, position, side="right"))
    request = backend.request.for_run(stats=spec, start_plane=plane_index)
    start = Start(field=field, z=position, plane_index=plane_index)
    session = start_session(backend, problem, request, start, owns=False)
    try:
        while not session.finished:
            batch = session.advance()
            for event in batch.events:
                if isinstance(event, ObservationReady):
                    return {
                        name: np.array(value) for name, value in event.stats.items()
                    }
            if batch.terminal is not None and batch.terminal.error is not None:
                raise batch.terminal.error
            session.acknowledge(batch.receipt)
    finally:
        session.close()
    msg = "the session reported no launch plane"
    raise AssertionError(msg)


def plane_fields(result: Run) -> list[np.ndarray]:
    """Return the field of every observation of a run but its launch plane.

    Parameters
    ----------
    result : sessions.executors.Run
        A run driven with an export recipe from a fresh start.

    Returns
    -------
    list of numpy.ndarray
        The fields, in order: the save planes, and the step planes where asked.
    """
    return [
        field
        for event, field in zip(result.observations, result.fields, strict=True)
        if event.kind is not ObservationKind.LAUNCH
    ]


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
    solprop.backends.multistep.CallLedger
        The ledger.
    """
    if backend.name == "numpy_ref":
        return backend._adams.ledger
    return CallLedger.from_record(
        cast("Mapping[str, object]", record["response_calls"])
    )


def launched_state(
    backend: Any, field: np.ndarray, noise: np.ndarray | None = None
) -> Any:
    """Return a native backend's native state, loaded with a field.

    For the kernel-level checks that call the state's own bindings (a trial, a
    finish), which the step driver's load reached before SEP 47 phase 5.

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend.
    field : numpy.ndarray
        The state, internal amplitude units and FFT convention.
    noise : numpy.ndarray or None, optional
        The ``modified`` noise model's spectrum the stages add, in the state's layout
        (:func:`solprop.translate.masks.noise_spectrum`), or ``None``.

    Returns
    -------
    solprop_native._core.State
        The backend's native state.
    """
    state = backend._native_state
    values = np.ascontiguousarray(field, dtype=np.complex128)
    added = (
        np.empty(0, dtype=np.complex128)
        if noise is None
        else np.ascontiguousarray(noise, dtype=np.complex128).reshape(-1)
    )
    state.set_initial(values.reshape(-1), added)
    state.clear_fsal()
    return state


def rk_trial(backend: Any, z: float, dz: float) -> np.ndarray:
    """Take one Runge-Kutta trial on a native backend's stage tables; its partials.

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend, its state loaded (:func:`launched_state`).
    z : float
        Where the trial starts, m.
    dz : float
        Its length, m.

    Returns
    -------
    numpy.ndarray
        The trial's scored partials.
    """
    backend._staged_rows()
    nodes = [float(node) for node in backend.request.stepper.tableau.nodes]
    partials = backend._native_state.rk_trial(z, dz, nodes, prepare_only=False)
    return np.array(partials, dtype=np.float64)


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
        return last_field(result), record, backend.memory_ledger()
    finally:
        backend.close()


def bind_native_monitor(
    backend: Any, problem: PropagationProblem, monitor: BoundaryMonitor
) -> BoundaryMonitor:
    """Hold a boundary monitor's weights and walk-off buckets in a native state.

    What the native session binds for its monitor (the backend's ``_bind_monitor``,
    the split collar's crossing the problem's), through the state's own bindings, for
    the kernel-level readings of :func:`native_reading`.

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
    monitor = dataclasses.replace(
        monitor, crossing=problem.split_crossing if monitor.split else None
    )
    state = backend._native_state
    columns = (
        np.empty(0, dtype=np.float64)
        if monitor.columns is None
        else np.ascontiguousarray(monitor.columns, dtype=np.float64)
    )
    state.bind_monitor(
        np.ascontiguousarray(monitor.weights, dtype=np.float64).reshape(-1), columns
    )
    if monitor.crossing is not None:
        state.bind_occupancy(
            np.ascontiguousarray(monitor.crossing.buckets, dtype=np.uint8).reshape(-1),
            SPLIT_RATE_BUCKETS,
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
        A state to load first (:func:`launched_state`); the state as it stands by
        default.

    Returns
    -------
    solprop.monitor.BoundaryEnergies
        The reading.
    """
    if field is not None:
        launched_state(backend, field)
    reading = np.asarray(backend._native_state.monitor_reading(), dtype=np.float64)
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


def native_source(backend: Any, field: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Evaluate a native backend's source once at a field: its whole stage source.

    A zero-length Runge-Kutta trial at `z` (:func:`rk_trial`), whose every stage reads
    the loaded field itself where the flow over no distance is exactly one: its first
    stage is the route's source untransported, the sample the reference's
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
        The source, shape ``(1, n_modes, n_pol, n_omega)`` as the reference's.
    """
    state = launched_state(backend, field)
    rk_trial(backend, z, 0.0)
    return np.array(state.stage_to_host(0)).reshape(1, *np.shape(field))


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
    attempts = history(result)
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
