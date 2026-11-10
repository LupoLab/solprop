# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native execution session, solprop SEP 47's run owner in C++ (prototype).

:class:`NativeSession` implements solprop's
:class:`~solprop.backends.session.RunSession` over ``_core.Session``, the native
session (``native/include/solprop_native/session.hpp``). The native session is the one
mutable owner of the trajectory: the march's policy, state and history, the plane
schedule, the periodic statistics rows (taken natively after each commit and kept in a
resident buffer until the next observation), the stop flag, the snapshot latch, the
committed-state generation and the pending batch. This adapter only translates: it
prepares the session once from a :class:`~solprop_native.backend.NativeBackend`'s
lowering (the native problem, the stage tables and the CPU execution record), turns
each batch's compact records into :mod:`solprop.backends.events` types, and fills
export destinations through the native recipe. It keeps no numerical state.

This is SEP 47's phase 1 prototype, selected only by tests and by an explicit
benchmark opt-in; production runs still use
:class:`~solprop.backends.driver_session.DriverSession`. It admits a subset and refuses
every other feature by name before allocating anything (:func:`refusal`):

- the native CPU build; Adams (``Multistep``) and the embedded Runge-Kutta tableaux,
  adaptive or replayed, on the routes the resident engines march without a host
  fallback;
- every periodic-capable statistic of the catalogue on these routes, plane and
  periodic, each evaluated and finished natively from a plan lowered once
  (:mod:`solprop_native._statistics`), the profile statistics from tables whose
  measured error the execution record carries; the plane-only statistics the host
  finishes (:data:`~solprop_native._statistics.PLANE_ONLY`) are refused;
- the temporal boundary monitor, which the native session owns (SEP 47 section 6.4):
  its every decision and record are native, a warning is a
  :class:`~solprop.backends.events.ServiceRequired` barrier and a stop a terminal
  failure, and a monitored resume continues the record natively;
- no transverse edge, dense output, low-storage replay, comparison method, custom
  controller, group, noise model, TDSE, resumed run or method-history snapshot.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from solprop.backends.events import (
    Eligibility,
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
)
from solprop.backends.protocol import BackendHistory, StepStats, StepSummary
from solprop.backends.session import (
    DEFAULT_DEADLINE,
    DEFAULT_WORK_LIMIT,
    SessionError,
    SessionState,
    SessionStatus,
    SnapshotReceipt,
)
from solprop.backends.spatial import FarFieldTables
from solprop.exceptions import BackendError, GridError, StopRequested
from solprop.monitor import (
    _log_finish,  # pyright: ignore[reportPrivateUsage] - one log line
)
from solprop.output.capability import (
    declared_capability,
    refusal_for,
    withheld_accessors,
)
from solprop.stats import PLASMA_INTEGRATION_STATISTICS
from solprop.stepper.controller import ControlRecord, PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import ComparisonMultistep, Multistep
from solprop.translate.linop import power_weight_at

from . import _core, _watch
from ._statistics import (
    ADMITTED_GENERAL_STATISTICS,
    ADMITTED_SPATIAL_STATISTICS,
    ADMITTED_STATISTICS,
    StatisticsLowering,
    lower,
    translated,
    uncovered,
)
from .backend import (
    NativeBackend,
    _aperture_payload,  # pyright: ignore[reportPrivateUsage]
    _NativeRungeKuttaMarch,  # pyright: ignore[reportPrivateUsage]
    _on_device,  # pyright: ignore[reportPrivateUsage]
    native_march_refusal,
)

if TYPE_CHECKING:
    from types import TracebackType

    from solprop.backends.events import Event
    from solprop.backends.export import ExportRecipe
    from solprop.backends.session import Start
    from solprop.grid import TimeGrid
    from solprop.log import StopFlag
    from solprop.modes.protocol import Mode
    from solprop.monitor import BoundaryMonitor
    from solprop.output.schema import MarchSnapshot
    from solprop.profiles import ZProfile
    from solprop.stats import StatsSpec
    from solprop.translate.problem import KernelProblem
    from solprop.translate.run_request import RunRequest

__all__ = [
    "ADMITTED_GENERAL_STATISTICS",
    "ADMITTED_SPATIAL_STATISTICS",
    "ADMITTED_STATISTICS",
    "NativeSession",
    "refusal",
]

CALL_ATTEMPTS = 256
"""A march call's most attempts, as the driver's native marches bound them (SEP 39)."""

CALL_SECONDS = 0.1
"""A march call's most wall time, s."""

ATTEMPT_CAPACITY = 4096
"""The most attempts the native session holds between two events; a full buffer yields
a progress event carrying them (SEP 47 section 5.3)."""

ROW_CAPACITY = 1024
"""The most periodic rows it holds between two events, likewise."""

_SESSION_IDS = itertools.count(1)
_KINDS = (
    ObservationKind.LAUNCH,
    ObservationKind.PLANE,
    ObservationKind.STEP,
    ObservationKind.NATURAL,
)
_STATES = (
    SessionState.READY,
    SessionState.PENDING,
    SessionState.FINISHED,
    SessionState.CLOSED,
)
_OUTCOMES = (None, Outcome.COMPLETED, Outcome.STOPPED, Outcome.FAILED)
_OBSERVATION, _PROGRESS, _SNAPSHOT, _TERMINAL, _WARNING = range(5)
_OVERRIDES = (
    "scale",
    "accepts",
    "propose",
    "rejection_factor",
    "start_factor",
    "equivalent_norm",
)
_EMPTY = np.empty(0, dtype=np.float64)


def refusal(
    problem: KernelProblem, request: RunRequest, start: Start | None = None
) -> str | None:
    """Return why the prototype session refuses a run, naming the feature, or ``None``.

    Decided from the problem, the request and the start alone, before anything is
    allocated. A march the backend itself would step on the host is refused once the
    backend is built (:class:`NativeSession`).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    start : solprop.backends.session.Start or None, optional
        Where the run starts.

    Returns
    -------
    str or None
        The reason, or ``None`` when the session admits the run.
    """
    if _on_device():
        return "a device build: the native session prototype runs on the CPU build only"
    if problem.shape[0] != 1:
        return f"a group of {problem.shape[0]} members (one trajectory a session)"
    method = request.integration.method
    if isinstance(method, LowStorageRungeKutta):
        return "the low-storage replay"
    if isinstance(method, ComparisonMultistep):
        return f"the comparison method {method.name}"
    if method is not None and not isinstance(method, Multistep):
        return f"the method {type(method).__name__}"
    controller = request.error_control.controller
    overridden = [
        name
        for name in _OVERRIDES
        if getattr(type(controller), name) is not getattr(PIController, name)
    ]
    if overridden:
        return "a controller with its own " + ", ".join(overridden)
    if request.observation.dense:
        return "dense output (interpolated save planes)"
    monitor = request.boundary.monitor
    if monitor is not None and monitor.transverse is not None:
        return "a boundary monitor's transverse edge"
    if problem.initial.noise_model != "none":
        return f"the {problem.initial.noise_model!r} noise model"
    if problem.tdse is not None:
        return "a TDSE response"
    if start is not None:
        if start.march is not None:
            return "a method-history resume (MarchSnapshot)"
        if start.ledgers is not None:
            return "restored call ledgers"
        if start.resume:
            return "a resumed run"
        if start.noise is not None:
            return "a noise realisation"
    march = native_march_refusal(problem)
    if march is not None:
        return march
    return _statistics_refusal(problem, request.observation.stats)


def _statistics_refusal(problem: KernelProblem, spec: StatsSpec) -> str | None:
    """Return why the session refuses a run's statistics, naming them, or ``None``.

    Every statistic the native plan evaluates on the route is admitted, plane and
    periodic (:mod:`solprop_native._statistics`); the plane-only statistics the host
    finishes are refused by name, and so is what the backend itself would refuse.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    spec : solprop.stats.StatsSpec
        The plane statistics and their periodic subset.

    Returns
    -------
    str or None
        The reason, or ``None``.
    """
    names = tuple(spec.names)
    if not names:
        return None
    general_modal = problem.transform.modal_tables is not None
    refused = refusal_for(
        declared_capability(
            general_modal=general_modal,
            spatial=problem.transform.geometry is not None,
        ),
        names,
        withheld=withheld_accessors(
            general_modal=general_modal,
            coupled_flow=problem.linear.coupled_flow is not None,
        ),
    )
    if refused is not None:
        return f"statistics the backend refuses ({refused})"
    left = uncovered(problem, names)
    if left:
        return (
            f"the statistics {left!r}, which the host finishes: a plane-only "
            "statistic, a metric that varies inside an interval, or not a "
            "statistic of this route"
        )
    responses = problem.responses
    if set(names) & set(PLASMA_INTEGRATION_STATISTICS) and (
        not responses.population_channels or responses.envelope_plasma is not None
    ):
        return (
            "the plasma integration diagnostics, which read the field-grid plasma "
            "column this problem does not carry"
        )
    if (
        set(names) & {"electron_density", "peak_ionisation_rate"}
        and not responses.population_channels
    ):
        return "the plasma statistics of a problem with no compiled plasma response"
    return None


class NativeSession:
    """A :class:`~solprop.backends.session.RunSession` owned by the native backend.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's policies.
    start : solprop.backends.session.Start
        Where the run starts.
    grid : solprop.grid.TimeGrid or None, optional
        The time grid, as :class:`~solprop_native.backend.NativeBackend` takes it.
    modes : sequence of Mode or None, optional
        The mode set, likewise.
    profiles : mapping or None, optional
        The z profiles, likewise.
    stop : solprop.log.StopFlag or None, optional
        The run's stop token. A :class:`~solprop.backends.session.StopToken` sets the
        native flag from any thread; any other flag is polled before each advance.
    listener : callable or None, optional
        Called with each attempt's end, length and acceptance. A per-attempt listener is
        a requested external service: the session then yields after every attempt and
        calls it before returning the batch (SEP 47 section 7.3). It observes only.
    backend : solprop_native.backend.NativeBackend or None, optional
        A backend already built for `problem`, holding no run, to run on; by default the
        session builds its own from `grid`, `modes` and `profiles`.
    owns : bool, optional
        Whether closing the session closes `backend`: false for a runner's backend
        reused across runs. A backend the session built is always its own.

    Raises
    ------
    solprop.exceptions.BackendError
        Naming the feature the prototype refuses, before anything is allocated.
    """

    def __init__(
        self,
        problem: KernelProblem,
        request: RunRequest,
        start: Start,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        stop: StopFlag | None = None,
        listener: Callable[[float, float, bool], None] | None = None,
        backend: NativeBackend | None = None,
        owns: bool = True,
    ) -> None:
        """Refuse what the prototype does not admit, then prepare and start it."""
        refused = refusal(problem, request, start)
        if refused is not None:
            msg = f"the native session does not run {refused}"
            raise BackendError(msg)
        self._problem = problem
        self._request = request
        self._monitor: BoundaryMonitor | None = request.boundary.monitor
        self._finish_logged = False
        self._sequence = 0
        self._id = next(_SESSION_IDS)
        self._listener = listener
        self._stop = stop
        self._detach: Callable[[], None] | None = None
        self._exports: dict[tuple[int, int, object], np.ndarray] = {}
        self._spatial = problem.transform.geometry is not None
        self._shape = tuple(problem.shape[1:])
        spec = request.observation.stats
        self._row_names = tuple(spec.periodic)
        self._spec = spec
        self._row_shapes: dict[str, tuple[int, ...]] = {}
        # The statistics are lowered once, before anything is allocated: plane and
        # periodic rows read the same native implementation of each.
        # The union of the plane and row statistics is lowered once; the rows' plan
        # reads the same lowered inputs.
        union = tuple(dict.fromkeys((*spec.names, *self._row_names)))
        # One set of far-field tables, read by the plan and handed to the backend.
        if backend is not None:
            grid, modes, profiles = backend._grid, backend._modes, backend._profiles  # pyright: ignore[reportPrivateUsage]
            self._far_field = backend._far_field  # pyright: ignore[reportPrivateUsage]
        else:
            self._far_field = FarFieldTables(problem)
            owns = True
        self._owns = owns
        lowering = self._lower(union, grid, modes, profiles)
        self._statistics_tables = lowering.tables
        self._plane_lowering = lowering.subset(spec.names)
        self._row_lowering = lowering.subset(self._row_names)
        # The attempts since the previous observation, for its summary alone.
        self._plane_norms: list[float] = []
        self._plane_steps = 0
        self._plane_rejected = 0
        self._plane_wall = 0.0
        if backend is None:
            backend = NativeBackend(
                problem,
                grid=grid,
                modes=modes,
                profiles=profiles,
                far_field=self._far_field,
            )
        self._backend: NativeBackend | None = backend
        try:
            kind, state = self._prepare(backend, start)
            self._native = backend._native_state.start_session(  # pyright: ignore[reportPrivateUsage]
                kind, state, self._plan(start), float(start.z)
            )
        except BaseException:
            if self._owns:
                backend.close()
            raise
        self._adams = kind == "adams"
        attach = getattr(stop, "attach", None)
        if callable(attach):
            self._detach = cast("Callable[[], None]", attach(self._native.request_stop))

    # -- preparation ----------------------------------------------------------------

    def _lower(
        self,
        names: tuple[str, ...],
        grid: TimeGrid | None,
        modes: Sequence[Mode] | None,
        profiles: Mapping[str, ZProfile] | None,
    ) -> StatisticsLowering:
        """Lower statistics for the native plan, with their far-field tables.

        Parameters
        ----------
        names : tuple of str
            The statistics.
        grid : solprop.grid.TimeGrid or None
            The time grid.
        modes : sequence of Mode or None
            The mode set.
        profiles : mapping or None
            The z profiles.

        Returns
        -------
        solprop_native._statistics.StatisticsLowering
            The lowering.

        Raises
        ------
        solprop.exceptions.BackendError
            Naming an input a statistic lacks.
        """
        apertures = None
        if self._spatial and names:
            tables = self._far_field.for_spec(self._spec)
            apertures = _aperture_payload(tables, int(self._problem.shape[-1]))
        try:
            return lower(
                self._problem,
                self._spec,
                names,
                grid=grid,
                modes=modes,
                profiles=profiles,
                apertures=apertures,
            )
        except BackendError as error:
            msg = f"the native session does not run {error}"
            raise BackendError(msg) from error

    def _prepare(
        self, backend: NativeBackend, start: Start
    ) -> tuple[str, dict[str, object]]:
        """Load the start and configure the backend's native march; return its state.

        Parameters
        ----------
        backend : solprop_native.backend.NativeBackend
            The freshly built backend.
        start : solprop.backends.session.Start
            Where the run starts.

        Returns
        -------
        tuple
            The march's kind and its state at the start, as the native call takes it.

        Raises
        ------
        solprop.exceptions.BackendError
            If the backend's own admission disagrees with :func:`refusal`'s, which
            decided before it was built.
        """
        driver = self._driver(backend)
        integrator = self._problem.stepper.integrator
        backend.set_initial(
            np.asarray(start.take_field())[np.newaxis],
            None,
            z=float(start.z),
            resume=False,
        )
        if isinstance(integrator, Multistep):
            march = driver._adams
            refused = march.refusal
        else:
            refused, _ = _NativeRungeKuttaMarch.refusal(backend)
            march = None if refused is not None else driver._rk_march()
        if refused is not None:
            msg = (
                f"the native backend steps this run's march on the host ({refused}), "
                "which the native session's admission did not foresee"
            )
            raise BackendError(msg)
        assert march is not None
        march._configure()
        monitor = self._monitor
        if monitor is not None:
            # The monitor's weights where the state lives, read at every check, and the
            # split collar's walk-off buckets (`attach_monitor`); the monitor stays
            # solprop's and fresh: the session owns its state natively.
            driver._bind_monitor(
                dataclasses.replace(
                    monitor,
                    crossing=self._problem.split_crossing if monitor.split else None,
                )
            )
        kind = "adams" if isinstance(integrator, Multistep) else "runge_kutta"
        return kind, march._state(False)

    @staticmethod
    def _driver(backend: NativeBackend | None) -> Any:
        """Return the backend as the untyped protected surface the session lowers from.

        The one place the adapter reaches the backend's protected members (its march
        adapters, their policy and state records, its statistics unpacking), as the
        backend's own march adapters reach them (``DRIVER_SURFACE``); confined here and
        typed ``Any`` because those members are the driver's private surface.

        Parameters
        ----------
        backend : solprop_native.backend.NativeBackend or None
            The backend.

        Returns
        -------
        Any
            The same object.
        """
        assert backend is not None
        return backend

    def _plan(self, start: Start) -> dict[str, object]:
        """Return the observation request and itinerary for the native session.

        Parameters
        ----------
        start : solprop.backends.session.Start
            Where the run starts.

        Returns
        -------
        dict
            The plan.
        """
        observation = self._request.observation
        schedule = (
            None if self._request.replay is None else self._request.replay.schedule
        )
        rows = bool(self._row_names)
        return {
            "planes": np.ascontiguousarray(observation.save_planes, dtype=np.float64),
            "next_plane": int(start.plane_index),
            "launch_pending": bool(start.plane_pending),
            "step_planes": int(observation.step_planes),
            "plane_statistics": self._plane_lowering.plan,
            "row_statistics": self._row_lowering.plan,
            "period": int(self._spec.period) if rows else 0,
            "attempt_capacity": ATTEMPT_CAPACITY,
            "row_capacity": ROW_CAPACITY,
            "replay": schedule is not None,
            "replay_steps": 0 if schedule is None else int(schedule.n_steps),
            "replay_length": 0.0 if schedule is None else float(schedule.total_length),
            "call_attempts": CALL_ATTEMPTS,
            "call_seconds": CALL_SECONDS,
            "monitor": self._monitor_plan(start),
        }

    def _monitor_plan(self, start: Start) -> dict[str, object] | None:
        """Return the native monitor's configuration and any record it continues.

        Parameters
        ----------
        start : solprop.backends.session.Start
            Where the run starts, with a monitored resume's continuation.

        Returns
        -------
        dict or None
            The monitor's plan, or ``None`` without a monitor.
        """
        monitor = self._monitor
        if monitor is None:
            return None
        backend = self._driver(self._backend)
        plan: dict[str, object] = {
            "config": _watch.monitor_config(
                monitor,
                self._problem.split_crossing if monitor.split else None,
                can_rest=bool(backend._collar_can_rest()),
            ).native(),
            "resume": start.monitor is not None,
        }
        if start.monitor is not None:
            plan.update(self._continuation(start.monitor))
        return plan

    @staticmethod
    def _continuation(continuation: Mapping[str, object]) -> dict[str, object]:
        """Return a monitored resume's continuation as the native session reads it.

        Parameters
        ----------
        continuation : collections.abc.Mapping
            As :attr:`solprop.backends.session.Start.monitor`.

        Returns
        -------
        dict
            The earlier record's resumption, whether the run resumes at a saved plane,
            and the stretches run since without a monitor.
        """
        record = continuation.get("record")
        return {
            "record": _watch.resumption(
                cast("Mapping[str, object]", record)
                if isinstance(record, Mapping)
                else None
            ).native(),
            "plane": bool(continuation.get("plane", False)),
            "unwatched": _watch.unwatched_of(
                cast("Sequence[tuple[float, float]]", continuation.get("unwatched", ()))
            ),
        }

    # -- protocol -------------------------------------------------------------------

    @property
    def native(self) -> _core.Session:
        """Return the native session, for the structural counters of the tests.

        Returns
        -------
        solprop_native._core.Session
            The native session.
        """
        return self._native

    @property
    def finished(self) -> bool:
        """Return whether the session yielded its terminal event.

        Returns
        -------
        bool
            True once finished.
        """
        return self._native.status().outcome != 0

    @property
    def closed(self) -> bool:
        """Return whether the session was closed.

        Returns
        -------
        bool
            True once closed.
        """
        return _STATES[self._native.status().state] is SessionState.CLOSED

    def status(self) -> SessionStatus:
        """Return a read-only summary of the session.

        Returns
        -------
        solprop.backends.session.SessionStatus
            Position, plane, attempts and lifecycle state.
        """
        status = self._native.status()
        return SessionStatus(
            z=status.z,
            plane_index=status.plane_index,
            attempts=status.attempts,
            rejected=status.rejected,
            state=_STATES[status.state],
            outcome=_OUTCOMES[status.outcome],
        )

    def advance(
        self,
        *,
        work_limit: int = DEFAULT_WORK_LIMIT,
        deadline: float = DEFAULT_DEADLINE,
    ) -> EventBatch:
        """Run to the next barrier, the attempt bound or the deadline, and report it.

        Parameters
        ----------
        work_limit : int, optional
            Attempts before a yield; one while a per-attempt listener is attached.
        deadline : float, optional
            Host seconds before a yield, s.

        Returns
        -------
        solprop.backends.events.EventBatch
            The events, ending at a barrier.

        Raises
        ------
        solprop.backends.session.SessionError
            Across a pending batch, after the terminal event or once closed.
        """
        stop = self._stop
        if stop is not None and stop.is_set():
            self._native.request_stop()
        limit = 1 if self._listener is not None else max(int(work_limit), 1)
        try:
            batch = self._native.advance(limit, float(deadline))
        except _core.SessionLifecycleError as error:
            raise SessionError(str(error)) from None
        receipt = Receipt(self._id, batch.sequence)
        self._sequence = batch.sequence
        # Each batch is one event and the chunk since the previous one, delivered once.
        chunk = self._history(batch.attempts)
        if self._listener is not None:
            for end, dz, accepted in zip(
                chunk.z.tolist(),
                chunk.dz.tolist(),
                chunk.accepted.tolist(),
                strict=True,
            ):
                self._listener(end, dz, bool(accepted))
        rows = self._chunk_rows(batch)
        return EventBatch(receipt, (self._event(batch, receipt, chunk, rows),))

    def acknowledge(self, receipt: Receipt) -> None:
        """Release the pending batch and allow advancing across its barrier.

        Parameters
        ----------
        receipt : solprop.backends.events.Receipt
            The pending batch's receipt.

        Raises
        ------
        solprop.backends.session.SessionError
            For a foreign, stale or repeated receipt, or once closed.
        """
        if self.closed:
            msg = "the session is closed"
            raise SessionError(msg)
        if receipt.session != self._id:
            msg = f"receipt {receipt} belongs to another session"
            raise SessionError(msg)
        try:
            self._native.acknowledge(receipt.sequence)
        except _core.SessionLifecycleError as error:
            raise SessionError(str(error)) from None
        self._exports.clear()

    def export(
        self,
        ticket: ExportTicket,
        recipe: ExportRecipe,
        out: npt.NDArray[np.complex128] | None = None,
    ) -> npt.NDArray[np.complex128]:
        """Fill a destination with a pending observation's field, converted natively.

        One native pass reads the resident state and writes the destination, applying
        the recipe (:func:`solprop.backends.export.apply_recipe`, operation for
        operation). A recipe exported into a new array is kept for the batch and handed
        to every consumer that asks again, read-only.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The pending observation's ticket.
        recipe : solprop.backends.export.ExportRecipe
            The conversion.
        out : numpy.ndarray or None, optional
            The destination, complex128, C-contiguous, writeable, the field's shape; a
            new array by default.

        Returns
        -------
        numpy.ndarray
            The converted field: `out` when given.

        Raises
        ------
        solprop.backends.session.SessionError
            If the ticket expired or belongs to another session.
        solprop.exceptions.BackendError
            If `out` cannot hold the field.
        solprop.exceptions.GridError
            If the route's power weight is not positive and finite.
        """
        if self.closed or ticket.receipt.session != self._id:
            msg = (
                f"the export ticket for z = {ticket.z!r} m has expired: its batch was "
                "acknowledged, or it belongs to another session"
            )
            raise SessionError(msg)
        if out is not None:
            self._check_destination(out)
        if recipe.power and self._spatial:
            # A spatial route's field measure is already the stored quantity.
            recipe = dataclasses.replace(recipe, power=False)
        key = (ticket.receipt.sequence, ticket.generation, recipe)
        held = self._exports.get(key)
        if held is not None:
            if out is None:
                return held
            np.copyto(out, held)
            return out
        weight = (
            np.ascontiguousarray(
                power_weight_at(self._problem, ticket.z), dtype=np.float64
            ).reshape(-1)
            if recipe.power
            else None
        )
        destination = np.empty(self._shape, dtype=np.complex128) if out is None else out
        try:
            self._native.export(
                ticket.receipt.sequence,
                ticket.generation,
                recipe.power,
                recipe.optics,
                weight,
                destination,
            )
        except _core.SessionLifecycleError as error:
            raise SessionError(str(error)) from None
        except ValueError as error:
            if "power weights" in str(error):
                raise GridError(str(error)) from None
            raise
        if out is None:
            destination.setflags(write=False)
            self._exports[key] = destination
        return destination

    def noise_phase(self, ticket: ExportTicket) -> np.ndarray | None:
        """Return ``None``: the prototype refuses noise models.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The pending observation's ticket.

        Returns
        -------
        None
            No noise phase.
        """
        del ticket

    def request_snapshot(self) -> SnapshotReceipt:
        """Latch a snapshot for the next eligible accepted endpoint.

        Returns
        -------
        solprop.backends.session.SnapshotReceipt
            The request's receipt; a request while one is outstanding joins it.

        Raises
        ------
        solprop.backends.session.SessionError
            Once closed.
        """
        try:
            request, coalesced = self._native.request_snapshot()
        except _core.SessionLifecycleError as error:
            raise SessionError(str(error)) from None
        return SnapshotReceipt(request, coalesced)

    def march_snapshot(self, ticket: ExportTicket) -> MarchSnapshot | None:
        """Return ``None``: the prototype keeps no method-history snapshot.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The pending observation's ticket.

        Returns
        -------
        None
            No snapshot.
        """
        del ticket

    def march_snapshot_nbytes(self) -> int | None:
        """Return ``None``: the prototype keeps no method-history snapshot.

        Returns
        -------
        None
            No snapshot.
        """
        return None

    def execution_record(self) -> Mapping[str, object]:
        """Return the session's execution facts.

        Returns
        -------
        collections.abc.Mapping
            The backend's build and execution record, with the session's counters.
        """
        status = self._native.status()
        record: dict[str, object] = {}
        if self._backend is not None and not self.closed:
            record.update(self._backend.metadata())
        if self._statistics_tables:
            # The profile statistics' tables and their measured errors (SEP 47 6.4).
            record["statistics_tables"] = dict(self._statistics_tables)
        record["native_session"] = {
            "march_calls": status.march_calls,
            "rows_taken": status.rows_taken,
            "batches": status.batches,
            "exports": status.exports,
            "export_bytes": status.export_bytes,
            "buffer_bytes": status.buffer_bytes,
        }
        monitor = self.monitor_record()
        if monitor is not None:
            # Where the driver's record carries its monitor's (`StepDriver.metadata`).
            record["boundary_monitor"] = monitor
        return record

    def monitor_record(self) -> Mapping[str, object] | None:
        """Return the boundary monitor's record, which the native session keeps.

        Returns
        -------
        collections.abc.Mapping or None
            The record as :meth:`solprop.monitor.BoundaryMonitor.summary` lays it out,
            or ``None`` without a monitor.
        """
        monitor = self._monitor
        if monitor is None:
            return None
        native = self._native.monitor_record()
        if native is None:
            return None
        return _watch.record_of(monitor.check, native)

    def continue_monitor(self, continuation: Mapping[str, object]) -> None:
        """Continue a resumed run's boundary-monitor record, before it is attached.

        Parameters
        ----------
        continuation : collections.abc.Mapping
            As :attr:`solprop.backends.session.Start.monitor`; one given there is kept.

        Raises
        ------
        solprop.backends.session.SessionError
            If the monitor already watches the run.
        """
        if self._monitor is None:
            return
        held = self._continuation(continuation)
        try:
            self._native.continue_monitor(
                cast("dict[str, object]", held["record"]),
                bool(held["plane"]),
                cast("list[list[float]]", held["unwatched"]),
            )
        except _core.SessionLifecycleError as error:
            raise SessionError(str(error)) from None

    def undelivered(self, ticket: ExportTicket) -> tuple[BackendHistory, StepStats]:
        """Return the attempts and rows made since the previous event: none.

        Every native batch delivers its whole chunk with its event (SEP 47 section
        5.3), and nothing advances between batches, so no attempt is ever left
        undelivered.

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The pending ticket a checkpoint is written for.

        Returns
        -------
        tuple
            An empty history and no rows.

        Raises
        ------
        solprop.backends.session.SessionError
            If the ticket belongs to another session.
        """
        if ticket.receipt.session != self._id:
            msg = f"the ticket for z = {ticket.z!r} m belongs to another session"
            raise SessionError(msg)
        return self._history(np.empty((0, 5))), self._no_rows()

    def pending_ticket(self) -> ExportTicket | None:
        """Return a ticket for the committed state while a batch is pending.

        Returns
        -------
        solprop.backends.events.ExportTicket or None
            A ticket bound to the pending batch, or ``None`` when none is pending or
            the session is closed: the native state exports only while its batch is.
        """
        status = self._native.status()
        if _STATES[status.state] is not SessionState.PENDING or self._sequence == 0:
            return None
        return ExportTicket(
            Receipt(self._id, self._sequence), status.generation, status.z
        )

    @property
    def reads_transverse_edge(self) -> bool:
        """Return whether the session's backend reads a transverse edge (SEP 40).

        Returns
        -------
        bool
            True: the native backend reads it, though this session refuses a
            monitor's edge for now.
        """
        return NativeBackend.reads_transverse_edge

    def close(self) -> None:
        """Release the native session and its backend; idempotent."""
        if self._detach is not None:
            self._detach()
            self._detach = None
        self._native.close()
        self._exports.clear()
        if self._backend is not None:
            if self._owns:
                self._backend.close()
            self._backend = None

    def __enter__(self) -> "NativeSession":  # noqa: UP037 - runtime type hook
        """Return the session.

        Returns
        -------
        NativeSession
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

    # -- translation ----------------------------------------------------------------

    def _check_destination(self, out: np.ndarray) -> None:
        """Refuse a destination that cannot hold an export (as ``apply_recipe`` does).

        Parameters
        ----------
        out : numpy.ndarray
            The proposed destination.

        Raises
        ------
        solprop.exceptions.BackendError
            Naming the mismatch.
        """
        if out.shape != self._shape:
            msg = f"an export destination has shape {out.shape}, expected {self._shape}"
            raise BackendError(msg)
        if (
            out.dtype != np.complex128
            or not out.flags.c_contiguous
            or not out.flags.writeable
        ):
            msg = "an export destination must be writeable, C-contiguous complex128"
            raise BackendError(msg)

    def _plane_stats(self, batch: _core.SessionBatch) -> dict[str, np.ndarray]:
        """Return a batch's plane statistics by name.

        Parameters
        ----------
        batch : solprop_native._core.SessionBatch
            The batch.

        Returns
        -------
        dict
            The statistics, empty when the batch carries none.
        """
        if not batch.event.has_stats:
            return {}
        return self._plane_lowering.unpack(batch.plane_stats[0])

    def _chunk_rows(self, batch: _core.SessionBatch) -> StepStats:
        """Return a batch's chunk of periodic rows, its steps indexing the chunk.

        Parameters
        ----------
        batch : solprop_native._core.SessionBatch
            The batch.

        Returns
        -------
        solprop.backends.protocol.StepStats
            The rows taken since the previous event.
        """
        z = batch.row_z.reshape(-1)
        step = batch.row_step.reshape(-1).astype(np.int64)
        if z.size == 0:
            return self._no_rows()
        unpacked = [self._row_lowering.unpack(row) for row in batch.row_values]
        values = {
            name: np.stack([row[name] for row in unpacked]) for name in self._row_names
        }
        self._row_shapes = {name: value.shape[1:] for name, value in values.items()}
        return StepStats(z, step, values)

    def _no_rows(self) -> StepStats:
        """Return no rows: no names before the first row, then each row's shape.

        Returns
        -------
        solprop.backends.protocol.StepStats
            Empty rows, as the driver's buffer drains them.
        """
        return StepStats(
            np.empty(0),
            np.empty(0, dtype=np.int64),
            {
                name: np.zeros((0, *shape), dtype=np.float64)
                for name, shape in self._row_shapes.items()
            },
        )

    @staticmethod
    def _history(attempts: np.ndarray) -> BackendHistory:
        """Return a chunk's attempt rows as a history.

        Parameters
        ----------
        attempts : numpy.ndarray
            The rows, ``(n, 5)``: end, step, norm, acceptance, wall time.

        Returns
        -------
        solprop.backends.protocol.BackendHistory
            The attempts.
        """
        return BackendHistory(
            z=np.ascontiguousarray(attempts[:, 0]),
            dz=np.ascontiguousarray(attempts[:, 1]),
            error_norm=np.ascontiguousarray(attempts[:, 2]),
            accepted=attempts[:, 3] != 0.0,
            wall_time=np.ascontiguousarray(attempts[:, 4]),
        )

    def _count(self, chunk: BackendHistory) -> None:
        """Count a delivered chunk's attempts toward the next observation's summary.

        Parameters
        ----------
        chunk : solprop.backends.protocol.BackendHistory
            The attempts one event delivered; their wall times start at that chunk.
        """
        self._plane_norms.extend(chunk.error_norm[chunk.accepted].tolist())
        self._plane_steps += chunk.n_steps
        self._plane_rejected += chunk.n_rejected
        if chunk.n_steps:
            self._plane_wall += float(chunk.wall_time[-1])

    def _summary(self, z: float) -> StepSummary:
        """Return the cost and tolerance of the propagation an observation reports.

        Formed from every chunk since the previous observation, as the driver forms
        it from the same attempts, and started afresh.

        Parameters
        ----------
        z : float
            Its position, m.

        Returns
        -------
        solprop.backends.protocol.StepSummary
            The summary.
        """
        record = ControlRecord.from_norms(
            self._problem.stepper.controller,
            self._plane_norms,
            n_rejected=self._plane_rejected,
        )
        summary = StepSummary(
            z=z,
            n_steps=self._plane_steps,
            n_rejected=self._plane_rejected,
            max_norm=record.max_norm,
            mean_norm=record.mean_norm,
            wall_time=self._plane_wall,
            met_tolerance=record.met_tolerance,
        )
        self._plane_norms, self._plane_steps, self._plane_rejected = [], 0, 0
        self._plane_wall = 0.0
        return summary

    def _event(
        self,
        batch: _core.SessionBatch,
        receipt: Receipt,
        chunk: BackendHistory,
        rows: StepStats,
    ) -> Event:
        """Translate a batch's native event record and its chunk.

        Parameters
        ----------
        batch : solprop_native._core.SessionBatch
            The batch.
        receipt : solprop.backends.events.Receipt
            Its receipt.
        chunk : solprop.backends.protocol.BackendHistory
            Its attempts, since the previous event.
        rows : solprop.backends.protocol.StepStats
            Its periodic rows, likewise.

        Returns
        -------
        solprop.backends.events.Event
            The event.
        """
        event = batch.event
        ticket = (
            ExportTicket(receipt, event.generation, event.z) if event.ticket else None
        )
        # Exactly-once chunked delivery (SEP 47 section 5.3): the event carries the
        # attempts and rows since the previous event, and the caller assembles them.
        self._count(chunk)
        if event.kind == _PROGRESS:
            return Progress(event.z, chunk, rows)
        if event.kind == _OBSERVATION:
            assert ticket is not None
            return ObservationReady(
                z=event.z,
                kind=_KINDS[event.observation],
                plane_index=event.plane_index,
                history=chunk,
                step_stats=rows,
                stats=self._plane_stats(batch),
                summary=self._summary(event.z),
                ticket=ticket,
            )
        if event.kind == _SNAPSHOT:
            return ServiceRequired(
                kind=ServiceKind.SNAPSHOT,
                z=event.z,
                message="snapshot at an accepted endpoint",
                ticket=ticket,
                history=chunk,
                step_stats=rows,
            )
        if event.kind == _WARNING:
            # A boundary warning (SEP 47 section 5.5): the caller emits it, and its
            # filters may raise it; the committed field and the chunk are there for
            # that failure's checkpoint.
            assert self._monitor is not None
            notice = _watch.Notice.of(event.notice)
            if notice.kind is _watch.NoticeKind.REMOVAL:
                self._log_finish()
            warning = _watch.notice_warning(self._monitor.check, notice)
            return ServiceRequired(
                kind=ServiceKind.WARNING,
                z=event.z,
                message=str(warning),
                category=type(warning),
                ticket=ticket,
                history=chunk,
                step_stats=rows,
                warning=warning,
            )
        outcome = _OUTCOMES[event.outcome]
        assert outcome is not None
        if outcome is Outcome.COMPLETED:
            self._log_finish()
            return Terminal(
                Outcome.COMPLETED,
                event.z,
                Eligibility.ELIGIBLE,
                None,
                None,
                chunk,
                rows,
            )
        return Terminal(
            outcome,
            event.z,
            Eligibility.ELIGIBLE,
            self._error(event, outcome),
            ticket,
            chunk,
            rows,
        )

    def _log_finish(self) -> None:
        """Log the monitor's removal and guard peaks once, at the end (D5)."""
        if self._finish_logged or self._monitor is None:
            return
        self._finish_logged = True
        record = self._native.monitor_record()
        if record is not None:
            _log_finish(
                float(record["removed_fraction"]),
                float(record["guard_low_max"]),
                float(record["guard_high_max"]),
            )

    def _error(self, event: _core.SessionEvent, outcome: Outcome) -> BaseException:
        """Return the exception a stopped or failed session reports.

        A failure is formatted by the backend's own march adapter, as its native call
        would have raised it.

        Parameters
        ----------
        event : solprop_native._core.SessionEvent
            The terminal record.
        outcome : solprop.backends.events.Outcome
            Stopped or failed.

        Returns
        -------
        BaseException
            The stop request or the primary failure.
        """
        if outcome is Outcome.STOPPED:
            return StopRequested(self._stop)
        notice = _watch.Notice.of(event.notice)
        if notice.kind is not _watch.NoticeKind.NONE and self._monitor is not None:
            return _watch.notice_error(self._monitor.check, notice)
        if event.schedule_exhausted:
            schedule = self._problem.stepper.schedule
            end = math.nan if schedule is None else float(schedule.z_end)
            return BackendError(
                f"the replayed schedule ends at {end:.6g} m and cannot reach "
                f"z = {event.target:.6g} m"
            )
        # A statistics evaluation the native plan refused: the reference's exception.
        refused = translated(BackendError(event.message))
        if not isinstance(refused, BackendError):
            return refused
        result = {
            "reason": event.reason,
            "cost": event.cost,
            "floor_proposal": event.floor_proposal,
            "replay_dz": event.replay_dz,
            "replay_end": event.replay_end,
            "replay_norm": event.replay_norm,
            "message": event.message,
            "category": event.category,
        }
        driver = self._driver(self._backend)
        if self._adams:
            march = driver._adams
        else:
            # The diverged replay's message names the cursor and the position.
            driver._schedule_index = int(event.cursor)
            driver._z = float(event.z)
            march = driver._rk_march()
        try:
            march._raise(int(event.reason), result)
        except Exception as error:
            return error
        return RuntimeError(event.message or "the native session failed")
