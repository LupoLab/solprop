# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native execution session, solprop SEP 47's run owner in C++.

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

Every run on a native backend uses it: the backend's prepared execution
(:class:`NativePrepared`, from ``NativeBackend.prepare``, which
:func:`solprop.backends.sessions.start_session` starts) opens a :class:`NativeSession`
for each start; there is no other native executor (SEP 47 phase 5). It admits the
following and refuses every other feature by name before allocating anything
(:func:`refusal`, and the backend where it is built):

- the native CPU build; Adams (``Multistep``) and the embedded Runge-Kutta tableaux,
  adaptive or replayed, on the routes the resident engines march without a host
  fallback;
- on a device build (CUDA or HIP), the same (SEP 47 phase 4):
  the session's own code is host code over the resident state wherever it lives, its
  decisions made on the host from the device's reductions, and an export crosses the
  field to the destination once and converts it there; the backend refuses a carrier
  frame and a z table that is not device-ready by name;
- the low-storage replay (``LowStorageRungeKutta``, SEP 47 section 6.3) on the
  diagonal-flow routes whose stage rows the native state reads from its tables: every
  step one in-place native step, its scalars formed natively, the replay loop the
  native Runge-Kutta march's; a failure in the middle of a step leaves no consistent
  state, and the terminal event is then poisoned, with no ticket;
- every periodic-capable statistic of the catalogue on these routes, plane and
  periodic, each evaluated and finished natively from a plan lowered once
  (:mod:`solprop_native._statistics`), the profile statistics from tables whose
  measured error the execution record carries, and the plane-only statistics, a
  spatial route's fluence widths and the on-axis plasma statistics among them (the
  axis row scanned natively by the node scan's own pass); what the problem does not
  carry is refused (:func:`solprop.backends.session.statistics_unavailable`);
- the boundary monitor, temporal and a free-space run's transverse edge, which the
  native session owns (SEP 47 section 6.4): its every decision and record are native,
  each warning is its own :class:`~solprop.backends.events.ServiceRequired` barrier
  and a stop a terminal failure, the edge's readings and a collar's current are taken
  from the resident state with no return to Python, and a monitored resume continues
  both records natively (:mod:`solprop_native._transverse`);
- dense output (``interpolate_planes``), for Adams and the embedded Runge-Kutta
  tableau alike: a plane the march steps across is interpolated natively inside the
  step that crossed it, the committed state held aside, and observed in order before
  the march goes on; a latched snapshot waits for the next accepted endpoint, every
  plane its step crossed observed first (SEP 47 sections 5.3 and 5.5);
- resumed runs: a field restart, restored call ledgers and the Adams method history
  (:class:`~solprop.output.schema.MarchSnapshot`), which the session exports at every
  accepted endpoint a ticket names and installs from its start, as the backend's own
  march does;
- the input and modified noise models, the input draw's vacuum floor read by the
  native monitor and the draw's readings by the native transverse edge, and the noise
  phase at each plane, formed on the host or, for a z table's propagator phase, read
  from its slot in the leased state;
- no group (:mod:`solprop_native.group` runs one native session per member); and no
  comparison method, controller that overrides a decision or TDSE response, which the
  backend refuses by name where it is built, before it allocates (SEP 47 section
  6.3).
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from solprop._interrupts import committing
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
from solprop.backends.export import destination_refusal
from solprop.backends.multistep import CallLedger, budget_error
from solprop.backends.populations import axis_refusal
from solprop.backends.protocol import BackendHistory, StepStats, StepSummary
from solprop.backends.session import (
    DEFAULT_DEADLINE,
    DEFAULT_WORK_LIMIT,
    ExecutionOptions,
    SessionError,
    SessionState,
    SessionStatus,
    SnapshotReceipt,
    statistics_unavailable,
)
from solprop.backends.session_core import check_preparation
from solprop.backends.spatial import FarFieldTables
from solprop.exceptions import (
    BackendError,
    ConvergenceError,
    GridError,
    StopRequested,
)
from solprop.monitor import (
    _log_finish,  # pyright: ignore[reportPrivateUsage] - one log line
)
from solprop.output.capability import (
    declared_capability,
    refusal_for,
    withheld_accessors,
)
from solprop.output.schema import MarchSnapshot
from solprop.stepper.controller import ControlRecord
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.tableau import continuous_extension
from solprop.translate.linop import power_weight_at

from . import _core, _counters, _transverse, _watch
from ._load import Loaded
from ._statistics import (
    ADMITTED_GENERAL_STATISTICS,
    ADMITTED_SPATIAL_STATISTICS,
    ADMITTED_STATISTICS,
    StatisticsLowering,
    lower,
    metric_refusal,
    translated,
)
from .backend import (
    _BUDGET,  # pyright: ignore[reportPrivateUsage]
    _DIVERGED,  # pyright: ignore[reportPrivateUsage]
    NativeBackend,
    _aperture_payload,  # pyright: ignore[reportPrivateUsage]
    _low_storage_payload,  # pyright: ignore[reportPrivateUsage]
    _NativeRungeKuttaMarch,  # pyright: ignore[reportPrivateUsage]
    _on_device,  # pyright: ignore[reportPrivateUsage]
    _raise_rk,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from types import TracebackType

    from solprop.backends.events import Event
    from solprop.backends.export import ExportRecipe
    from solprop.backends.session import RunSession, Start
    from solprop.grid import TimeGrid
    from solprop.log import StopFlag
    from solprop.modes.protocol import Mode
    from solprop.monitor import BoundaryMonitor
    from solprop.profiles import ZProfile
    from solprop.stats import StatsSpec
    from solprop.translate.problem import PropagationProblem, StepperSettings
    from solprop.translate.run_request import RunRequest

__all__ = [
    "ADMITTED_GENERAL_STATISTICS",
    "ADMITTED_SPATIAL_STATISTICS",
    "ADMITTED_STATISTICS",
    "NativePrepared",
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
_EVENT_NAMES = ("observation", "progress", "snapshot", "terminal", "warning")
"""The native event kinds by name, as the benchmark counters report a batch."""
_EMPTY = np.empty(0, dtype=np.float64)
_NO_ORDERS = 1 << 62
"""An order index past any run's, to read the march's state without its orders."""
_LEDGER_COUNTERS = (
    "startup",
    "predictor",
    "rejected",
    "accepted_steps",
    "rejected_steps",
    "restarts",
    "failure_restarts",
    "dense_observations",
)
"""The call ledger's counters the native Adams march keeps (``_MARCH_COUNTERS``)."""


def refusal(
    problem: PropagationProblem, request: RunRequest, start: Start | None = None
) -> str | None:
    """Return why the native session refuses a run, naming the feature, or ``None``.

    Decided from the problem, the request and the start alone, before anything is
    allocated: the statistics the session does not evaluate. Everything else native
    does not run -- a comparison method, a controller that overrides a decision, a
    TDSE response, a coupled flow beside the interaction formulation, stage rows the
    native tables do not read -- the backend refuses by name where it is built
    (``NativeBackend``), still before any native state (SEP 47 section 6.3).

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
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
    return _statistics_refusal(problem, request.observation.stats)


def _diverged(
    name: str, stepper: StepperSettings, event: _core.SessionEvent
) -> ConvergenceError:
    """Return the error of a low-storage step that ended in a non-finite state.

    The reference's message (``HostLowStorageMarch``): the step was accepted in place,
    so the state is lost with it, and the terminal event that carries this error is
    poisoned.

    Parameters
    ----------
    name : str
        The backend's name.
    stepper : solprop.translate.problem.StepperSettings
        The run's stepper section, whose method names the march.
    event : solprop_native._core.SessionEvent
        The terminal record: where the step started and its length.

    Returns
    -------
    solprop.exceptions.ConvergenceError
        The divergence.
    """
    method = cast("LowStorageRungeKutta", stepper.integrator)
    return ConvergenceError(
        f"{name}: a replayed {method.name} step of {event.replay_dz:.4g} m from z = "
        f"{event.z:.6g} m ended in a non-finite state: the field has diverged. A "
        "replayed schedule takes its steps as given and cannot reject one, so a step "
        "beyond an explicit stability limit -- the in-stage collars' -- grows until it "
        "overflows; the translation refuses a declared step past the collars' limit. "
        "The state was accepted in place and is lost; take shorter steps"
    )


def _statistics_refusal(problem: PropagationProblem, spec: StatsSpec) -> str | None:
    """Return why the session refuses a run's statistics, naming them, or ``None``.

    Every statistic the native plan evaluates on the route is admitted, plane and
    periodic (:mod:`solprop_native._statistics`), a spatial route's fluence widths
    and the on-axis plasma statistics among them. What the backend itself would
    refuse, and what the problem does not carry
    (:func:`solprop.backends.session.statistics_unavailable`, which every executor
    refuses before it starts, and
    :func:`solprop.backends.populations.axis_refusal`), are refused by name.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
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
    unavailable = statistics_unavailable(problem, (*names, *spec.periodic))
    if unavailable is not None:
        return unavailable
    refused = metric_refusal(problem, (*names, *spec.periodic))
    if refused is not None:
        return refused
    # Every other name the backend admits is the plan's on its route.
    return axis_refusal(problem, names)


class NativeSession:
    """A :class:`~solprop.backends.session.RunSession` owned by the native backend.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
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
    options : solprop.backends.session.ExecutionOptions or None, optional
        The run's execution options, from which the backend plans its node chunk;
        ``None`` for the automatic choice.
    backend : solprop_native.backend.NativeBackend or None, optional
        A backend already built for `problem` and holding no state, to run on in place
        of building one: a runner's or a run's (:func:`solprop.backends.sessions.
        start_session`), or a group's member, as a batched scan constructs it
        (:class:`~solprop_native.group.NativeGroupSession`). Its own time grid, modes,
        profiles, far-field tables and execution options are used, and `grid`,
        `modes`, `profiles` and `options` must then be left out. A failed construction
        leaves it to the caller.
    owns : bool, optional
        Whether closing the session closes `backend`: false for a runner's backend
        reused across runs. A backend the session built is always its own.
    started : callable or None, optional
        Called with the start, the end and the plane count when propagation begins:
        before the first advance past an owed launch plane, as the backend's own march
        announces it at its first propagation, and never for a start at the end.

    Raises
    ------
    solprop.exceptions.BackendError
        Naming the feature the session or the backend refuses, before anything is
        allocated; or if `backend` was built for another problem, or with context
        given beside it.
    """

    def __init__(
        self,
        problem: PropagationProblem,
        request: RunRequest,
        start: Start,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        stop: StopFlag | None = None,
        listener: Callable[[float, float, bool], None] | None = None,
        options: ExecutionOptions | None = None,
        backend: NativeBackend | None = None,
        owns: bool = True,
        started: Callable[[float, float, int], None] | None = None,
    ) -> None:
        """Refuse what the session does not admit, then prepare and start it."""
        refused = refusal(problem, request, start)
        if refused is not None:
            msg = f"the native session does not run {refused}"
            raise BackendError(msg)
        owned = backend is None
        if backend is not None:
            adopted = self._driver(backend)
            # A runner's backend, reused (not owned), was built for the problem this
            # one was derived from, which the runner checked compatible
            # (`check_derived_problem`); a backend handed over to own was built for
            # this very problem.
            if owns and adopted._problem is not problem:
                msg = "the native session's backend was built for another problem"
                raise BackendError(msg)
            if not (grid is modes is profiles is options is None):
                msg = (
                    "a native session over a built backend reads the backend's own "
                    "grid, modes, profiles and execution options"
                )
                raise BackendError(msg)
            grid, modes, profiles = adopted._grid, adopted._modes, adopted._profiles
        self._problem = problem
        self._request = request
        # The stepping policy as one record (`RunRequest.stepper`).
        self._stepper = request.stepper
        self._monitor: BoundaryMonitor | None = request.boundary.monitor
        self._finish_logged = False
        # A declared launch's share in the ramp's band is logged once it is launched.
        edge = None if self._monitor is None else self._monitor.transverse
        self._declared_logged = edge is None or not _transverse.declared(problem)
        self._sequence = 0
        self._id = next(_SESSION_IDS)
        self._listener = listener
        self._started = started
        self._launch_owed = bool(start.plane_pending)
        self._start_z = float(start.z)
        self._stop = stop
        self._detach: Callable[[], None] | None = None
        self._exports: dict[tuple[int, int, object], np.ndarray] = {}
        # The last native batch until its events are returned: a signal or an error
        # as its advance ends leaves its chunk to `undelivered`, for the failure's
        # checkpoint.
        self._held: tuple[int, _core.SessionBatch] | None = None
        # The Adams march's call ledger as the record states it: the restored counts
        # and orders, then the native march's (`_sync_calls`).
        self._calls: CallLedger | None = None
        self._orders_seen = 0
        self._march_refusal: str | None = None
        # The start as the backend loaded it (`NativeBackend._load`, in `_prepare`).
        self._loaded: Loaded | None = None
        self._method: Multistep | None = None
        self._max_calls: int | None = None
        self._rows_nbytes = 0
        # The input noise's readings the transverse edge's launch carries (`_prepare`).
        self._edge_noise: dict[str, object] | None = None
        self._spatial = problem.transform.geometry is not None
        self._shape = tuple(problem.shape)
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
        self._far_field = (
            FarFieldTables(problem, spec) if backend is None else backend._far_field  # pyright: ignore[reportPrivateUsage] - the tables it was built with
        )
        self._owns = owned or owns
        # Every position a plane statistic is taken at, where the request fixes them
        # all: its start and its save planes, with no step or interpolated plane.
        observation = request.observation
        self._planes: tuple[float, ...] | None = (
            None
            if observation.dense or observation.step_planes
            else (float(start.z), *(float(z) for z in observation.save_planes))
        )
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
                request,
                grid=grid,
                modes=modes,
                profiles=profiles,
                far_field=self._far_field,
                options=options,
            )
        self._backend: NativeBackend | None = backend
        try:
            kind, state = self._prepare(backend, start)
            self._native = backend._native_state.start_session(  # pyright: ignore[reportPrivateUsage]
                kind, state, self._plan(start), float(start.z)
            )
        except BaseException:
            if owned:
                backend.close()
            raise
        self._adams = kind == "adams"
        self._low_storage = kind == "low_storage"
        # The benchmark counters' name for the march, and the native march calls they
        # have counted (`_tally`).
        self._march_name = f"session_{kind}"
        self._calls_counted = 0
        self._poisoned = False
        self._abandoned: tuple[float, float] | None = None
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
                z_end=self._stepper.z_end,
                planes=self._planes,
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
        """
        driver = self._driver(backend)
        integrator = self._stepper.integrator
        # The native load: the field, the noise, the collar back in the stages and a
        # fresh Adams record (`NativeBackend._load`); then a resume's ledgers and its
        # method history, whose state replaces the loaded one.
        loaded = driver._load(
            np.asarray(start.take_field()),
            start.noise,
            z=float(start.z),
            resume=start.resume,
        )
        self._loaded = loaded
        self._march_refusal, proposal = driver._restore(start, loaded)
        if isinstance(integrator, Multistep):
            march = driver._adams
            march._configure()  # pyright: ignore[reportPrivateUsage]
        else:
            # The Runge-Kutta march, and the low-storage replay, which is the native
            # Runge-Kutta march's over the stage tables the state reads each step's
            # scalars from (SEP 47 section 6.3).
            driver._staged_rows()
            _NativeRungeKuttaMarch(backend, monitored=False)._configure()  # pyright: ignore[reportPrivateUsage]
        kind = "runge_kutta"
        if isinstance(integrator, Multistep):
            kind = "adams"
        elif isinstance(integrator, LowStorageRungeKutta):
            kind = "low_storage"
            # The flows `_ResidentLowStorage` holds, and the operator's origin, from
            # which a constant operator's table is built.
            payload = _low_storage_payload(
                self._problem, self._stepper, device=_on_device()
            )
            backend._native_state.low_storage_configure(  # pyright: ignore[reportPrivateUsage]
                int(cast("int", payload["low_storage_flows"])),
                driver._terms.separable is not None,
                float(self._problem.linear.z_span[0]),
            )
        monitor = self._monitor
        if monitor is not None:
            # The monitor's weights where the state lives, read at every check, and the
            # split collar's walk-off buckets; the monitor stays solprop's and fresh:
            # the session owns its state natively.
            driver._bind_monitor(
                dataclasses.replace(
                    monitor,
                    crossing=self._problem.split_crossing if monitor.split else None,
                )
            )
        if isinstance(integrator, Multistep):
            adams = driver._adams
            ledger = adams.ledger
            self._calls = dataclasses.replace(ledger, orders=list(ledger.orders))
            self._method = adams._method
            self._max_calls = adams._policy.max_calls
            # The history's persistent rows, sized while the state is not leased.
            self._rows_nbytes = int(adams._engine.snapshot_nbytes())
            state: dict[str, object] = adams.start_state(
                loaded.z, proposal, loaded.cursor
            )
        else:
            norms: list[float] = []
            state = {
                "z": float(loaded.z),
                "proposal": float(proposal),
                "norms": norms,
                "cursor": int(loaded.cursor),
            }
        self._edge_noise = self._noise_readings(driver, start)
        return kind, state

    def _noise_readings(self, driver: Any, start: Start) -> dict[str, object] | None:
        """Return the input noise's readings the transverse edge's launch carries.

        ``StepDriver.attach_monitor`` hands the edge's launch the input draw and the
        launch's columns without it (``_edge_clean_columns``), formed where the load
        holds both parts or, on a restart, from the loaded state, which is read here
        before the native session leases it. A launch that resumes an earlier edge's
        record is not judged, so it takes no such columns and no state is read for
        them: the native edge resumes exactly when the record's resumption holds a
        launch baseline (:func:`solprop_native._transverse.resumption`).

        Parameters
        ----------
        driver : solprop_native.backend.NativeBackend
            The loaded backend, its state not yet leased.
        start : solprop.backends.session.Start
            Where the run starts, with a monitored resume's continuation.

        Returns
        -------
        dict or None
            The readings (:func:`solprop_native._transverse.noise_readings`), or
            ``None`` without an edge or input noise.
        """
        edge = None if self._monitor is None else self._monitor.transverse
        assert self._loaded is not None  # `_prepare` loaded the start
        drawn = self._loaded.floor
        if edge is None or drawn is None:
            return None
        resumed = start.monitor is not None and "edge_baseline" in cast(
            "dict[str, object]", self._continuation(start.monitor)["edge_record"]
        )
        clean = None if resumed else driver._launch_clean_columns(edge)
        return _transverse.noise_readings(edge, np.asarray(drawn[0]), clean)

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
            "dense": self._dense_plan() if observation.dense else None,
        }

    def _dense_plan(self) -> dict[str, object]:
        """Return the dense output the native session interpolates planes with.

        Returns
        -------
        dict
            The tableau's dense-output weights, one per stage, empty for Adams: those
            of :func:`~solprop.stepper.tableau.continuous_extension`, which the
            translation requires of a Runge-Kutta run with interpolated planes.
        """
        stepper = self._stepper
        if isinstance(stepper.integrator, Multistep):
            return {"weights": _EMPTY}
        weights = continuous_extension(stepper.tableau)
        assert weights is not None  # the translation refuses dense output without one
        return {"weights": np.ascontiguousarray(weights, dtype=np.float64)}

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
                can_rest=bool(backend._collar_rests()),
            ).native(),
            "resume": start.monitor is not None,
        }
        assert self._loaded is not None  # `_prepare` loaded the start
        drawn = self._loaded.floor
        if drawn is not None:
            # The input noise's vacuum floor, read from its draw on the host once, as
            # `BoundaryMonitor.launch` reads it.
            floor = float(monitor.power(drawn).sum())
            plan["floor"] = [floor, *(floor * monitor.weights.mean(axis=1)).tolist()]
        if monitor.transverse is not None:
            plan["edge"] = _transverse.edge_config(
                monitor.transverse, self._problem, self._stepper
            ).native()
            plan["edge_noise"] = self._edge_noise
        if start.monitor is not None:
            plan.update(self._continuation(start.monitor))
        return plan

    def _continuation(self, continuation: Mapping[str, object]) -> dict[str, object]:
        """Return a monitored resume's continuation as the native session reads it.

        Parameters
        ----------
        continuation : collections.abc.Mapping
            As :attr:`solprop.backends.session.Start.monitor`.

        Returns
        -------
        dict
            The earlier record's resumption and its transverse edge's, whether the run
            resumes at a saved plane, and the stretches run since without a monitor.
        """
        record = continuation.get("record")
        edge = None if self._monitor is None else self._monitor.transverse
        faces = 0 if edge is None else len(edge._faces)  # pyright: ignore[reportPrivateUsage]
        return {
            "edge_record": _transverse.resumption(
                cast("Mapping[str, object]", record).get("transverse")
                if isinstance(record, Mapping)
                else None,
                faces,
            ),
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
        self._announce()
        stop = self._stop
        if stop is not None and stop.is_set():
            self._native.request_stop()
        limit = 1 if self._listener is not None else max(int(work_limit), 1)
        # A signal arriving during the native call is raised as this block ends, with
        # the batch pending and held (SEP 47 section 5.5): the failure's checkpoint
        # exports the batch's state and includes its chunk, once. The batch is held
        # before it is translated, so an error translating it loses no attempt; and
        # that error is kept as the cause of a signal that arrived meanwhile, which
        # the commit would otherwise drop beside it.
        failure: BaseException | None = None
        translated: tuple[BackendHistory, Event] | None = None
        try:
            with committing():
                try:
                    batch = self._native.advance(limit, float(deadline))
                except _core.SessionLifecycleError as error:
                    raise SessionError(str(error)) from None
                receipt = Receipt(self._id, batch.sequence)
                self._sequence = batch.sequence
                self._held = (batch.sequence, batch)
                try:
                    translated = self._translate(batch, receipt)
                except BaseException as error:
                    failure = error
        except BaseException as signalled:
            if failure is not None and signalled is not failure:
                raise signalled from failure
            raise
        if failure is not None:
            raise failure
        assert translated is not None
        chunk, event = translated
        if (counters := _counters.active) is not None:
            self._tally(counters, batch, chunk, limit)
        if self._listener is not None:
            for end, dz, accepted in zip(
                chunk.z.tolist(),
                chunk.dz.tolist(),
                chunk.accepted.tolist(),
                strict=True,
            ):
                self._listener(end, dz, bool(accepted))
        self._held = None
        return EventBatch(receipt, (event,))

    def _translate(
        self, batch: _core.SessionBatch, receipt: Receipt
    ) -> tuple[BackendHistory, Event]:
        """Return a native batch's chunk of attempts and its one event.

        Parameters
        ----------
        batch : solprop_native._core.SessionBatch
            The batch, pending.
        receipt : solprop.backends.events.Receipt
            Its receipt.

        Returns
        -------
        tuple
            The attempts since the previous event, and the event carrying them.
        """
        # Each batch is one event and the chunk since the previous one, delivered once.
        chunk = self._history(batch.attempts)
        rows = self._chunk_rows(batch)
        if not self._declared_logged:
            edge = self._native.edge_record()
            self._declared_logged = edge is None or _transverse.log_declared(edge)
        return chunk, self._event(batch, receipt, chunk, rows)

    def _announce(self) -> None:
        """Announce the propagation once, before the first advance that may step."""
        if self._started is None:
            return
        if self._launch_owed:
            # This advance commits the launch plane alone.
            self._launch_owed = False
            return
        announce, self._started = self._started, None
        settings = self._stepper
        if self._start_z < settings.z_end:
            announce(self._start_z, settings.z_end, len(settings.save_planes))

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
        if (counters := _counters.active) is not None:
            counters.crossing("export", destination.nbytes)
        if out is None:
            destination.setflags(write=False)
            self._exports[key] = destination
        return destination

    def noise_phase(self, ticket: ExportTicket) -> np.ndarray | None:
        """Return the noise phase at a pending observation, ``Im Phi`` there.

        Formed on the host from the problem's scalar tables, as the backend's own
        observation forms it (``StepDriver._noise_phase``); a z table's, which the
        leased state holds, is its slot at the plane, read through the native session
        under its lock as the backend reads it (``NativeBackend._unitary_phase``).

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            The pending observation's ticket.

        Returns
        -------
        numpy.ndarray or None
            The phase, shape ``(n_modes, n_pol, n_omega)``, or ``None`` without noise.

        Raises
        ------
        solprop.backends.session.SessionError
            If the ticket belongs to another session, or a z table's ticket expired.
        solprop.exceptions.BackendError
            If a z table cannot be read at the plane (outside its span).
        """
        if self._problem.initial.noise_model == "none":
            return None
        if self.closed or ticket.receipt.session != self._id:
            msg = f"the ticket for z = {ticket.z!r} m has expired"
            raise SessionError(msg)
        driver = self._driver(self._backend)
        z = float(ticket.z)
        tables = driver._z_tables
        if tables is not None and tables.phi:
            try:
                values = self._native.table_phi(
                    ticket.receipt.sequence, ticket.generation, z
                )
            except _core.SessionLifecycleError as error:
                raise SessionError(str(error)) from None
            except (ValueError, IndexError, RuntimeError) as error:
                # Too many positions, one outside the table's span, or no table of
                # `Phi`: the backend's error, as `_ZTables.load` raises the first.
                raise BackendError(str(error)) from error
            phase = np.imag(values).reshape(driver._layout)
        else:
            phase = driver._unitary_phase(driver._stage_scalars(z, z))
        # The phase keeps the operator table's leading axis of one.
        return np.array(np.broadcast_to(phase, (1, *self._problem.shape))[0])

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
        """Return the Adams method history at a pending ticket's committed state.

        The native session copies the march's state, the accepted field as the state
        holds it and every persistent history row once, losslessly, as
        solprop's host history lays them out
        (:meth:`solprop.backends.host.adams.HostAdamsMarch.snapshot`; SEP 47 section
        5.4).

        Parameters
        ----------
        ticket : solprop.backends.events.ExportTicket
            A ticket of the pending batch.

        Returns
        -------
        solprop.output.schema.MarchSnapshot or None
            The snapshot, or ``None`` for a Runge-Kutta march, which keeps no history,
            and before the Adams march holds any.

        Raises
        ------
        solprop.backends.session.SessionError
            If the ticket expired or belongs to another session.
        """
        if not self._adams:
            return None
        if self.closed or ticket.receipt.session != self._id:
            msg = f"the ticket for z = {ticket.z!r} m has expired"
            raise SessionError(msg)
        try:
            taken = self._native.march_snapshot(
                ticket.receipt.sequence, ticket.generation
            )
        except _core.SessionLifecycleError as error:
            raise SessionError(str(error)) from None
        if taken is None:
            return None
        state, field, rows = taken
        if (counters := _counters.active) is not None:
            counters.crossing(
                "snapshot", field.nbytes + sum(row.nbytes for row in rows.values())
            )
        if float(state["z"]) != ticket.z:
            # A dense plane's ticket: the march stands ahead of it, at its endpoint,
            # and keeps no history a resume from the plane could continue.
            return None
        method = self._method
        assert method is not None
        shape = tuple(self._problem.shape)
        history = state["history"]
        return MarchSnapshot(
            z=float(state["z"]),
            # The trajectory's state; the history rows keep the engine's layout, a
            # leading axis of one after the row axis (solprop SEP 47 section 6.7).
            state=field.reshape(shape),
            proposal=float(state["proposal"]),
            method=repr(method),
            order=int(state["order"]),
            rejections=int(state["rejections"]),
            last_rejected=bool(state["last_rejected"]),
            positions=np.array([entry[0] for entry in history], dtype=np.float64),
            identities=tuple(
                "predicted" if entry[1] else "accepted" for entry in history
            ),
            slots=np.array([entry[2] for entry in history], dtype=np.int64),
            rows={name: row.reshape(len(row), 1, *shape) for name, row in rows.items()},
            collar_was_quiet=bool(state["collar_was_quiet"]),
            history_key=method.history_key,
        )

    def march_snapshot_nbytes(self) -> int | None:
        """Return the bytes :meth:`march_snapshot` would copy now, copying nothing.

        Returns
        -------
        int or None
            The state and every history row, or ``None`` where a snapshot would be
            ``None``.
        """
        if not self._adams or self.closed:
            return None
        state = self._native.march_state(_NO_ORDERS)
        if state is None or not state["history"]:
            return None
        ticket = self.pending_ticket()
        if ticket is not None and float(state["z"]) != ticket.z:
            # A dense plane's: the march stands ahead of it (`march_snapshot`).
            return None
        return math.prod(self._problem.shape) * 16 + self._rows_nbytes

    def march_export_refusal(self) -> str | None:
        """Return ``None``: the session exports the history of every march it admits.

        Returns
        -------
        None
            The resident Adams engine exports and installs its rows; a Runge-Kutta
            march keeps no history to refuse.
        """
        return None

    @property
    def march_refusal(self) -> str | None:
        """Return why the start's method history was not installed, or ``None``.

        Returns
        -------
        str or None
            The backend march's reason (:meth:`~solprop.backends.driver.StepDriver.
            restore_march`), or ``None`` when it was installed or none was given.
        """
        return self._march_refusal

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
        calls = self._sync_calls()
        if calls is not None:
            # The march's ledger is the session's, not the backend's (SEP 47 5.2).
            record["response_calls"] = calls.to_record()
        if self._statistics_tables:
            # The profile statistics' tables and their measured errors (SEP 47 6.4).
            record["statistics_tables"] = dict(self._statistics_tables)
        if self._low_storage:
            # Four source evaluations a committed step (SEP 24 section 5), which the
            # native replay makes: none for a step abandoned or diverged.
            method = cast("LowStorageRungeKutta", self._stepper.integrator)
            record["source_evaluations"] = method.evaluations * status.attempts
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

    def _sync_calls(self) -> CallLedger | None:
        """Bring the Adams call ledger up to the native march's, and return it.

        The counters are the native march's, cumulative from the restored ones; the
        orders of the steps accepted since the last call are appended, so that the
        record's run-length encoding is extended rather than redone.

        Returns
        -------
        solprop.backends.multistep.CallLedger or None
            The ledger, or ``None`` for a Runge-Kutta march.
        """
        calls = self._calls
        if calls is None or self.closed:
            return calls
        state = self._native.march_state(self._orders_seen)
        if state is None:
            return calls
        for name in _LEDGER_COUNTERS:
            setattr(calls, name, int(state["ledger"][name]))
        orders = [int(order) for order in state["orders"]]
        calls.orders.extend(orders)
        self._orders_seen += len(orders)
        return calls

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
        record = _watch.record_of(monitor.check, native)
        edge = self._native.edge_record()
        if monitor.transverse is not None and edge is not None:
            record["transverse"] = _transverse.record_of(monitor.transverse, edge)
        return record

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
                cast("dict[str, object]", held["edge_record"]),
            )
        except _core.SessionLifecycleError as error:
            raise SessionError(str(error)) from None

    def undelivered(self, ticket: ExportTicket) -> tuple[BackendHistory, StepStats]:
        """Return the attempts and rows made since the previous event, once.

        Every native batch delivers its whole chunk with its event (SEP 47 section
        5.3), and nothing advances between batches; only a batch whose advance a signal
        or a listener's exception ended before it was returned leaves its chunk here.

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
        held, self._held = self._held, None
        if held is not None and held[0] == ticket.receipt.sequence:
            return self._history(held[1].attempts), self._chunk_rows(held[1])
        return self._history(np.empty((0, 5))), self._no_rows()

    def pending_ticket(self) -> ExportTicket | None:
        """Return a ticket for the committed state while a batch is pending.

        Returns
        -------
        solprop.backends.events.ExportTicket or None
            A ticket bound to the pending batch, or ``None`` when none is pending, the
            session is closed or its state poisoned: the native state exports only
            while its batch is.
        """
        if self.closed or self._poisoned:
            return None
        status = self._native.status()
        if _STATES[status.state] is not SessionState.PENDING or self._sequence == 0:
            return None
        return ExportTicket(
            Receipt(self._id, self._sequence), status.generation, status.z
        )

    def failure_ticket(self) -> ExportTicket | None:
        """Return a ticket for the committed state after a failure outside the session.

        When a batch is pending, the ticket is bound to it (:meth:`pending_ticket`);
        when none is, after an acknowledgment, the native session issues a failed
        terminal batch at its committed state, so that the state can be exported for a
        terminal checkpoint and nothing advances again, as the reference's
        :meth:`solprop.backends.host.session.HostSession.pending_ticket` does. The
        service loop's fallback for a failure between batches (a service failure, a
        signal); a session that never issued a batch has nothing to bind it to.

        Returns
        -------
        solprop.backends.events.ExportTicket or None
            The ticket, or ``None`` once the session ended, closed or was poisoned, or
            before its first batch.
        """
        if self.closed or self._poisoned or self._sequence == 0:
            return None
        failed = self._native.fail_outside()
        if failed is not None:
            # Its chunk, empty unless an aborted advance left attempts behind, is
            # delivered once, through `undelivered`.
            self._sequence = failed.sequence
            self._held = (failed.sequence, failed)
        return self.pending_ticket()

    @property
    def in_flight(self) -> tuple[float, float] | None:
        """Return the step a failure left in flight, or ``None``.

        A low-storage step overwrites the state in place, so a failure partway leaves
        neither of its ends: the session is poisoned, and the service loop falls back
        to its latest running checkpoint (SEP 47 section 5.5), naming this step.

        Returns
        -------
        tuple or None
            Where the abandoned step started and its length, m, or ``None``.
        """
        return self._abandoned

    @property
    def reads_transverse_edge(self) -> bool:
        """Return whether the session's backend reads a transverse edge (SEP 40).

        Returns
        -------
        bool
            True: the native session reads the edge from its resident state and
            judges it natively.
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
        refusal = destination_refusal(out, self._shape)
        if refusal is not None:
            raise BackendError(refusal)

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

    def _tally(
        self,
        counters: _counters.Counters,
        batch: _core.SessionBatch,
        chunk: BackendHistory,
        limit: int,
    ) -> None:
        """Count a delivered batch for the benchmarks: one return to Python.

        Parameters
        ----------
        counters : solprop_native._counters.Counters
            The active counters.
        batch : solprop_native._core.SessionBatch
            The batch.
        chunk : solprop.backends.protocol.BackendHistory
            Its attempts.
        limit : int
            The ``advance``'s attempt bound.
        """
        event = batch.event
        attempts = chunk.n_steps
        rows = int(batch.row_z.size)
        name = _EVENT_NAMES[event.kind]
        if event.kind == _PROGRESS:
            # The native session's yield (`Session::advance`): a full chunk buffer
            # first, then the attempt bound, else the time bound.
            if attempts >= ATTEMPT_CAPACITY or rows >= ROW_CAPACITY:
                name += ":buffer"
            elif attempts >= limit:
                name += ":work_limit"
            else:
                name += ":deadline"
        calls = int(self._native.status().march_calls)
        made, self._calls_counted = calls - self._calls_counted, calls
        counters.session_batch(
            name,
            attempts,
            attempts - chunk.n_rejected,
            rows=rows,
            evaluations=rows + int(event.has_stats),
            plane=event.kind == _OBSERVATION
            and _KINDS[event.observation] is not ObservationKind.LAUNCH,
            march=self._march_name,
            calls=made,
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
            self._stepper.controller,
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
            notice = _watch.Notice.of(event.notice)
            if notice.kind is _watch.NoticeKind.REMOVAL:
                self._log_finish()
            warning = self._warning(event.notice)
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
        # A stop or failure without a ticket left a low-storage step in flight: the
        # state is neither end of it and claims no checkpoint (SEP 47 section 5.5).
        self._poisoned = ticket is None
        if self._poisoned:
            self._abandoned = (event.z, self._abandoned_step(event))
        return Terminal(
            outcome,
            event.z,
            Eligibility.POISONED if self._poisoned else Eligibility.ELIGIBLE,
            self._error(event, outcome),
            ticket,
            chunk,
            rows,
        )

    def _budget_error(self, event: _core.SessionEvent) -> BaseException:
        """Return the error of a trial the call budget could not pay for.

        Formed from the session's own ledger, position and committed field, as the
        backend's march forms it from its own (:func:`~solprop.backends.multistep.
        budget_error`).

        Parameters
        ----------
        event : solprop_native._core.SessionEvent
            The terminal record, its batch pending.

        Returns
        -------
        BaseException
            The budget's error.
        """
        calls = self._sync_calls()
        assert calls is not None
        field = np.empty(self._shape, dtype=np.complex128)
        self._native.export(self._sequence, event.generation, False, False, None, field)
        return budget_error(
            self._max_calls,
            calls.total,
            float(event.z),
            int(event.cost),
            field,
        )

    def _abandoned_step(self, event: _core.SessionEvent) -> float:
        """Return the length of the step a poisoned terminal event left in flight.

        Parameters
        ----------
        event : solprop_native._core.SessionEvent
            The terminal record.

        Returns
        -------
        float
            The diverged step's length, or the schedule's step at the failure's cursor,
            m.
        """
        if event.reason == _DIVERGED:
            return float(event.replay_dz)
        schedule = self._stepper.schedule
        assert schedule is not None  # only a replayed low-storage step is in place
        return float(schedule.steps[int(event.cursor)])

    def _warning(self, native: Mapping[str, object]) -> Warning:
        """Return a warning barrier's warning, the monitor's or its edge's.

        Parameters
        ----------
        native : collections.abc.Mapping
            ``SessionEvent.notice``.

        Returns
        -------
        Warning
            The warning, as solprop's monitor or edge issues it.
        """
        monitor = self._monitor
        assert monitor is not None
        notice = _watch.Notice.of(native)
        if notice.kind is _watch.NoticeKind.TRANSVERSE:
            assert monitor.transverse is not None
            edge = _transverse.EdgeNotice.of(
                cast("Mapping[str, object]", native["edge"])
            )
            return _transverse.notice_warning(monitor.transverse, edge)
        return _watch.notice_warning(monitor.check, notice)

    def _log_finish(self) -> None:
        """Log the edge's verdict, then the monitor's removal and guard peaks, once.

        As :meth:`solprop.monitor.BoundaryMonitor.finish` logs them at the end (D5).
        """
        if self._finish_logged or self._monitor is None:
            return
        self._finish_logged = True
        edge = self._native.edge_record()
        if edge is not None:
            _transverse.log_finish(edge)
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
        if notice.kind is _watch.NoticeKind.TRANSVERSE and self._monitor is not None:
            assert self._monitor.transverse is not None
            edge = _transverse.EdgeNotice.of(
                cast("Mapping[str, object]", event.notice["edge"])
            )
            return _transverse.notice_error(self._monitor.transverse, edge)
        if notice.kind is not _watch.NoticeKind.NONE and self._monitor is not None:
            return _watch.notice_error(self._monitor.check, notice)
        if event.schedule_exhausted:
            schedule = self._stepper.schedule
            end = math.nan if schedule is None else float(schedule.z_end)
            return BackendError(
                f"the replayed schedule ends at {end:.6g} m and cannot reach "
                f"z = {event.target:.6g} m"
            )
        # A statistics evaluation the native plan refused: the reference's exception.
        failure = BackendError(event.message)
        refused = translated(failure)
        if refused is not failure:
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
        if self._adams and int(event.reason) == _BUDGET:
            return self._budget_error(event)
        driver = self._driver(self._backend)
        if self._low_storage and event.reason == _DIVERGED:
            return _diverged(driver.name, self._stepper, event)
        schedule = self._stepper.schedule
        try:
            if self._adams:
                driver._adams._raise(int(event.reason), result)
            else:
                # A native exception's message and class, as the Runge-Kutta march
                # maps them; a diverged replay's names the cursor and the position.
                _raise_rk(
                    int(event.reason),
                    result,
                    name=driver.name,
                    dz_min=float(self._stepper.dz_min),
                    at=(float(event.z), int(event.cursor)),
                    n_steps=0 if schedule is None else schedule.n_steps,
                )
        except Exception as error:
            return error
        return RuntimeError(event.message or "the native session failed")


class NativePrepared:
    """The native backend's prepared execution (SEP 47 section 5.2).

    The backend was built for the problem and a request -- it lowered the method, the
    error control, the itinerary, the planes and the statistics then -- so preparing
    binds the problem and the request one run serves, a request that differs from the
    backend's own only in its start, retention, monitor and history
    (:meth:`~solprop.translate.run_request.RunRequest.for_run`). Each :meth:`start`
    opens one :class:`NativeSession` over the backend, which refuses by name what it
    does not run.

    Parameters
    ----------
    driver : solprop_native.backend.NativeBackend
        The constructed backend; it stays its caller's unless a session owns it.
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's request.
    """

    def __init__(
        self,
        driver: NativeBackend,
        problem: PropagationProblem,
        request: RunRequest,
    ) -> None:
        """Bind the problem and the request to the backend.

        Raises
        ------
        solprop.backends.session.SessionError
            If the request is not one run of the backend's own
            (:func:`~solprop.backends.session_core.foreign_request`).
        solprop.exceptions.BackendError
            If the request asks for statistics the problem does not carry
            (:func:`~solprop.backends.session.statistics_unavailable`), or naming a
            varying metric's statistics
            (:func:`~solprop_native._statistics.metric_refusal`), or if the backend
            is a subclass, whose overrides the native session would not call.
        """
        check_preparation(driver.name, driver.request, problem, request)
        if type(driver) is not NativeBackend:
            msg = (
                f"native cannot prepare this run: the native session runs "
                f"NativeBackend itself, not the subclass {type(driver).__name__}, "
                "whose overrides it would not call"
            )
            raise BackendError(msg)
        stats = request.observation.stats
        refused = metric_refusal(problem, (*stats.names, *stats.periodic))
        if refused is not None:
            msg = f"native cannot prepare this run: native refuses {refused}"
            raise BackendError(msg)
        self._backend = driver
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
        return self._backend.name

    def start(
        self,
        start: Start,
        *,
        listener: Callable[[float, float, bool], None] | None = None,
        started: Callable[[float, float, int], None] | None = None,
        stop: StopFlag | None = None,
        owns: bool = True,
    ) -> RunSession:
        """Start the native session from an initial state or a checkpoint.

        Parameters
        ----------
        start : solprop.backends.session.Start
            Where the run starts; the session takes its field.
        listener : callable or None, optional
            Called with each attempt as it is made; the native session then yields
            after every attempt.
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
            The session.

        Raises
        ------
        solprop.backends.session.SessionError
            If this preparation is closed.
        """
        if self._closed:
            msg = "the prepared execution is closed"
            raise SessionError(msg)
        return NativeSession(
            self._problem,
            self._request,
            start,
            stop=stop,
            listener=listener,
            backend=self._backend,
            owns=owns,
            started=started,
        )

    def metadata(self) -> Mapping[str, object]:
        """Return the execution metadata recorded with the run.

        Returns
        -------
        collections.abc.Mapping
            The backend's own (:meth:`solprop.backends.protocol.Backend.metadata`).
        """
        return self._backend.metadata()

    def memory_ledger(self) -> Mapping[str, int]:
        """Return every retained buffer by name, bytes.

        Returns
        -------
        collections.abc.Mapping
            The backend's ledger.
        """
        return self._backend.memory_ledger()

    def close(self) -> None:
        """Refuse further starts and close the backend it was prepared on; idempotent.

        A caller that keeps the backend for later runs leaves the preparation open: a
        session it started with ``owns=False`` never closes the backend.
        """
        if self._closed:
            return
        self._closed = True
        self._backend.close()
