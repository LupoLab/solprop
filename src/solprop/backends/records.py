# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The run records: what one run's march, session and monitor read (SEP 52 §5.1).

Each record is a frozen dataclass formed once per run by the shared builders here, from
the :class:`~solprop.translate.problem.PropagationProblem`, the
:class:`~solprop.translate.run_request.RunRequest` and the
:class:`~solprop.backends.session.Start`, so that every number either backend uses has
one source. :func:`run_records` forms them all:

- the method's policy: :class:`~solprop.stepper.multistep.MarchPolicy` for the Adams
  march (:func:`adams_policy`), :class:`RungeKuttaPolicy` for the Runge-Kutta march and
  the low-storage replay (:func:`runge_kutta_policy`), and the low-storage march's
  :class:`LowStorageSettings`;
- the schedule, when one is fixed (a :class:`~solprop.stepper.schedule.FixedSchedule`);
- the :class:`PlanePlan`: the planes, step planes, statistics cadence, replay length and
  the native session's capacities;
- the boundary monitor's :class:`MonitorRecords`: its :class:`MonitorConfig`, a
  transverse edge's :class:`EdgeConfig`, and a resumed run's
  :class:`MonitorContinuation`.

The native adapter converts them into what its extension reads; the reference reads
them where it would otherwise derive the same facts. :class:`Notice` and
:class:`EdgeNotice` are the backend-neutral form of a session's boundary verdicts.
The statistics lowering stays each backend's own: the native plan is lowered in C++,
and the reference evaluates the :class:`~solprop.stats.StatsSpec` directly.

Nothing here imports a backend, so the records are available without the native
package installed.
"""

import dataclasses
import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
from jaxtyping import Float, UInt

from solprop.backends.exponential import carrier_reference
from solprop.backends.memory import low_storage_flows
from solprop.monitor import (
    _FLOOR,  # the floor the monitor judges by
    FLOOR_MARGIN,
    _count,  # one refusal of a count
)
from solprop.spatial_absorber import RAMP_K_FLOOR
from solprop.stepper.controller import START_GROWTH, PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import MarchPolicy, Multistep, march_policy
from solprop.stepper.schedule import LANDING_ULPS, FixedSchedule
from solprop.translate.routes import transverse_of
from solprop.transverse_monitor import BACKSTOP_READS, NOISE_FACTOR, REACH_GROWTH, READS

if TYPE_CHECKING:
    from solprop.backends.session import Start
    from solprop.boundaries import SplitCrossing
    from solprop.monitor import BoundaryMonitor
    from solprop.stepper.settings import StepperSettings
    from solprop.translate.problem import PropagationProblem
    from solprop.translate.run_request import RunRequest
    from solprop.transverse_monitor import TransverseEdge, TransverseStatus

__all__ = [
    "ATTEMPT_CAPACITY",
    "CALL_ATTEMPTS",
    "CALL_SECONDS",
    "PLANE_COLUMNS",
    "ROW_CAPACITY",
    "STATUSES",
    "EdgeConfig",
    "EdgeNotice",
    "EdgeNoticeKind",
    "EdgeResumption",
    "LowStorageSettings",
    "MonitorConfig",
    "MonitorContinuation",
    "MonitorRecords",
    "Notice",
    "NoticeKind",
    "PlanePlan",
    "Resumption",
    "RunRecords",
    "RungeKuttaPolicy",
    "adams_policy",
    "collar_rests",
    "declared",
    "edge_config",
    "edge_resumption",
    "low_storage_settings",
    "monitor_config",
    "monitor_continuation",
    "monitor_records",
    "plane_plan",
    "resumption",
    "run_records",
    "runge_kutta_policy",
]

CALL_ATTEMPTS = 256
"""A march call's most attempts, as the native marches bound them (SEP 39)."""

CALL_SECONDS = 0.1
"""A march call's most wall time, s."""

ATTEMPT_CAPACITY = 4096
"""The most attempts the native session holds between two events; a full buffer yields
a progress event carrying them (SEP 47 section 5.3)."""

ROW_CAPACITY = 1024
"""The most periodic rows it holds between two events, likewise."""

PLANE_COLUMNS = ("z", "collar_occupancy", "guard_low", "guard_high", "removed_fraction")
"""The monitor record's plane rows, in the native order (``watch::plane_columns``)."""

STATUSES: tuple["TransverseStatus", ...] = (
    "clear",
    "warned",
    "stopped",
    "insensitive",
    "coarse",
)
"""The transverse statuses in the native order (``transverse::Status``)."""

_NUMBERS = (
    "launch_energy",
    "launch_guard_low",
    "launch_guard_high",
    "launch_collar",
    "guard_low_max",
    "guard_high_max",
    "guard_low_at",
    "guard_high_at",
    "collar_max",
    "raw_removed_fraction",
    "removed_signed",
    "removed_fraction",
    "floor_removed_fraction",
    "raw_guard_max",
)
"""The numbers a resume reads from an earlier monitor record (``watch::Resumption``)."""

_COUNTS = ("steps", "skipped", "collar_left_out_checks")
"""The counts it reads, each whole or refused (``solprop.monitor._count``)."""

_EDGE_NUMBERS = (
    "launch_energy",
    "noise_allowance",
    "raw_ramp_removed",
    "noise_ramp_removed",
    "noise_share_in_ramp",
    "declared_share_in_ramp",
    "spatial_removed",
    "current",
    "current_baseline",
    "strip_baseline",
    "strip_floor",
    "strip_max",
    "returned_max",
    "largest_gap",
    "warned_at",
)
"""The numbers a resume reads from an earlier edge record (``transverse::Resumption``).
"""


# -- the method's policy ------------------------------------------------------------


@dataclass(frozen=True)
class RungeKuttaPolicy:
    """Every number the Runge-Kutta march reads and never changes during a run (SEP 45).

    The controller's decisions read the controller, the floor, the cap and the start-up
    flag (:mod:`solprop.backends.rk_core`); the landings read the rest, the fields of
    :class:`~solprop.stepper.multistep.MarchPolicy` of the same names. The native march
    receives it whole, with the controller's fields, and holds no default of its own.

    Attributes
    ----------
    controller : solprop.stepper.controller.PIController
        The controller, whose gains, factors, lag, estimator order, tolerances and
        acceptance rule the decisions read.
    dz_min, dz_max : float
        The floor and the cap on a proposal, m.
    auto_init : bool
        Whether the first accepted step takes the start-up rule.
    stretch, z_end : float
        The landing stretch (a ratio) and the run's end, m.
    save_planes : tuple of float
        The planes to land on, m; empty when the planes are interpolated.
    interpolate : bool
        Whether the save planes are observed densely.
    hard_kinks : tuple of float
        The operator's hard kinks, m.
    breakpoints : tuple of float
        The edges whose intervals cap a step, m: the operator's breakpoints with the
        interval cap, otherwise only its first and last.
    landing_ulps : int
        :data:`~solprop.stepper.schedule.LANDING_ULPS`.
    start_growth : float
        :data:`~solprop.stepper.controller.START_GROWTH`.
    quiet_interval : float or None
        The boundary monitor's interval, m, which caps a step while the collar rests;
        ``None`` for a session, whose monitor holds its own.
    """

    controller: PIController
    dz_min: float
    dz_max: float
    auto_init: bool
    stretch: float
    z_end: float
    save_planes: tuple[float, ...]
    interpolate: bool
    hard_kinks: tuple[float, ...]
    breakpoints: tuple[float, ...]
    landing_ulps: int
    start_growth: float
    quiet_interval: float | None


def runge_kutta_policy(
    problem: "PropagationProblem", stepper: "StepperSettings"
) -> RungeKuttaPolicy:
    """Return the Runge-Kutta march's policy of one run.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem, whose operator's breakpoints and hard kinks the landings read.
    stepper : solprop.stepper.settings.StepperSettings
        The run's stepper settings.

    Returns
    -------
    RungeKuttaPolicy
        The record, with no quiet interval.
    """
    breakpoints = np.asarray(problem.linear.phi_breakpoints, dtype=np.float64)
    if not stepper.interval_cap:
        # The cumulative table makes the flow exact across its intervals, so only the
        # table's two ends bound a step (SEP 14 section 4.1).
        breakpoints = breakpoints[[0, -1]]
    interpolate = bool(stepper.interpolate_planes)
    return RungeKuttaPolicy(
        controller=stepper.controller,
        dz_min=float(stepper.dz_min),
        dz_max=float(stepper.dz_max),
        auto_init=bool(stepper.auto_init),
        stretch=float(stepper.stretch),
        z_end=float(stepper.z_end),
        # Interpolated planes are stepped across (SEP 14 section 4.3).
        save_planes=()
        if interpolate
        else tuple(float(z) for z in np.asarray(stepper.save_planes)),
        interpolate=interpolate,
        hard_kinks=tuple(float(z) for z in np.asarray(problem.linear.hard_kinks)),
        breakpoints=tuple(float(z) for z in breakpoints),
        landing_ulps=LANDING_ULPS,
        start_growth=float(START_GROWTH),
        quiet_interval=None,
    )


def adams_policy(
    problem: "PropagationProblem",
    stepper: "StepperSettings",
    method: Multistep,
    *,
    carrier: bool,
) -> MarchPolicy:
    """Return the Adams march's policy of one run (SEP 39 section 4.2).

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem: the operator's breakpoints and kinks, and the in-stage temporal
        collar, whose largest rate bounds each order's stable step in the exponential
        formulation.
    stepper : solprop.stepper.settings.StepperSettings
        The run's stepper settings.
    method : solprop.stepper.multistep.Multistep
        The method.
    carrier : bool
        Whether the engine's carrier frame is active
        (:class:`~solprop.backends.exponential.CarrierReference`).

    Returns
    -------
    solprop.stepper.multistep.MarchPolicy
        The record, with no quiet interval.
    """
    responses = problem.responses
    collar = responses.temporal_absorber
    collar_rate = (
        0.5 * float(np.max(collar))
        if collar is not None
        and responses.absorber_scheme == "nonlinear"
        and method.formulation == "exponential"
        else 0.0
    )
    linear = problem.linear
    return march_policy(
        method,
        dz_min=stepper.dz_min,
        dz_max=stepper.dz_max,
        stretch=stepper.stretch,
        z_end=stepper.z_end,
        save_planes=np.asarray(stepper.save_planes, dtype=np.float64).tolist(),
        hard_kinks=np.asarray(linear.hard_kinks, dtype=np.float64).tolist(),
        breakpoints=np.asarray(linear.phi_breakpoints, dtype=np.float64).tolist(),
        interpolate=bool(stepper.interpolate_planes),
        collar_rate=collar_rate,
        carrier=carrier,
    )


@dataclass(frozen=True)
class LowStorageSettings:
    """What the low-storage replay reads beside the Runge-Kutta policy (SEP 24).

    Attributes
    ----------
    flows : int
        The flow volumes a step holds on a CPU
        (:func:`~solprop.backends.memory.low_storage_flows`): 0 recomputes each flow,
        1 holds a constant operator's table ``exp(h_t D / 2)``, 2 a varying operator's
        ``P`` and ``Q``.
    separable : bool
        Whether the operator is a separable geometry's, whose increments are the
        distances themselves.
    origin : float
        The operator's origin, ``linear.z_span[0]``, m, from which a table is built.
    """

    flows: int
    separable: bool
    origin: float


def low_storage_settings(
    problem: "PropagationProblem", stepper: "StepperSettings"
) -> LowStorageSettings | None:
    """Return the low-storage replay's settings, or ``None`` for another march.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    stepper : solprop.stepper.settings.StepperSettings
        The run's stepper settings.

    Returns
    -------
    LowStorageSettings or None
        The settings, when the run's integrator is the low-storage march.
    """
    if not isinstance(stepper.integrator, LowStorageRungeKutta):
        return None
    return LowStorageSettings(
        flows=low_storage_flows(problem, stepper),
        separable=problem.linear.separable is not None,
        origin=float(problem.linear.z_span[0]),
    )


def collar_rests(problem: "PropagationProblem", stepper: "StepperSettings") -> bool:
    """Return whether a run's stages may leave a quiet temporal collar out (SEP 21).

    One rule for every backend: a modal or radial stage with the in-stage
    (``"nonlinear"``) scheme and a temporal collar alone. The low-storage march, coupled
    flows, the general modal route, a real-space collar sharing the term, and the
    Cartesian, line and planar routes keep the collar in every stage (SEP 21 D15).

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    stepper : solprop.stepper.settings.StepperSettings
        The run's stepper settings.

    Returns
    -------
    bool
        True when the term may be left out while the collar is quiet.
    """
    responses = problem.responses
    geometry = problem.transform.geometry
    resting_geometry = geometry is None or (
        geometry.kind == "radial" and not np.any(geometry.absorber_rate)
    )
    return (
        not isinstance(stepper.integrator, LowStorageRungeKutta)
        and responses.temporal_absorber is not None
        and responses.absorber_scheme == "nonlinear"
        and problem.linear.coupled_flow is None
        and problem.transform.modal_tables is None
        and resting_geometry
    )


# -- the plane plan -------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PlanePlan:
    """Where a run is observed and how its session buffers (SEP 47 section 5.3).

    Equality is by identity (``eq=False``) because ``planes`` is an array.

    Attributes
    ----------
    z : float
        Where the run starts, m.
    planes : numpy.ndarray
        The save planes, m, shape ``(n_z,)``.
    next_plane : int
        The first save plane strictly beyond `z`.
    launch_pending : bool
        Whether the plane at `z` is owed.
    step_planes : int
        Save every this many accepted steps; 0 for none.
    dense : bool
        Whether the planes are interpolated rather than landed on.
    period : int
        Accepted steps between the periodic statistics rows; 0 without rows.
    replay_steps : int
        A fixed schedule's steps; 0 without one.
    replay_length : float
        A fixed schedule's total length, m; 0.0 without one.
    attempt_capacity, row_capacity : int
        The most attempts and periodic rows the native session holds between two
        events (:data:`ATTEMPT_CAPACITY`, :data:`ROW_CAPACITY`).
    call_attempts : int
        A march call's most attempts (:data:`CALL_ATTEMPTS`).
    call_seconds : float
        A march call's most wall time, s (:data:`CALL_SECONDS`).
    """

    z: float
    planes: Float[np.ndarray, " n_z"]
    next_plane: int
    launch_pending: bool
    step_planes: int
    dense: bool
    period: int
    replay_steps: int
    replay_length: float
    attempt_capacity: int
    row_capacity: int
    call_attempts: int
    call_seconds: float

    @property
    def fixed_positions(self) -> tuple[float, ...] | None:  # numpydoc ignore=RT01
        """Every position a plane is observed at, m, when the plan fixes them all.

        The start and the save planes; ``None`` with step planes or interpolated
        planes, whose positions the march decides.
        """
        if self.dense or self.step_planes:
            return None
        return (self.z, *(float(z) for z in self.planes))


def plane_plan(request: "RunRequest", start: "Start") -> PlanePlan:
    """Return a run's plane plan.

    Parameters
    ----------
    request : solprop.translate.run_request.RunRequest
        The run's request.
    start : solprop.backends.session.Start
        Where the run starts.

    Returns
    -------
    PlanePlan
        The plan, with the native session's capacities.
    """
    observation = request.observation
    spec = observation.stats
    schedule = None if request.replay is None else request.replay.schedule
    return PlanePlan(
        z=float(start.z),
        planes=np.ascontiguousarray(observation.save_planes, dtype=np.float64),
        next_plane=int(start.plane_index),
        launch_pending=bool(start.plane_pending),
        step_planes=int(observation.step_planes),
        dense=bool(observation.dense),
        period=int(spec.period) if spec.periodic else 0,
        replay_steps=0 if schedule is None else int(schedule.n_steps),
        replay_length=0.0 if schedule is None else float(schedule.total_length),
        attempt_capacity=ATTEMPT_CAPACITY,
        row_capacity=ROW_CAPACITY,
        call_attempts=CALL_ATTEMPTS,
        call_seconds=CALL_SECONDS,
    )


# -- the boundary monitor -------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class MonitorConfig:
    """A session's boundary monitor's configuration (``watch::Config``).

    Equality is by identity (``eq=False``) because ``rate`` is an array.

    Attributes
    ----------
    policy : collections.abc.Mapping
        The predictions' policy (:meth:`~solprop.monitor.BoundaryMonitor.watch_policy`'s
        fields), its floor regions and launch floor zero: they are formed at the
        launch.
    floor_share : float
        No light is judged below this share of the launch energy.
    floor_margin : float
        :data:`solprop.monitor.FLOOR_MARGIN`.
    budget : float
        The budget the finish's removal is judged against.
    rate : numpy.ndarray
        The collar's power rate on the coarse axis, 1/m, shape ``(n_t,)``.
    can_rest : bool
        Whether the route's stages may leave a quiet collar out (:func:`collar_rests`).
    """

    policy: Mapping[str, object]
    floor_share: float
    floor_margin: float
    budget: float
    rate: Float[np.ndarray, " n_t"]
    can_rest: bool


def monitor_config(
    monitor: "BoundaryMonitor", crossing: "SplitCrossing | None", *, can_rest: bool
) -> MonitorConfig:
    """Return a fresh monitor's configuration for a run.

    The predictions' policy is the monitor's own with the split collar's crossing limit
    the run gives it.

    Parameters
    ----------
    monitor : solprop.monitor.BoundaryMonitor
        The run's fresh monitor.
    crossing : solprop.boundaries.SplitCrossing or None
        The split collar's crossing limit, under the split scheme.
    can_rest : bool
        Whether the route's stages may leave a quiet collar out.

    Returns
    -------
    MonitorConfig
        The configuration.
    """
    configured = dataclasses.replace(monitor, crossing=crossing)
    policy = configured.watch_policy()
    policy["floor_regions"] = [0.0, 0.0, 0.0, 0.0]
    policy["launch_floor"] = 0.0
    return MonitorConfig(
        policy=policy,
        floor_share=_FLOOR,
        floor_margin=FLOOR_MARGIN,
        budget=float(monitor.check.budget),
        rate=np.ascontiguousarray(monitor.weights[3], dtype=np.float64),
        can_rest=bool(can_rest),
    )


@dataclass(frozen=True, eq=False)
class EdgeConfig:
    """A session's transverse edge's configuration (``transverse::Config``, SEP 40).

    Every threshold is formed as :class:`~solprop.transverse_monitor.TransverseEdge`
    forms it, so that a session compares the same numbers. Equality is by identity
    (``eq=False``) because it holds arrays.

    Attributes
    ----------
    faces : tuple of str
        The faces judged, in the edge's order.
    warn, stop : float
        ``budget ** 2`` and ``(stop_factor * budget) ** 2``.
    length : float
        The run's length, m.
    guard : float
        The guard's width.
    k_min : float
        The safety ramp's least transverse wavenumber, 1/m.
    k_floor : float
        :data:`~solprop.spatial_absorber.RAMP_K_FLOOR`.
    reads, backstop_reads, reach_growth, noise_factor : float
        :mod:`solprop.transverse_monitor`'s constants of those names.
    columns : numpy.ndarray
        The spectral columns' weights, shape ``(n_columns,)``.
    kappa : numpy.ndarray
        Each column's transverse wavenumber, 1/m, shape ``(n_kappa,)``.
    kappa_order : numpy.ndarray
        Its ascending order, ``uint64``, shape ``(n_kappa,)``.
    design : bool
        Whether a safety ramp is designed.
    ramp_rate : numpy.ndarray
        The ramp's rate by column, 1/m; empty without a design.
    collar, collar_rate : bool
        Whether the design has a collar, and a collar rate.
    declared : bool
        Whether the run declares content above ``discard_above`` (:func:`declared`).
    longest_step : float or None
        A fixed schedule's longest step, m; ``None`` without one.
    noise_share : float
        The input noise's share of the weights on the first face.
    noise_modes : float
        The input noise's count of modes.
    """

    faces: tuple[str, ...]
    warn: float
    stop: float
    length: float
    guard: float
    k_min: float
    k_floor: float
    reads: float
    backstop_reads: float
    reach_growth: float
    noise_factor: float
    columns: Float[np.ndarray, " n_columns"]
    kappa: Float[np.ndarray, " n_kappa"]
    kappa_order: UInt[np.ndarray, " n_kappa"]
    design: bool
    ramp_rate: Float[np.ndarray, " n_ramp"]
    collar: bool
    collar_rate: bool
    declared: bool
    longest_step: float | None
    noise_share: float
    noise_modes: float


def declared(problem: "PropagationProblem") -> bool:
    """Return whether a run declares content above ``discard_above``.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.

    Returns
    -------
    bool
        True when its transverse specification sets ``discard_above``.
    """
    design = transverse_of(problem.route)
    return design is not None and design.specification.discard_above is not None


def edge_config(
    edge: "TransverseEdge",
    *,
    steps: Float[np.ndarray, " n_steps"] | None,
    declared: bool,
) -> EdgeConfig:
    """Return a fresh transverse edge's configuration for its launch's arguments.

    Parameters
    ----------
    edge : solprop.transverse_monitor.TransverseEdge
        The run's fresh edge.
    steps : numpy.ndarray or None
        A fixed schedule's steps, m; ``None`` for an adaptive run.
    declared : bool
        Whether the run declares content above ``discard_above``.

    Returns
    -------
    EdgeConfig
        The configuration.
    """
    design = edge.design
    faces = edge._faces  # the edge's own geometry
    weights = edge._weights
    first = faces[0].edge
    modes = max(int(np.count_nonzero(first)), 1) * max(int(edge.band.size), 1)
    return EdgeConfig(
        faces=tuple(face.name for face in faces),
        warn=edge.budget**2,
        stop=(edge.stop_factor * edge.budget) ** 2,
        length=float(edge.length),
        guard=float(edge._guard),
        k_min=float(edge.k_min),
        k_floor=float(RAMP_K_FLOOR),
        reads=float(READS),
        backstop_reads=float(BACKSTOP_READS),
        reach_growth=float(REACH_GROWTH),
        noise_factor=float(NOISE_FACTOR),
        columns=np.ascontiguousarray(
            edge._columns,
            dtype=np.float64,
        ),
        kappa=np.ascontiguousarray(
            edge._kappa,
            dtype=np.float64,
        ),
        kappa_order=np.ascontiguousarray(
            edge._kappa_order,
            dtype=np.uint64,
        ),
        design=design is not None,
        ramp_rate=np.empty(0, dtype=np.float64)
        if design is None
        else np.ascontiguousarray(design.ramp_rate, dtype=np.float64),
        collar=design is not None and design.has_collar,
        collar_rate=design is not None and design.collar_rate is not None,
        declared=bool(declared),
        longest_step=float(np.max(steps)) if steps is not None and steps.size else None,
        noise_share=float(np.sum(weights[first])) / float(np.sum(weights)),
        noise_modes=float(modes),
    )


@dataclass(frozen=True)
class Resumption:
    """What a resume reads of an earlier monitor record (``watch::Resumption``).

    Only the record's types are read: which numbers are restored, their defaults and the
    fallbacks are the session's.

    Attributes
    ----------
    record : bool
        Whether there is a record at all.
    collar_quiet : bool
        ``bool(record.get("collar_quiet", False))``.
    numbers : collections.abc.Mapping
        Each number present in the record, by field; the counts already whole.
    unwatched : tuple or None
        The record's ``unwatched`` stretches, when it lists them.
    resumed_at : tuple or None
        The record's ``resumed_at``, when it lists them.
    planes : tuple or None
        The record's plane rows by :data:`PLANE_COLUMNS` (NaN for ``None``), when they
        have those five columns.
    """

    record: bool
    collar_quiet: bool = False
    numbers: Mapping[str, float] = field(default_factory=dict[str, float])
    unwatched: tuple[tuple[float, ...], ...] | None = None
    resumed_at: tuple[float, ...] | None = None
    planes: tuple[tuple[float, ...], ...] | None = None


def _number(record: Mapping[str, object], name: str) -> float | None:
    """Return a record's number, or ``None`` when it is absent or not a number.

    As ``solprop.monitor._number`` reads it, without its default, which the session's
    reconstruction applies.
    """
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def resumption(record: Mapping[str, object] | None) -> Resumption:
    """Return what a resume reads of an earlier execution's monitor record.

    Read as :meth:`~solprop.monitor.BoundaryMonitor.resume` reads it: each number
    present, the lists, and the plane rows when they have the five columns. A count
    that is not whole is refused as that resume refuses it, once the record's launch
    energy makes it read the counts.

    Parameters
    ----------
    record : collections.abc.Mapping or None
        The ``boundary_monitor`` record of the latest earlier execution, or ``None``.

    Returns
    -------
    Resumption
        What the session restores from.
    """
    if record is None:
        return Resumption(record=False)
    numbers: dict[str, float] = {}
    for name in _NUMBERS:
        value = _number(record, name)
        if value is not None:
            numbers[name] = value
    launch = numbers.get("launch_energy", math.nan)
    if launch > 0.0:
        # `BoundaryMonitor.resume` reads the counts only past its launch test.
        quiet = _number(record, "quiet_checks")
        if quiet is not None and math.isfinite(quiet):
            numbers["quiet_checks"] = float(_count(quiet, "quiet_checks"))
        for name in _COUNTS:
            value = _number(record, name)
            numbers[name] = float(_count(0.0 if value is None else value, name))
    listed = record.get("unwatched")
    resumed = record.get("resumed_at")
    rows = record.get("planes")
    planes = None
    if isinstance(rows, Mapping) and set(cast("Mapping[str, object]", rows)) == set(
        PLANE_COLUMNS
    ):
        columns = cast("Mapping[str, list[float | None]]", rows)
        planes = tuple(
            tuple(math.nan if one is None else float(one) for one in columns[name])
            for name in PLANE_COLUMNS
        )
    return Resumption(
        record=True,
        collar_quiet=bool(record.get("collar_quiet", False)),
        numbers=numbers,
        unwatched=tuple(
            tuple(float(one) for one in cast("list[float]", stretch))
            for stretch in cast("list[object]", listed)
        )
        if isinstance(listed, list)
        else None,
        resumed_at=tuple(float(one) for one in cast("list[float]", resumed))
        if isinstance(resumed, list)
        else None,
        planes=planes,
    )


@dataclass(frozen=True)
class EdgeResumption:
    """What a resume reads of an earlier edge record (``transverse::Resumption``).

    Attributes
    ----------
    record : bool
        Whether there is a record at all.
    numbers : collections.abc.Mapping
        Each number present in the record, by field.
    edge_baseline, edge_max, edge_max_raw : tuple of float or None
        The per-face lists, once the record's launch energy and baseline make the
        resume read them; each ``None`` otherwise or without a value per face.
    reads : int or None
        The reads, made whole, under the same condition.
    transverse_status : str or None
        The record's status, when it names one.
    """

    record: bool
    numbers: Mapping[str, float] = field(default_factory=dict[str, float])
    edge_baseline: tuple[float, ...] | None = None
    edge_max: tuple[float, ...] | None = None
    edge_max_raw: tuple[float, ...] | None = None
    reads: int | None = None
    transverse_status: str | None = None


def _per_face(
    record: Mapping[str, object], name: str, faces: int
) -> tuple[float, ...] | None:
    """Return a record's per-face list, or ``None`` when it has no value per face.

    As ``TransverseEdge.resume``'s ``numbers`` reads it.
    """
    value = record.get(name)
    if not isinstance(value, list) or len(cast("list[object]", value)) != faces:
        return None
    return tuple(float(cast("float", one)) for one in cast("list[object]", value))


def edge_resumption(record: object, faces: int) -> EdgeResumption:
    """Return what a resume reads of an earlier execution's transverse edge record.

    Read as :meth:`~solprop.transverse_monitor.TransverseEdge.resume` reads it: each
    number present, and, once the record's launch energy and baseline make that resume
    read further, each per-face maximum that has a value per face and the reads, made
    whole as it makes them.

    Parameters
    ----------
    record : object
        The ``transverse`` entry of the earlier monitor record, or anything else when
        there is none.
    faces : int
        The edge's face count.

    Returns
    -------
    EdgeResumption
        What the session restores from.
    """
    if not isinstance(record, Mapping):
        return EdgeResumption(record=False)
    mapping = cast("Mapping[str, object]", record)
    numbers: dict[str, float] = {}
    for name in _EDGE_NUMBERS:
        value = _number(mapping, name)
        if value is not None:
            numbers[name] = value
    baseline = _per_face(mapping, "edge_baseline", faces)
    launch = numbers.get("launch_energy", math.nan)
    status = mapping.get("transverse_status")
    if not (launch > 0.0 and baseline is not None):
        # `TransverseEdge.resume` reads the maxima and the count only past its test of
        # the launch and the baseline: a record it ignores is not read further.
        return EdgeResumption(
            record=True,
            numbers=numbers,
            transverse_status=status if isinstance(status, str) else None,
        )
    reads = _number(mapping, "reads")
    return EdgeResumption(
        record=True,
        numbers=numbers,
        edge_baseline=baseline,
        edge_max=_per_face(mapping, "edge_max", faces),
        edge_max_raw=_per_face(mapping, "edge_max_raw", faces),
        reads=int(0.0 if reads is None else reads),
        transverse_status=status if isinstance(status, str) else None,
    )


@dataclass(frozen=True)
class MonitorContinuation:
    """A monitored resume's continuation, as a session reads it.

    Attributes
    ----------
    record : Resumption
        The earlier monitor record's resumption.
    edge : EdgeResumption
        Its transverse edge record's.
    plane : bool
        Whether the run resumes at a saved plane.
    unwatched : tuple of tuple of float
        The ``(start, end)`` stretches, m, run since without a monitor.
    """

    record: Resumption
    edge: EdgeResumption
    plane: bool
    unwatched: tuple[tuple[float, float], ...]


def monitor_continuation(
    monitor: "BoundaryMonitor", continuation: Mapping[str, object]
) -> MonitorContinuation:
    """Return a monitored resume's continuation records.

    Parameters
    ----------
    monitor : solprop.monitor.BoundaryMonitor
        The run's fresh monitor, whose transverse edge gives the face count.
    continuation : collections.abc.Mapping
        As :attr:`solprop.backends.session.Start.monitor`.

    Returns
    -------
    MonitorContinuation
        The records.
    """
    record = continuation.get("record")
    mapping = (
        cast("Mapping[str, object]", record) if isinstance(record, Mapping) else None
    )
    edge = monitor.transverse
    faces = 0 if edge is None else len(edge._faces)
    stretches = cast("Sequence[tuple[float, float]]", continuation.get("unwatched", ()))
    return MonitorContinuation(
        record=resumption(mapping),
        edge=edge_resumption(
            None if mapping is None else mapping.get("transverse"), faces
        ),
        plane=bool(continuation.get("plane", False)),
        unwatched=tuple((float(start), float(end)) for start, end in stretches),
    )


@dataclass(frozen=True, eq=False)
class MonitorRecords:
    """A monitored run's boundary records.

    Attributes
    ----------
    config : MonitorConfig
        The temporal monitor's configuration.
    edge : EdgeConfig or None
        A free-space run's transverse edge's, or ``None`` without one.
    continuation : MonitorContinuation or None
        A monitored resume's continuation, or ``None`` when the start carries none
        (a session may still be continued before it watches).
    """

    config: MonitorConfig
    edge: EdgeConfig | None
    continuation: MonitorContinuation | None


def monitor_records(
    problem: "PropagationProblem", request: "RunRequest", start: "Start"
) -> MonitorRecords | None:
    """Return a run's boundary records, or ``None`` without a monitor.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem: the split collar's crossing limit, the quiet-collar rule and a
        declared transverse specification.
    request : solprop.translate.run_request.RunRequest
        The run's request, with its fresh monitor.
    start : solprop.backends.session.Start
        Where the run starts, with any monitor continuation.

    Returns
    -------
    MonitorRecords or None
        The records.
    """
    monitor = request.boundary.monitor
    if monitor is None:
        return None
    stepper = request.stepper
    edge = monitor.transverse
    schedule = stepper.schedule
    return MonitorRecords(
        config=monitor_config(
            monitor,
            problem.split_crossing if monitor.split else None,
            can_rest=collar_rests(problem, stepper),
        ),
        edge=None
        if edge is None
        else edge_config(
            edge,
            steps=None if schedule is None else np.asarray(schedule.steps),
            declared=declared(problem),
        ),
        continuation=None
        if start.monitor is None
        else monitor_continuation(monitor, start.monitor),
    )


# -- the boundary verdicts ------------------------------------------------------------


class NoticeKind(enum.IntEnum):
    """A boundary monitor's notice's kind (``session::Notice::Kind``).

    Attributes
    ----------
    NONE : int
        No notice.
    GUARD : int
        A guard occupancy, at the launch or at a check.
    REMOVAL : int
        The collar's removal at the end of a run.
    CROSSING : int
        A replay's step passes the split collar's cap.
    READING : int
        The split found nothing where the state holds light.
    DIVISION : int
        Light after a launch that held none, which the monitor divides by.
    TRANSVERSE : int
        The transverse edge's verdict (:class:`EdgeNotice`).
    """

    NONE = 0
    GUARD = 1
    REMOVAL = 2
    CROSSING = 3
    READING = 4
    DIVISION = 5
    TRANSVERSE = 6


@dataclass(frozen=True)
class Notice:
    """A boundary monitor's verdict (``session::Notice``): a warning or a stop.

    Attributes
    ----------
    kind : NoticeKind
        What it is.
    launch : bool
        For a guard: whether it is the launch's judgement.
    z : float
        Where, m.
    value : float
        A guard's amplitude, or the removed fraction.
    cap : float
        A crossing's cap, m.
    step : float
        A crossing's longest step before the next check, m.
    """

    kind: NoticeKind
    launch: bool
    z: float
    value: float
    cap: float
    step: float


class EdgeNoticeKind(enum.IntEnum):
    """A transverse edge's notice's kind (``transverse::Notice::Kind``).

    Attributes
    ----------
    NONE : int
        No notice.
    RAMP_BAND : int
        The launch's share in the safety ramp's band.
    CLIP : int
        A face's clip of the launch.
    INSENSITIVE : int
        The input noise's allowance exceeds the warning.
    COARSE : int
        A fixed schedule's longest step outruns the read spacing.
    FACE : int
        A face's judged share.
    RETURNED : int
        A requested collar's returned share.
    RAMP_REMOVAL : int
        The safety ramp's removal.
    """

    NONE = 0
    RAMP_BAND = 1
    CLIP = 2
    INSENSITIVE = 3
    COARSE = 4
    FACE = 5
    RETURNED = 6
    RAMP_REMOVAL = 7


@dataclass(frozen=True)
class EdgeNotice:
    """A transverse edge's verdict (``transverse::Notice``): a warning or a stop.

    Attributes
    ----------
    kind : EdgeNoticeKind
        What it is.
    stop : bool
        Whether it stops the run.
    face : int
        The face it names, by index.
    value : float
        The share, allowance or step it names.
    z : float
        Where, m.
    spacing : float
        A coarse schedule's read spacing, m.
    warned_at : float
        Where the edge first warned, for a face's stop, m; NaN where it did not.
    """

    kind: EdgeNoticeKind
    stop: bool
    face: int
    value: float
    z: float
    spacing: float
    warned_at: float


# -- one run's records ----------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RunRecords:
    """Every record of one run, formed once by :func:`run_records` (SEP 52 §5.1).

    Attributes
    ----------
    policy : solprop.stepper.multistep.MarchPolicy or RungeKuttaPolicy
        The method's policy: the Adams march's, or the Runge-Kutta march's, which the
        low-storage replay reads too.
    low_storage : LowStorageSettings or None
        The low-storage replay's settings; ``None`` for another march.
    schedule : solprop.stepper.schedule.FixedSchedule or None
        The fixed schedule, when the run replays one.
    planes : PlanePlan
        The plane plan.
    monitor : MonitorRecords or None
        The boundary records; ``None`` without a monitor.
    """

    policy: MarchPolicy | RungeKuttaPolicy
    low_storage: LowStorageSettings | None
    schedule: FixedSchedule | None
    planes: PlanePlan
    monitor: MonitorRecords | None


def run_records(
    problem: "PropagationProblem", request: "RunRequest", start: "Start"
) -> RunRecords:
    """Return one run's records.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's request.
    start : solprop.backends.session.Start
        Where the run starts; its field is not read.

    Returns
    -------
    RunRecords
        The records.

    Examples
    --------
    >>> from solprop.backends.records import run_records
    >>> from solprop.backends.session import Start
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.assembly import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> launch = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8)
    >>> problem, request = assemble(grid, gnlse.mode, [gnlse.kerr], launch.build(grid),
    ...                    controller=PIController(rtol=1e-8, atol=1e-16),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> records = run_records(problem, request, Start.entrance(problem))
    >>> type(records.policy).__name__, records.planes.planes.tolist()
    ('RungeKuttaPolicy', [0.01])
    >>> records.low_storage is None, records.monitor is None
    (True, True)
    """
    stepper = request.stepper
    method = stepper.integrator
    policy: MarchPolicy | RungeKuttaPolicy
    if isinstance(method, Multistep):
        reference = carrier_reference(problem, method)
        policy = adams_policy(
            problem,
            stepper,
            method,
            carrier=reference is not None and reference.active,
        )
    else:
        policy = runge_kutta_policy(problem, stepper)
    return RunRecords(
        policy=policy,
        low_storage=low_storage_settings(problem, stepper),
        schedule=stepper.schedule,
        planes=plane_plan(request, start),
        monitor=monitor_records(problem, request, start),
    )
