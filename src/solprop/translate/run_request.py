# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The immutable run request a backend prepares beside its problem (SEP 47 section 6.3).

The translated problem (:class:`solprop.translate.problem.PropagationProblem`) says
*what* is propagated. A :class:`RunRequest` says *how it is run*: which integration
method, how errors are controlled, which fixed itinerary is replayed, what is observed
and kept, what the boundary watches and which method history is retained. Python
resolves every default before preparation, so a backend never invents a policy, and the
request never changes during a run: stop and snapshot requests travel separately
(:mod:`solprop.backends.session`).

The request separates method selection from error scaling and acceptance, as SEP 47
section 6.3 specifies:

- :class:`IntegrationSpec` names the method -- an embedded Runge-Kutta tableau, the
  low-call Adams family or the fixed low-storage replay -- and its options;
- :class:`ErrorControlSpec` holds the tolerances, the norm, the controller and the
  step limits;
- :class:`ReplaySpec` holds a fixed itinerary, when one is replayed;
- :class:`ObservationRequest` holds the planes, dense output, step planes,
  statistics and field retention;
- :class:`BoundaryRequest` holds the boundary monitor's configuration;
- :class:`HistoryPolicy` says when the method's history is kept for a resume.

:func:`describe_run` builds one from the stepper section translation resolved
(:class:`~solprop.translate.problem.StepperSettings`) and the run's settings;
:func:`solprop.translate.problem.assemble` returns the request for a fresh run beside
its problem, and :meth:`RunRequest.for_run` adapts it to one run's start, retention,
monitor and history. The problem carries none of this (SEP 47 section 5.2).
"""

from __future__ import annotations

import dataclasses
import enum
import functools
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from solprop.exceptions import TranslationError
from solprop.monitor import BoundaryMonitor
from solprop.output.schema import FieldRetention
from solprop.stats import StatsSpec
from solprop.stepper.controller import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import ComparisonMultistep, Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.stepper.tableau import Tableau
from solprop.translate.problem import PropagatorMode, StepperSettings

__all__ = [
    "MARCH_SNAPSHOT_AUTO_BYTES",
    "BoundaryRequest",
    "ErrorControlSpec",
    "HistoryPolicy",
    "HistoryRetention",
    "IntegrationSpec",
    "MethodKind",
    "ObservationRequest",
    "ReplaySpec",
    "RunRequest",
    "describe_run",
]

MARCH_SNAPSHOT_AUTO_BYTES = 256 * 2**20
"""The automatic method-history retention limit, bytes (SEP 47 section 5.5)."""


class MethodKind(enum.Enum):
    """The integration method family.

    Attributes
    ----------
    RUNGE_KUTTA : str
        An embedded Runge-Kutta tableau, adaptive or replayed.
    ADAMS : str
        The low-call Adams family (``Multistep``), including its benchmark-only
        comparison variants, which only the reference runs.
    LOW_STORAGE : str
        The fixed-schedule low-storage RK4IP replay (SEP 24).
    """

    RUNGE_KUTTA = "runge_kutta"
    ADAMS = "adams"
    LOW_STORAGE = "low_storage"


@dataclass(frozen=True)
class IntegrationSpec:
    """Which method integrates the run, and its options.

    Attributes
    ----------
    kind : MethodKind
        The method family.
    tableau : solprop.stepper.tableau.Tableau
        The embedded Runge-Kutta tableau. An Adams run still names the tableau that
        starts it and that its error control is scaled against.
    method : Multistep, ComparisonMultistep, LowStorageRungeKutta or None
        The Adams or low-storage method object, immutable; ``None`` for Runge-Kutta.
    propagator_mode : {'auto', 'tabulate', 'recompute'}
        How the linear propagator is formed.
    """

    kind: MethodKind
    tableau: Tableau
    method: Multistep | ComparisonMultistep | LowStorageRungeKutta | None
    propagator_mode: PropagatorMode

    @property
    def reference_only(self) -> bool:
        """Return whether only the NumPy reference runs this method.

        Returns
        -------
        bool
            True for the benchmark-only comparison family, ``ComparisonMultistep``
            (SEP 47 section 6.3): PEC, PECE, explicit Adams-Bashforth and the matched
            pairs, which native refuses;
            :class:`~solprop.stepper.multistep.Multistep` is the selected method every
            backend runs.
        """
        return isinstance(self.method, ComparisonMultistep)


@dataclass(frozen=True)
class ErrorControlSpec:
    """Tolerances, the controller and the step limits.

    Attributes
    ----------
    controller : solprop.stepper.controller.PIController
        The controller and its norm. A subclass that overrides a decision method is
        a reference-only experiment (SEP 47 section 6.3).
    dz_min : float
        The step floor, m.
    dz_max : float
        The step ceiling, m.
    dz_init : float
        The first proposal, m.
    stretch : float
        The fraction by which a step may stretch to land on a plane.
    auto_init : bool
        Whether the first proposal is chosen from the launch state.
    interval_cap : bool
        Whether a step is capped to the interval it starts in.
    """

    controller: PIController
    dz_min: float
    dz_max: float
    dz_init: float
    stretch: float
    auto_init: bool
    interval_cap: bool


@dataclass(frozen=True)
class ReplaySpec:
    """A fixed itinerary, replayed rather than chosen.

    Attributes
    ----------
    schedule : solprop.stepper.schedule.FixedSchedule
        The step edges.
    """

    schedule: FixedSchedule


@dataclass(frozen=True, eq=False)
class ObservationRequest:
    """What a run observes and keeps.

    Attributes
    ----------
    save_planes : numpy.ndarray
        The requested planes, m, increasing, read-only.
    start_plane : int
        The first save plane strictly beyond the starting state.
    dense : bool
        Whether planes are interpolated from dense output rather than landed on.
    step_planes : int
        Observe the endpoint of every ``step_planes``-th accepted step; ``0`` for none.
    stats : solprop.stats.StatsSpec
        The plane statistics, and the periodic subset with its period.
    retention : solprop.output.schema.FieldRetention
        Which observed planes keep their field.
    """

    save_planes: Float[np.ndarray, " n_z"]
    start_plane: int
    dense: bool
    step_planes: int
    stats: StatsSpec
    retention: FieldRetention


@dataclass(frozen=True, eq=False)
class BoundaryRequest:
    """What the boundary watches.

    Attributes
    ----------
    monitor : solprop.monitor.BoundaryMonitor or None
        A fresh, unrecorded monitor -- weights, collar, thresholds and any transverse
        edge -- that the session takes over and feeds; ``None`` watches nothing.
    """

    monitor: BoundaryMonitor | None = None


class HistoryRetention(enum.Enum):
    """When the method's history is kept for a resume.

    Attributes
    ----------
    AUTO : str
        Kept while it is at most :data:`MARCH_SNAPSHOT_AUTO_BYTES`; omitted beyond
        that, which a resume reports.
    REQUIRED : str
        Always kept; a snapshot that cannot keep it fails rather than downgrading.
    OFF : str
        Never kept, by explicit choice.
    """

    AUTO = "auto"
    REQUIRED = "required"
    OFF = "off"


@dataclass(frozen=True)
class HistoryPolicy:
    """The method-history retention policy.

    Attributes
    ----------
    retention : HistoryRetention
        When to keep it.
    """

    retention: HistoryRetention = HistoryRetention.AUTO

    @classmethod
    def from_choice(cls, choice: bool | None) -> HistoryPolicy:
        """Return the policy for the public ``resume_history`` choice.

        Parameters
        ----------
        choice : bool or None
            ``True`` keeps it always, ``False`` never and ``None`` automatically.

        Returns
        -------
        HistoryPolicy
            The policy.
        """
        if choice is None:
            return cls(HistoryRetention.AUTO)
        return cls(HistoryRetention.REQUIRED if choice else HistoryRetention.OFF)


@dataclass(frozen=True, eq=False)
class RunRequest:
    """Every policy of one run, resolved before preparation.

    Attributes
    ----------
    integration : IntegrationSpec
        The method.
    error_control : ErrorControlSpec
        Tolerances, controller and step limits.
    replay : ReplaySpec or None
        A fixed itinerary; ``None`` for an adaptive run.
    observation : ObservationRequest
        Planes, statistics and field retention.
    boundary : BoundaryRequest
        The boundary monitor's configuration.
    history : HistoryPolicy
        Method-history retention for a resume.
    snapshots : bool
        Whether snapshot requests may arrive during the run.
    """

    integration: IntegrationSpec
    error_control: ErrorControlSpec
    replay: ReplaySpec | None
    observation: ObservationRequest
    boundary: BoundaryRequest
    history: HistoryPolicy
    snapshots: bool = False

    @functools.cached_property
    def stepper(self) -> StepperSettings:
        """Return the stepping policy as the one record translation resolved.

        The method, the error control, the itinerary and the planes, gathered back
        into :class:`~solprop.translate.problem.StepperSettings` for the code that
        reads them together: the march drivers, the request-aware checks
        (:func:`solprop.translate.problem.validate_run`) and the run's configuration
        record, whose hash they enter exactly as before.

        Returns
        -------
        solprop.translate.problem.StepperSettings
            The record, built once per request.

        Examples
        --------
        >>> from solprop.stepper import DOPRI5, PIController
        >>> from solprop.translate.problem import StepperSettings
        >>> from solprop.translate.run_request import describe_run
        >>> settings = StepperSettings(DOPRI5, PIController(rtol=1e-8, atol=1e-14),
        ...                            1e-9, 0.1, 1e-3, 1.1, None, [0.5, 1.0], "auto")
        >>> describe_run(settings).stepper.save_planes.tolist()
        [0.5, 1.0]
        """
        integration, control = self.integration, self.error_control
        observation = self.observation
        return StepperSettings(
            tableau=integration.tableau,
            controller=control.controller,
            dz_min=control.dz_min,
            dz_max=control.dz_max,
            dz_init=control.dz_init,
            stretch=control.stretch,
            schedule=None if self.replay is None else self.replay.schedule,
            save_planes=observation.save_planes,
            propagator_mode=integration.propagator_mode,
            integrator=integration.method,
            interval_cap=control.interval_cap,
            auto_init=control.auto_init,
            interpolate_planes=observation.dense,
            step_planes=observation.step_planes,
        )

    def with_stepper(self, stepper: StepperSettings) -> RunRequest:
        """Return this request with another stepper section, its other policies kept.

        The statistics, retention, start, monitor, history and snapshot policy stay;
        the method, the error control, the itinerary and the planes are `stepper`'s.
        :meth:`solprop.driver.simulation.Simulation.with_input` uses it for the planes
        left beyond an interior launch.

        Parameters
        ----------
        stepper : solprop.translate.problem.StepperSettings
            The new stepper section.

        Returns
        -------
        RunRequest
            The request.

        Raises
        ------
        TranslationError
            If this request's start plane is not a plane of the new section.

        Examples
        --------
        >>> import dataclasses
        >>> from solprop.stepper import DOPRI5, PIController
        >>> from solprop.translate.problem import StepperSettings
        >>> from solprop.translate.run_request import describe_run
        >>> settings = StepperSettings(DOPRI5, PIController(rtol=1e-8, atol=1e-14),
        ...                            1e-9, 0.1, 1e-3, 1.1, None, [0.5, 1.0], "auto")
        >>> request = describe_run(settings)
        >>> shorter = dataclasses.replace(settings, save_planes=[1.0])
        >>> later = request.with_stepper(shorter)
        >>> later.observation.save_planes.tolist()
        [1.0]
        """
        observation = self.observation
        return describe_run(
            stepper,
            stats=observation.stats,
            retention=observation.retention,
            start_plane=observation.start_plane,
            monitor=self.boundary.monitor,
            history=self.history,
            snapshots=self.snapshots,
        )

    def for_run(
        self,
        *,
        stats: StatsSpec | None = None,
        retention: FieldRetention | None = None,
        start_plane: int = 0,
        monitor: BoundaryMonitor | None = None,
        history: HistoryPolicy | None = None,
        snapshots: bool = False,
    ) -> RunRequest:
        """Return this request with one run's observation, boundary and history.

        The method, the error control, the itinerary and the planes are kept: a
        backend prepared for this request serves the returned one.

        Parameters
        ----------
        stats : solprop.stats.StatsSpec or None, optional
            The statistics; this request's own by default.
        retention : solprop.output.schema.FieldRetention or None, optional
            Field retention, already resolved against the planes; every field by
            default.
        start_plane : int, optional
            The first save plane beyond the starting state.
        monitor : solprop.monitor.BoundaryMonitor or None, optional
            A fresh boundary monitor, or ``None``.
        history : HistoryPolicy or None, optional
            Method-history retention; automatic by default.
        snapshots : bool, optional
            Whether snapshot requests may arrive.

        Returns
        -------
        RunRequest
            The run's request.

        Raises
        ------
        TranslationError
            If `start_plane` is not a plane of the run.

        Examples
        --------
        >>> from solprop.stepper import DOPRI5, PIController
        >>> from solprop.translate.problem import StepperSettings
        >>> from solprop.translate.run_request import describe_run
        >>> settings = StepperSettings(DOPRI5, PIController(rtol=1e-8, atol=1e-14),
        ...                            1e-9, 0.1, 1e-3, 1.1, None, [0.5, 1.0], "auto")
        >>> describe_run(settings).for_run(start_plane=1).observation.start_plane
        1
        """
        observation = self.observation
        planes = observation.save_planes.size
        if not 0 <= start_plane <= planes:
            msg = f"start_plane {start_plane} is not a plane of a run with {planes}"
            raise TranslationError(msg)
        return dataclasses.replace(
            self,
            observation=dataclasses.replace(
                observation,
                start_plane=start_plane,
                stats=observation.stats if stats is None else stats,
                retention=FieldRetention() if retention is None else retention,
            ),
            boundary=BoundaryRequest(monitor),
            history=HistoryPolicy() if history is None else history,
            snapshots=snapshots,
        )


def describe_run(
    stepper: StepperSettings,
    *,
    stats: StatsSpec | None = None,
    retention: FieldRetention | None = None,
    start_plane: int = 0,
    monitor: BoundaryMonitor | None = None,
    history: HistoryPolicy | None = None,
    snapshots: bool = False,
) -> RunRequest:
    """Build the run request from a resolved stepper section and the run's settings.

    Parameters
    ----------
    stepper : solprop.translate.problem.StepperSettings
        The stepper section translation resolved; it supplies the method, error
        control, itinerary and planes.
    stats : solprop.stats.StatsSpec or None, optional
        The statistics; none by default.
    retention : solprop.output.schema.FieldRetention or None, optional
        Field retention, already resolved against the planes; every field by default.
    start_plane : int, optional
        The first save plane beyond the starting state.
    monitor : solprop.monitor.BoundaryMonitor or None, optional
        A fresh boundary monitor to copy, or ``None``.
    history : HistoryPolicy or None, optional
        Method-history retention; automatic by default.
    snapshots : bool, optional
        Whether snapshot requests may arrive.

    Returns
    -------
    RunRequest
        The request.

    Raises
    ------
    TranslationError
        If `start_plane` is not a plane of the run.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> from solprop.translate.run_request import describe_run
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=1e-13, peak_power=1.0).build(grid)
    >>> problem, request = assemble(
    ...     grid, gnlse.mode, [gnlse.kerr], start,
    ...     controller=PIController(rtol=1e-8, atol=1e-14), save_planes=[1.0],
    ...     dz_max=0.1, dz_init=1e-3, shock=False, prefactor_index=1.0, density=1.0,
    ... )
    >>> again = describe_run(request.stepper)
    >>> again.integration.kind.value, again.observation.save_planes.tolist()
    ('runge_kutta', [1.0])
    """
    settings = stepper
    planes = np.array(settings.save_planes, dtype=np.float64)
    planes.setflags(write=False)
    if not 0 <= start_plane <= planes.size:
        msg = f"start_plane {start_plane} is not a plane of a run with {planes.size}"
        raise TranslationError(msg)
    integrator = settings.integrator
    if isinstance(integrator, LowStorageRungeKutta):
        kind = MethodKind.LOW_STORAGE
    elif isinstance(integrator, Multistep | ComparisonMultistep):
        kind = MethodKind.ADAMS
    else:
        kind = MethodKind.RUNGE_KUTTA
    return RunRequest(
        integration=IntegrationSpec(
            kind=kind,
            tableau=settings.tableau,
            method=integrator,
            propagator_mode=settings.propagator_mode,
        ),
        error_control=ErrorControlSpec(
            controller=settings.controller,
            dz_min=settings.dz_min,
            dz_max=settings.dz_max,
            dz_init=settings.dz_init,
            stretch=settings.stretch,
            auto_init=settings.auto_init,
            interval_cap=settings.interval_cap,
        ),
        replay=None if settings.schedule is None else ReplaySpec(settings.schedule),
        observation=ObservationRequest(
            save_planes=planes,
            start_plane=start_plane,
            dense=settings.interpolate_planes,
            step_planes=settings.step_planes,
            stats=StatsSpec() if stats is None else stats,
            retention=FieldRetention() if retention is None else retention,
        ),
        boundary=BoundaryRequest(monitor),
        history=HistoryPolicy() if history is None else history,
        snapshots=snapshots,
    )
