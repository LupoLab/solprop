# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""How a run is stepped, what it records and which backend executes it.

:class:`SimulationSettings` and the checks that admit it, and the settings defaults a
keyword route applies: its save planes, its controller and its absorbing boundaries
(SEP 21). The manual page is ``docs/guide/setup/running.md``.
"""

import dataclasses
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, SupportsIndex, cast

import numpy as np
from jaxtyping import Float

from solprop._display import Displayed, engineering, line, summarise
from solprop._ownership import frozen_copy
from solprop.adiabatic import AdiabaticAccuracy
from solprop.backends.registry import resolve_name
from solprop.backends.session import ExecutionOptions, positive_count
from solprop.boundaries import Absorber, SpatialAbsorber, no_absorber
from solprop.boundaries import absorber as collar
from solprop.driver.defaults import (
    default_controller,
    default_integrator,
    default_rtol,
    default_scheme,
    default_step,
)
from solprop.exceptions import BackendError, DriverError, GridError, SolpropWarning
from solprop.grid import TimeGrid
from solprop.modes.quadrature import TransverseQuadrature
from solprop.monitor import DEFAULT_BOUNDARY_CHECK, BoundaryCheck
from solprop.nonlinear.tdse import TDSEExecution
from solprop.output.records import DefaultsResolution
from solprop.output.store import FieldsChoice, field_retention_refusal
from solprop.profiles import ZProfile
from solprop.stats import StatsSpec
from solprop.stepper.controller import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.masks import AbsorberScheme
from solprop.translate.problem import PropagatorMode

__all__ = [
    "DEFAULT_N_SAVE",
    "LOOSE_ADAMS_RTOL",
    "LOW_STORAGE_BACKENDS",
    "NoiseSetting",
    "SimulationSettings",
    "default_collar",
    "keyword_settings",
]

LOOSE_ADAMS_RTOL = 5e-2
"""The ``rtol`` from which an adaptive Adams run is warned on.

An error of five per cent or more allowed per step compounds through the method's
history, and its predictor-corrector estimate is unreliable there
(``docs/guide/numerics/choosing-a-stepper.md``).
"""

DEFAULT_N_SAVE = 201
"""Planes a GNLSE run stores, counting the one at ``z = 0``."""

_RECIPE_EXECUTION = ("fields", "field_planes", "resume_history")
"""The settings a recipe's ``[execution]`` table holds: a change of any other one drops
the recipe, which would no longer describe the run (SEP 41, S2)."""


def _held_budget(value: SupportsIndex | None) -> int | None:
    """Return a settings node budget as an ``int``, refused before translation.

    Any integral budget, a NumPy integer among them, is held as an ``int``, and a
    budget that is not one is refused when the settings are made, rather than when the
    backend is built after the output has opened.

    - ``value``: the budget given, bytes.
    - Returns: the budget, or ``None`` for the backend's automatic choice.
    """
    if value is None:
        return None
    try:
        return positive_count("node_budget", value)
    except BackendError as error:
        raise DriverError(str(error), parameter="node_budget") from None


_EXECUTION_SETTINGS = (
    "backend",
    "node_budget",
    "resume_history",
    "tdse_execution",
    "fields",
    "field_planes",
    "boundary_check",
)
"""The settings read when a run executes, not when it is translated."""

NoiseSetting = Literal["default", "input", "modified"] | None
"""What a constructor's ``noise`` keyword takes (SEP 37).

A model, ``None`` for a noiseless run, or ``"default"``, the keyword's default, which
resolves to ``"input"`` wherever the run can carry noise and to none where it cannot,
recording which in the run's :class:`~solprop.output.records.NoiseResolution`.
"""


class _Defaulted(str):
    """A setting's value that a default rule chose rather than the caller.

    A copy made with :func:`dataclasses.replace` carries the marker with the value, so
    the copy resolves the rule again for its own route or planes (the collar scheme),
    while a value the caller names, even an equal one, is a plain string and stays as
    given.
    """

    __slots__ = ()


@dataclass(frozen=True, eq=False)
class SimulationSettings(Displayed):
    """How a run is stepped, what it records and which backend executes it.

    Everything that is *not* a physics object, in one group, so that a simulation can be
    constructed from the physics and this. Equality is by identity (``eq=False``)
    because ``save_planes`` is an array.

    Attributes
    ----------
    controller : solprop.stepper.controller.PIController or None
        Tolerances, safety and gains; the tolerances and the norm choice live here and
        nowhere else (``docs/dev/kernel-contract.md`` section 7). ``None``, the default,
        takes the route's from :data:`~solprop.driver.defaults.ROUTE_DEFAULTS` (SEP 37)
        when an interface builds the run: the
        :data:`~solprop.driver.defaults.SPATIAL_RTOL` and
        :data:`~solprop.driver.defaults.SPATIAL_ATOL` of
        :meth:`~solprop.driver.simulation.Simulation.from_free` on a radial grid and
        of :meth:`~solprop.driver.simulation.Simulation.from_planar`, the
        :data:`~solprop.driver.defaults.FREE_RTOL` and
        :data:`~solprop.driver.defaults.FREE_ATOL` of
        :meth:`~solprop.driver.simulation.Simulation.from_free` on a Cartesian or line
        grid. An explicit controller
        always wins, and settings that already belong to a route keep its default.
        The other fields are keyword-only.
    save_planes : numpy.ndarray
        Positions to save at, m, increasing and positive, shape ``(n_z,)``. The plane at
        ``z = 0`` is saved as well and is not listed here. A sequence is adopted as a
        read-only float64 array, so that two runs specified the same way record the same
        planes whichever type they passed and therefore hash alike.
    dz_max : float
        Cap on the controller's proposal, m.
    dz_init : float
        First step, m.
    dz_min : float or None
        Floor on the proposal, m; ``None`` takes
        :func:`solprop.stepper.controller.default_dz_min` of the run length. A proposal
        below it raises rather than being clamped silently.
    stretch : float or None
        Landing stretch factor; ``None`` takes
        :data:`solprop.translate.problem.DEFAULT_STRETCH`.
    auto_init : bool
        Whether the start-up rule takes the step after the first from the first step's
        own error estimate, rather than growing from `dz_init` at the growth cap (SEP
        14). The convenience constructors set it when no `dz_init` is given. It is off
        for a coupled flow and a reject-free controller.
    interval_cap : bool or None
        Whether a step is capped at the width of its operator-table interval. ``None``
        caps it only for a coupled flow or a reject-free controller (SEP 14).
    interpolate_planes : bool
        Observe the save planes inside accepted steps instead of landing on them (SEP
        14 phase 3): through the Adams march's own polynomial or, with
        ``RungeKutta()``, DOPRI5's continuous extension. Every plane but the run's end
        and a hard kink is then interpolated. Refused beside a coupled flow, the
        per-step temporal collar and the ``modified`` noise.
    step_planes : int
        A positive ``k`` also saves the end of every ``k``-th accepted step, the
        march's own positions, beside the requested planes (SEP 14 phase 3); ``0``
        saves the requested planes only. Not in scans.
    integrator : Multistep, RungeKutta, LowStorageRungeKutta or None
        The method that steps the run (``docs/guide/numerics/choosing-a-stepper.md``).
        ``None``, the default, takes ``Multistep()``, or ``RungeKutta()`` where the
        interface's builder sets it
        (:meth:`~solprop.driver.simulation.Simulation.from_gnlse`,
        :meth:`~solprop.driver.simulation.Simulation.from_waveguide` and
        :meth:`~solprop.driver.simulation.Simulation.from_free`;
        ``docs/guide/numerics/choosing-a-stepper.md``); the stored value is the resolved
        method. ``Multistep()`` is the low-call exponential Adams method, which
        ``numpy_ref`` and ``native`` execute for supported configurations;
        ``RungeKutta()`` is the interaction-picture DOPRI5 march, which every backend
        executes and which holds a third of the Adams history's memory.
        ``LowStorageRungeKutta()`` replays `schedule` with RK4 in three registers and no
        error estimate (SEP 24); every backend executes it
        (:data:`LOW_STORAGE_BACKENDS`), and ``"auto"`` resolves as for any integrator.
    schedule : solprop.stepper.schedule.FixedSchedule or None
        A fixed schedule to replay instead of stepping adaptively
        (``docs/dev/stepper.md`` section 8). Its edges must contain every save plane,
        which :meth:`solprop.stepper.schedule.FixedSchedule.uniform` guarantees.
    propagator_mode : {'auto', 'tabulate', 'recompute'}
        Whether the backend tabulates the stage propagators
        (``docs/dev/kernel-contract.md``).
    density : solprop.profiles.ZProfile or float or None
        Number density along z in m^-3 that multiplies the compiled responses; ``None``
        takes the mode's own path. The GNLSE passes 1.0
        (``docs/guide/waveguides/gnlse.md``).
        Refused beside a mixture and beside a gas medium, which carry their own.
    shock : bool
        Keep the ``omega`` dependence of the normalisation prefactor, which is the
        optical shock term.
    prefactor_index : float or None
        Use this constant in the prefactor and power weight; None uses Re n_eff.
        The GNLSE passes 1.0, leaving its power normalisation and Phi unchanged.
    absorber_scheme : {'nonlinear', 'split_step'} or None
        How the temporal collar is applied (``docs/guide/numerics/boundaries.md``).
        ``None``, the default, is the route's: ``"split_step"`` on the radial,
        free-space and planar routes unless planes are interpolated (SEP 21), and
        ``"nonlinear"`` elsewhere. ``"nonlinear"`` adds ``-alpha(t) E(t) / 2`` inside
        every stage; ``"split_step"`` multiplies the accepted state by
        ``exp(-alpha(t) h / 2)`` once per step, a Lie splitting, first order wherever
        the collar does any work and unconditionally stable.
    absorber : solprop.boundaries.Absorber or None
        The collar rates. Its spectral part folds into the linear operator and its
        temporal part crosses the seam as a rate
        (``docs/guide/numerics/boundaries.md``).
    quadrature : solprop.modes.quadrature.TransverseQuadrature or None
        The fixed transverse rule a multimode run integrates the nonlinear polarisation
        on (``docs/guide/waveguides/mode-sets.md``). ``None`` builds the mode-averaged
        single-node transform, which needs exactly one mode.
    components : sequence of str or None
        Which transverse Cartesian components the multimode transform resolves at each
        node, a non-empty subset of ``("x", "y")``; ``None`` takes the union of
        the modes' nonzero components at the entrance (``("x",)`` for HE_1m modes
        at ``phi = 0``; ``docs/dev/conventions.md`` section 5). Refused without a
        `quadrature`.
    ionisation_field_range : tuple of float or None
        Lowest and highest field amplitude in V/m the compiled ionisation-rate table
        has to cover. Required beside a plasma response whose rate is not already a
        table; the table holds its last value above the upper limit, so one set too
        low is a silent clip.
    node_budget : int or None
        Bytes of node-chunk buffers a response pass may hold
        (``docs/dev/architecture.md`` section 13). An execution setting, outside the
        resume key: the backend plans its node chunk from it when it is prepared, and
        it may change after translation. ``None``, the default, leaves the choice to
        the backend, which takes 1 GiB on a free-space, line or planar grid and 64 MiB
        otherwise (:data:`~solprop.backends.node_chunk.DEFAULT_SPATIAL_NODE_BUDGET`,
        :data:`~solprop.backends.node_chunk.DEFAULT_NODE_BUDGET`); an explicit budget
        is authoritative.
    stats : solprop.stats.StatsSpec or None
        Which statistics to collect at every saved plane; ``None`` collects none.
    polarisation_taper : bool
        Whether the spectral collar tapers the nonlinear polarisation as well as
        absorbing the field through the linear operator. Default ``False``, which is
        :func:`solprop.translate.assembly.assemble`'s: the taper costs a systematic
        error of the size of the field's content in the collar, and the rate absorber
        acts either way (``docs/guide/numerics/boundaries.md``).
    backend : str
        Registry name of the backend to execute the run. The default ``"auto"``
        resolves to ``"native"`` at settings construction, and refuses if it isn't
        installed or this processor is below its baseline.
        The stored name is concrete, including in scan workers. A backend's import or
        construction errors never trigger a fallback.
    resume_history : bool or None
        Whether the output keeps the Adams march's state at every save plane and
        terminal checkpoint, so that a resume continues the history -- its samples, its
        order and its proposed step -- as the uninterrupted run would, instead of
        starting it afresh at order one. ``None``, the default, keeps it while it is at
        most :data:`solprop.translate.run_request.MARCH_SNAPSHOT_AUTO_BYTES`; it is
        outside the resume key. Every backend's fused and exponential Adams engines, the
        native backend's resident one included, export and install their history.
    fields : {'all', 'planes', 'none'}
        Which saved planes keep their field (SEP 22): every one, the default; those
        of `field_planes`; or none, for a run wanted as statistics alone. A plane
        without its field stores no field and no noise phase, and an in-memory output
        is allocated for the kept fields only; the march is the same. A
        terminal checkpoint carries the field anyway, so a stopped run resumes, but a
        file whose last regular plane kept no field resumes only from one. Outside
        the resume key, like `resume_history`: it changes what is kept, not what is
        computed. In a scan, :func:`~solprop.driver.scan.run_scan`'s own `fields`
        decide, and a builder that sets this is refused. Beside
        ``LowStorageRungeKutta()``, a step abandoned in flight leaves no checkpoint, so
        a file resumes from the last plane that kept its field.
    field_planes : sequence of int
        Saved-plane indices for ``fields='planes'``, the entrance as 0 and negative
        from the end; empty otherwise. Refused beside `step_planes`, whose plane count
        is not known in advance.
    evanescent_rate : float or None
        The spatial operator's evanescent field attenuation cap in m^-1, positive and
        finite (SEP 22 section 4.4). ``None``, the default, keeps
        :data:`solprop.spatial.EVANESCENT_RATE`;
        :func:`solprop.spatial.physical_evanescent_rate` gives the rate at which the
        grid's non-propagating content decays physically. Refused without a spatial
        grid; a replayed step it would overflow is refused at translation.
    spatial_absorber : solprop.boundaries.SpatialAbsorber, False or None
        The transverse boundary of a free-space run (SEP 40). ``None``, the default,
        is ``SpatialAbsorber(real_space=False)``: a wall or a wrap at the grid's
        outer size and a safety ramp on its outer wavenumbers, folded into the
        operator as an exact diagonal rate. A
        :class:`~solprop.boundaries.SpatialAbsorber` asks for a real-space collar
        too, in the room the grid has outside its requested size
        (``collar_fraction``). ``False`` carries no ramp either; the boundary
        monitor still runs. A specification is refused on modal runs.
    adiabatic_accuracy : solprop.adiabatic.AdiabaticAccuracy or None
        The forward-field and reflected-energy budgets a radial medium varying
        along z must meet (SEP 34 section 4.5). ``None``, the default, is
        ``AdiabaticAccuracy()``, both 1e-4; a path exceeding them is refused.
    tdse_execution : solprop.nonlinear.tdse.TDSEExecution or None
        Explicit atomic runtime resources for a TDSE response. Requires an explicit
        ``backend="numpy_ref"`` and a Multistep integrator. Execution resources are
        outside the scientific resume identity; attempted-call budgets are cumulative.
    boundary_check : solprop.monitor.BoundaryCheck or None
        How the run watches its time window's edges (SEP 21): guard and collar
        occupancy at the launch and every committed step, a warning and then a stop
        when light reaches the edge, and the energy the temporal collar removed at the
        end. The default :class:`~solprop.monitor.BoundaryCheck` warns at a field-error
        budget of 1e-3 and stops at ten times it; ``None`` watches nothing.

    Raises
    ------
    DriverError
        If the save planes are empty, not increasing, or not positive. The same check
        runs again in :func:`solprop.translate.assembly.assemble`; here it fires before
        a run has built anything. Also if the field retention is not valid, asks for
        ``fields='planes'`` beside `step_planes`.

    Examples
    --------
    >>> from solprop.driver import SimulationSettings
    >>> from solprop.stepper import PIController
    >>> settings = SimulationSettings(
    ...     controller=PIController(rtol=1e-8, atol=1e-14),
    ...     save_planes=[0.5, 1.0], dz_max=0.1, dz_init=1e-3,
    ...     backend="numpy_ref",
    ... )
    >>> settings.save_planes.tolist(), settings.backend
    ([0.5, 1.0], 'numpy_ref')
    """

    controller: PIController | None = None
    _: dataclasses.KW_ONLY
    save_planes: Float[np.ndarray, " n_z"] | Sequence[float]
    dz_max: float
    dz_init: float
    dz_min: float | None = None
    stretch: float | None = None
    auto_init: bool = False
    interval_cap: bool | None = None
    interpolate_planes: bool = False
    step_planes: int = 0
    schedule: FixedSchedule | None = None
    propagator_mode: PropagatorMode = "auto"
    density: ZProfile | float | None = None
    shock: bool = True
    prefactor_index: float | None = None
    absorber: Absorber | None = None
    absorber_scheme: AbsorberScheme | None = None
    quadrature: TransverseQuadrature | None = None
    components: Sequence[str] | None = None
    ionisation_field_range: tuple[float, float] | None = None
    node_budget: int | SupportsIndex | None = None
    stats: StatsSpec | None = None
    polarisation_taper: bool = False
    backend: str = "auto"
    integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None
    resume_history: bool | None = None
    tdse_execution: TDSEExecution | None = None
    fields: FieldsChoice = "all"
    field_planes: Sequence[int] = ()
    evanescent_rate: float | None = None
    spatial_absorber: SpatialAbsorber | Literal[False] | None = None
    adiabatic_accuracy: AdiabaticAccuracy | None = None
    boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK
    _default_integrator: Multistep | RungeKutta | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    """The default the settings chose for an unnamed integrator. A copy made with
    :func:`dataclasses.replace` carries it with the integrator, so the copy still knows
    the caller named none; any other integrator counts as named."""
    _route: str | None = dataclasses.field(default=None, repr=False, compare=False)
    """The row of :data:`~solprop.driver.defaults.ROUTE_DEFAULTS` the interface that
    built the settings reads, or ``None`` for settings built by hand. A copy made with
    :func:`dataclasses.replace` carries it, so a copy takes the route's defaults
    again, and the route's choice never counts as named."""
    _default_controller: PIController | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    """The controller the settings took from the route's defaults. A copy carries it
    with the controller, so the copy still knows the caller gave none; any other
    controller counts as given."""
    # Whether the caller named the integrator, set per instance in __post_init__.
    integrator_named = False
    # Whether the controller is the route's default, set per instance in __post_init__.
    controller_defaulted = False

    def _planes_text(self) -> str:
        """Return the save planes summarised, for printing."""
        planes = np.asarray(self.save_planes, dtype=float)
        return summarise(planes, "m").replace("values", "planes")

    def __repr__(self) -> str:
        """Return a one-line description with the save planes summarised, not dumped.

        Returns
        -------
        str
            For example ``SimulationSettings(<201 planes, 0 m to 1.5 m>, rtol=1e-06,
            backend=auto)``.
        """
        return line(
            "SimulationSettings",
            [
                ("", self._planes_text()),
                *(
                    [("", "interface default tolerances")]
                    if self.controller is None
                    else [
                        ("rtol", f"{self.controller.rtol:g}"),
                        ("atol", f"{self.controller.atol:g}"),
                    ]
                ),
                ("backend", self.backend),
            ],
        )

    def _default_text(self, text: str) -> str:
        """Mark the integrator's name when it is the route's default.

        - Returns: `text`, then ``"(default)"`` or ``"(gnlse default)"`` where no
          integrator was named.
        """
        if self.integrator_named:
            return text
        return f"{text} ({'' if self._route is None else self._route + ' '}default)"

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        schedule = "adaptive" if self.schedule is None else type(self.schedule).__name__
        return "SimulationSettings", [
            ("save planes", self._planes_text()),
            ("integrator", self._default_text(type(self.integrator).__name__)),
            (
                "tolerances",
                "the interface's default"
                if self.controller is None
                else f"rtol {self.controller.rtol:g}, atol {self.controller.atol:g}"
                + (
                    f" ({self._route} default)"
                    if self.controller_defaulted and self._route is not None
                    else ""
                ),
            ),
            ("steps", f"{schedule}, dz_max {engineering(self.dz_max, 'm')}"),
            ("shock term", self.shock),
            (
                "absorber",
                (
                    f"{_scheme(self)} scheme"
                    + ("" if self.absorber_scheme_named else " (default)")
                )
                if self.absorber is not None
                else "none",
            ),
            ("statistics", "none" if self.stats is None else type(self.stats).__name__),
            ("backend", self.backend),
        ]

    def __post_init__(self) -> None:
        """Adopt a read-only plane array and check it.

        Raises
        ------
        DriverError
            If the planes are empty, not increasing, or not positive, or if
            `resume_history` is not a boolean or ``None``.
        """
        planes = frozen_copy(np.reshape(self.save_planes, -1), np.float64)
        if (
            planes.size == 0
            or not np.all(np.isfinite(planes))
            or not np.all(np.diff(planes) > 0.0)
            or planes[0] <= 0.0
        ):
            msg = (
                f"save_planes must be positive and increasing, got {planes}; the plane "
                "at z = 0 is saved anyway and is not one of them"
            )
            raise DriverError(msg, parameter="save_planes")
        object.__setattr__(self, "save_planes", planes)
        refusal = field_retention_refusal(self.fields, self.field_planes)
        if refusal is not None:
            raise DriverError(refusal, parameter=("fields", "field_planes"))
        object.__setattr__(self, "field_planes", tuple(self.field_planes))
        object.__setattr__(self, "node_budget", _held_budget(self.node_budget))
        if self.resume_history is not None and not isinstance(
            self.resume_history, bool
        ):
            # A truthy string such as a recipe's "none" would otherwise be read as
            # True by the history policy, the opposite of what it says (issue #703).
            msg = (
                "resume_history must be True, False or None, got "
                f"{self.resume_history!r}"
            )
            raise DriverError(msg, parameter="resume_history")
        self._resolve_defaults()
        if self.fields == "planes" and self.step_planes:
            msg = (
                "fields='planes' names saved planes by index, and step_planes saves "
                "the march's own steps as well, so the plane count is not known in "
                "advance; keep fields='all' or 'none' beside step_planes"
            )
            raise DriverError(
                msg,
                parameter=("fields", "step_planes"),
                rule="fields-planes-step-planes",
            )
        if self.tdse_execution is not None:
            _admit_tdse_execution(self)
        # The annotation refuses another integrator first wherever the annotations are
        # checked at run time, as in the test suite.
        if not isinstance(
            self.integrator, Multistep | RungeKutta | LowStorageRungeKutta
        ):  # pragma: no cover
            msg = (
                "integrator must be Multistep(), RungeKutta() or "
                f"LowStorageRungeKutta(), got {self.integrator!r}"
            )
            raise DriverError(msg, parameter="integrator")
        if isinstance(self.integrator, LowStorageRungeKutta):
            backend = _admit_low_storage(self.integrator, self.backend, self)
            object.__setattr__(self, "backend", backend)
        # An Adams integrator resolves "auto" as the Runge-Kutta march does: every
        # backend steps the exponential formulation on every route, and the
        # interaction one on every route but a coupled flow, which it refuses by name
        # (SEP 16 phases 3 and 5).
        object.__setattr__(self, "backend", resolve_name(self.backend))
        if isinstance(self.integrator, Multistep):
            rtol = getattr(self.controller, "rtol", None)
            if self.schedule is None and rtol is not None and rtol >= LOOSE_ADAMS_RTOL:
                msg = (
                    f"rtol={rtol:g} is loose for the Adams method: an error of this "
                    "size allowed per step compounds through its history, and its "
                    "error estimate is unreliable there. Take rtol at most 1e-2, or "
                    "integrator=RungeKutta() for a rough look"
                )
                warnings.warn(msg, SolpropWarning, stacklevel=3)

    def _resolve_defaults(self) -> None:
        """Take the route's default for each of the stepping choices the caller left.

        The integrator, the controller and the collar scheme each come from
        :data:`~solprop.driver.defaults.ROUTE_DEFAULTS` when not given. A default
        carried by a copy is resolved again, so it follows the copy's route and
        planes; a value the caller gave is kept.
        """
        carried = (
            self.integrator is not None and self.integrator is self._default_integrator
        )
        object.__setattr__(
            self, "integrator_named", self.integrator is not None and not carried
        )
        if not self.integrator_named:
            default = default_integrator(self._route)
            if carried and type(self.integrator) is type(default):
                default = self.integrator
            object.__setattr__(self, "integrator", default)
            object.__setattr__(self, "_default_integrator", default)
        controller = self.controller
        defaulted = controller is not None and controller is self._default_controller
        if controller is None and self._route is not None:
            controller = default_controller(self._route)
            defaulted = True
        object.__setattr__(self, "controller", controller)
        object.__setattr__(self, "controller_defaulted", defaulted)
        if defaulted:
            object.__setattr__(self, "_default_controller", controller)
        if self.absorber_scheme is None or isinstance(self.absorber_scheme, _Defaulted):
            scheme = default_scheme(
                self._route, interpolate_planes=self.interpolate_planes
            )
            object.__setattr__(self, "absorber_scheme", _Defaulted(scheme))

    @property
    def absorber_scheme_named(self) -> bool:  # numpydoc ignore=RT01
        """Whether the caller named the collar scheme, rather than taking the route's.

        False when :attr:`absorber_scheme` is the route's default.
        """
        return not isinstance(self.absorber_scheme, _Defaulted)

    def execution_options(self) -> ExecutionOptions:
        """Return the execution options a backend is prepared with (SEP 47).

        The node budget is an execution choice, not part of the scientific problem:
        the translation records none, and the backend plans its node chunk from it
        when it is prepared (:func:`solprop.backends.node_chunk.plan_node_chunk`).

        Returns
        -------
        solprop.backends.session.ExecutionOptions
            The explicit :attr:`node_budget`, or the backend's automatic choice.

        Examples
        --------
        >>> from solprop.driver import SimulationSettings
        >>> settings = SimulationSettings(
        ...     save_planes=[1.0], dz_max=0.1, dz_init=0.01, node_budget=2**26
        ... )
        >>> settings.execution_options().node_budget
        67108864
        """
        return ExecutionOptions(node_budget=self.node_budget)

    def defaults_resolution(self) -> DefaultsResolution:
        """Return which stepping choices took the route's default (#87, SEP 41).

        Returns
        -------
        solprop.output.records.DefaultsResolution
            The route, and which of the integrator, the relative tolerance and the
            collar scheme came from its row of the defaults table.

        Examples
        --------
        >>> from solprop.driver import SimulationSettings
        >>> settings = SimulationSettings(
        ...     save_planes=[1.0], dz_max=0.1, dz_init=0.01, backend="numpy_ref"
        ... )
        >>> settings.defaults_resolution()
        DefaultsResolution(route=None, defaulted=('integrator', 'absorber_scheme'))
        """
        defaulted = (
            *(() if self.integrator_named else ("integrator",)),
            *(("rtol",) if self.controller_defaulted else ()),
            *(() if self.absorber_scheme_named else ("absorber_scheme",)),
        )
        return DefaultsResolution(self._route, defaulted)

    def with_boundaries(
        self,
        grid: TimeGrid,
        *,
        absorber: Absorber | Literal[False] | None = None,
        absorber_scheme: AbsorberScheme | None = None,
        boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
        spatial_absorber: SpatialAbsorber | Literal[False] | None = None,
    ) -> "SimulationSettings":
        """Return these settings with a spatial constructor's boundary keywords applied.

        Each keyword may be given here or in the settings, not both (SEP 21 section 5,
        SEP 40); its default leaves the settings' own.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            The run's time grid, which sizes a collar switched off.
        absorber : solprop.boundaries.Absorber, False or None, optional
            The temporal and spectral collar. ``False`` zeroes the rates of the
            settings' absorber and keeps its lengths, or without one builds the hard
            band mask alone.
        absorber_scheme : {'nonlinear', 'split_step'} or None, optional
            How the temporal collar is applied.
        boundary_check : solprop.monitor.BoundaryCheck or None, optional
            How the run watches its time window's edges; ``None`` watches nothing.
        spatial_absorber : solprop.boundaries.SpatialAbsorber, False or None, optional
            The transverse boundary.

        Returns
        -------
        SimulationSettings
            The settings, replaced where a keyword applies.

        Raises
        ------
        DriverError
            If a keyword is given beside settings that declare their own, or
            ``absorber=False`` beside a scheme.

        Examples
        --------
        >>> from solprop.driver import SimulationSettings
        >>> from solprop.grid import EnvelopeGrid
        >>> grid = EnvelopeGrid.from_limits(0.01, 835e-9, (500e-9, 2e-6), 1e-12)
        >>> settings = SimulationSettings(save_planes=[1.0], dz_max=0.1, dz_init=0.01)
        >>> settings.with_boundaries(grid, boundary_check=None).boundary_check is None
        True
        """
        changes: dict[str, object] = {}
        if absorber is False:
            if absorber_scheme is not None:
                msg = (
                    "absorber=False switches the temporal collar off, so there is "
                    f"nothing for absorber_scheme={absorber_scheme!r} to apply; give "
                    "one or the other"
                )
                raise DriverError(msg)
            current = self.absorber
            changes["absorber"] = (
                no_absorber(grid)
                if current is None
                else dataclasses.replace(
                    current,
                    spectral_rate=np.zeros_like(current.spectral_rate),
                    user_rate=None,
                    temporal_rate=np.zeros(grid.n_t),
                    temporal_rate_fine=np.zeros(grid.n_to),
                    temporal=False,
                )
            )
        elif absorber is not None:
            if self.absorber is not None:
                msg = (
                    "the settings already declare an absorber; give it there or as "
                    "the absorber keyword, not both"
                )
                raise DriverError(msg)
            changes["absorber"] = absorber
        if absorber_scheme is not None:
            if self.absorber_scheme_named and self.absorber_scheme != absorber_scheme:
                msg = (
                    f"the settings already declare absorber_scheme="
                    f"{self.absorber_scheme!r}; give it there or as the keyword"
                )
                raise DriverError(msg)
            changes["absorber_scheme"] = absorber_scheme
        if not (
            boundary_check is DEFAULT_BOUNDARY_CHECK
            or boundary_check == self.boundary_check
        ):
            if self.boundary_check != DEFAULT_BOUNDARY_CHECK:
                msg = (
                    "the settings already declare a boundary_check; give it there or "
                    "as the boundary_check keyword, not both"
                )
                raise DriverError(msg)
            changes["boundary_check"] = boundary_check
        if spatial_absorber is not None and spatial_absorber != self.spatial_absorber:
            if self.spatial_absorber is not None:
                msg = (
                    "the settings already declare a spatial_absorber; give it there or "
                    "as the spatial_absorber keyword, not both"
                )
                raise DriverError(msg)
            changes["spatial_absorber"] = spatial_absorber
        return dataclasses.replace(self, **changes) if changes else self


def _admit_tdse_execution(settings: SimulationSettings) -> None:
    """Refuse TDSE execution resources beside settings a TDSE run cannot step with.

    Each refusal names the setting at fault, in the order a caller would fix them:
    the execution record, then the integrator.
    """
    if not isinstance(settings.tdse_execution, TDSEExecution):
        msg = (
            "tdse_execution must be a TDSEExecution, got "
            f"{type(settings.tdse_execution).__name__}"
        )
        raise DriverError(msg, parameter="tdse_execution")
    if not isinstance(settings.integrator, Multistep):
        msg = (
            "tdse_execution requires integrator=Multistep(), the Adams method the TDSE "
            f"response steps with; got {type(settings.integrator).__name__}()"
        )
        raise DriverError(
            msg, parameter=("tdse_execution", "integrator"), rule="tdse-multistep"
        )


def _quiet_collar(settings: SimulationSettings) -> bool:
    """Return whether a run's settings opt in to the quiet collar (ADR-057).

    - Returns: True when the boundary check asks for it.
    """
    check = settings.boundary_check
    return check is not None and check.quiet_collar


def keyword_settings(
    route: str,
    length: float,
    *,
    rtol: float | None,
    atol: float,
    dz_max: float | None,
    dz_init: float | None,
    # The other settings fields, each typed on SimulationSettings itself.
    **fields: Any,
) -> SimulationSettings:
    """Return the settings a keyword route builds: its controller and its steps.

    Parameters
    ----------
    route : str
        The row of :data:`~solprop.driver.defaults.ROUTE_DEFAULTS` the route reads.
    length : float
        The guide's length, m, which the default steps are fractions of.
    rtol : float or None
        The caller's relative tolerance, which is never raised silently; ``None``
        takes the route's (:func:`~solprop.driver.defaults.default_rtol`), and the
        settings record that the controller is the default.
    atol : float
        The absolute tolerance, in the route's field unit.
    dz_max, dz_init : float or None
        The largest and the first step, m; ``None`` takes
        :func:`~solprop.driver.defaults.default_step`, and a first step so chosen is
        refined by the start-up rule.
    **fields : Any
        Every other field of :class:`SimulationSettings`.

    Returns
    -------
    SimulationSettings
        The settings.

    Examples
    --------
    >>> from solprop.driver.settings import keyword_settings
    >>> settings = keyword_settings(
    ...     "gnlse", 0.4, rtol=None, atol=1e-14, dz_max=None, dz_init=None,
    ...     save_planes=[0.4],
    ... )
    >>> settings.controller.rtol, settings.dz_max, settings.controller_defaulted
    (1e-07, 0.2, True)
    """
    if rtol is None:
        controller = PIController(rtol=default_rtol(route), atol=atol)
        default = controller
    else:
        controller, default = PIController(rtol=rtol, atol=atol), None
    return SimulationSettings(
        controller=controller,
        _default_controller=default,
        _route=route,
        dz_max=default_step("dz_max", length) if dz_max is None else dz_max,
        dz_init=default_step("dz_init", length) if dz_init is None else dz_init,
        auto_init=dz_init is None,
        **fields,
    )


LOW_STORAGE_BACKENDS = ("native", "numpy_ref")
"""The backends that execute :class:`~solprop.stepper.low_storage.LowStorageRungeKutta`.

``numpy_ref`` from SEP 24's phase 1 and solprop-native from phase 3, on every route the
translation admits the march on; ``backend="auto"`` resolves as for every other
integrator (:func:`solprop.backends.registry.resolve_name`: the native backend).
"""


def _admit_low_storage(
    integrator: LowStorageRungeKutta, backend: str, settings: SimulationSettings
) -> str:
    """Refuse what the low-storage march cannot run, before translation (SEP 24 5).

    The march replays a prescribed schedule, lands on its edges and accepts every
    step, so it needs a schedule and has no interpolated planes. Every registered
    backend executes it (:data:`LOW_STORAGE_BACKENDS`), so the backend passes as asked.
    """
    if settings.schedule is None:
        msg = (
            f"the {integrator.name} march replays a prescribed schedule; pass "
            "schedule=FixedSchedule.uniform(...), or step adaptively with RungeKutta()"
        )
        raise DriverError(
            msg, parameter=("integrator", "schedule"), rule="low-storage-needs-schedule"
        )
    if settings.interpolate_planes:
        msg = (
            f"the {integrator.name} march lands on its schedule's edges and has no "
            "dense output; drop interpolate_planes"
        )
        raise DriverError(
            msg,
            parameter=("integrator", "interpolate_planes"),
            rule="low-storage-no-interpolation",
        )
    return backend


def _same_setting(old: object, new: object) -> bool:
    """Return whether two values of one settings field are the same setting.

    - Returns: True for the same object, equal arrays, or values that compare equal; an
      object compared by identity (``eq=False``) is the same only as itself, which is
      what :func:`dataclasses.replace` keeps for every field it is not given.
    """
    if old is new:
        return True
    if isinstance(old, np.ndarray) or isinstance(new, np.ndarray):
        # The other side is an array-like setting (save_planes given as a list).
        left = np.asarray(cast("np.ndarray", old))
        right = np.asarray(cast("np.ndarray", new))
        return left.shape == right.shape and bool(np.array_equal(left, right))
    try:
        return bool(old == new)
    except ValueError:
        # A container of arrays has no single truth value; it is not provably the same.
        return False


def _save_planes(
    length: float,
    save_planes: Sequence[float] | np.ndarray | None,
    n_save: int | None,
) -> Float[np.ndarray, " n_z"]:
    """Return the save planes a keyword interface asks for.

    - ``length``: fibre length, m.
    - ``save_planes``: positions given by the caller, m.
    - Returns: the planes after ``z = 0``, m, shape ``(n_z,)``.
    """
    if save_planes is not None:
        if n_save is not None:
            msg = "give save_planes or n_save, not both"
            raise DriverError(
                msg, parameter=("save_planes", "n_save"), rule="save-planes-or-n-save"
            )
        return np.ascontiguousarray(save_planes, dtype=np.float64).reshape(-1)
    count = DEFAULT_N_SAVE if n_save is None else n_save
    if count < 2:
        msg = (
            f"n_save counts the plane at z = 0, so it must be at least 2, got {count}; "
            "a run always stores the entrance"
        )
        raise DriverError(msg, parameter="n_save")
    return np.linspace(0.0, length, count)[1:]


def _scheme(settings: SimulationSettings) -> AbsorberScheme:
    """Return the collar scheme the settings resolve to."""
    return cast("AbsorberScheme", str(settings.absorber_scheme))


def _spatial_settings(settings: SimulationSettings, route: str) -> SimulationSettings:
    """Give a spatial run's settings the route's defaults (SEP 37, SEP 21).

    - Returns: `settings`, reading the route's row of
      :data:`~solprop.driver.defaults.ROUTE_DEFAULTS` for whatever the caller left: the
      controller (:data:`~solprop.driver.defaults.SPATIAL_RTOL` for radial and planar
      runs, :data:`~solprop.driver.defaults.FREE_RTOL` for Cartesian and line ones), the
      integrator (``RungeKutta()`` in free space, whose Adams
      history would be too large) and the collar scheme (``"split_step"`` unless planes
      are interpolated). Settings that already belong to a route, this one or another
      (settings copied from a modal run, for a comparison), keep what that route
      resolved and are returned as they are.
    """
    if settings._route is not None:
        return settings
    return dataclasses.replace(settings, _route=route)


def _boundary(
    absorber: Absorber | Literal[False] | None,
    scheme: AbsorberScheme | None,
    grid: TimeGrid,
    *,
    default: Callable[[TimeGrid], Absorber] | None = None,
) -> tuple[Absorber | None, AbsorberScheme | None]:
    """Resolve a builder's absorber keywords (SEP 21 section 5).

    - ``absorber``: the collar given; ``False`` switches both soft collars off and keeps
      the hard band mask; ``None`` takes `default`.
    - ``scheme``: the temporal scheme given; ``None`` is the default scheme.
    - ``default``: the route's collar when none is given; ``None`` is
      :func:`default_collar`.
    - Returns: the absorber (``None`` for none) and the scheme, ``None`` where the
      settings are to take the route's default.
    """
    if absorber is False:
        if scheme is not None:
            msg = (
                "absorber=False switches the temporal collar off, so there is nothing "
                f"for absorber_scheme={scheme!r} to apply; give one or the other"
            )
            raise DriverError(
                msg,
                parameter=("absorber", "absorber_scheme"),
                rule="absorber-scheme-needs-absorber",
            )
        return no_absorber(grid), None
    if absorber is None:
        absorber = default_collar(grid) if default is None else default(grid)
    return absorber, scheme


def default_collar(grid: TimeGrid) -> Absorber:
    """Return a route's default absorbing boundaries on a grid (SEP 21).

    :func:`solprop.boundaries.absorber` at its defaults, spectral-only when the grid
    leaves too little room outside the request for a temporal collar to taper
    (``collar_fraction=0`` and no rounding slack).

    Parameters
    ----------
    grid : TimeGrid
        The run's grid.

    Returns
    -------
    Absorber
        The default collar.

    Examples
    --------
    >>> from solprop.driver.settings import default_collar
    >>> from solprop.grid import FieldGrid
    >>> grid = FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15)
    >>> default_collar(grid).temporal
    True
    """
    try:
        return collar(grid)
    except GridError:
        # No room for a temporal taper outside the request: the spectral collar alone.
        return collar(grid, temporal=False)


def _spatial_default(
    settings: SimulationSettings, grid: TimeGrid
) -> SimulationSettings:
    """Give a spatial run whose settings declare no absorber the default collar.

    The temporal and spectral collars become the routes' default (SEP 21, D6); the
    transverse boundary is the settings' ``spatial_absorber`` (SEP 40).
    """
    if settings.absorber is not None:
        return settings
    return dataclasses.replace(settings, absorber=default_collar(grid))
