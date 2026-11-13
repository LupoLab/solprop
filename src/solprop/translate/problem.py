# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel contract: the one object that crosses into a backend.

:class:`PropagationProblem` is a frozen dataclass of plain NumPy arrays and scalars,
validated, versioned and documented field by field in
``docs/dev/kernel-contract.md``. A backend may assume everything stated there and
nothing else; the translation layer promises everything stated there and nothing else.
Changing
that document and this module together is the only way to extend the contract, and
``tests/translate/test_problem.py`` parses the document's tables and checks them against
this dataclass tree in both directions, so the two cannot drift.

The sections of the contract are grouped into one frozen dataclass each:
:class:`Flags` (section 1), :class:`solprop.translate.masks.Grids` (2),
:class:`solprop.translate.linop.LinearOperator` (3),
:class:`solprop.translate.modal.Transform` (4),
:class:`solprop.translate.responses.CompiledResponses` (5), :class:`Initial` (6),
:class:`StepperSettings` (7) and :class:`solprop.stats.StatsSpec` (8).

**Sign convention.** Every complex array here is in the **FFT convention** of
``docs/dev/conventions.md`` section 2 — raw forward DFTs, propagation
``exp(i(omega t - beta z))``, linear operator ``-i beta``. The user's optics-convention
inputs are converted by the pulse builders and the observations are converted back by
the output layer; backends never conjugate.
"""

import hashlib
import importlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import numpy as np
from jaxtyping import Complex, Float, Num

from solprop._ownership import frozen_copy, is_held
from solprop.boundaries import (
    STABILITY_RADIUS,
    SplitCrossing,
    stable_step,
)
from solprop.constants import EPS0, C
from solprop.exceptions import (
    ExponentBudgetWarning,
    TranslationError,
)
from solprop.fourier import GridKind, n_omega_for
from solprop.grid import Collar
from solprop.spatial_absorber import (
    CROSSING_ADAPTIVE,
    CROSSING_FIXED,
    DAMPING_BOUND,
    SpatialAbsorberDesign,
)
from solprop.stats import StatsSpec
from solprop.stepper.controller import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.stepper.tableau import Tableau, continuous_extension
from solprop.translate.admission import check_general_route, check_leaf_types
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.coupled import flow_edge_slack
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.linop import LinearOperator
from solprop.translate.masks import Grids
from solprop.translate.modal import Transform
from solprop.translate.plasma_source import plasma_source_record
from solprop.translate.radial import radial_rate_bounds
from solprop.translate.responses import CompiledResponses
from solprop.translate.routes import (
    GeneralRoute,
    ModalRoute,
    Route,
    SpatialRoute,
    TDSESource,
    route_name,
    tdse_of,
)
from solprop.translate.spatial import (
    RateBounds,
    SeparableOperator,
    rate_bounds,
)
from solprop.translate.tdse_source import (
    STEADY_CURRENT_CERTIFICATE_LIMIT,
    TDSE_UNAVAILABLE_STATISTICS,
    steady_current_certificate,
)

if TYPE_CHECKING:
    # The request module builds on this one: it imports the stepper section.
    from solprop.translate.run_request import RunRequest

__all__ = [
    "DEFAULT_COMPONENTS",
    "DEFAULT_STRETCH",
    "EXPONENT_BUDGET",
    "LOG_DBL_MAX",
    "MAX_N_COMPONENTS",
    "MAX_N_POL",
    "Flags",
    "Initial",
    "NoiseModel",
    "PropagationProblem",
    "PropagatorMode",
    "StepperSettings",
    "Translation",
    "TranslationReuse",
    "debug_summary",
    "validate_run",
]

_MOVED = {
    "assemble": "assembly",
    "MAX_GENERAL_FLOW_EXPONENT": "assembly",
    "validate_coupled_flow": "validation",
    "DEFAULT_FLOW_CONDITION_LIMIT": "validation",
    "MOVING_FLOW_CERTIFICATE": "validation",
    "TAPER_FLOW_CERTIFICATE": "validation",
}
"""Public names this module used to define, by the module that defines them now."""


def __getattr__(name: str) -> object:
    """Return a public name that moved to a module building on this one.

    ``solprop.translate.problem.assemble`` and the rest keep working: they are imported
    from their home on first use, since that module imports this one.

    Parameters
    ----------
    name : str
        The attribute looked up on the module.

    Returns
    -------
    object
        The name's value in its new home.

    Raises
    ------
    AttributeError
        If `name` is not one that moved.
    """
    if name in _MOVED:
        module = importlib.import_module(f"solprop.translate.{_MOVED[name]}")
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


DEFAULT_COMPONENTS = ("x",)
"""Canonical component of an unrotated HE_1m mode.

The component an HE_1m Marcatili mode points along at ``phi = 0`` (see
``docs/dev/decisions.md``). Assembly derives its actual default from the mode set's
``principal_component`` traits, so a rotated or vector set is resolved correctly.
"""


MAX_N_POL = 2
"""Most entries the state's polarisation axis may have.

The transverse components are carried either by the transform's rows or by the
state's polarisation axis, never by both: a row of ``synthesis`` is one Cartesian
component at one node, and so is one entry of the state's polarisation axis, so a
problem that set both would count them twice.

It stays 2 at kernel schema 14. A general modal guide's three Cartesian components
are *not* three state polarisations: one vector eigenmode has E_x, E_y and E_z and
stores one amplitude, so the guide carries ``n_pol = 1`` and
:data:`MAX_N_COMPONENTS` local field components. Setting this to
3 would instead admit three independent amplitudes per eigenmode, which is a
different and unphysical problem.
"""


MAX_N_COMPONENTS = 3
"""Cartesian components of the local physical field a general modal guide resolves.

The count of :class:`solprop.translate.modal_tables.ModalTransformTables`' component
axis -- E_x, E_y and E_z, and the matching H -- which is a property of the provider's
fields and not of the state's polarisation axis. The distinct check
:data:`MAX_N_POL` performs for the legacy routes is kept beside it, unchanged.
"""


_SEAM_DTYPES: tuple[type, ...] = (np.float64, np.complex128, np.bool_, np.intp)

"""The four dtypes every array crossing the seam may have."""


_INT32_LEAVES = frozenset(
    {
        "transform.modal_tables.region_index",
        "responses.regional_responses.response_region",
        "responses.regional_responses.kernel_valid_samples",
        "responses.dispersive_kerr.regions",
    }
)
"""The leaves that may be ``int32`` as well, by dotted name.

Kernel schema 12 admits ``int32`` for the index leaves of the general modal route --
the per-node material region of :class:`~solprop.translate.modal_tables.
ModalTransformTables` and the region and kernel-support indices of
:class:`~solprop.translate.responses.RegionalResponseTables` -- schema 24 for the
region of each dispersive channel on that route
(:class:`~solprop.translate.responses.DispersiveKerrTerms`), and for nothing else.
The set is keyed by name rather than widened globally because
:func:`_walk_arrays` applies the dtype rule to every leaf in the tree, so a global
``int32`` would also have admitted, say, an ``int32`` ``stepper.save_planes``.
"""


_UNBOUNDED_LEAVES = frozenset(
    {"stats_spec.energy_windows", "stats_spec.spectral_regions"}
)
"""The leaves that may hold infinite entries, by dotted name.

A region of transverse wavevector is a half-open box whose edges may be infinite, so
that half-planes and quadrants tile the columns (SEP 23 section 4.7), and an energy
window is a half-open frequency interval whose edges may be infinite, as its manual
documents. Both are read by the statistics alone, only in comparisons, and never enter
a kernel's arithmetic. NaN stays refused.
"""


_UINT8_LEAVES = frozenset({"split_crossing.buckets"})
"""The leaves that may be ``uint8``, by dotted name: the split collar's walk-off
bucket of every bin (kernel schema 43), a byte where a float would be a state's worth
of memory again."""


_STRIDED_LEAVES = frozenset({"initial.field0"})
"""The leaves that need not be C-contiguous, by dotted name.

The initial state is never indexed by a kernel: a backend takes it through
``set_initial``, which accepts any strides and copies it into its own state. So it may
be a zero-stride broadcast view, which the problem keeps instead of a full copy
(:class:`Initial`; kernel schema 42).
"""


DEFAULT_STRETCH = 1.1
"""Landing stretch factor (``stepper.md`` section 3).

It stays 1.1 whatever the growth cap: stretching a step by 2 to reach a plane could
multiply an order-five defect by 32 (SEP 14 section 4.2).
"""


NoiseModel = Literal["none", "input", "modified"]
"""Which shot-noise model the run carries (see ``docs/dev/decisions.md``)."""


PropagatorMode = Literal["auto", "tabulate", "recompute"]
"""Whether the backend tabulates the stage propagators or recomputes them.

``"auto"``, the default, is the backend's own choice, resolved at construction and
recorded in the run's metadata: ``tabulate`` for ``numpy_ref``, whose two complex
exponentials per stage the table is worth memoising, and ``recompute`` for
``native``, where both signs come from one real ``exp`` and a ``sincos`` in the
stage that formed them, so the table buys residency and nothing else
(``docs/dev/backends.md`` section 3.6). Either can still be asked for by name; the two
agree bitwise, so the choice is memory and not numbers.
"""


@dataclass(frozen=True)
class Flags:
    """Static booleans a kernel specialises on at compile time.

    Section 1 of the kernel contract. They are *derived* from the content of the problem
    by :meth:`from_content` and never passed in, so a flag cannot disagree with the
    arrays it describes.

    Attributes
    ----------
    single_mode : bool
        One mode: ``n_modes == 1``.
    scalar_pol : bool
        One polarisation component: ``n_pol == 1``.
    has_plasma : bool
        A plasma response is compiled.
    has_raman : bool
        A Raman kernel is compiled.
    has_noise : bool
        A shot-noise realisation is carried and added at every stage.
    has_temporal_absorber : bool
        A temporal collar rate is present.
    constant_linop : bool
        The linear operator does not vary along z, so the propagator is exact and cheap.
    mode_averaged : bool
        The single-node, effective-area case of the multimode pipeline.
    n_channels : int
        Number of independent labelled plasma populations, non-negative.
    n_raman : int
        Number of Raman-active components, non-negative; each may have several terms.

    Examples
    --------
    >>> from solprop.translate.problem import Flags
    >>> flags = Flags(True, True, False, False, False, False, True, True)
    >>> flags.constant_linop, flags.has_raman
    (True, False)
    """

    single_mode: bool
    scalar_pol: bool
    has_plasma: bool
    has_raman: bool
    has_noise: bool
    has_temporal_absorber: bool
    constant_linop: bool
    mode_averaged: bool
    n_channels: int = 0
    n_raman: int = 0

    @classmethod
    def from_content(
        cls,
        shape: tuple[int, int, int],
        linear: LinearOperator,
        transform: Transform,
        responses: CompiledResponses,
        initial: "Initial",
    ) -> "Flags":
        """Derive the flags from the problem's own content.

        Parameters
        ----------
        shape : tuple of int
            The canonical state shape ``(n_modes, n_pol, n_omega)``.
        linear : solprop.translate.linop.LinearOperator
            The cumulative operator.
        transform : solprop.translate.modal.Transform
            The modal transform.
        responses : solprop.translate.responses.CompiledResponses
            The compiled nonlinear term.
        initial : Initial
            The initial state and noise.

        Returns
        -------
        Flags
            The derived flags.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.nonlinear.gnlse import problem_from
        >>> from solprop.pulses import SechPulse
        >>> from solprop.stepper import PIController
        >>> from solprop.translate.problem import assemble
        >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
        >>> gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0,
        ...                      gamma=0.11, shock=False)
        >>> start = SechPulse(wavelength=835e-9, width=1e-13,
        ...                   peak_power=1.0).build(grid)
        >>> problem, request = assemble(
        ...     grid, gnlse.mode, [gnlse.kerr], start,
        ...     controller=PIController(rtol=1e-8, atol=1e-14),
        ...     save_planes=[1.0], dz_max=0.1, dz_init=1e-3,
        ...     shock=False, prefactor_index=1.0, density=1.0,
        ... )
        >>> problem.flags.mode_averaged, problem.flags.constant_linop
        (True, True)
        """
        return cls(
            single_mode=shape[0] == 1,
            scalar_pol=shape[1] == 1,
            has_plasma=bool(responses.population_channels),
            has_raman=bool(responses.raman_terms),
            has_noise=initial.noise0 is not None,
            has_temporal_absorber=responses.temporal_absorber is not None,
            constant_linop=linear.phi_representation in ("constant", "separable"),
            mode_averaged=transform.mode_averaged,
            n_channels=len(responses.population_channels),
            n_raman=len(responses.raman_terms),
        )


@dataclass(frozen=True, eq=False)
class Initial:
    """The initial state and the noise realisation of one trajectory.

    Section 6 of the kernel contract. Both arrays are in the lab frame and the FFT
    convention and read-only; the noise is C-contiguous, and so is the state unless it
    is a broadcast view.

    Every array is the problem's own, so that nothing the caller does to an array it
    passed, afterwards, changes the problem, and the caller's array is never frozen:

    - the state and the noise are copied, complex128, C-contiguous and read-only;
    - a broadcast state (zero strides, as from :func:`numpy.broadcast_to`) copies only
      the block it repeats, converted first, and broadcasts that copy, so it costs the
      block's bytes rather than one state's;
    - an array the problem already holds, which a replaced or re-translated problem
      passes back, is taken as it is, and so is one solprop built for the problem
      itself (``hold``), so that neither is copied twice.

    Attributes
    ----------
    field0 : numpy.ndarray
        Initial state, complex128, shape ``(n_modes, n_pol, n_omega)``, in the square
        root of watts; possibly a read-only broadcast view, never written. A group of
        trajectories is a sequence of problems, one per member (SEP 47 section 6.7),
        never a leading axis here.
    noise0 : numpy.ndarray or None
        The shot-noise realisation in the same layout, or ``None``. It is carried along
        z by the unitary part of the linear flow and added at every stage; the
        propagating field never contains it.
    noise_model : {'none', 'input', 'modified'}
        Which model the realisation belongs to.
    z : float, optional
        Absolute launch position in m, finite and non-negative. Input power
        normalisation uses the effective index at this position.

    Raises
    ------
    TranslationError
        If the noise layout does not match the state, or the model and the realisation
        disagree.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.problem import Initial
    >>> state = np.zeros((1, 1, 8), dtype=complex)
    >>> Initial(state, None, "none").noise_model
    'none'

    A broadcast input stays a broadcast, over a copy of the row it repeats:

    >>> row = np.ones((1, 1, 8), dtype=complex)
    >>> held = Initial(np.broadcast_to(row, (4, 1, 8)), None, "none").field0
    >>> held.shape, held.strides[0], np.shares_memory(held, row)
    ((4, 1, 8), 0, False)
    """

    field0: Complex[np.ndarray, "n_modes n_pol n_omega"]
    noise0: Complex[np.ndarray, "n_modes_n n_pol_n n_omega_n"] | None
    noise_model: NoiseModel
    z: float = 0.0

    def __post_init__(self) -> None:
        """Adopt the state as the class docstring says and check the noise against it.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        if not math.isfinite(self.z) or self.z < 0.0:
            raise TranslationError("initial.z must be finite and non-negative")
        state = _held_state(self.field0)
        object.__setattr__(self, "field0", state)
        if self.noise0 is not None:
            noise = frozen_copy(self.noise0, np.complex128)
            object.__setattr__(self, "noise0", noise)
            if noise.shape != state.shape:
                msg = (
                    f"noise0 has shape {noise.shape}, expected {state.shape} to match "
                    "field0"
                )
                raise TranslationError(msg)
        if (self.noise_model == "none") != (self.noise0 is None):
            msg = (
                f"noise_model is {self.noise_model!r} but the realisation is "
                f"{'absent' if self.noise0 is None else 'present'}"
            )
            raise TranslationError(msg)


_ADAMS_CONTROL = frozenset({"rtol", "atol", "weights", "region"})
"""The controller fields the Adams march reads. It chooses its steps by its own
``safety``, ``min_factor`` and ``max_growth``
(:class:`~solprop.stepper.multistep.Multistep`)."""


def _refuse_unread_control(controller: PIController) -> None:
    """Refuse a controller setting the Adams march would ignore, by name."""
    default = PIController(rtol=controller.rtol, atol=controller.atol)
    unread = [
        one.name
        for one in fields(PIController)
        if one.name not in _ADAMS_CONTROL
        and getattr(controller, one.name) != getattr(default, one.name)
    ]
    if unread:
        msg = (
            "Multistep() reads only the controller's rtol, atol, weights and region, "
            "and chooses its steps by its own safety, min_factor and max_growth: "
            f"PIController's {', '.join(unread)} would be ignored. Set them on "
            "Multistep, or leave them at their defaults"
        )
        raise TranslationError(msg)


@dataclass(frozen=True, eq=False)
class StepperSettings:
    """The tableau, the controller and the schedule of a run.

    Section 7 of the kernel contract. The tolerances and the norm choice live in the
    controller and nowhere else.

    Attributes
    ----------
    tableau : solprop.stepper.tableau.Tableau
        Nodes, matrix, solution weights, error weights ``b - b_hat``, orders and the
        first-same-as-last flag.
    controller : solprop.stepper.controller.PIController
        Tolerances, safety, growth caps, the PI exponents, the lag, the reject-free
        switch, the estimator order and the norm weights.
    dz_min : float
        Floor on the controller's *proposal*, m. A proposal below it raises rather than
        being clamped silently; it never applies to a landing step.
    dz_max : float
        Cap on the proposal, m.
    dz_init : float
        First step, m.
    stretch : float
        Landing stretch factor: a target within this multiple of the capped proposal is
        landed on directly. 1 switches it off.
    schedule : solprop.stepper.schedule.FixedSchedule or None
        A fixed schedule to replay, whose edges must contain every save plane and hard
        kink. ``None`` for an adaptive run; lockstep batches require one.
    save_planes : numpy.ndarray
        Positions in z the driver will ask for, m, shape ``(n_z,)``, increasing. Steps
        land on them exactly; there is no dense output.
    propagator_mode : {'auto', 'tabulate', 'recompute'}
        Whether the backend tabulates the stage propagator factors or recomputes them.
        ``"auto"`` leaves the choice to the backend, which records what it resolved.
    integrator : solprop.stepper.multistep.Multistep or LowStorageRungeKutta or None
        A low-call Adams method to step with instead of the tableau (SEP 13), the
        low-storage RK4 march (SEP 24), or ``None`` for the Runge-Kutta march. With a
        `schedule` an Adams method replays the schedule's steps (SEP 16 section 5);
        the low-storage march replays one and nothing else.
    interval_cap : bool
        Whether a step is capped at the width of the operator-table interval it starts
        in. The cumulative table makes the flow exact across intervals, so the cap is
        off on the diagonal-flow routes and on for a coupled flow, whose flow error the
        embedded estimate does not see (SEP 14 section 4.1).
    auto_init : bool
        Whether the start-up rule takes the step after the first from the first step's
        own error estimate (``PIController.start_factor``); ``False`` grows from
        `dz_init` at the growth cap.
    interpolate_planes : bool
        Step across the save planes and observe each inside its accepted step, through
        the tableau's continuous extension or the Adams march's own polynomial (SEP 14
        phase 3); ``False`` lands on them. The end of the run and every hard kink are
        landed on either way.
    step_planes : int
        A positive ``k`` also saves the end of every ``k``-th accepted step: the
        march's ``propagate_to`` returns there, short of its target, and is called
        again (SEP 14 phase 3). ``0`` saves the requested planes only.

    Raises
    ------
    TranslationError
        If the step caps are not ordered, the stretch is below one, the save
        planes are not increasing and positive, or the low-storage march is asked for
        without a schedule, with interpolated planes or with a schedule step longer
        than `dz_max`.

    Examples
    --------
    >>> from solprop.stepper import DOPRI5, PIController
    >>> from solprop.translate.problem import StepperSettings
    >>> settings = StepperSettings(DOPRI5, PIController(rtol=1e-8, atol=1e-14),
    ...                            1e-9, 0.1, 1e-3, 1.1, None, [0.5, 1.0], "tabulate")
    >>> settings.save_planes.tolist()
    [0.5, 1.0]
    """

    tableau: Tableau
    controller: PIController
    dz_min: float
    dz_max: float
    dz_init: float
    stretch: float
    schedule: FixedSchedule | None
    save_planes: Float[np.ndarray, " n_z"] | Sequence[float]
    propagator_mode: PropagatorMode
    integrator: Multistep | LowStorageRungeKutta | None = None
    interval_cap: bool = True
    auto_init: bool = False
    interpolate_planes: bool = False
    step_planes: int = 0

    def __post_init__(self) -> None:
        """Adopt a read-only plane array and check the caps.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        planes = frozen_copy(np.reshape(self.save_planes, -1), np.float64)
        object.__setattr__(self, "save_planes", planes)
        if planes.size == 0 or not np.all(np.diff(planes) > 0.0):
            msg = f"save_planes must be non-empty and increasing, got {planes}"
            raise TranslationError(msg)
        if planes[0] <= 0.0:
            msg = f"save_planes must be positive, got {planes[0]!r} first"
            raise TranslationError(msg)
        if not 0.0 < self.dz_min <= self.dz_init <= self.dz_max:
            msg = (
                f"the step caps must satisfy 0 < dz_min <= dz_init <= dz_max, got "
                f"{self.dz_min!r}, {self.dz_init!r}, {self.dz_max!r}"
            )
            raise TranslationError(msg)
        if not self.stretch >= 1.0:
            msg = f"stretch must be at least 1, got {self.stretch!r}"
            raise TranslationError(msg)
        if self.step_planes < 0:
            msg = f"step_planes must be 0 (off) or a positive count: {self.step_planes}"
            raise TranslationError(msg)
        if isinstance(self.integrator, LowStorageRungeKutta):
            self._check_low_storage(self.integrator)
        if isinstance(self.integrator, Multistep):
            _refuse_unread_control(self.controller)
        if self.interpolate_planes and self.schedule is not None:
            msg = "interpolated planes lie inside adaptive steps; a schedule lands"
            raise TranslationError(msg)
        if (
            self.interpolate_planes
            and self.integrator is None
            and continuous_extension(self.tableau) is None
        ):
            msg = f"{self.tableau.name} has no continuous extension; use DOPRI5"
            raise TranslationError(msg)

    def _check_low_storage(self, method: LowStorageRungeKutta) -> None:
        """Refuse what the low-storage march cannot replay, before any allocation.

        The march accepts every step in place and has no estimate, so it replays a
        prescribed schedule and nothing else (SEP 24 section 5); `dz_max` stays a true
        statement about the run it records, so a longer schedule step is refused.
        """
        if self.schedule is None:
            msg = (
                f"the {method.name} march replays a prescribed schedule; pass "
                "schedule=FixedSchedule.uniform(...), or step adaptively with "
                "RungeKutta()"
            )
            raise TranslationError(msg)
        if self.interpolate_planes:
            msg = (
                f"the {method.name} march lands on its schedule's edges and has no "
                "dense output; drop interpolate_planes"
            )
            raise TranslationError(msg)
        # A schedule's steps are differences of rounded positions, so one may exceed
        # the step it was cut to by a few ulps of those positions.
        span = max(abs(self.schedule.z_start), abs(self.schedule.z_end), self.dz_max)
        longest = self.schedule.max_step
        if longest > self.dz_max + 4.0 * float(np.spacing(span)):
            msg = (
                f"the schedule's longest step is {longest:.6g} m and dz_max is "
                f"{self.dz_max:.6g} m; the {method.name} march records dz_max as a "
                "limit of the run, so raise dz_max to the schedule's longest step"
            )
            raise TranslationError(msg)

    @property
    def z_end(self) -> float:  # numpydoc ignore=RT01
        """Last save plane, where the run stops, m.

        Examples
        --------
        >>> from solprop.stepper import DOPRI5, PIController
        >>> from solprop.translate.problem import StepperSettings
        >>> StepperSettings(DOPRI5, PIController(rtol=1e-8, atol=1e-14), 1e-9, 0.1,
        ...                 1e-3, 1.1, None, [0.5, 1.0], "tabulate").z_end
        1.0
        """
        return float(np.max(self.save_planes))


class _RunPolicy(NamedTuple):
    """The run's policy the request-aware checks read (:func:`validate_run`).

    Attributes
    ----------
    stepper : StepperSettings
        The method, step limits, schedule and planes.
    stats : solprop.stats.StatsSpec
        The statistics.
    """

    stepper: StepperSettings
    stats: StatsSpec


def _held_block(values: np.ndarray, dtype: np.dtype | type[np.generic]) -> np.ndarray:
    """Return an array as solprop holds it, a broadcast copying its block alone.

    - Returns: read-only, held and of `dtype`: for a broadcast view, a broadcast of the
      block it repeats, which is copied unless already held and C-contiguous; otherwise
      :func:`~solprop._ownership.frozen_copy` of `values`, so a strided view is copied
      to a C-contiguous array even when it views a held one.
    """
    repeated = tuple(
        size > 1 and stride == 0
        for size, stride in zip(values.shape, values.strides, strict=True)
    )
    if not any(repeated):
        return frozen_copy(values, dtype)
    sliced = values[tuple(slice(0, 1) if axis else slice(None) for axis in repeated)]
    block = frozen_copy(sliced, dtype)
    if block is sliced and is_held(values):
        # Already a broadcast of a held block: taken back as it is.
        return values
    # A read-only view of a held block, so itself held.
    return np.broadcast_to(block, values.shape)


def _held_state(field0: Num[np.ndarray, "*shape"]) -> Complex[np.ndarray, "*shape"]:
    """Return the initial state as :class:`Initial` holds it, copying only the block.

    - Returns: Complex128 and read-only: `field0` itself when it is already held as
      complex128; for a broadcast view, a broadcast of a held copy of the block it
      repeats; otherwise a held C-contiguous copy (:func:`_held_block`).
    """
    return _held_block(np.asarray(field0), np.complex128)


@dataclass(frozen=True, eq=False)
class PropagationProblem:
    """Everything a backend needs, and nothing it does not.

    The seam of ``docs/dev/architecture.md`` section 3: the numerical meaning of a
    run and nothing of how it is run. The tableau, controller, step limits, schedule,
    planes and statistics are the run's policy, which a
    :class:`~solprop.translate.run_request.RunRequest` carries beside it (SEP 47
    section 5.2; kernel contract sections 7 and 8). Construct both with
    :func:`~solprop.translate.assembly.assemble` rather than by hand; a backend checks
    the pair with :func:`validate_run` before it allocates, which
    :func:`~solprop.translate.assembly.assemble` already does.

    Attributes
    ----------
    grid_kind : {'field', 'envelope'}
        Selects r2c/c2r against c2c transforms and the response formulas.
    shape : tuple of int
        The canonical single-trajectory state shape ``(n_modes, n_pol, n_omega)``.
    flags : Flags
        Static booleans, derived from the content (section 1).
    grids : solprop.translate.masks.Grids
        Frequency axis, masks and oversampling data (section 2).
    linear : solprop.translate.linop.LinearOperator
        The cumulative linear operator (section 3).
    responses : solprop.translate.responses.CompiledResponses
        The compiled nonlinear term (section 5).
    initial : Initial
        Initial state and noise (section 6).
    route : solprop.translate.routes.ModalRoute, GeneralRoute or SpatialRoute
        What differs between routes, in exactly one variant of a closed union, each
        with the fields its executor needs (SEP 47 section 6.2): the transform
        (section 4) and the route's own leaves (:mod:`solprop.translate.routes`).
    split_crossing : solprop.boundaries.SplitCrossing or None, optional
        The split temporal collar's crossing limit (version 43): each bin's walk-off
        bucket, the collar's width and the launch's limit, which the boundary
        monitor reads again from the state at every check. ``None`` under the
        in-stage scheme or without a temporal collar.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0, gamma=0.11,
    ...                      shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=1e-13, peak_power=1.0).build(grid)
    >>> problem, request = assemble(
    ...     grid, gnlse.mode, [gnlse.kerr], start,
    ...     controller=PIController(rtol=1e-8, atol=1e-14), save_planes=[1.0],
    ...     dz_max=0.1, dz_init=1e-3, shock=False, prefactor_index=1.0, density=1.0,
    ... )
    >>> problem.shape, problem.grid_kind
    ((1, 1, 3072), 'envelope')
    """

    grid_kind: GridKind
    shape: tuple[int, int, int]
    flags: Flags
    grids: Grids
    linear: LinearOperator
    responses: CompiledResponses
    initial: Initial
    route: Route
    split_crossing: SplitCrossing | None = None

    @property
    def transform(self) -> Transform:  # numpydoc ignore=RT01
        """The route's transform, which every route variant carries.

        Synthesis and projection (section 4).
        """
        return self.route.transform

    @property
    def device_ready(self) -> bool:  # numpydoc ignore=RT01
        """Whether both z-varying arrays are in the seam's device form.

        True when ``Phi`` and the prefactor each evaluate as a weighted sum of a fixed
        set of arrays over host scalars -- the constant fast path with one term and the
        distance as its weight, or a parameter expansion with ``n_terms`` of them
        (:mod:`solprop.translate.stage`). False when either took the z table, whose
        evaluation is a Clenshaw pass over an array of the state's size times the
        degree: the general representation, which the CPU backends evaluate (the
        native CPU build into term slots of its own, SEP 29 section 6.6) and an
        accelerator build refuses through
        :func:`solprop.translate.stage.require_device_ready`.

        Whether neither representation is ``"table"``.

        Examples
        --------
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.nonlinear.gnlse import problem_from
        >>> from solprop.pulses import SechPulse
        >>> from solprop.stepper import PIController
        >>> from solprop.translate.problem import assemble
        >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
        >>> gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0,
        ...                      gamma=0.11, shock=False)
        >>> start = SechPulse(wavelength=835e-9, width=1e-13,
        ...                   peak_power=1.0).build(grid)
        >>> assemble(grid, gnlse.mode, [gnlse.kerr], start,
        ...          controller=PIController(rtol=1e-8, atol=1e-14),
        ...          save_planes=[1.0], dz_max=0.1, dz_init=1e-3, shock=False,
        ...          prefactor_index=1.0, density=1.0).problem.device_ready
        True
        """
        return (
            self.linear.phi_representation != "table"
            and self.responses.prefactor_representation != "table"
        )

    def validate(self) -> None:
        """Check every promise the problem makes alone, raising on the first that fails.

        The part of :func:`validate_run` that needs no run request: every array
        finite, read-only, of the stated dtype and shape, and C-contiguous but for the
        initial state, which may be a broadcast view (:class:`Initial`); the masks
        Boolean; the breakpoints strictly increasing and every table along z spanning
        the same range and, for the prefactor, carrying the state's value shape; a
        constant prefactor, when present, standing in for a table that is one interval
        of degree zero and agreeing with it; the hard kinks interval edges; the Raman
        rows agreeing with the fine grid and with each other; the general modal route
        and the coupled flow ones solprop executes
        (:func:`solprop.translate.admission.check_general_route`). What depends on the
        method, the step limits, the schedule, the planes or the statistics -- the
        tables covering the run, the schedule landing on its targets, the collar's
        stability, the exponent budget, the split collar's crossing and the method's
        admission -- is checked by :func:`validate_run`, which a backend calls before
        it allocates.

        Quadrature weights are checked where they exist, in
        :mod:`solprop.modes.quadrature`: by the time the transform is built only their
        product with the mode fields survives, so this checks that product is finite
        instead.

        Raises
        ------
        TranslationError
            Naming the field that failed.

        Examples
        --------
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.nonlinear.gnlse import problem_from
        >>> from solprop.pulses import SechPulse
        >>> from solprop.stepper import PIController
        >>> from solprop.translate.problem import assemble
        >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
        >>> gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0,
        ...                      gamma=0.11, shock=False)
        >>> start = SechPulse(wavelength=835e-9, width=1e-13,
        ...                   peak_power=1.0).build(grid)
        >>> problem, request = assemble(
        ...     grid, gnlse.mode, [gnlse.kerr], start,
        ...     controller=PIController(rtol=1e-8, atol=1e-14), save_planes=[1.0],
        ...     dz_max=0.1, dz_init=1e-3, shock=False, prefactor_index=1.0,
        ...     density=1.0,
        ... )
        >>> problem.validate() is None
        True
        """
        self._validate(None)

    def _validate(self, run: _RunPolicy | None) -> None:
        """Run the checks in their one order, those of the run only with a request.

        - ``run``: the run's stepper section and statistics, or ``None`` for the checks
          the problem makes alone.
        """
        if run is not None:
            self._check_integrator(run.stepper)
        self._check_route(run)
        if run is not None and not self.initial.z < run.stepper.save_planes[0]:
            raise TranslationError("save planes must follow the initial position")
        self._check_arrays(run)
        self._check_shapes()
        self._check_power_weight()
        self._check_spatial(run)
        self._check_carrier()
        self._check_species()
        self._check_plasma()
        self._check_envelope_factors()
        self._check_raman()
        self._check_dispersive()
        self._check_modal_leaves(run)
        check_general_route(self)  # the route's policy, once its leaves are consistent
        self._check_tables(run)

    def validate_launch(self, request: "RunRequest") -> None:
        """Check the launch against the save planes, and a schedule against them.

        The part of :func:`validate_run` a replaced input with a new launch position
        needs again (:meth:`solprop.driver.simulation.Simulation.with_input`): every
        other leaf is shared, unchanged, with a problem that was validated whole.

        Parameters
        ----------
        request : solprop.translate.run_request.RunRequest
            The run's request, with the planes left beyond the new launch.

        Raises
        ------
        TranslationError
            If a save plane does not follow the launch, or a replayed schedule does
            not land on every save plane and hard kink.

        Examples
        --------
        >>> from solprop.driver import Simulation
        >>> simulation = Simulation.from_gnlse(
        ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
        ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
        ...     width=100e-15, peak_power=9.8, shape="sech",
        ... )
        >>> problem = simulation.translate()
        >>> problem.validate_launch(simulation._run_request()) is None
        True
        """
        stepper = request.stepper
        if not self.initial.z < stepper.save_planes[0]:
            raise TranslationError("save planes must follow the initial position")
        if stepper.schedule is not None:
            self._check_schedule(stepper)

    def _check_route(self, run: _RunPolicy | None) -> None:
        """Visit the route variant and check what it shares with the other groups.

        Each variant checks its own fields (``validate``, also run on construction);
        what spans groups is checked here, one variant at a time (SEP 47 section 6.2).
        """
        # Again here, as every group's own checks are: a problem may hold a route
        # that was changed after it was built.
        self.route.validate()
        match self.route:
            case ModalRoute(tdse=TDSESource() as source):
                self._check_tdse(source, None if run is None else run.stepper)
                unavailable = (
                    set()
                    if run is None
                    else TDSE_UNAVAILABLE_STATISTICS.intersection(
                        (*run.stats.names, *run.stats.periodic)
                    )
                )
                if unavailable:
                    msg = (
                        "TDSE does not define legacy statistics "
                        f"{sorted(unavailable)}; request explicitly labelled atomic "
                        "diagnostics"
                    )
                    raise TranslationError(msg)
            case ModalRoute() | GeneralRoute():
                pass
            case SpatialRoute(flux=flux):
                if not np.array_equal(flux.base.active, self.grids.active_mask):
                    msg = "the plane measure's active mask must match the temporal grid"
                    raise TranslationError(msg)
            case _:  # pragma: no cover - the union is closed and every variant is above
                msg = f"a problem carries a route of type {type(self.route).__name__}"
                raise TranslationError(msg)

    def _check_integrator(self, stepper: StepperSettings) -> None:
        """Refuse what the low-storage march does not step, by name (SEP 24 section 5).

        Its half-step flows are exact exponentials of commuting diagonal values, so a
        coupled flow, whose flows are not, would leave an error outside the
        Runge-Kutta order; the TDSE response is the Adams route's alone.
        """
        method = stepper.integrator
        if not isinstance(method, LowStorageRungeKutta):
            return
        if self.linear.coupled_flow is not None:
            msg = (
                f"the {method.name} march steps a diagonal linear flow only: a coupled "
                "flow's half-step flows are not exponentials of commuting values, and "
                "their error would lie outside the Runge-Kutta order. Step it with "
                "RungeKutta() or Multistep(formulation='exponential')"
            )
            raise TranslationError(msg)
        if tdse_of(self.route) is not None:
            msg = (
                f"the {method.name} march does not step the TDSE response, which "
                "runs on the Adams route alone; use Multistep()"
            )
            raise TranslationError(msg)

    def _check_tdse(self, source: TDSESource, stepper: StepperSettings | None) -> None:
        """Refuse a manually constructed atomic seam outside the admitted route.

        - ``stepper``: the run's stepper section; ``None`` leaves the method to
          :func:`validate_run`.
        """
        descriptor = source.descriptor
        if (
            self.grid_kind != "field"
            or self.shape[:2] != (1, 1)
            or self.transform.n_pol != 1
            or self.flags.mode_averaged
            # The input draw is part of the launch and drives the atom with it; the
            # modified one is kept apart and is not (SEP 37, section 6).
            or self.initial.noise_model == "modified"
            or self.flags.has_plasma
            or self.flags.has_raman
            or (
                stepper is not None
                and (
                    not isinstance(stepper.integrator, Multistep)
                    or stepper.interpolate_planes
                    or stepper.schedule is not None
                )
            )
            or self.linear.coupled_flow is not None
            or self.responses.absorber_scheme != "nonlinear"
        ):
            raise TranslationError(
                "TDSE requires the scalar resolved modal Adams route"
            )
        if (
            descriptor.samples != self.grids.n_to
            or descriptor.spacing != self.grids.dt_fine
            or descriptor.time_origin != -self.grids.n_to * self.grids.dt_fine / 2
        ):
            raise TranslationError(
                "TDSE descriptor sampling differs from the response grid"
            )
        if descriptor.gated:
            configuration = json.loads(descriptor.configuration_json)
            collar = Collar(**configuration["lattice"]["time_collar"])
            times = (
                np.arange(self.grids.n_to) - self.grids.n_to / 2
            ) * self.grids.dt_fine
            if not np.array_equal(self.grids.t_window_fine, collar.window(times)):
                raise TranslationError(
                    "TDSE source collar differs from its recorded policy"
                )
        if descriptor.gated:
            certificate = steady_current_certificate(
                descriptor, self.grids.omega, self.grids.active_mask
            )
            if not certificate <= STEADY_CURRENT_CERTIFICATE_LIMIT:
                raise TranslationError(
                    "the gated native TDSE source is not certified on this "
                    "lattice and band: its steady-current certificate "
                    f"{certificate:.3g} exceeds {STEADY_CURRENT_CERTIFICATE_LIMIT:g}: "
                    "the sampled tapers do not resolve the gate; widen the tapers or "
                    "refine the response lattice (TDSESampling)"
                )
        expected = np.where(
            self.grids.active_mask, -self.grids.crop_scale / (2 * EPS0 * C), 0.0
        )
        if source.current_numerator.shape != self.shape or not np.array_equal(
            source.current_numerator, expected[None, None, :]
        ):
            raise TranslationError(
                "TDSE current numerator differs from the physical coefficient"
            )
        cubic = self.responses.kerr_coefficients
        if descriptor.kerr_model == "optmat_cubic":
            if cubic is None or cubic.cubic_form != "real" or cubic.analytic_signal:
                raise TranslationError(
                    "TDSE optmat_cubic requires its real full cubic coefficient"
                )
        elif cubic is not None:
            raise TranslationError(
                "TDSE shadow/native model must not also own an optmat cubic"
            )
        if self.responses.dispersive_kerr is not None:
            raise TranslationError(
                "TDSE owns the complete electronic response of its gas and refuses a "
                "dispersive Kerr response beside it"
            )

    def _check_arrays(self, run: _RunPolicy | None) -> None:
        """Check every stored array is finite, contiguous, read-only and typed.

        The initial state alone may be strided (:data:`_STRIDED_LEAVES`).

        - ``run``: the run's stepper section and statistics, whose arrays are held to
          the same rules under their names ``stepper`` and ``stats_spec``; ``None``
          checks the problem's alone.
        """
        for name, values in _walk_arrays(self, run):
            if not values.flags.c_contiguous and name not in _STRIDED_LEAVES:
                msg = f"{name} is not C-contiguous; a kernel indexes it directly"
                raise TranslationError(msg)
            if values.flags.writeable:
                msg = f"{name} is writeable; arrays crossing the seam are frozen"
                raise TranslationError(msg)
            allowed = (
                _SEAM_DTYPES
                + ((np.int32,) if name in _INT32_LEAVES else ())
                + ((np.uint8,) if name in _UINT8_LEAVES else ())
            )
            if values.dtype not in allowed:
                msg = (
                    f"{name} has dtype {values.dtype}; the contract carries float64, "
                    "complex128, bool and intp, and int32 and uint8 only at the "
                    "leaves it declares"
                )
                raise TranslationError(msg)
            if name in _UNBOUNDED_LEAVES and not np.any(np.isnan(values)):
                continue
            if values.dtype != np.bool_ and not np.all(np.isfinite(values)):
                bad = int(np.sum(~np.isfinite(values)))
                msg = f"{name} has {bad} of {values.size} entries that are not finite"
                raise TranslationError(msg)
        for name in ("inband_mask", "active_mask", "norm_mask"):
            mask = getattr(self.grids, name)
            if mask.dtype != np.bool_:
                msg = f"grids.{name} must be Boolean, got {mask.dtype}"
                raise TranslationError(msg)

    def _check_shapes(self) -> None:
        """Check every array agrees with the canonical state shape."""
        n_modes, n_pol, n_omega = self.shape
        if self.initial.field0.shape != self.shape:
            msg = (
                f"initial.field0 has shape {self.initial.field0.shape}, expected "
                f"{self.shape}"
            )
            raise TranslationError(msg)
        if self.grids.n_omega != n_omega:
            msg = (
                f"grids.omega has {self.grids.n_omega} bins, expected {n_omega} to "
                "match the state"
            )
            raise TranslationError(msg)
        # The operator's values keep a leading broadcast axis of one, the layout the
        # native build reads; the state has none (SEP 47 section 6.2).
        value = self.linear.value_shape
        if value != (1, n_modes, n_pol, n_omega):
            msg = (
                f"linear carries values of shape {value}, expected "
                f"{(1, n_modes, n_pol, n_omega)}"
            )
            raise TranslationError(msg)
        value = self.responses.prefactor_value_shape
        if value != (n_modes, n_pol, n_omega):
            msg = (
                f"responses.prefactor carries values of shape {value}, expected "
                f"{(n_modes, n_pol, n_omega)}"
            )
            raise TranslationError(msg)
        self._check_prefactor_constant()
        if self.transform.n_modes != n_modes:
            msg = (
                f"transform carries {self.transform.n_modes} modes, expected {n_modes}"
            )
            raise TranslationError(msg)
        if self.transform.n_pol * n_pol > MAX_N_POL:
            msg = (
                f"transform resolves {self.transform.n_pol} transverse components and "
                f"the state carries {n_pol}; the components belong to one of the two "
                "and not to both (solprop.modes.collection.ModeCollection."
                "with_polarisations builds the polarisation-resolved mode set a "
                "component-resolving transform expects)"
            )
            raise TranslationError(msg)
        for name, expected in (
            ("temporal_absorber", self.grids.n_t),
            ("temporal_absorber_fine", self.grids.n_to),
        ):
            rate = getattr(self.responses, name)
            if rate is not None and rate.size != expected:
                msg = f"responses.{name} has {rate.size} samples, expected {expected}"
                raise TranslationError(msg)

    def _check_power_weight(self) -> None:
        """Require a positive modal weight representation with matching shape."""
        linear = self.linear
        kind = linear.power_weight_representation
        table = linear.power_weight
        expansion = linear.power_weight_expansion
        constant = linear.power_weight_constant
        if self.transform.geometry is not None:
            if any(value is not None for value in (kind, table, expansion, constant)):
                msg = "power_weight must be absent for spatial geometry"
                raise TranslationError(msg)
            return
        if (
            kind not in ("constant", "expansion", "table")
            or (table is not None) != (kind in ("constant", "table"))
            or (expansion is not None) != (kind == "expansion")
            or (constant is not None) != (kind == "constant")
        ):
            msg = "power_weight representation and present arrays disagree"
            raise TranslationError(msg)
        value = expansion if expansion is not None else table
        assert value is not None
        if value.value_shape != self.shape:
            msg = f"power_weight shape {value.value_shape} does not match {self.shape}"
            raise TranslationError(msg)
        for z in value.z_span:
            if np.any(value(z) <= 0):
                msg = "power_weight must be positive"
                raise TranslationError(msg)
        if constant is not None and (
            table is None
            or table.degrees.tolist() != [0]
            or constant.shape != self.shape
            or not np.array_equal(table(0.0), constant)
        ):
            msg = "power_weight_constant must match its degree-zero table"
            raise TranslationError(msg)

    def _check_spatial(self, run: _RunPolicy | None) -> None:
        """Check spatial geometry and the shared separable representation agree.

        - ``run``: the run's stepper section and statistics; ``None`` leaves the
          propagator mode to :func:`validate_run`.
        """
        geometry = self.transform.geometry
        separable = self.linear.separable
        if self.responses.separable is not separable:
            msg = "linear and responses must share the same separable coefficients"
            raise TranslationError(msg)
        if separable is not None:
            if geometry is None or geometry.kind not in ("free", "slab"):
                msg = "a separable spatial operator requires free or slab geometry"
                raise TranslationError(msg)
            if not np.array_equal(separable.active, self.grids.active_mask):
                msg = "separable active mask must match the temporal grid"
                raise TranslationError(msg)
            if geometry.kind == "slab":
                self._check_slab(separable)
            else:
                if separable.kappa_sq is not None:
                    msg = "separable kappa_sq belongs to the slab geometry alone"
                    raise TranslationError(msg)
                kx, ky = geometry.spectral_axes
                expected = (kx[None, :] ** 2 + ky[:, None] ** 2).reshape(-1)
                if not np.array_equal(separable.kperp_sq, expected):
                    msg = (
                        "separable kperp_sq must match the spatial FFT axes with x "
                        "fastest"
                    )
                    raise TranslationError(msg)
        if geometry is not None and geometry.kind in ("free", "slab"):
            if separable is None:
                msg = (
                    f"{geometry.kind} propagation requires a constant separable "
                    "operator; varying media are unsupported"
                )
                raise TranslationError(msg)
            # The low-storage march tabulates one volume, exp(h D / 2), which every
            # step of a fixed schedule reuses (SEP 24 section 4.3); the Runge-Kutta
            # march's six fractions would need twelve.
            if (
                run is not None
                and run.stepper.propagator_mode == "tabulate"
                and not isinstance(run.stepper.integrator, LowStorageRungeKutta)
            ):
                msg = (
                    f"{geometry.kind} propagation refuses propagator_mode='tabulate' "
                    "except under LowStorageRungeKutta(); use 'recompute'"
                )
                raise TranslationError(msg)

    def _check_slab(self, separable: SeparableOperator) -> None:
        """Check the hybrid planar operator against its geometry and transform.

        - ``separable``: the shared coefficients, which must carry one wall term per
          channel.
        """
        geometry = self.transform.geometry
        assert geometry is not None
        n_channels = self.transform.synthesis.shape[1]
        if separable.kappa_sq is None or separable.n_channels != n_channels:
            msg = (
                "slab propagation requires separable kappa_sq with one row per "
                f"channel of the transform ({n_channels})"
            )
            raise TranslationError(msg)
        (k_y,) = geometry.spectral_axes
        if not np.array_equal(separable.kperp_sq, np.repeat(k_y**2, n_channels)):
            msg = (
                "separable kperp_sq must be k_y**2 per column with the channel fastest"
            )
            raise TranslationError(msg)

    def _check_prefactor_constant(self) -> None:
        """Check a present constant prefactor against the representation beside it.

        The fast path is a claim about the prefactor's own representation, not about
        the operator: a table must be one interval of degree zero, an
        expansion one term, and either must hold this array, so that a backend taking
        the array sees what a Clenshaw pass would.
        """
        constant = self.responses.prefactor_constant
        if constant is None:
            return
        table = self.responses.prefactor
        expansion = self.responses.prefactor_expansion
        if table is not None and (
            table.degrees.size != 1 or int(table.degrees[0]) != 0
        ):
            msg = (
                "responses.prefactor_constant is present but the prefactor table "
                f"has {table.degrees.size} interval(s) of degree up to "
                f"{int(table.degrees.max())}; the fast path is only for a prefactor "
                "that does not vary along z"
            )
            raise TranslationError(msg)
        if expansion is not None and expansion.n_terms != 1:
            msg = (
                "responses.prefactor_constant is present but the prefactor expansion "
                f"has {expansion.n_terms} terms; the fast path is only for a prefactor "
                "that does not vary along z"
            )
            raise TranslationError(msg)
        # Evaluated from the representation itself and not through `prefactor_at`,
        # which would hand back the very array under test.
        low = self.responses.prefactor_span[0]
        sampled = (
            expansion(low)
            if expansion is not None
            else cast("PiecewiseChebyshev", table)(low)
        )
        if not np.allclose(sampled, constant, rtol=1e-12, atol=0.0):
            msg = (
                "responses.prefactor_constant does not agree with the prefactor "
                "representation it stands in for"
            )
            raise TranslationError(msg)

    def _check_collar_stability(self, stepper: StepperSettings) -> None:
        """Refuse a step the run cannot avoid that is too long for an in-stage collar.

        The ``"nonlinear"`` absorber scheme makes the collar's largest rate an
        eigenvalue of the linearised problem, so an explicit method must keep
        ``h alpha / 2`` inside its stability region
        (:func:`solprop.boundaries.stable_step`). An
        *adaptive* run finds that for itself -- it costs steps rather than correctness,
        and only until the limit is above the step accuracy wants -- and is not
        refused. Two configurations cannot find it: a replayed schedule, whose steps are
        already fixed, and a floor ``dz_min`` set above the limit. Past the limit the
        collar amplifies the field it should absorb, and the run diverges, so they are
        refused rather than run.
        """
        geometry = self.transform.geometry
        if geometry is not None and np.any(geometry.absorber_rate):
            self._check_spatial_collar_stability(stepper)
            self._check_collar_crossing(stepper)
            return
        rate = self.responses.temporal_absorber
        if rate is None or self.responses.absorber_scheme != "nonlinear":
            return
        radius = self._stability_radius(stepper)
        limit = stable_step(rate) * radius / STABILITY_RADIUS
        schedule = stepper.schedule
        longest, source = (
            (float(np.max(schedule.steps)), "the replayed schedule's longest step")
            if schedule is not None
            else (stepper.dz_min, "the step floor dz_min")
        )
        if longest <= limit:
            return
        printed = _printed_down(limit)
        advice = (
            _history_replay_advice(printed, split_step=True, integrator=True)
            if schedule is not None
            else ""
        )
        msg = (
            f"{source} is {longest:.3e} m and the temporal collar is stable in the "
            f"nonlinear absorber scheme only up to {printed} m: the collar reaches "
            f"{float(np.max(rate)):.3e} 1/m, and an explicit method needs "
            f"h alpha / 2 below {radius}, so the run would diverge. "
            "Take shorter steps, weaken the collar with a larger boundary_length, "
            f"{self._runge_kutta_advice(stepper)}or "
            'ask for absorber_scheme="split_step", which integrates the rate exactly '
            "and is unconditionally stable at the cost of first-order convergence."
            f"{advice}"
        )
        raise TranslationError(msg)

    def _check_exponent_budget(self, stepper: StepperSettings) -> None:
        """Refuse a step the run cannot shrink whose propagator factor overflows.

        SEP 22 section 4.5, on the separable operator: a replayed schedule's longest
        step, or the floor ``dz_min``, ``h``, forms ``exp(h lambda)`` with ``lambda``
        the larger of the loss bound and ``sigma`` times the gain bound
        (:func:`solprop.translate.spatial.rate_bounds`), ``sigma`` being one for the
        Runge-Kutta and low-storage marches and ``q + 1`` for an Adams method of
        largest order ``q``, whose exponential carry spans that many steps. Past
        ``log(DBL_MAX)`` the factor is infinite on the first such step, so the run
        cannot complete and is refused; past nine tenths of it the run completes only
        while its field stays small, and translation warns. An adaptive run is held to
        the budget only at its floor ``dz_min``; above it, its controller rejects a
        non-finite trial and shrinks. The radial kind carries a dense operator and
        forms the same inverse factor from it, so it is bounded by
        :func:`solprop.translate.radial.radial_rate_bounds` instead (SEP 22, Q16);
        the modal kinds are not checked.
        """
        operator = self.linear.separable
        geometry = self.transform.geometry
        if operator is not None:
            bounds = rate_bounds(operator)
            attenuation = "the medium's attenuation"
        elif geometry is not None and geometry.kind == "radial":
            bounds = radial_rate_bounds(self.linear)
            attenuation = "the radial operator's attenuation"
        else:
            return
        schedule = stepper.schedule
        step, source = (
            (float(np.max(schedule.steps)), "replayed step")
            if schedule is not None
            else (stepper.dz_min, "step floor dz_min")
        )
        method = stepper.integrator
        span = method.order + 1 if isinstance(method, Multistep) else 1
        rate = max(bounds.loss, span * bounds.gain)
        exponent = step * rate
        if exponent <= EXPONENT_BUDGET * LOG_DBL_MAX:
            return
        message = _budget_message(source, step, bounds, span, exponent, attenuation)
        if exponent > LOG_DBL_MAX:
            raise TranslationError(
                message
                + " that overflows; "
                + _budget_remedy(step, bounds, span, LOG_DBL_MAX, attenuation)
            )
        warnings.warn(
            message
            + f" within a tenth of the overflow limit exp({LOG_DBL_MAX:.2f}): the run "
            "completes only while the field stays below exp of the difference; "
            + _budget_remedy(
                step, bounds, span, EXPONENT_BUDGET * LOG_DBL_MAX, attenuation
            ),
            ExponentBudgetWarning,
            stacklevel=5,
        )

    @staticmethod
    def _runge_kutta_advice(stepper: StepperSettings) -> str:
        """Name the Runge-Kutta march where an Adams method's interval is the limit.

        - Returns: a clause ending in ", " for an Adams run, otherwise empty.
        """
        if not isinstance(stepper.integrator, Multistep):
            return ""
        return (
            f"step with integrator=RungeKutta(), whose interval is {STABILITY_RADIUS}, "
        )

    @staticmethod
    def _stability_radius(stepper: StepperSettings) -> float:
        """Return the stepper's negative-real stability interval for a damping term.

        DOPRI5's :data:`solprop.boundaries.STABILITY_RADIUS`, the Adams method's far
        smaller one at its highest order (SEP 16 admits Adams schedules), or RK4's
        2.785 under the low-storage march (SEP 24 section 4.6).
        """
        method = stepper.integrator
        if method is None:
            return STABILITY_RADIUS
        return min(STABILITY_RADIUS, method.real_stability_interval)

    def _check_spatial_collar_stability(self, stepper: StepperSettings) -> None:
        """Refuse a step past the spatial collar's limit in the energy-weighted basis.

        The step checked is a replayed schedule's longest, or else the step floor.
        """
        geometry = self.transform.geometry
        if geometry is None:
            return
        # Weighted inverse synthesis is T for the radial QDHT. F is its weighted
        # adjoint, so the damping operator is negative semidefinite and its largest
        # rate is at most max(alpha)*||T||_2^2. Include the SVD/scale rounding model;
        # exact orthogonality is neither assumed nor enforced.
        operations = 128 * self.transform.n_modes
        rounding = operations * np.finfo(float).eps
        rounding /= 1 - rounding
        if geometry.kind == "radial":
            inverse = self.transform.synthesis / self.grids.pad_scale
            weighted = (
                np.sqrt(geometry.spatial_weights)[:, None]
                * inverse
                / np.sqrt(geometry.spectral_weights)[None, :]
            )
            factor = (float(np.linalg.norm(weighted, ord=2)) * (1 + rounding)) ** 2
        else:
            factor = 1 + rounding
        largest = float(np.max(geometry.absorber_rate)) * factor
        temporal = self.responses.temporal_absorber_fine
        if temporal is not None and self.responses.absorber_scheme == "nonlinear":
            largest += float(np.max(temporal))
        limit = 2 * self._stability_radius(stepper) / largest
        schedule = stepper.schedule
        longest = stepper.dz_min if schedule is None else float(np.max(schedule.steps))
        if longest > limit:
            printed = _printed_down(limit)
            advice = (
                _history_replay_advice(printed, split_step=False, integrator=True)
                if schedule is not None
                else ""
            )
            msg = (
                f"the fixed step or step floor {longest:.3e} m exceeds the spatial "
                f"collar stability limit {printed} m (effective power rate at most "
                f"{largest:.3e} 1/m). Take shorter steps, "
                f"{self._runge_kutta_advice(stepper)}or widen the collar (a larger "
                "collar_fraction designs a weaker one); the spatial absorber is "
                f"always in-stage and cannot use split_step.{advice}"
            )
            raise TranslationError(msg)

    def _check_collar_crossing(self, stepper: StepperSettings) -> None:
        """Refuse a fixed schedule a requested collar cannot be accurate under.

        SEP 40 section 4.7: a fixed in-stage schedule is checked at the crossing
        limit's ``c_x = 0.15`` (:data:`solprop.spatial_absorber.CROSSING_FIXED`),
        because it is bound first by damping accuracy, ``alpha0 h / 2 <= 1.3``
        (:data:`solprop.spatial_absorber.DAMPING_BOUND`). An adaptive run is held to
        the crossing limit through ``dz_max`` instead (:func:`assemble`).
        """
        route = self.route
        design = route.transverse if isinstance(route, SpatialRoute) else None
        schedule = stepper.schedule
        if design is None or not design.has_collar or schedule is None:
            return
        longest = float(np.max(schedule.steps))
        crossing = design.crossing_step * CROSSING_FIXED / CROSSING_ADAPTIVE
        damping = 2.0 * DAMPING_BOUND / design.alpha0
        bound, name = min(
            (crossing, "the collar-crossing limit"), (damping, "the damping bound")
        )
        if longest > bound:
            printed = _printed_down(bound)
            advice = _history_replay_advice(printed, split_step=False, integrator=False)
            msg = (
                f"the fixed schedule's longest step {longest:.3e} m exceeds {name} of "
                f"the requested collar, {printed} m (SEP 40 section 4.7): a step "
                "that long could carry the protected band across the collar unseen or "
                "damp it inaccurately. Take shorter steps, or an adaptive run, which "
                f"the crossing limit holds.{advice}"
            )
            raise TranslationError(msg)

    def _check_carrier(self) -> None:
        """Check the harmonic carrier against the Kerr term that needs it."""
        wanted = self.responses.harmonic_carrier
        if (self.grids.carrier_2w0 is not None) != wanted:
            msg = (
                "grids.carrier_2w0 must be present exactly when a third-harmonic Kerr "
                "term, constant or dispersive, is compiled; the term is "
                f"{'present' if wanted else 'absent'} "
                f"and the carrier is "
                f"{'present' if self.grids.carrier_2w0 is not None else 'absent'}"
            )
            raise TranslationError(msg)

    def _check_plasma(self) -> None:
        """Validate every channel and its identity with the reported rate tuple."""
        responses = self.responses
        if bool(responses.plasma_channels):
            plasma_source_record(self.grids)
        channels = responses.population_channels
        if not channels:
            return
        expected_grid = "envelope" if responses.envelope_plasma is not None else "field"
        if self.grid_kind != expected_grid:
            raise TranslationError(
                f"the compiled plasma requires a {expected_grid!r} grid, "
                f"got {self.grid_kind!r}"
            )
        if len(responses.ionisation_table) != len(channels):
            raise TranslationError(
                "ionisation_table must contain one rate per plasma channel"
            )
        seen = set()
        for channel, table in zip(channels, responses.ionisation_table, strict=True):
            if channel.label in seen or channel.label not in responses.labels:
                raise TranslationError(
                    f"invalid or duplicate plasma channel label {channel.label!r}"
                )
            seen.add(channel.label)
            if channel.rate is not table:
                raise TranslationError(
                    "ionisation_table is not the rate table plasma channel "
                    f"{channel.label!r} carries"
                )
            if float(table(0.0)) != 0.0:
                raise TranslationError(
                    f"plasma channel {channel.label!r}: rate must be zero at zero field"
                )
            _check_scalar_domain(channel.density, f"density:{channel.label}")
            _check_scalar_domain(
                channel.preionisation, f"preionisation:{channel.label}", upper=1.0
            )

    def _check_envelope_factors(self) -> None:
        """Bind source multipliers to the coarse absolute-frequency axis."""
        assembly = self.responses.envelope_plasma
        if assembly is None:
            return
        factors = assembly.factors
        active = self.grids.active_mask & (self.grids.omega > 0)
        if factors.omega0 != self.grids.omega0 or not np.array_equal(
            factors.active, active
        ):
            raise TranslationError("envelope source factors disagree with the grid")
        inverse = np.zeros_like(self.grids.omega)
        inverse[active] = 1 / (
            self.grids.omega[active]
            if factors.carrier_policy == "absolute-frequency"
            else self.grids.omega0
        )
        if not np.array_equal(
            factors.drude, -(inverse * inverse)
        ) or not np.array_equal(factors.loss, -1j * inverse):
            raise TranslationError("envelope source factors have incorrect multipliers")

    def _check_raman(self) -> None:
        """Check every Raman component's shape, amplitude and compatible drive."""
        terms = self.responses.raman_terms
        if not terms:
            return
        if self.transform.n_pol * self.shape[1] > 1 and not all(
            term.tensor_capable for term in terms
        ):
            raise TranslationError(
                "vector Raman requires tensor-capable terms; found scalar_only"
            )
        expected = 2 * self.grids.n_to
        seen = set()
        for compiled in terms:
            if compiled.label not in self.responses.labels or compiled.label in seen:
                raise TranslationError(
                    f"invalid or duplicate Raman component label {compiled.label!r}"
                )
            seen.add(compiled.label)
            if compiled.undamped.shape[1] != expected:
                raise TranslationError(
                    f"raman_terms[{compiled.label!r}].undamped has shape "
                    f"{compiled.undamped.shape}, "
                    f"expected (n_terms, {expected}) on the doubled fine grid"
                )
            if compiled.density is None:
                raise TranslationError(
                    f"Raman component {compiled.label!r} needs a density table"
                )
            _check_scalar_domain(compiled.density, f"Raman density:{compiled.label}")
            if self.grid_kind == "field":
                electronic = self.responses.electronic_thg
                if electronic is None or compiled.thg != electronic:
                    raise TranslationError(
                        f"Raman component {compiled.label!r} and Kerr must agree on thg"
                    )
        if len({term.thg for term in terms}) != 1:
            raise TranslationError("Raman component terms must agree on thg")

    def _check_dispersive(self) -> None:
        """Check the dispersive electronic leaf against the grid and the species.

        On the general modal route each channel names a node region of the guide
        rather than a population of the run's material.
        """
        leaf = self.responses.dispersive_kerr
        if leaf is None:
            return
        n_omega_fine = n_omega_for(self.grids.n_to, self.grid_kind)
        if leaf.positive.shape != (n_omega_fine,):
            msg = (
                f"responses.dispersive_kerr is sampled on {leaf.positive.shape[0]} "
                f"bins and the fine {self.grid_kind} grid has {n_omega_fine}"
            )
            raise TranslationError(msg)
        tables = self.transform.modal_tables
        if (leaf.regions is None) != (tables is None):
            raise TranslationError(
                "responses.dispersive_kerr.regions is present exactly on the general "
                "modal route, whose channels act in the guide's node regions"
            )
        if tables is not None and leaf.regions is not None:
            declared = int(np.max(tables.region_index)) + 1
            if int(np.max(leaf.regions)) >= declared:
                msg = (
                    "responses.dispersive_kerr names region "
                    f"{int(np.max(leaf.regions))} and the node rule declares "
                    f"{declared}"
                )
                raise TranslationError(msg)
        else:
            unknown = sorted(set(leaf.labels) - set(self.responses.labels))
            if unknown:
                msg = f"responses.dispersive_kerr names unknown populations {unknown}"
                raise TranslationError(msg)
        wanted = self.grid_kind == "envelope" and leaf.sector == "full"
        if leaf.carrier != wanted:
            raise TranslationError(
                "responses.dispersive_kerr.carrier must be true exactly for the full "
                "sector on an envelope grid"
            )
        for label, table in zip(leaf.labels, leaf.densities, strict=True):
            _check_scalar_domain(table, f"dispersive Kerr density:{label}")

    def _check_species(self) -> None:
        """Validate the declared species order and all population density tables."""
        responses = self.responses
        labels = responses.labels
        if (
            not labels
            or len(set(labels)) != len(labels)
            or any(not label for label in labels)
        ):
            raise TranslationError("species labels must be nonempty and unique")
        if len(labels) != len(responses.species_names) or len(labels) != len(
            responses.species_densities
        ):
            raise TranslationError(
                "species labels, names and densities must have matching lengths"
            )
        for label, table in zip(labels, responses.species_densities, strict=True):
            _check_scalar_domain(table, f"species density:{label}")
        _check_scalar_domain(responses.density, "total density")
        if (
            responses.kerr_coefficients is not None
            and responses.kerr_coefficients.density is None
        ):
            raise TranslationError("Kerr coefficients need their own density table")

    def _check_modal_leaves(self, run: _RunPolicy | None) -> None:
        """Check the four schema-12 general-guide leaves against the state.

        The contract's tree walk *does* recurse into these records: every array they
        hold is checked for dtype, finiteness, contiguity and ownership like any
        other leaf. What the walk does not do is relate any of them to the canonical
        state, and that is this method's work -- each leaf's type, the channel and
        component counts, the frequency axes and the positions along z.

        - ``run``: the run's stepper section and statistics; ``None`` leaves the run's
          extent along z to :func:`validate_run`.
        """
        n_modes, n_pol, _ = self.shape
        tables = self.transform.modal_tables
        metric = self.linear.modal_metric
        flow = self.linear.coupled_flow
        regional = self.responses.regional_responses
        check_leaf_types(self)
        if tables is not None:
            if self.transform.geometry is not None:
                msg = (
                    "transform.modal_tables is a modal E/H reference and "
                    "transform.geometry is spatial; a problem carries one or the "
                    "other"
                )
                raise TranslationError(msg)
            if n_pol != 1:
                msg = (
                    f"transform.modal_tables carries {MAX_N_COMPONENTS} physical "
                    f"components per channel and the state has n_pol = {n_pol}; a "
                    "general modal guide puts every physical channel, including a "
                    "selected parity partner, in n_modes and keeps n_pol = 1"
                )
                raise TranslationError(msg)
            channels = int(tables.electric.shape[3])
            if channels != n_modes:
                msg = (
                    f"transform.modal_tables describes {channels} channel(s) and the "
                    f"state carries {n_modes}"
                )
                raise TranslationError(msg)
        if metric is not None and metric.n_modes not in (0, n_modes):
            msg = (
                f"linear.modal_metric describes {metric.n_modes} channel(s) and the "
                f"state carries {n_modes}"
            )
            raise TranslationError(msg)
        if metric is not None:  # its rows, before the route's policy reads them
            metric.check_rows()
        if flow is not None and flow.n_modes != n_modes:
            msg = (
                f"linear.coupled_flow couples {flow.n_modes} channel(s) and the state "
                f"carries {n_modes}"
            )
            raise TranslationError(msg)
        if regional is not None:
            if tables is None:
                msg = (
                    "responses.regional_responses indexes the node regions of "
                    "transform.modal_tables, which is absent"
                )
                raise TranslationError(msg)
            regions = int(np.max(tables.region_index)) + 1
            if int(np.max(regional.response_region)) >= regions:
                msg = (
                    "responses.regional_responses names region "
                    f"{int(np.max(regional.response_region))} and the node rule "
                    f"declares {regions}"
                )
                raise TranslationError(msg)
            components = int(tables.electric.shape[2])
            if regional.n_components != components:
                msg = (
                    f"responses.regional_responses contracts over "
                    f"{regional.n_components} Cartesian component(s) and "
                    f"transform.modal_tables resolves {components}"
                )
                raise TranslationError(msg)
        self._check_modal_grids(run)

    def _check_modal_grids(self, run: _RunPolicy | None) -> None:
        """Check each general-guide leaf is tabulated on the canonical grids.

        The canonical state lives on the coarse grid, so a leaf's
        ``omega`` is ``grids.omega`` itself -- not a sub-band, not a resampling --
        with the bins outside the guide's active support explicitly zero rather than
        evaluated, and ``modal_tables.omega_fine`` is the problem's fine spectrum.
        The fine axis is not a contract row of its own, so it is pinned by the two
        facts the contract does carry: it is as long as the fine spectrum, and the
        coarse bins sit in it exactly where ``grids.crop_index`` says they do.

        Every leaf tabulated along z must also cover the run, as
        :meth:`_check_tables` requires of every other z table.

        - ``run``: the run's stepper section and statistics, whose last plane the leaves
          must reach; ``None`` checks the axes alone.
        """
        if self.transform.modal_tables is not None and self.grid_kind != "field":
            # Named first: every leaf below is tabulated on the increasing coarse axis
            # of a field grid, so on another grid its axis checks would misname this.
            msg = (
                "solprop executes a general modal guide (transform.modal_tables) on a "
                "field grid, whose canonical axis is increasing, and this problem is "
                f"{self.grid_kind}"
            )
            raise TranslationError(msg)
        omega = self.grids.omega
        z_end = None if run is None else run.stepper.z_end
        for name, leaf in (
            ("transform.modal_tables", self.transform.modal_tables),
            ("linear.modal_metric", self.linear.modal_metric),
            ("linear.coupled_flow", self.linear.coupled_flow),
        ):
            if leaf is None:
                continue
            if not np.array_equal(leaf.omega, omega):
                msg = (
                    f"{name}.omega is not grids.omega: it carries "
                    f"{leaf.omega.size} frequencies and the canonical coarse grid "
                    f"has {omega.size}, and the values must agree bin for bin. A "
                    "general-modal leaf is tabulated on the state's own grid, zero "
                    "outside its active support; an envelope grid's FFT-ordered "
                    "axis is not increasing and cannot carry one"
                )
                raise TranslationError(msg)
            low, high = float(leaf.z_edges[0]), float(leaf.z_edges[-1])
            if z_end is not None and (low > 0.0 or high < z_end):
                msg = (
                    f"{name}.z_edges covers [{low}, {high}] m, which does not "
                    f"contain the run [0, {z_end}] m"
                )
                raise TranslationError(msg)
        tables = self.transform.modal_tables
        if tables is None:
            return
        grids = self.grids
        expected = grids.n_to // 2 + 1 if self.grid_kind == "field" else grids.n_to
        if tables.omega_fine.size != expected or not np.array_equal(
            tables.omega_fine[grids.crop_index], omega
        ):
            msg = (
                f"transform.modal_tables.omega_fine carries "
                f"{tables.omega_fine.size} frequencies and the problem's fine "
                f"spectrum has {expected}, with the coarse bins at grids.crop_index"
            )
            raise TranslationError(msg)

    def _check_tables(self, run: _RunPolicy | None) -> None:
        """Check the z tables cover the run, and the schedule contains its targets.

        - ``run``: the run's stepper section and statistics; ``None`` checks the tables
          against each other alone, and leaves the run's extent, its schedule, the
          collar's stability and the exponent budget to :func:`validate_run`.
        """
        edges = self.linear.phi_breakpoints
        if not np.all(np.diff(edges) > 0.0):
            msg = f"linear.phi_breakpoints must be strictly increasing, got {edges}"
            raise TranslationError(msg)
        low, high = self.linear.z_span
        z_end = None if run is None else run.stepper.z_end
        if z_end is not None and (low > 0.0 or high < z_end):
            msg = (
                f"the Phi table covers [{low}, {high}] m, which does not contain the "
                f"run [0, {z_end}] m"
            )
            raise TranslationError(msg)
        interior = edges[1:-1]
        for kink in self.linear.hard_kinks:
            if interior.size == 0 or np.min(np.abs(interior - kink)) > _EDGE_SLACK * (
                high - low
            ):
                msg = (
                    f"linear.hard_kinks: the kink at z = {float(kink)!r} m is not an "
                    "interval edge, so the table was built across it"
                )
                raise TranslationError(msg)
        for name, table in _walk_tables(self):
            span = table.z_span
            if abs(span[0] - low) > _SPAN_SLACK * (high - low) or abs(
                span[1] - high
            ) > _SPAN_SLACK * (high - low):
                msg = (
                    f"{name} covers {span} m while the Phi table covers "
                    f"{(low, high)} m; every table along z must span the same range"
                )
                raise TranslationError(msg)
        if run is None:
            return
        if run.stepper.schedule is not None:
            self._check_schedule(run.stepper)
        self._check_collar_stability(run.stepper)
        self._check_exponent_budget(run.stepper)

    def _check_schedule(self, stepper: StepperSettings) -> None:
        """Check a replayed schedule lands on every save plane and hard kink.

        A coupled flow makes the hard kinks stricter. Its sections are placed on
        them, and the section a stage is evaluated on is matched within
        :func:`solprop.translate.coupled.flow_edge_slack` -- a far tighter window
        than this check's own ``1e-9`` of the run. A schedule edge between the two
        is accepted here, anchors its step in the section that **closes** at the
        kink, and then runs past the corner against a frozen block none of its
        stages read: the residual one stage sees jumps by the whole connection and
        the run falls from fifth order to first with nothing said. Such a schedule
        is refused by name rather than integrated; adaptive
        stepping needs no such rule, because the driver lands on every hard kink
        itself.
        """
        schedule = stepper.schedule
        assert schedule is not None
        edges = schedule.edges
        scale = max(schedule.total_length, 1.0)
        flow = self.linear.coupled_flow
        flow_slack = None if flow is None else flow_edge_slack(flow)
        for name, targets in (
            ("save plane", stepper.save_planes),
            ("hard kink", self.linear.hard_kinks),
        ):
            for target in targets:
                offset = float(np.min(np.abs(edges - target)))
                if offset > _EDGE_SLACK * scale:
                    msg = (
                        f"stepper.schedule does not land on the {name} at z = "
                        f"{float(target)!r} m; a fixed schedule must contain every one "
                        "of them, which FixedSchedule.uniform guarantees"
                    )
                    raise TranslationError(msg)
                if (
                    name == "hard kink"
                    and flow_slack is not None
                    and offset > flow_slack
                ):
                    msg = (
                        f"stepper.schedule comes within {offset!r} m of the hard kink "
                        f"at z = {float(target)!r} m but does not land on it, and "
                        "linear.coupled_flow places a section edge there: a step that "
                        f"straddles it by more than {flow_slack!r} m is integrated "
                        "against a frozen block its own stages do not read, which "
                        "costs the run its order. Land the schedule on the kink "
                        "exactly, which FixedSchedule.uniform guarantees"
                    )
                    raise TranslationError(msg)


class Translation(NamedTuple):
    """What translation returns: the numerical problem and its run's request.

    Attributes
    ----------
    problem : PropagationProblem
        The numerical meaning (SEP 47 section 5.2).
    request : solprop.translate.run_request.RunRequest
        Every policy of a fresh run of it, its step limits resolved against the
        problem; :meth:`~solprop.translate.run_request.RunRequest.for_run` adapts it
        to one run.
    """

    problem: PropagationProblem
    request: "RunRequest"


def validate_run(problem: PropagationProblem, request: "RunRequest") -> None:
    """Check a problem and the request it is run with, before anything is allocated.

    Everything :meth:`PropagationProblem.validate` checks, in the same order, and
    with it what depends on the run's policy: the method's admission on the route,
    the save planes following the launch, the statistics the route admits, the
    request's own arrays, the propagator mode on a separable route, the tables and a
    general guide's leaves covering the run, a replayed schedule landing on every save
    plane and hard kink, the in-stage collar's stability, the split collar's crossing,
    the exponent budget and, last, dense observation's admission (SEP 14 section 5).
    A backend calls this where it is built (SEP 47 section 5.2), so an unsupported
    combination is refused by name before any workspace exists.

    Parameters
    ----------
    problem : PropagationProblem
        The translated problem.
    request : solprop.translate.run_request.RunRequest
        The run's request.

    Raises
    ------
    TranslationError
        Naming the field or setting that failed.

    Warns
    -----
    solprop.exceptions.ExponentBudgetWarning
        When a step the run cannot shrink comes within a tenth of the exponent limit.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.translate.problem import validate_run
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech",
    ... )
    >>> validate_run(simulation.translate(), simulation._run_request()) is None
    True
    """
    problem._validate(_RunPolicy(request.stepper, request.observation.stats))
    _admit_dense(problem, request.stepper)


def _admit_dense(problem: PropagationProblem, stepper: StepperSettings) -> None:
    """Refuse dense observation where its interpolant is not the run's field.

    SEP 14 section 5's admission matrix, for the Runge-Kutta extension and for SEP 13's
    Adams interpolant alike. The per-step temporal collar changes the state after the
    step, so an interpolated plane would be uncollared; a coupled flow's action and the
    ``modified`` noise's draw are outside the interpolant.
    """
    if not stepper.interpolate_planes:
        return
    responses = problem.responses
    reasons = []
    if (
        responses.temporal_absorber is not None
        and responses.absorber_scheme == "split_step"
    ):
        reasons.append("the per-step temporal collar (absorber_scheme='split_step')")
    if problem.linear.coupled_flow is not None:
        reasons.append("a coupled linear flow")
    if problem.initial.noise_model == "modified":
        reasons.append("the 'modified' noise model")
    if reasons:
        msg = (
            "dense observation interpolates inside an accepted step and is refused "
            f"beside {', '.join(reasons)}: land on the save planes instead"
        )
        raise TranslationError(msg)


_EDGE_SLACK = 1e-9


_SPAN_SLACK = 1e-12


LOG_DBL_MAX = math.log(np.finfo(np.float64).max)
"""``log(DBL_MAX)``, about 709.78: the largest exponent ``exp`` returns finite.

IEEE Std 754-2019, doi:10.1109/IEEESTD.2019.8766229. A propagator factor
``exp(h lambda)`` past it is infinite on every backend (SEP 22 section 4.5).
"""


EXPONENT_BUDGET = 0.9
"""Fraction of :data:`LOG_DBL_MAX` above which translation warns of a step it cannot
shrink: such a run completes only while its field stays below ``exp`` of the margin.
"""


def _budget_term(
    bounds: RateBounds, span: int, attenuation: str
) -> tuple[str, float, float | None]:
    """Return the exponent budget's binding term: its name, value and what remains.

    - Returns: the term's description, its rate in m^-1, and the rest of the loss bound
      beside it (``None`` for a term no setting moves).
    """
    if span * bounds.gain > bounds.loss:
        return f"the medium's gain over {span} step(s)", span * bounds.gain, None
    terms = (
        ("evanescent_rate", bounds.evanescent, 1.0),
        ("safety ramp strength", bounds.transverse, 2.0),
        ("spectral absorber rate", bounds.spectral, 2.0),
        (attenuation, bounds.propagating, 0.0),
    )
    name, value, scale = max(terms, key=lambda term: term[1])
    if scale == 0.0:
        return name, value, None
    return name, scale * value, bounds.loss - value


def _budget_message(  # noqa: PLR0917 - called positionally
    source: str,
    step: float,
    bounds: RateBounds,
    span: int,
    exponent: float,
    attenuation: str,
) -> str:
    """Describe the factor a step forms, naming the binding term.

    - ``step``: the step, m.
    """
    name, value, _ = _budget_term(bounds, span, attenuation)
    return (
        f"{source} {step:.3g} m with {name} {value:.3g} 1/m forms a propagator "
        f"factor exp({exponent:.0f})"
    )


def _budget_remedy(
    step: float, bounds: RateBounds, span: int, limit: float, attenuation: str
) -> str:
    """Name the admissible step, and the admissible rate when a setting binds.

    - ``step``: the step, m.
    """
    name, value, rest = _budget_term(bounds, span, attenuation)
    rate = max(bounds.loss, span * bounds.gain)
    remedy = f"take steps of at most {limit / rate:.3g} m"
    if rest is not None:
        scale = 1.0 if name == "evanescent_rate" else 2.0
        admissible = scale * (limit / step - rest)
        if name == "safety ramp strength":
            # The ramp is designed, not set: its levers are the specification's.
            remedy += (
                " or a weaker safety ramp (a smaller SpatialAbsorber attenuation or a "
                "wider safety_fraction; spatial_absorber=False removes it)"
            )
        elif 0 < admissible < value:
            remedy += f" or {name} at most {admissible:.3g} 1/m"
    return remedy


def _check_scalar_domain(
    table: PiecewiseChebyshev, name: str, *, upper: float = math.inf
) -> None:
    """Check a scalar polynomial at endpoints and every real stationary point.

    ``upper`` is inclusive, in the table's value units.
    """
    if table.value_shape != ():
        raise TranslationError(f"{name} must be a scalar table")
    for row, degree in zip(table.coefficients, table.degrees, strict=True):
        coefficients = row[: int(degree) + 1]
        roots = np.polynomial.chebyshev.chebroots(
            np.polynomial.chebyshev.chebder(coefficients)
        )
        interior = np.real(roots[np.isreal(roots) & (np.abs(roots) < 1)])
        values = np.polynomial.chebyshev.chebval(
            np.r_[-1.0, interior, 1.0], coefficients
        )
        if np.any(values < 0.0) or np.any(values > upper):
            raise TranslationError(f"{name} must lie in [0, {upper}]")


def _walk_arrays(
    problem: PropagationProblem, run: _RunPolicy | None = None
) -> list[tuple[str, np.ndarray]]:
    """Collect every NumPy array in the problem tree, with its dotted name.

    - Returns: ``(name, array)`` pairs, in field order.
    """
    found: list[tuple[str, np.ndarray]] = []

    def visit(prefix: str, node: object) -> None:
        """Add every array reachable from one node.

        - ``node``: a dataclass, an array, or anything else.
        """
        if isinstance(node, np.ndarray):
            found.append((prefix, node))
        elif isinstance(node, PiecewiseChebyshev):
            for name in ("coefficients", "breakpoints", "degrees", "hard_kinks"):
                found.append((f"{prefix}.{name}", getattr(node, name)))
        elif isinstance(node, Mapping):
            for key, value in node.items():
                visit(f"{prefix}[{key!r}]", value)
        elif isinstance(node, tuple | list):
            for index, value in enumerate(node):
                visit(f"{prefix}[{index}]", value)
        elif is_dataclass(node) and not isinstance(node, type):
            for entry in fields(node):
                if node is problem and entry.name == "route":
                    if run is not None:
                        visit("stepper", run.stepper)
                        visit("stats_spec", run.stats)
                    # The route's leaves keep the names they have always had:
                    # ``transform.synthesis``, ``flux.base.kperp_sq``.
                    visit(prefix, getattr(node, entry.name))
                    continue
                name = f"{prefix}.{entry.name}" if prefix else entry.name
                visit(name, getattr(node, entry.name))

    visit("", problem)
    return found


def _walk_tables(problem: PropagationProblem) -> list[tuple[str, PiecewiseChebyshev]]:
    """Collect every Chebyshev table along z, with its dotted name.

    - Returns: ``(name, table)`` pairs.
    """
    candidates = [
        ("linear.beta0", problem.linear.beta0),
        ("linear.power_weight", problem.linear.power_weight),
        (
            "linear.power_weight_expansion.weights",
            None
            if problem.linear.power_weight_expansion is None
            else problem.linear.power_weight_expansion.weights,
        ),
        ("linear.beta1", problem.linear.beta1),
        ("transform.taper_scale", problem.transform.taper_scale),
        ("transform.aeff", problem.transform.aeff),
        ("responses.density", problem.responses.density),
        ("responses.prefactor", problem.responses.prefactor),
        (
            "responses.prefactor.weights",
            None
            if problem.responses.prefactor_expansion is None
            else problem.responses.prefactor_expansion.weights,
        ),
        (
            "linear.phi_expansion.weights",
            None
            if problem.linear.phi_expansion is None
            else problem.linear.phi_expansion.weights,
        ),
    ]
    responses = problem.responses
    if responses.kerr_coefficients is not None:
        candidates.append(
            ("responses.kerr_coefficients.density", responses.kerr_coefficients.density)
        )
    if responses.dispersive_kerr is not None:
        candidates.extend(
            (f"responses.dispersive_kerr.densities[{i}]", table)
            for i, table in enumerate(responses.dispersive_kerr.densities)
        )
    candidates.extend(
        (f"responses.species_densities[{i}]", table)
        for i, table in enumerate(responses.species_densities)
    )
    for i, channel in enumerate(responses.population_channels):
        candidates.extend(
            (
                (f"responses.plasma_channels[{i}].density", channel.density),
                (
                    f"responses.plasma_channels[{i}].preionisation",
                    channel.preionisation,
                ),
            )
        )
    for i, term in enumerate(responses.raman_terms):
        candidates.append((f"responses.raman_terms[{i}].density", term.density))
        candidates.extend(
            (f"responses.raman_terms[{i}].decay_rates[{j}]", table)
            for j, table in enumerate(term.decay_rates)
        )
    return [(name, table) for name, table in candidates if table is not None]


def _state(
    field0: Complex[np.ndarray, "*shape"],
    n_omega: int,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Bring an initial field to the canonical single-trajectory layout.

    The shape is checked here rather than by an annotation, so that a wrong rank is
    reported by name in production too. A group of trajectories is a sequence of
    problems (SEP 47 section 6.7), never a stacked field.

    - Returns: the state, complex128, the problem's own: copied once here as
      :class:`Initial` holds it, so that it is not copied again there, and taken as it
      is when the problem already holds it.
    """
    values = np.asarray(field0)
    if values.ndim != 3:
        msg = f"field0 must have shape (n_modes, n_pol, n_omega), got {values.shape}"
        raise TranslationError(msg)
    if values.shape[-1] != n_omega:
        raise TranslationError(
            f"field0 has {values.shape[-1]} spectral bins, expected {n_omega} "
            "to match the grid"
        )
    return _held_state(values)


@dataclass(frozen=True, eq=False)
class TranslationReuse:
    """Carry prequalified reusable data into a material-path update.

    This host-side construction record is outside the kernel contract. Its caller
    must establish unchanged geometry, grid and operator functions before supplying
    their data; :meth:`solprop.driver.simulation.Simulation.with_mixture` performs
    those checks. New parameter paths are checked against each expansion's ranges.

    Attributes
    ----------
    transform : Transform
        Unchanged transverse geometry, quadrature and grid transform.
    initial : Initial
        Unchanged initial field, noise and batch parameters.
    phi_expansion : ParameterExpansion or None
        Cumulative-operator coefficients, only if its sampled function is unchanged.
    prefactor_expansion : ParameterExpansion or None
        Prefactor coefficients, under the same function-identity requirement.
    raman_undamped : Mapping
        Component label to undamped rows, for unchanged Raman models and time grid.
    transverse : SpatialAbsorberDesign or None
        The transverse boundary the reused transform and expansions were built with,
        kept as they are (SEP 40).
    """

    transform: Transform
    initial: Initial
    phi_expansion: ParameterExpansion | None
    prefactor_expansion: ParameterExpansion | None
    raman_undamped: Mapping[str, Float[np.ndarray, "... n_to_doubled"]]
    transverse: SpatialAbsorberDesign | None = None


def _printed_down(value: float) -> str:
    """Return a positive limit in the ``.3e`` form, rounded down rather than to nearest.

    A refusal prints the step a caller has to stay under, so the printed number must
    not exceed the limit itself: ``5.2386e-03`` prints as ``5.238e-03``, where rounding
    to nearest would give ``5.239e-03``, a step the check still refuses. The decimal
    expansion of the double is exact, so the floor is too.

    - ``value``: the limit, positive and finite.
    """
    exact = Decimal(value)
    exponent = exact.adjusted()
    mantissa = exact.scaleb(-exponent).quantize(Decimal("0.001"), rounding=ROUND_FLOOR)
    return f"{mantissa}e{exponent:+03d}"


def _history_replay_advice(printed: str, *, split_step: bool, integrator: bool) -> str:
    """Return the remedies for replaying an adaptive run's own step history.

    An adaptive run is not held to a fixed schedule's limits: an empty collar produces
    no error, and a requested spatial collar holds an adaptive step only to its
    crossing limit, so its accepted steps can be longer than a replay may take, and
    a schedule read from them
    (:meth:`solprop.stepper.schedule.FixedSchedule.from_history`) is then refused.
    Capping the discovery run with ``dz_max`` under the limit always replays; the
    split temporal collar lifts the temporal collar's limit alone.

    - ``printed``: the limit as the refusal prints it, rounded down, m.
    """
    split = (
        'either give absorber_scheme="split_step" to both the discovery run and '
        "the replay, or "
        if split_step
        else ""
    )
    method = (
        " The limit is the replay's integrator's, so run the discovery with that "
        "integrator."
        if integrator
        else ""
    )
    return (
        " To replay an adaptive run's own steps (FixedSchedule.from_history), "
        f"{split}hold the discovery run below this limit with dz_max at or under "
        f"{printed} m and replay its history.{method}"
    )


def debug_summary(problem: PropagationProblem, stats: StatsSpec | None = None) -> str:
    """Return a short array-free description of an assembled problem, for a log.

    The translation summary ``AGENTS.md`` asks the translation layer to log: what
    was built, in which representation, and with which certificates. It reads the
    assembled record and nothing else, so it can be called on a problem from any
    source, and it is not a kernel-contract field -- adding a row here is not a
    seam change.

    For a **general modal guide** it carries the certificates that route rests on
    and that nothing downstream reads again: the sampled Gram defect the identity
    metric was admitted with, the declared profile error of the frequency
    interpolation, the gauge, and the guide's own support in wavelength.

    Parameters
    ----------
    problem : PropagationProblem
        A problem, assembled and validated.
    stats : solprop.stats.StatsSpec or None, optional
        The statistics its run collects, listed when given; they are the run's
        policy, not the problem's.

    Returns
    -------
    str
        Several lines of ``key: value``, with no array in them.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.translate.problem import debug_summary
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech", n_save=3,
    ... )
    >>> print(debug_summary(simulation.translate()).splitlines()[0])
    kernel problem: envelope grid, route modal
    """
    linear, transform = problem.linear, problem.transform
    tables, metric = transform.modal_tables, linear.modal_metric
    rows = [
        f"kernel problem: {problem.grid_kind} grid, route {route_name(problem.route)}",
        f"  shape:         {problem.shape} (modes, pol, omega)",
        f"  flow:          phi {linear.phi_representation}, "
        f"{linear.n_intervals} interval(s), {linear.nbytes / 2**20:.3f} MB",
        f"  power weight:  {linear.power_weight_representation}",
        "  responses:     order "
        f"{[str(one.value) for one in problem.responses.response_order]}",
        f"  noise:         {problem.initial.noise_model}",
    ]
    if stats is not None:
        rows.append(f"  statistics:    {', '.join(stats.names) or 'none'}")
    if metric is not None:
        rows.append(
            f"  modal metric:  {metric.kind}, defect "
            f"{float(np.max(metric.error)):.3e}, certificate "
            f"{metric.certificate_id}"
        )
    if tables is not None:
        support = np.flatnonzero(np.any(np.all(tables.valid, axis=1), axis=0))
        low = float(tables.omega_fine[support[0]])
        high = float(tables.omega_fine[support[-1]])
        rows.extend(
            [
                f"  modal tables:  {int(tables.electric.shape[3])} channel(s) on "
                f"{int(tables.electric.shape[1])} nodes, "
                f"{tables.omega.size} coarse and {tables.omega_fine.size} fine bins",
                f"  guide support: {2 * np.pi * C / high * 1e9:.1f} to "
                f"{2 * np.pi * C / low * 1e9:.1f} nm, frozen rule "
                f"{tables.frozen_rule}",
                f"  profile error: {float(np.max(tables.profile_error)):.3e}",
                f"  gauge:         {tables.gauge_id}, policy {tables.policy_id}",
                # The physical model identity is the provider's whole serialised
                # state and runs to hundreds of characters; a summary prints what
                # distinguishes two of them, which is exactly a digest of it.
                "  model digest:  "
                f"{hashlib.sha256(tables.model_id.encode()).hexdigest()[:16]}",
            ]
        )
    regional = problem.responses.regional_responses
    if regional is not None:
        rows.append(
            f"  region source: {regional.n_responses} region(s), "
            f"{regional.model_id}, {regional.n_components} components"
        )
    return "\n".join(rows)
