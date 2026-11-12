# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What translation refuses, and what it warns about, before a problem is returned.

The policy :func:`solprop.translate.assembly.assemble` applies to its arguments and to
the problem it assembled: the refusals of a route's settings (a wall off the capillary
route, operator settings without a spatial grid, a save plane on an index jump, a frame
that removes ``beta_0`` beside the third harmonic, a TDSE response off its route, the
general modal route's legacy knobs), the split collar's fixed-schedule limit, dense
observation where its interpolant is not the run's field, and the coupled flow's
advisory step (:func:`validate_coupled_flow`). The kernel contract's own checks are
:meth:`solprop.translate.problem.PropagationProblem.validate` and
:func:`solprop.translate.problem.validate_run`.
"""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Bool, Complex, Float

from solprop.adiabatic import AdiabaticAccuracy
from solprop.boundaries import (
    Absorber,
    SplitCrossing,
)
from solprop.exceptions import (
    SolpropWarning,
    TranslationError,
)
from solprop.grid import FreeGrid, RadialGrid, TimeGrid
from solprop.linear import Frame, LossPolicy
from solprop.modes.capillary import CoherentWall
from solprop.modes.quadrature import TransverseQuadrature
from solprop.modes.slab import PlanarGrid
from solprop.profiles import ZProfile
from solprop.spatial import SpatialMedium
from solprop.spatial_absorber import SpatialAbsorber
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.coupled import (
    CoupledFlowTables,
    advisory_step,
    residual_truncation,
)
from solprop.translate.general import (
    GeneralModalRoute,
    check_noise_basis,
)
from solprop.translate.masks import AbsorberScheme
from solprop.translate.problem import (
    NoiseModel,
    PropagationProblem,
    StepperSettings,
    TranslationReuse,
    validate_run,
)
from solprop.translate.responses import CompiledResponses
from solprop.translate.tdse import TDSEDescriptor

if TYPE_CHECKING:
    from solprop.translate.run_request import RunRequest

__all__ = [
    "DEFAULT_FLOW_CONDITION_LIMIT",
    "MOVING_FLOW_CERTIFICATE",
    "TAPER_FLOW_CERTIFICATE",
    "validate_coupled_flow",
]


_INPUT_NOISE_ADMITTED = (
    "The 'input' model is admitted: its draw is part of the entrance state, in the "
    "channel basis, which is orthonormal in true power on every coupled-flow route, "
    "and the flow propagates it like any field"
)
"""The clause every coupled-flow refusal of the modified noise model ends with.

SEP 37 section 6. The input draw is one photon per channel and bin only in a basis
orthonormal in true power. The capillary's ``HE_1m`` set (and the vector wall's
``TE_0m``/``TM_0m`` families) are real Bessel profiles, orthogonal on the core and
each normalised to its own power by
:meth:`solprop.modes.capillary.MarcatiliMode.normalisation`. A general modal guide's
basis is gated by :func:`solprop.translate.general.check_noise_basis` before any flow
is built, at the entrance of a moving guide as everywhere else.
"""


@dataclass(frozen=True)
class _Assembly:
    """The arguments of :func:`assemble` that choosing a route needs, in one record.

    Passed to the path helpers below so that each takes the run's objects and this
    one bag rather than a dozen keywords; the record is private and holds exactly
    the caller's own values.
    """

    quadrature: TransverseQuadrature | None
    components: Sequence[str] | None
    spatial_grid: RadialGrid | FreeGrid | PlanarGrid | None
    medium: SpatialMedium | None
    density: ZProfile | float | None
    prefactor_index: float | None
    shock: bool
    responses: Sequence[object]
    n_pol: int
    z_max: float
    absorber: Absorber | None
    loss: LossPolicy
    margin: float
    tol: float
    has_noise: bool
    dz_max: float
    reuse: "TranslationReuse | None"
    kinks: tuple[tuple[float, str], ...] = ()
    taper_coupling: bool = True
    evanescent_rate: float | None = None
    spatial_absorber: SpatialAbsorber | None = None
    launch: Complex[np.ndarray, "n_modes n_pol n_omega"] | None = None
    adiabatic_accuracy: AdiabaticAccuracy | None = None
    noise_model: NoiseModel = "none"


def _check_operator_settings(
    *,
    spatial: bool,
    evanescent_rate: float | None,
    spatial_absorber: SpatialAbsorber | None,
) -> None:
    """Refuse the separable-operator settings where they cannot apply.

    - ``evanescent_rate``: the requested evanescent cap, m^-1.
    """
    if not spatial and (evanescent_rate is not None or spatial_absorber is not None):
        msg = (
            "evanescent_rate and spatial_absorber set the spatial operator, and "
            "this run has no spatial grid"
        )
        raise TranslationError(msg)
    if evanescent_rate is not None and (
        not math.isfinite(evanescent_rate) or evanescent_rate <= 0
    ):
        msg = f"evanescent_rate must be positive and finite, got {evanescent_rate!r}"
        raise TranslationError(msg)


def _refuse_planes_at_jumps(
    medium: SpatialMedium | None, planes: Float[np.ndarray, " n_z"]
) -> None:
    """Refuse a save plane on a declared index jump, whose side would be ambiguous.

    SEP 34 section 4.4: a plane at a jump measures one side's forward closure, and
    a request that does not say which is ambiguous.

    - ``medium``: the run's medium; None for a modal run, which has none.
    - ``planes``: the requested save positions in m.
    """
    if medium is None:
        return
    for name, path in medium.parameter_paths.items():
        jumps = np.asarray(path.jumps)
        hit = planes[np.isin(planes, jumps)]
        if hit.size:
            msg = (
                f"save plane z = {float(hit[0])!r} m lies on a declared jump of the "
                f"medium's {name} path, where its side would be ambiguous; move the "
                "plane off the jump"
            )
            raise TranslationError(msg)


def _refuse_split_crossing(
    crossing: SplitCrossing | None, schedule: FixedSchedule | None
) -> None:
    """Refuse a fixed schedule whose longest step exceeds the launch's crossing limit.

    - ``schedule``: the schedule to replay, or ``None`` for an adaptive run.
    """
    if crossing is None or schedule is None:
        return
    longest = float(np.max(schedule.steps))
    if longest > crossing.launch_limit:
        msg = (
            f"the fixed schedule's longest step {longest:.3e} m exceeds the split "
            f"temporal collar's crossing limit {crossing.launch_limit:.3e} m, "
            f"{crossing.factor:g} of the collar's {crossing.width:.3e} s width over "
            "the walk-off of the light the launch holds: a step that long could "
            "carry light through the collar and round the window undamped. Take "
            "shorter steps, an adaptive run, which the limit holds, or "
            'absorber_scheme="nonlinear"'
        )
        raise TranslationError(msg)


DEFAULT_FLOW_CONDITION_LIMIT: float = 1.0e4
"""Largest inverse conditioning a general modal coupled flow is admitted at.

Dimensionless, and the same number
:attr:`solprop.modes.capillary.CoherentWall.condition_limit` defaults to, because it
bounds the same thing: how far the interaction picture's inverse action may amplify
the nonlinear source over one step. A general guide's block is far below it -- a
unitary rotation of a lossless split block has an exactly Hermitian-free part and
reads 1.0 -- so this is a guard and not a tuning knob, and there is no declaration
object on this route to put it on.
"""


MOVING_FLOW_CERTIFICATE: str = "general-moving-frozen-residual"
"""The ``certificate_id`` a moving solid guide's coupled flow carries."""


TAPER_FLOW_CERTIFICATE: str = "taper-frozen-residual"
"""The ``certificate_id`` of a multimode capillary taper's coupling without a wall."""


def _validated(problem: "PropagationProblem", request: "RunRequest") -> None:
    """Validate an assembled problem and run every diagnostic that only warns."""
    validate_run(problem, request)
    validate_coupled_flow(problem, request.stepper)
    _admit_dense(problem, request.stepper)


def validate_coupled_flow(
    problem: "PropagationProblem",
    stepper: StepperSettings,
) -> None:
    """Warn when a run's declared steps exceed its coupled flow's advisory limit.

    A ``validate`` diagnostic in the manner of the in-stage absorber's
    ``stable_step`` warning: it names the term, the
    section, the frequency, the residual norm ``K``, the conditioning factor and the
    limit, and it **does not change the caller's tolerances or schedule**. The limit is
    ``0.9/(K ||U_s|| ||U_s^-1||)`` (:func:`solprop.translate.coupled.advisory_step`),
    and the warning says plainly that it is advisory: a wall *difference* can have
    positive real eigenvalues -- local growth relative to the frozen attenuating frame
    -- and no positive step makes that residual alone contractive, so neither its
    spectral radius nor a test of the stability polynomial is a complete rule.

    The three steps a run can declare are checked: ``dz_init``, ``dz_max`` and, when a
    fixed schedule is replayed, its own largest step, which the controller can neither
    shorten nor reject.

    Parameters
    ----------
    problem : PropagationProblem
        The assembled contract. A problem carrying no coupled flow, or one whose
        residual is identically zero, has nothing to warn about and is returned from
        in silence.
    stepper : StepperSettings
        The run's stepper section, whose declared steps are checked.

    Warns
    -----
    solprop.exceptions.SolpropWarning
        Naming the step that exceeds the limit, with the term, the section, the
        frequency, ``K``, the conditioning factor and the limit.

    Examples
    --------
    >>> from solprop.translate.validation import validate_coupled_flow
    >>> validate_coupled_flow.__name__
    'validate_coupled_flow'
    """
    flow = problem.linear.coupled_flow
    if flow is None:
        return
    # Over the bins the explicit residual is applied on: the requested band.
    _warn_advisory_step(flow, stepper, problem.grids.inband_mask)


def _declared_steps(settings: StepperSettings) -> dict[str, float]:
    """Return the steps a run declares, by name: the two limits and a replay's largest.

    - Returns: ``dz_init``, ``dz_max`` and, for a fixed schedule, its largest step, in
      m.
    """
    declared = {
        "dz_init": float(settings.dz_init),
        "dz_max": float(settings.dz_max),
    }
    if settings.schedule is not None:
        declared["the fixed schedule's largest step"] = float(
            np.max(settings.schedule.steps)
        )
    return declared


def _warn_advisory_step(
    flow: CoupledFlowTables,
    settings: StepperSettings,
    scored: Bool[np.ndarray, " n_omega"],
) -> None:
    """Warn when a declared step exceeds the explicit residual's advisory limit."""
    limits = advisory_step(flow, scored)
    section, bin_index = np.unravel_index(int(np.argmin(limits)), limits.shape)
    smallest = float(limits[section, bin_index])
    declared = _declared_steps(settings)
    over = {name: value for name, value in declared.items() if value > smallest}
    if not over:
        return
    residual = float(flow.residual_norm_bound[section, bin_index])
    conditioning = float(flow.inverse_condition_bound[section, bin_index])
    named = ", ".join(f"{name} = {value:.4g} m" for name, value in over.items())
    defect = residual_truncation(max(over.values()) * residual * conditioning)
    owner, advice = {
        MOVING_FLOW_CERTIFICATE: (
            "the moving guide's",
            "cap the step, or divide the sections "
            "(ModalGuide.from_provider(sections=...)), deliberately",
        ),
        TAPER_FLOW_CERTIFICATE: (
            "the taper coupling's",
            "cap the step (dz_max, or the schedule) deliberately",
        ),
    }.get(
        flow.certificate_id,
        (
            "the coherent wall's",
            "cap the step, or divide the sections (CoherentWall.sections), "
            "deliberately",
        ),
    )
    warnings.warn(
        f"{owner} explicit residual is the limiting explicit term of this "
        f"run: its interval bound is K = {residual:.4g} 1/m on section {int(section)} "
        f"at bin {int(bin_index)}, {float(flow.omega[bin_index]):.6g} rad/s, and the "
        f"interaction picture's conditioning factor there is {conditioning:.4g}, so "
        f"the advisory step 0.9/(K cond) is {smallest:.4g} m and {named} exceeds it, "
        f"for a local frozen truncation defect of {defect:.4g} charged to the linear "
        "share of the budget. "
        "Advisory only; a growth certificate is required -- a residual with positive "
        "real eigenvalues is locally growing relative to the frozen frame and no "
        f"positive step makes it contractive. Nothing here changes your schedule: "
        f"{advice}",
        SolpropWarning,
        stacklevel=3,
    )


def _refuse_a_wall_off_the_capillary_route(
    wall: CoherentWall | None,
    *,
    general: bool,
    spatial: bool,
    taper_coupling: bool = True,
) -> None:
    """Refuse a coherent wall declared beside a route that cannot carry it.

    The **first** door of the route, before anything is assembled, because it is the
    one a wall silently fell through: :func:`assemble` reads ``wall`` only in its
    modal branch, so a general modal or spatial run carrying one used to translate
    with ``linear.coupled_flow`` absent, no ``modal.flow`` key on its record and no
    word said. The other two doors are the mode set's
    (:func:`solprop.modes.capillary.wall_operator`) and the backend's.
    """
    if wall is not None and not taper_coupling:
        msg = (
            "taper_coupling=False declines the taper connection, and a coherent wall "
            "block always carries it: its basis is attached to the local radius. "
            "Leave taper_coupling at its default with a wall, or drop the wall "
            "for the adiabatic route"
        )
        raise TranslationError(msg)
    if wall is None or not (general or spatial):
        return
    route = "the general modal route" if general else "a spatial geometry"
    msg = (
        f"a coherent wall block is the capillary route's and this run takes "
        f"{route}: a spatial run carries no mode set to condense the boundary "
        "coordinate onto, and a general modal guide assembles its own flow from its "
        "provider. Declare the wall on Simulation.from_capillary, or leave it None"
    )
    raise TranslationError(msg)


def _check_general(
    route: GeneralModalRoute,
    grid: TimeGrid,
    refusals: _Assembly,
) -> None:
    """Refuse every legacy knob and every grid a general modal route cannot take.

    - ``refusals``: the assembly's knobs; each legacy one is refused by name when it is
      set, and the declared responses must be empty, a general guide's belonging to its
      material regions.
    """
    named = (
        ("quadrature", refusals.quadrature),
        ("components", refusals.components),
        ("spatial_grid", refusals.spatial_grid),
        ("medium", refusals.medium),
        ("density", refusals.density),
        ("prefactor_index", refusals.prefactor_index),
    )
    for name, value in named:
        if value is not None:
            msg = (
                f"{name} belongs to the static transverse quadrature and the "
                "general modal route carries its nodes, its "
                "materials and its normalisation in transform.modal_tables and "
                "responses.regional_responses instead"
            )
            raise TranslationError(msg)
    if refusals.responses:
        msg = (
            f"{len(refusals.responses)} declared response(s) beside a general modal "
            "route: "
            "its responses are its material regions' and reach the seam through "
            "responses.regional_responses"
        )
        raise TranslationError(msg)
    if not refusals.shock:
        msg = (
            "shock=False beside a general modal route: the route's source carries "
            "the full -i omega of its own source_frequency_factor and has no "
            "frozen-frequency form"
        )
        raise TranslationError(msg)
    if grid.kind != "field":
        msg = (
            "a general modal route is a field-grid route: its leaves are tabulated "
            "on the canonical coarse grid, and an envelope grid's axis is in FFT "
            "order about omega_0 and is not the increasing axis a leaf declares"
        )
        raise TranslationError(msg)
    if refusals.n_pol != 1:
        msg = (
            f"a general modal route puts every physical channel in n_modes and "
            f"keeps n_pol = 1; the state carries {refusals.n_pol}"
        )
        raise TranslationError(msg)
    check_noise_basis(route.metric, has_noise=refusals.has_noise)
    support = np.all(route.tables.valid, axis=1)[:, grid.crop_index]
    outside = grid.active_mask[None, :] & ~support
    if np.any(outside):
        first = float(grid.omega[np.flatnonzero(np.any(outside, axis=0))[0]])
        msg = (
            f"the run's active band reaches {first:.6e} rad/s, where the guide's "
            "reference is not defined: outside its support every stored profile is "
            "an explicit zero, so the band must lie inside it"
        )
        raise TranslationError(msg)


def _check_tdse_assembly_route(
    descriptor: TDSEDescriptor | None,
    *,
    modal: bool,
    scheduled: bool,
    absorber_scheme: AbsorberScheme,
    density: ZProfile | float | None,
) -> None:
    """Refuse paths whose optical ownership is outside the atomic contract.

    - ``descriptor``: atomic declaration, or None for an ordinary response.
    - ``density``: requested density override in m^-3, refused for an atomic
      declaration.
    """
    if descriptor is None:
        return
    if not modal or scheduled or absorber_scheme != "nonlinear":
        raise TranslationError(
            "TDSE requires the uncoupled modal Adams route with in-stage collar"
        )
    if density is not None:
        raise TranslationError(
            "TDSE takes density from its helium fill; drop the density override"
        )


def _admit_dense(problem: "PropagationProblem", stepper: StepperSettings) -> None:
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


def _check_harmonic_frame(harmonic: bool, compiled: CompiledResponses) -> None:
    """Check the frame's harmonic rule against the compiled carrier.

    The frame is chosen before anything compiles, from
    :func:`solprop.translate.responses.declares_harmonic_carrier`; the operator it
    writes is right only if the compiled pass agrees on whether it carries the
    harmonic (:attr:`solprop.translate.responses.CompiledResponses.harmonic_carrier`).
    """
    if compiled.harmonic_carrier != harmonic:
        msg = (
            "internal inconsistency: the frame was chosen for "
            f"{'a' if harmonic else 'no'} third-harmonic carrier "
            "(declares_harmonic_carrier), but the compiled responses "
            f"{'do not carry' if harmonic else 'carry'} one, so the linear operator "
            "would be written in the wrong frame"
        )
        raise TranslationError(msg)


def _refuse_carrier_removing_frame(frame: Frame | None, harmonic: bool) -> None:
    """Refuse a frame that removes ``beta_0`` beside the third harmonic's carrier.

    Removing ``beta_0``, and measuring the delay from ``omega_0`` with it, takes a
    constant phase rate out of the envelope that the harmonic term's
    ``A**3 exp(2 i omega_0 t)`` does not share, so the harmonic would be driven with a
    spurious mismatch (:class:`solprop.linear.Frame`,
    :func:`solprop.linear.frame_origin`).

    - ``frame``: the frame the caller gave, or ``None`` for the co-moving one.
    """
    if frame is not None and frame.subtract_beta0 and harmonic:
        msg = (
            "a frame with subtract_beta0=True was given beside a third-harmonic term "
            "on an envelope grid: the harmonic's exp(2i omega0 t) carrier is tied to "
            "the whole carrier phase, so removing beta_0 would drive it with a "
            "spurious phase mismatch. Give "
            "Frame(omega0, subtract_beta0=False), or leave frame=None for "
            "Frame.comoving, which keeps beta_0 here"
        )
        raise TranslationError(msg)
