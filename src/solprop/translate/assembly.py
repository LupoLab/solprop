# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Assembly: the physics objects of a run turned into a validated kernel problem.

:func:`assemble` is the translation layer's entry point. It chooses the route -- the
mode-set route (with the coupled flows of :mod:`solprop.translate.walls`), the general
modal route or a spatial geometry -- builds its operator, transform and prefactor,
compiles the responses, resolves the step limits the collars set, and returns the
:class:`~solprop.translate.problem.PropagationProblem` with its run's request,
validated.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from jaxtyping import Complex, Float

from solprop._ownership import hold
from solprop._process import host_blas
from solprop.adiabatic import AdiabaticAccuracy, certify_radial_path
from solprop.boundaries import (
    Absorber,
    SplitCrossing,
)
from solprop.exceptions import TranslationError
from solprop.grid import FreeGrid, RadialGrid, TimeGrid
from solprop.linear import DEFAULT_LOSS_POLICY, Frame, LossPolicy
from solprop.modes.capillary import (
    CoherentWall,
    _shared_sampling,
    taper_coupling_block,
)
from solprop.modes.protocol import (
    ChannelLabel,
    CrossSection,
    Mode,
    PolarisationComponent,
)
from solprop.modes.quadrature import TransverseQuadrature
from solprop.modes.slab import PlanarGrid
from solprop.noise import NoiseRealisation
from solprop.physdata import Gas
from solprop.physdata.mixtures import Mixture
from solprop.profiles import FromCallable, ZProfile
from solprop.spatial import SpatialDispersion, SpatialMedium
from solprop.spatial_absorber import (
    SpatialAbsorber,
    SpatialAbsorberDesign,
)
from solprop.stats import StatsSpec
from solprop.stepper.controller import PIController, default_dz_min
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.stepper.tableau import DOPRI5, Tableau
from solprop.translate.chebyshev import (
    DEFAULT_TOL,
    PiecewiseChebyshev,
)
from solprop.translate.coupled import (
    CoupledFlowTables,
    constant_schur_flow,
    frozen_residual_flow,
)
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.flux import flux_plan
from solprop.translate.free import free_operators, free_transform
from solprop.translate.general import GeneralModalRoute
from solprop.translate.linop import (
    DEFAULT_MARGIN,
    LinearOperator,
    power_weight_row,
    walk_off_buckets,
)
from solprop.translate.masks import (
    DEFAULT_ABSORBER_SCHEME,
    AbsorberScheme,
    build_masks,
    noise_spectrum,
)
from solprop.translate.modal import (
    Transform,
    general_modal_transform,
    mode_averaged_transform,
    multimode_transform,
)
from solprop.translate.planar import planar_operators, planar_transform
from solprop.translate.problem import (
    _EDGE_SLACK,
    DEFAULT_COMPONENTS,
    DEFAULT_STRETCH,
    LOG_DBL_MAX,
    MAX_N_POL,
    Flags,
    Initial,
    PropagationProblem,
    PropagatorMode,
    StepperSettings,
    Translation,
    TranslationReuse,
    _state,
    debug_summary,
)
from solprop.translate.radial import (
    radial_operators,
    radial_transform,
)
from solprop.translate.responses import (
    CompiledResponses,
    compile_responses,
    declares_harmonic_carrier,
    dispersive_regions,
    response_kinks,
    unlanded_kinks,
)
from solprop.translate.routes import (
    GeneralRoute,
    ModalRoute,
    Route,
    SpatialRoute,
    TDSESource,
    spatial_kind,
)
from solprop.translate.spatial import (
    FluxPlan,
    SeparableOperator,
)
from solprop.translate.tdse_source import (
    admit_tdse,
    current_numerator,
)
from solprop.translate.transverse import checked_design, resolve_transverse
from solprop.translate.validation import (
    _INPUT_NOISE_ADMITTED,
    DEFAULT_FLOW_CONDITION_LIMIT,
    MOVING_FLOW_CERTIFICATE,
    _Assembly,
    _check_general,
    _check_harmonic_frame,
    _check_operator_settings,
    _check_tdse_assembly_route,
    _refuse_a_wall_off_the_capillary_route,
    _refuse_carrier_removing_frame,
    _refuse_planes_at_jumps,
    _refuse_split_crossing,
    _validated,
)
from solprop.translate.walls import (
    _coupled_assembly,
    _taper_assembly,
    _wall_assembly,
)

__all__ = [
    "MAX_GENERAL_FLOW_EXPONENT",
    "assemble",
]

logger = logging.getLogger(__name__)


_COMPONENT_NAMES = ("x", "y")


_GENERAL_INDEX = 1.0
"""Index override the general modal route builds its operator and weight with.

A general guide's basis is Poynting-normalised by construction, so its amplitude is
already the true-power amplitude every route carries and the statistics power
weight is the constant one: the legacy ``sqrt(Re n_eff)`` boundary is not
crossed on this route. The same override removes the cumulative normalisation term
from ``Phi``, which is the term that renormalises a vacuum amplitude along a varying
index and has no meaning for a basis that carries its own flux metric.
"""


def _spatial_assembly(
    spatial_grid: RadialGrid | FreeGrid | PlanarGrid,
    medium: SpatialMedium,
    grid: TimeGrid,
    frame: Frame,
    build: _Assembly,
) -> tuple[
    LinearOperator,
    Transform,
    PiecewiseChebyshev | ParameterExpansion | SeparableOperator,
    FluxPlan,
    SpatialAbsorberDesign | None,
]:
    """Assemble the operator, the transform, the prefactor and the plane measure.

    - ``spatial_grid``: the transverse geometry; a
      :class:`solprop.modes.slab.PlanarGrid` is the hybrid planar transform (slab modes
      in x, an FFT in y).
    - Returns: the operator, the transform, the already translated prefactor, the run's
      plane measure and its transverse boundary's design (SEP 40), ``None`` for none.
    """
    dispersion = (
        SpatialDispersion(medium, grid, frame)
        if build.evanescent_rate is None
        else SpatialDispersion(
            medium, grid, frame, evanescent_rate=build.evanescent_rate
        )
    )
    reuse = build.reuse
    design = _transverse_design(spatial_grid, dispersion, build)
    ramp = None if design is None else design.ramp_rate
    collar_rate = None if design is None else design.collar_rate
    if isinstance(spatial_grid, PlanarGrid):
        if build.has_noise:
            msg = (
                "a planar grid carries no noise yet: its columns are slab amplitudes, "
                "whose photon measure no draw declares"
            )
            raise TranslationError(msg)
        linear, prefactor = planar_operators(
            spatial_grid,
            dispersion,
            z_max=build.z_max,
            n_pol=build.n_pol,
            absorber=build.absorber,
            transverse_rate=ramp,
            margin=build.margin,
        )
        transform = (
            reuse.transform
            if reuse is not None
            else planar_transform(
                spatial_grid,
                grid,
                z_max=build.z_max,
                n_pol=build.n_pol,
                collar_rate=collar_rate,
                margin=build.margin,
            )
        )
        return (
            linear,
            transform,
            prefactor,
            flux_plan(spatial_grid, dispersion, linear, tol=build.tol),
            design,
        )
    if isinstance(spatial_grid, FreeGrid):
        linear, prefactor = free_operators(
            spatial_grid,
            dispersion,
            z_max=build.z_max,
            n_pol=build.n_pol,
            absorber=build.absorber,
            margin=build.margin,
            transverse_rate=ramp,
        )
        transform = (
            reuse.transform
            if reuse is not None
            else free_transform(
                spatial_grid,
                z_max=build.z_max,
                n_pol=build.n_pol,
                collar_rate=collar_rate,
                margin=build.margin,
            )
        )
        return (
            linear,
            transform,
            prefactor,
            flux_plan(spatial_grid, dispersion, linear, tol=build.tol),
            design,
        )
    admission = None
    if not all(path.is_constant for path in medium.parameter_paths.values()):
        # SEP 34 section 4.5: every active propagating column must meet the
        # adiabatic budgets over the whole run path before the forward amplitude
        # transport is trusted.
        admission = replace(
            certify_radial_path(
                dispersion,
                spatial_grid.k_r**2,
                (0.0, build.z_max),
                build.adiabatic_accuracy,
            ),
            # The operator tables resolve the represented path to this tolerance.
            interpolation_envelope=build.tol,
        )
    linear, prefactor = radial_operators(
        spatial_grid,
        dispersion,
        z_max=build.z_max,
        n_pol=build.n_pol,
        absorber=build.absorber,
        margin=build.margin,
        tol=build.tol,
        transverse_rate=ramp,
        reuse_linear=None if reuse is None else reuse.phi_expansion,
        reuse_prefactor=None if reuse is None else reuse.prefactor_expansion,
    )
    transform = (
        reuse.transform
        if reuse is not None
        else radial_transform(
            spatial_grid,
            grid,
            z_max=build.z_max,
            collar_rate=collar_rate,
            margin=build.margin,
        )
    )
    return (
        linear,
        transform,
        prefactor,
        flux_plan(spatial_grid, dispersion, linear, tol=build.tol, admission=admission),
        design,
    )


def _crossing_clamped(
    design: SpatialAbsorberDesign | None,
    dz_max: float,
    dz_init: float,
) -> tuple[float, float]:
    """Return the step bounds held to a collar's crossing limit (SEP 40 section 4.7).

    No adaptive step carries the protected band across the collar unseen; a run
    without a collar keeps its bounds. Every transverse design also holds the step to
    the safety ramp's exponent budget (section 6): the propagator's inverse factor
    ``exp(alpha_s h / 2)`` stays within half binary64's range, so a strong ramp on a
    fine grid cannot overflow it.

    - ``dz_max``: the requested largest step, m.
    - ``dz_init``: the requested first step, m.
    - Returns: the largest step and the first step, m.
    """
    if design is None:
        return dz_max, dz_init
    # A resolved design's ramp is never zero: every axis has a nonzero largest |k|.
    dz_max = min(dz_max, LOG_DBL_MAX / design.ramp_strength)
    if design.has_collar:
        dz_max = min(dz_max, design.crossing_step)
    return dz_max, min(dz_init, dz_max)


def _split_crossing(
    compiled: CompiledResponses,
    linear: LinearOperator,
    grid: TimeGrid,
    launch: Complex[np.ndarray, "n_modes n_pol n_omega"],
    columns: Float[np.ndarray, " n_modes"] | None,
) -> SplitCrossing | None:
    """Return the split temporal collar's crossing limit, or ``None`` without one.

    Under the ``"split_step"`` scheme the collar damps the state once per accepted
    step, so a step whose walk-off exceeds the collar's width can carry light through
    it and round the periodic window, undamped, where on a linear stretch the error
    estimate sees nothing. No step may walk the light more than
    :data:`~solprop.boundaries.SPLIT_CROSSING` of the collar
    (:class:`~solprop.boundaries.SplitCrossing`, the buckets of
    :func:`~solprop.translate.linop.walk_off_buckets`), read from where the light is:
    here from the launch, which holds an adaptive run's ``dz_max``
    (:func:`_split_clamped`) and a fixed schedule (:func:`_refuse_split_crossing`), and
    then by the boundary monitor from the state at every check. The in-stage scheme,
    and a run without a temporal collar, have no such limit.

    - ``columns``: the modes' quadrature weights, as the boundary monitor weighs them,
      or ``None`` for unit weights.
    - Returns: the limit, or ``None`` when none applies.
    """
    if compiled.absorber_scheme != "split_step" or compiled.temporal_absorber is None:
        return None
    # A nonzero temporal rate lies in a collar of nonzero width.
    width = min(grid.t_collar.width_low, grid.t_collar.width_high)
    buckets, fastest = walk_off_buckets(linear, grid.omega, grid.active_mask)
    crossing = SplitCrossing(buckets, width, fastest=fastest)
    return replace(
        crossing,
        launch_limit=crossing.limit(crossing.occupancy(launch[np.newaxis], columns)),
    )


def _split_clamped(
    crossing: SplitCrossing | None,
    schedule: FixedSchedule | None,
    dz_max: float,
    dz_init: float,
) -> tuple[float, float]:
    """Return an adaptive run's step bounds held to the launch's crossing limit.

    - ``schedule``: the schedule to replay, which keeps its bounds, or ``None``.
    - ``dz_max``: the requested largest step, m.
    - ``dz_init``: the requested first step, m.
    - Returns: the largest step and the first step, m. Lowering ``dz_max`` is logged at
      ``INFO``, naming the limit and what sets it.
    """
    if crossing is None or schedule is not None:
        return dz_max, dz_init
    limit = crossing.launch_limit
    if limit < dz_max:
        logger.info(
            "the split temporal collar's crossing limit holds dz_max to %.3e m, from "
            "%.3e m: %g of the collar's %.3e s width over the walk-off of the light "
            "the launch holds; the boundary monitor reads it again at every check",
            limit,
            dz_max,
            crossing.factor,
            crossing.width,
        )
    dz_max = min(dz_max, limit)
    return dz_max, min(dz_init, dz_max)


def _transverse_design(
    spatial_grid: RadialGrid | FreeGrid | PlanarGrid,
    dispersion: SpatialDispersion,
    build: _Assembly,
) -> SpatialAbsorberDesign | None:
    """Return the run's transverse boundary, resolved or carried over (SEP 40).

    A material update (:class:`TranslationReuse`) keeps the design its reused
    transform and expansions were built with, as it keeps the transform itself.

    - Returns: the design, or ``None`` when the run asked for no transverse boundary.
    """
    if build.reuse is not None:
        return build.reuse.transverse
    if build.spatial_absorber is None:
        return None
    assert build.launch is not None
    design = resolve_transverse(
        build.spatial_absorber, spatial_grid, dispersion, build.launch, build.z_max
    )
    return checked_design(design, spatial_grid, dispersion, build.launch, build.z_max)


def _general_assembly(
    route: GeneralModalRoute,
    grid: TimeGrid,
    modes: Sequence[Mode],
    frame: Frame,
    build: _Assembly,
) -> tuple[LinearOperator, Transform, PiecewiseChebyshev]:
    """Assemble the operator, the transform and the prefactor of a general guide.

    The linear half of the general modal route is the **existing** cumulative
    diagonal flow: ``Phi`` comes from the channels' own index tables through the one
    table-to-operator site, in the one shared co-moving frame, with ``beta``
    certified by the guide's constructor. Nothing new
    happens here; what is new is the transform, which carries the E/H tables instead
    of a static matrix, and the metric, which records the basis certificate.

    **Unless the route declares a coupled block**, which happens in one case: a
    declared unitary basis of a *split* block leaves
    ``U.H diag(beta) U`` off the diagonal and the channels genuinely beat. Then
    ``Phi`` is built from the route's reference channel alone, replicated, so that it
    is a multiple of the identity in the channel axis and commutes with the block,
    and every channel's departure from that reference -- its own dispersion included
    -- goes into ``linear.coupled_flow`` beside it
    (:func:`_general_flow`). This is the general assembly's own flow and it shares
    nothing with the capillary's but
    :func:`solprop.translate.coupled.constant_schur_flow` itself: there is no wall
    vector, no validity ratio and no Marcatili mode in it.

    - Returns: the operator carrying the identity metric and the unit power weight, the
      transform carrying the tables, and the prefactor row.
    """
    _check_general(route, grid, build)
    coupled = route.coupled_generator
    # A coupled block needs a carrier that commutes with it, which is what the
    # reference channel replicated gives; with no block every channel keeps its
    # own, and the route is the plain diagonal flow. A moving guide's carrier is the
    # reference channel's **entrance** dispersion, declaring the geometry's kinks so
    # that the driver lands on them.
    carried: Sequence[Mode] = (
        modes
        if coupled is None and not route.moving
        else (_carrier(modes[route.reference_channel], route),) * len(modes)
        if route.moving
        else (modes[route.reference_channel],) * len(modes)
    )
    _, linear = _coupled_assembly(
        carried,
        modes,
        grid,
        frame,
        build,
        index=_GENERAL_INDEX,
        reuse=None if build.reuse is None else build.reuse.phi_expansion,
    )
    transform = general_modal_transform(
        modes,
        route.tables,
        z_max=build.z_max,
        margin=build.margin,
        tol=build.tol,
    )
    prefactor = _general_prefactor(grid, linear, n_pol=build.n_pol)
    logger.debug(
        "general modal route: %d channels, Gram defect %.3e, %d response region(s), "
        "power weight 1, prefactor the crop scale %.6g",
        route.n_modes,
        route.gram_defect,
        0 if route.responses is None else route.responses.n_responses,
        grid.crop_scale,
    )
    metric = replace(linear, modal_metric=route.metric)
    if route.moving:
        return (
            replace(metric, coupled_flow=_moving_flow(route, grid, build)),
            transform,
            prefactor,
        )
    if coupled is None:
        return metric, transform, prefactor
    return (
        replace(metric, coupled_flow=_general_flow(route, coupled, grid, build)),
        transform,
        prefactor,
    )


MAX_GENERAL_FLOW_EXPONENT: float = 1.0e3
"""Largest ``dz_max max|G|`` a general modal coupled flow is executed at.

Dimensionless. The backend takes the flow's action by scaling and squaring a Schur
factor, whose relative error grows with the exponent's norm: at ``1e3`` the action is
still good to about ``1e-13`` relative, which is the accuracy the diagonal route it
replaces has. The exponent a *split block* produces is tiny -- the departure of each
channel from the reference is the block's own splitting, 744 1/m on the TE01/TM01
pair, and ``dz_max`` is microns -- so this limit exists to catch a block built from
channels that were never a degenerate group, whose departures are the whole
dispersion and which would silently lose digits.
"""


def _general_flow(
    route: GeneralModalRoute,
    coupled: Complex[np.ndarray, "n_modes_gf_out n_modes_gf_in n_omega_gf"],
    grid: TimeGrid,
    build: _Assembly,
) -> CoupledFlowTables:
    """Build the constant coupled flow of a general modal guide's split block.

    One constant section, because the geometry does not move: the guide is a
    snapshot and only the *coordinates* are rotated, so the block is the same matrix
    at every position and :func:`solprop.translate.coupled.constant_schur_flow` is
    the method. A guide that varies along z needs the transform maps to move with
    it and is built by :func:`_moving_flow` instead.

    - ``route``: its gauge identity is carried by the flow, so that a resume comparing
      the flow compares the basis it was built in.
    - ``coupled``: the block relative to the reference channel in 1/m, shape ``(n_modes,
      n_modes, n_omega)``.
    - ``grid``: the run's grid, whose coarse axis the leaf is tabulated on.
    """
    if build.noise_model == "modified":
        msg = (
            "a split block's coupled flow refuses the 'modified' noise model: its "
            "draw is transported by the unitary phase of a diagonal flow, and the "
            "transport of a draw through a nonunitary coupled flow is not measured "
            "in this package; the backend refuses it too. " + _INPUT_NOISE_ADMITTED
        )
        raise TranslationError(msg)
    exponent = float(build.dz_max) * float(np.max(abs(coupled)))
    if exponent > MAX_GENERAL_FLOW_EXPONENT:
        worst = int(np.argmax(np.max(abs(coupled), axis=(0, 1))))
        msg = (
            f"a general modal coupled block reaches {exponent:.4g} over the "
            f"{float(build.dz_max):.4g} m largest step this run declares, at "
            f"{float(grid.omega[worst]):.6g} rad/s, above the "
            f"{MAX_GENERAL_FLOW_EXPONENT:.4g} its matrix exponential is executed "
            "at: the block carries every channel's departure from the reference "
            "one, so a declared basis over channels that are not a degenerate group "
            "puts their whole dispersion difference in it. Cap dz_max, or declare "
            "the basis over channels whose propagation constants are close"
        )
        raise TranslationError(msg)
    flow = constant_schur_flow(
        grid.omega,
        np.array([0.0, float(build.z_max) * (1.0 + float(build.margin))]),
        coupled,
        dz_max=build.dz_max,
        condition_limit=DEFAULT_FLOW_CONDITION_LIMIT,
        gauge_id=route.tables.gauge_id,
        certificate_id="general-constant-schur",
    )
    logger.debug(
        "general modal coupled flow: %d channels relative to channel %d, largest "
        "%.3e 1/m, exponent %.3e over dz_max %.3g m, conditioning %.6g",
        flow.n_modes,
        route.reference_channel,
        float(np.max(abs(coupled))),
        exponent,
        build.dz_max,
        float(np.max(flow.inverse_condition_bound)),
    )
    return flow


def _landed_kinks(
    build: _Assembly, linear: LinearOperator
) -> Float[np.ndarray, " n_kinks"]:
    """Return the operator's hard kinks, refusing a response kink they do not hold.

    - Returns: ``linear.hard_kinks``, in m, shape ``(n_kinks,)``.
    """
    straddled = unlanded_kinks(
        build.kinks,
        linear.hard_kinks,
        build.z_max,
        slack=_EDGE_SLACK * linear.z_span[1],
    )
    if straddled:
        z, name = straddled[0]
        msg = (
            f"the {name} path kinks at z = {z!r} m, and this route cannot make steps "
            "land there (only the uncoupled modal route adds a response's kinks to "
            "Phi); give the path no interior breakpoints, or run it on that route"
        )
        raise TranslationError(msg)
    return linear.hard_kinks


@dataclass(frozen=True, eq=False)
class _PathCarrier:
    """Carry a moving guide's ``Phi``: one channel's entrance dispersion, and its kinks.

    ``Phi`` beside a coupled flow must be a multiple of the identity in the channel
    axis, and beside a ``"frozen_residual"`` flow it must be a weighted sum of fixed
    terms (``numpy_ref``'s admission; a z table has no stage-bundle form). A moving
    solid guide's channels have no scalar parameter they are functions of, so ``Phi``
    carries the reference channel's **entrance** dispersion, z-constant, and the flow
    carries every channel's departure from it -- the reference channel's own motion
    included (:func:`_moving_flow`). What is subtracted is exactly what is carried,
    so the frame cancels between the two.

    The carrier declares one path, the position itself with the maps' interior
    section edges -- the provider's kinks among them -- as its breakpoints, on which
    its dispersion does **not** depend: the expansion is then resolved at degree zero
    (a declared path is a statement of z ownership), and the edges become
    ``Phi``'s hard kinks, which the driver lands on and a fixed schedule is held to,
    because the connection jumps with the slope at a kink.
    """

    entrance: Mode
    path: ZProfile

    @property
    def z_constant(self) -> bool:
        """Return False, so that the declared kinks reach the operator's tables."""
        return False

    @property
    def scale_invariant(self) -> bool:
        """Return False: a solid guide's modes have no scaling law."""
        return False

    @property
    def channel_label(self) -> ChannelLabel:
        """Return the reference channel's label."""
        return self.entrance.channel_label

    @property
    def azimuthal_order(self) -> int:
        """Return the reference channel's azimuthal order."""
        return self.entrance.azimuthal_order

    def principal_component(self, z: float = 0.0) -> PolarisationComponent | None:
        """Return the reference channel's principal component at the entrance.

        - ``z``: position in m; ignored.
        """
        del z
        return self.entrance.principal_component()

    def neff(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the entrance index at every position.

        - ``omega``: angular frequencies in rad/s.
        - ``z``: position in m; ignored, the carrier being the entrance's.
        """
        del z
        return self.entrance.neff(omega)

    def neff_minus_one(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the entrance ``n_eff - 1`` at every position.

        - ``omega``: angular frequencies in rad/s.
        - ``z``: position in m; ignored.
        """
        del z
        return self.entrance.neff_minus_one(omega)

    def dneff_domega(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the entrance index derivative at every position.

        - ``omega``: angular frequencies in rad/s.
        - ``z``: position in m; ignored.
        - Returns: ``d n_eff / d omega`` in s/rad, complex.
        """
        del z
        return self.entrance.dneff_domega(omega)

    def aeff(self, z: float = 0.0) -> float:
        """Return the entrance surrogate area.

        - ``z``: position in m; ignored.
        """
        del z
        return self.entrance.aeff()

    def dimlimits(
        self, z: float = 0.0
    ) -> tuple[CrossSection, tuple[float, ...], tuple[float, ...]]:
        """Return the entrance domain.

        - ``z``: position in m; ignored.
        """
        del z
        return self.entrance.dimlimits()

    def field(
        self, points: Float[np.ndarray, "n_points 2"], z: float = 0.0
    ) -> Float[np.ndarray, "n_points 2"]:
        """Refuse, as the provider channel it carries does.

        - ``points``: Cross-section positions in m.
        - ``z``: position in m.
        """
        return self.entrance.field(points, z)

    @property
    def parameter_paths(self) -> Mapping[str, ZProfile]:
        """Return the one declared path: the position, kinked where the maps are.

        - Returns: ``{"position": path}``.
        """
        return MappingProxyType({"position": self.path})

    def at_parameters(self, values: Mapping[str, float]) -> Mode:
        """Return the entrance channel, whatever the position."""
        if set(values) - {"position"}:
            msg = f"a moving guide's carrier declares one path, got {sorted(values)}"
            raise TranslationError(msg)
        return self.entrance


def _carrier(mode: Mode, route: GeneralModalRoute) -> _PathCarrier:
    """Return the carrier of a moving guide's ``Phi`` for its reference channel."""
    entrance = getattr(mode, "entrance", None)
    base = cast("Mode", entrance() if callable(entrance) else mode)
    edges = route.tables.z_edges
    # Every section edge, not the provider's kinks alone: a subdivision edge is where
    # the frozen block and both expansions change, and landing there costs a step at
    # most while a kink straddled costs the run its order.
    return _PathCarrier(
        base,
        FromCallable(
            _position,
            (float(edges[0]), float(edges[-1])),
            np.asarray(edges[1:-1], dtype=float),
        ),
    )


def _position(z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
    """Return the position itself, the carrier's declared path.

    - ``z``: positions in m.
    """
    return np.asarray(z, dtype=float)


def _moving_flow(
    route: GeneralModalRoute, grid: TimeGrid, build: _Assembly
) -> CoupledFlowTables:
    """Build a moving guide's frozen-residual coupled flow.

    The route's generator (every channel less the reference channel's entrance
    diagonal, which ``Phi`` carries) and its connection ``Gamma`` are stored on the
    maps' own sections, and :func:`solprop.translate.coupled.frozen_residual_flow`
    freezes each section's generator at its midpoint and leaves the rest -- the
    variation along the section and the whole connection -- to the explicit residual
    the interaction picture's source carries, as for a varying capillary. The
    flow's sections are the maps' here, which is natural and not required: the
    stage bundle carries the two weight rows separately (kernel schema 14).
    """
    if build.noise_model == "modified":
        msg = (
            "a moving guide's coupled flow refuses the 'modified' noise model: its "
            "draw is transported by the unitary phase of a diagonal flow, and its "
            "transport through a moving guide's coupled flow is not measured in "
            "this package; the backend refuses it too. " + _INPUT_NOISE_ADMITTED
        )
        raise TranslationError(msg)
    assert route.moving_generator is not None
    return frozen_residual_flow(
        grid.omega,
        route.tables.z_edges,
        route.moving_generator,
        route.moving_connection,
        dz_max=build.dz_max,
        condition_limit=DEFAULT_FLOW_CONDITION_LIMIT,
        representation_error=route.moving_error,
        gauge_id=route.tables.gauge_id,
        certificate_id=MOVING_FLOW_CERTIFICATE,
    )


def _modal_linear(
    wall: CoherentWall | None,
    modes: Sequence[Mode],
    grid: TimeGrid,
    frame: Frame,
    build: _Assembly,
) -> LinearOperator:
    """Return the linear operator of a mode-set route, with or without a wall block.

    - ``wall``: the coherent wall declaration, or ``None``. Without one, a multimode
      capillary taper carries the taper coupling when ``build.taper_coupling`` says so,
      and every other set -- or that one with the coupling declined -- takes the
      uncoupled diagonal route.
    - Returns: section 3 of the contract, carrying ``coupled_flow`` exactly when a wall
      was declared or the taper coupling applies (:func:`_taper_assembly`).
    """
    if wall is not None:
        # Every channel's frequency-only terms are sampled once for the whole wall
        # assembly rather than at each of its positions.
        with _shared_sampling():
            return _wall_assembly(wall, modes, grid, frame, build)
    block = taper_coupling_block(modes) if build.taper_coupling else None
    if block is not None:
        with _shared_sampling():
            return _taper_assembly(block, grid, frame, build)
    # The frame tables read the reference mode at every sample along z, and each
    # read would sample its fill and wall again (SEP 3, Q22): once per frequency
    # vector here too.
    with _shared_sampling():
        _, linear = _coupled_assembly(
            modes,
            modes,
            grid,
            frame,
            build,
            index=build.prefactor_index,
            reuse=None if build.reuse is None else build.reuse.phi_expansion,
            kinks=[z for z, _ in build.kinks],
        )
    return linear


def _general_prefactor(
    grid: TimeGrid, linear: LinearOperator, *, n_pol: int
) -> PiecewiseChebyshev:
    """Return the general modal route's prefactor row: the crop scale alone.

    Every other route's prefactor is the UPPE factor ``-i omega / (2 eps0 c
    Re n_eff)`` with the crop scale folded in. On the general route that factor is
    replaced by the record's own ``source_frequency_factor`` and by the reciprocal
    normalisation inside Q, so what is left for this row is the crop scale
    ``n_t / n_to`` -- which stays here, where it has always been, because the
    projection is frequency dependent and folding a grid convention into it would
    hide it. It is real, and it is exactly zero outside the
    active band as every other route's prefactor is, so no unnormalised
    polarisation survives there for the stages or the error estimate to see.

    - ``linear``: the assembled operator, whose z span every other table must share.
    - Returns: one interval of degree zero with value shape ``(n_modes, n_pol,
      n_omega)``.
    """
    n_modes = linear.value_shape[1]
    values = np.zeros((1, 1, n_modes, n_pol, grid.n_omega), dtype=np.complex128)
    values[:, :, :, :, grid.active_mask] = grid.crop_scale
    return PiecewiseChebyshev(
        hold(values),
        hold(np.array(linear.z_span, dtype=np.float64)),
        hold(np.zeros(1, dtype=np.intp)),
        hold(np.zeros(0)),
    )


def _material_mixture(
    modes: Sequence[Mode],
    medium: SpatialMedium | None,
    density: ZProfile | float | None,
) -> Mixture | None:
    """Resolve the material shared by the linear and response paths.

    - ``modes``: ordered mode set, empty for spatial geometry.
    - ``medium``: spatial material, or None for modal geometry.
    - ``density``: optional explicit response density in m^-3.
    - Returns: the declared mixture, or None for the legacy single material.
    """
    if medium is not None and isinstance(medium.material, Gas) and density is not None:
        # The linear operator takes the medium's own density path; an explicit one
        # would scale only the nonlinear responses, so the two would describe
        # different fills.
        raise TranslationError(
            f"a {medium.material} medium carries its density (its pressure and "
            "temperature set it, for the linear and the nonlinear terms alike); drop "
            "the density argument, or give the medium the density you mean "
            "(SpatialMedium.at_density)"
        )
    mixture = None
    if medium is not None and isinstance(medium.material, Mixture):
        mixture = medium.material
    elif modes:
        declared = getattr(modes[0], "mixture", None)
        if isinstance(declared, Mixture):
            mixture = declared
    if mixture is not None and density is not None:
        raise TranslationError(
            "a mixture carries its densities; drop the density argument"
        )
    return mixture


def _stepper_settings(
    linear: LinearOperator,
    *,
    controller: PIController,
    interval_cap: bool | None,
    auto_init: bool,
    **fields: Any,
) -> StepperSettings:
    """Return the stepper settings, with the route's step limits resolved.

    SEP 14 phase 1 relaxes the step limits on the diagonal-flow routes only. A coupled
    flow's error lies outside the embedded estimate, and a reject-free run cannot undo
    an overshoot, so both keep the version-16 policy: the interval cap and no start-up
    rule.

    - ``interval_cap``: the interval cap asked for; ``None`` takes the route's.
    """
    restrained = linear.coupled_flow is not None or controller.reject_free
    return StepperSettings(
        controller=controller,
        interval_cap=restrained if interval_cap is None else interval_cap,
        auto_init=auto_init and not restrained,
        **fields,
    )


@host_blas()
def assemble(  # noqa: PLR0915 - one keyword per physics object of a run
    grid: TimeGrid,
    modes: Mode | Sequence[Mode] | None,
    responses: Sequence[object],
    field0: Complex[np.ndarray, "*shape"],
    *,
    controller: PIController,
    save_planes: Sequence[float] | np.ndarray,
    dz_max: float,
    dz_init: float,
    density: ZProfile | float | None = None,
    noise: NoiseRealisation | None = None,
    frame: Frame | None = None,
    loss: LossPolicy = DEFAULT_LOSS_POLICY,
    absorber: Absorber | None = None,
    absorber_scheme: AbsorberScheme = DEFAULT_ABSORBER_SCHEME,
    quadrature: TransverseQuadrature | None = None,
    components: Sequence[str] | None = None,
    tableau: Tableau = DOPRI5,
    integrator: Multistep | LowStorageRungeKutta | None = None,
    schedule: FixedSchedule | None = None,
    dz_min: float | None = None,
    stretch: float | None = None,
    interval_cap: bool | None = None,
    auto_init: bool = False,
    interpolate_planes: bool = False,
    step_planes: int = 0,
    propagator_mode: PropagatorMode = "auto",
    stats_spec: StatsSpec | None = None,
    shock: bool = True,
    prefactor_index: float | None = None,
    ionisation_field_range: tuple[float, float] | None = None,
    spatial_grid: RadialGrid | FreeGrid | PlanarGrid | None = None,
    medium: SpatialMedium | None = None,
    margin: float = DEFAULT_MARGIN,
    tol: float = DEFAULT_TOL,
    polarisation_taper: bool = False,
    general: GeneralModalRoute | None = None,
    wall: CoherentWall | None = None,
    taper_coupling: bool = True,
    reuse: TranslationReuse | None = None,
    initial_z: float = 0.0,
    evanescent_rate: float | None = None,
    spatial_absorber: SpatialAbsorber | None = None,
    adiabatic_accuracy: AdiabaticAccuracy | None = None,
) -> Translation:
    """Turn the physics objects of a run into a validated kernel problem.

    This is the whole of the translation layer's public entry point: it builds the
    cumulative linear operator, the masks, the modal transform and the compiled
    responses, derives the flags from what it built, and validates the result before
    returning it. Everything it does is expensive once and free thereafter
    (``docs/dev/architecture.md`` section 7).

    The run covers ``[initial_z, max(save_planes)]``. Tables retain absolute
    positions from zero and cover a further `margin` beyond the last plane.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The field or envelope grid.
    modes : Mode or sequence of Mode
        The mode, or the ordered set of a multimode problem. A set needs `quadrature`.
    responses : sequence
        The declared nonlinear responses; see
        :func:`solprop.translate.responses.compile_responses`. A TDSE response
        owns the complete gas nonlinearity and refuses another Kerr or plasma term.
    field0 : numpy.ndarray
        Initial state, shape ``(n_modes, n_pol, n_omega)`` or already batched, complex,
        in the square root of watts and the internal FFT convention — a
        :meth:`solprop.pulses.Pulse.build` result, or a stack of them.
    controller : solprop.stepper.controller.PIController
        Tolerances, norm and step-control parameters.
    save_planes : array_like
        Positions in z the driver will ask for, m, increasing and positive.
    dz_max : float
        Cap on the controller's proposal, m.
    dz_init : float
        First step, m.
    density : solprop.profiles.ZProfile or float or None, optional
        Number density along z in m^-3. ``None`` takes the mode's own ``density`` path
        when it declares one and 1.0 otherwise, which is the convention a medium with an
        absolute ``chi3`` uses. Refused beside a mixture and beside a gas `medium`,
        which carry their own.
    noise : solprop.noise.NoiseRealisation or None, optional
        A shot-noise realisation to carry along the run.
    frame : solprop.linear.Frame or None, optional
        The co-moving frame. ``None`` builds
        :meth:`solprop.linear.Frame.comoving` for the grid, keeping ``beta_0`` when a
        third-harmonic term is present on an envelope grid
        (:func:`solprop.translate.responses.declares_harmonic_carrier`). A frame given
        with ``subtract_beta0=True`` beside that term is refused.
    loss : solprop.linear.LossPolicy, optional
        What to do with an attenuation too large to represent.
    absorber : solprop.boundaries.Absorber or None, optional
        The collar rates. Its spectral part is folded into the linear operator and its
        temporal part crosses the seam as a rate.
    absorber_scheme : {'split_step', 'nonlinear'}, optional
        Which application of the temporal absorber the backend will use.
    quadrature : solprop.modes.quadrature.TransverseQuadrature or None, optional
        The fixed transverse rule of a multimode problem. ``None`` builds the
        mode-averaged, single-node transform, which needs exactly one mode.
    components : sequence of str or None, optional
        The transverse Cartesian components the multimode transform resolves at each
        node, a non-empty subset of ``("x", "y")``. ``None`` takes the union of
        the mode set's nonzero components at the entrance, from each mode's
        ``principal_component(0.0)``: ``("x",)`` for HE_1m modes at ``phi = 0``,
        and both for a rotated or mixed set; pass
        ``("x", "y")`` for a polarisation-resolved mode set
        (:meth:`solprop.modes.collection.ModeCollection.with_polarisations`). Only
        meaningful with a `quadrature`, and refused without one. The product of its
        length and the state's polarisation axis may not exceed two: the components
        are carried by the transform or by the state, never by both.
    tableau : solprop.stepper.tableau.Tableau, optional
        The Runge-Kutta tableau. Default DOPRI5.
    integrator : solprop.stepper.multistep.Multistep or LowStorageRungeKutta, optional
        A low-call Adams method instead of the tableau (SEP 13), or the low-storage RK4
        march, which replays `schedule` (SEP 24).
    schedule : solprop.stepper.schedule.FixedSchedule or None, optional
        A fixed schedule to replay. Its edges must contain every save plane and hard
        kink.
    dz_min : float or None, optional
        Floor on the controller's proposal, m. ``None`` uses
        :func:`solprop.stepper.controller.default_dz_min` of the run length.
    stretch : float or None, optional
        Landing stretch factor. ``None`` uses
        :data:`~solprop.translate.problem.DEFAULT_STRETCH`.
    interval_cap : bool or None, optional
        Whether a step is capped at its operator-table interval's width. ``None`` caps
        it only for a coupled flow or a reject-free controller (SEP 14 section 4.1).
    auto_init : bool, optional
        Whether the start-up rule chooses the step after the first; `dz_init` is the
        first. Default ``False``, the convenience interfaces' choice when no `dz_init`
        is given. Off for a coupled flow or a reject-free controller.
    interpolate_planes : bool, optional
        Whether save planes are observed inside accepted steps -- through DOPRI5's
        continuous extension or the Adams march's polynomial -- rather than landed on
        (SEP 14 phase 3). Refused beside a coupled flow, the per-step temporal collar
        and the ``modified`` noise model.
    step_planes : int, optional
        Also save the end of every `step_planes`-th accepted step (SEP 14 phase 3);
        ``0``, the default, saves the requested planes only.
    propagator_mode : {'auto', 'tabulate', 'recompute'}, optional
        Whether the backend tabulates the stage propagators.
    stats_spec : solprop.stats.StatsSpec or None, optional
        Which statistics the run collects, carried by the returned request.
        ``None`` collects none.
    shock : bool, optional
        Keep the ``omega`` dependence of the prefactor's numerator, which is the optical
        shock term. Default ``True``.
    prefactor_index : float or None, optional
        Use this constant in the prefactor and power weight; None uses Re n_eff.
        The GNLSE passes 1.0, leaving its power normalisation and Phi unchanged.
    ionisation_field_range : tuple of float or None, optional
        Field amplitudes in V/m a plasma response's rate table must cover.
    spatial_grid : RadialGrid, FreeGrid, PlanarGrid or None, optional
        Radial or Cartesian geometry, or the hybrid planar grid
        (:class:`solprop.modes.slab.PlanarGrid`); requires ``modes=None`` and a bulk
        `medium`. The initial columns then contain Hankel spectra in V m, raw
        Cartesian DFT spectra in V/m, or y-DFT spectra of slab amplitudes in
        sqrt(W/m) (the planar grid carries its own modes).
    medium : solprop.spatial.SpatialMedium or None, optional
        Material and density path for spatial propagation. Its density is used for
        nonlinear responses; an explicit `density` is refused beside a gas or a
        mixture medium, and scales the responses of a glass or a constant index.
    margin : float, optional
        Fraction of the run length the tables extend beyond its end.
    tol : float, optional
        Target bound on the uniform error of the tables. Raised automatically to the
        measured round-off floor of the operator (:mod:`solprop.translate.linop`).
    polarisation_taper : bool, optional
        Whether the spectral collar tapers the nonlinear polarisation as well as
        absorbing the field through the operator. Default ``False``: the taper sets a
        floor on a broad pulse of the size of its content in the collar.
    general : solprop.translate.general.GeneralModalRoute or None, optional
        The seam leaves of a general modal guide. With it the
        three-component E/H route is assembled instead of the static transverse
        quadrature: ``modes`` are the guide's channels in state order and supply the
        diagonal operator, the power weight is the constant one, the prefactor is the
        crop scale alone and the responses are the route's own per-region record. A
        `quadrature`, `components`, `spatial_grid`, `medium`, `density`,
        `prefactor_index`, declared `responses` or ``shock=False`` is refused beside
        it by name, and the grid must be a field grid whose active band lies inside
        the guide's support.
    wall : solprop.modes.capillary.CoherentWall or None, optional
        The declaration of a scalar coherent wall block on the capillary route
        or, with ``impedance="te_tm"``, of one azimuthal order, on a constant or a
        varying guide. With it ``Phi`` carries the mode-independent carrier alone
        and the modal part of the flow becomes ``linear.coupled_flow``, which every
        backend executes. A general modal guide,
        a spatial geometry, a polarisation axis and a noise model are refused beside
        it by name.
    taper_coupling : bool, optional
        Carry the taper coupling on a multimode capillary whose radius varies along
        z: the moving-basis connection between the ``HE_1m`` channels, in a
        frozen-residual ``linear.coupled_flow`` whose diagonal is each channel's own
        propagation constant, lossless or not
        (:func:`solprop.modes.capillary.taper_coupling_block` says which sets). The
        default ``True``; ``False`` keeps the adiabatic diagonal route, in which each
        channel propagates on its own. It has no effect on any other mode set or
        route, and ``False`` beside a `wall` is refused, because the wall always
        carries its connection.
    reuse : TranslationReuse or None, optional
        Immutable data prequalified for the same grid, input, geometry and operator
        functions. New paths still undergo sampled-range checks. None builds fresh.
    initial_z : float, optional
        Absolute launch position in m, non-negative and before the first save.
        ``field0`` is a true-power spectrum at this position; the input boundary
        applies the corresponding modal power weight exactly once.
    evanescent_rate : float or None, optional
        The spatial operator's evanescent field attenuation cap in m^-1, positive and
        finite (SEP 22 section 4.4); ``None`` keeps
        :data:`solprop.spatial.EVANESCENT_RATE`. Refused without a spatial grid.
    spatial_absorber : solprop.spatial_absorber.SpatialAbsorber or None, optional
        The transverse boundary (SEP 40): resolved against the grid, the medium and
        the noise-free `field0` into the safety ramp, folded into the operator as
        ``SeparableOperator.kperp_rate``, and a real-space collar on request, as the
        geometry's ``absorber_rate``; its design is the spatial route's
        ``transverse`` (:class:`~solprop.translate.routes.SpatialRoute`). ``None``,
        the default here, carries neither. Refused without a spatial grid.
    adiabatic_accuracy : solprop.adiabatic.AdiabaticAccuracy or None, optional
        The budgets a radial medium varying along z must meet for its forward
        amplitude transport (SEP 34 section 4.5); None, the default, is
        ``AdiabaticAccuracy()``. A path exceeding them is refused.

    Returns
    -------
    Translation
        The problem and the request of a fresh run of it, with the step limits the
        route resolved and clamped, validated together
        (:func:`~solprop.translate.problem.validate_run`) and ready for a backend.

    Raises
    ------
    TranslationError
        If a response cannot be compiled, an argument is out of range, a frame that
        removes ``beta_0`` is given beside a third-harmonic term on an envelope grid,
        or the assembled problem and its request fail
        :func:`~solprop.translate.problem.validate_run`.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.assembly import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0, gamma=0.11,
    ...                      shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=1e-13, peak_power=1.0).build(grid)
    >>> problem, request = assemble(
    ...     grid, gnlse.mode, [gnlse.kerr], start,
    ...     controller=PIController(rtol=1e-8, atol=1e-14), save_planes=[0.5, 1.0],
    ...     dz_max=0.1, dz_init=1e-3, shock=False, prefactor_index=1.0, density=1.0,
    ... )
    >>> problem.flags.single_mode, problem.linear.n_intervals
    (True, 1)
    >>> request.observation.save_planes.tolist()
    [0.5, 1.0]
    """
    _refuse_a_wall_off_the_capillary_route(
        wall,
        general=general is not None,
        spatial=spatial_grid is not None or medium is not None,
        taper_coupling=taper_coupling,
    )
    state = _state(field0, grid.n_omega)
    n_modes, n_pol, n_omega = state.shape
    mode_set = (
        ()
        if modes is None
        else tuple(modes)
        if isinstance(modes, Sequence)
        else (modes,)
    )
    responses, tdse = admit_tdse(
        responses,
        grid,
        mode_set,
        shape=state.shape,
        quadrature=quadrature,
        integrator=integrator,
        interpolate_planes=interpolate_planes,
        modified_noise=noise is not None and noise.model == "modified",
        shock=shock,
        prefactor_index=prefactor_index,
    )
    _check_tdse_assembly_route(
        tdse,
        modal=spatial_grid is None and general is None and wall is None,
        scheduled=schedule is not None,
        absorber_scheme=absorber_scheme,
        density=density,
    )
    if (spatial_grid is None) != (medium is None):
        msg = "spatial_grid and medium must be given together"
        raise TranslationError(msg)
    _check_operator_settings(
        spatial=spatial_grid is not None,
        evanescent_rate=evanescent_rate,
        spatial_absorber=spatial_absorber,
    )
    if spatial_grid is None and len(mode_set) != n_modes:
        msg = (
            f"field0 carries {n_modes} modes but {len(mode_set)} were given; the state "
            "layout and the mode set must agree"
        )
        raise TranslationError(msg)
    if spatial_grid is not None:
        if modes is not None or quadrature is not None or components is not None:
            msg = (
                "spatial assembly requires modes=None and no modal quadrature "
                "or components"
            )
            raise TranslationError(msg)
        if n_modes != spatial_grid.n_columns:
            msg = "field0 spatial columns must match spatial_grid.n_columns"
            raise TranslationError(msg)
        if not shock or prefactor_index is not None or loss != DEFAULT_LOSS_POLICY:
            msg = (
                "spatial assembly uses exact material normalisation; shock=False, "
                "prefactor_index and modal loss policies are unsupported"
            )
            raise TranslationError(msg)
    planes = np.asarray(save_planes, dtype=np.float64).reshape(-1)
    z_max = float(np.max(planes)) if planes.size else 0.0
    if z_max <= 0.0:
        msg = f"save_planes must contain a positive position, got {planes}"
        raise TranslationError(msg)
    _refuse_planes_at_jumps(medium, planes)

    harmonic = declares_harmonic_carrier(responses, grid)
    _refuse_carrier_removing_frame(frame, harmonic)
    working_frame = frame if frame is not None else Frame.comoving(grid, thg=harmonic)
    build = _Assembly(
        quadrature=quadrature,
        components=components,
        spatial_grid=spatial_grid,
        medium=medium,
        density=density,
        prefactor_index=prefactor_index,
        shock=shock,
        responses=responses,
        n_pol=n_pol,
        z_max=z_max,
        absorber=absorber,
        loss=loss,
        margin=margin,
        tol=tol,
        has_noise=noise is not None,
        noise_model="none" if noise is None else noise.model,
        dz_max=dz_max,
        reuse=reuse,
        kinks=response_kinks(
            responses, density if general is None and spatial_grid is None else None
        ),
        taper_coupling=taper_coupling,
        evanescent_rate=evanescent_rate,
        spatial_absorber=spatial_absorber,
        launch=state,
        adiabatic_accuracy=adiabatic_accuracy,
    )
    # Each mode's frequency-only terms are sampled once per frequency vector for the
    # whole assembly: the dispersion, power-weight and prefactor tables all read them.
    with _shared_sampling():
        translated_prefactor = None
        flux = None
        design = None
        if general is not None:
            linear, transform, translated_prefactor = _general_assembly(
                general, grid, mode_set, working_frame, build
            )
        elif spatial_grid is not None and medium is not None:
            linear, transform, translated_prefactor, flux, design = _spatial_assembly(
                spatial_grid, medium, grid, working_frame, build
            )
        else:
            linear = _modal_linear(wall, mode_set, grid, working_frame, build)
            transform = (
                reuse.transform
                if reuse is not None
                else _transform(
                    mode_set,
                    grid,
                    quadrature,
                    z_max=z_max,
                    n_pol=n_pol,
                    components=components,
                    margin=margin,
                    tol=tol,
                )
            )
        compiled = compile_responses(
            responses,
            grid,
            mode_set if spatial_grid is None and general is None else None,
            density=medium.density
            if medium is not None and density is None
            # A general modal guide's susceptibilities are absolute and per region, so
            # its density path is the dimensionless one.
            else 1.0
            if general is not None
            else _density(mode_set, density),
            z_max=z_max,
            n_pol=n_pol,
            mode_averaged=transform.mode_averaged,
            shock=shock,
            prefactor_index=prefactor_index,
            ionisation_field_range=ionisation_field_range,
            absorber=absorber,
            absorber_scheme=absorber_scheme,
            # The Phi table's hard kinks are the union of the mode set's declared
            # breakpoints (and of the response paths' own), so passing them is
            # what makes the two tables split on the same edges without either
            # re-deriving them.
            mandatory_edges=_landed_kinks(build, linear),
            margin=margin,
            tol=tol,
            translated_prefactor=translated_prefactor,
            mixture=_material_mixture(
                mode_set if general is None else (), medium, density
            ),
            regional_responses=None if general is None else general.responses,
            regional_dispersive=None if general is None else general.dispersive,
            reuse_prefactor=None if reuse is None else reuse.prefactor_expansion,
            reuse_raman=None if reuse is None else reuse.raman_undamped,
        )
    # The frame above was chosen from the declared responses; the operator it wrote is
    # only right if the compiled pass agrees on whether it carries the harmonic.
    _check_harmonic_frame(harmonic, compiled)
    # SEP 6, section 4.10: classified once, at translation, on the whole active band;
    # a field reaching beyond a gas's support is logged here and never refused.
    dispersive_regions(responses, grid, vector=transform.n_pol * n_pol > 1)
    initial = _power_initial(
        Initial(
            field0=state,
            noise0=None if noise is None else noise_spectrum(noise, state.shape),
            noise_model="none" if noise is None else noise.model,
            z=initial_z,
        ),
        linear,
        prefactor_index,
        None if reuse is None else reuse.initial,
    )
    dz_max, dz_init = _crossing_clamped(design, dz_max, dz_init)
    geometry = transform.geometry
    crossing = _split_crossing(
        compiled,
        linear,
        grid,
        initial.field0,
        None
        if geometry is None or geometry.kind != "radial"
        else np.asarray(geometry.spectral_weights, dtype=np.float64),
    )
    dz_max, dz_init = _split_clamped(crossing, schedule, dz_max, dz_init)
    settings = _stepper_settings(
        linear,
        tableau=tableau,
        controller=controller,
        dz_min=default_dz_min(z_max) if dz_min is None else dz_min,
        dz_max=dz_max,
        dz_init=dz_init,
        stretch=DEFAULT_STRETCH if stretch is None else stretch,
        schedule=schedule,
        save_planes=planes,
        propagator_mode=propagator_mode,
        integrator=integrator,
        interval_cap=interval_cap,
        auto_init=auto_init,
        interpolate_planes=interpolate_planes,
        step_planes=step_planes,
    )
    problem = PropagationProblem(
        grid_kind=grid.kind,
        shape=(n_modes, n_pol, n_omega),
        flags=Flags.from_content(
            (n_modes, n_pol, n_omega), linear, transform, compiled, initial
        ),
        grids=build_masks(
            grid,
            polarisation_taper=polarisation_taper,
            region=getattr(settings.controller, "region", "active"),
            harmonic_carrier=compiled.harmonic_carrier,
        ),
        linear=linear,
        responses=compiled,
        initial=initial,
        route=_route(
            transform,
            tdse=None
            if tdse is None
            else TDSESource(descriptor=tdse, current_numerator=current_numerator(grid)),
            flux=flux,
            transverse=design,
        ),
        split_crossing=crossing,
    )
    # The run module builds on this one, for the stepper section.
    from solprop.translate.run_request import describe_run  # noqa: PLC0415

    statistics = StatsSpec() if stats_spec is None else stats_spec
    request = describe_run(settings, stats=statistics)
    _validated(problem, request)
    # After validation: a schedule the stability or transverse checks refuse names them.
    _refuse_split_crossing(crossing, schedule)
    logger.debug(
        "kernel problem: %s grid, shape %s, %d Phi intervals (%.3f MB), %d save "
        "planes, %d hard kinks, flags %s",
        problem.grid_kind,
        problem.shape,
        linear.n_intervals,
        linear.nbytes / 2**20,
        planes.size,
        linear.hard_kinks.size,
        problem.flags,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("%s", debug_summary(problem, statistics))
    return Translation(problem, request)


def _route(
    transform: Transform,
    *,
    tdse: TDSESource | None,
    flux: FluxPlan | None,
    transverse: SpatialAbsorberDesign | None,
) -> Route:
    """Return the route variant an assembled transform and its leaves describe.

    - ``transform``: the assembled transform: a spatial geometry makes a spatial route,
      general modal tables a general one, and anything else a modal one.
    """
    geometry = transform.geometry
    if (geometry is None) != (flux is None):
        msg = (
            "a spatial problem requires its plane measure (flux), and a problem "
            "without spatial geometry carries none"
        )
        raise TranslationError(msg)
    if tdse is not None and (
        geometry is not None or transform.modal_tables is not None
    ):
        raise TranslationError("TDSE requires the scalar resolved modal Adams route")
    if geometry is not None:
        assert flux is not None
        return SpatialRoute(
            kind=spatial_kind(geometry),
            transform=transform,
            flux=flux,
            transverse=transverse,
        )
    if transverse is not None:
        raise TranslationError("a transverse boundary belongs to a spatial route")
    if transform.modal_tables is not None:
        return GeneralRoute(transform=transform)
    return ModalRoute(transform=transform, tdse=tdse)


def _power_initial(
    initial: Initial,
    linear: LinearOperator,
    index: float | None,
    reused: Initial | None,
) -> Initial:
    """Convert a fresh true-power launch, retaining reuse only for identical weights.

    The entrance weight comes from the assembled row rather than from a second
    evaluation of the mode, so that ``power_amplitude(initial.field0,
    power_weight_row(linear, 0.0))`` returns the requested launch to one rounding of the
    divide and multiply, and so that re-translating the same input reproduces the same
    array bit for bit. A row built by an adaptive Chebyshev fit differs from a direct
    ``neff`` call at the last few bits, and the two conventions disagreeing at the
    entrance was measured at 3e-15 of the launch.

    - ``initial``: fresh true-power launch and noise in canonical shape.
    - ``linear``: its modal power-weight row is the entrance weight; its absence marks
      spatial geometry, which carries no modal conversion.
    - ``index``: constant index override, or None for the physical effective index.
    - ``reused``: previously scaled launch retained only when its arrays match.
    - Returns: internal vacuum-normalised launch and noise.
    """
    if linear.power_weight_representation is not None and index != 1.0:
        root_weight = np.sqrt(power_weight_row(linear, initial.z))
        # The quotients are the problem's own, held without a second copy.
        initial = replace(
            initial,
            field0=hold(initial.field0 / root_weight),
            noise0=None
            if initial.noise0 is None
            else hold(initial.noise0 / root_weight),
        )
    if (
        reused is not None
        and initial.z == reused.z
        and np.array_equal(initial.field0, reused.field0)
        and (
            (initial.noise0 is None and reused.noise0 is None)
            or (
                initial.noise0 is not None
                and reused.noise0 is not None
                and np.array_equal(initial.noise0, reused.noise0)
            )
        )
    ):
        return reused
    return initial


def _density(
    modes: Sequence[Mode], density: ZProfile | float | None
) -> ZProfile | float:
    """Choose the density path a compiled response multiplies its coefficient by.

    - ``density``: the caller's choice, or ``None`` to take the mode's.
    - Returns: the density, or 1.0 when neither the caller nor the mode states one.
    """
    if density is not None:
        return density
    declared = getattr(modes[0], "density", None)
    return 1.0 if declared is None else declared


def _components(
    components: Sequence[str] | None,
    quadrature: TransverseQuadrature | None,
    modes: Sequence[Mode],
) -> tuple[str, ...]:
    """Resolve and validate the transverse components a transform resolves.

    - ``components``: what the caller asked for, or ``None`` to resolve the mode set's
      components.
    - ``quadrature``: the fixed rule, or ``None`` for the mode-averaged case, which has
      no components to resolve.
    - ``modes``: modes whose continuous profiles declare the nonzero components.
    """
    if components is None:
        if quadrature is None:
            return DEFAULT_COMPONENTS
        principal = {mode.principal_component(0.0) for mode in modes}
        return tuple(
            name for name in _COMPONENT_NAMES if None in principal or name in principal
        )
    if quadrature is None:
        msg = (
            f"components={tuple(components)!r} was given without a transverse "
            "quadrature; the mode-averaged transform is one scalar factor and resolves "
            "no components"
        )
        raise TranslationError(msg)
    resolved = tuple(components)
    if (
        not resolved
        or len(set(resolved)) != len(resolved)
        or any(name not in _COMPONENT_NAMES for name in resolved)
    ):
        msg = (
            f"components must be a non-empty subset of {_COMPONENT_NAMES}, got "
            f"{resolved!r}"
        )
        raise TranslationError(msg)
    return resolved


def _transform(
    modes: Sequence[Mode],
    grid: TimeGrid,
    quadrature: TransverseQuadrature | None,
    *,
    z_max: float,
    n_pol: int,
    components: Sequence[str] | None,
    margin: float,
    tol: float,
) -> Transform:
    """Build the mode-averaged or the multimode transform, as the problem asks.

    - ``quadrature``: the fixed rule, or ``None`` for the mode-averaged case.
    - ``z_max``: length the run covers, m.
    - ``components``: the transverse components the rule resolves, or ``None`` for the
      union of the modes' nonzero components at the entrance (:func:`_components`).
    """
    # A trait, not a class: any future non-degenerate partner pair (a tabulated
    # pair, Zeisberger) declares it and is refused here without naming a class.
    if n_pol > 1 and any(
        not getattr(mode, "polarisation_degenerate", True) for mode in modes
    ):
        raise TranslationError(
            "these modes' polarisations have distinct effective indices; use "
            "separate x/y mode rows with one state component per row"
        )
    resolved = _components(components, quadrature, modes)
    if quadrature is not None:
        if len(resolved) * n_pol > MAX_N_POL:
            msg = (
                f"a transform resolving {resolved!r} and a state with {n_pol} "
                "polarisation components would carry the transverse components twice; "
                "resolve them in the transform and give the state one component "
                "(a polarisation-resolved mode set, "
                "solprop.modes.collection.ModeCollection.with_polarisations), or "
                "resolve one component and let the state carry the pair"
            )
            raise TranslationError(msg)
        for component in resolved:
            if all(
                mode.principal_component(0.0) not in (None, component) for mode in modes
            ):
                descriptions = ", ".join(
                    f"{mode!r} (phi={getattr(mode, 'phi', 'not declared')})"
                    for mode in modes
                )
                raise TranslationError(
                    f"component {component!r} is identically zero for every mode: "
                    f"{descriptions}"
                )
        return multimode_transform(
            modes,
            quadrature,
            grid,
            z_max=z_max,
            components=resolved,
            margin=margin,
            tol=tol,
        )
    if len(modes) != 1:
        msg = (
            f"{len(modes)} modes were given without a transverse quadrature; a "
            "multimode problem needs one, and a mode-averaged problem needs one mode"
        )
        raise TranslationError(msg)
    return mode_averaged_transform(modes[0], grid, z_max=z_max, margin=margin, tol=tol)
