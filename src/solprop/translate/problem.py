# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel contract: the one object that crosses into a backend.

:class:`KernelProblem` is a frozen dataclass of plain NumPy arrays and scalars,
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
import json
import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
from jaxtyping import Bool, Complex, Float

from solprop.adiabatic import AdiabaticAccuracy, certify_radial_path
from solprop.boundaries import (
    STABILITY_RADIUS,
    Absorber,
    TransverseCollar,
    stable_step,
    transverse_collar_rate,
)
from solprop.constants import EPS0, C
from solprop.exceptions import (
    ExponentBudgetWarning,
    GrazingIncidenceWarning,
    MaterialRangeWarning,
    SolpropWarning,
    TranslationError,
)
from solprop.fourier import GridKind, n_omega_for
from solprop.grid import Collar, FreeGrid, RadialGrid, TimeGrid
from solprop.linear import DEFAULT_LOSS_POLICY, Frame, LossPolicy
from solprop.modes.capillary import (
    CoherentWall,
    MarcatiliMode,
    VectorWallOperator,
    WallImpedance,
    WallLimits,
    WallOperator,
    _shared_sampling,
    taper_connection,
    taper_coupling_block,
    vector_wall_operator,
    wall_connection,
    wall_operator,
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
from solprop.spatial import FAR_FIELD_STATISTICS, SpatialDispersion, SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper.controller import PIController, default_dz_min
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.stepper.tableau import DOPRI5, Tableau, continuous_extension
from solprop.translate.chebyshev import DEFAULT_TOL, PiecewiseChebyshev
from solprop.translate.coupled import (
    CF4_GROWTH_LIMIT,
    CoupledFlowTables,
    advisory_step,
    cf4_defect,
    cf4_flow,
    constant_schur_flow,
    flow_edge_slack,
    frozen_residual_flow,
    residual_truncation,
    section_expansion,
)
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.flux import flux_plan
from solprop.translate.free import free_operators, free_transform
from solprop.translate.general import (
    GeneralModalRoute,
    check_noise_basis,
)
from solprop.translate.linop import (
    DEFAULT_MARGIN,
    LinearOperator,
    build_phi,
    delta_phi,
    power_weight_row,
    with_power_weight,
)
from solprop.translate.masks import (
    DEFAULT_ABSORBER_SCHEME,
    AbsorberScheme,
    Grids,
    build_masks,
    noise_spectrum,
)
from solprop.translate.modal import (
    BUFFERS_PER_NODE,
    DEFAULT_NODE_BUDGET,
    DEFAULT_SPATIAL_NODE_BUDGET,
    Transform,
    general_modal_transform,
    mode_averaged_transform,
    multimode_transform,
)
from solprop.translate.planar import planar_operators, planar_transform
from solprop.translate.plasma_source import plasma_source_record
from solprop.translate.precision import DOUBLE, Precision
from solprop.translate.radial import (
    radial_operators,
    radial_rate_bounds,
    radial_transform,
)
from solprop.translate.responses import (
    CompiledResponses,
    ResponseStage,
    compile_responses,
    dispersive_regions,
    response_kinks,
    unlanded_kinks,
)
from solprop.translate.spatial import (
    FluxPlan,
    RateBounds,
    SeparableOperator,
    rate_bounds,
)
from solprop.translate.tdse import TDSEDescriptor
from solprop.translate.tdse_source import (
    STEADY_CURRENT_CERTIFICATE_LIMIT,
    TDSE_UNAVAILABLE_STATISTICS,
    admit_tdse,
    current_numerator,
    steady_current_certificate,
)

__all__ = [
    "DEFAULT_COMPONENTS",
    "DEFAULT_FLOW_CONDITION_LIMIT",
    "DEFAULT_STRETCH",
    "EXPONENT_BUDGET",
    "KERNEL_SCHEMA_VERSION",
    "LOG_DBL_MAX",
    "LOG_FLT_MAX",
    "MAX_GENERAL_FLOW_EXPONENT",
    "MAX_N_COMPONENTS",
    "MAX_N_POL",
    "MOVING_FLOW_CERTIFICATE",
    "TAPER_FLOW_CERTIFICATE",
    "Flags",
    "Initial",
    "KernelProblem",
    "NoiseModel",
    "PropagatorMode",
    "StepperSettings",
    "TranslationReuse",
    "assemble",
    "debug_summary",
    "dispersive_node_buffers",
    "validate_coupled_flow",
]

logger = logging.getLogger(__name__)

KERNEL_SCHEMA_VERSION = 39
"""Version of the kernel contract this module implements.

Incremented on every change to the field list of ``docs/dev/kernel-contract.md``. A
backend refuses a problem whose version it does not know. Version 2 replaced the
``prefactor`` array with a table along z and added ``prefactor_constant`` beside it, so
that the effective index's z profile is carried rather than frozen at the entrance.
Version 3 replaced the z tables of the operator and the prefactor with expansions in
the scalar parameters they depend on, the z table staying behind the same evaluation
functions, moved the harmonic carrier into the grids as ``carrier_2w0``, and gathered
the Raman rows into ``raman_coefficients``.

Version 4 adds spatial geometry, transverse integration and absorber data, and the
constant separable spatial operator/prefactor in the problem and stage bundles.

Version 5 adds required material and transverse square residuals to the separable
record, resolving Cartesian cutoff arithmetic from unsquared binary inputs.

Version 6 adds mixture channels and their stage bundles. Version 7 adds the three
periodic statistics storage hooks to StepDriver, allowing device rows to be copied
once per plane. Problem fields, stage bundles and numerical conventions are unchanged.
Version 8 adds the batch protocol, independent member observations and metadata, and
the backend free-device-memory query for admission of fixed-schedule fibre batches.

The number also covers the seam a compiled backend subclasses and
packs from: the protected surface of :class:`solprop.backends.driver.StepDriver` a
subclass overrides or reads, and the fields of
:class:`solprop.translate.stage.StageTerms` and
:class:`solprop.translate.stage.StageScalars`. ``tests/backends/test_driver.py``
holds that surface as a literal, so a change to it fails there until this number is
bumped and the literal rewritten in the same commit. A separately built backend
refuses a problem of another version and looks the members it relies on up by name; it
never compares solprop's own version.

Version 9 adds envelope plasma cycle data, labelled channels and explicit spectral
assembly, and retains rate conventions and species bindings in rate tables. Field
plasma channels and stage scalar field names remain unchanged; the population
scalar tuples cover whichever plasma representation is active.

Version 10 adds the explicit numeric Raman tensor weights and the
per-component capability declaration; scalar Raman retains its operation order.
Version 11 adds the modal power-weight row (see ``docs/dev/decisions.md``) in its three
representations and the endpoint block of a cumulative parameter expansion; the
driver surface and the stage bundles are unchanged and every backend consumes the
weight in its statistics pass alone.

Version 12 adds the four optional leaves of the general
modal route -- ``linear.coupled_flow``, ``linear.modal_metric``,
``transform.modal_tables`` and ``responses.regional_responses`` -- each ``None`` by
default, and admits ``int32`` across the seam for the region and support indices they
carry. A problem with all four ``None`` is the version-11 problem, bit for bit, and
allocates no new buffer. The driver surface and the stage bundles are unchanged.

Version 13 adds the two coupled-flow rows of the stage
bundle -- ``flow_section``, which section of ``linear.coupled_flow`` the step's anchor
lies in, and ``flow_weights``, the weights its ``(n_intervals, n_deg)`` coefficient
block is combined over at the stage's position -- so that a varying guide's frozen
generator and explicit residual are evaluated from host scalars rather than from a
Clenshaw pass over a leaf. Nothing else moves: no ``LinearOperator`` row, no
``StepDriver`` signature, no dtype. A problem carrying no coupled flow reports
``flow_section = -1`` and ``flow_weights = None`` and is the version-12 problem bit for
bit. ``numpy_ref``, ``numba_cpu`` and ``native`` execute both ``"frozen_residual"`` and
``"constant_schur"`` sections. The ``"cf4"`` sections move nothing either:
the Gauss nodes of a CF4 step are not stage positions, and ``numpy_ref`` evaluates the
leaf's own expansion at them from ``z_edges``, the degree axis and the step's
``flow_section``, which the leaf and the bundle already carry.

Version 14 adds one row of the stage bundle,
``transform_weights``: the weights a **moving** guide's ``transform.modal_tables``
synthesis and projection maps are combined over at the stage's position, their
``(n_intervals, n_deg)`` coefficient axes flattened, separate from ``flow_weights``
so that the transform's sectioning is its own. A problem whose maps do not move
reports ``None`` and is the version-13 problem bit for bit; nothing else moves.

Version 15 adds the hybrid planar geometry: the
``SpatialGeometry.kind`` ``"slab"`` (slab modes in x on the transform's own
synthesis and projection, an FFT in y, columns ``(p, k_y)`` with the mode index
fastest) and one appended ``SeparableOperator`` field, ``kappa_sq``, each channel's
complex, frequency-dependent wall term subtracted from the column's longitudinal
square. It is ``None`` on every other kind, whose problems are the version-14
problems bit for bit. No ``KernelProblem`` field, stage bundle or ``StepDriver``
signature moves. ``numpy_ref``, ``numba_cpu`` and native CPU execute the kind.

Version 16 appends ``StepperSettings.integrator``: SEP 13's low-call Adams method
stepping the problem instead of the tableau, or ``None`` for the Runge-Kutta march,
whose problems are the version-15 problems bit for bit. Only ``numpy_ref`` executes an
integrator; ``numba_cpu`` and ``native`` refuse it by name, and the driver's settings
refuse it before translation on any other backend. No ``StepDriver`` signature moves:
the Adams march is the reference backend's own ``_march_to``.

Version 17 appends ``StepperSettings.interval_cap`` and ``auto_init`` (SEP 14 phase
1): whether the march caps a step at the width of the operator-table interval it
starts in, and whether the start-up rule takes the step after the first from that
step's own error estimate. ``assemble`` resolves both from the route: off and on on
the diagonal-flow routes, on and off for a coupled flow or a reject-free controller. A
problem with the cap on and the rule off is the version-16 problem bit for bit. Every
backend inherits the shared march, so none moves; Kokkos follows the version only.

Version 18 appends ``StepperSettings.interpolate_planes`` (SEP 14 phase 3): ``True``
observes a save plane inside an accepted step -- through DOPRI5's continuous extension,
or the Adams march's own polynomial -- instead of landing on it; ``False`` (the
version-17 problem bit for bit) lands. It replaces SEP 13's ``Multistep.dense``. The
shared march does it with one more ``StepDriver`` hook, ``_dense_stages``, the accepted
step's stage derivatives on the host. It also appends ``StepperSettings.step_planes``:
a positive ``k`` stops ``propagate_to`` short of its target at the end of every
``k``-th accepted step, so the driver saves the march's own steps as planes; ``0``
(the version-17 march) never stops short.

Version 20 (SEP 16 phase 2) gives ``StepperSettings.integrator`` a ``formulation``
field, ``"interaction"`` (the version-19 method bit for bit) or ``"exponential"``, and
adds one protected ``StepDriver`` hook, ``_adams_lab_source``: one source evaluation
returned in the lab frame, untransported, which the exponential formulation integrates
against the linear flow. A backend without the hook refuses the exponential
formulation by name; no ``KernelProblem`` field or stage bundle moves.

Version 21 (SEP 16 phase 3b) adds the protected hook ``_adams_engine``: a backend may
return an engine that keeps the Adams history and every state-sized operation of the
fused and exponential paths where its state lives
(``solprop.backends.multistep.AdamsEngine``); ``None``, the default, keeps the host
engine over the other Adams hooks, the version-20 arithmetic bit for bit. No
``KernelProblem`` field or stage bundle moves.

Version 22 (SEP 16 phase 5) adds the protected hook ``_adams_block``: a backend that
holds a coupled flow returns the frozen block of the section a step is anchored in,
whose matrix moments the exponential formulation's trial takes. ``None``, the default,
is a problem without one. No ``KernelProblem`` field or stage bundle moves.
Version 23 adds ``grids.norm_mask``, the bins the error norm scores: the controller's
``region``, by default every active bin rather than the requested band alone, which
``grids.inband_mask`` still carries.

Version 24 (SEP 6 phase 1) appends ``responses.dispersive_kerr``, the
frequency-dependent electronic response's leaf
(``translate.responses.DispersiveKerrTerms``): its evaluator and sector, the fine-grid
multipliers of each population's channel and their density tables, beside the
``dispersive_kerr`` stage of ``response_order``. It appends one row to the stage
bundle, ``StageScalars.dispersive_kerr_densities``, each channel's number
density at the stage. ``grids.carrier_2w0`` is also present for the dispersive full
cubic on an envelope grid. A problem without the leaf is the version-23 problem bit for
bit, its new bundle row empty. On the general modal route the leaf's ``regions`` names
each channel's node region, a gas region's species. ``numpy_ref`` executes the leaf on
every route; ``numba_cpu`` and the native backend refuse it by name.

Version 25 (SEP 18 phase 1) appends ``responses.plasma_integral``, the running-integral
rule of the field-grid plasma column
(:class:`solprop.nonlinear.ionisation.RunningIntegral`): one per problem, because the
channels share the two current integrals. The ``"trapezoid"`` rule is the version-24
column bit for bit. The ``"band"`` rule filters each of the three integrands with the
grid's symmetric taps before the trapezoid, closes the window's ends by point
reflection and floors the exposure at zero, so ``plasma_scan`` and ``plasma_current``
read ``K`` samples ahead of the one they finalise. ``numpy_ref``, ``numba_cpu`` (SEP
18 phase 2) and native CPU (phase 3) execute both rules; a native device build refuses
``"band"`` by name. No stage bundle or ``StepDriver`` hook moves.

Version 26 (SEP 19 phase 1) appends ``transform.kind``, what the transform represents,
stored instead of inferred from the node count: ``"mode_averaged"`` for the
effective-area route alone, ``"modal"`` for every resolved quadrature of any node count
and for the general modal tables, and the spatial geometry's kind otherwise. A one-node
quadrature is therefore an ordinary modal transform, with no area conversion in its
prefactor. Every existing problem keeps its arithmetic bit for bit.

Version 27 (SEP 20 phase 1) widens what ``tdse`` may carry without changing any
field: a ``solprop.tdse-descriptor.v3`` descriptor names a haCC model and its pinned
setup record, and may carry an atomic window whose gate, read through
``TDSEDescriptor.source_gate()``, replaces ``grids.t_window_fine`` as the
``time_collar`` source gate. A version-26 consumer would misread such a descriptor,
so the version moves; version-1 and version-2 descriptors are unchanged bit for bit.

Version 28 (SEP 25) moves no field: ``StepDriver._stage_bundles``, a trial's stage
bundles read together, joins the driver surface a separately built backend is built
against, so a version-27 consumer refuses the problem.

Version 29 (SEP 24 phase 1) widens ``StepperSettings.integrator`` to
``Multistep | LowStorageRungeKutta | None``: the low-storage interaction-picture RK4
march, which replays a schedule in three registers and records no estimate. It adds the
protected hook ``StepDriver._low_storage_engine`` and the driver attributes
``_low_storage`` and ``_in_flight`` to the surface, and ``_stage_bundles`` keeps the
bundles of every anchor it is asked for until the next hoist (the march reads three).
A backend that does not execute the march refuses it by name, because a widened field
would otherwise read as no integrator and run DOPRI5. Every problem without it keeps
its arithmetic bit for bit.

Version 30 (SEP 26 phase 1) moves no field: ``tdse`` may carry a
``solprop.tdse-descriptor.v4`` descriptor, the ``gated_derivative`` source filter of an
SAE or haCC response, windowed or on the grid collar. Its source subtracts the gate's
edge term, ``F[g' S] / (i omega)``, using ``TDSEDescriptor.source_gate_derivative()``,
so a version-29 consumer would misread it; translation admits it only when its
steady-current certificate is at most 1e-10 (SEP 26's further fixed 128-sample taper
minimum was later dropped as a defect, with no field change).

Version 31 (SEP 22 phase 2) appends ``SeparableOperator.kperp_rate``, a transverse
collar's power rate per column, which joins the field attenuation on the active band
(``D -= kperp_rate / 2`` there); zeros, the default, are the version-30 operator to
the last bit. ``StepDriver.observe`` gains the keyword ``field``: ``False`` observes
the step history and statistics' position without crossing the field or the noise
phase to the host, and :class:`solprop.backends.protocol.Observation` carries
``field=None``. ``StepDriver.held_observation`` returns the state a low-storage march
held at its last field-free plane. ``SeparableOperator.evanescent_rate`` may now be
set by the run (``SimulationSettings.evanescent_rate``).

Version 32 (SEP 24 phase 3) moves no field of the problem: solprop-native executes the
low-storage march with a resident engine, so an older wheel, which refuses it, must be
refused in turn. :class:`solprop.backends.low_storage.LowStorageEngine` gains
``atomic``, whether its step is one native call no signal interrupts (the march then
takes the step, the finish and the move in one commit), and ``StepDriver`` gains
``_hold`` and ``_held_field`` on its surface, so a backend may hold the field-free
plane's state where its own state lives.

Within version 32 the source filter was removed with descriptor versions 1 to 4:
every ``tdse`` descriptor is now ``solprop.tdse-descriptor.v5``, and a native
descriptor's source is always gated and edge-free (``TDSEDescriptor.gated``). Removing
values no consumer could misread needs no bump.

Version 33 (SEP 27) adds ``grids.t_window_fine_derivative`` and changes the field
plasma column's outputs to the two products ``g J`` and ``g' J`` alongside collared
non-plasma polarisation. Route assembly combines their spectra with ``1/(i omega)``
and ``-1/(i omega)**2`` only on positive active bins.

Version 34 (SEP 29 phase 2) moves no field of the problem: ``StepDriver`` gains the
class attribute ``helpers``, the backend's implementation of the per-stage Chebyshev
rows and the Adams basis, which the shared driver, march and engines pass down
(``solprop._helpers``). An older native build does not set it and would run solprop's
Numba helpers, so it is refused.

Version 35 (SEP 34) adds ``flux``, the spatial run's plane-measure coefficients
(:class:`solprop.translate.spatial.FluxPlan`): every spatial statistic is the axial
Poynting flux of the medium at its plane, weighted by the admittance
``c k_z / omega`` of that plane's root, with the complex admittance for local flux.
A radial gas path's cumulative operator gains the forward amplitude term
``-ln(Y_z(z)/Y_z(0))/2``. A problem without spatial geometry carries no plan.

Version 36 (SEP 33) adds ``precision``, the arithmetic the run is held in
(:class:`solprop.translate.precision.Precision`). Every table stays binary64 at the
seam; a backend that admits single precision narrows its tables on upload, and every
other backend refuses a single-precision problem by name.

Version 37 (SEP 35) gives ``StepperSettings.integrator`` a ``carrier_frame`` field,
``"off"`` (the version-36 method bit for bit) or ``"scalar"`` (a ``"harmonic"`` value
existed during SEP 35's development and was removed by its gate): the carrier phase the
exponential formulation removes from its history inside each step. A backend that does
not deliver the shift refuses it by name, and a build against an older version refuses
the problem rather than step a requested shift unshifted; no ``KernelProblem`` field or
stage bundle moves.

Version 38 adds two members to the driver's protected surface: the boundary monitor's
reading moves into the backend. ``StepDriver._bind_monitor`` hands the backend a run's
monitor (or ``None``), and ``StepDriver._boundary_energies`` returns the monitor's five
numbers of a state and what a split-step collar took since the last reading
(:class:`solprop.monitor.BoundaryEnergies`), so a device backend reduces where its state
lives; no ``KernelProblem`` field or stage bundle moves.

Version 39 (SEP 39) lets a backend run the exponential Adams march natively: the
march's policy is a record (:class:`solprop.stepper.multistep.MarchPolicy`), every
number its algorithm reads, and the driver's protected surface gains
``StepDriver._record_attempt`` and ``StepDriver._skip_step``, the bookkeeping of one
attempt and of an accepted step on which the driver has no work, beside the frozen
``_take_step_stats``, ``_natural_stop`` and ``_natural_stop_due``; no ``KernelProblem``
field or stage bundle moves.
"""

DEFAULT_COMPONENTS = ("x",)
"""Canonical component of an unrotated HE_1m mode.

The component an HE_1m Marcatili mode points along at ``phi = 0`` (see
``docs/dev/decisions.md``). Assembly derives its actual default from the mode set's
``principal_component`` traits, so a rotated or vector set is resolved correctly.
"""

_COMPONENT_NAMES = ("x", "y")

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
``numba_cpu``, where both signs come from one real ``exp`` and a ``sincos`` in the
stage that formed them, so the table buys residency and nothing else -- 561 us against
574 and 128 KiB against 768 on the soliton case (``docs/dev/backends.md``
section 3.6). Either can still be asked for by name; the two
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
    single_batch : bool
        One batch member: ``n_batch == 1``.
    has_plasma : bool
        A plasma response is compiled.
    has_raman : bool
        A Raman kernel is compiled.
    has_spectral_responses : bool
        Responses are evaluated after the forward transform.
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
    >>> flags = Flags(True, True, True, False, False, False, False, False, True, True)
    >>> flags.constant_linop, flags.has_raman
    (True, False)
    """

    single_mode: bool
    scalar_pol: bool
    single_batch: bool
    has_plasma: bool
    has_raman: bool
    has_spectral_responses: bool
    has_noise: bool
    has_temporal_absorber: bool
    constant_linop: bool
    mode_averaged: bool
    n_channels: int = 0
    n_raman: int = 0

    @classmethod
    def from_content(
        cls,
        shape: tuple[int, int, int, int],
        linear: LinearOperator,
        transform: Transform,
        responses: CompiledResponses,
        initial: "Initial",  # noqa: UP037 - bound after this class body
    ) -> "Flags":  # noqa: UP037 - beartype resolves it while the body still runs
        """Derive the flags from the problem's own content.

        Parameters
        ----------
        shape : tuple of int
            The canonical state shape ``(n_batch, n_modes, n_pol, n_omega)``.
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
        >>> problem = assemble(
        ...     grid, gnlse.mode, [gnlse.kerr], start,
        ...     controller=PIController(rtol=1e-8, atol=1e-14),
        ...     save_planes=[1.0], dz_max=0.1, dz_init=1e-3,
        ...     shock=False, prefactor_index=1.0, density=1.0,
        ... )
        >>> problem.flags.mode_averaged, problem.flags.constant_linop
        (True, True)
        """
        return cls(
            single_mode=shape[1] == 1,
            scalar_pol=shape[2] == 1,
            single_batch=shape[0] == 1,
            has_plasma=bool(responses.population_channels),
            has_raman=bool(responses.raman_terms),
            has_spectral_responses=bool(responses.spectral_responses),
            has_noise=initial.noise0 is not None,
            has_temporal_absorber=responses.temporal_absorber is not None,
            constant_linop=linear.phi_representation in ("constant", "separable"),
            mode_averaged=transform.mode_averaged,
            n_channels=len(responses.population_channels),
            n_raman=len(responses.raman_terms),
        )


@dataclass(frozen=True, eq=False)
class Initial:
    """The initial state, the noise realisation and the batch parameters.

    Section 6 of the kernel contract. Both arrays are in the lab frame and the FFT
    convention, read-only and C-contiguous.

    Attributes
    ----------
    field0 : numpy.ndarray
        Initial state, complex128, shape ``(n_batch, n_modes, n_pol, n_omega)``, in the
        square root of watts.
    noise0 : numpy.ndarray or None
        The shot-noise realisation in the same layout, or ``None``. It is carried along
        z by the unitary part of the linear flow and added at every stage; the
        propagating field never contains it.
    noise_model : {'none', 'input', 'modified'}
        Which model the realisation belongs to.
    batch_params : mapping
        Per-member scalars recorded in the output, each a read-only ``(n_batch,)``
        float64 array. Empty unless a batch runner filled it (work package 16). The
        annotation gives the length its own axis name so that ``__post_init__`` is what
        checks it against the state, which keeps the check reachable in production.
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
    >>> state = np.zeros((1, 1, 1, 8), dtype=complex)
    >>> Initial(state, None, "none", {}).noise_model
    'none'
    """

    field0: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    noise0: Complex[np.ndarray, "n_batch_n n_modes_n n_pol_n n_omega_n"] | None
    noise_model: NoiseModel
    batch_params: Mapping[str, Float[np.ndarray, " n_params"]]
    z: float = 0.0

    def __post_init__(self) -> None:
        """Adopt read-only copies and check the noise against the state.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        if not math.isfinite(self.z) or self.z < 0.0:
            raise TranslationError("initial.z must be finite and non-negative")
        state = np.ascontiguousarray(self.field0, dtype=np.complex128)
        state.setflags(write=False)
        object.__setattr__(self, "field0", state)
        if self.noise0 is not None:
            noise = np.ascontiguousarray(self.noise0, dtype=np.complex128)
            noise.setflags(write=False)
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
        parameters = {}
        for name, values in self.batch_params.items():
            entry = _readonly(np.asarray(values, dtype=np.float64).reshape(-1))
            if entry.size != state.shape[0]:
                msg = (
                    f"batch_params[{name!r}] has {entry.size} entries, expected "
                    f"{state.shape[0]}, one per batch member"
                )
                raise TranslationError(msg)
            parameters[name] = entry
        object.__setattr__(self, "batch_params", MappingProxyType(parameters))


_ADAMS_CONTROL = frozenset({"rtol", "atol", "weights", "region"})
"""The controller fields the Adams march reads. It chooses its steps by its own
``safety``, ``min_factor`` and ``max_growth``
(:class:`~solprop.stepper.multistep.Multistep`)."""


def _refuse_unread_control(controller: PIController) -> None:
    """Refuse a controller setting the Adams march would ignore, by name.

    Parameters
    ----------
    controller : solprop.stepper.controller.PIController
        The run's controller.

    Raises
    ------
    TranslationError
        Naming each field set away from its default that the Adams march does not
        read.
    """
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
        planes = _readonly(np.asarray(self.save_planes, dtype=np.float64).reshape(-1))
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

        Parameters
        ----------
        method : solprop.stepper.low_storage.LowStorageRungeKutta
            The march asked for.

        Raises
        ------
        TranslationError
            Naming the setting and its remedy.
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
    def z_end(self) -> float:
        """Return the last save plane, where the run stops.

        Returns
        -------
        float
            The largest save plane, m.

        Examples
        --------
        >>> from solprop.stepper import DOPRI5, PIController
        >>> from solprop.translate.problem import StepperSettings
        >>> StepperSettings(DOPRI5, PIController(rtol=1e-8, atol=1e-14), 1e-9, 0.1,
        ...                 1e-3, 1.1, None, [0.5, 1.0], "tabulate").z_end
        1.0
        """
        return float(np.max(self.save_planes))


def _readonly(values: np.ndarray) -> np.ndarray:
    """Return a C-contiguous, read-only view or copy of an array.

    Parameters
    ----------
    values : numpy.ndarray
        The array to freeze.

    Returns
    -------
    numpy.ndarray
        The same values, C-contiguous and not writeable.
    """
    frozen = np.ascontiguousarray(values)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, eq=False)
class KernelProblem:
    """Everything a backend needs, and nothing it does not.

    The seam of ``docs/dev/architecture.md`` section 3. Construct it with
    :func:`assemble` rather than by hand; call :meth:`validate` before handing it to a
    backend, which :func:`assemble` already does.

    Attributes
    ----------
    schema_version : int
        The contract version, :data:`KERNEL_SCHEMA_VERSION`.
    grid_kind : {'field', 'envelope'}
        Selects r2c/c2r against c2c transforms and the response formulas.
    shape : tuple of int
        The canonical state shape ``(n_batch, n_modes, n_pol, n_omega)``.
    flags : Flags
        Static booleans, derived from the content (section 1).
    grids : solprop.translate.masks.Grids
        Frequency axis, masks and oversampling data (section 2).
    linear : solprop.translate.linop.LinearOperator
        The cumulative linear operator (section 3).
    transform : solprop.translate.modal.Transform
        Synthesis and projection (section 4).
    responses : solprop.translate.responses.CompiledResponses
        The compiled nonlinear term (section 5).
    initial : Initial
        Initial state, noise and batch parameters (section 6).
    stepper : StepperSettings
        Tableau, controller, caps and schedule (section 7).
    stats_spec : solprop.stats.StatsSpec
        Which statistics to collect (section 8).
    tdse : solprop.translate.tdse.TDSEDescriptor or None, optional
        Immutable atomic and response-lattice scientific identity. Execution plans
        remain owned by the driver and cross the seam as an explicit binding.
    current_numerator : numpy.ndarray or None, optional
        Independent current source numerator, ``-crop_scale/(2 epsilon_0 c)``,
        shape ``(n_modes, n_pol, n_omega)``. Zero outside the retained band.
        Divide by the stage's real effective index without dividing by frequency.
    flux : solprop.translate.spatial.FluxPlan or None, optional
        A spatial run's plane-measure coefficients (version 35); required with
        spatial geometry and absent otherwise.
    precision : solprop.translate.precision.Precision, optional
        The arithmetic the run is held in (version 36): binary64 by default, or
        binary32 under SEP 33's contract. Tables are binary64 here either way.

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
    >>> problem = assemble(
    ...     grid, gnlse.mode, [gnlse.kerr], start,
    ...     controller=PIController(rtol=1e-8, atol=1e-14), save_planes=[1.0],
    ...     dz_max=0.1, dz_init=1e-3, shock=False, prefactor_index=1.0, density=1.0,
    ... )
    >>> problem.shape, problem.grid_kind, problem.schema_version
    ((1, 1, 1, 3072), 'envelope', 39)
    """

    schema_version: int
    grid_kind: GridKind
    shape: tuple[int, int, int, int]
    flags: Flags
    grids: Grids
    linear: LinearOperator
    transform: Transform
    responses: CompiledResponses
    initial: Initial
    stepper: StepperSettings
    stats_spec: StatsSpec
    tdse: TDSEDescriptor | None = None
    current_numerator: Float[np.ndarray, "n_modes n_pol n_omega"] | None = None
    flux: FluxPlan | None = None
    precision: Precision = DOUBLE

    @property
    def device_ready(self) -> bool:
        """Return whether both z-varying arrays are in the seam's device form.

        True when ``Phi`` and the prefactor each evaluate as a weighted sum of a fixed
        set of arrays over host scalars -- the constant fast path with one term and the
        distance as its weight, or a parameter expansion with ``n_terms`` of them
        (:mod:`solprop.translate.stage`). False when either took the z table, whose
        evaluation is a Clenshaw pass over an array of the state's size times the
        degree: the general representation, which the CPU backends evaluate (the
        native CPU build into term slots of its own, SEP 29 section 6.6) and an
        accelerator build refuses through
        :func:`solprop.translate.stage.require_device_ready`.

        Returns
        -------
        bool
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
        ...          prefactor_index=1.0, density=1.0).device_ready
        True
        """
        return (
            self.linear.phi_representation != "table"
            and self.responses.prefactor_representation != "table"
        )

    def validate(self) -> None:
        """Check every promise the contract makes, raising on the first that fails.

        A backend calls this and refuses anything inconsistent
        (``docs/dev/architecture.md`` section 7). What is checked: the schema version;
        every array finite, C-contiguous, read-only and of the stated dtype and shape;
        the masks Boolean; the breakpoints strictly increasing and covering the run
        including its margin; the hard kinks interval edges; every table along z
        spanning the same range and, for the prefactor, carrying the state's value
        shape; a constant prefactor, when present, standing in for a table that is one
        interval of degree zero and agreeing with it; the Raman rows agreeing with the
        fine grid and with each other; the save planes and, when one is replayed, the
        schedule's edges containing every plane and kink; and a replayed schedule
        or a step floor inside the stability limit of an in-stage collar.

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
        >>> problem = assemble(
        ...     grid, gnlse.mode, [gnlse.kerr], start,
        ...     controller=PIController(rtol=1e-8, atol=1e-14), save_planes=[1.0],
        ...     dz_max=0.1, dz_init=1e-3, shock=False, prefactor_index=1.0,
        ...     density=1.0,
        ... )
        >>> problem.validate() is None
        True
        """
        if self.schema_version != KERNEL_SCHEMA_VERSION:
            msg = (
                f"schema_version {self.schema_version} is not the kernel schema "
                f"{KERNEL_SCHEMA_VERSION} this translation layer writes"
            )
            raise TranslationError(msg)
        if (self.tdse is None) != (self.current_numerator is None):
            raise TranslationError("TDSE descriptor and current numerator must coexist")
        self._check_integrator()
        self._check_precision()
        if self.tdse is not None:
            self._check_tdse()
            unavailable = TDSE_UNAVAILABLE_STATISTICS.intersection(
                (*self.stats_spec.names, *self.stats_spec.periodic)
            )
            if unavailable:
                raise TranslationError(
                    f"TDSE does not define legacy statistics {sorted(unavailable)}; "
                    "request explicitly labelled atomic diagnostics"
                )
        if self.current_numerator is not None:
            values = self.current_numerator
            if (
                values.shape != self.shape[1:]
                or values.dtype != np.float64
                or values.flags.writeable
                or not values.flags.c_contiguous
                or not np.all(np.isfinite(values))
                or np.any(values > 0.0)
            ):
                raise TranslationError(
                    "current numerator must be a frozen finite source row"
                )
        if not self.initial.z < self.stepper.save_planes[0]:
            raise TranslationError("save planes must follow the initial position")
        self._check_arrays()
        self._check_shapes()
        self._check_power_weight()
        self._check_spatial()
        self._check_carrier()
        self._check_species()
        self._check_plasma()
        self._check_envelope_factors()
        self._check_raman()
        self._check_dispersive()
        self._check_modal_leaves()
        self._check_tables()

    def validate_launch(self) -> None:
        """Check the launch against the save planes, and a schedule against them.

        The part of :meth:`validate` a replaced input with a new launch position
        needs again (:meth:`solprop.driver.simulation.Simulation.with_input`): every
        other leaf is shared, unchanged, with a problem that was validated whole.

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
        >>> simulation.translate().validate_launch() is None
        True
        """
        if not self.initial.z < self.stepper.save_planes[0]:
            raise TranslationError("save planes must follow the initial position")
        if self.stepper.schedule is not None:
            self._check_schedule()

    def _check_integrator(self) -> None:
        """Refuse what the low-storage march does not step, by name (SEP 24 section 5).

        Its half-step flows are exact exponentials of commuting diagonal values, so a
        coupled flow, whose flows are not, would leave an error outside the
        Runge-Kutta order; the TDSE response is the Adams route's alone.

        Raises
        ------
        TranslationError
            Naming the configuration and the integrator that steps it.
        """
        method = self.stepper.integrator
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
        if self.tdse is not None:
            msg = (
                f"the {method.name} march does not step the TDSE response, which "
                "runs on the Adams route alone; use Multistep()"
            )
            raise TranslationError(msg)

    def _check_precision(self) -> None:
        """Refuse, by name, what no backend runs in single precision (SEP 33 5.1).

        The general modal route (a moving guide among it), the hybrid planar kind,
        coupled flows (``"cf4"`` sections among them), TDSE responses, the dispersive
        electronic response, the far-field statistics and noise models are outside
        SEP 33's scope, so every backend refuses them under ``precision="single"``.

        Raises
        ------
        TranslationError
            Naming the refused route, response or statistic.
        """
        if not isinstance(
            self.precision, Precision
        ):  # pragma: no cover - refused by the annotation where it is checked
            raise TranslationError("precision must be a Precision record")
        if not self.precision.single:
            return
        geometry = self.transform.geometry
        refused = [
            (self.transform.modal_tables is not None, "the general modal route"),
            (
                geometry is not None and geometry.kind == "slab",
                "the hybrid planar kind",
            ),
            (self.linear.coupled_flow is not None, "a coupled flow"),
            (self.tdse is not None, "a TDSE response"),
            (
                self.responses.dispersive_kerr is not None,
                "the dispersive electronic response",
            ),
            (
                bool(
                    set(FAR_FIELD_STATISTICS).intersection(
                        (*self.stats_spec.names, *self.stats_spec.periodic)
                    )
                ),
                "the far-field statistics",
            ),
            (
                self.flags.has_noise or self.initial.noise_model != "none",
                "a noise model (a known limitation of single precision)",
            ),
        ]
        for refuse, name in refused:
            if refuse:
                msg = (
                    f"precision='single' refuses {name}: no backend runs it in "
                    "binary32 (SEP 33 section 5.1); use precision='double'"
                )
                raise TranslationError(msg)
        self._check_single_exponent()

    def _check_single_exponent(self) -> None:
        """Refuse a step whose propagator or its inverse overflows binary32.

        The interaction picture multiplies each stage by ``exp(-dPhi)`` and the state
        by ``exp(dPhi)``; held in binary32, either is infinite once ``|Re dPhi|``
        passes ``log(FLT_MAX)``, about 88.72, where binary64 reaches 709.78. A strong
        spectral collar on a long step is the usual cause. As
        :meth:`_check_exponent_budget` does for binary64, a replayed schedule's
        longest step, or the floor ``dz_min``, is held to the limit -- an adaptive
        controller rejects a non-finite trial above its floor and shrinks -- and every
        kind is checked: the exponent is evaluated from the operator itself over that
        step at the start and the end of the run, which for a varying operator samples
        its range.

        Raises
        ------
        TranslationError
            Past ``log(FLT_MAX)``, naming the exponent, the step and the remedies.

        Warns
        -----
        solprop.exceptions.ExponentBudgetWarning
            Between nine tenths of the limit and the limit.
        """
        schedule = self.stepper.schedule
        step, source = (
            (float(np.max(schedule.steps)), "replayed step")
            if schedule is not None
            else (self.stepper.dz_min, "step floor dz_min")
        )
        start = float(self.initial.z)
        end = float(self.stepper.save_planes[-1])
        step = min(step, end - start)
        exponent = 0.0
        for first in {start, end - step}:
            phase = delta_phi(self.linear, first, first + step)
            exponent = max(exponent, float(np.max(np.abs(np.real(phase)))))
        if exponent <= EXPONENT_BUDGET * LOG_FLT_MAX:
            return
        message = (
            f"precision='single': the {source} {step:.3e} m gives a propagator "
            f"exponent |Re dPhi| of {exponent:.1f}"
        )
        remedy = (
            "take steps shorter than "
            f"{step * EXPONENT_BUDGET * LOG_FLT_MAX / exponent:.3e} m, weaken the "
            "collar with a larger boundary_length, or use precision='double'"
        )
        if exponent > LOG_FLT_MAX:
            msg = (
                f"{message}, past binary32's range exp({LOG_FLT_MAX:.2f}): the "
                f"interaction picture's inverse propagator overflows; {remedy}"
            )
            raise TranslationError(msg)
        warnings.warn(
            f"{message}, within a tenth of binary32's range exp({LOG_FLT_MAX:.2f}): "
            f"the run completes only while the field stays small; {remedy}",
            ExponentBudgetWarning,
            stacklevel=4,
        )

    def _check_tdse(self) -> None:
        """Refuse a manually constructed atomic seam outside the admitted route."""
        descriptor = self.tdse
        assert descriptor is not None
        method = self.stepper.integrator
        if (
            self.grid_kind != "field"
            or self.shape[:3] != (1, 1, 1)
            or self.transform.geometry is not None
            or self.transform.modal_tables is not None
            or self.transform.n_pol != 1
            or self.flags.mode_averaged
            # The input draw is part of the launch and drives the atom with it; the
            # modified one is kept apart and is not (SEP 37, section 6).
            or self.initial.noise_model == "modified"
            or self.flags.has_plasma
            or self.flags.has_raman
            or not isinstance(method, Multistep)
            or self.stepper.interpolate_planes
            or self.stepper.schedule is not None
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
        if self.current_numerator is None or not np.array_equal(
            self.current_numerator, expected[None, None, :]
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

    def _check_arrays(self) -> None:
        """Check every stored array is finite, contiguous, read-only and typed.

        Raises
        ------
        TranslationError
            Naming the field that failed.
        """
        for name, values in _walk_arrays(self):
            if not values.flags.c_contiguous:
                msg = f"{name} is not C-contiguous; a kernel indexes it directly"
                raise TranslationError(msg)
            if values.flags.writeable:
                msg = f"{name} is writeable; arrays crossing the seam are frozen"
                raise TranslationError(msg)
            allowed = _SEAM_DTYPES + ((np.int32,) if name in _INT32_LEAVES else ())
            if values.dtype not in allowed:
                msg = (
                    f"{name} has dtype {values.dtype}; the contract carries float64, "
                    "complex128, bool and intp, and int32 only at the index leaves "
                    "it declares"
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
        """Check every array agrees with the canonical state shape.

        Raises
        ------
        TranslationError
            Naming the field that failed.
        """
        n_batch, n_modes, n_pol, n_omega = self.shape
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
        value = self.linear.value_shape
        allowed = ((1, n_modes, n_pol, n_omega), (n_batch, n_modes, n_pol, n_omega))
        if value not in allowed:
            msg = (
                f"linear carries values of shape {value}, expected "
                f"{allowed[0]} or {allowed[1]}"
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
        if value.value_shape != self.shape[1:]:
            msg = (
                f"power_weight shape {value.value_shape} "
                f"does not match {self.shape[1:]}"
            )
            raise TranslationError(msg)
        for z in value.z_span:
            if np.any(value(z) <= 0):
                msg = "power_weight must be positive"
                raise TranslationError(msg)
        if constant is not None and (
            table is None
            or table.degrees.tolist() != [0]
            or constant.shape != self.shape[1:]
            or not np.array_equal(table(0.0), constant)
        ):
            msg = "power_weight_constant must match its degree-zero table"
            raise TranslationError(msg)

    def _check_flux(self) -> None:
        """Check a spatial problem carries its plane measure on the run's band."""
        if (self.transform.geometry is None) != (self.flux is None):
            msg = (
                "a spatial problem requires its plane measure (flux), and a problem "
                "without spatial geometry carries none"
            )
            raise TranslationError(msg)
        if self.flux is not None and not np.array_equal(
            self.flux.base.active, self.grids.active_mask
        ):
            msg = "the plane measure's active mask must match the temporal grid"
            raise TranslationError(msg)

    def _check_spatial(self) -> None:
        """Check spatial geometry and the shared separable representation agree."""
        geometry = self.transform.geometry
        separable = self.linear.separable
        if self.responses.separable is not separable:
            msg = "linear and responses must share the same separable coefficients"
            raise TranslationError(msg)
        self._check_flux()
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
            if self.stepper.propagator_mode == "tabulate" and not isinstance(
                self.stepper.integrator, LowStorageRungeKutta
            ):
                msg = (
                    f"{geometry.kind} propagation refuses propagator_mode='tabulate' "
                    "except under LowStorageRungeKutta(); use 'recompute'"
                )
                raise TranslationError(msg)

    def _check_slab(self, separable: SeparableOperator) -> None:
        """Check the hybrid planar operator against its geometry and transform.

        Parameters
        ----------
        separable : solprop.translate.spatial.SeparableOperator
            The shared coefficients, which must carry one wall term per channel.

        Raises
        ------
        TranslationError
            If the wall terms are absent, their channels are not the transform's,
            or a column's ``kperp_sq`` is not its ``k_y**2`` with the channel fastest.
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

        Raises
        ------
        TranslationError
            If the representation is not degenerate, or does not agree with the array.
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

    def _check_collar_stability(self) -> None:
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

        Raises
        ------
        TranslationError
            Naming the limit, the offending step and the scheme that lifts it.
        """
        geometry = self.transform.geometry
        if geometry is not None and np.any(geometry.absorber_rate):
            self._check_spatial_collar_stability()
            return
        rate = self.responses.temporal_absorber
        if rate is None or self.responses.absorber_scheme != "nonlinear":
            return
        radius = self._stability_radius()
        limit = stable_step(rate) * radius / STABILITY_RADIUS
        schedule = self.stepper.schedule
        longest, source = (
            (float(np.max(schedule.steps)), "the replayed schedule's longest step")
            if schedule is not None
            else (self.stepper.dz_min, "the step floor dz_min")
        )
        if longest <= limit:
            return
        msg = (
            f"{source} is {longest:.3e} m and the temporal collar is stable in the "
            f"nonlinear absorber scheme only up to {limit:.3e} m: the collar reaches "
            f"{float(np.max(rate)):.3e} 1/m, and an explicit method needs "
            f"h alpha / 2 below {radius}, so the run would diverge. "
            "Take shorter steps, weaken the collar with a larger boundary_length, "
            f"{self._runge_kutta_advice()}or "
            'ask for absorber_scheme="split_step", which integrates the rate exactly '
            "and is unconditionally stable at the cost of first-order convergence."
        )
        raise TranslationError(msg)

    def _check_exponent_budget(self) -> None:
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

        Raises
        ------
        TranslationError
            Naming the binding term, its value, the factor and the admissible step
            and rate.

        Warns
        -----
        solprop.exceptions.ExponentBudgetWarning
            Between nine tenths of the limit and the limit.
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
        schedule = self.stepper.schedule
        step, source = (
            (float(np.max(schedule.steps)), "replayed step")
            if schedule is not None
            else (self.stepper.dz_min, "step floor dz_min")
        )
        method = self.stepper.integrator
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
            stacklevel=4,
        )

    def _runge_kutta_advice(self) -> str:
        """Name the Runge-Kutta march where an Adams method's interval is the limit.

        Returns
        -------
        str
            A clause ending in ", " for an Adams run, otherwise empty.
        """
        if not isinstance(self.stepper.integrator, Multistep):
            return ""
        return (
            f"step with integrator=RungeKutta(), whose interval is {STABILITY_RADIUS}, "
        )

    def _stability_radius(self) -> float:
        """Return the stepper's negative-real stability interval for a damping term.

        DOPRI5's :data:`solprop.boundaries.STABILITY_RADIUS`, the Adams method's far
        smaller one at its highest order (SEP 16 admits Adams schedules), or RK4's
        2.785 under the low-storage march (SEP 24 section 4.6).

        Returns
        -------
        float
            The largest ``h |lambda|`` an explicitly integrated damping rate allows.
        """
        method = self.stepper.integrator
        if method is None:
            return STABILITY_RADIUS
        return min(STABILITY_RADIUS, method.real_stability_interval)

    def _check_spatial_collar_stability(self) -> None:
        """Refuse a step past the spatial collar's limit in the energy-weighted basis.

        Raises
        ------
        TranslationError
            When a replayed schedule's longest step, or the step floor, exceeds
            the limit.
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
        limit = 2 * self._stability_radius() / largest
        schedule = self.stepper.schedule
        longest = (
            self.stepper.dz_min if schedule is None else float(np.max(schedule.steps))
        )
        if longest > limit:
            msg = (
                f"the fixed step or step floor {longest:.3e} m exceeds the spatial "
                f"collar stability limit {limit:.3e} m (effective power rate at most "
                f"{largest:.3e} 1/m). Take shorter steps, "
                f"{self._runge_kutta_advice()}or increase boundary_length; the "
                "spatial absorber is always in-stage and cannot use split_step."
            )
            raise TranslationError(msg)

    def _check_carrier(self) -> None:
        """Check the harmonic carrier against the Kerr term that needs it.

        Raises
        ------
        TranslationError
            If the carrier is present without a harmonic term or absent with one.
        """
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
        if self.transform.n_pol * self.shape[2] > 1 and not all(
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

        Raises
        ------
        TranslationError
            If its arrays are not on the fine grid's spectral layout, a channel names
            an unknown population or region, its carrier flag disagrees with the grid
            and the sector, or a density table does not cover the run.
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

    def _check_modal_leaves(self) -> None:
        """Check the four schema-12 general-guide leaves against the state.

        The contract's tree walk *does* recurse into these records: every array they
        hold is checked for dtype, finiteness, contiguity and ownership like any
        other leaf. What the walk does not do is relate any of them to the canonical
        state, and that is this method's work -- the channel counts, the component
        count, the frequency axes and the positions along z.
        Every leaf is ``None`` on every route that exists today, and the whole
        method is then a single test of four references.

        Raises
        ------
        TranslationError
            Naming the leaf that disagrees with the state.
        """
        _, n_modes, n_pol, _ = self.shape
        tables = self.transform.modal_tables
        metric = self.linear.modal_metric
        flow = self.linear.coupled_flow
        regional = self.responses.regional_responses
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
        self._check_modal_grids()

    def _check_modal_grids(self) -> None:
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

        Raises
        ------
        TranslationError
            Naming the leaf whose axis is not the state's.
        """
        omega = self.grids.omega
        z_end = self.stepper.z_end
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
            if low > 0.0 or high < z_end:
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

    def _check_tables(self) -> None:
        """Check the z tables cover the run, and the schedule contains its targets.

        Raises
        ------
        TranslationError
            Naming the field that failed.
        """
        edges = self.linear.phi_breakpoints
        if not np.all(np.diff(edges) > 0.0):
            msg = f"linear.phi_breakpoints must be strictly increasing, got {edges}"
            raise TranslationError(msg)
        low, high = self.linear.z_span
        z_end = self.stepper.z_end
        if low > 0.0 or high < z_end:
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
        if self.stepper.schedule is not None:
            self._check_schedule()
        self._check_collar_stability()
        self._check_exponent_budget()

    def _check_schedule(self) -> None:
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

        Raises
        ------
        TranslationError
            If a target is not an edge of the schedule, or if a coupled flow's kink
            is straddled rather than landed on.
        """
        schedule = self.stepper.schedule
        assert schedule is not None
        edges = schedule.edges
        scale = max(schedule.total_length, 1.0)
        flow = self.linear.coupled_flow
        flow_slack = None if flow is None else flow_edge_slack(flow)
        for name, targets in (
            ("save plane", self.stepper.save_planes),
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


_EDGE_SLACK = 1e-9
_SPAN_SLACK = 1e-12

LOG_DBL_MAX = math.log(np.finfo(np.float64).max)
"""``log(DBL_MAX)``, about 709.78: the largest exponent ``exp`` returns finite.

IEEE Std 754-2019, doi:10.1109/IEEESTD.2019.8766229. A propagator factor
``exp(h lambda)`` past it is infinite on every backend (SEP 22 section 4.5).
"""

LOG_FLT_MAX = math.log(float(np.finfo(np.float32).max))
"""``log(FLT_MAX)``, about 88.72: binary32's largest finite ``exp`` (SEP 33)."""

EXPONENT_BUDGET = 0.9
"""Fraction of :data:`LOG_DBL_MAX` above which translation warns of a step it cannot
shrink: such a run completes only while its field stays below ``exp`` of the margin.
"""


def _budget_term(
    bounds: RateBounds, span: int, attenuation: str
) -> tuple[str, float, float | None]:
    """Return the exponent budget's binding term: its name, value and what remains.

    Parameters
    ----------
    bounds : solprop.translate.spatial.RateBounds
        The operator's bounds.
    span : int
        Steps a gain factor spans.
    attenuation : str
        The name of the `propagating` term, which no setting moves.

    Returns
    -------
    tuple
        The term's description, its rate in m^-1, and the rest of the loss bound
        beside it (``None`` for a term no setting moves).
    """
    if span * bounds.gain > bounds.loss:
        return f"the medium's gain over {span} step(s)", span * bounds.gain, None
    terms = (
        ("evanescent_rate", bounds.evanescent, 1.0),
        ("transverse_absorber strength", bounds.transverse, 2.0),
        ("spectral absorber rate", bounds.spectral, 2.0),
        (attenuation, bounds.propagating, 0.0),
    )
    name, value, scale = max(terms, key=lambda term: term[1])
    if scale == 0.0:
        return name, value, None
    return name, scale * value, bounds.loss - value


def _budget_message(
    source: str,
    step: float,
    bounds: RateBounds,
    span: int,
    exponent: float,
    attenuation: str,
) -> str:
    """Describe the factor a step forms, naming the binding term.

    Parameters
    ----------
    source : str
        Which step: the replayed schedule's or the floor.
    step : float
        The step, m.
    bounds : solprop.translate.spatial.RateBounds
        The operator's bounds.
    span : int
        Steps a gain factor spans.
    exponent : float
        The factor's exponent.
    attenuation : str
        The name of the operator's own attenuation term.

    Returns
    -------
    str
        The opening of the message.
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

    Parameters
    ----------
    step : float
        The step, m.
    bounds : solprop.translate.spatial.RateBounds
        The operator's bounds.
    span : int
        Steps a gain factor spans.
    limit : float
        The exponent to stay within.
    attenuation : str
        The name of the operator's own attenuation term.

    Returns
    -------
    str
        The remedy.
    """
    name, value, rest = _budget_term(bounds, span, attenuation)
    rate = max(bounds.loss, span * bounds.gain)
    remedy = f"take steps of at most {limit / rate:.3g} m"
    if rest is not None:
        scale = 1.0 if name == "evanescent_rate" else 2.0
        admissible = scale * (limit / step - rest)
        if 0 < admissible < value:
            remedy += f" or {name} at most {admissible:.3g} 1/m"
    return remedy


_GENERAL_INDEX = 1.0
"""Index override the general modal route builds its operator and weight with.

A general guide's basis is Poynting-normalised by construction, so its amplitude is
already the true-power amplitude every route carries and the statistics power
weight is the constant one: the legacy ``sqrt(Re n_eff)`` boundary is not
crossed on this route. The same override removes the cumulative normalisation term
from ``Phi``, which is the term that renormalises a vacuum amplitude along a varying
index and has no meaning for a basis that carries its own flux metric.
"""


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
    node_budget: int
    has_noise: bool
    dz_max: float
    reuse: "TranslationReuse | None"  # noqa: UP037 - bound after this class body
    kinks: tuple[tuple[float, str], ...] = ()
    taper_coupling: bool = True
    evanescent_rate: float | None = None
    transverse_absorber: TransverseCollar | None = None
    adiabatic_accuracy: AdiabaticAccuracy | None = None
    noise_model: NoiseModel = "none"


def _check_operator_settings(
    *,
    spatial: bool,
    evanescent_rate: float | None,
    transverse_absorber: TransverseCollar | None,
) -> None:
    """Refuse the separable-operator settings where they cannot apply.

    Parameters
    ----------
    spatial : bool
        Whether the run has a spatial grid.
    evanescent_rate : float or None
        The requested evanescent cap, m^-1.
    transverse_absorber : solprop.boundaries.TransverseCollar or None
        The requested transverse collar.

    Raises
    ------
    TranslationError
        If either is set on a run without a spatial grid, or the rate is not
        positive and finite.
    """
    if not spatial and (evanescent_rate is not None or transverse_absorber is not None):
        msg = (
            "evanescent_rate and transverse_absorber set the spatial operator, and "
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

    Parameters
    ----------
    medium : solprop.spatial.SpatialMedium or None
        The run's medium; None for a modal run, which has none.
    planes : numpy.ndarray
        The requested save positions in m.

    Raises
    ------
    TranslationError
        If a plane lies exactly on a declared jump.
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
]:
    """Assemble the operator, the transform, the prefactor and the plane measure.

    Parameters
    ----------
    spatial_grid : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The transverse geometry; a :class:`solprop.modes.slab.PlanarGrid` is the
        hybrid planar transform (slab modes in x, an FFT in y).
    medium : solprop.spatial.SpatialMedium
        Material and density path.
    grid : solprop.grid.TimeGrid
        The run's temporal grid.
    frame : solprop.linear.Frame
        The co-moving frame.
    build : _Assembly
        The assembly's knobs.

    Returns
    -------
    tuple
        The operator, the transform, the already translated prefactor and the
        run's plane measure.
    """
    dispersion = (
        SpatialDispersion(medium, grid, frame)
        if build.evanescent_rate is None
        else SpatialDispersion(
            medium, grid, frame, evanescent_rate=build.evanescent_rate
        )
    )
    collar = build.transverse_absorber
    if collar is not None and not isinstance(spatial_grid, FreeGrid):
        msg = (
            "transverse_absorber acts on a periodic Cartesian grid (FreeGrid or "
            f"LineGrid), not on {type(spatial_grid).__name__}: the radial kind is not "
            "periodic, and the hybrid planar kind's k_y-only collar is not offered "
            "(SEP 22, Q5)"
        )
        raise TranslationError(msg)
    reuse = build.reuse
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
                absorber=build.absorber,
                node_budget=build.node_budget,
                margin=build.margin,
            )
        )
        return (
            linear,
            transform,
            prefactor,
            flux_plan(spatial_grid, dispersion, linear, tol=build.tol),
        )
    if isinstance(spatial_grid, FreeGrid):
        linear, prefactor = free_operators(
            spatial_grid,
            dispersion,
            z_max=build.z_max,
            n_pol=build.n_pol,
            absorber=build.absorber,
            margin=build.margin,
            transverse_rate=None
            if collar is None
            else transverse_collar_rate(spatial_grid, collar),
        )
        transform = (
            reuse.transform
            if reuse is not None
            else free_transform(
                spatial_grid,
                grid,
                z_max=build.z_max,
                n_pol=build.n_pol,
                absorber=build.absorber,
                node_budget=build.node_budget,
                margin=build.margin,
            )
        )
        return (
            linear,
            transform,
            prefactor,
            flux_plan(spatial_grid, dispersion, linear, tol=build.tol),
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
            n_pol=build.n_pol,
            absorber=build.absorber,
            node_budget=build.node_budget,
            margin=build.margin,
        )
    )
    return (
        linear,
        transform,
        prefactor,
        flux_plan(spatial_grid, dispersion, linear, tol=build.tol, admission=admission),
    )


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

    Parameters
    ----------
    route : solprop.translate.general.GeneralModalRoute
        The guide's leaves and its Gram certificate.
    grid : solprop.grid.TimeGrid
        The run's field grid.
    modes : sequence of Mode
        The guide's channels in state order.
    frame : solprop.linear.Frame
        The one shared co-moving frame.
    build : _Assembly
        The assembly's knobs, including the legacy ones it refuses.

    Returns
    -------
    tuple
        The operator carrying the identity metric and the unit power weight, the
        transform carrying the tables, and the prefactor row.

    Raises
    ------
    TranslationError
        Naming what cannot be combined with a general modal route.
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
    linear = with_power_weight(
        build_phi(
            carried,
            grid,
            frame,
            z_max=build.z_max,
            n_pol=build.n_pol,
            absorber_rate=None
            if build.absorber is None
            else build.absorber.spectral_rate,
            loss=build.loss,
            margin=build.margin,
            tol=build.tol,
            reuse=None if build.reuse is None else build.reuse.phi_expansion,
            index=_GENERAL_INDEX,
        ),
        modes,
        grid,
        _GENERAL_INDEX,
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

    Parameters
    ----------
    route : solprop.translate.general.GeneralModalRoute
        The route, whose gauge identity the flow carries so that a resume comparing
        the flow compares the basis it was built in.
    coupled : numpy.ndarray
        The block relative to the reference channel in 1/m, shape
        ``(n_modes, n_modes, n_omega)``.
    grid : solprop.grid.TimeGrid
        The run's grid, whose coarse axis the leaf is tabulated on.
    build : _Assembly
        The assembly's knobs, for ``dz_max`` and the run's length.

    Returns
    -------
    solprop.translate.coupled.CoupledFlowTables
        The validated leaf, one ``"constant_schur"`` section.

    Raises
    ------
    TranslationError
        If the run declares the ``'modified'`` noise model, if the block's exponent
        over ``dz_max`` exceeds :data:`MAX_GENERAL_FLOW_EXPONENT`, or on anything
        ``constant_schur_flow`` refuses.
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

    Parameters
    ----------
    build : _Assembly
        The assembly's knobs, carrying the response kinks.
    linear : LinearOperator
        The route's operator, whose hard kinks are what every step lands on.

    Returns
    -------
    numpy.ndarray
        ``linear.hard_kinks``, in m, shape ``(n_kinks,)``.

    Raises
    ------
    TranslationError
        Naming the path and the position, if a response kink strictly inside the run
        is not a hard kink: only the uncoupled modal route adds them to ``Phi``.
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
        """Return False, so that the declared kinks reach the operator's tables.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False

    @property
    def scale_invariant(self) -> bool:
        """Return False: a solid guide's modes have no scaling law.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False

    @property
    def channel_label(self) -> ChannelLabel:
        """Return the reference channel's label.

        Returns
        -------
        solprop.modes.protocol.ChannelLabel
            The entrance channel's own.
        """
        return self.entrance.channel_label

    @property
    def azimuthal_order(self) -> int:
        """Return the reference channel's azimuthal order.

        Returns
        -------
        int
            The entrance channel's own.
        """
        return self.entrance.azimuthal_order

    def principal_component(self, z: float = 0.0) -> PolarisationComponent | None:
        """Return the reference channel's principal component at the entrance.

        Parameters
        ----------
        z : float, optional
            Position in m; ignored.

        Returns
        -------
        str or None
            The entrance channel's own.
        """
        del z
        return self.entrance.principal_component()

    def neff(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the entrance index at every position.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s.
        z : float, optional
            Position in m; ignored, the carrier being the entrance's.

        Returns
        -------
        numpy.ndarray
            The entrance ``n_eff``, complex.
        """
        del z
        return self.entrance.neff(omega)

    def neff_minus_one(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the entrance ``n_eff - 1`` at every position.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s.
        z : float, optional
            Position in m; ignored.

        Returns
        -------
        numpy.ndarray
            The entrance ``n_eff - 1``, complex.
        """
        del z
        return self.entrance.neff_minus_one(omega)

    def dneff_domega(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the entrance index derivative at every position.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s.
        z : float, optional
            Position in m; ignored.

        Returns
        -------
        numpy.ndarray
            ``d n_eff / d omega`` in s/rad, complex.
        """
        del z
        return self.entrance.dneff_domega(omega)

    def aeff(self, z: float = 0.0) -> float:
        """Return the entrance surrogate area.

        Parameters
        ----------
        z : float, optional
            Position in m; ignored.

        Returns
        -------
        float
            Area in m².
        """
        del z
        return self.entrance.aeff()

    def dimlimits(
        self, z: float = 0.0
    ) -> tuple[CrossSection, tuple[float, ...], tuple[float, ...]]:
        """Return the entrance domain.

        Parameters
        ----------
        z : float, optional
            Position in m; ignored.

        Returns
        -------
        tuple
            The entrance channel's own.
        """
        del z
        return self.entrance.dimlimits()

    def field(
        self, points: Float[np.ndarray, "n_points 2"], z: float = 0.0
    ) -> Float[np.ndarray, "n_points 2"]:
        """Refuse, as the provider channel it carries does.

        Parameters
        ----------
        points : numpy.ndarray
            Cross-section positions in m.
        z : float, optional
            Position in m.

        Returns
        -------
        numpy.ndarray
            Never returns for a provider channel.
        """
        return self.entrance.field(points, z)

    @property
    def parameter_paths(self) -> Mapping[str, ZProfile]:
        """Return the one declared path: the position, kinked where the maps are.

        Returns
        -------
        collections.abc.Mapping
            ``{"position": path}``.
        """
        return MappingProxyType({"position": self.path})

    def at_parameters(self, values: Mapping[str, float]) -> Mode:
        """Return the entrance channel, whatever the position.

        Parameters
        ----------
        values : collections.abc.Mapping
            The position, by name.

        Returns
        -------
        solprop.modes.protocol.Mode
            The entrance channel.

        Raises
        ------
        TranslationError
            If a name other than ``"position"`` is given.
        """
        if set(values) - {"position"}:
            msg = f"a moving guide's carrier declares one path, got {sorted(values)}"
            raise TranslationError(msg)
        return self.entrance


def _carrier(mode: Mode, route: GeneralModalRoute) -> _PathCarrier:
    """Return the carrier of a moving guide's ``Phi`` for its reference channel.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        The reference channel, a moving provider channel.
    route : solprop.translate.general.GeneralModalRoute
        The moving route, whose guide identity names the path's sections.

    Returns
    -------
    _PathCarrier
        The carrier.
    """
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

    Parameters
    ----------
    z : numpy.ndarray
        Positions in m.

    Returns
    -------
    numpy.ndarray
        The same positions, float64.
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

    Parameters
    ----------
    route : solprop.translate.general.GeneralModalRoute
        The moving route.
    grid : solprop.grid.TimeGrid
        The run's grid.
    build : _Assembly
        The assembly's knobs, for ``dz_max`` and the noise refusal.

    Returns
    -------
    solprop.translate.coupled.CoupledFlowTables
        The validated leaf, every section ``"frozen_residual"``.

    Raises
    ------
    TranslationError
        If the run declares the ``'modified'`` noise model, or on anything the flow
        refuses.
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

    Parameters
    ----------
    wall : solprop.modes.capillary.CoherentWall or None
        The coherent wall declaration, or ``None``. Without one, a multimode
        capillary taper carries the taper coupling when ``build.taper_coupling``
        says so, and every other set -- or that one with the coupling declined --
        takes the uncoupled diagonal route.
    modes : sequence of Mode
        The mode set in state order.
    grid : solprop.grid.TimeGrid
        The run's grid.
    frame : solprop.linear.Frame
        The one shared co-moving frame.
    build : _Assembly
        The assembly's knobs, including ``dz_max`` and the prequalified reuse.

    Returns
    -------
    LinearOperator
        Section 3 of the contract, carrying ``coupled_flow`` exactly when a wall was
        declared or the taper coupling applies (:func:`_taper_assembly`).
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
        linear = build_phi(
            modes,
            grid,
            frame,
            z_max=build.z_max,
            n_pol=build.n_pol,
            absorber_rate=None
            if build.absorber is None
            else build.absorber.spectral_rate,
            loss=build.loss,
            margin=build.margin,
            tol=build.tol,
            reuse=None if build.reuse is None else build.reuse.phi_expansion,
            index=build.prefactor_index,
            kinks=[z for z, _ in build.kinks],
        )
    return with_power_weight(linear, modes, grid, build.prefactor_index)


def _wall_assembly(
    wall: CoherentWall,
    modes: Sequence[Mode],
    grid: TimeGrid,
    frame: Frame,
    build: _Assembly,
) -> LinearOperator:
    """Assemble the linear operator of a capillary carrying a coherent wall block.

    The split is the one a constant section takes: the
    **shared scalar** part of the flow stays in the cumulative diagonal ``Phi``, where
    it is integrated analytically and commutes with the whole channel block, and the
    modal part -- the paraxial transverse term and the rank-one wall together -- is
    the coupled flow. Nothing may be split between them that does not commute, and a
    mode-dependent diagonal does not: ``exp(A) exp(B)`` is ``exp(A + B)`` only when
    ``A`` is a multiple of the identity, so ``Phi`` is built over ``N`` copies of the
    **first** channel and carries no mode dependence at all.

    What is subtracted from the block is therefore exactly what ``Phi`` applies per
    metre, and the co-moving frame cancels between the two: with ``k`` the in-core
    wavenumber and ``beta_11`` the first channel's own Marcatili propagation constant,
    the stored generator is ``conj(G + i(k - beta_11) I)``, the complex conjugate of
    the **whole** optics-convention equation (``docs/dev/conventions.md`` section 2).
    Its first diagonal entry is then ``conj(i(beta_wall - beta_11))``, which is the
    difference between the block's own HE_11 and this package's Marcatili one and is
    small by a derived bound, rather than either of them twice.
    The per-channel true-power weight is unaffected: it is a row of its own and is
    built from the real mode set, so a saved amplitude still carries each channel's
    own ``sqrt(Re n_eff)``.

    Parameters
    ----------
    wall : solprop.modes.capillary.CoherentWall
        The declaration, carrying the two limits the block is admitted under.
    modes : sequence of Mode
        The contiguous ``HE_11`` to ``HE_1N`` block in state order.
    grid : solprop.grid.TimeGrid
        The run's grid; supplies the frequency axis and the active band.
    frame : solprop.linear.Frame
        The one shared co-moving frame.
    build : _Assembly
        The assembly's knobs, including the ``dz_max`` the conditioning bound is
        declared over.

    Returns
    -------
    LinearOperator
        The mode-independent carrier, the real mode set's power weight and the
        coupled flow leaf.

    Raises
    ------
    TranslationError
        If the run is not one this block is derived for, if the carrier is not the
        exact constant representation, or if the total generator is not passive.
    """
    block = tuple(modes)
    _check_wall_route(build)
    _check_wall_method(wall, cast("MarcatiliMode", block[0]))
    if wall.impedance == "te_tm":
        return _vector_wall_assembly(wall, block, grid, frame, build)
    if not cast("MarcatiliMode", block[0]).z_constant:
        return _varying_wall_assembly(wall, block, grid, frame, build)
    operator = wall_operator(
        cast("Sequence[MarcatiliMode]", block),
        grid.omega,
        z=0.0,
        active=grid.active_mask,
        scored=grid.inband_mask,
        validity_limit=wall.validity_limit,
    )
    carrier = build_phi(
        (block[0],) * len(block),
        grid,
        frame,
        z_max=build.z_max,
        n_pol=build.n_pol,
        absorber_rate=None if build.absorber is None else build.absorber.spectral_rate,
        loss=build.loss,
        margin=build.margin,
        tol=build.tol,
        index=build.prefactor_index,
    )
    # The carrier is built from N copies of one channel this function has already
    # established to be `z_constant` -- a varying one took `_varying_wall_assembly`
    # above -- and `build_phi` takes `constant_phi` for such a set whatever else it
    # is asked for, so this is an invariant of the constant route rather than a
    # condition on the run. It is asserted, not refused, because nothing can reach
    # the refusal; the varying route's own carrier is checked where it is built.
    assert carrier.phi_representation == "constant"
    assert carrier.phi_constant is not None
    linear = with_power_weight(carrier, block, grid, build.prefactor_index)
    generator = _wall_shifted_generator(
        operator.generator, cast("MarcatiliMode", block[0]), grid
    )
    _check_wall_total(generator, linear, len(block))
    flow = constant_schur_flow(
        grid.omega,
        np.asarray(carrier.phi_breakpoints, dtype=float)[[0, -1]],
        generator,
        dz_max=build.dz_max,
        condition_limit=wall.condition_limit,
        wall_vectors=operator.trace_vector,
        wall_weights=np.conj(operator.coupling),
        gauge_id="capillary HE_1m",
        certificate_id="constant-schur",
    )
    logger.debug(
        "coherent wall: %d channels, worst validity ratio %.3e (%s), largest "
        "conditioning %.3e over dz_max %.3g m, representation error %.3e",
        len(block),
        operator.limits.worst,
        operator.limits.named()[0],
        float(np.max(flow.inverse_condition_bound)),
        build.dz_max,
        float(np.max(flow.generator_error)),
    )
    return replace(linear, coupled_flow=flow)


def _check_wall_method(wall: CoherentWall, first: MarcatiliMode) -> None:
    """Refuse, by name, a CF4 declaration the block it would integrate cannot take.

    A guide that does not vary along z has one constant section, whose flow the
    constant-Schur action integrates **exactly**, so there is nothing for a
    commutator-free exponential to order and the declaration would otherwise be
    silently ignored; a declaration that would change nothing is refused rather
    than dropped. CF4 is admitted beside the vector block of ``impedance="te_tm"``
    too: it integrates the stored expansion of whichever block the route built and
    reads nothing else of it.

    Parameters
    ----------
    wall : solprop.modes.capillary.CoherentWall
        The declaration.
    first : solprop.modes.capillary.MarcatiliMode
        The block's first channel, whose profiles are every channel's.

    Raises
    ------
    TranslationError
        If CF4 is declared on a constant guide.
    """
    if wall.method != "cf4":
        return
    if first.z_constant:
        msg = (
            "CoherentWall(method='cf4') orders the exponentials of a generator that "
            "varies along z, and this guide's does not: its one constant section is "
            "integrated exactly by the constant-Schur action, which a CF4 "
            "step would only reproduce to rounding. Declare the default method, or a "
            "radius or pressure that varies"
        )
        raise TranslationError(msg)


def _vector_wall_assembly(
    wall: CoherentWall,
    block: tuple[Mode, ...],
    grid: TimeGrid,
    frame: Frame,
    build: _Assembly,
) -> LinearOperator:
    """Assemble the linear operator of a capillary carrying a vector wall block.

    The split is the scalar wall's, unchanged: the shared scalar carrier stays in the
    cumulative diagonal ``Phi``, built over ``N`` copies of the **first** channel so
    that it is a multiple of the identity and commutes with the block, and the whole
    modal part -- the paraxial transverse term and the **rank-two** wall together --
    is the coupled flow (:func:`solprop.modes.capillary.vector_wall_operator`). What
    differs is the block itself and two admissions.

    **No wall pair is stored.** The leaf's optional factored pair is one vector and
    one weight, which reconstructs a rank-**one** off-diagonal; the vector block's
    wall term is ``V W V^T`` with two columns. Storing half of it would be worse than
    storing none, so the dense generator stands alone and the constructor is given no
    pair to check.

    **A guide that varies along z** takes the frozen-residual route
    (:func:`_varying_wall_assembly`) with this block in place of the scalar one and
    the vector connection, which is block diagonal -- each family's own ``C_mn`` of
    its Bessel order, and no term between the families -- as derived and as measured
    (:func:`solprop.modes.capillary.wall_connection`).

    Parameters
    ----------
    wall : solprop.modes.capillary.CoherentWall
        The declaration, carrying ``impedance="te_tm"`` and the two limits.
    block : tuple of Mode
        One azimuthal order of the capillary, in state order.
    grid : solprop.grid.TimeGrid
        The run's grid.
    frame : solprop.linear.Frame
        The one shared co-moving frame.
    build : _Assembly
        The assembly's knobs.

    Returns
    -------
    LinearOperator
        The mode-independent carrier, the real mode set's power weight and the
        constant-Schur coupled flow of the vector block, or the varying route's flow.

    Raises
    ------
    TranslationError
        If the total generator is not passive, or on everything the varying route
        refuses.
    """
    first = cast("MarcatiliMode", block[0])
    if not first.z_constant:
        return _varying_wall_assembly(wall, block, grid, frame, build)
    operator = vector_wall_operator(
        block,
        grid.omega,
        z=0.0,
        active=grid.active_mask,
        scored=grid.inband_mask,
        validity_limit=wall.validity_limit,
    )
    carrier = build_phi(
        (first,) * len(block),
        grid,
        frame,
        z_max=build.z_max,
        n_pol=build.n_pol,
        absorber_rate=None if build.absorber is None else build.absorber.spectral_rate,
        loss=build.loss,
        margin=build.margin,
        tol=build.tol,
        index=build.prefactor_index,
    )
    # As on the constant scalar route: the carrier is N copies of a channel already
    # established to be `z_constant`, and `build_phi` takes `constant_phi` for such a
    # set whatever else it is asked for, so this is an invariant and not a condition.
    assert carrier.phi_representation == "constant"
    linear = with_power_weight(carrier, block, grid, build.prefactor_index)
    generator = _wall_shifted_generator(operator.generator, first, grid)
    _check_wall_total(generator, linear, len(block))
    flow = constant_schur_flow(
        grid.omega,
        np.asarray(carrier.phi_breakpoints, dtype=float)[[0, -1]],
        generator,
        dz_max=build.dz_max,
        condition_limit=wall.condition_limit,
        gauge_id=f"capillary azimuthal order nu={operator.sector}",
        certificate_id="constant-schur",
    )
    logger.debug(
        "coherent wall, vector: nu = %d, %s, %d channels, worst validity ratio %.3e "
        "(%s), anisotropy %.3e of the mean impedance, largest conditioning %.3e over "
        "dz_max %.3g m, representation error %.3e",
        operator.sector,
        "+".join(family.label for family in operator.families),
        len(block),
        operator.limits.worst,
        operator.limits.named()[0],
        float(
            np.max(
                abs(operator.impedance[0] - operator.impedance[1])[operator.active]
                / abs(operator.impedance[0] + operator.impedance[1])[operator.active]
            )
        ),
        float(np.max(flow.inverse_condition_bound)),
        build.dz_max,
        float(np.max(flow.generator_error)),
    )
    return replace(linear, coupled_flow=flow)


def _varying_wall_assembly(
    wall: CoherentWall,
    block: tuple[Mode, ...],
    grid: TimeGrid,
    frame: Frame,
    build: _Assembly,
) -> LinearOperator:
    """Assemble the coherent wall of a capillary whose geometry or fill varies along z.

    The **complete** generator of each section -- the paraxial transverse term, the
    rank-one wall and the carrier shift together -- is expanded in Chebyshev
    coefficients along z; the block frozen inside the exponential flow is that
    expansion at the section's midpoint, and

    ``R_s(z) = D(z) - D(z_s) + H(z) - H(z_s) - C(z)``

    is the explicit residual the interaction-picture source carries, with ``C`` the
    HE1m moving-basis connection
    (:func:`solprop.modes.capillary.wall_connection`). With ``R`` evaluated exactly,
    freezing is a rearrangement and not an approximation. The **vector** block of
    ``impedance="te_tm"`` takes the same route unchanged but for the block
    (:func:`solprop.modes.capillary.vector_wall_operator`) and its connection, which
    is each radial family's own and nothing between them.

    Two things differ from the constant section of a guide that does not vary. The
    cumulative ``Phi`` is
    no longer the exact constant representation but the **parameter expansion** of the
    lowest channel, which is still a multiple of the identity in the channel axis and
    therefore still commutes with the block -- that, and not constancy, is what the
    product of the two exponentials needs; the rest of each channel's own
    normalisation is on the generator's diagonal (:func:`_channel_normalisation_at`).
    And the section edges are the operator
    table's own breakpoints, which already land on every declared kink of the radius
    and the density, subdivided ``wall.sections`` times: a step that crosses one is
    still the same equation, because the residual is defined against whichever block
    the step's anchor names, so the edges need not be landed on exactly.

    Parameters
    ----------
    wall : solprop.modes.capillary.CoherentWall
        The declaration, carrying the two limits and the section count.
    block : tuple of Mode
        The contiguous ``HE_11`` to ``HE_1N`` set in state order, or one azimuthal
        order of the capillary for ``wall.impedance == "te_tm"``.
    grid : solprop.grid.TimeGrid
        The run's grid.
    frame : solprop.linear.Frame
        The one shared co-moving frame.
    build : _Assembly
        The assembly's knobs.

    Returns
    -------
    LinearOperator
        The mode-independent carrier, the real mode set's power weight and the
        frozen-residual coupled flow -- or, for ``wall.method == "cf4"``, the CF4 flow
        of the same expansion (:func:`solprop.translate.coupled.cf4_flow`).

    Raises
    ------
    TranslationError
        If the carrier is not mode independent, if the total generator grows, or on
        everything :func:`solprop.translate.coupled.frozen_residual_flow` or
        :func:`solprop.translate.coupled.cf4_flow` refuses.
    """
    first = cast("MarcatiliMode", block[0])
    carrier = build_phi(
        (first,) * len(block),
        grid,
        frame,
        z_max=build.z_max,
        n_pol=build.n_pol,
        absorber_rate=None if build.absorber is None else build.absorber.spectral_rate,
        loss=build.loss,
        margin=build.margin,
        tol=build.tol,
        index=build.prefactor_index,
    )
    linear = with_power_weight(carrier, block, grid, build.prefactor_index)
    edges = _section_edges(np.asarray(carrier.phi_breakpoints, dtype=float), wall)
    sampled: list[tuple[float, WallLimits]] = []
    coefficients, tail = section_expansion(
        lambda positions: np.stack(
            [
                _wall_generator_at(block, grid, float(z), sampled, wall.impedance)
                for z in positions
            ]
        ),
        edges,
        name="the coherent wall's generator",
    )
    link = None
    if not first.radius_is_constant:
        link, link_tail = section_expansion(
            lambda positions: np.stack(
                [
                    _wall_connection_at(block, grid, float(z), wall.impedance)
                    for z in positions
                ]
            ),
            edges,
            name="the coherent wall's taper connection",
        )
        tail = max(tail, link_tail)
        # One rectangular degree axis serves both: a Chebyshev series padded with
        # zeros is the same function, so whichever resolved at a lower degree is
        # lengthened rather than either being rebuilt.
        degrees = max(coefficients.shape[1], link.shape[1])
        coefficients = _padded_degrees(coefficients, degrees)
        link = _padded_degrees(link, degrees)
    # The admission is over every position the expansion was built from, and the one
    # call that carries the caller's limit is at the worst of them, so a varying guide
    # is refused or warned about once and names the geometry that did it.
    position = sampled[int(np.argmax([limits.worst for _, limits in sampled]))][0]
    operator = _wall_block_at(
        block, grid, position, wall.validity_limit, wall.impedance
    )
    gauge_id = (
        "capillary HE_1m"
        if isinstance(operator, WallOperator)
        else f"capillary azimuthal order nu={operator.sector}"
    )
    _check_wall_total_varying(coefficients, linear, edges, block, grid, tail)
    if wall.method == "cf4":
        flow = cf4_flow(
            grid.omega,
            edges,
            coefficients,
            link,
            dz_max=build.dz_max,
            condition_limit=wall.condition_limit,
            growth_limit=CF4_GROWTH_LIMIT
            if wall.growth_limit is None
            else wall.growth_limit,
            fallback=wall.fallback,
            representation_error=tail,
            gauge_id=gauge_id,
            certificate_id="cf4",
        )
    else:
        flow = frozen_residual_flow(
            grid.omega,
            edges,
            coefficients,
            link,
            dz_max=build.dz_max,
            condition_limit=wall.condition_limit,
            representation_error=tail,
            gauge_id=gauge_id,
            certificate_id="frozen-residual",
        )
    logger.debug(
        "coherent wall, varying (%s): %d channels over %d section(s), worst validity "
        "ratio %.3e (%s) at z = %.6g m, residual bound %.3e 1/m, largest conditioning "
        "%.3e over dz_max %.3g m, representation error %.3e",
        gauge_id,
        len(block),
        len(flow.section_methods),
        operator.limits.worst,
        operator.limits.named()[0],
        position,
        float(np.max(flow.residual_norm_bound)),
        float(np.max(flow.inverse_condition_bound)),
        build.dz_max,
        float(np.max(flow.generator_error)),
    )
    return replace(linear, coupled_flow=flow)


def _taper_assembly(
    block: tuple[MarcatiliMode, ...],
    grid: TimeGrid,
    frame: Frame,
    build: _Assembly,
) -> LinearOperator:
    """Assemble a multimode capillary taper's coupled flow without a wall.

    The varying coherent wall's route (:func:`_varying_wall_assembly`) with the wall
    term zero, which is the whole of the derivation: in a basis attached to the local
    radius the equation of the true-power amplitude is ``b' = (i B(z) - C(z)) b``,
    with ``B`` the channels' own propagation constants and ``C`` the HE1m connection
    (:func:`solprop.modes.capillary.taper_connection`). Without the wall's rank-one
    block the diagonal is the **modes' own** ``beta_n(z)`` -- the Marcatili index of
    whatever model and loss the channels declare, lossless or not -- rather than the
    wall block's paraxial condensation of it, so that with ``C`` dropped the flow
    propagates exactly what the diagonal route propagates. The cumulative ``Phi``
    carries the lowest channel for every channel, as beside a wall; the flow's
    generator is each channel's rate beyond it, ``conj(i(beta_n - beta_1))``, with
    each channel's normalisation completed (:func:`_channel_normalisation_at`),
    frozen per section of the carrier's own table with the variation and the
    connection in the explicit residual (:func:`frozen_residual_flow`).

    Parameters
    ----------
    block : tuple of MarcatiliMode
        The contiguous ``HE_11`` to ``HE_1N`` set on a varying radius, from
        :func:`solprop.modes.capillary.taper_coupling_block`.
    grid : solprop.grid.TimeGrid
        The run's grid.
    frame : solprop.linear.Frame
        The one shared co-moving frame.
    build : _Assembly
        The assembly's knobs.

    Returns
    -------
    LinearOperator
        The mode-independent carrier, the real mode set's power weight and the
        frozen-residual coupled flow.

    Raises
    ------
    TranslationError
        If the run carries a noise model or a second field component, if the total
        generator grows, or on everything
        :func:`solprop.translate.coupled.frozen_residual_flow` refuses.
    solprop.exceptions.ModeError
        If the channels do not declare one guide with analytic profile derivatives.
    """
    _check_taper_route(build, grid)
    # The block's own door, before the generator differentiates the profiles: it
    # names an unslope-able radius or disagreeing channels, and the opt-out.
    taper_connection(block)
    first = block[0]
    carrier = build_phi(
        (first,) * len(block),
        grid,
        frame,
        z_max=build.z_max,
        n_pol=build.n_pol,
        absorber_rate=None if build.absorber is None else build.absorber.spectral_rate,
        loss=build.loss,
        margin=build.margin,
        tol=build.tol,
        index=build.prefactor_index,
        kinks=[z for z, _ in build.kinks],
    )
    linear = with_power_weight(carrier, block, grid, build.prefactor_index)
    edges = np.asarray(carrier.phi_breakpoints, dtype=float)
    coefficients, tail = section_expansion(
        lambda positions: np.stack(
            [_taper_generator_at(block, grid, float(z)) for z in positions]
        ),
        edges,
        name="the taper coupling's generator",
    )
    link, link_tail = section_expansion(
        lambda positions: np.stack(
            [
                _true_power_connection(
                    taper_connection(block, float(z)), block, grid, float(z)
                )
                for z in positions
            ]
        ),
        edges,
        name="the taper connection",
    )
    tail = max(tail, link_tail)
    degrees = max(coefficients.shape[1], link.shape[1])
    coefficients = _padded_degrees(coefficients, degrees)
    link = _padded_degrees(link, degrees)
    _check_wall_total_varying(coefficients, linear, edges, block, grid, tail)
    try:
        flow = frozen_residual_flow(
            grid.omega,
            edges,
            coefficients,
            link,
            dz_max=build.dz_max,
            condition_limit=DEFAULT_FLOW_CONDITION_LIMIT,
            representation_error=tail,
            gauge_id="capillary HE_1m",
            certificate_id=TAPER_FLOW_CERTIFICATE,
        )
    except TranslationError as error:
        # The flow's own refusals name its section count and its limit, which are a
        # wall's declarations; the taper coupling has neither, and has the opt-out.
        msg = (
            f"{error}. This is the taper coupling of a multimode capillary, "
            "which declares no sections and no limit: cap "
            "dz_max, or pass taper_coupling=False for the adiabatic route"
        )
        raise TranslationError(msg) from error
    logger.debug(
        "taper coupling: %d channels over %d section(s), residual bound %.3e 1/m, "
        "largest conditioning %.3e over dz_max %.3g m, representation error %.3e",
        len(block),
        len(flow.section_methods),
        float(np.max(flow.residual_norm_bound)),
        float(np.max(flow.inverse_condition_bound)),
        build.dz_max,
        float(np.max(flow.generator_error)),
    )
    return replace(linear, coupled_flow=flow)


def _check_taper_route(build: _Assembly, grid: TimeGrid) -> None:
    """Refuse, by name, a run the taper coupling cannot be carried in.

    Parameters
    ----------
    build : _Assembly
        The assembly's knobs.
    grid : solprop.grid.TimeGrid
        The run's grid, which must be a field grid.

    Raises
    ------
    TranslationError
        For an envelope grid, a second field component or the ``'modified'`` noise
        model, naming the opt-out.
    """
    if grid.kind != "field":
        msg = (
            "the taper coupling of a multimode capillary is a coupled flow tabulated "
            "on an increasing frequency axis, which a field grid has and an "
            f"{grid.kind} grid, in FFT order about its carrier, does not; no coupled "
            "flow is built on one in this package. Pass envelope=False, or "
            "taper_coupling=False for the adiabatic route"
        )
        raise TranslationError(msg)
    if build.n_pol != 1:
        msg = (
            f"the taper coupling of a multimode capillary carries one field "
            f"component and this run carries n_pol = {build.n_pol}: the coupled "
            "flow's action contracts the channel axis alone. Pass "
            "taper_coupling=False for the adiabatic route"
        )
        raise TranslationError(msg)
    if build.noise_model == "modified":
        msg = (
            "the taper coupling of a multimode capillary refuses the 'modified' "
            "noise model: its draw is transported by the unitary phase of a "
            "diagonal flow, and its transport through a coupled flow is not "
            "measured in this package. Pass taper_coupling=False for the adiabatic "
            "route, whose diagonal flow carries the draw. " + _INPUT_NOISE_ADMITTED
        )
        raise TranslationError(msg)


def _taper_generator_at(
    block: tuple[MarcatiliMode, ...], grid: TimeGrid, z: float
) -> Complex[np.ndarray, "n_modes_t_out n_modes_t_in n_omega_t"]:
    """Return the taper coupling's internal-convention generator at one position.

    Diagonal: each channel's own propagation constant beyond the lowest channel's,
    which the cumulative ``Phi`` carries for every channel, conjugated as the whole
    optics-convention equation is (``docs/dev/conventions.md`` section 2), so
    ``conj(i(beta_n - beta_1))`` -- exactly zero for the lowest channel -- plus the
    rest of each channel's normalisation rate (:func:`_channel_normalisation_at`).
    The difference is formed from ``n_eff - 1`` without the round trip through unity.

    Parameters
    ----------
    block : tuple of MarcatiliMode
        The contiguous ``HE_11`` to ``HE_1N`` set.
    grid : solprop.grid.TimeGrid
        The run's grid; the generator is exactly zero off its active band.
    z : float
        Position in m.

    Returns
    -------
    numpy.ndarray
        The generator in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``.
    """
    band = grid.omega[grid.active_mask]
    reference = np.asarray(block[0].neff_minus_one(band, z))
    count = len(block)
    generator = np.zeros((count, count, grid.n_omega), dtype=np.complex128)
    for index, mode in enumerate(block[1:], start=1):
        rate = (np.asarray(mode.neff_minus_one(band, z)) - reference) * band / C
        generator[index, index, grid.active_mask] = np.conj(1j * rate)
    generator[np.arange(count), np.arange(count)] += _channel_normalisation_at(
        block, grid, z
    )
    return generator


def _section_edges(
    breakpoints: Float[np.ndarray, " n_breakpoints"], wall: CoherentWall
) -> Float[np.ndarray, " n_edges"]:
    """Return the flow's section edges: the table's own, subdivided.

    The **placement** is the operator table's, which already lands on every declared
    kink of the radius and the density and never spans one; `CoherentWall.sections` is
    the refinement inside each of those, in equal parts.

    Parameters
    ----------
    breakpoints : numpy.ndarray
        The carrier's interval edges in m, strictly increasing.
    wall : solprop.modes.capillary.CoherentWall
        The declaration, carrying the section count.

    Returns
    -------
    numpy.ndarray
        The edges in m, strictly increasing, with the same two ends.
    """
    if wall.sections == 1:
        return np.array(breakpoints, dtype=float)
    fractions = np.arange(wall.sections) / wall.sections
    inner = (
        breakpoints[:-1, None] + fractions[None, :] * np.diff(breakpoints)[:, None]
    ).reshape(-1)
    return np.concatenate([inner, breakpoints[-1:]])


def _padded_degrees(
    coefficients: Complex[np.ndarray, "n_intervals_p n_deg_p *rest_p"], degrees: int
) -> np.ndarray:
    """Return a coefficient block lengthened along its degree axis with exact zeros.

    Parameters
    ----------
    coefficients : numpy.ndarray
        The block, shape ``(n_intervals, n_deg, ...)``.
    degrees : int
        The degree axis to reach, at least the block's own.

    Returns
    -------
    numpy.ndarray
        The block, with the same values and a degree axis of `degrees`.
    """
    pad = [(0, 0)] * coefficients.ndim
    pad[1] = (0, degrees - coefficients.shape[1])
    return np.pad(coefficients, pad)


def _wall_block_at(
    block: tuple[Mode, ...],
    grid: TimeGrid,
    z: float,
    validity_limit: float,
    impedance: WallImpedance,
) -> WallOperator | VectorWallOperator:
    """Return the wall block the declared impedance names, at one position.

    Parameters
    ----------
    block : tuple of Mode
        The channels in state order.
    grid : solprop.grid.TimeGrid
        The run's grid.
    z : float
        Position in m.
    validity_limit : float
        Largest admitted validity ratio, as :class:`CoherentWall` declares it.
    impedance : str
        ``"mean"`` builds :func:`solprop.modes.capillary.wall_operator`, ``"te_tm"``
        :func:`solprop.modes.capillary.vector_wall_operator`.

    Returns
    -------
    WallOperator or VectorWallOperator
        The block, built on the grid's active band with its validity ratios taken
        over the requested band (``grid.inband_mask``).
    """
    if impedance == "mean":
        return wall_operator(
            cast("Sequence[MarcatiliMode]", block),
            grid.omega,
            z=z,
            active=grid.active_mask,
            scored=grid.inband_mask,
            validity_limit=validity_limit,
        )
    return vector_wall_operator(
        block,
        grid.omega,
        z=z,
        active=grid.active_mask,
        scored=grid.inband_mask,
        validity_limit=validity_limit,
    )


def _wall_generator_at(
    block: tuple[Mode, ...],
    grid: TimeGrid,
    z: float,
    sampled: list[tuple[float, WallLimits]],
    impedance: WallImpedance = "mean",
) -> Complex[np.ndarray, "n_modes_s_out n_modes_s_in n_omega_s"]:
    """Return the complete internal-convention wall generator at one position.

    The same split as the constant section, evaluated at `z`: the stored generator is
    ``conj(G + i(k - beta_11) I)``, so that what the cumulative ``Phi`` applies per
    metre is exactly what is subtracted here and the co-moving frame cancels between
    the two. Both the block and the reference channel are read **at this position**,
    which is the whole difference from the constant section.

    Parameters
    ----------
    block : tuple of Mode
        The contiguous ``HE_11`` to ``HE_1N`` set.
    grid : solprop.grid.TimeGrid
        The run's grid.
    z : float
        Position in m.
    sampled : list
        Collects the limits of every position sampled, for the admission that follows.
    impedance : str, optional
        Which block to sample, ``"mean"`` or ``"te_tm"``.

    Returns
    -------
    numpy.ndarray
        The generator in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``, exactly
        zero off the active band.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GrazingIncidenceWarning)
        warnings.simplefilter("ignore", MaterialRangeWarning)
        operator = _wall_block_at(block, grid, z, math.inf, impedance)
    sampled.append((z, operator.limits))
    generator = _wall_shifted_generator(
        operator.generator, cast("MarcatiliMode", block[0]), grid, z
    )
    count = len(block)
    generator[np.arange(count), np.arange(count)] += _channel_normalisation_at(
        block, grid, z
    )
    return generator


def _channel_normalisation_at(
    block: tuple[Mode, ...], grid: TimeGrid, z: float
) -> Float[np.ndarray, "n_modes_q n_omega_q"]:
    """Return each channel's normalisation rate beyond the reference channel's.

    The cumulative ``Phi`` beside a coupled block is built from the lowest channel
    alone, so the term that renormalises a vacuum amplitude along a varying index,
    ``-0.5 ln[w(z)/w(0)]`` with ``w = Re n_eff``, is channel 1's for every
    channel. Each channel's own is completed here, as the rest of its rate:
    ``-0.5 d ln(w_n/w_1)/dz = -0.5 (w_n'/w_n - w_1'/w_1)``, real, so the
    optics/internal conjugation leaves it alone, zero for channel 1 by construction
    and zero where the guide does not vary. It is part of the generator, so the
    frozen block carries it at the section midpoint, the explicit residual carries
    its variation, and CF4 integrates it with the rest. The slope is the analytic
    :meth:`solprop.modes.capillary.MarcatiliMode.dneff_dz`, never a difference.

    Parameters
    ----------
    block : tuple of Mode
        The contiguous ``HE_11`` to ``HE_1N`` set.
    grid : solprop.grid.TimeGrid
        The run's grid; the rate is exactly zero off its active band.
    z : float
        Position in m.

    Returns
    -------
    numpy.ndarray
        The rate in 1/m, shape ``(n_modes, n_omega)``.
    """
    band = grid.omega[grid.active_mask]
    logarithmic = np.stack(
        [
            np.real(cast("MarcatiliMode", mode).dneff_dz(band, z))
            / np.real(np.asarray(mode.neff(band, z)))
            for mode in block
        ]
    )
    rate = np.zeros((len(block), grid.n_omega))
    rate[:, grid.active_mask] = -0.5 * (logarithmic - logarithmic[:1])
    return rate


def _wall_shifted_generator(
    generator: Complex[np.ndarray, "n_modes_g_out n_modes_g_in n_omega_g"],
    first: MarcatiliMode,
    grid: TimeGrid,
    z: float = 0.0,
) -> Complex[np.ndarray, "n_modes_g2_out n_modes_g2_in n_omega_g2"]:
    """Return a wall block's generator in the internal convention, with its carrier.

    ``Gamma_internal = conj(i(k - beta_0 - beta_1 dw) I + G)`` and the carrier ``Phi``
    applies ``conj(i(beta_11 - beta_0 - beta_1 dw))``; the co-moving frame cancels
    between them and what is left on the diagonal is ``conj(i(k - beta_11))``, the
    difference between the block's own in-core carrier ``k = n_core omega/c`` and the
    lowest channel's Marcatili propagation constant. The conjugation is of the
    **whole** equation (``docs/dev/conventions.md`` section 2): it is not ``-G``, and
    a passive channel propagated with that would grow.

    ``k - beta_11`` is formed as ``((n_core - 1) - (n_eff - 1)) omega/c``
    (:meth:`~solprop.modes.capillary.MarcatiliMode.core_index_minus_one`,
    :meth:`~solprop.modes.capillary.MarcatiliMode.neff_minus_one`), as the taper
    coupling forms its own diagonal, and never as the difference of the two
    wavenumbers: those agree to five or six digits, so their difference would carry
    the absolute rounding of ``k`` (about 1e-9 1/m) instead of that of the
    refractivities.

    Parameters
    ----------
    generator : numpy.ndarray
        The optics-convention block in 1/m, shape ``(n_modes, n_modes, n_omega)``.
    first : solprop.modes.capillary.MarcatiliMode
        The channel the cumulative carrier was built from.
    grid : solprop.grid.TimeGrid
        The run's grid.
    z : float, optional
        Position in m both are read at.

    Returns
    -------
    numpy.ndarray
        The internal-convention generator in 1/m, same shape.
    """
    band = grid.omega[grid.active_mask]
    refractivity = np.asarray(first.core_index_minus_one(band, z)) - np.asarray(
        first.neff_minus_one(band, z)
    )
    shift = np.zeros(grid.n_omega, dtype=np.complex128)
    shift[grid.active_mask] = refractivity * band / C
    shifted = np.conj(generator)
    count = shifted.shape[0]
    shifted[np.arange(count), np.arange(count)] += np.conj(1j * shift)
    return shifted


def _wall_connection_at(
    block: tuple[Mode, ...], grid: TimeGrid, z: float, impedance: WallImpedance = "mean"
) -> Complex[np.ndarray, "n_modes_c_out n_modes_c_in n_omega_c"]:
    """Return the wall block's connection at one position, on the run's frequency axis.

    :func:`solprop.modes.capillary.wall_connection` is the connection of the
    field-normalised Bessel basis, antisymmetric in the **true-power** amplitude
    ``b = W^(1/2) a`` with ``W = diag(Re n_eff)`` -- which is where coupled local-mode
    theory puts it, and what makes a lossless taper conserve power exactly. The
    kernel's channel amplitude is ``a``, so the stored term is
    ``W^(-1/2) C W^(1/2)``, entry by entry ``sqrt(w_n / w_m) C_mn``: real, so the
    optics/internal conjugation of ``docs/dev/conventions.md`` section 2 leaves it
    alone, and frequency dependent through the weights alone. It is exactly zero
    off the active band, because the
    residual it joins is, and a bin the wall formula was never evaluated at must carry
    no term at all. The vector block's connection (``impedance="te_tm"``) is carried
    the same way: each channel's weight is its own, whichever family it is in, and
    the terms between the families are zero before the weights and after them.

    Parameters
    ----------
    block : tuple of Mode
        The contiguous ``HE_11`` to ``HE_1N`` set, or one azimuthal order.
    grid : solprop.grid.TimeGrid
        The run's grid.
    z : float
        Position in m.
    impedance : str, optional
        Which block the connection is of, ``"mean"`` or ``"te_tm"``.

    Returns
    -------
    numpy.ndarray
        ``C(z)`` in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``.
    """
    return _true_power_connection(
        wall_connection(block, z, impedance=impedance), block, grid, z
    )


def _true_power_connection(
    link: Float[np.ndarray, "n_modes_l_out n_modes_l_in"],
    block: tuple[Mode, ...],
    grid: TimeGrid,
    z: float,
) -> Complex[np.ndarray, "n_modes_w_out n_modes_w_in n_omega_w"]:
    """Return a connection carried to the kernel's amplitude on the run's band.

    ``W^(-1/2) C W^(1/2)`` with ``W = diag(Re n_eff)`` at `z`, entry by entry
    ``sqrt(w_n / w_m) C_mn``, exactly zero off the active band
    (:func:`_wall_connection_at` gives the reason).

    Parameters
    ----------
    link : numpy.ndarray
        ``C`` in 1/m, real, shape ``(n_modes, n_modes)``.
    block : tuple of Mode
        The channels in state order.
    grid : solprop.grid.TimeGrid
        The run's grid.
    z : float
        Position in m.

    Returns
    -------
    numpy.ndarray
        The term in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``.
    """
    band = grid.omega[grid.active_mask]
    weight = np.stack([np.real(np.asarray(mode.neff(band, z))) for mode in block])
    root = np.sqrt(weight)
    scattered = np.zeros((*link.shape, grid.n_omega), dtype=np.complex128)
    scattered[:, :, grid.active_mask] = (
        link[:, :, None] * root[None, :, :] / root[:, None, :]
    )
    return scattered


def _check_wall_total_varying(
    coefficients: Complex[
        np.ndarray, "n_intervals_v n_deg_v n_modes_v_out n_modes_v_in n_omega_v"
    ],
    linear: LinearOperator,
    edges: Float[np.ndarray, " n_edges_v"],
    block: tuple[Mode, ...],
    grid: TimeGrid,
    tail: float = 0.0,
) -> None:
    """Refuse a varying coupled route whose total internal generator gains on average.

    The varying counterpart of :func:`_check_wall_total`, and the same door: a
    conjugation applied to the block and not to the carrier turns the diagonal into
    gain. What is compared is each section's **mean** rate, because ``Phi`` has no
    pointwise rate this layer can read and its exact increment over the section does:
    the mean of a Chebyshev series on its own interval is
    ``c_0 - c_2/3 - c_4/15 - ...``, and ``delta_phi`` over the section divided by its
    width is the carrier's. A sign slipped into this arithmetic is a *global*
    property of the conjugation, which is what this catches; the block's own pointwise
    passivity is checked at every sampled position by
    :func:`solprop.modes.capillary.wall_operator`.

    Parameters
    ----------
    coefficients : numpy.ndarray
        The stored generator's Chebyshev coefficients in 1/m.
    linear : solprop.translate.linop.LinearOperator
        The carrier the flow is multiplied by.
    edges : numpy.ndarray
        The section edges in m.
    block : tuple of Mode
        The mode set. ``Phi`` was built from its lowest channel and the flow's diagonal
        completes every other channel's normalisation, so the allowance is each
        channel's own.
    grid : solprop.grid.TimeGrid
        The run's grid, which supplies the band the allowance is taken over.
    tail : float, optional
        The expansion's relative Chebyshev tail, dimensionless. The interpolant's
        section mean departs from the function's by about the size of its last
        coefficients, so twice the tail times the block's scale joins the rounding
        budget: the normalisation term is compared through the interpolant on one side
        and exactly on the other.

    Raises
    ------
    TranslationError
        If any channel grows on average over any section by more than its own
        normalisation term accounts for.
    """
    count = len(block)
    degrees = np.arange(coefficients.shape[1])
    # The mean of T_k on its own interval is zero for odd k and 1/(1 - k^2) for even,
    # which is one at k = 0 and never divides by zero, k = 1 being odd.
    means = np.zeros(degrees.size)
    even = degrees % 2 == 0
    means[even] = 1.0 / (1.0 - degrees[even] ** 2.0)
    scale = float(np.max(abs(coefficients)))
    budget = (64 * count * float(np.finfo(float).eps) + 2.0 * tail) * scale
    for index in range(edges.size - 1):
        lo, hi = float(edges[index]), float(edges[index + 1])
        carrier = np.real(delta_phi(linear, lo, hi))[0, :, 0, :] / (hi - lo)
        frozen = np.tensordot(means, coefficients[index], axes=1)
        allowance = budget + _normalisation_rate(block, grid, lo, hi)
        total = (
            carrier + np.real(frozen[np.arange(count), np.arange(count)]) - allowance
        )
        largest = float(np.max(total))
        if largest > 0.0:
            channel, bin_index = np.unravel_index(int(np.argmax(total)), total.shape)
            msg = (
                f"the coupled flow's total internal generator grows on "
                f"section {index} of [{lo:.6g}, {hi:.6g}] m: channel "
                f"{int(channel)} gains {largest:.4g} 1/m on average at bin "
                f"{int(bin_index)} beyond its own normalisation term and the "
                f"{budget:.4g} 1/m rounding of a block "
                f"whose own scale is {scale:.4g} 1/m. Crossing the optics/internal "
                "boundary conjugates the whole equation, carrier and block alike "
                "(docs/dev/conventions.md section 2)"
            )
            raise TranslationError(msg)


def _validated(problem: "KernelProblem") -> "KernelProblem":  # noqa: UP037
    """Validate an assembled problem and run every diagnostic that only warns.

    Parameters
    ----------
    problem : KernelProblem
        The assembled contract.

    Returns
    -------
    KernelProblem
        The same problem.
    """
    problem.validate()
    validate_coupled_flow(problem)
    _admit_dense(problem)
    return problem


def _normalisation_rate(
    block: tuple[Mode, ...], grid: TimeGrid, lo: float, hi: float
) -> Float[np.ndarray, "n_modes_r n_omega_r"]:
    """Return the mean rate each channel's normalisation term adds over one section.

    ``Phi`` carries ``-0.5 ln[w(z)/w(0)]`` beside the integral of ``D``, which
    renormalises a vacuum amplitude along a varying effective index
    (``docs/dev/kernel-contract.md`` section 3), and on the wall route it is the lowest
    channel's; the flow's diagonal completes every other channel's to its own
    (:func:`_channel_normalisation_at`). The term is **not sign definite**: along a
    falling pressure the index falls and it is positive, so the total internal
    generator's real part is positive by that much with nothing wrong. The passivity
    guard allows exactly it, per channel, and nothing else -- a conjugation slipped
    into the block turns the diagonal into gain of the order of the whole wall loss,
    three orders above this -- and the allowance is each channel's term's own exact
    mean over the section, ``|ln(w_n(hi)/w_n(lo))|/(2 (hi - lo))``.

    Parameters
    ----------
    block : tuple of Mode
        The channels, in state order.
    grid : solprop.grid.TimeGrid
        The run's grid; the allowance is zero off its active band, where the block is.
    lo, hi : float
        The section's ends in m.

    Returns
    -------
    numpy.ndarray
        The non-negative allowance in 1/m, shape ``(n_modes, n_omega)``.
    """
    band = grid.omega[grid.active_mask]
    rate = np.zeros((len(block), grid.n_omega))
    for channel, mode in enumerate(block):
        wide = np.real(np.asarray(mode.neff(band, lo)))
        narrow = np.real(np.asarray(mode.neff(band, hi)))
        ratio = np.divide(narrow, wide, out=np.ones_like(narrow), where=wide > 0.0)
        rate[channel, grid.active_mask] = abs(
            0.5 * np.log(np.where(ratio > 0.0, ratio, 1.0))
        ) / (hi - lo)
    return rate


def validate_coupled_flow(problem: "KernelProblem") -> None:  # noqa: UP037
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

    A flow with a ``"cf4"`` section has no explicit residual there and no advisory step,
    but a flow error the embedded pair cannot see: the largest declared step's
    step-doubling defect estimate (:func:`solprop.translate.coupled.cf4_defect`) is
    warned on when it exceeds the controller's relative tolerance -- an asymptotic
    estimator, and advisory like the first.

    Parameters
    ----------
    problem : KernelProblem
        The assembled contract. A problem carrying no coupled flow, or one whose
        residual is identically zero, has nothing to warn about and is returned from
        in silence.

    Warns
    -----
    solprop.exceptions.SolpropWarning
        Naming the step that exceeds the limit, with the term, the section, the
        frequency, ``K``, the conditioning factor and the limit; or, for a CF4 flow,
        the step, the section, the frequency, the defect estimate and the tolerance.

    Examples
    --------
    >>> from solprop.translate.problem import validate_coupled_flow
    >>> validate_coupled_flow.__name__
    'validate_coupled_flow'
    """
    flow = problem.linear.coupled_flow
    if flow is None:
        return
    # Over the bins the explicit residual is applied on: the requested band.
    _warn_advisory_step(flow, problem.stepper, problem.grids.inband_mask)
    if "cf4" in flow.section_methods:
        _warn_cf4_defect(flow, problem.stepper, problem.linear.hard_kinks)


def _declared_steps(settings: StepperSettings) -> dict[str, float]:
    """Return the steps a run declares, by name: the two limits and a replay's largest.

    Parameters
    ----------
    settings : StepperSettings
        The run's stepper section.

    Returns
    -------
    dict
        ``dz_init``, ``dz_max`` and, for a fixed schedule, its largest step, in m.
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
    """Warn when a declared step exceeds the explicit residual's advisory limit.

    Parameters
    ----------
    flow : solprop.translate.coupled.CoupledFlowTables
        The leaf.
    settings : StepperSettings
        The run's stepper section.
    scored : numpy.ndarray of bool
        The error norm's bins, on which alone the residual is applied.

    Warns
    -----
    solprop.exceptions.SolpropWarning
        Naming the step, the term, the section, the frequency, ``K``, the
        conditioning factor and the limit.
    """
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


def _warn_cf4_defect(
    flow: CoupledFlowTables,
    settings: StepperSettings,
    kinks: Float[np.ndarray, " n_kinks"],
) -> None:
    """Warn when a CF4 flow's estimated local defect exceeds the controller's tolerance.

    The CF4 counterpart of the advisory step, and as advisory as it: a CF4
    section's flow error is invisible to the embedded Runge-Kutta pair, which compares
    two combinations of the same stages, so the controller can neither see nor bound
    it, and the only step that governs it is the one the run declares. The
    step-doubling estimate of :func:`solprop.translate.coupled.cf4_defect` at the
    largest declared step is compared with the controller's relative tolerance, which
    is what the pair holds the rest of each step to. It is an **asymptotic
    estimator**, not a bound, and the warning says so; nothing here changes the
    caller's tolerances or schedule.

    Parameters
    ----------
    flow : solprop.translate.coupled.CoupledFlowTables
        The leaf, carrying at least one ``"cf4"`` section.
    settings : StepperSettings
        The run's stepper section.
    kinks : numpy.ndarray
        The run's hard kinks in m, at which every step ends.

    Warns
    -----
    solprop.exceptions.SolpropWarning
        Naming the step, the section, the frequency, the estimate and the tolerance.
    """
    declared = _declared_steps(settings)
    name = max(declared, key=lambda key: declared[key])
    step = declared[name]
    defect = cf4_defect(flow, step, kinks)
    section, bin_index = np.unravel_index(int(np.argmax(defect)), defect.shape)
    largest = float(defect[section, bin_index])
    tolerance = float(settings.controller.rtol)
    if largest <= tolerance:
        return
    warnings.warn(
        f"the coherent wall's CF4 flow has an estimated local defect of {largest:.4g} "
        f"per step of {step:.4g} m ({name}) on section {int(section)} at bin "
        f"{int(bin_index)}, {float(flow.omega[bin_index]):.6g} rad/s, above the "
        f"controller's relative tolerance {tolerance:.4g}: the embedded pair does not "
        "see a CF4 flow's error, so only the declared step governs it, and at this "
        "one the run is at most fourth order with a linear share the controller "
        "cannot hold. The estimate is step doubling, asymptotic and not a bound. "
        "Nothing here changes your schedule: cap the step, or "
        "declare the default method, deliberately",
        SolpropWarning,
        stacklevel=3,
    )


def _check_wall_route(build: _Assembly) -> None:
    """Refuse, by name, every run a coherent wall block is not derived for.

    The two the *assembly* owns. The route the run takes is refused earlier still
    (:func:`_refuse_a_wall_off_the_capillary_route`, the first thing
    :func:`assemble` does), the mode set by
    :func:`solprop.modes.capillary.wall_operator`, and an empty active band by that
    same function, which names the mask it was given and is the layer that evaluates
    over it.

    Parameters
    ----------
    build : _Assembly
        The assembly's knobs.

    Raises
    ------
    TranslationError
        Naming what the block cannot carry.
    """
    if build.n_pol != 1:
        msg = (
            f"a coherent wall block carries one field component and this run carries "
            f"n_pol = {build.n_pol}: the flow's action contracts the channel axis "
            "alone, so a block beside two components would be applied to each as "
            "though it described both. CoherentWall(impedance='te_tm') does not "
            "admit a second component either: it names the wall's own TE and TM "
            "impedances, not the state's polarisation axis, and its mode families "
            "are channels of the mode set, one component each"
        )
        raise TranslationError(msg)
    if build.noise_model == "modified":
        msg = (
            "a coherent wall block refuses the 'modified' noise model: its draw is "
            "transported by the unitary phase of a diagonal flow, and the transport "
            "of a draw through a nonunitary coupled flow is not measured in this "
            "package. " + _INPUT_NOISE_ADMITTED
        )
        raise TranslationError(msg)


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

    Parameters
    ----------
    wall : solprop.modes.capillary.CoherentWall or None
        What the run declared.
    general : bool
        Whether the run takes the general modal route.
    spatial : bool
        Whether the run carries a spatial geometry.
    taper_coupling : bool, optional
        Whether the run keeps the taper coupling, which a wall cannot decline.

    Raises
    ------
    TranslationError
        Naming the route that cannot carry the block, or the declined coupling.
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


def _check_wall_total(
    generator: Complex[np.ndarray, "n_modes_t_out n_modes_t_in n_omega_t"],
    linear: LinearOperator,
    count: int,
) -> None:
    """Refuse a wall route whose total internal generator shows gain.

    The last of the three doors the sign of this operator passes through. The wall
    block itself is checked for passivity in its own coordinates
    (:func:`solprop.modes.capillary.wall_operator`); here the **carrier** is added
    back, so a conjugation applied to one of the two terms and not the other -- the
    easy mistake of writing ``-i B`` where ``-i conj(B)`` was meant -- turns the
    diagonal into gain and is caught.

    Parameters
    ----------
    generator : numpy.ndarray
        The stored internal-convention generator in 1/m.
    linear : LinearOperator
        The carrier, whose constant term is what the flow is multiplied by.
    count : int
        Channel count, used in the refusal.

    Raises
    ------
    TranslationError
        If any channel grows at any frequency.
    """
    assert linear.phi_constant is not None
    total = np.real(linear.phi_constant[0, :, 0, :]) + np.real(
        generator[np.arange(count), np.arange(count)]
    )
    scale = float(np.max(abs(generator))) + float(np.max(abs(linear.phi_constant)))
    budget = 64 * count * float(np.finfo(float).eps) * scale
    largest = float(np.max(total))
    if largest > budget:
        channel, bin_index = np.unravel_index(int(np.argmax(total)), total.shape)
        msg = (
            f"the coherent wall route's total internal generator grows: channel "
            f"{int(channel)} gains {largest:.4g} 1/m at bin {int(bin_index)}, above "
            f"the {budget:.4g} 1/m rounding of a problem whose own scale is "
            f"{scale:.4g} 1/m. Crossing the optics/internal boundary conjugates the "
            "whole equation, carrier and block alike (docs/dev/conventions.md "
            "section 2)"
        )
        raise TranslationError(msg)


def _check_general(
    route: GeneralModalRoute,
    grid: TimeGrid,
    refusals: _Assembly,
) -> None:
    """Refuse every legacy knob and every grid a general modal route cannot take.

    Parameters
    ----------
    route : solprop.translate.general.GeneralModalRoute
        The guide's leaves.
    grid : solprop.grid.TimeGrid
        The run's grid.
    refusals : _Assembly
        The assembly's knobs; each legacy one is refused by name when it is set,
        and the declared responses must be empty, a general guide's belonging to
        its material regions.

    Raises
    ------
    TranslationError
        Naming what cannot be combined with a general modal route.
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

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The run's grid, which supplies the crop scale and the active band.
    linear : solprop.translate.linop.LinearOperator
        The assembled operator, whose z span every other table must share.
    n_pol : int
        State polarisations, one on this route.

    Returns
    -------
    solprop.translate.chebyshev.PiecewiseChebyshev
        One interval of degree zero with value shape ``(n_modes, n_pol, n_omega)``.
    """
    n_modes = linear.value_shape[1]
    values = np.zeros((1, 1, n_modes, n_pol, grid.n_omega), dtype=np.complex128)
    values[:, :, :, :, grid.active_mask] = grid.crop_scale
    return PiecewiseChebyshev(
        _readonly(values),
        _readonly(np.asarray(linear.z_span, dtype=np.float64)),
        _readonly(np.zeros(1, dtype=np.intp)),
        _readonly(np.zeros(0)),
    )


def _check_scalar_domain(
    table: PiecewiseChebyshev, name: str, *, upper: float = math.inf
) -> None:
    """Check a scalar polynomial at endpoints and every real stationary point.

    Parameters
    ----------
    table : PiecewiseChebyshev
        Scalar profile table checked over every polynomial interval.
    name : str
        Table name used in a refusal diagnostic.
    upper : float
        Inclusive upper domain bound, in the table value units; infinity by default.
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


def _walk_arrays(problem: KernelProblem) -> list[tuple[str, np.ndarray]]:
    """Collect every NumPy array in the problem tree, with its dotted name.

    Parameters
    ----------
    problem : KernelProblem
        The problem to walk.

    Returns
    -------
    list of tuple
        ``(name, array)`` pairs, in field order.
    """
    found: list[tuple[str, np.ndarray]] = []

    def visit(prefix: str, node: object) -> None:
        """Add every array reachable from one node.

        Parameters
        ----------
        prefix : str
            Dotted name of the node.
        node : object
            A dataclass, an array, or anything else.
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
                name = f"{prefix}.{entry.name}" if prefix else entry.name
                visit(name, getattr(node, entry.name))

    visit("", problem)
    return found


def _walk_tables(problem: KernelProblem) -> list[tuple[str, PiecewiseChebyshev]]:
    """Collect every Chebyshev table along z, with its dotted name.

    Parameters
    ----------
    problem : KernelProblem
        The problem to walk.

    Returns
    -------
    list of tuple
        ``(name, table)`` pairs.
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


def _even_chunk(count: int, limit: int) -> int:
    """Return the smallest chunk that covers `count` items in as few passes as `limit`.

    Parameters
    ----------
    count : int
        Items to cover (positive).
    limit : int
        Largest admitted chunk (positive).

    Returns
    -------
    int
        ``ceil(count / ceil(count / limit))``, at most `limit` and `count`.

    Examples
    --------
    >>> _even_chunk(33, 32), _even_chunk(33, 33), _even_chunk(100, 30)
    (17, 33, 25)
    """
    passes = -(-count // limit)
    return -(-count // passes)


def _balanced_chunk(transform: Transform) -> Transform:
    """Spread the nodes evenly over the passes the admitted chunk already needs.

    The budget admissions each take ``budget // per_node``, which can leave a
    last pass of a node or two: 33 RDW nodes under a limit of 32 ran a whole
    32-node pass and then a one-node tail on one thread, about 1.6 times the cost
    of one 33-node pass. The same number of passes of near-equal size (17 and 16)
    never exceeds the admitted chunk, so every budget still holds. A planar chunk
    of whole transverse rows stays whole rows; one shorter than a row is kept.

    Parameters
    ----------
    transform : Transform
        Transform with its admitted chunk.

    Returns
    -------
    Transform
        The transform, with the smallest chunk that needs no more passes.
    """
    row_nodes = (
        transform.geometry.shape[0]
        if transform.geometry is not None and transform.geometry.kind == "slab"
        else 1
    )
    rows = transform.n_nodes // row_nodes
    rows_per_chunk, partial = divmod(transform.node_chunk, row_nodes)
    if partial or rows_per_chunk >= rows:
        return transform
    chunk = _even_chunk(rows, rows_per_chunk) * row_nodes
    return (
        transform
        if chunk == transform.node_chunk
        else replace(transform, node_chunk=chunk)
    )


def _envelope_chunk(
    transform: Transform, n_pol: int, n_to: int, budget: int, *, enabled: bool
) -> Transform:
    """Include current streams and histories in envelope plasma chunk admission.

    Parameters
    ----------
    transform : Transform
        Existing transform and baseline chunk size.
    n_pol : int
        State polarisation components.
    n_to : int
        Fine temporal samples per node.
    budget : int
        Node workspace budget in bytes.
    enabled : bool
        Whether the compiled problem carries envelope plasma sources.

    Returns
    -------
    Transform
        Transform with a bounded chunk. Modal projected arrays are accounted
        separately by backend ledgers because they do not scale with chunk nodes.

    Raises
    ------
    TranslationError
        If one complete node history and its sources do not fit.
    """
    if not enabled:
        return transform
    components = transform.n_pol * n_pol
    free = transform.geometry is not None and transform.geometry.kind in (
        "free",
        "slab",
    )
    baseline = (20 if free else 3) * 16 * components
    # Bound both host and device storage: two complex sources, up to six float
    # histories, per-sample angular status, and fixed256 scan totals/carries.
    # Cartesian chunks also own projected spectra and projection scratch.
    extra = 32 * components + 56 + (80 * components if free else 0)
    per_node = n_to * (baseline + extra) + 16 * ((n_to + 255) // 256) + 8
    if budget < per_node:
        raise TranslationError(
            "node_budget cannot hold one envelope plasma source history"
        )
    return replace(transform, node_chunk=min(transform.node_chunk, budget // per_node))


def _held_current_chunk(
    transform: Transform,
    responses: CompiledResponses,
    n_pol: int,
    n_to: int,
    budget: int,
) -> Transform:
    """Admit complete held-current histories and their nodal transforms.

    Modal projected streams do not scale with the node chunk and belong to
    total-memory admission. The allowance here includes the largest nodal
    transform layout (radial), the band scan, Raman and dispersive buffers
    together, so a mixed response cannot consume the same budget twice.

    Parameters
    ----------
    transform : Transform
        Existing spatial or modal transform and its current chunk limit.
    responses : CompiledResponses
        Compiled sources; unchanged when no field plasma is present.
    n_pol : int
        State components.
    n_to : int
        Fine temporal samples.
    budget : int
        Node workspace budget, bytes.

    Returns
    -------
    Transform
        Transform with all node-dependent source buffers admitted.

    Raises
    ------
    TranslationError
        If one complete node, or planar transverse row, does not fit.
    """
    if not responses.plasma_channels:
        return transform
    spatial = transform.geometry is not None
    free = transform.geometry is not None and transform.geometry.kind in (
        "free",
        "slab",
    )
    components = transform.n_pol * n_pol
    baseline = _FREE_COLUMN_BUFFERS if free else BUFFERS_PER_NODE
    planar = transform.geometry is not None and transform.geometry.kind == "slab"
    # J and its two products; three real FFT inputs and spectra; cropped nodal
    # output. Each complex FFT row is bounded by n_to complex values, including
    # the DC/Nyquist pair. The device band's three scalar and three component
    # histories bound the host layouts too; tiled prefixes need one total per
    # component per minimum-size tile. Channels reuse those histories.
    source_bytes = (
        components * n_to * (3 * 8 + (3 * 8 + 4 * 16 if spatial and not planar else 0))
    )
    histories = (
        (3 + 3 * components) * n_to * 8
        + components * math.ceil(n_to / 256) * 8
        + len(responses.plasma_channels) * 4 * 8
    )
    radial_projection = (
        transform.n_modes * transform.n_pol * 8 if spatial and not free else 0
    )
    raman = (3 * (48 * n_to + 16)) if responses.raman_terms else 0
    dispersive = dispersive_node_buffers(responses, spatial=free)
    per_node = (
        (baseline + dispersive) * components * n_to * 16
        + source_bytes
        + histories
        + radial_projection
        + raman
    )
    row_nodes = (
        transform.geometry.shape[0]
        if transform.geometry is not None and transform.geometry.kind == "slab"
        else 1
    )
    # A planar row contracts n_x physical nodes to n_p modal columns before its
    # temporal FFT. Charge the two projected sources and spectra, plus Numba's
    # shared projection scratch, per modal column rather than per physical node.
    # This matters when a full row is the smallest possible chunk.
    projected_row = (
        transform.synthesis.shape[1]
        * components
        * (3 * n_to * 8 + 2 * (n_to // 2 + 1) * 16)
        if planar
        else 0
    )
    per_row = per_node * row_nodes + projected_row
    if budget < per_row:
        raise TranslationError(
            "node_budget cannot hold one held plasma current history"
            " (one full transverse row for a planar grid)"
        )
    # Planar projection contracts complete x rows. Admit the rounded row here,
    # before a backend would otherwise have to increase a sub-row allocation.
    chunk = (
        min(max(1, transform.node_chunk // row_nodes), budget // per_row) * row_nodes
    )
    return (
        transform
        if chunk == transform.node_chunk
        else replace(transform, node_chunk=chunk)
    )


def _raman_chunk(
    transform: Transform,
    responses: CompiledResponses,
    n_pol: int,
    n_to: int,
    budget: int,
) -> Transform:
    """Admit complete Raman histories, including combined envelope plasma storage.

    Parameters
    ----------
    transform : Transform
        Existing transform and baseline chunk limit.
    responses : CompiledResponses
        Declared convolution channels and optional plasma streams.
    n_pol : int
        State polarisation components, one or two.
    n_to : int
        Fine temporal samples per physical node.
    budget : int
        Node workspace budget in bytes.

    Returns
    -------
    Transform
        Transform with a complete-node chunk within the workspace budget.
    """
    if not responses.raman_terms:
        return transform
    components = transform.n_pol * n_pol
    drives = (
        3
        if components == 2
        and any(np.any(term.tensor_weights[:, 1]) for term in responses.raman_terms)
        else 1
    )
    free = transform.geometry is not None and transform.geometry.kind in (
        "free",
        "slab",
    )
    # Cartesian admission already reserves three complex-equivalent Raman rows
    # per component. Replace that allowance by the selected real drive layout.
    baseline = (17 if free else 3) * 16 * components * n_to
    extra = (48 * n_to + 16) * drives
    if responses.envelope_plasma is not None:
        extra += n_to * (32 * components + 56 + (80 * components if free else 0))
        extra += 16 * ((n_to + 255) // 256) + 8
    per_node = baseline + extra
    if budget < per_node:
        raise TranslationError(
            "node_budget cannot hold one complete Raman node history"
        )
    # Full and actual-width tail plans share these arrays. Opaque provider plan
    # storage and node-independent kernels appear separately in backend ledgers.
    chunk = min(transform.node_chunk, budget // per_node)
    return (
        transform
        if chunk == transform.node_chunk
        else replace(transform, node_chunk=chunk)
    )


_FREE_COLUMN_BUFFERS = 20
"""Complex-equivalent buffers per component a Cartesian or planar column holds.

The count ``free_transform`` and ``planar_transform`` admit a column by
(``translate/free.py``); the dispersive admission adds its own to it.
"""


def dispersive_node_buffers(responses: CompiledResponses, *, spatial: bool) -> int:
    """Return the buffers per node and component SEP 6's dispersive response adds.

    On the fused node path a backend forms the response's products at the nodes:
    one node buffer beside other responses for the symmetric evaluator, which a lone
    one shares with the bare field, and two for the polynomial, whose cube and
    derivative products take them in turn. A Cartesian or planar column also holds
    its filtered fields and products in time and their spectra: two buffers per
    product, the symmetric evaluator's one (a mixture's under its mixture filter) and
    the polynomial's ``1 + 2 G`` for ``G`` groups of equal leg bound
    (:meth:`~solprop.translate.responses.DispersiveKerrTerms.groups`), and the planar
    nodes' one or two besides.

    Parameters
    ----------
    responses : CompiledResponses
        The compiled responses, with or without the dispersive leaf.
    spatial : bool
        Whether the route's nodes are Cartesian or planar columns.

    Returns
    -------
    int
        Complex-equivalent buffers per node and component; zero without the leaf.
    """
    leaf = responses.dispersive_kerr
    if leaf is None:
        return 0
    symmetric = leaf.evaluator == "symmetric"
    alone = (
        set(responses.response_order) - {ResponseStage.ACCUMULATE}
        == {ResponseStage.DISPERSIVE_KERR}
        and responses.envelope_plasma is None
    )
    groups = len(leaf.groups())
    nodes = (0 if alone and groups == 1 else 1) if symmetric else 2
    if not spatial:
        return nodes
    products = groups if symmetric else 1 + 2 * groups
    return nodes + 2 * products


def _dispersive_chunk(
    transform: Transform,
    responses: CompiledResponses,
    n_pol: int,
    n_to: int,
    budget: int,
) -> Transform:
    """Admit the dispersive response's node buffers into the chunk's budget.

    Parameters
    ----------
    transform : Transform
        Existing transform and baseline chunk limit.
    responses : CompiledResponses
        The compiled responses; nothing changes without the dispersive leaf.
    n_pol : int
        State polarisation components, one or two.
    n_to : int
        Fine temporal samples per node.
    budget : int
        Node workspace budget in bytes.

    Returns
    -------
    Transform
        Transform with a chunk whose every buffer fits the budget. A general modal
        guide sizes its own chunk from its maps and is returned unchanged.

    Raises
    ------
    TranslationError
        If one node and the response's buffers do not fit.
    """
    if responses.dispersive_kerr is None or transform.modal_tables is not None:
        return transform
    spatial = transform.geometry is not None and transform.geometry.kind in (
        "free",
        "slab",
    )
    components = transform.n_pol * n_pol
    baseline = _FREE_COLUMN_BUFFERS if spatial else BUFFERS_PER_NODE
    extra = dispersive_node_buffers(responses, spatial=spatial)
    per_node = (baseline + extra) * components * n_to * 16
    if budget < per_node:
        raise TranslationError(
            "node_budget cannot hold one node of the dispersive electronic response"
        )
    chunk = min(transform.node_chunk, budget // per_node)
    return (
        transform
        if chunk == transform.node_chunk
        else replace(transform, node_chunk=chunk)
    )


def _state(
    field0: Complex[np.ndarray, "*shape"],
    n_omega: int,
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    """Bring an initial field to the canonical layout, adding the batch axis.

    Parameters
    ----------
    field0 : numpy.ndarray
        A ``Pulse.build`` result of shape ``(n_modes, n_pol, n_omega)``, or a stack of
        them with a leading batch axis. The shape is checked here rather than by an
        annotation, so that a wrong rank is reported by name in production too.
    n_omega : int
        Required number of spectral bins from the grid.

    Returns
    -------
    numpy.ndarray
        The state with four axes, complex128.

    Raises
    ------
    TranslationError
        If the array has neither three nor four axes or the grid extent differs.
    """
    values = np.asarray(field0, dtype=np.complex128)
    if values.ndim in (3, 4) and values.shape[-1] != n_omega:
        raise TranslationError(
            f"field0 has {values.shape[-1]} spectral bins, expected {n_omega} "
            "to match the grid"
        )
    if values.ndim == 3:
        return values[np.newaxis]
    if values.ndim != 4:
        msg = (
            "field0 must have shape (n_modes, n_pol, n_omega) or (n_batch, n_modes, "
            f"n_pol, n_omega), got {values.shape}"
        )
        raise TranslationError(msg)
    return values


def _material_mixture(
    modes: Sequence[Mode],
    medium: SpatialMedium | None,
    density: ZProfile | float | None,
) -> Mixture | None:
    """Resolve the material shared by the linear and response paths.

    Parameters
    ----------
    modes : sequence of Mode
        Ordered mode set, empty for spatial geometry.
    medium : SpatialMedium or None
        Spatial material, or None for modal geometry.
    density : ZProfile, float or None
        Optional explicit response density in m^-3.

    Returns
    -------
    Mixture or None
        The declared mixture, or None for the legacy single material.

    Raises
    ------
    TranslationError
        If an explicit density conflicts with a mixture or a gas medium.
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
    """

    transform: Transform
    initial: Initial
    phi_expansion: ParameterExpansion | None
    prefactor_expansion: ParameterExpansion | None
    raman_undamped: Mapping[str, Float[np.ndarray, "... n_to_doubled"]]


def _check_tdse_assembly_route(
    descriptor: TDSEDescriptor | None,
    *,
    modal: bool,
    scheduled: bool,
    absorber_scheme: AbsorberScheme,
    density: ZProfile | float | None,
) -> None:
    """Refuse paths whose optical ownership is outside the atomic contract.

    Parameters
    ----------
    descriptor : TDSEDescriptor or None
        Atomic declaration, or None for an ordinary response.
    modal : bool
        Whether the geometry is the uncoupled modal route.
    scheduled : bool
        Whether a fixed step schedule was supplied.
    absorber_scheme : str
        Temporal collar application policy.
    density : ZProfile, float or None
        Requested density override in m^-3, refused for an atomic declaration.
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


def _admit_dense(problem: "KernelProblem") -> None:  # noqa: UP037 - defined below
    """Refuse dense observation where its interpolant is not the run's field.

    SEP 14 section 5's admission matrix, for the Runge-Kutta extension and for SEP 13's
    Adams interpolant alike. The per-step temporal collar changes the state after the
    step, so an interpolated plane would be uncollared; a coupled flow's action and the
    ``modified`` noise's draw are outside the interpolant.

    Parameters
    ----------
    problem : KernelProblem
        The assembled problem.

    Raises
    ------
    TranslationError
        Naming the combination refused.
    """
    if not problem.stepper.interpolate_planes:
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

    Parameters
    ----------
    linear : LinearOperator
        The problem's linear operator, whose coupled flow decides the policy.
    controller : solprop.stepper.controller.PIController
        The controller, whose reject-free switch decides the policy too.
    interval_cap : bool or None
        The interval cap asked for; ``None`` takes the route's.
    auto_init : bool
        Whether the start-up rule was asked for.
    **fields
        The other :class:`StepperSettings` fields.

    Returns
    -------
    StepperSettings
        The settings.
    """
    restrained = linear.coupled_flow is not None or controller.reject_free
    return StepperSettings(
        controller=controller,
        interval_cap=restrained if interval_cap is None else interval_cap,
        auto_init=auto_init and not restrained,
        **fields,
    )


def assemble(  # noqa: PLR0913, PLR0915 - one keyword per physics object of a run
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
    batch_params: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    shock: bool = True,
    prefactor_index: float | None = None,
    ionisation_field_range: tuple[float, float] | None = None,
    node_budget: int | None = None,
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
    transverse_absorber: TransverseCollar | None = None,
    adiabatic_accuracy: AdiabaticAccuracy | None = None,
    precision: Precision = DOUBLE,
) -> KernelProblem:
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
        third-harmonic Kerr term is present on an envelope grid.
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
        Landing stretch factor. ``None`` uses :data:`DEFAULT_STRETCH`.
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
        Which statistics to collect. ``None`` collects none.
    batch_params : mapping or None, optional
        Per-member scalars to record, each of length ``n_batch``.
    shock : bool, optional
        Keep the ``omega`` dependence of the prefactor's numerator, which is the optical
        shock term. Default ``True``.
    prefactor_index : float or None, optional
        Use this constant in the prefactor and power weight; None uses Re n_eff.
        The GNLSE passes 1.0, leaving its power normalisation and Phi unchanged.
    ionisation_field_range : tuple of float or None, optional
        Field amplitudes in V/m a plasma response's rate table must cover.
    node_budget : int or None, optional
        Bytes of node-chunk buffers a multimode response pass may hold. ``None``
        takes :data:`~solprop.translate.modal.DEFAULT_SPATIAL_NODE_BUDGET` on a
        free-space, line or planar grid and
        :data:`~solprop.translate.modal.DEFAULT_NODE_BUDGET` otherwise.
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
        backend executes (a CF4 section ``numpy_ref`` alone). A general modal guide,
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
    transverse_absorber : solprop.boundaries.TransverseCollar or None, optional
        A transverse-wavevector safety ramp folded into the Cartesian operator as
        ``SeparableOperator.kperp_rate`` (SEP 22 section 4.3). Refused on the radial
        and hybrid planar kinds and without a spatial grid.
    adiabatic_accuracy : solprop.adiabatic.AdiabaticAccuracy or None, optional
        The budgets a radial medium varying along z must meet for its forward
        amplitude transport (SEP 34 section 4.5); None, the default, is
        ``AdiabaticAccuracy()``. A path exceeding them is refused.
    precision : solprop.translate.precision.Precision, optional
        The arithmetic the run is held in (SEP 33): :data:`~.precision.DOUBLE`, the
        default, or :data:`~.precision.SINGLE`. The tables are built in binary64
        either way; what single precision refuses is refused here.

    Returns
    -------
    KernelProblem
        Validated, versioned and ready for a backend.

    Raises
    ------
    TranslationError
        If a response cannot be compiled, an argument is out of range, or the assembled
        problem fails :meth:`KernelProblem.validate`.

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
    >>> problem = assemble(
    ...     grid, gnlse.mode, [gnlse.kerr], start,
    ...     controller=PIController(rtol=1e-8, atol=1e-14), save_planes=[0.5, 1.0],
    ...     dz_max=0.1, dz_init=1e-3, shock=False, prefactor_index=1.0, density=1.0,
    ... )
    >>> problem.flags.single_mode, problem.linear.n_intervals
    (True, 1)
    """
    if node_budget is None:
        node_budget = (
            DEFAULT_SPATIAL_NODE_BUDGET
            if isinstance(spatial_grid, FreeGrid | PlanarGrid)
            else DEFAULT_NODE_BUDGET
        )
    _refuse_a_wall_off_the_capillary_route(
        wall,
        general=general is not None,
        spatial=spatial_grid is not None or medium is not None,
        taper_coupling=taper_coupling,
    )
    state = _state(field0, grid.n_omega)
    n_batch, n_modes, n_pol, n_omega = state.shape
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
        transverse_absorber=transverse_absorber,
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

    thg = any(getattr(one, "thg", False) for one in responses)
    working_frame = frame if frame is not None else Frame.comoving(grid, thg=thg)
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
        node_budget=node_budget,
        has_noise=noise is not None,
        noise_model="none" if noise is None else noise.model,
        dz_max=dz_max,
        reuse=reuse,
        kinks=response_kinks(
            responses, density if general is None and spatial_grid is None else None
        ),
        taper_coupling=taper_coupling,
        evanescent_rate=evanescent_rate,
        transverse_absorber=transverse_absorber,
        adiabatic_accuracy=adiabatic_accuracy,
    )
    # Each mode's frequency-only terms are sampled once per frequency vector for the
    # whole assembly: the dispersion, power-weight and prefactor tables all read them.
    with _shared_sampling():
        translated_prefactor = None
        flux = None
        if general is not None:
            linear, transform, translated_prefactor = _general_assembly(
                general, grid, mode_set, working_frame, build
            )
        elif spatial_grid is not None and medium is not None:
            linear, transform, translated_prefactor, flux = _spatial_assembly(
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
                    node_budget=node_budget,
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
    # SEP 6, section 4.10: classified once, at translation, on the whole active band;
    # a field reaching beyond a gas's support is logged here and never refused.
    dispersive_regions(responses, grid, vector=transform.n_pol * n_pol > 1)
    transform = _envelope_chunk(
        transform,
        n_pol,
        grid.n_to,
        node_budget,
        enabled=compiled.envelope_plasma is not None,
    )
    transform = _dispersive_chunk(
        _raman_chunk(transform, compiled, n_pol, grid.n_to, node_budget),
        compiled,
        n_pol,
        grid.n_to,
        node_budget,
    )
    transform = _balanced_chunk(
        _held_current_chunk(transform, compiled, n_pol, grid.n_to, node_budget)
    )
    initial = _power_initial(
        Initial(
            field0=state,
            noise0=None if noise is None else noise_spectrum(noise, state.shape),
            noise_model="none" if noise is None else noise.model,
            batch_params=_batch_params(batch_params, n_batch),
            z=initial_z,
        ),
        linear,
        prefactor_index,
        None if reuse is None else reuse.initial,
    )
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
    problem = KernelProblem(
        schema_version=KERNEL_SCHEMA_VERSION,
        grid_kind=grid.kind,
        shape=(n_batch, n_modes, n_pol, n_omega),
        flags=Flags.from_content(
            (n_batch, n_modes, n_pol, n_omega), linear, transform, compiled, initial
        ),
        grids=build_masks(
            grid,
            polarisation_taper=polarisation_taper,
            region=getattr(settings.controller, "region", "active"),
            harmonic_carrier=compiled.harmonic_carrier,
        ),
        linear=linear,
        transform=transform,
        responses=compiled,
        initial=initial,
        stepper=settings,
        stats_spec=StatsSpec() if stats_spec is None else stats_spec,
        tdse=tdse,
        current_numerator=None if tdse is None else current_numerator(grid),
        flux=flux,
        precision=precision,
    )
    _validated(problem)
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
        logger.debug("%s", debug_summary(problem))
    return problem


def debug_summary(problem: KernelProblem) -> str:
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
    problem : KernelProblem
        A problem, assembled and validated.

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
    kernel problem schema 39: envelope grid, route modal
    """
    linear, transform = problem.linear, problem.transform
    tables, metric = transform.modal_tables, linear.modal_metric
    route = (
        "general modal guide"
        if tables is not None
        else "spatial"
        if transform.geometry is not None
        else "modal"
    )
    rows = [
        f"kernel problem schema {problem.schema_version}: "
        f"{problem.grid_kind} grid, route {route}",
        f"  shape:         {problem.shape} (batch, modes, pol, omega)",
        f"  flow:          phi {linear.phi_representation}, "
        f"{linear.n_intervals} interval(s), {linear.nbytes / 2**20:.3f} MB",
        f"  power weight:  {linear.power_weight_representation}",
        "  responses:     order "
        f"{[str(one.value) for one in problem.responses.response_order]}",
        f"  noise:         {problem.initial.noise_model}",
        f"  statistics:    {', '.join(problem.stats_spec.names) or 'none'}",
    ]
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

    Parameters
    ----------
    initial : Initial
        Fresh true-power launch and noise in canonical shape.
    linear : solprop.translate.linop.LinearOperator
        Assembled linear group; its modal power-weight row is the entrance weight, and
        its absence marks spatial geometry, which carries no modal conversion.
    index : float or None
        Constant index override, or None for the physical effective index.
    reused : Initial or None
        Previously scaled launch retained only when its arrays match.

    Returns
    -------
    Initial
        Internal vacuum-normalised launch and noise.
    """
    if linear.power_weight_representation is not None and index != 1.0:
        root_weight = np.sqrt(power_weight_row(linear, initial.z))
        initial = replace(
            initial,
            field0=initial.field0 / root_weight,
            noise0=None if initial.noise0 is None else initial.noise0 / root_weight,
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


def _batch_params(
    batch_params: Mapping[str, Sequence[float] | np.ndarray] | None, n_batch: int
) -> Mapping[str, Float[np.ndarray, " n_batch"]]:
    """Bring the per-member scalars to float64 arrays of the batch length.

    Parameters
    ----------
    batch_params : mapping or None
        The caller's per-member scalars, or ``None``.
    n_batch : int
        Batch members the state carries.

    Returns
    -------
    mapping
        Name to a ``(n_batch,)`` float64 array; empty when nothing was given.

    Raises
    ------
    TranslationError
        If an entry is not one value per member.
    """
    if batch_params is None:
        return {}
    entries: dict[str, np.ndarray] = {}
    for name, values in batch_params.items():
        entry = np.asarray(values, dtype=np.float64).reshape(-1)
        if entry.size != n_batch:
            msg = (
                f"batch_params[{name!r}] has {entry.size} entries, expected {n_batch}, "
                "one per batch member"
            )
            raise TranslationError(msg)
        entries[name] = entry
    return entries


def _density(
    modes: Sequence[Mode], density: ZProfile | float | None
) -> ZProfile | float:
    """Choose the density path a compiled response multiplies its coefficient by.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set; a gas-filled capillary declares its own ``density`` path.
    density : solprop.profiles.ZProfile or float or None
        The caller's choice, or ``None`` to take the mode's.

    Returns
    -------
    solprop.profiles.ZProfile or float
        The density, or 1.0 when neither the caller nor the mode states one.
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

    Parameters
    ----------
    components : sequence of str or None
        What the caller asked for, or ``None`` to resolve the mode set's components.
    quadrature : solprop.modes.quadrature.TransverseQuadrature or None
        The fixed rule, or ``None`` for the mode-averaged case, which has no components
        to resolve.
    modes : sequence of Mode
        Modes whose continuous profiles declare the nonzero components.

    Returns
    -------
    tuple of str
        The resolved components.

    Raises
    ------
    TranslationError
        If components are given without a rule, or are not a non-empty subset of
        ``("x", "y")``.
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


def _transform(  # noqa: PLR0913 - one call site, and each argument is a distinct choice
    modes: Sequence[Mode],
    grid: TimeGrid,
    quadrature: TransverseQuadrature | None,
    *,
    z_max: float,
    n_pol: int,
    components: Sequence[str] | None,
    node_budget: int,
    margin: float,
    tol: float,
) -> Transform:
    """Build the mode-averaged or the multimode transform, as the problem asks.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set.
    grid : solprop.grid.TimeGrid
        The grid.
    quadrature : solprop.modes.quadrature.TransverseQuadrature or None
        The fixed rule, or ``None`` for the mode-averaged case.
    z_max : float
        Length the run covers, m.
    n_pol : int
        Polarisation components of the state, 1 or 2.
    components : sequence of str or None
        The transverse components the rule resolves, or ``None`` for the union of
        the modes' nonzero components at the entrance (:func:`_components`).
    node_budget : int
        Bytes of node-chunk buffers a multimode response pass may hold.
    margin : float
        Fraction of `z_max` the tables extend beyond the end.
    tol : float
        Target bound on the uniform error of the tables.

    Returns
    -------
    Transform
        Section 4 of the kernel contract.

    Raises
    ------
    TranslationError
        If a mode set is given without a rule, if the components are invalid, or if the
        transform and the state would both carry a polarisation axis.
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
            node_budget=node_budget,
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
