# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The simulation object: physics objects in, a written, resumable run out.

A :class:`Simulation` holds what a run is made of -- the grid, the modes, the responses,
the input, the noise and the numerical settings -- translates it **once** into a
:class:`solprop.translate.problem.KernelProblem`, and can then be run as often as one
likes, into memory or into a file. It owns three things the loop of
:mod:`solprop.driver.run` does not: the translation and its cache, the
:class:`solprop.output.schema.RunMetadata` that says what produced a file, and the
resume decision.

**Resume.** Given a path, a run checks the file that is already there with
:func:`solprop.output.hdf5.check_cache`: the digest covers the grid, the modes, the
whole serialised configuration, the noise and the stepper settings, so a file written
for different physics is refused by name rather than continued from someone else's
state. If it passes, the last plane comes back through
:func:`solprop.output.hdf5.resume`, the backend is loaded at that plane's position, and
the run carries on into the same file with ``HDF5Output(path, append=True)``.

Layer D of ``docs/dev/architecture.md`` section 3: this module imports the physics, the
translation, the backends and the output layer, and nothing imports it. The manual page
is ``docs/guide/setup/running.md``.
"""

import copy
import dataclasses
import functools
import json
import logging
import math
import threading
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
from jaxtyping import Complex, Float
from scipy.special import j0, jn_zeros

import solprop.driver.tdse as tdse_runtime
from solprop._display import Displayed, engineering, line, summarise
from solprop.adiabatic import AdiabaticAccuracy
from solprop.backends.multistep import CallLedger
from solprop.backends.protocol import Backend
from solprop.backends.registry import metal_available, resolve_name, select
from solprop.boundaries import (
    Absorber,
    SpatialAbsorber,
    no_absorber,
    temporal_collar_rate,
    warn_walkoff,
)
from solprop.boundaries import absorber as collar
from solprop.constants import C
from solprop.coupling import (
    CouplingLaunch,
    CouplingModel,
    FieldPlane,
    InputPort,
    PlaneBasis,
    PlaneGeometry,
    PlaneReference,
    basis_identity,
    couple,
    modal_basis_identity,
    space_record,
)
from solprop.driver import tdse_consistency
from solprop.driver.material_output import (
    material_manifest,
    material_record,
    processing_record,
)
from solprop.driver.memory import MemoryAdmission, admit_guide_memory, admit_memory
from solprop.driver.mixtures import (
    ResponseSelection,
    capillary_fill,
    is_air,
    mixture_profiles,
)
from solprop.driver.run import (
    check_snapshot_request,
    field_retention,
    open_run_record,
    run_problem,
)
from solprop.exceptions import (
    AdiabaticTaperWarning,
    BoundaryWarning,
    CouplingError,
    DriverError,
    GasFlowWarning,
    GrazingIncidenceWarning,
    GridError,
    MaterialRangeWarning,
    MemoryBudgetError,
    MemoryBudgetWarning,
    OutputError,
    SampledPeakWarning,
    SolpropWarning,
    TDSEArchiveError,
    TDSECostWarning,
    TranslationError,
)
from solprop.fourier import power_amplitude
from solprop.gasflow import VacuumEnd
from solprop.grid import (
    EnvelopeGrid,
    FieldGrid,
    FreeGrid,
    LineGrid,
    RadialGrid,
    TimeGrid,
)
from solprop.linear import Frame, frame_origin, frame_quantities
from solprop.log import DEFAULT_PROGRESS_PERIOD, ProgressChoice, ProgressReporter
from solprop.modes.antiresonant import TubeLattice
from solprop.modes.capillary import (
    CoherentWall,
    GlassIndex,
    LossSpec,
    MarcatiliMode,
    MarcatiliModel,
    taper_coupling_block,
    worst_wall_limits,
)
from solprop.modes.collection import ModeCollection, transverse_measure
from solprop.modes.graded_index import GradedIndexProvider
from solprop.modes.guide import ModalGuide, StepIndexProvider
from solprop.modes.material import JsonValue, MaterialRegion
from solprop.modes.protocol import Mode, ParameterisedMode
from solprop.modes.quadrature import (
    RadialRule,
    TransverseQuadrature,
    cartesian_nodes,
    mode_matrix,
    on_axis_rule,
    transverse_quadrature,
)
from solprop.modes.slab import PlanarGrid, SlabMode
from solprop.modes.stepindex import StepIndexMode
from solprop.modes.tabulated import IndexTable
from solprop.monitor import DEFAULT_BOUNDARY_CHECK, BoundaryCheck, BoundaryMonitor
from solprop.noise import (
    NoiseRealisation,
    input_noise,
    modified_noise,
    spatial_noise,
    transformed_noise,
)
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.gnlse import (
    GnlseProblem,
    SpecifiedMode,
    attenuation_from_db,
    problem_from,
)
from solprop.nonlinear.ionisation import IntegrationPolicy, IonisationRate
from solprop.nonlinear.mixtures import electronic_response, responses_for
from solprop.nonlinear.optmat_raman import mixture_raman, raman_status
from solprop.nonlinear.optmat_rates import (
    OptmatRate,
    PeakFieldReport,
    bind_reference_field,
    peak_field_report,
    selected_rate,
)
from solprop.nonlinear.raman import (
    DampedOscillator,
    MolecularOptions,
    RamanResponse,
    RotationalLines,
    SilicaTensorResponse,
)
from solprop.nonlinear.responses import (
    DispersiveKerrResponse,
    KerrResponse,
    PlasmaResponse,
)
from solprop.nonlinear.tdse import TDSEAtomicWindow, TDSEExecution, TDSEGasResponse
from solprop.output.compatibility import RecordComparison, dotted
from solprop.output.hdf5 import (
    HDF5Output,
    ResumeState,
    _compare_cache,
    _resume_refusal,
    _stored_metadata,
    _stored_stat_shapes,
)
from solprop.output.hdf5 import resume as resume_state
from solprop.output.memory import MemoryOutput
from solprop.output.schema import (
    INTERPOLATED_RESUMES_ANNOTATION,
    PEAK_FIELDS_ANNOTATION,
    CheckpointOutput,
    FieldsChoice,
    GridRecord,
    ModeRecord,
    NoiseRecord,
    NoiseResolution,
    Output,
    PlaneStore,
    RunMetadata,
    StepLimits,
    StepperRecord,
    field_retention_refusal,
    recorded_input,
    serialise,
)
from solprop.output.snapshots import SnapshotController
from solprop.output.tdse_archive import TDSEArchive, TDSEArchiveWriter
from solprop.physdata import (
    Gas,
    Mixture,
    _optmat,
    as_gas,
    fused_silica,
    species,
)
from solprop.physdata.glasses import SilicaBundle
from solprop.physdata.mixtures import Pressure
from solprop.physdata.nonlinear_data import carriers_agree
from solprop.physdata.selection import (
    DISPERSIVE_POLICIES,
    MaterialLog,
    MaterialManifest,
    MaterialSelection,
)
from solprop.physdata.sellmeier import GasSusceptibility
from solprop.physdata.species import Species, built_in_population, default_species
from solprop.profiles import (
    PressureGradient,
    ZProfile,
    as_profile,
)
from solprop.pulses import (
    GaussianPulse,
    PlanarPulse,
    Pulse,
    PulseSum,
    SechPulse,
    SpatialPulse,
)
from solprop.spatial import SpatialDispersion, SpatialFlux, SpatialMedium, peak_field
from solprop.spatial_absorber import SpatialAbsorberDesign
from solprop.stats import (
    PLASMA_INTEGRATION_STATISTICS,
    StatsSpec,
    analytic_field,
    default_spec,
)
from solprop.stepper.controller import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.stepper.tableau import DOPRI5
from solprop.translate.coupled import (
    CF4_GROWTH_LIMIT,
    taper_identity,
    wall_identity,
)
from solprop.translate.coupling import frame_record
from solprop.translate.coupling import input_port as fill_port
from solprop.translate.envelope_plasma import (
    CycleIntegralData,
    assembly_record,
    cycle_moments,
)
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.flux import flux_record
from solprop.translate.general import (
    GeneralModalRoute,
    general_modal_route,
    modal_identity,
)
from solprop.translate.linop import power_weight_at
from solprop.translate.masks import (
    DEFAULT_ABSORBER_SCHEME,
    AbsorberScheme,
    noise_spectrum,
)
from solprop.translate.modal import DEFAULT_NODE_BUDGET
from solprop.translate.plasma_integration import (
    integration_record,
)
from solprop.translate.plasma_source import _BENCHMARK_GATED, plasma_source_record
from solprop.translate.precision import (
    SINGLE_RTOL_CAP,
    SINGLE_RTOL_FLOOR,
    Precision,
    PrecisionName,
    admit_precision,
    precision_named,
)
from solprop.translate.problem import (
    DEFAULT_COMPONENTS,
    KernelProblem,
    PropagatorMode,
    StepperSettings,
    TranslationReuse,
    assemble,
)
from solprop.translate.responses import KerrForm
from solprop.translate.tdse import AtomicCallLedger, TDSEBinding, refine_response_grid
from solprop.translate.tdse_source import TDSE_UNAVAILABLE_STATISTICS
from solprop.translate.transverse import (
    band_wavenumbers,
    checked_design,
    resolve_transverse,
)
from solprop.transverse_monitor import TransverseEdge

__all__ = [
    "BACKEND_STATISTICS",
    "CAPILLARY_RTOL",
    "DEFAULT_ATOL",
    "DEFAULT_N_SAVE",
    "DEFAULT_RTOL",
    "FREE_ATOL",
    "FREE_NOISE",
    "FREE_RTOL",
    "INTERIOR_LAUNCH_NOISE",
    "LOOSE_ADAMS_RTOL",
    "LOW_STORAGE_BACKENDS",
    "ORTHONORMALITY_TOLERANCE",
    "PLANAR_NOISE",
    "PLASMA_STATISTICS",
    "SINGLE_PRECISION_NOISE",
    "SLAB_NOISE",
    "SPATIAL_ATOL",
    "SPATIAL_RTOL",
    "TDSE_NOISE",
    "WAVEGUIDE_ATOL",
    "WAVEGUIDE_RTOL",
    "NoiseModel",
    "NoiseSetting",
    "Simulation",
    "SimulationRunner",
    "SimulationSettings",
    "check_derived_problem",
    "default_collar",
    "default_field_range",
    "default_radial_order",
    "metadata_for",
]

logger = logging.getLogger(__name__)

type _ResumeChoice = bool | Literal["interpolated", "force"]
"""What ``resume`` takes: no, yes, from an interpolated plane, or across compared
differences (SEP 42 section 5.5)."""

_RESUME_CHOICES: Final = (False, True, "interpolated", "force")


LOOSE_ADAMS_RTOL = 5e-2
"""The ``rtol`` from which an adaptive Adams run is warned on.

An error of five per cent or more allowed per step compounds through the method's
history, and its predictor-corrector estimate is unreliable there
(``docs/guide/numerics/choosing-a-stepper.md``).
"""

DEFAULT_RTOL = 1e-7
"""Relative tolerance a GNLSE run asks for unless told otherwise.

Set by SEP 37's convergence protocol on the supercontinuum of Dudley, Genty and Coen
(the benchmark suite's ``gnlse_raman_shock``): the loosest of three levels a decade
apart at which the ensemble-mean spectrum stays within 0.5 dB above -40 dB of its peak
and the coherence within 0.02, over paired seeds, with the default ``RungeKutta()``.
A noiseless soliton-recurrence study asks for its own, tighter value.
"""

DEFAULT_ATOL = 1e-14
"""Absolute tolerance, in the square root of watts, of the same default."""

DEFAULT_N_SAVE = 201
"""Planes a GNLSE run stores, counting the one at ``z = 0``."""

CAPILLARY_RTOL = 1e-6
"""Relative tolerance a capillary run asks for unless told otherwise.

The default of :meth:`Simulation.from_capillary` and
:func:`~solprop.driver.interface.prop_capillary` with their default ``Multistep()``.
SEP 37's convergence protocol on the benchmark suite's ``rdw`` case passes it, and a
noisy comb-to-continuum check needs it with this integrator (``1e-5`` fails the
coherence budget there). It is a starting point, not a
guarantee: a production run should calibrate it (``docs/dev/stepper.md`` section 3).
"""

WAVEGUIDE_RTOL = 1e-5
"""Relative tolerance a modal waveguide run asks for unless told otherwise.

The default of :meth:`Simulation.from_waveguide` and
:func:`~solprop.driver.interface.prop_waveguide` with their default ``RungeKutta()``,
set by SEP 37's convergence protocol on the benchmark suite's ``grin_nonlinear``
window. A starting
point, not a guarantee: a production run should calibrate it.
"""

SPATIAL_RTOL = 1e-5
"""Relative tolerance of the controller a radial or planar run takes by default.

Set by SEP 37's convergence protocol on the benchmark suite's ``radial_filament_256``
and ``planar`` cases. It applies only where the caller's
:class:`SimulationSettings` gives no controller.
"""

SPATIAL_ATOL = 0.0
"""Absolute tolerance of the same default controller: none, as the suite runs it."""

FREE_RTOL = 1e-4
"""Relative tolerance of the controller a Cartesian or line run takes by default.

With the default ``RungeKutta()`` (SEP 37): the Adams history's memory rules the
multistep method out for a large transverse state. On a self-focusing Kerr beam the
spectrum stays within 0.002 dB of a tight reference even at 1e-3; 1e-4 cuts the field
error by 2.5 times for a sixth more steps. Applies only where the caller's
:class:`SimulationSettings` gives no controller.
"""

FREE_ATOL = 0.0
"""Absolute tolerance of the same default controller."""

WAVEGUIDE_ATOL = 1e-10
"""Absolute tolerance of a modal waveguide run, in sqrt(W).

The companion of :data:`CAPILLARY_RTOL` and :data:`WAVEGUIDE_RTOL`. It was never
binding in SEP 37's measurements.
"""

PLASMA_STATISTICS = ("electron_density", "peak_ionisation_rate")
"""Backend statistics that read the plasma the run compiled; refused without one."""

BACKEND_STATISTICS = ("peak_intensity", "quadrature_error", *PLASMA_STATISTICS)
"""The statistics only a backend can compute, recorded by a capillary run by default.

Each needs something the kernel contract does not carry to a later pass: the quadrature
nodes the plasma was evaluated at, the mode set the axis needs, or the embedded
coarse-rule projection, which is gone once the step is accepted. The two plasma entries
are recorded only by a run that has a plasma response; the backend refuses them
otherwise, and a default that always asked for them would make ``plasma=False`` fail.
"""

ORTHONORMALITY_TOLERANCE = 1e-6
"""How far a given mode set may stray from orthonormality before it is refused.

The transform is one synthesis matrix and its dual, which represent an orthonormal
basis and nothing else, so a set that is not one is propagating a different problem
from the one it describes. The bound is loose because the overlaps are integrated
adaptively by :func:`scipy.integrate.dblquad`, whose own accuracy on a set of Bessel
profiles is around 1e-9.
"""

_MINIMUM_RADIAL_ORDER = 32
"""Floor on the default radial order; see :func:`default_radial_order`."""

_RADIAL_NODES_PER_MODE = 8
"""Radial nodes per mode in the default order; see :func:`default_radial_order`."""

_FIELD_RANGE_BELOW = 1e-3
"""Lower end of the default ionisation table, as a factor on the input's peak field."""

_FIELD_RANGE_ABOVE = 30.0
"""Upper end of the default ionisation table, as a factor on the input's peak field."""

_PRESSURE_FORM_ENTRIES = 2
"""Entries in either of the two pair forms of the ``pressure`` keyword."""

_EXECUTION_SETTINGS = (
    "backend",
    "resume_history",
    "tdse_execution",
    "fields",
    "field_planes",
    "boundary_check",
)
"""The settings read when a run executes, not when it is translated."""

NoiseModel = Literal["input", "modified"]
"""Which shot-noise model a run carries; ``None`` carries none.

The two models are described in ``docs/guide/media/noise.md``.
"""

NoiseSetting = Literal["default", "input", "modified"] | None
"""What a constructor's ``noise`` keyword takes (SEP 37).

A model, ``None`` for a noiseless run, or ``"default"``, the keyword's default, which
resolves to ``"input"`` wherever the run can carry noise and to none where it cannot,
recording which in the run's :class:`~solprop.output.schema.NoiseResolution`.
"""

SINGLE_PRECISION_NOISE = (
    "single precision carries no noise: binary32 rounding swamps a one-photon seed"
)
"""Why a default resolves to no noise under ``precision="single"`` (SEP 37)."""

PLANAR_NOISE = "the hybrid planar grid carries no noise: no photon measure is declared"
"""Why a planar run carries no noise (SEP 37)."""

SLAB_NOISE = (
    "a slab mode set carries no noise: no photon measure is declared for its "
    "amplitudes per unit width"
)
"""Why a waveguide run of slab modes carries no noise, as the planar grid does not.

A slab amplitude is in sqrt(W/m), and the one-photon draw is defined per mode in
sqrt(W); one photon per metre of width is a convention no derivation has fixed for
these modes (the line grid declares its own), so a default resolves to none and an
explicit model is refused.
"""

FREE_NOISE = (
    "a Cartesian or line run carries no noise by default: it is slow, and rarely about "
    "noise sensitivity"
)
"""Why a default resolves to no noise on the free-space interface (SEP 37)."""

TDSE_NOISE = (
    "a TDSE run carries no noise by default: it is expensive, rarely repeated as an "
    "ensemble, and its checks are coherent"
)
"""Why a default resolves to no noise beside the TDSE response (SEP 37)."""

INTERIOR_LAUNCH_NOISE = (
    "an interior launch takes its spectrum as given, which may already carry a draw"
)
"""Why a default noise is dropped at an interior launch (SEP 37)."""


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
        takes the interface's own (SEP 37): :data:`SPATIAL_RTOL` and
        :data:`SPATIAL_ATOL` for :meth:`Simulation.from_radial` and
        :meth:`Simulation.from_planar`, :data:`FREE_RTOL` and :data:`FREE_ATOL` for
        :meth:`Simulation.from_free`; under single precision the tolerance is at least
        :data:`~solprop.translate.precision.SINGLE_RTOL_FLOOR` and at most
        :data:`~solprop.translate.precision.SINGLE_RTOL_CAP`. An explicit controller
        always wins. The other fields are keyword-only.
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
        ``None``, the default, takes ``Multistep()`` under double precision, or
        ``RungeKutta()`` where the interface's builder sets it
        (:meth:`Simulation.from_gnlse`, :meth:`Simulation.from_waveguide` and
        :meth:`Simulation.from_free`; ``docs/guide/numerics/choosing-a-stepper.md``),
        and ``RungeKutta()`` under single precision, which refuses the Adams method
        until its single-precision survival is established (SEP 33, Q2); the stored
        value is the resolved method. ``Multistep()`` is the low-call exponential Adams
        method, which
        ``numpy_ref``, ``numba_cpu`` and ``native`` execute for supported
        configurations;
        ``RungeKutta()`` is the interaction-picture DOPRI5 march, which every backend
        executes and which holds a third of the Adams history's memory.
        ``LowStorageRungeKutta()`` replays `schedule` with RK4 in three registers and
        no error estimate (SEP 24); every backend executes it
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
        (``docs/dev/architecture.md`` section 13). ``None``, the default, takes 1 GiB
        on a free-space, line or planar grid and 64 MiB otherwise
        (:data:`~solprop.translate.modal.DEFAULT_SPATIAL_NODE_BUDGET`,
        :data:`~solprop.translate.modal.DEFAULT_NODE_BUDGET`).
    stats : solprop.stats.StatsSpec or None
        Which statistics to collect at every saved plane; ``None`` collects none.
    polarisation_taper : bool
        Whether the spectral collar tapers the nonlinear polarisation as well as
        absorbing the field through the linear operator. Default ``False``, which is
        :func:`solprop.translate.problem.assemble`'s: the taper costs a systematic
        error of the size of the field's content in the collar, and the rate absorber
        acts either way (``docs/guide/numerics/boundaries.md``).
    backend : str
        Registry name of the backend to execute the run. The default ``"auto"``
        resolves to ``"native"`` when installed, otherwise, with a warning, to the
        slower ``"numba_cpu"``, at settings construction; under single precision, to
        ``"mlx"`` (`precision`).
        The stored name is concrete, including in scan workers. An installed
        backend's import or construction errors never trigger a fallback.
    precision : {'double', 'single'}
        The arithmetic the run is held in (SEP 33,
        ``docs/guide/performance/single-precision.md``). ``"double"``, the default, is
        binary64 throughout and changes nothing.
        ``"single"`` holds the state, the stages, the transforms and the responses in
        binary32, with the controller and every host scalar in binary64. It admits
        ``backend="numpy_ref"`` (the single-precision oracle) and the experimental
        ``backend="mlx"``; refuses an adaptive ``rtol`` below
        :data:`~solprop.translate.precision.SINGLE_RTOL_FLOOR`, the Adams integrator
        and an explicit noise model (a default noise resolves to none); and is
        refused by ``numba_cpu`` and ``native``. ``backend="auto"`` resolves to
        ``"mlx"`` when MLX and a Metal GPU are present and is otherwise refused.
        Part of the hashed configuration; outputs are binary64 either way.
    resume_history : bool or None
        Whether the output keeps the Adams march's state at every save plane and
        terminal checkpoint, so that a resume continues the history -- its samples, its
        order and its proposed step -- as the uninterrupted run would, instead of
        starting it afresh at order one. ``None``, the default, keeps it while it is at
        most :data:`solprop.driver.run.MARCH_SNAPSHOT_AUTO_BYTES`; it is outside the
        resume key. Every backend's fused and exponential Adams engines, the native
        backend's resident one included, export and install their history; a march
        that cannot (the general path of a comparison method) restarts it and warns.
    fields : {'all', 'planes', 'none'}
        Which saved planes keep their field (SEP 22): every one, the default; those
        of `field_planes`; or none, for a run wanted as statistics alone. A plane
        without its field stores no field and no noise phase, and an in-memory output
        is charged and allocated for the kept fields only; the march is the same. A
        terminal checkpoint carries the field anyway, so a stopped run resumes, but a
        file whose last regular plane kept no field resumes only from one. Outside
        the resume key, like `resume_history`: it changes what is kept, not what is
        computed. In a scan, :func:`~solprop.driver.scan.run_scan`'s own `fields`
        decide, and a builder that sets this is refused. Beside
        ``LowStorageRungeKutta()`` the backend holds the last field-free plane's
        state, one more state volume charged at admission, so a step abandoned in
        flight still leaves a resumable checkpoint (SEP 22, Q14).
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
        runs again in :func:`solprop.translate.problem.assemble`; here it fires before a
        run has built anything. Also if the field retention is not valid, asks for
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
    node_budget: int | None = None
    stats: StatsSpec | None = None
    polarisation_taper: bool = False
    backend: str = "auto"
    integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None
    precision: PrecisionName = "double"
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
    _route_integrator: RungeKutta | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    """The binary64 default of the interface that built the settings, where it is not
    the settings' own ``Multistep()``: ``RungeKutta()`` for the GNLSE, waveguide and
    free-space interfaces (SEP 37). A copy made with :func:`dataclasses.replace`
    carries it, so a copy in the other precision takes the interface's default, not
    the settings' own, and the interface's choice never counts as named."""
    # Whether the caller named the integrator, set per instance in __post_init__.
    integrator_named = False

    def _planes_text(self) -> str:
        """Return the save planes summarised, for printing.

        Returns
        -------
        str
            How many planes there are and the positions they span.
        """
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

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        schedule = "adaptive" if self.schedule is None else type(self.schedule).__name__
        return "SimulationSettings", [
            ("save planes", self._planes_text()),
            (
                "tolerances",
                "the interface's default"
                if self.controller is None
                else f"rtol {self.controller.rtol:g}, atol {self.controller.atol:g}",
            ),
            ("steps", f"{schedule}, dz_max {engineering(self.dz_max, 'm')}"),
            ("shock term", self.shock),
            (
                "absorber",
                f"{_scheme(self)}" if self.absorber is not None else "none",
            ),
            ("statistics", "none" if self.stats is None else type(self.stats).__name__),
            ("backend", self.backend),
        ]

    def __post_init__(self) -> None:
        """Adopt a read-only plane array and check it.

        Raises
        ------
        DriverError
            If the planes are empty, not increasing, or not positive.
        """
        planes = np.ascontiguousarray(self.save_planes, dtype=np.float64).reshape(-1)
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
            raise DriverError(msg)
        planes.setflags(write=False)
        object.__setattr__(self, "save_planes", planes)
        refusal = field_retention_refusal(self.fields, self.field_planes)
        if refusal is not None:
            raise DriverError(refusal)
        object.__setattr__(self, "field_planes", tuple(self.field_planes))
        precision = _precision(self.precision)
        carried = (
            self.integrator is not None and self.integrator is self._default_integrator
        )
        object.__setattr__(
            self, "integrator_named", self.integrator is not None and not carried
        )
        if not self.integrator_named:
            default = (
                RungeKutta()
                if precision.single
                else self._route_integrator or Multistep()
            )
            if carried and type(self.integrator) is type(default):
                default = self.integrator
            object.__setattr__(self, "integrator", default)
            object.__setattr__(self, "_default_integrator", default)
        if precision.single:
            object.__setattr__(self, "backend", _single_backend(self.backend))
        if self.fields == "planes" and self.step_planes:
            msg = (
                "fields='planes' names saved planes by index, and step_planes saves "
                "the march's own steps as well, so the plane count is not known in "
                "advance; keep fields='all' or 'none' beside step_planes"
            )
            raise DriverError(msg)
        if self.tdse_execution is not None:
            _admit_tdse_execution(self, precision)
        # The annotation refuses another integrator first wherever the annotations are
        # checked at run time, as in the test suite.
        if not isinstance(
            self.integrator, Multistep | RungeKutta | LowStorageRungeKutta
        ):  # pragma: no cover
            msg = (
                "integrator must be Multistep(), RungeKutta() or "
                f"LowStorageRungeKutta(), got {self.integrator!r}"
            )
            raise DriverError(msg)
        if isinstance(self.integrator, LowStorageRungeKutta):
            backend = _admit_low_storage(self.integrator, self.backend, self)
            object.__setattr__(self, "backend", backend)
        # An Adams integrator resolves "auto" as the Runge-Kutta march does: every
        # backend steps the exponential formulation on every route, and the
        # interaction one on every route but a coupled flow, which it refuses by name
        # (SEP 16 phases 3 and 5).
        object.__setattr__(self, "backend", resolve_name(self.backend))
        # The precision's refusals come before the integrator's backend check, so a
        # single-precision run is told what single precision refuses -- the Adams
        # integrator -- rather than that a binary32 backend does not step it.
        try:
            admit_precision(
                precision,
                backend=self.backend,
                adams=isinstance(self.integrator, Multistep),
                rtol=getattr(self.controller, "rtol", None),
                schedule=self.schedule is not None,
            )
        except TranslationError as error:
            raise DriverError(str(error)) from error
        if isinstance(self.integrator, Multistep):
            _admit_integrator(self.integrator, self.backend)
            rtol = getattr(self.controller, "rtol", None)
            if self.schedule is None and rtol is not None and rtol >= LOOSE_ADAMS_RTOL:
                msg = (
                    f"rtol={rtol:g} is loose for the Adams method: an error of this "
                    "size allowed per step compounds through its history, and its "
                    "error estimate is unreliable there. Take rtol at most 1e-2, or "
                    "integrator=RungeKutta() for a rough look"
                )
                warnings.warn(msg, SolpropWarning, stacklevel=3)


def _admit_tdse_execution(settings: SimulationSettings, precision: Precision) -> None:
    """Refuse TDSE execution resources beside settings a TDSE run cannot step with.

    Each refusal names the setting at fault, in the order a caller would fix them:
    the precision, then the backend, then the integrator.

    Parameters
    ----------
    settings : SimulationSettings
        The settings being checked, whose ``tdse_execution`` is given.
    precision : solprop.translate.precision.Precision
        Their resolved precision.

    Raises
    ------
    DriverError
        If the resources are not a :class:`~solprop.nonlinear.tdse.TDSEExecution`,
        or the run is in single precision, on a backend other than ``numpy_ref`` or
        stepped by an integrator other than ``Multistep()``.
    """
    if not isinstance(settings.tdse_execution, TDSEExecution):
        msg = (
            "tdse_execution must be a TDSEExecution, got "
            f"{type(settings.tdse_execution).__name__}"
        )
        raise DriverError(msg)
    if precision.single:
        msg = (
            "precision='single' refuses a TDSE response: no backend runs it in "
            "binary32 (SEP 33 section 5.1); use precision='double'"
        )
        raise DriverError(msg)
    if settings.backend != "numpy_ref":
        msg = (
            f"tdse_execution requires an explicit backend='numpy_ref', the one backend "
            f"that runs the TDSE response; got backend={settings.backend!r}"
        )
        raise DriverError(msg)
    if not isinstance(settings.integrator, Multistep):
        msg = (
            "tdse_execution requires integrator=Multistep(), the Adams method the TDSE "
            f"response steps with; got {type(settings.integrator).__name__}()"
        )
        raise DriverError(msg)


def _default_rtol(default: float, precision: PrecisionName) -> float:
    """Return an interface's default relative tolerance in the run's precision.

    The one rule every interface's default follows, keyword and spatial alike. Binary64
    takes the interface's own. Binary32 takes the larger of it and
    :data:`~solprop.translate.precision.SINGLE_RTOL_FLOOR`, so it is never tighter than
    its binary64 twin nor than binary32 rounding can serve, and at most
    :data:`~solprop.translate.precision.SINGLE_RTOL_CAP`, the loosest value the
    single-precision survival record measured (the maintainers' ruling of 2026-10-28).
    Single precision so takes 1e-6 on the GNLSE and capillary interfaces, 1e-5 on the
    waveguide, radial and planar ones, and 1e-5 in free space.

    Parameters
    ----------
    default : float
        The interface's binary64 default.
    precision : {'double', 'single'}
        The run's precision.

    Returns
    -------
    float
        The default relative tolerance.

    Examples
    --------
    >>> from solprop.driver.simulation import _default_rtol
    >>> [_default_rtol(rtol, "single") for rtol in (1e-7, 1e-6, 1e-5, 1e-4)]
    [1e-06, 1e-06, 1e-05, 1e-05]
    >>> _default_rtol(1e-7, "double")
    1e-07
    """
    if precision != "single":
        return default
    return min(max(default, SINGLE_RTOL_FLOOR), SINGLE_RTOL_CAP)


def _quiet_collar(settings: SimulationSettings) -> bool:
    """Return whether a run's settings opt in to the quiet collar (ADR-057).

    Parameters
    ----------
    settings : SimulationSettings
        The settings.

    Returns
    -------
    bool
        True when the boundary check asks for it.
    """
    check = settings.boundary_check
    return check is not None and check.quiet_collar


def _tolerance(rtol: float | None, precision: PrecisionName, default: float) -> float:
    """Return a keyword interface's relative tolerance under its precision.

    Parameters
    ----------
    rtol : float or None
        The caller's tolerance; ``None`` takes the default.
    precision : {'double', 'single'}
        The run's precision.
    default : float
        The interface's binary64 default.

    Returns
    -------
    float
        `rtol` when given, which is never raised silently; otherwise the default
        :func:`_default_rtol` gives `default` in this precision.
    """
    if rtol is not None:
        return rtol
    return _default_rtol(default, precision)


def _precision(name: PrecisionName) -> Precision:
    """Return the record of a public precision value, refusing an unknown one.

    Parameters
    ----------
    name : {'double', 'single'}
        The settings' value.

    Returns
    -------
    solprop.translate.precision.Precision
        The record.

    Raises
    ------
    DriverError
        If `name` is not a public value.
    """
    if name not in ("double", "single"):  # pragma: no cover - refused by its annotation
        msg = f"precision must be 'double' or 'single', got {name!r}"
        raise DriverError(msg)
    return precision_named(name)


def _single_backend(backend: str) -> str:
    """Resolve ``"auto"`` for a single-precision run (SEP 33, Q8).

    ``"auto"`` resolves to ``"mlx"`` when MLX and a Metal device are present, and
    otherwise is refused: it never resolves to a binary64 backend, and
    never to ``numpy_ref``, the oracle, which is not a production backend. An explicit
    name passes through to :func:`~solprop.translate.precision.admit_precision`.

    Parameters
    ----------
    backend : str
        The requested backend name.

    Returns
    -------
    str
        The concrete backend name.

    Raises
    ------
    DriverError
        If `backend` is ``"auto"`` and no single-precision GPU backend is installed.
    """
    if backend != "auto":
        return backend
    if metal_available():
        return "mlx"
    msg = (
        "precision='single' with backend='auto' resolves to the mlx backend, which "
        "needs the mlx extra (pip install solprop[mlx]) on an Apple-silicon Mac with a "
        "Metal GPU; pass backend='numpy_ref' for the single-precision reference (the "
        "oracle, not a production backend) or use precision='double'"
    )
    raise DriverError(msg)


def _stepper_record(settings: StepperSettings) -> StepperRecord:
    """Return the run's stepper record, naming an Adams integrator where one steps it.

    The method is recorded in full, and the interval cap and start-up rule with the
    limits, so that a run and its resume hash the stepping they were taken with.

    Parameters
    ----------
    settings : solprop.translate.problem.StepperSettings
        The contract's stepper settings.

    Returns
    -------
    solprop.output.schema.StepperRecord
        The record.
    """
    return StepperRecord.from_settings(
        settings.tableau,
        settings.controller,
        schedule=settings.schedule,
        limits=StepLimits(
            dz_min=settings.dz_min,
            dz_max=settings.dz_max,
            dz_init=settings.dz_init,
            stretch=settings.stretch,
            interval_cap=settings.interval_cap,
            auto_init=settings.auto_init,
            interpolate_planes=settings.interpolate_planes,
            step_planes=settings.step_planes,
        ),
        method=None if settings.integrator is None else repr(settings.integrator),
    )


LOW_STORAGE_BACKENDS = ("native", "numba_cpu", "numpy_ref")
"""The backends that execute :class:`~solprop.stepper.low_storage.LowStorageRungeKutta`.

``numpy_ref`` from SEP 24's phase 1, ``numba_cpu`` from phase 2 and solprop-native from
phase 3, on every route the translation admits the march on; ``backend="auto"``
resolves as for every other integrator (:func:`solprop.backends.registry.resolve_name`:
the native backend when it is installed, otherwise ``numba_cpu``).
"""


def _admit_low_storage(
    integrator: LowStorageRungeKutta, backend: str, settings: SimulationSettings
) -> str:
    """Refuse what the low-storage march cannot run, before translation (SEP 24 5).

    The march replays a prescribed schedule, lands on its edges and accepts every
    step, so it needs a schedule and has no interpolated planes. Every registered
    backend executes it (:data:`LOW_STORAGE_BACKENDS`), so the backend passes as asked.

    Parameters
    ----------
    integrator : solprop.stepper.low_storage.LowStorageRungeKutta
        The requested march.
    backend : str
        The requested backend name, or ``"auto"``.
    settings : SimulationSettings
        The settings being checked, for the schedule and the planes.

    Returns
    -------
    str
        The backend to run, as requested: every registered backend executes the
        march (:data:`LOW_STORAGE_BACKENDS`), and ``"auto"`` resolves as for any
        integrator.

    Raises
    ------
    DriverError
        Naming the setting and its remedy.
    """
    if settings.schedule is None:
        msg = (
            f"the {integrator.name} march replays a prescribed schedule; pass "
            "schedule=FixedSchedule.uniform(...), or step adaptively with RungeKutta()"
        )
        raise DriverError(msg)
    if settings.interpolate_planes:
        msg = (
            f"the {integrator.name} march lands on its schedule's edges and has no "
            "dense output; drop interpolate_planes"
        )
        raise DriverError(msg)
    return backend


def _admit_integrator(integrator: Multistep, backend: str) -> None:
    """Refuse an Adams integrator on a backend that does not execute it.

    Refused here, before translation, so that no backend -- in particular a
    separately built one reading the contract by name -- can silently step the problem
    with its Runge-Kutta march instead (SEP 13 section 6.1).

    Parameters
    ----------
    integrator : solprop.stepper.multistep.Multistep
        The requested method.
    backend : str
        The requested backend name.

    Raises
    ------
    DriverError
        Naming the conflict and the Runge-Kutta march every backend executes.
    """
    if backend not in ("auto", "numpy_ref", "numba_cpu", "native"):
        msg = (
            f"the {integrator.name} integrator is implemented on numpy_ref, numba_cpu "
            f"and native only, and backend {backend!r} would step the problem with its "
            "Runge-Kutta march; pass one of those or 'auto', or "
            "integrator=RungeKutta()"
        )
        raise DriverError(msg)


def metadata_for(  # noqa: PLR0913 - the record's parts, each a distinct object
    problem: KernelProblem,
    grid: TimeGrid,
    *,
    modes: Sequence[Mode] = (),
    configuration: Mapping[str, object] | None = None,
    noise: NoiseRealisation | None = None,
    modal: Mapping[str, JsonValue] | None = None,
    materials: MaterialManifest | None = None,
    quadrature: TransverseQuadrature | None = None,
    frame: Frame | None = None,
    noise_resolution: NoiseResolution | None = None,
) -> RunMetadata:
    """Build the record that says what produced a run, and whether it may be resumed.

    Everything that describes *what is computed* goes into the digest
    :meth:`solprop.output.schema.RunMetadata.configuration_hash` takes -- the grid, the
    modes, the configuration, the noise and the stepper settings, plus the save planes
    and the statistics, which are read off the contract here -- and nothing that
    describes *when* or *with which build*, so a resume survives an unrelated commit or
    a NumPy upgrade.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem the run will step.
    grid : solprop.grid.TimeGrid
        The grid the state lives on.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order.
    configuration : collections.abc.Mapping, optional
        The physics objects that built the run -- responses, the input, profiles,
        boundaries -- recorded by :func:`solprop.output.schema.serialise`. Nothing is
        dropped: a callable is recorded as its ``repr`` and marked unreproducible, which
        also makes the digest unstable across processes, so prefer objects that carry
        their data (``docs/guide/setup/running.md``).
    noise : solprop.noise.NoiseRealisation or None, optional
        The realisation the run carries, recorded with its seed.
    modal : collections.abc.Mapping or None, optional
        The general modal identity, as
        :func:`solprop.translate.general.modal_identity` writes it: the channel
        labels, the table, metric, response and guide identities, the frame, the
        active band and the accessors the basis admits. ``None`` for every other
        route. It is part of the resume key.
    materials : solprop.physdata.selection.MaterialManifest or None, optional
        The run's material manifest
        (:func:`solprop.driver.material_output.material_manifest`); ``None`` is the
        empty manifest of a run of solprop's own materials. Its identities and
        policies are part of the resume key.
    quadrature : solprop.modes.quadrature.TransverseQuadrature or None, optional
        The transverse rule of a resolved modal run, recorded with its nodes and
        physical weights on the first mode's cross-section and part of the resume
        key (SEP 19). ``None`` for every other route.
    frame : solprop.linear.Frame or None, optional
        The run's frame, whose reference frequency is the origin of the linear frame
        term on an envelope grid; ``None`` is the grid's co-moving frame. It enters
        the plane-reference record (SEP 31), not the hash.
    noise_resolution : solprop.output.schema.NoiseResolution or None, optional
        How the noise model was chosen (SEP 37). Outside the hash.

    Returns
    -------
    solprop.output.schema.RunMetadata
        The record, ready for :meth:`solprop.output.schema.Output.open`.

    Examples
    --------
    >>> from solprop.driver.simulation import metadata_for
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> grid = EnvelopeGrid.from_limits(0.01, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0, gamma=0.11)
    >>> start = SechPulse(wavelength=835e-9, width=1e-13, peak_power=1.0).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-14),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> metadata = metadata_for(problem, grid, modes=[gnlse.mode])
    >>> metadata.simulation_kind, len(metadata.configuration_hash())
    ('mode_averaged', 64)
    """
    settings = problem.stepper
    record = dict(configuration or {})
    if bool(problem.responses.plasma_channels):
        # The equation and admission policy are identity; the measured residual
        # belongs to backend execution evidence, like the TDSE certificate. Its
        # last bits may change with the host FFT implementation.
        source = plasma_source_record(problem.grids)
        identity = {key: value for key, value in source.items() if key != "certificate"}
        if _BENCHMARK_GATED.get():
            identity["form"] = "gated_benchmark"
            record["_benchmark_plasma_source"] = "gated_nonresumable"
        record["plasma_source"] = identity
    # The planes and the statistics come from the contract rather than from the caller:
    # a resume continues at a plane index, so two runs that differ in either are not the
    # same run, and neither is carried by StepperRecord.
    record["save_planes"] = np.asarray(settings.save_planes, dtype=np.float64)
    record["statistics"] = problem.stats_spec
    measure = transverse_measure(modes) if modes else "area"
    geometry = problem.transform.geometry
    if geometry is not None and geometry.kind == "free" and geometry.shape[1] == 1:
        # A Cartesian line (one y row of unit measure) is per metre of y however the
        # simulation was built, not only through from_free.
        measure = "per_unit_width"
    if measure != "area":
        # A run uniform in y reports its powers per metre of y; processing and
        # plotting label them from this key (RunMetadata.transverse_measure). It is
        # written only where it is not the default, so every area run keeps the
        # digest it was stamped with.
        record["transverse_measure"] = measure
    if problem.responses.envelope_plasma is not None:
        record["statistic_rate_conventions"] = {
            name: "cycle-averaged"
            for name in ("peak_ionisation_rate", "species_peak_ionisation_rate")
            if name in problem.stats_spec.names or name in problem.stats_spec.periodic
        }
        record["envelope_plasma"] = assembly_record(
            problem.responses.envelope_plasma, problem.grids.omega
        )
    if problem.responses.plasma_integral.policy != "trapezoid":
        # The band rule's taps and policies (SEP 18, Q9): a resume that would
        # integrate the plasma differently is refused through the hash. Written only
        # for that rule, so that every trapezoid run keeps the digest it had.
        record["plasma_integration"] = integration_record(
            problem.responses.plasma_integral
        )
    material = record.get("medium")
    if isinstance(material, SpatialMedium):
        record["spatial_medium"] = material
        material = material.material
    if not isinstance(material, Mixture) and modes:
        material = getattr(modes[0], "mixture", None)
    if isinstance(material, Mixture):
        record["medium"] = material_record(material)
        record["mixture_processing"] = processing_record(problem, grid, material, modes)
    if problem.linear.separable is not None:
        # A Cartesian checkpoint must not resume with changed cutoff bins.
        record["spatial_arithmetic"] = "split-square-v1"
    geometry = problem.transform.geometry
    spatial_record: dict[str, object] | None = None
    if geometry is not None:
        # A hybrid planar column is a slab amplitude per y frequency: its x axis is the
        # slab rule's nodes, its spectrum the mode index and k_y, and its stored unit a
        # modal amplitude's.
        slab = geometry.kind == "slab"
        spatial_record = {
            "kind": geometry.kind,
            "shape": geometry.shape,
            "axes": geometry.axes,
            "axis_names": ("r",) if geometry.kind == "radial" else ("x", "y"),
            "axis_units": "m",
            "spectral_axes": geometry.spectral_axes,
            "spectral_axis_names": ("k_r",)
            if geometry.kind == "radial"
            else ("k_y",)
            if slab
            else ("k_x", "k_y"),
            "spectral_axis_units": "rad/m",
            "column_order": "r"
            if geometry.kind == "radial"
            else "p + n_p*iy"
            if slab
            else "ix + n_x*iy",
            "physical_field_units": "V/m",
            "stored_field_units": "V m"
            if geometry.kind == "radial"
            else "sqrt(W/m)"
            if slab
            else "V/m",
            "spatial_weights": geometry.spatial_weights,
            "spectral_weights": geometry.spectral_weights,
            "forward_scale": geometry.forward_scale,
            "inverse_scale": geometry.inverse_scale,
            "window": geometry.window,
            "absorber_rate": geometry.absorber_rate,
            "flux_model": "scalar-axial-admittance",
            "flux_plan": flux_record(problem.flux),
        }
    tdse_record = None if problem.tdse is None else dataclasses.asdict(problem.tdse)
    return RunMetadata(
        reference=_plane_reference_record(problem, grid, record, frame, modes, modal),
        transverse=None if problem.transverse is None else problem.transverse.record(),
        grid=GridRecord.from_grid(grid),
        modes=tuple(ModeRecord.from_mode(mode) for mode in modes),
        simulation_kind=(
            geometry.kind
            if geometry is not None
            else "mode_averaged"
            if problem.flags.mode_averaged
            else "multimode"
        ),
        configuration=record,
        stepper=_stepper_record(settings),
        noise=None if noise is None else NoiseRecord.from_realisation(noise),
        noise_resolution=noise_resolution,
        spatial=spatial_record,
        quadrature=None
        if quadrature is None or not modes
        else quadrature.record(modes[0].dimlimits(0.0)),
        modal=None if modal is None else dict(modal),
        materials=MaterialManifest() if materials is None else materials,
        initial_z=problem.initial.z,
        tdse=tdse_record,
    )


def _plane_reference_record(
    problem: KernelProblem,
    grid: TimeGrid,
    configuration: Mapping[str, object],
    frame: Frame | None,
    modes: Sequence[Mode],
    modal: Mapping[str, JsonValue] | None,
) -> dict[str, object]:
    """Return the plane-reference record of a run (SEP 31).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem, whose frame tables are integrated.
    grid : solprop.grid.TimeGrid
        Its temporal grid.
    configuration : collections.abc.Mapping
        The run's configuration; a ``coupling`` entry supplies the inherited origin
        and the parent's provenance, and a ``spatial_grid`` the transverse basis.
    frame : solprop.linear.Frame or None
        The run's frame, or ``None`` for the co-moving one.
    modes : collections.abc.Sequence
        The propagated modes of a modal run.
    modal : collections.abc.Mapping or None
        The general modal identity, for a general guide.

    Returns
    -------
    dict
        ``carrier``, ``origin``, ``parent``, ``frame`` and ``basis``, as
        :attr:`solprop.output.schema.RunMetadata.reference` documents them.
    """
    # The co-moving frame keeps the carrier phase exactly when a harmonic term is
    # present, and then measures its delay from zero frequency (`frame_origin`).
    if frame is None:
        frame = Frame.comoving(grid, thg=problem.grids.carrier_2w0 is not None)
    coupling = configuration.get("coupling")
    origin: object = PlaneReference().record()
    parent: object = None
    if isinstance(coupling, Mapping):
        coupled = cast("Mapping[str, object]", coupling)
        origin = coupled["origin"]
        parent = coupled["source"]
    return {
        "carrier": float(grid.omega0),
        "origin": origin,
        "parent": parent,
        "frame": frame_record(problem.linear, frame_origin(frame, grid.kind)),
        "basis": _plane_bases(problem, grid, configuration, modes, modal),
    }


def _plane_bases(
    problem: KernelProblem,
    grid: TimeGrid,
    configuration: Mapping[str, object],
    modes: Sequence[Mode],
    modal: Mapping[str, JsonValue] | None,
) -> dict[str, object]:
    """Return a run's route and its basis identity at each planned plane.

    A spatial basis is its transverse grid and a general one its checked reference,
    the same at every plane; an ordinary mode set's basis is evaluated at each plane
    (:func:`solprop.coupling.plane.modal_basis_identity`), because a guide may vary.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem: its launch position and save planes.
    grid : solprop.grid.TimeGrid
        Its temporal grid.
    configuration : collections.abc.Mapping
        The run's configuration, carrying a spatial run's ``spatial_grid``.
    modes : collections.abc.Sequence
        The propagated modes of a modal run.
    modal : collections.abc.Mapping or None
        The general modal identity.

    Returns
    -------
    dict
        ``route``, ``z`` (the launch plane first, then the save planes, m) and
        ``identity`` (one digest per plane, ``None`` on a planar run).
    """
    planes = [float(problem.initial.z), *map(float, problem.stepper.save_planes)]
    geometry = problem.transform.geometry
    identity: str | None
    if geometry is not None:
        route = {"radial": "radial", "free": "cartesian", "slab": "planar"}[
            geometry.kind
        ]
        space = configuration.get("spatial_grid")
        identity = (
            basis_identity(cast("Literal['radial']", route), space_record(space))
            if isinstance(space, RadialGrid | FreeGrid)
            else None
        )
        identities = [identity] * len(planes)
    elif modal is not None and modal.get("basis") == "general_modal":
        route = "general_modal"
        metric = cast("Mapping[str, JsonValue]", modal.get("metric") or {})
        record = {
            "channels": modal.get("channels"),
            "tables": modal.get("tables"),
            "guide": modal.get("guide"),
            "metric": {"kind": metric.get("kind"), "gauge_id": metric.get("gauge_id")},
        }
        identities = [basis_identity("general_modal", record)] * len(planes)
    else:
        route = "modal"
        identities = [modal_basis_identity(modes, grid, z) for z in planes]
    return {"route": route, "z": planes, "identity": identities}


def _one_material_context[**P, R](build: Callable[P, R]) -> Callable[P, R]:
    """Run a constructor inside one material evaluation context, the run's own.

    The simulation it builds adopts the context's log
    (:class:`solprop.physdata.selection.MaterialLog`), and its translation, record and
    run resume it, so that every range, status and dilute-scaling warning of its
    materials is raised once per run, however its work is spread in time.

    Parameters
    ----------
    build : callable
        A ``Simulation.from_*`` constructor.

    Returns
    -------
    callable
        The constructor, run inside the context.
    """

    @functools.wraps(build)
    def inside(*args: P.args, **kwargs: P.kwargs) -> R:
        """Build the simulation with the run's material context open.

        Parameters
        ----------
        *args : object
            The constructor's positional arguments.
        **kwargs : object
            Its keywords.

        Returns
        -------
        object
            What the constructor returns.
        """
        with _optmat.material_context():
            return build(*args, **kwargs)

    return inside


def _same_setting(old: object, new: object) -> bool:
    """Return whether two values of one settings field are the same setting.

    Parameters
    ----------
    old, new : object
        The field's current and proposed values.

    Returns
    -------
    bool
        True for the same object, equal arrays, or values that compare equal; an
        object compared by identity (``eq=False``) is the same only as itself, which
        is what :func:`dataclasses.replace` keeps for every field it is not given.
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


@dataclass(frozen=True)
class _AtomicContext:
    """Hold what preparing a TDSE run's atomic resources needs from the run.

    Attributes
    ----------
    handler : Output
        The optical output, not yet opened.
    state : ResumeState or None
        What a resumed file holds.
    metadata : RunMetadata
        The run's record.
    admission : MemoryAdmission
        The run's optical memory admission.
    """

    handler: Output
    state: ResumeState | None
    metadata: RunMetadata
    admission: MemoryAdmission


class Simulation(Displayed):
    """A run that can be translated once and executed many times.

    Construct it from the physics objects and a :class:`SimulationSettings`, or through
    :meth:`from_gnlse` for a GNLSE fibre. :meth:`translate` assembles the kernel problem
    and caches it, so a second :meth:`run` -- into another handler, or resuming a
    file -- reuses it. :attr:`settings` may be replaced until that first translation;
    after it only its ``backend`` may change.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The field or envelope grid.
    modes : solprop.modes.protocol.Mode or sequence of Mode
        The mode, or the ordered set of a multimode problem.
    responses : collections.abc.Sequence
        The declared nonlinear responses, as
        :func:`solprop.translate.responses.compile_responses` takes them.
    initial : solprop.pulses.Pulse or solprop.pulses.SpatialPulse or numpy.ndarray
        The input: a :class:`solprop.pulses.Pulse` -- anything with a ``build(grid)``
        method -- or an already-built spectrum of shape ``(n_modes, n_pol, n_omega)`` in
        the square root of watts and the internal convention. A pulse is the better
        choice: it is a small specification, while a built array is recorded in the
        metadata as itself.
    settings : SimulationSettings
        How to step it, what to record and which backend to use.
    initial_z : float, optional
        Absolute position of an array input, m, default zero. Every requested save
        plane must follow it. An interior launch keeps the original profiles and
        propagation frame; it requires a modal array input, no noise and no fixed
        step schedule.
    noise : solprop.noise.NoiseRealisation or None, optional
        A shot-noise realisation to carry along the run (``docs/guide/media/noise.md``).
    noise_resolution : solprop.output.schema.NoiseResolution or None, optional
        How the noise model was chosen, as the constructors record it (SEP 37); ``None``
        records the realisation as asked for explicitly.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name, for the ``density``, ``pressure`` and
        ``core_radius`` statistics. Passed to the backend, which computes them.
    configuration : collections.abc.Mapping, optional
        Extra provenance to record beside what the simulation can see for itself, such
        as the coefficients a mapping came from.
    spatial_grid : RadialGrid, FreeGrid, PlanarGrid or None, optional
        Transverse grid for spatial propagation; use an empty mode sequence (a
        :class:`solprop.modes.slab.PlanarGrid` carries its own slab modes).
    medium : solprop.spatial.SpatialMedium or None, optional
        Bulk material, required with a spatial grid.
    frame : solprop.linear.Frame or None, optional
        Propagation frame. None uses the grid and responses' normal default.
    general : solprop.translate.general.GeneralModalRoute or None, optional
        The seam leaves of one general modal guide. Built by :meth:`from_waveguide` from
        a ``provider=``; with it the run takes the general modal route, whose responses
        are the guide's material regions' and whose declared `responses` must therefore
        be empty.
    guide : solprop.modes.guide.ModalGuide or None, optional
        The general modal guide `general` was built from, kept so that the run's
        material manifest finds what its regions hold (their laws and delayed
        responses); it enters no translation. Set by :meth:`from_waveguide`.
    wall : solprop.modes.capillary.CoherentWall or None, optional
        The declaration of a coherent wall block. Built by :meth:`from_capillary`
        from a ``wall=``; with it the mode set must be the contiguous ``HE_11`` to
        ``HE_1N`` block -- or, with ``impedance="te_tm"``, one azimuthal order -- on a
        constant or a varying guide, the cumulative operator carries the
        mode-independent carrier alone and the modal part of the flow becomes
        ``linear.coupled_flow``, which all three backends execute (a
        ``method="cf4"`` wall ``numpy_ref`` alone).
    taper_coupling : bool, optional
        Carry the **taper coupling** of a multimode capillary whose radius varies along
        z: the connection ``<psi_m|d_z psi_n>`` between the ``HE_1m`` channels, lossless
        or not, in a coupled flow beside each channel's own propagation constant.
        ``True``, the default; ``False`` is the explicit opt-out to the **adiabatic**
        route, on which each channel follows its local mode and no power is exchanged.
        No effect on a set the coupling does not apply to
        (:func:`solprop.modes.capillary.taper_coupling_block`); ``False`` beside a
        `wall` is refused, because the wall carries its connection.
    materials : solprop.physdata.selection.MaterialSelection or None, optional
        The material selection the run was built with, recorded in its material
        manifest with every optmat-backed object the run carries.
        The objects themselves are taken as they are: the ``from_*`` constructors
        resolve the built-in materials under the selection. ``None`` records none.

    Raises
    ------
    DriverError
        If `initial` is neither an array nor something with a ``build`` method.

    Examples
    --------
    >>> from solprop.driver import Simulation, SimulationSettings
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> grid = EnvelopeGrid.from_limits(0.01, 835e-9, (500e-9, 2e-6), 1.2e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0, gamma=0.11,
    ...                      shock=False)
    >>> simulation = Simulation(
    ...     grid, gnlse.mode, [gnlse.kerr],
    ...     SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8),
    ...     settings=SimulationSettings(
    ...         controller=PIController(rtol=1e-6, atol=1e-16),
    ...         save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...         density=1.0, shock=False, prefactor_index=1.0,
    ...     ),
    ... )
    >>> handler = simulation.run()
    >>> handler.n_saved, simulation.translate().shape
    (2, (1, 1, 1, 1024))
    """

    # How a constructor that drew the noise without a seed draws it again from a given
    # one: set per instance by `_redrawable` and carried to a run derived with its
    # noise kept (`_rebound_noise`), read by `_reseed`.
    _noise_redraw: Callable[[int | None], NoiseRealisation | None] | None = None

    def __init__(  # noqa: PLR0913 - the physics of a run, each a distinct object
        self,
        grid: TimeGrid,
        modes: Mode | Sequence[Mode],
        responses: Sequence[object],
        initial: Pulse | SpatialPulse | np.ndarray,
        *,
        settings: SimulationSettings,
        initial_z: float = 0.0,
        noise: NoiseRealisation | None = None,
        noise_resolution: NoiseResolution | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        configuration: Mapping[str, object] | None = None,
        spatial_grid: RadialGrid | FreeGrid | PlanarGrid | None = None,
        medium: SpatialMedium | None = None,
        frame: Frame | None = None,
        general: GeneralModalRoute | None = None,
        guide: ModalGuide | None = None,
        wall: CoherentWall | None = None,
        taper_coupling: bool = True,
        materials: MaterialSelection | None = None,
    ) -> None:
        """Adopt the physics objects and the settings; translate nothing yet."""
        _check_selection_carrier(materials, grid.reference_wavelength)
        self._base_grid = grid
        declared = tuple(responses)
        _check_electronic_policy(materials, declared)
        grid, settings = _resolve_tdse_grid(grid, declared, settings)
        self.grid = grid
        self.materials = materials
        # The run's material log: the one its constructor opened, or a new one, which
        # its translation, record and run resume (`_one_material_context`).
        self._material_log = _optmat.open_log() or MaterialLog()
        self.general = general
        self.guide = guide
        self.wall = wall
        self.taper_coupling = taper_coupling
        self.modes: tuple[Mode, ...] = (
            tuple(modes) if isinstance(modes, Sequence) else (modes,)
        )
        self.responses = declared
        self.initial = initial
        self.initial_z = initial_z
        self._problem: KernelProblem | None = None
        self.settings = settings
        if settings.controller is None:
            msg = (
                "settings.controller is None: Simulation.from_radial and "
                "Simulation.from_planar supply their interface's default tolerance, "
                "but a Simulation built directly needs a PIController"
            )
            raise DriverError(msg)
        self.noise = noise
        self.noise_resolution = (
            NoiseResolution(
                None if noise is None else noise.model,
                None if noise is None else noise.model,
            )
            if noise_resolution is None
            else noise_resolution
        )
        self.profiles = dict(profiles or {})
        self.configuration = dict(configuration or {})
        if (spatial_grid is None) != (medium is None):
            msg = "spatial_grid and medium must be supplied together"
            raise DriverError(msg)
        if spatial_grid is not None and self.modes:
            msg = "spatial propagation requires an empty mode sequence"
            raise DriverError(msg)
        self.spatial_grid = spatial_grid
        self.medium = medium
        self.frame = frame
        _check_initial_position(self, settings)
        _check_tdse_execution(self, settings)
        self._metadata: RunMetadata | None = None
        self._mixture_selection: ResponseSelection | None = None
        self._peak_fields: tuple[PeakFieldReport, ...] = ()
        self._reuse: TranslationReuse | None = None
        if isinstance(initial, SpatialPulse):
            if spatial_grid is None or medium is None:
                msg = "a SpatialPulse requires spatial_grid and medium"
                raise DriverError(msg)
            if self.frame is None:
                self.frame = Frame.comoving(
                    grid, thg=any(bool(getattr(one, "thg", False)) for one in responses)
                )
            self._field0 = initial.build(
                grid, spatial_grid, medium=medium, z=initial_z, frame=self.frame
            )
        else:
            self._field0 = _built(initial, grid)

    @property
    def settings(self) -> SimulationSettings:
        """Return how the run steps, records and executes; fixed once translated.

        Assigning a new :class:`SimulationSettings` before the first :meth:`translate`
        (or :meth:`run`) takes effect. After it the translated problem is cached, so an
        assignment that changes anything the translation consumed is refused, because
        it could no longer change the run; the ``backend`` and ``tdse_execution``
        resources, which are read when the run executes, may still change.

        Returns
        -------
        SimulationSettings
            The settings the run is, or will be, translated with.
        """
        return self._settings

    @settings.setter
    def settings(self, settings: SimulationSettings) -> None:
        """Replace settings; after translation, execution resources alone.

        Parameters
        ----------
        settings : SimulationSettings
            The replacement.

        Raises
        ------
        DriverError
            If the simulation has already been translated and the replacement changes
            a setting other than ``backend``, ``resume_history``, ``fields``,
            ``field_planes``, ``tdse_execution`` or ``boundary_check``, naming them.
        """
        if self._problem is not None:
            changed = [
                one.name
                for one in dataclasses.fields(SimulationSettings)
                if one.name not in _EXECUTION_SETTINGS
                and not one.name.startswith("_")
                and not _same_setting(
                    getattr(self._settings, one.name), getattr(settings, one.name)
                )
            ]
            if changed:
                msg = (
                    f"this Simulation is already translated, so a new "
                    f"{', '.join(changed)} would be ignored; set the settings before "
                    "the first translate() or run(), or build a new Simulation (only "
                    "the backend, resume_history, fields, field_planes, the TDSE "
                    "execution resources and boundary_check may change after "
                    "translation)"
                )
                raise DriverError(msg)
        if hasattr(self, "noise"):
            _check_initial_position(self, settings)
            _check_tdse_execution(self, settings)
            if _quiet_collar(settings) != _quiet_collar(self._settings):
                # The opt-in is part of the resume key: the record is built again.
                self._metadata = None
        self._settings = settings

    def _kind(self) -> str:
        """Return what kind of run this is, for printing.

        Returns
        -------
        str
            ``"modal"`` with its mode count, the guide's channel count, or the
            spatial grid's class.
        """
        if self.spatial_grid is not None:
            return type(self.spatial_grid).__name__
        count = len(self.modes)
        if self.general is not None:
            return f"{count} guide channel" + ("" if count == 1 else "s")
        return f"{count} mode" if count == 1 else f"{count} modes"

    def __repr__(self) -> str:
        """Return a one-line description: the grid, the geometry and the responses.

        Returns
        -------
        str
            For example ``Simulation(FieldGrid(...), 2 modes, responses=[Kerr, Plasma],
            backend=auto)``.
        """
        responses = ", ".join(type(response).__name__ for response in self.responses)
        return line(
            "Simulation",
            [
                ("", repr(self.grid)),
                ("", self._kind()),
                ("responses", f"[{responses}]"),
                ("backend", self.settings.backend),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        names = [type(response).__name__ for response in self.responses]
        initial = (
            summarise(self.initial)
            if isinstance(self.initial, np.ndarray)
            else repr(self.initial)
        )
        rows: list[tuple[str, object]] = [("grid", repr(self.grid))]
        if self.spatial_grid is not None:
            rows.append(("space", repr(self.spatial_grid)))
        if self.general is not None:
            rows.append(("guide", repr(self.general)))
        rows.extend(
            (f"mode {index}", repr(mode)) for index, mode in enumerate(self.modes)
        )
        rows.extend(
            [
                (
                    "responses",
                    ", ".join(names)
                    or ("the guide's regions" if self.general is not None else "none"),
                ),
                ("input", initial),
                (
                    "noise",
                    "none"
                    if self.noise is None
                    else f"{self.noise.model} model, seed {self.noise.seed}",
                ),
                ("settings", str(self.settings)),
            ]
        )
        return f"Simulation: {self._kind()}", rows

    def translate(self) -> KernelProblem:
        """Assemble the kernel problem, once, and return it.

        Everything expensive about a run that does not depend on the state happens here
        (``docs/dev/architecture.md`` section 7): the cumulative linear operator, the
        masks, the modal transform and the compiled responses. The result is cached, so
        two runs of the same simulation translate once.

        Returns
        -------
        solprop.translate.problem.KernelProblem
            Validated, versioned and ready for a backend.

        Examples
        --------
        >>> from solprop.driver import Simulation
        >>> simulation = Simulation.from_gnlse(
        ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
        ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
        ...     width=100e-15, peak_power=9.8, shape="sech", n_save=3,
        ... )
        >>> problem = simulation.translate()
        >>> problem.shape, simulation.translate() is problem
        ((1, 1, 1, 768), True)
        """
        if self._problem is None:
            settings = self.settings
            if (
                self._reuse is None
                and self.spatial_grid is None
                and self.general is None
                and self.wall is None
                and self.taper_coupling
            ):
                _warn_adiabatic_taper(self.modes)
            with _optmat.material_context(self._material_log):
                self._problem = assemble(
                    self.grid,
                    self.modes if self.spatial_grid is None else None,
                    list(self.responses),
                    self._field0,
                    initial_z=self.initial_z,
                    # Simulation.__init__ refuses a missing controller.
                    controller=cast("PIController", settings.controller),
                    save_planes=settings.save_planes,
                    dz_max=settings.dz_max,
                    dz_init=settings.dz_init,
                    dz_min=settings.dz_min,
                    stretch=settings.stretch,
                    auto_init=settings.auto_init,
                    interval_cap=settings.interval_cap,
                    interpolate_planes=settings.interpolate_planes,
                    step_planes=settings.step_planes,
                    # The contract carries the Runge-Kutta march as its tableau and
                    # no integrator; an Adams method or the low-storage march as the
                    # integrator, beside an unused DOPRI5.
                    tableau=(
                        settings.integrator.tableau
                        if isinstance(settings.integrator, RungeKutta)
                        else DOPRI5
                    ),
                    integrator=(
                        None
                        if isinstance(settings.integrator, RungeKutta)
                        else settings.integrator
                    ),
                    schedule=settings.schedule,
                    propagator_mode=settings.propagator_mode,
                    density=settings.density,
                    noise=self.noise,
                    absorber=settings.absorber,
                    absorber_scheme=_scheme(settings),
                    quadrature=settings.quadrature,
                    components=settings.components,
                    ionisation_field_range=(
                        settings.ionisation_field_range if self._reuse is None else None
                    ),
                    node_budget=settings.node_budget,
                    stats_spec=settings.stats,
                    shock=settings.shock,
                    prefactor_index=settings.prefactor_index,
                    polarisation_taper=settings.polarisation_taper,
                    spatial_grid=self.spatial_grid,
                    medium=self.medium,
                    frame=self.frame,
                    general=self.general,
                    wall=self.wall,
                    taper_coupling=self.taper_coupling,
                    reuse=self._reuse,
                    evanescent_rate=settings.evanescent_rate,
                    spatial_absorber=_transverse_specification(
                        settings.spatial_absorber, self.spatial_grid
                    ),
                    adiabatic_accuracy=settings.adiabatic_accuracy,
                    precision=precision_named(settings.precision),
                )
        return self._problem

    def input_port(self, z: float | None = None) -> InputPort:
        """Return what this simulation can receive at its entrance (SEP 31).

        The port is the receiving side of a transfer between simulations: its route,
        temporal lattice and basis with their identities, the entrance plane and the
        frame accumulated there, the native metric at the entrance, the reference
        index of its entrance medium, the coupling models it admits and what they
        need: an ordinary mode set's scalar profiles, or a general guide's E/H
        reference at its nodes and its regions' contrast. A launch computed for it
        (:func:`solprop.coupling.coupler.couple`) is accepted by :meth:`with_input`,
        and a launch computed for any other port is refused.

        Parameters
        ----------
        z : float or None, optional
            An interior entrance in m, for a modal run continued from inside its own
            path; None is the simulation's launch position.

        Returns
        -------
        solprop.coupling.InputPort
            The port and the models it admits.

        Raises
        ------
        DriverError
            For a hybrid planar run, whose slab columns are not exchanged in this
            release, or an interior entrance outside the run.
        """
        problem = self.translate()
        metadata = self.metadata()
        reference = cast("Mapping[str, object]", metadata.reference)
        bases = cast("Mapping[str, object]", reference["basis"])
        route = cast("str", bases["route"])
        if route == "planar":
            msg = (
                "a hybrid planar run's columns are slab amplitudes per y frequency; "
                "planar planes are not exchanged in this release"
            )
            raise DriverError(msg)
        if z is not None and not (
            math.isfinite(z) and 0.0 <= z < float(np.max(self.settings.save_planes))
        ):
            msg = f"an interior entrance lies inside the run, got z = {z!r} m"
            raise DriverError(msg)
        identity = cast("list[str]", bases["identity"])[0]
        if route == "modal" and z is not None:
            identity = modal_basis_identity(self.modes, self.grid, z)
        labels = tuple(mode.channel_label.display_label for mode in self.modes)
        space = (
            self.spatial_grid
            if isinstance(self.spatial_grid, RadialGrid | FreeGrid)
            else None
        )
        basis = PlaneBasis(
            cast("Literal['modal', 'general_modal', 'radial', 'cartesian']", route),
            identity,
            labels,
            space,
        )
        return fill_port(
            problem,
            self.grid,
            route=basis.kind,
            basis=basis,
            frame=cast("dict[str, object] | None", reference["frame"]),
            medium=self.medium,
            tables=None if self.general is None else self.general.tables,
            z=z,
            modes=tuple(self.modes) if route == "modal" else (),
            guide=self.guide if isinstance(self.guide, ModalGuide) else None,
        )

    def with_input(
        self,
        initial: Pulse | SpatialPulse | np.ndarray | CouplingLaunch,
        *,
        noise: NoiseRealisation | Literal["keep"] | None = "keep",
        initial_z: float | None = None,
        configuration: Mapping[str, object] | None = None,
    ) -> "Simulation":  # noqa: UP037 - runtime hook evaluates the class body
        """Replace the input while sharing the translated propagation problem.

        Only what changed is validated (SEP 22 section 5.2): the replaced input's
        shape, dtype and one finiteness pass, and the replaced noise's, with every
        other leaf of the translated problem shared by identity rather than scanned
        again. A new `initial_z`, which changes the save planes, also keeps the
        stepper's checks: the launch before the first plane, and a replayed schedule
        containing every plane. An array input above
        :data:`~solprop.output.schema.INLINE_ARRAY_LIMIT` elements is recorded by its
        digest (:class:`~solprop.output.schema.ArrayDigest`), not inline.

        Parameters
        ----------
        initial : Pulse, SpatialPulse, numpy.ndarray or solprop.coupling.CouplingLaunch
            Pulse or internal-convention spectrum on the existing grid. The built
            shape must match the original input, including modes and polarisation.
            Modal amplitudes are in sqrt(W); spatial units follow the spatial grid.
            A :class:`~solprop.coupling.port.CouplingLaunch` (SEP 31) must be bound to
            this simulation's :meth:`input_port` -- its route, lattice, basis,
            entrance and metric -- or it is refused whatever its shape; its
            entrance sets the launch position, and its provenance and inherited
            reference origin are recorded as ``configuration["coupling"]``.
        noise : NoiseRealisation, {'keep'} or None, optional
            Keep the existing noise, replace it explicitly, or disable it with None.
            Noise spectra must match the existing grid and input shape. A launch
            carries the realised field of its source, so for a launch ``'keep'``
            means no noise: no second floor is drawn unless a realisation is given.
            Kept noise that was drawn without a seed can still be drawn again from
            the seed a resume's file or a scan point supplies.
        initial_z : float or None, optional
            Absolute launch position in metres. None retains this simulation's
            launch. An explicit position keeps only existing save planes beyond
            it, preserving all absolute profiles, propagation tables and the frame.
        configuration : collections.abc.Mapping or None, optional
            Caller keys merged into the derived run's configuration record, and so
            into its configuration hash: the delay of a scan, the label of an input.
            A key the run already records is refused.

        Returns
        -------
        Simulation
            Translated simulation sharing all propagation tables. Metadata is
            rebuilt for the replacement input and noise. Existing ionisation
            table ranges remain in force, and so does the reference field of an
            optmat rate's constant energy per event, which is the original
            input's peak; use a fresh translation to change them.

        Raises
        ------
        DriverError
            If the new field or noise shape differs or any sample is not finite, a
            `configuration` key is one the run records itself, or a launch is bound
            to another port or given beside an `initial_z`.

        Warns
        -----
        MaterialRangeWarning
            Once per run, when the new input's peak exceeds an optmat rate's maximum
            field, or the input is not linearly polarised at a Keldysh parameter
            above one there (SEP 3, D12).
        """
        if isinstance(noise, str) and noise != "keep":
            raise DriverError("noise must be keep, None, or a NoiseRealisation")
        coupling = None
        if isinstance(initial, CouplingLaunch):
            if initial_z is not None:
                msg = "a launch carries its own entrance: give no initial_z beside it"
                raise DriverError(msg)
            entrance = initial.port.geometry.z
            interior = None if entrance == self.initial_z else entrance
            refusal = self.input_port(interior).accepts(initial.port)
            if refusal is not None:
                msg = f"this launch was computed for another input port: {refusal}"
                raise DriverError(msg)
            coupling = initial.record()
            initial_z = interior
            noise = None if isinstance(noise, str) else noise
            initial = initial.internal()
        original = self.translate()
        launch_z = self.initial_z if initial_z is None else initial_z
        settings = self.settings
        if initial_z is not None:
            remaining = np.asarray(settings.save_planes)
            remaining = remaining[remaining > launch_z]
            if not np.isfinite(launch_z) or launch_z < 0 or remaining.size == 0:
                raise DriverError(
                    "initial_z must be finite, nonnegative and before the end"
                )
            settings = dataclasses.replace(settings, save_planes=remaining)
        replacement_noise = self.noise if isinstance(noise, str) else noise
        resolution = (
            self.noise_resolution
            if isinstance(noise, str)
            else NoiseResolution(
                None if noise is None else noise.model,
                None if noise is None else noise.model,
            )
        )
        if (
            isinstance(noise, str)
            and launch_z > 0
            and self.noise_resolution.requested == "default"
        ):
            # SEP 37: a default draw is not added at an interior launch, whose given
            # spectrum may already carry the realisation of an earlier run.
            replacement_noise = None
            resolution = NoiseResolution("default", None, INTERIOR_LAUNCH_NOISE)
        recorded = self._derived_configuration(initial, configuration, coupling)
        updated = Simulation(
            self._base_grid,
            self.modes,
            self.responses,
            initial,
            settings=settings,
            initial_z=launch_z,
            noise=replacement_noise,
            noise_resolution=resolution,
            profiles=self.profiles,
            configuration=recorded,
            spatial_grid=self.spatial_grid,
            medium=self.medium,
            frame=self.frame,
            general=self.general,
            guide=self.guide,
            wall=self.wall,
            taper_coupling=self.taper_coupling,
            materials=self.materials,
        )
        # A derived run starts from what its parent's run has warned and recorded,
        # and keeps its own from here: its siblings never see each other's.
        updated._material_log = self._material_log.copy()
        if isinstance(noise, str) and replacement_noise is not None:
            updated._noise_redraw = self._noise_redraw
        self._check_replaced_input(
            updated._field0, None if isinstance(noise, str) else replacement_noise
        )
        record = dataclasses.replace(
            original.initial,
            z=launch_z,
            field0=np.asarray(updated._field0[np.newaxis], dtype=np.complex128),
        )
        if original.transform.geometry is None:
            record = dataclasses.replace(
                record,
                field0=power_amplitude(
                    record.field0, power_weight_at(original, launch_z), inverse=True
                ),
            )
        updated._problem = _with_noise(
            dataclasses.replace(
                original,
                initial=record,
                stepper=dataclasses.replace(
                    original.stepper, save_planes=settings.save_planes
                ),
            ),
            replacement_noise,
        )
        updated._problem = dataclasses.replace(
            updated._problem,
            transverse=_rechecked(self, original, updated._problem),
        )
        if initial_z is not None:
            # A new launch changes the save planes: the launch must precede them, and
            # a replayed schedule contain them (SEP 22 section 5.2).
            updated._problem.validate_launch()
        updated._mixture_selection = self._mixture_selection
        with _optmat.material_context(updated._material_log):
            _check_rate_inputs(updated, initial)
        return updated

    def _derived_configuration(
        self,
        initial: Pulse | SpatialPulse | np.ndarray,
        configuration: Mapping[str, object] | None,
        coupling: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Return a derived run's extra configuration: this run's, the caller's keys.

        Parameters
        ----------
        initial : Pulse, SpatialPulse or numpy.ndarray
            The replacement input.
        configuration : collections.abc.Mapping or None
            The caller's keys.
        coupling : collections.abc.Mapping or None, optional
            The provenance of a coupling launch, recorded as ``coupling``.

        Returns
        -------
        dict
            The configuration the derived simulation is built with.

        Raises
        ------
        DriverError
            If a caller's key is one the run records itself.
        """
        recorded = dict(self.configuration)
        # Nor does the constructor's incident-beam account describe it, nor a
        # coupling that launched this run.
        recorded.pop("launch", None)
        recorded.pop("coupling", None)
        # A constructor's pulse recipe must never describe a replacement spectrum.
        # Arrays carry their exact input in metadata, without an inferred recipe.
        if "pulse" in recorded:
            recorded.pop("pulse")
            if not isinstance(initial, np.ndarray):
                recorded["pulse"] = initial
        if configuration:
            clash = sorted(
                set(configuration) & ({"coupling"} | set(self.metadata().configuration))
            )
            if clash:
                msg = (
                    f"configuration keys {clash} are recorded by the run itself; "
                    "name the caller's keys differently"
                )
                raise DriverError(msg)
            recorded.update(configuration)
        if coupling is not None:
            recorded["coupling"] = dict(coupling)
        return recorded

    def _check_replaced_input(
        self,
        field0: Complex[np.ndarray, "n_modes n_pol n_omega"],
        noise: NoiseRealisation | None,
    ) -> None:
        """Validate a replaced input alone: its shape and one finiteness pass.

        Every other leaf of the derived problem is this one's, shared by identity and
        validated when it was translated (SEP 22 section 5.2); the dtype is fixed by
        the build.

        Parameters
        ----------
        field0 : numpy.ndarray
            The replacement's built spectrum.
        noise : solprop.noise.NoiseRealisation or None
            A replacement realisation, or ``None`` when the noise is kept or removed.

        Raises
        ------
        DriverError
            If a shape differs from this run's input or a sample is not finite.
        """
        if field0.shape != self._field0.shape:
            raise DriverError("with_input requires the existing mode and grid shape")
        if not np.all(np.isfinite(field0)):
            raise DriverError("with_input requires a finite initial field")
        if noise is None:
            return
        if noise.spectrum.shape != self._field0.shape:
            raise DriverError("replacement noise must match the input field shape")
        if not np.all(np.isfinite(noise.spectrum)):
            raise DriverError("with_input requires a finite noise realisation")

    def runner(self, *, memory_budget: int | str | None = None) -> "SimulationRunner":  # noqa: UP037 - defined below
        """Return a runner that keeps this translation and one backend for many inputs.

        Translates (or reuses the cached translation), admits memory and constructs
        the backend once; each :meth:`SimulationRunner.run` then runs one input on it
        (SEP 22 section 5.2). Close it, or use it in a ``with`` block.

        Parameters
        ----------
        memory_budget : int or str or None, optional
            The budget the backend is admitted against, as for :meth:`run`.

        Returns
        -------
        SimulationRunner
            The open runner.

        Raises
        ------
        DriverError
            If the run has a TDSE response.
        MemoryBudgetError
            If the backend does not fit its budget.
        """
        return SimulationRunner(self, memory_budget=memory_budget)

    def with_mixture(self, mixture: Mixture) -> "Simulation":  # noqa: UP037 - runtime hook evaluates the class body
        """Update material paths while preserving the run's geometry and input.

        Rate tables and unchanged undamped Raman kernels are shared by identity.
        Expansion terms are reused only for the identical sampled function with
        covered parameter ranges; changing fill fractions invalidates a pressure
        expansion. Every replacement weight and density or decay table is rebuilt.

        Parameters
        ----------
        mixture : Mixture
            New paths and populations with the same ordered labels and species data.
            Public capillary response selections are retained; changed temperature
            or Raman component options rebuild the affected molecular models.

        Returns
        -------
        Simulation
            A translated simulation sharing the grid, input field, noise realisation,
            numerical settings and transverse geometry with this one. Noise drawn
            without a seed can still be drawn again from the seed a resume's file or
            a scan point supplies; on a spatial route, where the draw is measured by
            the medium's flux, it is drawn again now from the same seed on the new
            medium. A realisation the caller seeded or supplied is shared unchanged.

        Raises
        ------
        DriverError
            If there is no mixture declaration, or labels, species or unsupported
            mode geometries differ.
        """
        old = _simulation_mixture(self)
        if old.labels != mixture.labels:
            raise DriverError(
                f"with_mixture requires the same ordered labels: {old.labels}, "
                f"got {mixture.labels}"
            )
        for before, after in zip(old.components, mixture.components, strict=True):
            if serialise(before.species) != serialise(after.species):
                raise DriverError(
                    f"with_mixture cannot change species data for {before.label!r}"
                )
        original = self.translate()
        declared = (
            self.responses
            if self._mixture_selection is None
            else self._mixture_selection.build(mixture)
        )
        channels = {
            channel.label: channel for channel in original.responses.population_channels
        }
        responses = tuple(
            dataclasses.replace(
                response,
                rate=channels[_population_label(response, mixture)].rate,
                preionisation=mixture[
                    _population_label(response, mixture)
                ].preionisation,
            )
            if isinstance(response, PlasmaResponse | EnvelopePlasmaResponse)
            else response
            for response in declared
        )
        _refuse_idle_preionisation(mixture, responses)
        modes = tuple(_rebind_mode(mode, mixture) for mode in self.modes)
        _warn_boreless_gradient(mixture, _flow_radius(self.modes, None))
        profiles = dict(self.profiles)
        profiles.pop("pressure", None)
        profiles.update(mixture_profiles(mixture, pressure="pressure" in self.profiles))
        configuration = dict(self.configuration)
        configuration["medium"] = mixture
        if "capillary" in configuration:
            configuration["capillary"] = {
                **cast("Mapping[str, object]", self.configuration["capillary"]),
                "temperature": mixture.temperature,
            }
        if "waveguide" in configuration:
            configuration["waveguide"] = {
                **cast("Mapping[str, object]", self.configuration["waveguide"]),
                "modes": [ModeRecord.from_mode(mode) for mode in modes],
                "temperature": mixture.temperature,
            }
        updated = Simulation(
            self._base_grid,
            modes,
            responses,
            self._field0,
            settings=self.settings,
            initial_z=self.initial_z,
            noise=self.noise,
            noise_resolution=self.noise_resolution,
            profiles=profiles,
            configuration=configuration,
            spatial_grid=self.spatial_grid,
            medium=None
            if self.medium is None
            else dataclasses.replace(self.medium, material=mixture),
            frame=self.frame,
            wall=self.wall,
            taper_coupling=self.taper_coupling,
            materials=self.materials,
        )
        updated._material_log = self._material_log.copy()
        with _optmat.material_context(updated._material_log):
            updated.noise, updated._noise_redraw = self._rebound_noise(updated.medium)
        updated.initial = self.initial
        updated._mixture_selection = self._mixture_selection
        updated._reuse = _mixture_reuse(self, updated, old, mixture, original)
        updated.translate()
        return updated

    def metadata(self) -> RunMetadata:
        """Build the record of this run, once, and return it.

        Cached because the record stamps the time it was created: two runs of the same
        simulation are the same run, and their files must carry the same digest.

        Returns
        -------
        solprop.output.schema.RunMetadata
            What produced the run, and the key a resume is checked against.

        Examples
        --------
        >>> from solprop.driver import Simulation
        >>> simulation = Simulation.from_gnlse(
        ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
        ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
        ...     width=100e-15, peak_power=9.8, shape="sech", n_save=3,
        ... )
        >>> metadata = simulation.metadata()
        >>> metadata.simulation_kind, metadata.spectral_convention
        ('mode_averaged', 'optics')
        >>> metadata.configuration_hash() == simulation.metadata().configuration_hash()
        True
        """
        # The backend's name is deliberately not here. A run continued on the other
        # backend agrees with an uninterrupted one to 2.2-2.7e-16, which is smaller
        # than the 4.1e-16 between two uninterrupted runs, so it is the same run and
        # the digest keys on the problem and the state alone. What executed the run
        # is recorded instead, outside the key, in `RunMetadata.backend`.
        if self._metadata is None:
            configuration = dict(self.configuration)
            configuration.update(
                {
                    "responses": list(self.responses),
                    # A gigabyte array is recorded by its digest (SEP 22, Q4).
                    "initial": recorded_input(self.initial),
                    "profiles": dict(self.profiles),
                    "density": self.settings.density,
                    "shock": self.settings.shock,
                    "prefactor_index": self.settings.prefactor_index,
                    "absorber": self.settings.absorber,
                    "absorber_scheme": _scheme(self.settings),
                    "propagator_mode": self.settings.propagator_mode,
                    # Every run records its precision, double included (SEP 33
                    # section 5.3, ADR-051), so a resume in the other is refused.
                    "precision": self.settings.precision,
                }
            )
            if self.frame is not None:
                configuration["frame"] = self.frame
            # The operator settings of SEP 22 phase 2, recorded only where set, so
            # every earlier run keeps its digest; a collar as its two numbers.
            if self.settings.evanescent_rate is not None:
                configuration["evanescent_rate"] = self.settings.evanescent_rate
            if self.settings.spatial_absorber is not None:
                configuration["spatial_absorber"] = self.settings.spatial_absorber
            if self.settings.adiabatic_accuracy is not None:
                configuration["adiabatic_accuracy"] = self.settings.adiabatic_accuracy
            # A quiet collar changes what the stages compute (ADR-057), so the opt-in
            # is part of the resume key, recorded only where taken; the rest of the
            # check is the execution's, recorded in its record.
            if _quiet_collar(self.settings):
                configuration["quiet_collar"] = True
            if self.spatial_grid is not None:
                configuration.update(
                    spatial_grid=self.spatial_grid, medium=self.medium, frame=self.frame
                )
            with _optmat.material_context(self._material_log):
                self._metadata = metadata_for(
                    self.translate(),
                    self.grid,
                    modes=self.modes,
                    configuration=configuration,
                    noise=self.noise,
                    modal=self._modal_identity(),
                    materials=self.material_manifest(),
                    quadrature=self.settings.quadrature,
                    frame=self.frame,
                    noise_resolution=self.noise_resolution,
                )
        return self._metadata

    def material_manifest(self) -> MaterialManifest:
        """Collect the run's material manifest: what it took from optmat, and how.

        Every optmat-backed object the run carries is found where it sits -- the
        mixture, the modes' fills and claddings, a planar grid's slab modes, a guide's
        material regions, a bulk medium -- with the selection the run was built with,
        the policies in force and what was measured about them: the dilute-scaling error
        over the run's whole band, collars included, the loss a transparent solid
        dropped, and the bands outside a model's qualified intervals that the run's
        range warnings named. :meth:`metadata` records it once; call it after
        :meth:`translate` for a complete measurement. After :meth:`run` it also
        carries the post-run diagnostic of each optmat ionisation rate
        (``peak_fields``), which the file's record, written before the run, does not:
        the run writes it beside the record instead, as the output's post-run
        annotation :data:`~solprop.output.schema.PEAK_FIELDS_ANNOTATION`, outside the
        configuration hash.

        Returns
        -------
        solprop.physdata.selection.MaterialManifest
            The manifest; empty for a run that takes nothing from optmat.

        Examples
        --------
        >>> from solprop.driver import Simulation
        >>> simulation = Simulation.from_gnlse(
        ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
        ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12, raman=True,
        ...     width=100e-15, peak_power=9.8, shape="sech", n_save=3,
        ... )
        >>> [record.name for record in simulation.material_manifest().records]
        ['kerr/SiO2', 'silica_raman/hollenbeck_2002', 'silica_bundle/default']
        """
        # The band edge of a dispersive law depends on whether the node field is a
        # vector: the translation decides it on the modal and spatial routes, and a
        # general modal guide's node field always has three components.
        vector = self.general is not None and self.general.dispersive is not None
        leaf = None
        if any(isinstance(one, DispersiveKerrResponse) for one in self.responses):
            problem = self.translate()
            vector = problem.transform.n_pol * problem.shape[2] > 1
            leaf = problem.responses.dispersive_kerr
        elif vector:
            leaf = self.translate().responses.dispersive_kerr
        with _optmat.material_context(self._material_log):
            return material_manifest(
                (
                    self.configuration,
                    self.modes,
                    self.responses,
                    self.medium,
                    self.spatial_grid,
                    self.general,
                    self.guide,
                    self.wall,
                ),
                selection=self.materials,
                omega=self.grid.omega,
                log=self._material_log,
                peak_fields=self._peak_fields,
                carrier=self.grid.reference_wavelength,
                active_band=(0.5 * self.grid.omega_min, self.grid.omega_max_window),
                vector=vector,
                leaf=leaf,
            )

    def _modal_identity(self) -> Mapping[str, JsonValue] | None:
        """Return the modal identity of this run's record, or none at all.

        Three routes declare one: a general modal guide, whose identity names the
        provider, the tables and the basis; a capillary carrying a coherent wall block,
        whose identity names the flow instead; and a multimode capillary taper carrying
        the taper coupling without a wall, whose identity names that flow
        (:func:`solprop.translate.coupled.taper_identity`). A run with none carries no
        identity.

        The wall block's validity ratios are re-derived here from the mode set rather
        than carried out of the translation layer: they are a handful of Bessel zeros
        over a wavenumber, the record must hold scalars and not arrays, and a second
        evaluation of the same declared geometry cannot disagree with the first. It
        is the **second** evaluation, though, so it says nothing the first did not:
        ``translate()`` above has already built the same block and raised the same
        grazing-incidence and material-range warnings, and a caller warned twice
        about one geometry would reasonably look for two of them. They are silenced
        here and nowhere else.

        For a guide that varies along z the scan is over the **edges and the midpoints
        of the flow's own sections**: the midpoints are where the frozen blocks were
        built, and the edges are where a monotone profile is extreme, so the recorded
        ratios are the worst geometry the run actually reaches rather than the
        entrance's, and they are the geometry the admission check warns about. Scanning
        the midpoints alone would record a radius the run merely passes through -- on a
        50 to 30 micrometre linear taper, 40 rather than 30 micrometres, a third
        optimistic in every ratio (:func:`solprop.modes.capillary.worst_wall_limits`).

        Returns
        -------
        collections.abc.Mapping or None
            The ``modal`` record, or ``None``.
        """
        if self.general is not None:
            return modal_identity(
                self.general,
                self.grid,
                channels=[mode.channel_label for mode in self.modes],
                has_noise=self.noise is not None,
                # A declared basis of a split block puts a coupled flow on the
                # general route too, and a file that did not say so could resume
                # against a different block in the same gauge.
                flow=self.translate().linear.coupled_flow,
            )
        flow = self.translate().linear.coupled_flow
        if flow is None:
            return None
        if self.wall is None:
            # The taper coupling of a multimode capillary taper:
            # the only coupled flow a mode-set route builds without a wall.
            problem = self.translate()
            return taper_identity(
                flow,
                channels=[mode.channel_label for mode in self.modes],
                active=self.grid.active_mask,
                scored=problem.grids.inband_mask,
                n_components=problem.transform.n_pol,
                has_noise=self.noise is not None,
            )
        edges = np.asarray(flow.z_edges, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GrazingIncidenceWarning)
            warnings.simplefilter("ignore", MaterialRangeWarning)
            limits = worst_wall_limits(
                cast("Sequence[MarcatiliMode]", self.modes),
                self.grid.omega,
                positions=np.unique(
                    np.concatenate([edges, 0.5 * (edges[:-1] + edges[1:])])
                ).tolist(),
                active=self.grid.active_mask,
                scored=self.translate().grids.inband_mask,
                validity_limit=self.wall.validity_limit,
                impedance=self.wall.impedance,
            )
        cf4 = self.wall.method == "cf4"
        return wall_identity(
            flow,
            limits,
            channels=[mode.channel_label for mode in self.modes],
            impedance=self.wall.impedance,
            growth_limit=(
                CF4_GROWTH_LIMIT
                if self.wall.growth_limit is None
                else self.wall.growth_limit
            )
            if cf4
            else None,
            fallback=self.wall.fallback,
            dz_max=float(self.translate().stepper.dz_max) if cf4 else None,
            kinks=self.translate().linear.hard_kinks,
            scored=self.translate().grids.inband_mask,
            n_components=self.translate().transform.n_pol,
            has_noise=self.noise is not None,
        )

    def _boundary_monitor(self) -> BoundaryMonitor | None:
        """Return the boundary monitor of a fresh run, or ``None`` without a check.

        Returns
        -------
        solprop.monitor.BoundaryMonitor or None
            Weights for this grid, its temporal collar and, on a radial run, the
            Hankel quadrature of its columns.
        """
        check = self.settings.boundary_check
        if check is None:
            return None
        boundary = self.settings.absorber
        rate = (
            None
            if boundary is None or not np.any(boundary.temporal_rate)
            else boundary.temporal_rate
        )
        space = self.spatial_grid
        columns = (
            np.asarray(space.hankel.spectral_weights, dtype=np.float64)
            if isinstance(space, RadialGrid)
            else None
        )
        monitor = BoundaryMonitor.build(
            self.grid,
            rate,
            check,
            columns=columns,
            split=rate is not None and _scheme(self.settings) == "split_step",
        )
        if isinstance(space, RadialGrid | FreeGrid | PlanarGrid) and (
            self.medium is not None
        ):
            monitor.transverse = self._transverse_edge(check, space, self.medium)
        return monitor

    def _transverse_edge(
        self,
        check: BoundaryCheck,
        space: RadialGrid | FreeGrid | PlanarGrid,
        medium: SpatialMedium,
    ) -> TransverseEdge:
        """Return the transverse edge a free-space run's monitor reads (SEP 40 5.3).

        Parameters
        ----------
        check : solprop.monitor.BoundaryCheck
            The run's thresholds.
        space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
            Its transverse grid.
        medium : solprop.spatial.SpatialMedium
            Its medium.

        Returns
        -------
        solprop.transverse_monitor.TransverseEdge
            The edge, with nothing recorded.
        """
        problem = self.translate()
        length = float(np.max(problem.stepper.save_planes)) - float(problem.initial.z)
        dispersion = SpatialDispersion(medium, self.grid, Frame.comoving(self.grid))
        k, _ = band_wavenumbers(space, dispersion, max(length, 0.0))
        grid = self.grid
        return TransverseEdge(
            space,
            problem.transverse,
            check.edge_budget,
            check.stop_factor,
            length,
            np.nonzero(grid.inband_mask & grid.active_mask)[0],
            float(np.min(k)),
        )

    def run(  # noqa: PLR0913 - independent execution and output controls
        self,
        output: Output | Path | str | None = None,
        *,
        progress: ProgressReporter | ProgressChoice | None = None,
        progress_period: float = DEFAULT_PROGRESS_PERIOD,
        resume: _ResumeChoice = False,
        memory_budget: int | str | None = None,
        tdse_archive: TDSEArchive | None = None,
        snapshots: SnapshotController | None = None,
    ) -> Output:
        """Execute the run and return the handler it was written to.

        Parameters
        ----------
        output : solprop.output.schema.Output or pathlib.Path or str or None, optional
            Where the planes go: a handler, a path to write an HDF5 file at, or ``None``
            for a :class:`solprop.output.memory.MemoryOutput`. The handler is closed
            when the run ends, so a file handler is returned closed and is read back
            with :func:`solprop.output.hdf5.open_file`; a memory handler stays readable.
            A run with optmat ionisation rates then annotates it with their peak-field
            report (:data:`~solprop.output.schema.PEAK_FIELDS_ANNOTATION`).
        progress : ProgressReporter, {'auto', 'bar', 'log', 'none'} or None, optional
            Where progress goes; ``None`` takes the process default: a
            :class:`solprop.log.BarReporter` on a terminal or in a notebook, a
            :class:`solprop.log.LogReporter` otherwise
            (:func:`solprop.driver.run.run_problem`).
        progress_period : float, optional
            Seconds between the line reporter's periodic lines, s; ``0`` leaves only
            the plane and closing lines. A bar has its own redraw period and a given
            reporter its own business.
        resume : bool or {'interpolated', 'force'}, optional
            Continue the file at `output` instead of replacing it, when its record
            and this run's agree on what matters to the run
            (``docs/guide/results/resuming.md``, SEP 42): the layout and the physics
            must agree, numbers to a rounding-level tolerance; a changed tolerance or
            integrator continues while there is work left, with a warning, and is
            recorded in the session's execution record; provenance is ignored. A
            finished file is returned only when every session ran under this run's
            numerics. Only a path can be resumed: the driver has to open the file in
            append mode itself. Noise a constructor drew without a seed is drawn again
            from the seed the file recorded, so re-running the same call continues its
            file. A file whose last plane was interpolated inside a step is refused
            unless this is ``"interpolated"``: it then continues afresh from the
            interpolated field, and the position is recorded in the annotation
            :data:`~solprop.output.schema.INTERPOLATED_RESUMES_ANNOTATION`.
            ``"force"`` also continues across changed physics while the layout agrees,
            the statistics fit and work is left, restarting the Adams history and
            marking the file, past which its record no longer describes its planes; a
            forced file then continues only under ``"force"``.
        memory_budget : int or str or None, optional
            The memory budget this run is admitted against before its backend
            allocates: bytes, or a string such as ``"16 GiB"`` or ``"warn:16 GiB"``,
            or ``"none"``; ``None`` takes ``SOLPROP_MEMORY_BUDGET`` and otherwise four
            fifths of the machine's memory, which only warns
            (:mod:`solprop.driver.memory`).
        tdse_archive : solprop.output.TDSEArchive or None, optional
            Archive every raw TDSE provider evaluation of a run with a TDSE
            response in this separate HDF5 file (SEP 20 section 5.8); ``None``
            archives nothing. It is opened and validated before any atomic work, a
            resume appends a new session to its own run's archive, and the optical
            output records the archive's run UUID and relative path. It changes
            neither the propagation nor the call ledger.
        snapshots : solprop.output.SnapshotController or None, optional
            Publish immutable local checkpoints without rebuilding the backend or
            atomic provider. Requires HDF5, ``step_planes=1``, ``fields='all'``
            and no interpolated output. Adams history cannot be explicitly disabled.
            A configured archive is copied together with the optical file.

        Returns
        -------
        solprop.output.schema.Output
            The handler the run was written to.

        Raises
        ------
        ValueError
            If `resume` is not one of its four values.
        DriverError
            If a resume is asked for and `output` is not a path, or the file's last
            plane was interpolated and `resume` is neither ``"interpolated"`` nor
            ``"force"``; or `tdse_archive` is given without a TDSE response or at the
            path of the optical file.
        TDSEArchiveError
            If the archive's path holds another file, or a write to it fails; the
            run then stops at its last optical checkpoint.
        MemoryBudgetError
            If the run's estimated persistent memory exceeds its budget; nothing has
            been allocated or written.
        OutputError
            If a file is there and its layout or physics differ (unless forced), a
            forced file is resumed without ``"force"``, a finished file was computed
            under other numerics, or a forced run's statistics do not fit the file's;
            the message names what changed.

        Warns
        -----
        SolpropWarning
            When a resume continues under changed numerics, is forced, or returns a
            finished forced file.

        Examples
        --------
        >>> from solprop.driver import Simulation
        >>> simulation = Simulation.from_gnlse(
        ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
        ...     wavelength_limits=(500e-9, 2e-6), trange=1.2e-12,
        ...     width=100e-15, peak_power=9.8, shape="sech", n_save=3, rtol=1e-6,
        ... )
        >>> handler = simulation.run()
        >>> handler.n_saved, handler["z"].tolist()
        (3, [0.0, 0.005, 0.01])
        """
        _check_resume_choice(resume)
        with _optmat.material_context(self._material_log):
            if resume:
                self._adopt_recorded_seed(output)
            problem = self.translate()
            _refuse_pending_check(problem)
            metadata = self.metadata()
            handler, state, comparison = _open_output(output, metadata, resume=resume)
            continuation: _Continuation | None = None
            if state is not None and comparison is not None:
                next_plane = _validate_resume(
                    problem, state, self.settings.tdse_execution
                )
                continuation = _resume_decision(
                    Path(cast("Path | str", output)),
                    problem,
                    state,
                    next_plane,
                    comparison,
                    resume,
                )
                if continuation is None:
                    handler.open(state.metadata)
                    handler.close()
                    return handler
            _warn_sampled_peak(self.settings.quadrature, problem.stats_spec)
            # Admitted before the output is opened and the backend allocates, so
            # that a refused run leaves nothing behind.
            retention = field_retention(
                problem, self.settings.fields, self.settings.field_planes
            )
            admission = admit_memory(
                problem,
                self.settings.backend,
                memory_budget,
                output=handler,
                retention=retention,
            )
            if problem.tdse is not None and not isinstance(handler, CheckpointOutput):
                raise DriverError(
                    "TDSE output must support terminal checkpoints and call ledgers"
                )
            _check_tdse_archive(tdse_archive, problem, handler)
            check_snapshot_request(
                problem,
                handler,
                snapshots,
                self.settings.resume_history,
                retention.fields,
            )
            if (
                snapshots is not None
                and problem.tdse is not None
                and state is not None
                and _recorded_archive_uuid(state) is not None
                and tdse_archive is None
            ):
                raise DriverError(
                    "snapshots require the archive referenced by "
                    "the resumed optical file"
                )
            backend: Backend | None = None
            binding: TDSEBinding | None = None
            archive: TDSEArchiveWriter | None = None
            failure: BaseException | None = None
            try:
                if problem.tdse is not None:
                    binding, archive = self._prepare_atomic(
                        problem,
                        _AtomicContext(handler, state, metadata, admission),
                        tdse_archive,
                    )
                backend = select(
                    self.settings.backend,
                    problem,
                    grid=self.grid,
                    modes=self.modes,
                    profiles=self.profiles,
                    tdse_binding=binding,
                )
                start_plane = self._prime(
                    backend,
                    problem,
                    state,
                    restart=continuation is not None and continuation.forced,
                )
                _confirm_continuation(continuation, backend, problem, output)
                self._consistency_gate(
                    binding, problem, handler, metadata, backend, admission
                )
                run_problem(
                    problem,
                    backend,
                    handler,
                    metadata=metadata,
                    stats_spec=problem.stats_spec,
                    progress=progress,
                    progress_period=progress_period,
                    start_plane=start_plane,
                    memory=admission,
                    tdse_binding=binding,
                    resume_history=self.settings.resume_history,
                    fields=retention.fields,
                    field_planes=retention.field_planes,
                    snapshots=snapshots,
                    tdse_archive=archive,
                    boundary_monitor=self._boundary_monitor(),
                    resume_record=None if continuation is None else continuation.entry,
                )
            except BaseException as error:
                failure = error
                raise
            finally:
                _close_run(backend, binding, archive, handler, failure)
            self._peak_fields = _peak_field_reports(
                problem, handler, self.responses, _session_planes(state, continuation)
            )
        if (
            state is not None
            and _interpolated(state)
            and isinstance(handler, PlaneStore)
        ):
            # Allowed only by resume="interpolated" or "force"; the record says where.
            earlier = handler.annotations.get(INTERPOLATED_RESUMES_ANNOTATION, [])
            handler.annotate(
                INTERPOLATED_RESUMES_ANNOTATION,
                [*cast("list[JsonValue]", earlier), float(state.plane.z)],
            )
        if self._peak_fields and isinstance(handler, PlaneStore):
            # Learned only now the run has closed, so outside its record and its
            # configuration hash: a post-run annotation (SEP 3, Q13).
            handler.annotate(
                PEAK_FIELDS_ANNOTATION,
                [cast("JsonValue", report.to_dict()) for report in self._peak_fields],
            )
        return handler

    def _prepare_atomic(
        self,
        problem: KernelProblem,
        context: _AtomicContext,
        tdse_archive: TDSEArchive | None,
    ) -> tuple[TDSEBinding, TDSEArchiveWriter | None]:
        """Admit a TDSE run's atomic memory, open its archive and prepare its binding.

        Parameters
        ----------
        problem : KernelProblem
            The translated problem, with its TDSE descriptor.
        context : _AtomicContext
            The optical output, resume state, record and memory admission.
        tdse_archive : TDSEArchive or None
            The archive request.

        Returns
        -------
        tuple
            The binding and the open archive session, or None without a request.
            A failure closes the archive before it propagates.
        """
        descriptor = problem.tdse
        assert descriptor is not None
        _warn_tdse_cost(self.settings.quadrature)
        execution = self.settings.tdse_execution
        if execution is None:
            raise DriverError("TDSE requires explicit tdse_execution settings")
        execution = _tdse_memory_execution(
            problem,
            execution,
            context.admission,
            archive=None if tdse_archive is None else tdse_archive.outputs,
        )
        # Opened and validated before any atomic work (SEP 20, 5.8).
        archive = _open_tdse_archive(
            tdse_archive,
            problem,
            context.handler,
            context.state,
            context.metadata,
            execution,
        )
        try:
            # A haCC binding builds its propagation plans only after the
            # consistency check, so the check's one-row plan never adds to their
            # peak memory (_consistency_gate).
            binding = tdse_runtime.prepare_binding(
                descriptor,
                execution,
                problem.transform.n_nodes,
                host_memory_bytes=context.admission.estimate.total,
                defer_plans=descriptor.model == "hacc",
                archive=archive,
            )
        except BaseException as error:
            _close_tdse_archive(archive, error)
            raise
        return binding, archive

    def _consistency_gate(
        self,
        binding: TDSEBinding | None,
        problem: KernelProblem,
        handler: Output,
        metadata: RunMetadata,
        backend: Backend,
        admission: MemoryAdmission,
    ) -> None:
        """Check a haCC run's weak-field linear response before propagating.

        The cached or computed result is attached to the binding's provenance, under
        ``tdse.consistency``. The binding was prepared with deferred plans, so the
        check's temporary one-row plan is closed before the propagation plans are
        built here. A refused or invalid result is written to the requested
        output, as the opened run record without any plane or checkpoint, before
        :class:`~solprop.exceptions.TDSEConsistencyError` is raised (SEP 20, 5.4).

        Parameters
        ----------
        binding : TDSEBinding or None
            The run's open atomic binding; None for a run without one.
        problem : KernelProblem
            The translated problem.
        handler : Output
            The run's output handler, not yet opened.
        metadata : RunMetadata
            The run's record.
        backend : Backend
            The prepared optical backend.
        admission : MemoryAdmission
            The run's memory admission.
        """
        descriptor = problem.tdse
        if binding is None or descriptor is None or descriptor.model != "hacc":
            return
        declared = next(
            one for one in self.responses if isinstance(one, TDSEGasResponse)
        )
        if not isinstance(binding, tdse_consistency.GateBinding):
            raise DriverError("a haCC run needs a binding with a diagnostic row")
        build = cast("dict[str, object]", binding.metadata().get("build", {}))
        native = cast("dict[str, object]", build.get("native", {}))
        result = tdse_consistency.gate_run(
            binding,
            declared,
            reference_wavelength=self.grid.reference_wavelength,
            provider_revision=tdse_runtime.pinned_revision(),
            computed_on={
                "execution_space": native.get("execution_space"),
                "build": build,
                "threads": native.get("requested_threads"),
            },
        )
        binding.record_consistency(result.to_record())
        if result.outcome in ("refused", "invalid"):
            try:
                open_run_record(handler, metadata, backend, binding, admission)
            except Exception:
                logger.exception("could not record the refused consistency check")
        tdse_consistency.apply_outcome(result)
        # The diagnostic's temporary plan is closed: only now the propagation plans.
        if not binding.plans_built:
            binding.build_plans()

    def _reseed(self, seed: int) -> bool:
        """Draw a noise realisation that was drawn without a seed again, from `seed`.

        A constructor given ``seed=None`` draws operating-system entropy, so building
        the same call twice gives two realisations and two configuration hashes. A
        resume (:meth:`run`) and a scan point (:func:`~solprop.driver.scan.run_scan`)
        pass the seed that makes the run the one they continue or reproduce. A
        realisation the caller seeded or supplied is never redrawn.

        Parameters
        ----------
        seed : int
            The entropy to draw from.

        Returns
        -------
        bool
            Whether the realisation was drawn again. A translation keeps every leaf
            but the initial noise, which is all a realisation reaches; the record is
            rebuilt on its next use.
        """
        if self._noise_redraw is None or self.noise is None or self.noise.seed == seed:
            return False
        self.noise = self._noise_redraw(seed)
        if self._problem is not None:
            self._problem = _with_noise(self._problem, self.noise)
        self._metadata = None
        return True

    def _reseeded(self, seed: int) -> "Simulation":  # noqa: UP037 - runtime hook evaluates the class body
        """Return this run with unseeded noise drawn from `seed`, leaving it as it is.

        A scan's builder may return one object for every point, and a batch keeps each
        member by reference, so a point's realisation must not land on the object the
        other points share. Translated first, so that they share its translation.

        Parameters
        ----------
        seed : int
            The entropy to draw from.

        Returns
        -------
        Simulation
            A copy carrying the new realisation and sharing the translation's other
            leaves, with its own material log as any derived run has; or this run,
            when there is nothing to draw again (:meth:`_reseed`).
        """
        if self._noise_redraw is None or self.noise is None or self.noise.seed == seed:
            return self
        self.translate()
        seeded = copy.copy(self)
        seeded._material_log = self._material_log.copy()
        seeded._reseed(seed)
        return seeded

    def _rebound_noise(
        self, medium: SpatialMedium | None
    ) -> tuple[
        NoiseRealisation | None,
        Callable[[int | None], NoiseRealisation | None] | None,
    ]:
        """Return this run's noise and its redraw for the same run in a new medium.

        A modal draw depends only on the grid, which the updated run shares. A spatial
        draw is measured by the medium's flux, so it is rebuilt there, and the
        realisation drawn again from its own seed: what a run carries must be what its
        redraw gives, or a resume drawing from the recorded seed would disagree with
        its own file. A realisation seeded or supplied by the caller stays as it is.

        Parameters
        ----------
        medium : solprop.spatial.SpatialMedium or None
            The updated run's medium; ``None`` on a modal route.

        Returns
        -------
        tuple
            The realisation and the redraw the updated run carries.
        """
        noise, redraw = self.noise, self._noise_redraw
        if noise is None or redraw is None or medium is None:
            return noise, redraw
        redraw = _spatial_draw(noise.model, self, medium)
        return redraw(noise.seed), redraw

    def _adopt_recorded_seed(self, output: Output | Path | str | None) -> None:
        """Take the noise seed of the file a resume continues, where the call gave none.

        The maintainers' ruling on resuming a default call (``seed=None``): the
        realisation is drawn again from the seed the file recorded, so the re-run
        call describes the run it continues. Nothing happens when there is no file,
        the file recorded no seed or another noise model, or the realisation was
        seeded or supplied by the caller; :func:`~solprop.output.hdf5.check_cache`
        then judges the run as before.

        Parameters
        ----------
        output : solprop.output.schema.Output or pathlib.Path or str or None
            Where the resumed run is written.
        """
        if (
            self._noise_redraw is None
            or self.noise is None
            or not isinstance(output, Path | str)
            or not Path(output).exists()
        ):
            return
        stored = _stored_metadata(output)[0].noise
        if stored is None or stored.seed is None or stored.model != self.noise.model:
            return
        if self._reseed(stored.seed):
            logger.info(
                "resuming %s: the noise was drawn without a seed, so it is drawn again "
                "from the seed the file recorded, %d",
                output,
                stored.seed,
            )

    def _prime(
        self,
        backend: Backend,
        problem: KernelProblem,
        state: ResumeState | None,
        *,
        restart: bool = False,
    ) -> int:
        """Load the backend with the state the run starts from, and say where that is.

        Parameters
        ----------
        backend : solprop.backends.protocol.Backend
            The backend about to step.
        problem : solprop.translate.problem.KernelProblem
            The translated problem, which carries the initial state.
        state : solprop.output.hdf5.ResumeState or None
            What a file the run is continuing holds, or ``None`` to start at the
            entrance.
        restart : bool, optional
            Whether to start the Adams history afresh: a forced resume across
            compared differences, whose stored history was stepped under the other
            physics (SEP 42 section 5.5).

        Returns
        -------
        int
            Index of the first save plane still to propagate to.
        """
        if state is None:
            backend.set_initial(problem.initial.field0, None, z=problem.initial.z)
            return 0
        next_plane = _validate_resume(problem, state, self.settings.tdse_execution)
        logger.info(
            "resuming at z = %.6e m, plane %d of %d",
            state.plane.z,
            state.n_saved,
            len(problem.stepper.save_planes) + 1,
        )
        # A field-free plane resumes only a finished run, which never gets here.
        assert state.plane.field is not None
        field = state.plane.field[np.newaxis]
        if problem.transform.geometry is None:
            field = power_amplitude(
                field, power_weight_at(problem, state.plane.z), inverse=True
            )
        backend.set_initial(field, None, z=state.plane.z, resume=True)
        if state.outcome is not None:
            backend.restore_call_ledgers(
                state.outcome.logical_calls, state.outcome.atomic_calls
            )
        if restart:
            # The snapshot's state was weighted by the other physics, and its energy
            # would jump at the join: restoring it measured an error of 1.3e-4
            # against 4.9e-8 starting afresh, on a capillary forced from 1 to 2 bar.
            logger.info("the Adams history starts afresh: a forced resume")
        else:
            self._continue_march(
                backend, state, adams=isinstance(problem.stepper.integrator, Multistep)
            )
        return next_plane

    def _continue_march(
        self, backend: Backend, state: ResumeState, *, adams: bool = True
    ) -> None:
        """Continue the Adams history from the file's snapshot, where one fits.

        Parameters
        ----------
        backend : solprop.backends.protocol.Backend
            The backend, loaded at the resume position.
        state : solprop.output.hdf5.ResumeState
            What the file holds.
        adams : bool, optional
            Whether this run steps with an Adams method. A run continued under
            another integrator keeps no history, by its own choice, which is
            recorded as a numerics change and needs no warning (SEP 42 section 5.6).
        """
        restore = getattr(backend, "restore_march", None)
        if restore is None:
            return
        if self.settings.resume_history is False:
            if state.march is not None:
                logger.info("the Adams history starts afresh: resume_history=False")
            return
        # A march that cannot carry its history restarts it without having been asked
        # to: say so once, as a warning (SEP 29 section 6.6).
        if not adams:
            if state.march is not None:
                logger.info(
                    "the Adams history is not continued: this run steps "
                    "with another integrator"
                )
            return
        refusal = getattr(backend, "march_export_refusal", lambda: None)()
        if refusal is not None:
            warnings.warn(
                f"the Adams history starts afresh: {refusal}",
                SolpropWarning,
                stacklevel=2,
            )
            return
        if state.march is None:
            return
        reason = restore(state.march)
        if reason is None:
            logger.info(
                "continuing the Adams history at order %d, %d samples",
                state.march.order,
                len(state.march.identities),
            )
        else:
            # Asked to continue a history this backend cannot install (another
            # engine's rows, another build's shapes): the run is still correct to the
            # integrator's tolerance, but it does less than was asked, so say so.
            warnings.warn(
                f"the Adams history starts afresh: {reason}",
                SolpropWarning,
                stacklevel=2,
            )

    @classmethod
    @_one_material_context
    def from_radial(  # noqa: PLR0913 - distinct physics objects and execution settings
        cls,
        grid: TimeGrid,
        space: RadialGrid,
        medium: SpatialMedium,
        responses: Sequence[object],
        initial: SpatialPulse | np.ndarray,
        *,
        settings: SimulationSettings,
        frame: Frame | None = None,
        noise: NoiseSetting = "default",
        seed: int | None = None,
        materials: MaterialSelection | None = None,
        absorber: Absorber | Literal[False] | None = None,
        absorber_scheme: AbsorberScheme | None = None,
        boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
        spatial_absorber: SpatialAbsorber | Literal[False] | None = None,
    ) -> "Simulation":  # noqa: UP037 - runtime hook evaluates while class is defined
        """Build a radial propagation using physical fields and the scalar QDHT.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            Field or envelope time grid, in SI units.
        space : solprop.grid.RadialGrid
            Radial nodes, outer radius and spatial collar.
        medium : solprop.spatial.SpatialMedium
            Homogeneous transverse material; gas density may vary along z.
        responses : collections.abc.Sequence
            Existing Kerr, plasma or explicitly declared Raman response objects.
        initial : solprop.pulses.SpatialPulse or numpy.ndarray
            Beam specification or internal Hankel spectrum in V m, with shape
            ``(n_r, n_pol, n_omega)``.
        settings : SimulationSettings
            Step tolerances, positive save positions in m, statistics and backend.
            Absolute tolerance has the input spectrum's units, V m.
        frame : solprop.linear.Frame or None, optional
            Moving frame. None uses the declared responses' THG choice.
        noise : {'default', None, 'input', 'modified'}, optional
            One photon per transverse column, polarisation and active bin.
            ``"default"``, the default, resolves to ``"input"`` wherever the run can
            carry noise, and records the resolution (SEP 37).
        seed : int or None, optional
            Random generator seed; None draws and records system entropy.
        materials : solprop.physdata.selection.MaterialSelection or None, optional
            Take the medium's gas from optmat under this selection: a
            :class:`~solprop.spatial.SpatialMedium` naming a built-in gas becomes the
            one-population :class:`~solprop.physdata.mixtures.Mixture` of that gas
            under the selection at the same pressure and temperature (a supplied
            susceptibility is kept). A mixture, glass or constant index is taken as
            it is. An optmat rate's energy per event is bound to the input's peak
            field (SEP 3, Q14); every response, a
            :class:`~solprop.nonlinear.responses.KerrResponse` included, is the
            caller's (``gamma3("Ar", wavelength=...)`` is optmat's coefficient at a
            carrier). The run records its material manifest.
            ``None`` (the default) is ``MaterialSelection()``.
        absorber : solprop.boundaries.Absorber, False or None, optional
            Overrides `settings`' absorber, which must then declare none. ``False``
            switches the temporal and spectral collars off and keeps the hard band
            mask; the transverse collars keep their own settings (SEP 21 section 5).
        absorber_scheme : {'nonlinear', 'split_step'} or None, optional
            Overrides `settings`' temporal scheme, which must then be the default.
        boundary_check : solprop.monitor.BoundaryCheck or None, optional
            How the run watches its time window's edges (SEP 21), when `settings`
            name the default check: the default warns, then stops, when light above
            the shot-noise floor reaches them; ``None`` watches nothing. Refused
            beside settings that name another check.

        spatial_absorber : solprop.boundaries.SpatialAbsorber, False or None, optional
            The transverse boundary (SEP 40), when `settings` declare none: ``None``
            keeps the settings' own, by default the safety ramp alone; a
            :class:`~solprop.boundaries.SpatialAbsorber` asks for a designed collar in
            the grid's room outside its requested size; ``False`` carries no ramp.

        Returns
        -------
        Simulation
            Cached translation and the ordinary resumable execution interface.

        Raises
        ------
        DriverError
            If a noise seed is given without noise or the input shape is invalid.
        """
        materials = MaterialSelection() if materials is None else materials
        return cls._from_spatial(
            grid,
            space,
            _selected_medium(medium, materials),
            responses,
            initial,
            settings=_bounded(
                _watched(
                    _with_boundary(settings, grid, absorber, absorber_scheme),
                    boundary_check,
                ),
                spatial_absorber,
            ),
            frame=frame,
            noise=noise,
            seed=seed,
            materials=materials,
        )

    @classmethod
    @_one_material_context
    def from_free(  # noqa: PLR0913 - distinct physics objects and execution settings
        cls,
        grid: TimeGrid,
        space: FreeGrid,
        medium: SpatialMedium,
        responses: Sequence[object],
        initial: SpatialPulse | np.ndarray,
        *,
        settings: SimulationSettings,
        frame: Frame | None = None,
        noise: NoiseSetting = "default",
        seed: int | None = None,
        materials: MaterialSelection | None = None,
        absorber: Absorber | Literal[False] | None = None,
        absorber_scheme: AbsorberScheme | None = None,
        boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
        spatial_absorber: SpatialAbsorber | Literal[False] | None = None,
    ) -> "Simulation":  # noqa: UP037 - runtime hook evaluates while class is defined
        """Build a Cartesian propagation with constant media and transverse FFTs.

        A :class:`solprop.grid.LineGrid` runs the same transform with one y column:
        the beam is uniform in y, the run records ``transverse_measure =
        "per_unit_width"``, and its energies and powers are per metre of y.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            Field or envelope time grid, in SI units.
        space : solprop.grid.FreeGrid
            Cartesian axes, area measure and spatial collar, or a
            :class:`solprop.grid.LineGrid`.
        medium : solprop.spatial.SpatialMedium
            Constant bulk material; varying profiles are refused.
        responses : collections.abc.Sequence
            Existing Kerr, plasma or explicitly declared Raman response objects.
        initial : solprop.pulses.SpatialPulse or numpy.ndarray
            Beam specification or internal Cartesian DFT spectrum in V/m, with shape
            ``(n_x*n_y, n_pol, n_omega)``.
        settings : SimulationSettings
            Step tolerances, positive save positions in m, statistics and backend.
            Absolute tolerance has the input spectrum's units, V/m.
        frame : solprop.linear.Frame or None, optional
            Moving frame. None uses the declared responses' THG choice.
        noise : {'default', None, 'input', 'modified'}, optional
            One photon per transverse column, polarisation and active bin.
            ``"default"``, the default, resolves to no noise on a Cartesian or line
            grid and records why (:data:`FREE_NOISE`, SEP 37); pass ``"input"`` or
            ``"modified"`` for a noise model.
        seed : int or None, optional
            Random generator seed; None draws and records system entropy.
        materials : solprop.physdata.selection.MaterialSelection or None, optional
            Take the medium's gas from optmat under this selection: a
            :class:`~solprop.spatial.SpatialMedium` naming a built-in gas becomes the
            one-population :class:`~solprop.physdata.mixtures.Mixture` of that gas
            under the selection at the same pressure and temperature (a supplied
            susceptibility is kept). A mixture, glass or constant index is taken as
            it is. An optmat rate's energy per event is bound to the input's peak
            field (SEP 3, Q14); every response, a
            :class:`~solprop.nonlinear.responses.KerrResponse` included, is the
            caller's (``gamma3("Ar", wavelength=...)`` is optmat's coefficient at a
            carrier). The run records its material manifest.
            ``None`` (the default) is ``MaterialSelection()``.
        absorber : solprop.boundaries.Absorber, False or None, optional
            Overrides `settings`' absorber, which must then declare none. ``False``
            switches the temporal and spectral collars off and keeps the hard band
            mask; the transverse collars keep their own settings (SEP 21 section 5).
        absorber_scheme : {'nonlinear', 'split_step'} or None, optional
            Overrides `settings`' temporal scheme, which must then be the default.
        boundary_check : solprop.monitor.BoundaryCheck or None, optional
            How the run watches its time window's edges (SEP 21), when `settings`
            name the default check: the default warns, then stops, when light above
            the shot-noise floor reaches them; ``None`` watches nothing. Refused
            beside settings that name another check.

        spatial_absorber : solprop.boundaries.SpatialAbsorber, False or None, optional
            The transverse boundary (SEP 40), when `settings` declare none: ``None``
            keeps the settings' own, by default the safety ramp alone; a
            :class:`~solprop.boundaries.SpatialAbsorber` asks for a designed collar in
            the grid's room outside its requested size; ``False`` carries no ramp.

        Returns
        -------
        Simulation
            Cached translation and the ordinary resumable execution interface.

        Raises
        ------
        DriverError
            If a noise seed is given without noise or the input shape is invalid.
        """
        materials = MaterialSelection() if materials is None else materials
        medium = _selected_medium(medium, materials)
        if not medium.density.is_constant:
            msg = (
                "Cartesian free propagation requires constant media; "
                "varying density profiles are unsupported"
            )
            raise DriverError(msg)
        simulation = cls._from_spatial(
            grid,
            space,
            medium,
            responses,
            initial,
            settings=_bounded(
                _watched(
                    _with_boundary(settings, grid, absorber, absorber_scheme),
                    boundary_check,
                ),
                spatial_absorber,
            ),
            frame=frame,
            noise=noise,
            seed=seed,
            materials=materials,
        )
        if isinstance(space, LineGrid):
            # The run's powers and energies are per metre of y; processing and
            # plotting label them from this key, as for a slab.
            simulation.configuration["transverse_measure"] = space.transverse_measure
        return simulation

    @classmethod
    @_one_material_context
    def from_planar(  # noqa: PLR0913 - distinct physics objects and execution settings
        cls,
        grid: TimeGrid,
        space: PlanarGrid,
        responses: Sequence[object],
        initial: PlanarPulse | Pulse | np.ndarray,
        *,
        settings: SimulationSettings,
        frame: Frame | None = None,
        materials: MaterialSelection | None = None,
        absorber: Absorber | Literal[False] | None = None,
        absorber_scheme: AbsorberScheme | None = None,
        boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
        spatial_absorber: SpatialAbsorber | Literal[False] | None = None,
    ) -> "Simulation":  # noqa: UP037 - runtime hook evaluates while class is defined
        """Build a hybrid planar propagation: slab modes in x, an FFT in y.

        The hollow planar guide resolved in both transverse directions: the slab modes
        of `space` in x and its periodic y axis, exact (non-paraxial) in y, so
        whole-beam self-focusing and the modulational instability in y -- which a slab
        uniform in y cannot show -- are in the model. The fill is the modes' own, as a
        bulk medium; it and the gap must be constant along z. The geometry integrates
        over y, so energies are in J and powers in W. The collar in y absorbs only when
        `settings` declares an absorber, as for :meth:`from_free`.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            Field or envelope time grid, in SI units.
        space : solprop.modes.slab.PlanarGrid
            The slab modes (one gap, one polarisation family, one constant fill) and
            the y axis with its collar.
        responses : collections.abc.Sequence
            Existing Kerr, plasma or explicitly declared Raman response objects,
            scaled by the fill's number density.
        initial : solprop.pulses.PlanarPulse or numpy.ndarray
            The astigmatic launch, or an internal ``(p, k_y)`` spectrum of slab
            amplitudes in sqrt(W/m), shape ``(n_p*n_y, 1, n_omega)``, the mode
            index fastest. Any other pulse is refused by name.
        settings : SimulationSettings
            Step tolerances, positive save positions in m, statistics and backend.
            Absolute tolerance has the input spectrum's units, sqrt(W/m).
        frame : solprop.linear.Frame or None, optional
            Moving frame. None uses the declared responses' THG choice, following
            the first mode at ``k_y = 0`` as the modal route does.
        materials : solprop.physdata.selection.MaterialSelection or None, optional
            Take the slab modes' built-in materials from optmat under this selection
            and record it in the run's manifest: a fill of a
            built-in gas with its default susceptibility takes the selection's
            single-species susceptibility (no mixture is needed), and a slab without
            a declared cladding takes optmat's recommended fused silica (SEP 3, Q5).
            A supplied susceptibility or cladding is taken as it is. An optmat
            rate's energy per event is bound to the input's peak field (SEP 3, Q14),
            and the input is checked against each rate. ``None`` (the default) is
            ``MaterialSelection()``.
        absorber : solprop.boundaries.Absorber, False or None, optional
            Overrides `settings`' absorber, which must then declare none. ``False``
            switches the temporal and spectral collars off and keeps the hard band
            mask; the transverse collars keep their own settings (SEP 21 section 5).
        absorber_scheme : {'nonlinear', 'split_step'} or None, optional
            Overrides `settings`' temporal scheme, which must then be the default.
        boundary_check : solprop.monitor.BoundaryCheck or None, optional
            How the run watches its time window's edges (SEP 21), when `settings`
            name the default check: the default warns, then stops, when light above
            the shot-noise floor reaches them; ``None`` watches nothing. Refused
            beside settings that name another check.

        spatial_absorber : solprop.boundaries.SpatialAbsorber, False or None, optional
            The transverse boundary (SEP 40), when `settings` declare none: ``None``
            keeps the settings' own, by default the safety ramp alone; a
            :class:`~solprop.boundaries.SpatialAbsorber` asks for a designed collar in
            the grid's room outside its requested size; ``False`` carries no ramp.

        Returns
        -------
        Simulation
            Cached translation and the ordinary resumable execution interface.

        Raises
        ------
        DriverError
            If the fill varies along z or is a mixture, the input is neither a
            :class:`solprop.pulses.PlanarPulse` nor a spectrum of the planar shape.
        """
        materials = MaterialSelection() if materials is None else materials
        space = _selected_planar(space, materials)
        medium = _planar_medium(space)
        if not isinstance(initial, PlanarPulse | np.ndarray):
            msg = (
                "a planar run launches a PlanarPulse (waist_x projected onto the slab "
                "modes, waist in y) or takes a built (p, k_y) spectrum; got "
                f"{type(initial).__name__}"
            )
            raise DriverError(msg)
        chosen_frame = frame
        if chosen_frame is None:
            chosen_frame = Frame.comoving(
                grid,
                thg=any(
                    bool(getattr(response, "thg", False)) for response in responses
                ),
            )
        settings = _spatial_default(
            _bounded(
                _watched(
                    _with_boundary(settings, grid, absorber, absorber_scheme),
                    boundary_check,
                ),
                spatial_absorber,
            ),
            grid,
        )
        simulation = cls(
            grid,
            (),
            responses,
            initial,
            settings=_spatial_settings(settings, "planar"),
            noise_resolution=NoiseResolution("default", None, PLANAR_NOISE),
            profiles={"density": medium.density},
            spatial_grid=space,
            medium=medium,
            frame=chosen_frame,
            materials=materials,
        )
        field = simulation._field0
        if field.shape != (space.n_columns, 1, grid.n_omega):
            msg = (
                "a planar input has shape (n_p*n_y, 1, n_omega) = "
                f"{(space.n_columns, 1, grid.n_omega)}, got {field.shape}"
            )
            raise DriverError(msg)
        _bind_spatial_rates(simulation, initial)
        return simulation

    @classmethod
    def _from_spatial(  # noqa: PLR0913 - distinct physics objects and execution settings
        cls,
        grid: TimeGrid,
        space: RadialGrid | FreeGrid,
        medium: SpatialMedium,
        responses: Sequence[object],
        initial: SpatialPulse | np.ndarray,
        *,
        settings: SimulationSettings,
        frame: Frame | None = None,
        noise: NoiseSetting = "default",
        seed: int | None = None,
        materials: MaterialSelection | None = None,
    ) -> "Simulation":  # noqa: UP037 - runtime hook evaluates while class is defined
        """Build a radial or Cartesian propagation using physical fields.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            Field or envelope time grid, in SI units.
        space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
            Radial nodes, outer radius and spatial collar, or the Cartesian axes and
            their collar.
        medium : solprop.spatial.SpatialMedium
            Homogeneous transverse material; gas density may vary along z.
        responses : collections.abc.Sequence
            Existing Kerr, plasma or explicitly declared Raman response objects.
        initial : solprop.pulses.SpatialPulse or numpy.ndarray
            Beam specification or internal spectrum, a Hankel spectrum in V m or a
            raw Cartesian DFT spectrum in V/m, with shape ``(n_columns, n_pol,
            n_omega)``.
        settings : SimulationSettings
            Step tolerances, positive save positions in m, statistics and backend.
            Absolute tolerance has the input spectrum's units.
        frame : solprop.linear.Frame or None, optional
            Moving frame. None uses the declared responses' THG choice.
        noise : {'default', None, 'input', 'modified'}, optional
            One photon per transverse column, polarisation and active bin.
            ``"default"``, the default, resolves to ``"input"`` on a radial grid and to
            no noise on a Cartesian or line one (:data:`FREE_NOISE`), and records the
            resolution (SEP 37).
        seed : int or None, optional
            Random generator seed; None draws and records system entropy.
        materials : solprop.physdata.selection.MaterialSelection or None, optional
            The run's material selection, recorded in its manifest.

        Returns
        -------
        Simulation
            Cached translation and the ordinary resumable execution interface.

        Raises
        ------
        DriverError
            If a noise seed is given without noise or the input shape is invalid.
        """
        chosen_frame = frame
        if chosen_frame is None:
            chosen_frame = Frame.comoving(
                grid,
                thg=any(
                    bool(getattr(response, "thg", False)) for response in responses
                ),
            )
        if isinstance(medium.material, Mixture) and settings.stats is None:
            base = default_spec("radial" if isinstance(space, RadialGrid) else "free")
            settings = dataclasses.replace(
                settings,
                stats=dataclasses.replace(
                    base,
                    names=(
                        *base.names,
                        "species_ionisation",
                        "species_electron_density",
                        "species_peak_ionisation_rate",
                        "species_density",
                        "species_ionisation_energy",
                    ),
                ),
            )
        settings = _spatial_default(settings, grid)
        simulation = cls(
            grid,
            (),
            responses,
            initial,
            settings=_spatial_settings(
                settings, "radial" if isinstance(space, RadialGrid) else "free"
            ),
            profiles=mixture_profiles(medium.material)
            if isinstance(medium.material, Mixture)
            else {"density": medium.density},
            spatial_grid=space,
            medium=medium,
            frame=chosen_frame,
            materials=materials,
        )
        field = simulation._field0
        if (
            field.ndim != 3
            or field.shape[0] != space.n_columns
            or field.shape[2] != grid.n_omega
        ):
            msg = "spatial input must have shape (n_columns, n_pol, n_omega)"
            raise DriverError(msg)
        noise_model, resolution = _resolve_noise(
            noise,
            seed,
            settings.precision,
            quiet=None if isinstance(space, RadialGrid) else FREE_NOISE,
        )
        simulation.noise_resolution = resolution
        if noise_model is not None:
            draw = _spatial_draw(noise_model, simulation, medium)
            simulation.noise = draw(seed)
            _redrawable(simulation, draw, seed)
        _bind_spatial_rates(simulation, initial)
        return simulation

    @classmethod
    @_one_material_context
    def from_gnlse(  # noqa: PLR0913 - the GNLSE's keyword surface
        cls,
        gamma: float | None,
        length: float,
        betas: Sequence[float] | np.ndarray | None = None,
        *,
        wavelength: float,
        wavelength_limits: tuple[float, float] | None = None,
        trange: float | None = None,
        grid: TimeGrid | None = None,
        beta: tuple[Float[np.ndarray, " n_nodes"], Float[np.ndarray, " n_beta_nodes"]]
        | Callable[[Float[np.ndarray, " n_omega"]], Float[np.ndarray, " n_omega"]]
        | None = None,
        fibre: Mode | IndexTable | None = None,
        n2: float | str | None = None,
        fwhm: float | None = None,
        width: float | None = None,
        shape: str = "gauss",
        energy: float | None = None,
        peak_power: float | None = None,
        phases: Sequence[float] | np.ndarray = (),
        pulses: Sequence[Pulse] | None = None,
        loss_db_per_m: float | None = None,
        shock: bool = True,
        raman: bool = False,
        raman_fraction: float | None = None,
        noise: NoiseSetting = "default",
        seed: int | None = None,
        rtol: float | None = None,
        atol: float = DEFAULT_ATOL,
        dz_max: float | None = None,
        dz_init: float | None = None,
        schedule: FixedSchedule | None = None,
        integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None,
        interpolate_planes: bool = False,
        step_planes: int = 0,
        save_planes: Sequence[float] | np.ndarray | None = None,
        n_save: int | None = None,
        stats: StatsSpec | None = None,
        polarisation_taper: bool = False,
        absorber: Absorber | Literal[False] | None = None,
        absorber_scheme: AbsorberScheme | None = None,
        boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
        backend: str = "auto",
        precision: PrecisionName = "double",
        materials: MaterialSelection | None = None,
    ) -> "Simulation":  # noqa: UP037 - beartype resolves it while the body still runs
        """Build a simulation of a GNLSE fibre from its coefficients.

        The constructor :func:`solprop.driver.interface.prop_gnlse` calls; its docstring
        documents every keyword, and this method takes them unchanged. The GNLSE is not
        a separate solver path: it is the mode-averaged envelope problem with a
        dispersion-specified mode, ``density = 1``, ``prefactor_index = 1`` and the
        shock term supplied by the prefactor's ``omega`` dependence
        (``docs/guide/waveguides/gnlse.md``).

        Parameters
        ----------
        gamma : float or None
            Nonlinear coefficient in 1/(W m).
        length : float
            Fibre length, m, positive.
        betas : array_like, optional
            Taylor coefficients of ``beta`` about `wavelength`, ``beta_0, beta_1, ...``
            in s^n/m. Give exactly one of this, `beta` or `fibre`.
        wavelength : float
            Reference wavelength, m: the ``gamma`` mapping's, the Taylor expansion's,
            the grid's and the pulse's centre.
        wavelength_limits : tuple of float, optional
            Shortest and longest wavelength of the frequency window, m. Required unless
            `grid` is given.
        trange : float, optional
            Extent of the time window, s. Required unless `grid` is given.
        grid : solprop.grid.TimeGrid, optional
            An envelope grid to use instead of building one. A field grid is refused.
        beta : tuple or callable, optional
            The propagation constant as data: a pair of arrays ``(omega, beta)`` in
            rad/s and 1/m, or a callable of ``omega`` sampled onto the grid's active
            band. Both go through :class:`solprop.nonlinear.gnlse.TabulatedDispersion`.
        fibre : Mode or IndexTable, optional
            Whole dispersion and loss instead of `betas` or `beta`. A mode goes
            through :meth:`solprop.nonlinear.gnlse.SpecifiedMode.from_mode`, so one
            whose geometry moves along z -- a tapered mode, a moving guide's
            provider channel, a ``SpecifiedMode`` with dispersion and area paths --
            hands over its dispersion and its area along z; only the fixed-step
            scaffold requires a z-constant fibre. Its fill is ignored as a
            nonlinear material: the GNLSE uses gamma or n2, retaining the mode only
            for dispersion, loss and area. A table's area is evaluated at
            `wavelength`, and so is a provider channel's, by a fresh solve there (along
            the path of a moving guide); without an area column a table requires
            `gamma` and uses the 1 m^2 convention.
        n2 : float or str, optional
            Nonlinear index in m^2/W, requiring `fibre` and ``gamma=None``.
            Exactly one of `gamma` and `n2` is required. The mapping is
            ``gamma = n2 omega0 / (c A_eff)``, with unit prefactor index. Under
            `materials`, ``"recommended"`` takes optmat's silica bundle's total
            ``n2`` at `wavelength`.
        fwhm : float, optional
            Intensity full width at half maximum of the input, s.
        width : float, optional
            Natural width of a ``sech**2`` input, s. Give this or `fwhm`.
        shape : {'gauss', 'sech'}, optional
            Shape of the input.
        energy : float, optional
            Pulse energy, J. Give this or `peak_power`.
        peak_power : float, optional
            Peak power, W.
        phases : array_like, optional
            Optics-convention spectral phase about `wavelength`: CEP in rad, then group
            delay, GDD, TOD, ... in s^n.
        pulses : sequence of solprop.pulses.Pulse or None, optional
            Several pulses launched together, for two-colour and multi-colour
            inputs: each a :mod:`solprop.pulses` object with its own wavelength,
            duration, energy or peak power, phase and delay; the input is the sum of
            their fields (:class:`~solprop.pulses.PulseSum`). Replaces the
            single-pulse keywords `fwhm`, `width`, `shape`, `energy`,
            `peak_power` and `phases`, which must then be left unset; `wavelength` still
            sets the grid's reference wavelength.
        loss_db_per_m : float or None, optional
            Omitted means zero loss on the coefficient routes. Refused beside
            `fibre`, which carries its own loss. Power loss in dB/m, converted by
            :func:`solprop.nonlinear.gnlse.attenuation_from_db`.
        shock : bool, optional
            Keep the prefactor's ``omega`` dependence, which is the shock term.
        raman : bool, optional
            Whether a fused-silica Raman response takes `raman_fraction` of ``gamma``.
        raman_fraction : float or None, optional
            Share of ``gamma`` held back from the Kerr term, between 0 and 1. Omitted,
            the silica bundle's ``f_R`` with `raman` true and 0 with it false; given,
            held back either way.
        noise : {'default', None, 'input', 'modified'}, optional
            Which shot-noise model to carry (``docs/guide/media/noise.md``).
            ``"default"``, the default, resolves to ``"input"`` wherever the run can
            carry noise, and records the resolution (SEP 37).
        seed : int, optional
            Entropy of the generator the realisation is drawn from, recorded in the
            output so a run is reproducible.
        rtol : float or None, optional
            Relative tolerance of the step controller.
        atol : float, optional
            Absolute tolerance, in the square root of watts.
        dz_max : float, optional
            Cap on the controller's proposal, m; ``None`` uses half the length.
        dz_init : float, optional
            First step, m; ``None`` uses a ten-thousandth of the length.
        schedule : solprop.stepper.schedule.FixedSchedule, optional
            A fixed schedule to replay instead of stepping adaptively
            (``docs/dev/stepper.md`` section 8).
        integrator : Multistep, RungeKutta or LowStorageRungeKutta, optional
            The method that steps the run
            (``docs/guide/numerics/choosing-a-stepper.md``).
            ``None``, the default, takes ``RungeKutta()``, the interaction-picture
            DOPRI5 march, which SEP 37 measured fastest at matched accuracy on the
            GNLSE. ``Multistep()`` is the low-call exponential Adams method, one
            response evaluation per step. ``LowStorageRungeKutta()`` replays
            `schedule` in three registers (SEP 24).
        interpolate_planes : bool, optional
            Observe the save planes inside accepted steps instead of landing on them
            (SEP 14): through the Adams march's own polynomial, or DOPRI5's
            continuous extension with ``RungeKutta()``. Default ``False``.
        step_planes : int, optional
            Also save the end of every `step_planes`-th accepted step, the
            march's own positions, beside the requested planes (SEP 14);
            ``0``, the default, saves the requested planes only.
        save_planes : array_like, optional
            Positions to save at, m. Give this or `n_save`.
        n_save : int, optional
            Number of planes to store, counting the one at ``z = 0``, spread evenly over
            the fibre. Default 201.
        stats : solprop.stats.StatsSpec, optional
            Which statistics to record; ``None`` records
            :func:`solprop.stats.default_spec` of a mode-averaged run.
        polarisation_taper : bool, optional
            Whether the spectral collar tapers the nonlinear polarisation. Default
            ``False``: the taper costs a systematic error of the size of the field's
            content in the collar, and the rate absorber acts either way.
        absorber : solprop.boundaries.Absorber, False or None, optional
            The collar rates; ``None`` builds the default collar of the grid
            (:func:`default_collar`), and ``False`` switches both soft collars off
            and keeps the hard band mask (:func:`solprop.boundaries.no_absorber`).
        absorber_scheme : {'nonlinear', 'split_step'} or None, optional
            How a temporal collar is applied (ADR-025); ``None`` is the default,
            inside every stage.
        boundary_check : solprop.monitor.BoundaryCheck or None, optional
            How the run watches its time window's edges (SEP 21): the default warns,
            then stops, when light above the shot-noise floor reaches them; ``None``
            watches nothing, for a window kept narrow on purpose.
        backend : str, optional
            Registry name, or ``"auto"`` (the default) to prefer ``native`` if installed
            and otherwise use Numba CPU. Import or construction errors do not fall back.
        precision : {'double', 'single'}, optional
            The arithmetic the run is held in (SEP 33): ``"double"``, the default, or
            ``"single"``, which takes ``rtol`` at the floor
            :data:`~solprop.translate.precision.SINGLE_RTOL_FLOOR` by default and
            ``RungeKutta()``, carries no noise, and resolves ``backend="auto"`` to the
            experimental ``mlx`` where MLX and a Metal GPU are present, with
            ``numpy_ref`` as the reference (:attr:`SimulationSettings.precision`).
        materials : solprop.physdata.selection.MaterialSelection or None, optional
            Take fused silica's nonlinear response from optmat's silica bundle
            (:class:`~solprop.physdata.glasses.SilicaBundle`): with `raman` on, an
            omitted fraction is the bundle's ``f_R`` and the response its sampled
            shape; ``n2="recommended"`` is its total ``n2``. The fibre's dispersion,
            area and a given ``gamma`` or ``n2`` are the caller's. The run records
            its manifest, the held-back fraction and its origin
            included. ``None`` (the default) is ``MaterialSelection()``.

        Returns
        -------
        Simulation
            Ready to :meth:`run`.

        Raises
        ------
        DriverError
            If the fibre, the grid, the input or the save planes are specified in a way
            this interface does not accept; the message names what is wrong.

        Examples
        --------
        >>> from solprop.driver import Simulation
        >>> simulation = Simulation.from_gnlse(
        ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
        ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
        ...     width=100e-15, peak_power=9.8, shape="sech", n_save=3,
        ... )
        >>> simulation.settings.save_planes.tolist()
        [0.005, 0.01]
        """
        materials = MaterialSelection() if materials is None else materials
        _refuse_dispersive_policy(materials, "a GNLSE run's fitted gamma")
        working = _gnlse_grid(
            grid,
            length,
            wavelength,
            wavelength_limits,
            trange,
        )
        _check_selection_carrier(materials, working.reference_wavelength, wavelength)
        mapping = _gnlse_fibre(
            gamma,
            betas,
            beta,
            working,
            wavelength=wavelength,
            loss=loss_db_per_m,
            fibre=fibre,
            n2=n2,
            shock=shock,
            raman=raman,
            raman_fraction=raman_fraction,
            selection=materials,
        )
        pulse = _shaped_pulse(
            shape,
            wavelength=wavelength,
            fwhm=fwhm,
            width=width,
            energy=energy,
            peak_power=peak_power,
            phases=phases,
            pulses=pulses,
        )
        boundary, scheme = _boundary(absorber, absorber_scheme, working)
        settings = SimulationSettings(
            controller=PIController(
                rtol=_tolerance(rtol, precision, DEFAULT_RTOL), atol=atol
            ),
            save_planes=_save_planes(length, save_planes, n_save),
            # Relative defaults, so that the first step and the cap scale with the
            # fibre.
            dz_max=length / 2 if dz_max is None else dz_max,
            dz_init=length * 1e-4 if dz_init is None else dz_init,
            auto_init=dz_init is None,
            schedule=schedule,
            integrator=integrator,
            _route_integrator=RungeKutta(),
            interpolate_planes=interpolate_planes,
            step_planes=step_planes,
            density=1.0,
            shock=shock,
            prefactor_index=mapping.prefactor_index,
            stats=default_spec("mode_averaged") if stats is None else stats,
            absorber=boundary,
            absorber_scheme=scheme,
            boundary_check=boundary_check,
            polarisation_taper=polarisation_taper,
            backend=backend,
            precision=precision,
        )
        noise_model, resolution = _resolve_noise(noise, seed, precision)
        draw = functools.partial(_shot_noise, noise_model, working)
        simulation = cls(
            working,
            mapping.mode,
            list(mapping.responses),
            pulse,
            settings=settings,
            noise=draw(seed),
            noise_resolution=resolution,
            configuration={
                "gnlse": mapping,
                "loss_db_per_m": 0.0 if loss_db_per_m is None else loss_db_per_m,
            },
            materials=materials,
        )
        return _redrawable(simulation, draw, seed)

    @classmethod
    @_one_material_context
    def from_capillary(  # noqa: PLR0913 - the interface's keyword surface
        cls,
        radius: float | ZProfile,
        length: float,
        gas: Gas | str | Mapping[str, float] | Mixture,
        pressure: Pressure | None = None,
        *,
        pressure_floor: float | None = None,
        vacuum_end: VacuumEnd | None = None,
        wavelength: float,
        wavelength_limits: tuple[float, float] | None = None,
        trange: float | None = None,
        grid: TimeGrid | None = None,
        envelope: bool = False,
        fwhm: float | None = None,
        width: float | None = None,
        shape: str = "gauss",
        energy: float | None = None,
        peak_power: float | None = None,
        phases: Sequence[float] | np.ndarray = (),
        pulses: Sequence[Pulse] | None = None,
        modes: int | Sequence[Mode] = 1,
        mode_average: bool = False,
        lattice: TubeLattice | None = None,
        wall: CoherentWall | None = None,
        taper_coupling: bool = True,
        model: MarcatiliModel = "full",
        loss: LossSpec = True,
        cladding: GlassIndex | complex | None = None,
        susceptibility: GasSusceptibility | None = None,
        temperature: float | ZProfile | None = None,
        kerr: bool = True,
        thg: bool | None = None,
        plasma: bool
        | IonisationRate
        | Mapping[str, bool | IonisationRate]
        | None = None,
        preionisation: float | ZProfile | Mapping[str, float | ZProfile] | None = None,
        plasma_integration: IntegrationPolicy | None = None,
        raman: bool
        | Sequence[str]
        | Mapping[str, bool | MolecularOptions]
        | None = None,
        n_r: int | None = None,
        n_theta: int | None = None,
        radial_rule: RadialRule = "gauss_kronrod",
        ionisation_field_range: tuple[float, float] | None = None,
        absorber: Absorber | Literal[False] | None = None,
        absorber_scheme: AbsorberScheme | None = None,
        boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
        noise: NoiseSetting = "default",
        seed: int | None = None,
        rtol: float | None = None,
        atol: float = WAVEGUIDE_ATOL,
        dz_max: float | None = None,
        dz_init: float | None = None,
        schedule: FixedSchedule | None = None,
        integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None,
        interpolate_planes: bool = False,
        step_planes: int | None = None,
        save_planes: Sequence[float] | np.ndarray | None = None,
        n_save: int | None = None,
        stats: StatsSpec | None = None,
        node_budget: int = DEFAULT_NODE_BUDGET,
        backend: str = "auto",
        precision: PrecisionName = "double",
        polarisation: str | float | None = None,
        materials: MaterialSelection | None = None,
        tdse: TDSEGasResponse | None = None,
        tdse_execution: TDSEExecution | None = None,
    ) -> "Simulation":  # noqa: UP037 - beartype resolves it while the body still runs
        """Build a simulation of a pulse in a gas-filled hollow waveguide.

        The constructor :func:`solprop.driver.interface.prop_capillary` calls. The mode
        set is the first `modes` HE_1m modes of the capillary unless a set is given; the
        transverse integral is a fixed quadrature rule
        (``docs/guide/waveguides/mode-sets.md``); the
        gas density along z is the pressure path through the equation of state, and the
        same path sets the modes' core index, so a gradient fill disperses and broadens
        consistently.

        See ``docs/guide/setup/running.md`` for the discussion and a worked example.

        Parameters
        ----------
        radius : float or solprop.profiles.ZProfile
            Core radius, m, constant or along z.
        length : float
            Capillary length, m.
        gas : Gas, str, Mapping or Mixture
            Built-in gas, mole fractions keyed by species name, or ordered populations
            carrying their own density paths. A mapping uses total pressure.
        pressure : float, sequence, ZProfile or None, optional
            None uses one bar for a gas or fractions. Omit pressure beside a Mixture;
            every explicit pressure is refused because it already carries its paths.
            Fill pressure in **bar**: a number for a constant fill, a pair
            ``(p_in, p_out)`` for the steady-flow gradient of
            :meth:`solprop.profiles.PressureGradient.two_point`, or ``(z, p)`` with the
            positions and the pressures there for
            :meth:`solprop.profiles.PressureGradient.from_stages`.
        pressure_floor : float or None, optional
            Lowest pressure the fill is allowed to reach, bar; ``None`` takes
            :data:`solprop.profiles.DEFAULT_PRESSURE_FLOOR_FRACTION` of the highest
            stage pressure. With ``vacuum_end="floor"``, pass ``0.0`` for a true
            vacuum end, which warns, because ``dp/dz`` diverges there.
        vacuum_end : {'choked', 'floor'} or None, optional
            How a pressure gradient's outer end below its choked pressure is treated.
            ``None`` (the default) means ``"choked"`` for a circular hollow guide: the
            fill follows the bore (Pennetta et al. 2019, eq. 6) and an end exhausting
            into vacuum takes the choked pressure of the flow
            (:func:`solprop.gasflow.steady_fill`), with a warning if the flow is not
            laminar. ``"floor"`` concerns the ends only: it keeps the declared end
            pressures, raised only to `pressure_floor`, for a comparison with a model
            that imposes them, while a taper's fill still follows the bore (on a
            uniform bore, bitwise the uniform law). Refused beside a constant
            pressure, a pressure profile or a Mixture, which have no end to choke,
            and on the general route. Every mode declaring ``core_radius`` -- every
            ``MarcatiliMode`` subclass, a ``TabulatedMode`` built from arrays
            included -- is taken as a circular bore of that radius; a rectangular
            guide keeps the uniform law.
        wavelength : float
            Reference wavelength, m, and the pulse's centre wavelength.
        wavelength_limits : tuple of float or None, optional
            Shortest and longest wavelength of the frequency window, m. Required unless
            `grid` is given.
        trange : float or None, optional
            Requested width of the time window, s. Required unless `grid` is given.
        grid : solprop.grid.TimeGrid or None, optional
            A grid of one's own, instead of the three keywords above; the only way to
            reach ``dt``, ``ffac`` or the transform size.
        envelope : bool, optional
            Propagate an envelope. Plasma is off by default; explicit ``plasma=True``
            selects cycle averaging of instantaneous rates. Plasma forbids THG.
        fwhm : float or None, optional
            Pulse duration at half maximum, s.
        width : float or None, optional
            Natural width of a sech pulse, s; give this or `fwhm`.
        shape : {'gauss', 'sech'}, optional
            Transform-limited shape of the input.
        energy : float or None, optional
            Pulse energy, J. Give this or `peak_power`.
        peak_power : float or None, optional
            Peak power, W, realised after the spectral phases.
        phases : sequence of float, optional
            Taylor spectral phases about `wavelength`, optics convention: CEP in rad,
            group delay in s, GDD in s^2, and so on.
        pulses : sequence of solprop.pulses.Pulse or None, optional
            Several pulses launched together, for two-colour and multi-colour
            inputs: each a :mod:`solprop.pulses` object with its own wavelength,
            duration, energy or peak power, phase and delay; the input is the sum of
            their fields (:class:`~solprop.pulses.PulseSum`). Replaces the
            single-pulse keywords `fwhm`, `width`, `shape`, `energy`,
            `peak_power` and `phases`, which must then be left unset; `wavelength` still
            sets the grid's reference wavelength.
        modes : int or sequence of Mode, optional
            How many HE_1m modes to propagate, or the mode set itself. A set is checked
            for orthonormality. Supplied sets may include TabulatedHollowMode or
            TabulatedMode with their own tables. A supplied set **carries its own
            geometry**: `radius` must agree with every mode's core radius along the run,
            and a `model`, `loss` or `cladding` declared away from its default must be
            every mode's own, or the call is refused by name rather than the argument
            ignored.
        mode_average : bool, optional
            Propagate the one mode **mode-averaged**, as a uniform disc of its
            effective area, instead of resolving it on the radial rule
            (``docs/guide/waveguides/mode-averaging.md``). The plasma then sees the
            area-mean field, below the on-axis one (``HE_11``'s axis intensity is
            1.768 times ``P/A_eff``), and ``peak_intensity`` stays on the axis.
            ``False``, the default, resolves the mode. Refused beside more than one
            mode and beside `n_r`, `n_theta` or a `radial_rule`.
        lattice : TubeLattice or None, optional
            Tube geometry for an integer mode count, selecting Vincetti modes.
            Refused with supplied modes, which carry their own geometry, and with
            ``model="reduced"``. None selects ordinary capillary modes.
        wall : solprop.modes.CoherentWall or None, optional
            Declare the scalar **coherent** wall block: the outgoing Robin boundary
            condition condensed onto the retained basis, which couples every ``HE_1m``
            channel to every other with a definite phase instead of giving each its own
            Marcatili attenuation. It needs the contiguous set ``HE_11`` to ``HE_1N`` --
            or, with ``impedance="te_tm"``, one azimuthal order -- on a constant or a
            varying guide, and a scalar run; it carries the default ``"input"`` noise
            and refuses ``"modified"``. All three backends execute it (a
            ``method="cf4"`` wall ``numpy_ref`` alone); every other mode set or
            geometry is refused by name. ``None``, the default, carries no wall:
            the modes keep their own Marcatili loss, and a multimode taper its taper
            coupling (below).
        taper_coupling : bool, optional
            On a multimode capillary whose radius varies along z, carry the
            **taper coupling** -- the connection ``<psi_m|d_z psi_n>`` that moves
            power between the ``HE_1m`` channels as the bore changes -- lossless or
            not. ``True``, the default; ``False`` is the explicit opt-out to the
            **adiabatic** route, on which each channel follows its local mode and
            exchanges no power. It is refused beside `wall`, which always carries
            the connection, and has no effect on one channel, a constant bore (a
            pressure gradient included, which does not change the modes' shape) or
            a set that is not the contiguous ``HE_11`` to ``HE_1N`` block
            (``docs/guide/waveguides/capillaries.md``, "Tapers").
        model : {'full', 'reduced'}, optional
            Which Marcatili model the modes use; beside a supplied set, refused unless
            it is the default or the modes' own.
        loss : bool, float or callable, optional
            Propagation loss of the modes; beside a supplied set, refused unless it
            is the default or the modes' own.
        cladding : GlassIndex, complex or None, optional
            Cladding index; ``None`` is the package's cited silica. Beside a
            supplied set, refused unless it is the default or the modes' own.
        susceptibility : GasSusceptibility or None, optional
            A gas susceptibility replacing the one `gas` selects, for a comparison
            against another code's refractive data. Refused for a fraction mapping
            or an existing mixture, which own their species models.
        temperature : float, ZProfile or None, optional
            Gas temperature, K, or a path of it along z in m. Sets the density the
            pressure path maps to, so it changes both the dispersion and the
            nonlinearity. A path also weights a steady-flow gradient by the gas's
            ``eta(T) T`` (docs/guide/waveguides/profiles.md), makes a shared mixture's
            populations independent parameters, and holds the rotational Raman
            populations at one reference temperature with a
            :class:`~solprop.exceptions.RamanTemperatureWarning`
            (:func:`solprop.nonlinear.mixtures.raman_temperature`). A constant path
            is the number, bit for bit. ``None`` (the default) is 293.15 K. A Mixture
            owns its temperature: beside one, a temperature must be omitted or be the
            mixture's own (the same number or path object), and any other is refused by
            name.
        kerr : bool, optional
            Include the Kerr response.
        thg : bool or None, optional
            Include third-harmonic generation in the Kerr term; ``None`` takes
            ``not envelope``.
        plasma : bool, IonisationRate, Mapping or None, optional
            Include photoionisation. ``None`` takes ``not envelope``. ``True`` selects
            optmat's recommended rate model of each built-in gas (SEP 3, D1;
            :class:`~solprop.nonlinear.optmat_rates.OptmatRate`, or the model the
            selection's ``ionisation`` names), for a single gas and for every
            component of a mixture; a custom species needs its own rate. A
            rate object is used as given; a mapping selects a rate or switch per
            component label, and omitted labels are off.
        preionisation : float, ZProfile, Mapping or None, optional
            Initial ionised fraction in [0, 1], globally or per label: a number, or a
            path along z in m giving the fraction ionised before the pulse arrives at
            each position. None and omitted mapping labels retain the component's
            existing declaration.
        plasma_integration : {'band', 'trapezoid'} or None, optional
            How the field-grid plasma forms its population and free-current integrals
            (SEP 18): ``'band'`` filters each integrand with taps designed for the
            grid, exact to a certified 1e-5 across the active band; ``'trapezoid'``
            is the plain cumulative trapezoid, short by up to 11 % per integral at
            the top of the default active band, which the frozen witness comparisons
            use. ``None`` takes :class:`~solprop.nonlinear.responses.PlasmaResponse`'s
            default. Refused with envelope plasma, whose currents are converted
            exactly, and when the plasma is off.
        raman : bool, sequence of str, Mapping or None, optional
            ``None`` (the default) turns the Raman response on for every species
            whose optmat response is curated -- N2, O2, H2, D2, CH4 and SF6 -- leaves
            an atom without one, and warns (:class:`solprop.exceptions.SolpropWarning`)
            for a molecule whose response is provisional (N2O, unless the selection's
            ``raman`` is ``"provisional"``) rather than leaving it out silently; with
            ``kerr=False`` the default is off. ``False`` disables it; ``True`` selects
            the available models of a mixture and is refused by name for a single gas
            without data; explicit labels or
            :class:`~solprop.physdata.raman_types.MolecularOptions` select components
            and terms, and a named unavailable model fails explicitly.
        n_r : int or None, optional
            Radial nodes of the fixed rule. ``None`` takes
            :func:`default_radial_order` of the mode count.
        n_theta : int or None, optional
            Azimuthal nodes. Refused for a set of purely radial modes, which need none.
        radial_rule : str, optional
            The radial rule family (SEP 19): ``"gauss_kronrod"`` (the default), whose
            embedded Gauss subset is the quadrature alarm's error estimate;
            ``"gauss"``, with no alarm; or ``"field_scale"`` and
            ``"field_scale_radau"``, Gaussian in the HE_11 field scale for expensive
            responses, which need exactly one Marcatili HE_11 mode and have no alarm
            (``docs/guide/waveguides/radial-quadrature.md``).
        ionisation_field_range : tuple of float or None, optional
            Field amplitudes in V/m the rate table must cover. ``None`` derives them
            from the input's peak field over the axis and the transverse nodes; see
            :func:`solprop.driver.simulation.default_field_range`.
        absorber : solprop.boundaries.Absorber, False or None, optional
            The collar rates; ``None`` builds the default collar of the grid, and
            ``False`` switches both soft collars off and keeps the hard band mask
            (:func:`solprop.boundaries.no_absorber`; SEP 21).
        absorber_scheme : {'nonlinear', 'split_step'} or None, optional
            How the temporal collar is applied (ADR-025): inside every stage, the
            default, or once per accepted step. The low-call Adams methods of
            SEP 13 step much further with ``'split_step'``, whose first-order
            splitting error stays inside the collar.
        boundary_check : solprop.monitor.BoundaryCheck or None, optional
            How the run watches its time window's edges (SEP 21): the default warns,
            then stops, when light above the shot-noise floor reaches them; ``None``
            watches nothing, for a window kept narrow on purpose.
        noise : {'default', None, 'input', 'modified'}, optional
            Which shot-noise model to carry.
            ``"default"``, the default, resolves to ``"input"`` wherever the run can
            carry noise, and records the resolution (SEP 37).
        seed : int or None, optional
            Entropy of the noise generator.
        rtol : float or None, optional
            Relative step tolerance; the default is :data:`CAPILLARY_RTOL`, the
            capillary default. Calibrate it for a production run
            (``docs/dev/stepper.md`` section 3).
        atol : float, optional
            Absolute step tolerance, in the square root of watts.
        dz_max : float or None, optional
            Cap on the step, m; ``None`` takes half the capillary.
        dz_init : float or None, optional
            First step, m; ``None`` takes ``length * 1e-4``.
        schedule : solprop.stepper.schedule.FixedSchedule or None, optional
            A fixed schedule to replay instead of stepping adaptively.
        integrator : Multistep, RungeKutta or LowStorageRungeKutta, optional
            The method that steps the run
            (``docs/guide/numerics/choosing-a-stepper.md``).
            ``None``, the default, takes ``Multistep()``: the low-call exponential
            Adams method, one response evaluation per step. ``RungeKutta()`` is the
            interaction-picture DOPRI5 march, for a cheap response or a state too
            large for the Adams history, and faster on a long run with Raman or
            plasma (SEP 37). ``LowStorageRungeKutta()`` replays
            `schedule` in three registers (SEP 24).
        interpolate_planes : bool, optional
            Observe the save planes inside accepted steps instead of landing on them
            (SEP 14): through the Adams march's own polynomial, or DOPRI5's
            continuous extension with ``RungeKutta()``. Default ``False``.
        step_planes : int or None, optional
            Also save the end of every `step_planes`-th accepted step, the
            march's own positions, beside the requested planes (SEP 14); ``0``
            saves the requested planes only. ``None``, the default, is ``0``, or
            ``1`` with a `tdse` response, which saves every accepted step without
            imposing a landing (SEP 20 section 5.8).
        save_planes : sequence of float or None, optional
            Positions to save at, m. Give this or `n_save`.
        n_save : int or None, optional
            How many planes to save, counting ``z = 0``.
        stats : solprop.stats.StatsSpec or None, optional
            Which statistics to record; ``None`` takes
            :func:`solprop.stats.default_spec` for a multimode run with the profiles,
            or for a mode-averaged one under `mode_average`, plus every statistic
            only a backend can compute.
        node_budget : int, optional
            Bytes of node-chunk buffers the response pass may hold.
        backend : str, optional
            Registry name, or ``"auto"`` (the default) to prefer ``native`` if installed
            and otherwise use Numba CPU. Import or construction errors do not fall back.
        precision : {'double', 'single'}, optional
            The arithmetic the run is held in (SEP 33): ``"double"``, the default, or
            ``"single"``, which takes ``rtol`` at the floor
            :data:`~solprop.translate.precision.SINGLE_RTOL_FLOOR` by default and
            ``RungeKutta()``, carries no noise, and resolves ``backend="auto"`` to the
            experimental ``mlx`` where MLX and a Metal GPU are present, with
            ``numpy_ref`` as the reference (:attr:`SimulationSettings.precision`).
        polarisation : str, float or None, optional
            Refused; this path is scalar.
        materials : solprop.physdata.selection.MaterialSelection or None, optional
            Take the run's built-in materials from optmat under this selection: a gas or
            fraction mapping of built-in gases, ``"air"`` and ``Gas.Air`` resolve
            through :func:`solprop.physdata.species` and
            :meth:`solprop.physdata.mixtures.Mixture.air` with it, and the default
            cladding (``cladding=None`` with a numbered mode set) is optmat's
            recommended silica. A supplied mixture, mode set, cladding or susceptibility
            is taken as it is. The run records its material manifest.
            ``None`` (the default) is ``MaterialSelection()``.

        tdse : solprop.nonlinear.tdse.TDSEGasResponse or None, optional
            A helium TDSE response, SAE or haCC, used as the complete electronic
            response of the fill instead of the Kerr, plasma and Raman responses
            (SEP 20 section 5.2); ``plasma``, ``raman``, ``preionisation``,
            ``plasma_integration``, ``ionisation_field_range`` and ``thg`` must then
            be left unset and ``kerr`` true. ``backend="auto"`` resolves to
            ``"numpy_ref"``, the only backend that executes it; statistics default to
            those the route defines; omitted `save_planes` and `n_save` save only the
            final position beside every accepted step. An atomic window whose duration
            is unresolved takes `trange`. ``None`` keeps the rate-based responses.
        tdse_execution : solprop.nonlinear.tdse.TDSEExecution or None, optional
            Atomic runtime resources of `tdse`: device, memory limit, rows per call
            and call cap. ``None`` beside `tdse` takes ``TDSEExecution()``; refused
            without `tdse`.

        Returns
        -------
        Simulation
            The simulation, ready to translate and run.

        Raises
        ------
        DriverError
            If a keyword is missing, inconsistent, or names something this path does
            not carry.

        Examples
        --------
        >>> from solprop.driver import Simulation
        >>> simulation = Simulation.from_capillary(
        ...     125e-6, 0.1, "He", (0.8, 0.1), wavelength=800e-9,
        ...     wavelength_limits=(150e-9, 2e-6), trange=200e-15,
        ...     fwhm=10e-15, energy=100e-6, modes=2, n_save=3,
        ... )
        >>> len(simulation.modes), simulation.settings.quadrature.n_nodes
        (2, 33)
        >>> sorted(simulation.profiles)
        ['core_radius', 'density', 'pressure']
        """
        return cls._from_hollow(
            radius=radius,
            length=length,
            gas=gas,
            pressure=pressure,
            pressure_floor=pressure_floor,
            vacuum_end=vacuum_end,
            wavelength=wavelength,
            wavelength_limits=wavelength_limits,
            trange=trange,
            grid=grid,
            envelope=envelope,
            fwhm=fwhm,
            width=width,
            shape=shape,
            energy=energy,
            peak_power=peak_power,
            phases=phases,
            pulses=pulses,
            modes=modes,
            mode_average=mode_average,
            lattice=lattice,
            wall=wall,
            taper_coupling=taper_coupling,
            model=model,
            loss=loss,
            cladding=cladding,
            susceptibility=susceptibility,
            temperature=temperature,
            kerr=kerr,
            thg=thg,
            plasma=plasma,
            preionisation=preionisation,
            plasma_integration=plasma_integration,
            raman=raman,
            n_r=n_r,
            n_theta=n_theta,
            radial_rule=radial_rule,
            ionisation_field_range=ionisation_field_range,
            absorber=absorber,
            absorber_scheme=absorber_scheme,
            boundary_check=boundary_check,
            noise=noise,
            seed=seed,
            rtol=_tolerance(rtol, precision, CAPILLARY_RTOL),
            atol=atol,
            dz_max=dz_max,
            dz_init=dz_init,
            schedule=schedule,
            integrator=integrator,
            interpolate_planes=interpolate_planes,
            step_planes=step_planes,
            save_planes=save_planes,
            n_save=n_save,
            stats=stats,
            node_budget=node_budget,
            backend=backend,
            precision=precision,
            polarisation=polarisation,
            materials=MaterialSelection() if materials is None else materials,
            tdse=tdse,
            tdse_execution=tdse_execution,
        )

    @classmethod
    @_one_material_context
    def from_waveguide(  # noqa: PLR0913 - the interface's keyword surface
        cls,
        modes: Sequence[Mode],
        length: float,
        gas: Gas | str | Mapping[str, float] | Mixture | None = None,
        pressure: Pressure | None = None,
        *,
        provider: ModalGuide | None = None,
        channel: int | str | None = None,
        incident: SpatialPulse | None = None,
        coupling: CouplingModel | None = None,
        kerr_form: KerrForm = "real",
        observable_rtol: float | None = None,
        observable_atol: float | None = None,
        pressure_floor: float | None = None,
        vacuum_end: VacuumEnd | None = None,
        wavelength: float,
        wavelength_limits: tuple[float, float] | None = None,
        trange: float | None = None,
        grid: TimeGrid | None = None,
        envelope: bool = False,
        fwhm: float | None = None,
        width: float | None = None,
        shape: str = "gauss",
        energy: float | None = None,
        peak_power: float | None = None,
        phases: Sequence[float] | np.ndarray = (),
        pulses: Sequence[Pulse] | None = None,
        susceptibility: GasSusceptibility | None = None,
        temperature: float | ZProfile | None = None,
        kerr: bool = True,
        thg: bool | None = None,
        plasma: bool
        | IonisationRate
        | Mapping[str, bool | IonisationRate]
        | None = None,
        preionisation: float | ZProfile | Mapping[str, float | ZProfile] | None = None,
        plasma_integration: IntegrationPolicy | None = None,
        raman: bool
        | Sequence[str]
        | Mapping[str, bool | MolecularOptions]
        | None = None,
        mode_average: bool = False,
        n_r: int | None = None,
        n_theta: int | None = None,
        radial_rule: RadialRule = "gauss_kronrod",
        ionisation_field_range: tuple[float, float] | None = None,
        absorber: Absorber | Literal[False] | None = None,
        absorber_scheme: AbsorberScheme | None = None,
        boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
        noise: NoiseSetting = "default",
        seed: int | None = None,
        rtol: float | None = None,
        atol: float = WAVEGUIDE_ATOL,
        dz_max: float | None = None,
        dz_init: float | None = None,
        schedule: FixedSchedule | None = None,
        integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None,
        save_planes: Sequence[float] | np.ndarray | None = None,
        n_save: int | None = None,
        stats: StatsSpec | None = None,
        node_budget: int = DEFAULT_NODE_BUDGET,
        backend: str = "auto",
        precision: PrecisionName = "double",
        polarisation: str | float | None = None,
        taper_coupling: bool = True,
        memory_budget: int | str | None = None,
        materials: MaterialSelection | None = None,
    ) -> "Simulation":  # noqa: UP037 - beartype resolves it while the body still runs
        """Build a pulse simulation in a waveguide: a gas-filled mode set, or a guide.

        **Without `provider`** this is the gas-filled route: modes keep their
        geometry, cladding and loss and receive the declared fill, their nonzero
        field components determine the transverse quadrature layout, and rectangular
        polarisation partners are separate rows with separate operators. A hollow
        planar set (:class:`~solprop.modes.slab.SlabMode`) is uniform in y and runs
        **per unit width**: `energy` is J/m, `peak_power` W/m, and the run records
        ``transverse_measure = "per_unit_width"``. Step-index fibres use
        ``prop_gnlse(fibre=...)`` instead.

        **With `provider`** this is the **general modal route**
        (``docs/guide/waveguides/general-modal.md``): a guide that solves its own
        three-component
        E and H at quadrature nodes, whose material regions own their responses and
        whose channels are Poynting-normalised, so that a channel amplitude is a
        true-power amplitude by construction. It is qualified on ``numpy_ref``,
        ``numba_cpu`` and the native CPU backend, a declared split block's coupled
        flow included, and propagates constant geometry on a **field** grid.
        Every gas keyword is refused by name beside a provider, because a guide's
        materials are its regions' and not a fill.

        Parameters
        ----------
        modes : sequence of solprop.modes.protocol.Mode
            Non-empty gas-filled mode set with a common transverse geometry. Each
            mode is rebound to this constructor's fill (``with_mixture``): its own
            gas, pressure and temperature -- a temperature path included -- are
            replaced by `gas`, `pressure` and `temperature`, and only its geometry is
            kept. Beside
            `provider` it must be empty or exactly the guide's own ordered channels:
            a guide supplies its channels itself, and the provider has already
            validated their order against its fields.
        length : float
            Waveguide length, m.
        gas : Gas, str, Mapping, Mixture or None
            Built-in gas, mole fractions keyed by species name, or ordered populations
            carrying their own density paths. A mapping uses total pressure. Required
            without `provider` and refused beside one.
        pressure : float, sequence, ZProfile or None, optional
            None uses one bar for a gas or fractions. Omit pressure beside a Mixture;
            every explicit pressure is refused because it already carries its paths.
            Fill pressure in **bar**: a number for a constant fill, a pair
            ``(p_in, p_out)`` for the steady-flow gradient of
            :meth:`solprop.profiles.PressureGradient.two_point`, or ``(z, p)`` with the
            positions and the pressures there for
            :meth:`solprop.profiles.PressureGradient.from_stages`.
        provider : solprop.modes.guide.ModalGuide or None, optional
            A guide built by
            :meth:`solprop.modes.guide.ModalGuide.from_provider`, so that it carries
            the sampling its dense E/H reference is rebuilt from. Its knots must
            span the run's whole **active** band, which starts at half the lowest
            declared frequency; a narrower guide is refused with both bands in the
            message.
        channel : int or str or None, optional
            Which channel the pulse enters, by index or by display label
            (``"HE1,1:even"``); ``None`` takes the first, as a gas-filled multimode
            run puts a single pulse in its fundamental. The launch is the pulse's
            amplitude placed in that channel unchanged -- a Poynting basis needs no
            conversion -- so the launched true-power energy is the pulse's. Requires
            `provider`.
        incident : solprop.pulses.SpatialPulse or None, optional
            An **incident beam** to launch instead of a named channel: the beam
            and its temporal pulse, whose energy or peak power is met over the
            **whole** beam in the paraxial measure of the guide's core medium
            (:meth:`solprop.pulses.SpatialPulse.incident`). The beam is coupled into
            the channels by :func:`solprop.coupling.coupler.couple` under `coupling`
            (SEP 31): it is sampled on a radial plane that resolves it, evaluated at
            the guide's nodes, and its coordinates are recovered with the flux
            pairing and the forward closure ``H_t = Y0 z x E_t``. The recovered
            amplitudes are launched as they are. **Light the channels do not
            capture is lost, not restored**: a beam much wider than the core
            launches a small share of its energy, and a beam the channels cannot see
            launches nothing, which is a valid zero launch. The coupling's
            :class:`~solprop.coupling.coupler.CouplingReport` -- incident and launched
            energy per frequency and in total, and its admission budget -- is
            recorded as ``configuration["launch"]``, and the launch's provenance as
            ``configuration["coupling"]``. The beam is launched at its waist
            (``propz=0``) and centred on the guide's axis. The pulse keywords
            (`fwhm`, `width`, `energy`, `peak_power`, `phases`, `pulses` and
            `shape`) are refused beside it, because the beam carries its own
            pulse. Requires `provider` and `coupling`, and is refused beside
            `channel`.
        coupling : solprop.coupling.models.CouplingModel or None, optional
            The model an `incident` beam is coupled under:
            ``CouplingModel("vector_paraxial", model_error=...)``, whose budget the
            guide's contrast, closure and angular content must fit (SEP 31, section
            4.3); a strongly guiding core is refused. Requires `incident`.
        kerr_form : {'real', 'analytic'}, optional
            Which instantaneous cubic a general modal run's material regions respond
            with. ``"real"``, the default, is the real-field cubic
            ``P = eps0 chi3 (E.E) E`` the legacy field route evaluates, third harmonic
            included; ``"analytic"`` is the analytic-signal form
            ``eps0 chi3 (2 |E|^2 E + (E.E) E*) / 4``, which carries the fundamental
            band alone and is what a comparison with an envelope route wants. The real
            form needs a grid with ``n_to >= 2 n_t``, which is ``FieldGrid``'s default
            ``ffac=6``, and is refused on a narrower one. Under one of SEP 6's
            dispersive electronic policies a region's gas fill takes the dispersive
            cubic in the matching sector: ``"real"`` its full cubic, ``"analytic"``
            its number-conserving sector. Requires `provider`.
        observable_rtol, observable_atol : float or None, optional
            Requested accuracy of the named physical observables of a general modal run.
            ``None``, the only value accepted, means **the solver's own** `rtol` and
            `atol`: the route's basis is orthonormal in true power, so the controller's
            weighted norm on the amplitudes already is a norm on the physical channel
            powers. A different number is refused by name, because the separate error
            allocation it would ask for -- over the table's interpolation, the
            transverse rule and the step -- is not implemented. Require `provider`.
        pressure_floor : float or None, optional
            Lowest pressure the fill is allowed to reach, bar; ``None`` takes
            :data:`solprop.profiles.DEFAULT_PRESSURE_FLOOR_FRACTION` of the highest
            stage pressure. With ``vacuum_end="floor"``, pass ``0.0`` for a true
            vacuum end, which warns, because ``dp/dz`` diverges there.
        vacuum_end : {'choked', 'floor'} or None, optional
            How a pressure gradient's outer end below its choked pressure is treated.
            ``None`` (the default) means ``"choked"`` for a circular hollow guide: the
            fill follows the bore (Pennetta et al. 2019, eq. 6) and an end exhausting
            into vacuum takes the choked pressure of the flow
            (:func:`solprop.gasflow.steady_fill`), with a warning if the flow is not
            laminar. ``"floor"`` concerns the ends only: it keeps the declared end
            pressures, raised only to `pressure_floor`, for a comparison with a model
            that imposes them, while a taper's fill still follows the bore (on a
            uniform bore, bitwise the uniform law). Refused beside a constant
            pressure, a pressure profile or a Mixture, which have no end to choke,
            and on the general route. Every mode declaring ``core_radius`` -- every
            ``MarcatiliMode`` subclass, a ``TabulatedMode`` built from arrays
            included -- is taken as a circular bore of that radius; a rectangular
            guide keeps the uniform law.
        wavelength : float
            Reference wavelength, m, and the pulse's centre wavelength.
        wavelength_limits : tuple of float or None, optional
            Shortest and longest wavelength of the frequency window, m. Required unless
            `grid` is given.
        trange : float or None, optional
            Requested width of the time window, s. Required unless `grid` is given.
        grid : solprop.grid.TimeGrid or None, optional
            A grid of one's own, instead of the three keywords above; the only way to
            reach ``dt``, ``ffac`` or the transform size.
        envelope : bool, optional
            Propagate an envelope. Plasma is off by default; explicit ``plasma=True``
            selects cycle averaging of instantaneous rates. Plasma forbids THG.
        fwhm : float or None, optional
            Pulse duration at half maximum, s.
        width : float or None, optional
            Natural width of a sech pulse, s; give this or `fwhm`.
        shape : {'gauss', 'sech'}, optional
            Transform-limited shape of the input.
        energy : float or None, optional
            Pulse energy, J. Give this or `peak_power`.
        peak_power : float or None, optional
            Peak power, W, realised after the spectral phases.
        phases : sequence of float, optional
            Taylor spectral phases about `wavelength`, optics convention: CEP in rad,
            group delay in s, GDD in s^2, and so on.
        pulses : sequence of solprop.pulses.Pulse or None, optional
            Several pulses launched together, for two-colour and multi-colour
            inputs: each a :mod:`solprop.pulses` object with its own wavelength,
            duration, energy or peak power, phase and delay; the input is the sum of
            their fields (:class:`~solprop.pulses.PulseSum`). Replaces the
            single-pulse keywords `fwhm`, `width`, `shape`, `energy`,
            `peak_power` and `phases`, which must then be left unset; `wavelength` still
            sets the grid's reference wavelength.
        susceptibility : GasSusceptibility or None, optional
            A gas susceptibility replacing the one `gas` selects, for a comparison
            against another code's refractive data. Refused for a fraction mapping
            or an existing mixture, which own their species models.
        temperature : float, ZProfile or None, optional
            Gas temperature, K, or a path of it along z in m. Sets the density the
            pressure path maps to, so it changes both the dispersion and the
            nonlinearity. A path also weights a steady-flow gradient by the gas's
            ``eta(T) T`` (docs/guide/waveguides/profiles.md), makes a shared mixture's
            populations independent parameters, and holds the rotational Raman
            populations at one reference temperature with a
            :class:`~solprop.exceptions.RamanTemperatureWarning`
            (:func:`solprop.nonlinear.mixtures.raman_temperature`). A constant path
            is the number, bit for bit. ``None`` (the default) is 293.15 K. A Mixture
            owns its temperature: beside one, a temperature must be omitted or be the
            mixture's own (the same number or path object), and any other is refused by
            name.
        kerr : bool, optional
            Include the Kerr response.
        thg : bool or None, optional
            Include third-harmonic generation in the Kerr term; ``None`` takes
            ``not envelope``.
        plasma : bool, IonisationRate, Mapping or None, optional
            Include photoionisation. ``None`` takes ``not envelope``. ``True`` selects
            optmat's recommended rate model of each built-in gas (SEP 3, D1;
            :class:`~solprop.nonlinear.optmat_rates.OptmatRate`, or the model the
            selection's ``ionisation`` names), for a single gas and for every
            component of a mixture; a custom species needs its own rate. A
            rate object is used as given; a mapping selects a rate or switch per
            component label, and omitted labels are off.
        preionisation : float, ZProfile, Mapping or None, optional
            Initial ionised fraction in [0, 1], globally or per label: a number, or a
            path along z in m giving the fraction ionised before the pulse arrives at
            each position. None and omitted mapping labels retain the component's
            existing declaration.
        plasma_integration : {'band', 'trapezoid'} or None, optional
            How the field-grid plasma forms its population and free-current integrals
            (SEP 18): ``'band'`` filters each integrand with taps designed for the
            grid, exact to a certified 1e-5 across the active band; ``'trapezoid'``
            is the plain cumulative trapezoid, short by up to 11 % per integral at
            the top of the default active band, which the frozen witness comparisons
            use. ``None`` takes :class:`~solprop.nonlinear.responses.PlasmaResponse`'s
            default. Refused with envelope plasma, whose currents are converted
            exactly, and when the plasma is off.
        raman : bool, sequence of str, Mapping or None, optional
            ``None`` (the default) turns the Raman response on for every species
            whose optmat response is curated -- N2, O2, H2, D2, CH4 and SF6 -- leaves
            an atom without one, and warns (:class:`solprop.exceptions.SolpropWarning`)
            for a molecule whose response is provisional (N2O, unless the selection's
            ``raman`` is ``"provisional"``) rather than leaving it out silently; with
            ``kerr=False`` the default is off. ``False`` disables it; ``True`` selects
            the available models of a mixture and is refused by name for a single gas
            without data; explicit labels or
            :class:`~solprop.physdata.raman_types.MolecularOptions` select components
            and terms, and a named unavailable model fails explicitly.
        mode_average : bool, optional
            Propagate the one mode **mode-averaged**, as a uniform disc of its
            effective area, instead of resolving it on the transverse rule
            (``docs/guide/waveguides/mode-averaging.md``). The plasma then sees the
            area-mean field, below the on-axis one (``HE_11``'s axis intensity is
            1.768 times ``P/A_eff``), and ``peak_intensity`` stays on the axis.
            ``False``, the default, resolves the mode. Refused beside more than one
            mode and beside `n_r`, `n_theta` or a `radial_rule`.
            Refused beside `provider`.
        n_r : int or None, optional
            Radial nodes of the fixed rule, or the x-axis nodes of a Cartesian
            product or slab rule. ``None`` takes :func:`default_radial_order` of
            the mode count, or :func:`~solprop.modes.quadrature.cartesian_nodes` of
            the set's highest transverse order where that is larger.
        n_theta : int or None, optional
            Azimuthal nodes, or the y-axis nodes of a Cartesian product rule, whose
            default is :func:`~solprop.modes.quadrature.cartesian_nodes`. Refused
            for a set of purely radial modes, which need none, and for a slab set,
            which is uniform in y.
        radial_rule : str, optional
            The radial rule family (SEP 19): ``"gauss_kronrod"`` (the default), whose
            embedded Gauss subset is the quadrature alarm's error estimate;
            ``"gauss"``, with no alarm; or ``"field_scale"`` and
            ``"field_scale_radau"``, Gaussian in the HE_11 field scale for expensive
            responses, which need exactly one Marcatili HE_11 mode and have no alarm
            (``docs/guide/waveguides/radial-quadrature.md``).
        ionisation_field_range : tuple of float or None, optional
            Field amplitudes in V/m the rate table must cover. ``None`` derives them
            from the input's peak field over the axis and the transverse nodes; see
            :func:`solprop.driver.simulation.default_field_range`.
        absorber : solprop.boundaries.Absorber, False or None, optional
            The collar rates; ``None`` builds the default collar of the grid, and
            ``False`` switches both soft collars off and keeps the hard band mask
            (:func:`solprop.boundaries.no_absorber`; SEP 21).
        absorber_scheme : {'nonlinear', 'split_step'} or None, optional
            How the temporal collar is applied (ADR-025); ``None`` is the default,
            inside every stage.
        boundary_check : solprop.monitor.BoundaryCheck or None, optional
            How the run watches its time window's edges (SEP 21): the default warns,
            then stops, when light above the shot-noise floor reaches them; ``None``
            watches nothing, for a window kept narrow on purpose.
        noise : {'default', None, 'input', 'modified'}, optional
            Which shot-noise model to carry.
            ``"default"``, the default, resolves to ``"input"`` wherever the run can
            carry noise, and records the resolution (SEP 37).
        seed : int or None, optional
            Entropy of the noise generator.
        rtol : float or None, optional
            Relative step tolerance; the default is :data:`WAVEGUIDE_RTOL`, the modal
            capillary default. Calibrate it for a production run
            (``docs/dev/stepper.md`` section 3).
        atol : float, optional
            Absolute step tolerance, in the square root of watts.
        dz_max : float or None, optional
            Cap on the step, m; ``None`` takes half the waveguide.
        dz_init : float or None, optional
            First step, m; ``None`` takes ``length * 1e-4``.
        schedule : solprop.stepper.schedule.FixedSchedule or None, optional
            A fixed schedule to replay instead of stepping adaptively.
        integrator : Multistep, RungeKutta or LowStorageRungeKutta, optional
            The method that steps the run
            (``docs/guide/numerics/choosing-a-stepper.md``).
            ``None``, the default, takes ``RungeKutta()``, the interaction-picture
            DOPRI5 march (SEP 37). ``Multistep()`` is the low-call exponential Adams
            method. ``LowStorageRungeKutta()`` replays `schedule` in three registers.
        save_planes : sequence of float or None, optional
            Positions to save at, m. Give this or `n_save`.
        n_save : int or None, optional
            How many planes to save, counting ``z = 0``.
        stats : solprop.stats.StatsSpec or None, optional
            Which statistics to record; ``None`` takes
            :func:`solprop.stats.default_spec` for a multimode run with the profiles,
            or for a mode-averaged one under `mode_average`, plus every statistic
            only a backend can compute.
        node_budget : int, optional
            Bytes of node-chunk buffers the response pass may hold.
        backend : str, optional
            Registry name, or ``"auto"`` (the default) to prefer ``native`` if installed
            and otherwise use Numba CPU. Import or construction errors do not fall back.
        precision : {'double', 'single'}, optional
            The arithmetic the run is held in (SEP 33): ``"double"``, the default, or
            ``"single"``, which keeps ``rtol`` at :data:`WAVEGUIDE_RTOL` by default
            (``docs/guide/performance/single-precision.md``), takes ``RungeKutta()``,
            carries no noise, and resolves ``backend="auto"`` to the experimental
            ``mlx`` where MLX and a Metal GPU are present, with ``numpy_ref`` as the
            reference (:attr:`SimulationSettings.precision`).
        polarisation : str, float or None, optional
            Refused; this path is scalar.
        taper_coupling : bool, optional
            Carry the taper coupling of a supplied capillary set on a varying radius, as
            :meth:`from_capillary` does; ``False`` is the opt-out to the adiabatic
            route. Refused as ``False`` beside a `provider`, whose moving guide carries
            its own connection.
        memory_budget : int or str or None, optional
            The budget a guide's route is admitted against **before** its dense E/H
            reference and maps are built, which happens here rather than when the run
            starts (:func:`solprop.driver.memory.admit_guide_memory`): a declaration as
            :meth:`run` takes it, and ``None`` takes ``SOLPROP_MEMORY_BUDGET`` or the
            machine's memory, warned on. Requires `provider`.
        materials : solprop.physdata.selection.MaterialSelection or None, optional
            Take the fill from optmat under this selection: a gas or fraction mapping
            of built-in gases, ``"air"`` and ``Gas.Air`` resolve through
            :func:`solprop.physdata.species` and
            :meth:`solprop.physdata.mixtures.Mixture.air` with it. The supplied modes,
            their claddings, a supplied mixture or susceptibility, and a provider's
            material regions are taken as they are. The run records its material
            manifest. ``None`` (the default) is
            ``MaterialSelection()``.

        Returns
        -------
        Simulation
            A configured simulation, ready to translate or run.

        Raises
        ------
        DriverError
            If the modes cannot bind a gas mixture, a gas keyword is given beside a
            provider, a general-guide keyword is given without one, or any supplied
            setting is invalid.
        MemoryBudgetError
            If a guide's reference and maps exceed a budget declared to refuse;
            nothing of them has been built.
        """
        if provider is not None:
            return cls._from_general(
                provider,
                modes,
                length,
                gas=gas,
                pressure=pressure,
                pressure_floor=pressure_floor,
                vacuum_end=vacuum_end,
                channel=channel,
                incident=incident,
                coupling=coupling,
                kerr_form=kerr_form,
                observable_rtol=observable_rtol,
                observable_atol=observable_atol,
                wavelength=wavelength,
                wavelength_limits=wavelength_limits,
                trange=trange,
                grid=grid,
                envelope=envelope,
                fwhm=fwhm,
                width=width,
                shape=shape,
                energy=energy,
                peak_power=peak_power,
                phases=phases,
                pulses=pulses,
                susceptibility=susceptibility,
                temperature=temperature,
                kerr=kerr,
                thg=thg,
                plasma=plasma,
                preionisation=preionisation,
                plasma_integration=plasma_integration,
                raman=raman,
                mode_average=mode_average,
                n_r=n_r,
                n_theta=n_theta,
                radial_rule=radial_rule,
                ionisation_field_range=ionisation_field_range,
                absorber=absorber,
                absorber_scheme=absorber_scheme,
                boundary_check=boundary_check,
                noise=noise,
                seed=seed,
                rtol=rtol,
                atol=atol,
                dz_max=dz_max,
                dz_init=dz_init,
                schedule=schedule,
                integrator=integrator,
                save_planes=save_planes,
                n_save=n_save,
                stats=stats,
                node_budget=node_budget,
                backend=backend,
                precision=precision,
                memory_budget=memory_budget,
                polarisation=polarisation,
                taper_coupling=taper_coupling,
                materials=MaterialSelection() if materials is None else materials,
            )
        for name, value in (
            ("memory_budget", memory_budget),
            ("channel", channel),
            ("incident", incident),
            ("coupling", coupling),
            ("kerr_form", None if kerr_form == "real" else kerr_form),
            ("observable_rtol", observable_rtol),
            ("observable_atol", observable_atol),
        ):
            if value is not None:
                msg = (
                    f"{name}= belongs to the general modal route and needs a "
                    "provider= beside it; the gas-filled waveguide route launches "
                    "one pulse into its lowest mode and controls the step alone"
                )
                raise DriverError(msg)
        if gas is None:
            msg = (
                "from_waveguide needs a gas fill for its supplied modes, or a "
                "provider= whose material regions own their own laws and responses "
                "(the general modal route)"
            )
            raise DriverError(msg)
        return cls._from_hollow(
            radius=None,
            modes=modes,
            mode_average=mode_average,
            waveguide=True,
            length=length,
            gas=gas,
            pressure=pressure,
            pressure_floor=pressure_floor,
            vacuum_end=vacuum_end,
            wavelength=wavelength,
            wavelength_limits=wavelength_limits,
            trange=trange,
            grid=grid,
            envelope=envelope,
            fwhm=fwhm,
            width=width,
            shape=shape,
            energy=energy,
            peak_power=peak_power,
            phases=phases,
            pulses=pulses,
            susceptibility=susceptibility,
            temperature=temperature,
            kerr=kerr,
            thg=thg,
            plasma=plasma,
            preionisation=preionisation,
            plasma_integration=plasma_integration,
            raman=raman,
            n_r=n_r,
            n_theta=n_theta,
            radial_rule=radial_rule,
            ionisation_field_range=ionisation_field_range,
            absorber=absorber,
            absorber_scheme=absorber_scheme,
            boundary_check=boundary_check,
            noise=noise,
            seed=seed,
            rtol=_tolerance(rtol, precision, WAVEGUIDE_RTOL),
            atol=atol,
            dz_max=dz_max,
            dz_init=dz_init,
            schedule=schedule,
            integrator=integrator,
            route_integrator=RungeKutta(),
            save_planes=save_planes,
            n_save=n_save,
            stats=stats,
            node_budget=node_budget,
            backend=backend,
            precision=precision,
            polarisation=polarisation,
            taper_coupling=taper_coupling,
            materials=MaterialSelection() if materials is None else materials,
        )

    @classmethod
    def _from_general(  # noqa: PLR0913 - it refuses from_waveguide's whole surface
        cls,
        provider: ModalGuide,
        modes: Sequence[Mode],
        length: float,
        *,
        channel: int | str | None,
        incident: SpatialPulse | None,
        coupling: CouplingModel | None,
        kerr_form: KerrForm,
        observable_rtol: float | None,
        observable_atol: float | None,
        wavelength: float,
        wavelength_limits: tuple[float, float] | None,
        trange: float | None,
        grid: TimeGrid | None,
        fwhm: float | None,
        width: float | None,
        shape: str,
        energy: float | None,
        peak_power: float | None,
        phases: Sequence[float] | np.ndarray,
        pulses: Sequence[Pulse] | None,
        absorber: Absorber | Literal[False] | None,
        absorber_scheme: AbsorberScheme | None,
        boundary_check: BoundaryCheck | None,
        noise: NoiseSetting,
        seed: int | None,
        rtol: float | None,
        atol: float,
        dz_max: float | None,
        dz_init: float | None,
        schedule: FixedSchedule | None,
        save_planes: Sequence[float] | np.ndarray | None,
        n_save: int | None,
        stats: StatsSpec | None,
        backend: str,
        precision: PrecisionName,
        memory_budget: int | str | None,
        materials: MaterialSelection | None = None,
        integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None,
        **refused: object,
    ) -> "Simulation":  # noqa: UP037 - beartype resolves it while the body still runs
        """Build a general modal run from a guide that solves its own fields.

        The provider half of :meth:`from_waveguide`, kept apart because it shares no
        physics with the gas-filled one: no fill, no mixture, no static transverse
        quadrature, no declared response list, and a launch that crosses no
        normalisation boundary. Every keyword of the gas-filled path arrives in
        `refused` and is refused by name if it was set.

        Parameters
        ----------
        provider : solprop.modes.guide.ModalGuide
            The guide, carrying its sampling, its channels and its regions.
        modes : sequence of solprop.modes.protocol.Mode
            Empty, or exactly the guide's own ordered channels.
        length : float
            Guide length, m.
        channel : int or str or None
            Which channel the pulse enters; ``None`` takes the first.
        incident : solprop.pulses.SpatialPulse or None
            An incident beam to couple into the channels instead, as
            :meth:`from_waveguide` documents it.
        coupling : solprop.coupling.models.CouplingModel or None
            The model it is coupled under; required beside `incident` and refused
            without it.
        kerr_form : {'real', 'analytic'}
            Which instantaneous cubic the regions respond with, as
            :meth:`from_waveguide` documents it.
        observable_rtol, observable_atol : float or None
            Accepted only as ``None``, which means the solver's own tolerances.
        wavelength, wavelength_limits, trange, grid : object
            The grid, as :func:`_capillary_grid` takes them; a **field** grid.
        fwhm, width, shape, energy, peak_power, phases, pulses : object
            The input pulse, as :func:`_shaped_pulse` takes it.
        absorber : solprop.boundaries.Absorber, False or None
            The collar rates; ``None`` builds the grid's default collar, and ``False``
            switches both soft collars off and keeps the hard band mask.
        absorber_scheme : {'nonlinear', 'split_step'} or None
            How the temporal collar is applied; ``None`` is the default scheme.
        boundary_check : solprop.monitor.BoundaryCheck or None
            How the run watches its time window's edges, as :meth:`from_waveguide`
            takes it.
        noise : {'default', None, 'input', 'modified'}
            Which shot-noise model to carry (``docs/guide/media/noise.md``).
            ``"default"``, the default, resolves to ``"input"`` wherever the run can
            carry noise, and records the resolution (SEP 37).
        seed : int or None
            Entropy of the noise generator.
        rtol, atol, dz_max, dz_init, schedule, save_planes, n_save : object
            The step controller and the schedule, as the other entry points take them.
        stats : solprop.stats.StatsSpec or None
            Which statistics to record; ``None`` takes :func:`_general_stats`.
        backend : str
            Registry name, or ``"auto"``.
        precision : {'double', 'single'}
            The arithmetic the run is held in; the general route is refused under
            single precision.
        memory_budget : int or str or None
            The budget the guide's route is admitted against before it is built.
        materials : solprop.physdata.selection.MaterialSelection or None, optional
            The run's material selection, recorded in its manifest. Under it a
            region's default :class:`~solprop.nonlinear.raman.SilicaTensorResponse`
            is optmat's tensor bundle (SEP 3, Q8), and a region's gas fill
            (:class:`~solprop.modes.material.GasKerr`) takes its electronic policy:
            the Kerr coefficient at the carrier, or SEP 6's dispersive cubic.
        integrator : Multistep, RungeKutta or LowStorageRungeKutta, optional
            The method that steps the run; ``None`` takes ``Multistep()``, and
            ``LowStorageRungeKutta()`` replays `schedule` in three registers.
        **refused : object
            Every gas-filled keyword; each is refused by name when it is set.

        Returns
        -------
        Simulation
            A configured general modal simulation, ready to translate or run.

        Raises
        ------
        DriverError
            Naming the keyword, the grid or the channel that cannot be carried.
        """
        _refuse_gas_keywords(refused)
        provider = _selected_guide(
            provider, MaterialSelection() if materials is None else materials
        )
        for name, value in (
            ("observable_rtol", observable_rtol),
            ("observable_atol", observable_atol),
        ):
            if value is not None:
                msg = (
                    f"{name}={value!r}: a general modal run's observables are its "
                    "channel amplitudes in a basis orthonormal in true power, so "
                    "the controller's own rtol and atol already bound them, and "
                    "the separate allocation a different number would ask for -- "
                    "over the table's frequency interpolation, the transverse rule "
                    "and the step -- is not implemented. Pass None, and set rtol "
                    "and atol"
                )
                raise DriverError(msg)
        working = _capillary_grid(
            grid,
            length,
            wavelength,
            wavelength_limits,
            trange,
            envelope=False,
            thg=False,
        )
        if working.kind != "field":
            msg = (
                f"a general modal guide propagates on a field grid and this run is "
                f"on an {working.kind} grid: its leaves are tabulated on the "
                "canonical increasing coarse axis, and its source is the "
                "analytic-signal polarisation of a carrier-resolved field"
            )
            raise DriverError(msg)
        channels = tuple(provider.mode_collection.modes)
        if tuple(modes) not in ((), channels):
            msg = (
                f"a general modal guide supplies its own {len(channels)} ordered "
                f"channel(s), validated against the provider's fields, and this "
                f"call passes {len(tuple(modes))} mode(s) of its own: pass "
                "modes=() beside provider=, or the guide's own "
                "provider.mode_collection.modes"
            )
            raise DriverError(msg)
        planes = _save_planes(length, save_planes, n_save)
        pulse = _general_pulse(
            incident,
            channel,
            shape,
            wavelength=wavelength,
            fwhm=fwhm,
            width=width,
            energy=energy,
            peak_power=peak_power,
            phases=phases,
            pulses=pulses,
        )
        if provider.sampling is not None:
            # Before the route and not after it: the route builds the dense E/H
            # reference and the maps, which on a large guide are nearly all of the
            # run's memory, and admission after them could only say so.
            admit_guide_memory(
                provider,
                working,
                z_max=float(np.max(planes)),
                backend=backend,
                memory_budget=memory_budget,
            )
        built = (pulse.temporal if isinstance(pulse, SpatialPulse) else pulse).build(
            working
        )
        try:
            route = general_modal_route(
                provider,
                working,
                z_max=float(np.max(planes)),
                omega_reference=float(working.omega0),
                kerr_form=kerr_form,
                selection=MaterialSelection() if materials is None else materials,
                # The pulse is built first so that the route's occupied-support basis
                # gate is weighted by where this run's light actually is rather than
                # over the whole declared band. The launch's channel distribution does
                # not enter that weight -- it is summed over channels -- so the pulse
                # alone is enough, which is what lets a projected launch, which needs
                # the route's own dual, still be weighted by its own spectrum. It
                # reaches no leaf.
                launch=built,
            )
        except TranslationError as error:
            # The bridge belongs to the translation layer and says so; a caller of
            # `from_waveguide` is told what this entry point promises, and
            # `DriverError` is what its `Raises` names. The message is the bridge's
            # own, unchanged, and the cause is kept.
            raise DriverError(str(error)) from error
        beam = pulse if isinstance(pulse, SpatialPulse) else None
        if (beam is None) != (coupling is None):
            msg = (
                "an incident beam is coupled under a declared model: give incident= "
                "with coupling=CouplingModel('vector_paraxial', model_error=...), or "
                "neither"
            )
            raise DriverError(msg)
        if beam is not None:
            # Checked before the port is built: a guide whose first region is off the
            # axis has no core for a beam to enter through.
            _core_index(provider.regions, working)
            # The launch is the coupling's, installed through `with_input` below.
            field0 = np.zeros((len(channels), 1, working.n_omega), dtype=np.complex128)
        else:
            field0 = _in_channel(built, provider, channel)
        boundary, scheme = _boundary(absorber, absorber_scheme, working)
        settings = SimulationSettings(
            controller=PIController(
                rtol=_tolerance(rtol, precision, WAVEGUIDE_RTOL), atol=atol
            ),
            save_planes=planes,
            dz_max=length / 2 if dz_max is None else dz_max,
            dz_init=length * 1e-4 if dz_init is None else dz_init,
            auto_init=dz_init is None,
            schedule=schedule,
            integrator=integrator,
            _route_integrator=RungeKutta(),
            absorber=boundary,
            absorber_scheme=scheme,
            boundary_check=boundary_check,
            # The route carries its nodes, its materials and its normalisation in
            # its own leaves: the static transverse quadrature, the component
            # selection, the density path and the index override are all refused
            # by the translation layer and are not settings of this run.
            stats=_general_stats() if stats is None else stats,
            backend=backend,
            precision=precision,
        )
        noise_model, resolution = _resolve_noise(noise, seed, precision)
        draw = functools.partial(_general_noise, noise_model, working, guide=provider)
        simulation = cls(
            working,
            channels,
            [],
            field0,
            settings=settings,
            noise=draw(seed),
            noise_resolution=resolution,
            configuration={
                "waveguide": {
                    "modes": [ModeRecord.from_mode(mode) for mode in channels],
                    "length": length,
                    "provider": dict(provider.identity),
                },
                **({} if beam is not None else {"pulse": pulse}),
            },
            general=route,
            guide=provider,
            materials=materials,
        )
        _redrawable(simulation, draw, seed)
        if beam is None:
            return simulation
        return _incident_launch(simulation, beam, cast("CouplingModel", coupling))

    @classmethod
    def _from_hollow(  # noqa: PLR0913, PLR0915 - the interface's keyword surface
        cls,
        radius: float | ZProfile | None,
        length: float,
        gas: Gas | str | Mapping[str, float] | Mixture,
        pressure: Pressure | None = None,
        *,
        waveguide: bool = False,
        pressure_floor: float | None = None,
        vacuum_end: VacuumEnd | None = None,
        wavelength: float,
        wavelength_limits: tuple[float, float] | None = None,
        trange: float | None = None,
        grid: TimeGrid | None = None,
        envelope: bool = False,
        fwhm: float | None = None,
        width: float | None = None,
        shape: str = "gauss",
        energy: float | None = None,
        peak_power: float | None = None,
        phases: Sequence[float] | np.ndarray = (),
        pulses: Sequence[Pulse] | None = None,
        modes: int | Sequence[Mode] = 1,
        mode_average: bool = False,
        lattice: TubeLattice | None = None,
        wall: CoherentWall | None = None,
        taper_coupling: bool = True,
        model: MarcatiliModel = "full",
        loss: LossSpec = True,
        cladding: GlassIndex | complex | None = None,
        susceptibility: GasSusceptibility | None = None,
        temperature: float | ZProfile | None = None,
        kerr: bool = True,
        thg: bool | None = None,
        plasma: bool
        | IonisationRate
        | Mapping[str, bool | IonisationRate]
        | None = None,
        preionisation: float | ZProfile | Mapping[str, float | ZProfile] | None = None,
        plasma_integration: IntegrationPolicy | None = None,
        raman: bool
        | Sequence[str]
        | Mapping[str, bool | MolecularOptions]
        | None = None,
        n_r: int | None = None,
        n_theta: int | None = None,
        radial_rule: RadialRule = "gauss_kronrod",
        ionisation_field_range: tuple[float, float] | None = None,
        absorber: Absorber | Literal[False] | None = None,
        absorber_scheme: AbsorberScheme | None = None,
        boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
        noise: NoiseSetting = "default",
        seed: int | None = None,
        rtol: float,
        atol: float = WAVEGUIDE_ATOL,
        dz_max: float | None = None,
        dz_init: float | None = None,
        schedule: FixedSchedule | None = None,
        integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None,
        route_integrator: RungeKutta | None = None,
        interpolate_planes: bool = False,
        step_planes: int | None = None,
        save_planes: Sequence[float] | np.ndarray | None = None,
        n_save: int | None = None,
        stats: StatsSpec | None = None,
        node_budget: int = DEFAULT_NODE_BUDGET,
        backend: str = "auto",
        precision: PrecisionName = "double",
        polarisation: str | float | None = None,
        materials: MaterialSelection | None = None,
        tdse: TDSEGasResponse | None = None,
        tdse_execution: TDSEExecution | None = None,
    ) -> "Simulation":  # noqa: UP037 - beartype resolves it while the body still runs
        """Build a simulation of a pulse in a gas-filled hollow waveguide.

        The shared body of :meth:`from_capillary` and :meth:`from_waveguide`. The mode
        set is the first `modes` HE_1m modes of the capillary unless a set is given; the
        transverse integral is a fixed quadrature rule
        (``docs/guide/waveguides/mode-sets.md``); the
        gas density along z is the pressure path through the equation of state, and the
        same path sets the modes' core index, so a gradient fill disperses and broadens
        consistently.

        See ``docs/guide/setup/running.md`` for the discussion and a worked example.

        Parameters
        ----------
        radius : float or solprop.profiles.ZProfile
            Core radius, m, constant or along z.
        length : float
            Capillary length, m.
        gas : Gas, str, Mapping or Mixture
            Built-in gas, mole fractions keyed by species name, or ordered populations
            carrying their own density paths. A mapping uses total pressure.
        pressure : float, sequence, ZProfile or None, optional
            None uses one bar for a gas or fractions. Omit pressure beside a Mixture;
            every explicit pressure is refused because it already carries its paths.
            Fill pressure in **bar**: a number for a constant fill, a pair
            ``(p_in, p_out)`` for the steady-flow gradient of
            :meth:`solprop.profiles.PressureGradient.two_point`, or ``(z, p)`` with the
            positions and the pressures there for
            :meth:`solprop.profiles.PressureGradient.from_stages`.
        waveguide : bool, optional
            Bind supplied geometry instead of constructing a capillary.
        pressure_floor : float or None, optional
            Lowest pressure the fill is allowed to reach, bar; ``None`` takes
            :data:`solprop.profiles.DEFAULT_PRESSURE_FLOOR_FRACTION` of the highest
            stage pressure. With ``vacuum_end="floor"``, pass ``0.0`` for a true
            vacuum end, which warns, because ``dp/dz`` diverges there.
        vacuum_end : {'choked', 'floor'} or None, optional
            How a pressure gradient's outer end below its choked pressure is treated.
            ``None`` (the default) means ``"choked"`` for a circular hollow guide: the
            fill follows the bore (Pennetta et al. 2019, eq. 6) and an end exhausting
            into vacuum takes the choked pressure of the flow
            (:func:`solprop.gasflow.steady_fill`), with a warning if the flow is not
            laminar. ``"floor"`` concerns the ends only: it keeps the declared end
            pressures, raised only to `pressure_floor`, for a comparison with a model
            that imposes them, while a taper's fill still follows the bore (on a
            uniform bore, bitwise the uniform law). Refused beside a constant
            pressure, a pressure profile or a Mixture, which have no end to choke,
            and on the general route. Every mode declaring ``core_radius`` -- every
            ``MarcatiliMode`` subclass, a ``TabulatedMode`` built from arrays
            included -- is taken as a circular bore of that radius; a rectangular
            guide keeps the uniform law.
        wavelength : float
            Reference wavelength, m, and the pulse's centre wavelength.
        wavelength_limits : tuple of float or None, optional
            Shortest and longest wavelength of the frequency window, m. Required unless
            `grid` is given.
        trange : float or None, optional
            Requested width of the time window, s. Required unless `grid` is given.
        grid : solprop.grid.TimeGrid or None, optional
            A grid of one's own, instead of the three keywords above; the only way to
            reach ``dt``, ``ffac`` or the transform size.
        envelope : bool, optional
            Propagate an envelope. Plasma is off by default; explicit ``plasma=True``
            selects cycle averaging of instantaneous rates. Plasma forbids THG.
        fwhm : float or None, optional
            Pulse duration at half maximum, s.
        width : float or None, optional
            Natural width of a sech pulse, s; give this or `fwhm`.
        shape : {'gauss', 'sech'}, optional
            Transform-limited shape of the input.
        energy : float or None, optional
            Pulse energy, J. Give this or `peak_power`.
        peak_power : float or None, optional
            Peak power, W, realised after the spectral phases.
        phases : sequence of float, optional
            Taylor spectral phases about `wavelength`, optics convention: CEP in rad,
            group delay in s, GDD in s^2, and so on.
        pulses : sequence of solprop.pulses.Pulse or None, optional
            Several pulses launched together, for two-colour and multi-colour
            inputs: each a :mod:`solprop.pulses` object with its own wavelength,
            duration, energy or peak power, phase and delay; the input is the sum of
            their fields (:class:`~solprop.pulses.PulseSum`). Replaces the
            single-pulse keywords `fwhm`, `width`, `shape`, `energy`,
            `peak_power` and `phases`, which must then be left unset; `wavelength` still
            sets the grid's reference wavelength.
        modes : int or sequence of Mode, optional
            How many HE_1m modes to propagate, or the mode set itself. A set is checked
            for orthonormality. The own HE_1m set is synthesised along x; a supplied set
            resolves the transverse components its modes need -- E_x and E_y for TE_0m,
            TM_0m, EH, higher HE or rotated modes -- with or without a coherent wall.
        mode_average : bool, optional
            Propagate the one mode **mode-averaged**: a uniform disc of the mode's
            effective area instead of a transverse rule
            (``docs/guide/waveguides/mode-averaging.md``). The plasma then sees the
            area-mean field, below the on-axis one (``HE_11``'s axis intensity is
            1.768 times ``P/A_eff``), while ``peak_intensity`` stays on the axis.
            ``False``, the default, resolves the mode on the radial rule. Refused
            beside more than one mode and beside `n_r`, `n_theta` or a `radial_rule`.
        lattice : TubeLattice or None, optional
            Tube geometry for an integer mode count, selecting Vincetti modes.
            Refused with supplied modes, which carry their own geometry, and with
            ``model="reduced"``. None selects ordinary capillary modes.
        wall : solprop.modes.CoherentWall or None, optional
            Declare the scalar **coherent** wall block: the outgoing Robin boundary
            condition condensed onto the retained basis, which couples every ``HE_1m``
            channel to every other with a definite phase instead of giving each its own
            Marcatili attenuation. It needs the contiguous set ``HE_11`` to ``HE_1N`` --
            or, with ``impedance="te_tm"``, one azimuthal order -- on a constant or a
            varying guide, and a scalar run; it carries the default ``"input"`` noise
            and refuses ``"modified"``. All three backends execute it (a
            ``method="cf4"`` wall ``numpy_ref`` alone); every other mode set or
            geometry is refused by name. ``None``, the default, carries no wall:
            the modes keep their own Marcatili loss, and a multimode taper its taper
            coupling (below).
        taper_coupling : bool, optional
            On a multimode capillary whose radius varies along z, carry the
            **taper coupling** -- the connection ``<psi_m|d_z psi_n>`` that moves
            power between the ``HE_1m`` channels as the bore changes -- lossless or
            not. ``True``, the default; ``False`` is the explicit opt-out to the
            **adiabatic** route, on which each channel follows its local mode and
            exchanges no power. It is refused beside `wall`, which always carries
            the connection, and has no effect on one channel, a constant bore (a
            pressure gradient included, which does not change the modes' shape) or
            a set that is not the contiguous ``HE_11`` to ``HE_1N`` block
            (``docs/guide/waveguides/capillaries.md``, "Tapers").
        model : {'full', 'reduced'}, optional
            Which Marcatili model the modes use; beside a supplied set, refused unless
            it is the default or the modes' own.
        loss : bool, float or callable, optional
            Propagation loss of the modes; beside a supplied set, refused unless it
            is the default or the modes' own.
        cladding : GlassIndex, complex or None, optional
            Cladding index; ``None`` is the package's cited silica. Beside a
            supplied set, refused unless it is the default or the modes' own.
        susceptibility : GasSusceptibility or None, optional
            A gas susceptibility replacing the one `gas` selects, for a comparison
            against another code's refractive data. Refused for a fraction mapping
            or an existing mixture, which own their species models.
        temperature : float, ZProfile or None, optional
            Gas temperature, K, or a path of it along z in m. Sets the density the
            pressure path maps to, so it changes both the dispersion and the
            nonlinearity. A path also weights a steady-flow gradient by the gas's
            ``eta(T) T`` (docs/guide/waveguides/profiles.md), makes a shared mixture's
            populations independent parameters, and holds the rotational Raman
            populations at one reference temperature with a
            :class:`~solprop.exceptions.RamanTemperatureWarning`
            (:func:`solprop.nonlinear.mixtures.raman_temperature`). A constant path
            is the number, bit for bit. ``None`` (the default) is 293.15 K. A Mixture
            owns its temperature: beside one, a temperature must be omitted or be the
            mixture's own (the same number or path object), and any other is refused by
            name.
        kerr : bool, optional
            Include the Kerr response.
        thg : bool or None, optional
            Include third-harmonic generation in the Kerr term; ``None`` takes
            ``not envelope``.
        plasma : bool, IonisationRate, Mapping or None, optional
            Include photoionisation. ``None`` takes ``not envelope``. ``True`` selects
            optmat's recommended rate model of each built-in gas (SEP 3, D1;
            :class:`~solprop.nonlinear.optmat_rates.OptmatRate`, or the model the
            selection's ``ionisation`` names), for a single gas and for every
            component of a mixture; a custom species needs its own rate. A
            rate object is used as given; a mapping selects a rate or switch per
            component label, and omitted labels are off.
        preionisation : float, ZProfile, Mapping or None, optional
            Initial ionised fraction in [0, 1], globally or per label: a number, or a
            path along z in m giving the fraction ionised before the pulse arrives at
            each position. None and omitted mapping labels retain the component's
            existing declaration.
        plasma_integration : {'band', 'trapezoid'} or None, optional
            How the field-grid plasma forms its population and free-current integrals
            (SEP 18): ``'band'`` filters each integrand with taps designed for the
            grid, exact to a certified 1e-5 across the active band; ``'trapezoid'``
            is the plain cumulative trapezoid, short by up to 11 % per integral at
            the top of the default active band, which the frozen witness comparisons
            use. ``None`` takes :class:`~solprop.nonlinear.responses.PlasmaResponse`'s
            default. Refused with envelope plasma, whose currents are converted
            exactly, and when the plasma is off.
        raman : bool, sequence of str, Mapping or None, optional
            ``None`` (the default) turns the Raman response on for every species
            whose optmat response is curated -- N2, O2, H2, D2, CH4 and SF6 -- leaves
            an atom without one, and warns (:class:`solprop.exceptions.SolpropWarning`)
            for a molecule whose response is provisional (N2O, unless the selection's
            ``raman`` is ``"provisional"``) rather than leaving it out silently; with
            ``kerr=False`` the default is off. ``False`` disables it; ``True`` selects
            the available models of a mixture and is refused by name for a single gas
            without data; explicit labels or
            :class:`~solprop.physdata.raman_types.MolecularOptions` select components
            and terms, and a named unavailable model fails explicitly.
        n_r : int or None, optional
            Radial nodes of the fixed rule. ``None`` takes
            :func:`default_radial_order` of the mode count.
        n_theta : int or None, optional
            Azimuthal nodes. Refused for a set of purely radial modes, which need none.
        radial_rule : str, optional
            The radial rule family (SEP 19): ``"gauss_kronrod"`` (the default), whose
            embedded Gauss subset is the quadrature alarm's error estimate;
            ``"gauss"``, with no alarm; or ``"field_scale"`` and
            ``"field_scale_radau"``, Gaussian in the HE_11 field scale for expensive
            responses, which need exactly one Marcatili HE_11 mode and have no alarm
            (``docs/guide/waveguides/radial-quadrature.md``).
        ionisation_field_range : tuple of float or None, optional
            Field amplitudes in V/m the rate table must cover. ``None`` derives them
            from the input's peak field over the axis and the transverse nodes; see
            :func:`solprop.driver.simulation.default_field_range`.
        absorber : solprop.boundaries.Absorber, False or None, optional
            The collar rates; ``None`` builds the default collar of the grid, and
            ``False`` switches both soft collars off and keeps the hard band mask
            (:func:`solprop.boundaries.no_absorber`; SEP 21).
        absorber_scheme : {'nonlinear', 'split_step'} or None, optional
            How the temporal collar is applied (ADR-025): inside every stage, the
            default, or once per accepted step. The low-call Adams methods of
            SEP 13 step much further with ``'split_step'``, whose first-order
            splitting error stays inside the collar.
        boundary_check : solprop.monitor.BoundaryCheck or None, optional
            How the run watches its time window's edges (SEP 21): the default warns,
            then stops, when light above the shot-noise floor reaches them; ``None``
            watches nothing, for a window kept narrow on purpose.
        noise : {'default', None, 'input', 'modified'}, optional
            Which shot-noise model to carry.
            ``"default"``, the default, resolves to ``"input"`` wherever the run can
            carry noise, and records the resolution (SEP 37).
        seed : int or None, optional
            Entropy of the noise generator.
        rtol : float
            Relative step tolerance, as the calling interface resolved it from its own
            default (:func:`_tolerance`). Calibrate it for a production run
            (``docs/dev/stepper.md`` section 3).
        atol : float, optional
            Absolute step tolerance, in the square root of watts.
        dz_max : float or None, optional
            Cap on the step, m; ``None`` takes half the capillary.
        dz_init : float or None, optional
            First step, m; ``None`` takes ``length * 1e-4``.
        schedule : solprop.stepper.schedule.FixedSchedule or None, optional
            A fixed schedule to replay instead of stepping adaptively.
        integrator : Multistep, RungeKutta or LowStorageRungeKutta, optional
            The method that steps the run
            (``docs/guide/numerics/choosing-a-stepper.md``).
            ``None``, the default, takes ``Multistep()``: the low-call exponential
            Adams method, one response evaluation per step. ``RungeKutta()`` is the
            interaction-picture DOPRI5 march, for a cheap response or a state too
            large for the Adams history. ``LowStorageRungeKutta()`` replays
            `schedule` in three registers (SEP 24).
        route_integrator : RungeKutta or None, optional
            The interface's binary64 default where `integrator` is ``None``, if not
            the settings' own ``Multistep()``: ``RungeKutta()`` for
            :meth:`from_waveguide`, carried by its settings so that copies keep it.
        interpolate_planes : bool, optional
            Observe the save planes inside accepted steps instead of landing on them
            (SEP 14): through the Adams march's own polynomial, or DOPRI5's
            continuous extension with ``RungeKutta()``. Default ``False``.
        step_planes : int or None, optional
            Also save the end of every `step_planes`-th accepted step (SEP 14);
            ``None`` is ``0``, or ``1`` with `tdse` (SEP 20 section 5.8).
        save_planes : sequence of float or None, optional
            Positions to save at, m. Give this or `n_save`.
        n_save : int or None, optional
            How many planes to save, counting ``z = 0``.
        stats : solprop.stats.StatsSpec or None, optional
            Which statistics to record; ``None`` takes
            :func:`solprop.stats.default_spec` for a multimode run with the profiles,
            or for a mode-averaged one under `mode_average`, plus every statistic
            only a backend can compute.
        node_budget : int, optional
            Bytes of node-chunk buffers the response pass may hold.
        backend : str, optional
            Registry name, or ``"auto"`` (the default) to prefer ``native`` if installed
            and otherwise use Numba CPU. Import or construction errors do not fall back.
        precision : {'double', 'single'}, optional
            The arithmetic the run is held in (SEP 33): ``"double"``, the default, or
            ``"single"``, which takes ``RungeKutta()``, carries no noise, and
            resolves ``backend="auto"`` to the experimental ``mlx`` where MLX and a
            Metal GPU are present, with ``numpy_ref`` as the reference
            (:attr:`SimulationSettings.precision`); `rtol` comes resolved.
        polarisation : str, float or None, optional
            Refused; this path is scalar.
        materials : solprop.physdata.selection.MaterialSelection or None, optional
            Take the run's built-in materials from optmat under this selection: a gas or
            fraction mapping of built-in gases, ``"air"`` and ``Gas.Air`` resolve
            through :func:`solprop.physdata.species` and
            :meth:`solprop.physdata.mixtures.Mixture.air` with it, and the default
            cladding (``cladding=None`` with a numbered mode set) is optmat's
            recommended silica. A supplied mixture, mode set, cladding or susceptibility
            is taken as it is. The run records its material manifest.
            ``None`` (the default) is ``MaterialSelection()``.

        tdse : solprop.nonlinear.tdse.TDSEGasResponse or None, optional
            The complete electronic response of a capillary fill
            (:meth:`from_capillary`).
        tdse_execution : solprop.nonlinear.tdse.TDSEExecution or None, optional
            Its atomic runtime resources.

        Returns
        -------
        Simulation
            The simulation, ready to translate and run.

        Raises
        ------
        DriverError
            If a keyword is missing, inconsistent, or names something this path does
            not carry.
        """
        # The TDSE response owns the electronic response: build no other.
        atomic, plasma, raman, backend, n_save, step_planes = _entry_overrides(
            _tdse_entry(
                tdse,
                tdse_execution,
                trange=trange,
                kerr=kerr,
                mode_average=mode_average,
                unset={
                    "plasma": plasma,
                    "raman": raman,
                    "preionisation": preionisation,
                    "plasma_integration": plasma_integration,
                    "ionisation_field_range": ionisation_field_range,
                    "thg": thg,
                },
                backend=backend,
                step_planes=step_planes,
                save_planes=save_planes,
                n_save=n_save,
            ),
            plasma,
            raman,
            backend,
            n_save,
            step_planes,
        )
        if raman and not kerr:
            msg = (
                "raman=True needs kerr=True: a medium with a Raman response has an "
                "electronic one too, and on a field grid the drive form reaches the "
                "kernel through the Kerr term (docs/guide/media/raman.md)"
            )
            raise DriverError(msg)
        if polarisation is not None:
            msg = (
                f"polarisation={polarisation!r}: this path is scalar and propagates "
                "one transverse component per mode row. Declare the polarisation "
                "in the modes instead -- a rectangular or slab guide's own "
                "polarisation, each partner a row of its own -- or, for a vector "
                "capillary run, a polarisation-resolved mode set, which is not "
                "validated end to end yet"
            )
            raise DriverError(msg)
        _check_wall_declaration(
            wall,
            backend=backend,
            model=model,
            loss=loss,
            lattice=lattice,
            taper_coupling=taper_coupling,
        )
        thg, plasma = (
            (not envelope) if thg is None else thg,
            (not envelope) if plasma is None else plasma,
        )
        active_plasma = (
            any(value is not False for value in plasma.values())
            if isinstance(plasma, Mapping)
            else plasma is not False
        )
        _refuse_plasma_settings(
            plasma_integration, envelope=envelope, active=active_plasma, thg=thg
        )

        working = _capillary_grid(
            grid,
            length,
            wavelength,
            wavelength_limits,
            trange,
            envelope=envelope,
            thg=thg,
        )
        materials = MaterialSelection() if materials is None else materials
        _check_selection_carrier(materials, working.reference_wavelength, wavelength)
        bore = (
            _flow_radius(modes, pressure)
            if waveguide
            else as_profile(cast("float | ZProfile", radius))
        )
        mixture = capillary_fill(
            gas,
            pressure,
            length=length,
            temperature=temperature,
            floor=pressure_floor,
            susceptibility=susceptibility,
            preionisation=preionisation,
            radius=bore,
            vacuum_end=vacuum_end,
            selection=materials,
        )
        _warn_boreless_gradient(mixture, bore)
        if waveguide:
            if isinstance(modes, int):
                raise DriverError("from_waveguide requires supplied gas-filled modes")
            mode_set = _given_gas_modes(modes, mixture)
        else:
            mode_set = _capillary_modes(
                modes,
                cast("float | ZProfile", radius),
                mixture,
                length=length,
                lattice=lattice,
                model=model,
                loss=loss,
                cladding=_default_cladding(cladding)
                if isinstance(modes, int)
                else cladding,
            )
        rule = (
            _no_quadrature(mode_set, n_r, n_theta, radial_rule=radial_rule)
            if mode_average
            else _capillary_quadrature(mode_set, n_r, n_theta, radial_rule=radial_rule)
        )
        profiles = mixture_profiles(mixture)
        if waveguide:
            for mode in mode_set:
                paths: Mapping[str, ZProfile] = getattr(mode, "parameter_paths", {})
                profiles.update(
                    {
                        name: path
                        for name, path in paths.items()
                        if name not in profiles
                        and (not path.is_constant or name == "core_radius")
                    }
                )
        else:
            profiles["core_radius"] = as_profile(cast("float | ZProfile", radius))
        if raman is None:
            # The default follows the Kerr switch: a run without the electronic
            # response is a deliberately reduced model, and a field-grid Raman term
            # cannot stand without a Kerr term beside it anyway.
            raman = kerr and _default_raman(
                mixture, provisional=materials.raman == "provisional"
            )
        if atomic is not None:
            responses = [atomic.response]
        elif (
            isinstance(gas, (Gas, str))
            and not is_air(gas)
            and isinstance(raman, bool)
            and not isinstance(plasma, Mapping)
            and not isinstance(preionisation, Mapping)
        ):
            # The Raman response is optmat's, built from the fill (SEP 3); the other
            # responses keep this path.
            responses = _capillary_responses(
                gas,
                mixture.components[0].species,
                kerr=kerr,
                plasma=plasma,
                preionisation=0.0 if preionisation is None else preionisation,
                integration=plasma_integration,
                wavelength=wavelength,
                envelope=envelope,
                # The single gas's electronic response is optmat's, at the carrier
                # or dispersive, as the selection's electronic policy says (SEP 6).
                electronic=electronic_response(
                    mixture.components[0].species,
                    selection=materials,
                    wavelength=wavelength,
                    thg=thg,
                )
                if kerr
                else None,
                selection=materials,
            )
            responses += (
                _optmat_raman(gas, mixture, materials, thg, wavelength, length)
                if raman
                else []
            )
        else:
            responses = list(
                responses_for(
                    mixture,
                    kerr=kerr,
                    thg=thg,
                    plasma=plasma,
                    raman=raman,
                    wavelength=wavelength,
                    envelope=envelope,
                    length=length,
                    selection=materials,
                    plasma_integration=plasma_integration,
                )
            )
        _refuse_idle_preionisation(mixture, responses)
        pulse = _shaped_pulse(
            shape,
            wavelength=wavelength,
            fwhm=fwhm,
            width=width,
            energy=energy,
            peak_power=peak_power,
            phases=phases,
            pulses=pulses,
        )
        field0 = _in_lowest_mode(pulse.build(working), len(mode_set))
        ionising = any(
            isinstance(one, PlasmaResponse | EnvelopePlasmaResponse)
            for one in responses
        )
        boundary, scheme = _boundary(absorber, absorber_scheme, working)
        # A hollow guide's default is the grid's collar, so there is always one.
        _warn_capillary_walkoff(mode_set.modes, working, cast("Absorber", boundary))
        responses, field_range, reference = _ionisation_inputs(
            responses,
            ionisation_field_range,
            _InputPeak(field0, mode_set, working, rule) if ionising else None,
        )
        settings = SimulationSettings(
            controller=PIController(rtol=rtol, atol=atol),
            save_planes=_save_planes(length, save_planes, n_save),
            # Relative defaults, so that the first step and the cap scale with the
            # capillary.
            dz_max=length / 2 if dz_max is None else dz_max,
            dz_init=length * 1e-4 if dz_init is None else dz_init,
            auto_init=dz_init is None,
            schedule=schedule,
            integrator=integrator,
            _route_integrator=route_integrator,
            interpolate_planes=interpolate_planes,
            step_planes=step_planes,
            density=None,
            absorber=boundary,
            absorber_scheme=scheme,
            boundary_check=boundary_check,
            quadrature=rule,
            components=None
            if rule is None
            else _hollow_components(supplied=not isinstance(modes, int)),
            ionisation_field_range=field_range,
            node_budget=node_budget,
            stats=(
                _with_integration_statistics(
                    _hollow_stats(
                        with_plasma=ionising,
                        mixture=mixture,
                        profiles=profiles,
                        rule=rule,
                    )
                    if waveguide
                    else _capillary_stats(
                        with_plasma=ionising, mixture=mixture, averaged=rule is None
                    ),
                    responses,
                )
                if stats is None
                else stats
            )
            if atomic is None or stats is not None
            else _tdse_stats(
                _capillary_stats(
                    with_plasma=False, mixture=mixture, averaged=rule is None
                )
            ),
            backend=backend,
            precision=precision,
            tdse_execution=None if atomic is None else atomic.execution,
        )
        noise_model, resolution = _resolve_noise(
            noise,
            seed,
            precision,
            tdse=atomic is not None,
            unmeasured=(
                SLAB_NOISE
                if any(isinstance(mode, SlabMode) for mode in mode_set.modes)
                else None
            ),
        )
        draw = functools.partial(
            _shot_noise, noise_model, working, n_modes=len(mode_set)
        )
        simulation = cls(
            working,
            mode_set.modes,
            responses,
            field0,
            settings=settings,
            noise=draw(seed),
            noise_resolution=resolution,
            profiles=profiles,
            wall=wall,
            taper_coupling=taper_coupling,
            materials=materials,
            configuration={
                "medium": mixture,
                **(
                    {
                        "waveguide": {
                            "modes": [ModeRecord.from_mode(mode) for mode in mode_set],
                            "length": length,
                            "temperature": mixture.temperature,
                        }
                    }
                    if waveguide
                    else {
                        "capillary": {
                            "radius": radius,
                            "length": length,
                            "temperature": mixture.temperature,
                        }
                    }
                ),
                "pulse": pulse,
                **({} if atomic is None else {"tdse_entry": atomic.record}),
            },
        )
        # A TDSE run keeps no selection: a mixture update must not rebuild the
        # rate-based responses in its place.
        simulation._mixture_selection = (
            None
            if atomic is not None
            else ResponseSelection(
                kerr,
                thg,
                plasma,
                raman,
                wavelength,
                envelope,
                length,
                selection=materials,
                reference_field=reference,
                field_range=field_range,
                plasma_integration=plasma_integration,
            )
        )
        return _redrawable(simulation, draw, seed)


class SimulationRunner(Displayed):
    """One translated problem and one backend, run for many inputs (SEP 22 section 5.2).

    Built by :meth:`Simulation.runner`: the problem is translated (or the cached
    translation reused), memory is admitted and one backend is constructed, once. Each
    :meth:`run` then derives the input with :meth:`Simulation.with_input`, loads it
    into the same backend and runs the save-plane loop, returning that input's own
    handler. A large problem so pays its backend's allocation and transform planning
    once rather than for every input.

    A run on the reused backend repeats a fresh backend's run, to the rounding that
    separates any two runs: loading an input restores everything the march reads
    (:meth:`solprop.backends.driver.StepDriver.set_initial`). Each run's record is
    its own input's, so each has its own configuration hash.

    Use it as a context manager, or call :meth:`close`: the backend holds about
    fifteen copies of the state until then. A closed runner refuses :meth:`run`. A
    run that raises leaves the runner open, since the next input resets everything;
    a backend that closed itself closes the runner.

    Parameters
    ----------
    simulation : Simulation
        The run to repeat. Its settings, and the translation they fix, are used as
        they are; only the input changes from run to run.
    memory_budget : int or str or None, optional
        The budget the backend is admitted against
        (:func:`solprop.driver.memory.admit_memory`); each run's in-memory output is
        admitted against it again when the run starts.

    Raises
    ------
    DriverError
        If the simulation has a TDSE response, whose atomic binding has its own
        lifecycle and is never reused.
    MemoryBudgetError
        If the backend does not fit its budget; nothing has been allocated.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.pulses import SechPulse
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1.2e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech", n_save=3, rtol=1e-6,
    ... )
    >>> def pulse(power):
    ...     return SechPulse(wavelength=835e-9, width=100e-15, peak_power=power)
    >>> with simulation.runner() as runner:
    ...     outputs = [
    ...         runner.run(pulse(power), configuration={"peak_power": power},
    ...                    progress="none")
    ...         for power in (5.0, 9.8)
    ...     ]
    >>> [output.n_saved for output in outputs], runner.closed
    ([3, 3], True)
    """

    def __init__(
        self, simulation: Simulation, *, memory_budget: int | str | None = None
    ) -> None:
        """Translate, admit and construct the backend, once."""
        problem = simulation.translate()
        if problem.tdse is not None:
            msg = (
                "a TDSE response binds an atomic runtime with its own lifecycle, which "
                "a runner cannot reuse across inputs; run each input with "
                "simulation.with_input(...).run()"
            )
            raise DriverError(msg)
        settings = simulation.settings
        self._simulation = simulation
        self._problem = problem
        self._memory_budget = memory_budget
        self._lock = threading.Lock()
        self._closed = False
        self.runs = 0
        with _optmat.material_context(simulation._material_log):
            _warn_sampled_peak(settings.quadrature, problem.stats_spec)
            # Admitted before the backend allocates, so a refused runner holds nothing.
            admit_memory(problem, settings.backend, memory_budget)
            self._backend = select(
                settings.backend,
                problem,
                grid=simulation.grid,
                modes=simulation.modes,
                profiles=simulation.profiles,
            )

    def __repr__(self) -> str:
        """Return a one-line description: the simulation, the runs and the state.

        Returns
        -------
        str
            For example ``SimulationRunner(Simulation(...), 3 runs, open)``.
        """
        return line(
            "SimulationRunner",
            [
                ("", repr(self._simulation)),
                ("", f"{self.runs} runs"),
                ("", "closed" if self._closed else "open"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "SimulationRunner", [
            ("simulation", repr(self._simulation)),
            ("backend", self._simulation.settings.backend),
            ("runs", self.runs),
            ("state", "closed" if self._closed else "open"),
        ]

    def __enter__(self) -> "SimulationRunner":  # noqa: UP037 - bound after the class body
        """Return the runner, so that a ``with`` block closes it on the way out.

        Returns
        -------
        SimulationRunner
            This runner.
        """
        return self

    def __exit__(self, *details: object) -> None:
        """Close the runner when the block ends, whatever happened inside it.

        Parameters
        ----------
        *details : object
            The exception being raised, if any; it is not suppressed.
        """
        self.close()

    @property
    def closed(self) -> bool:
        """Return whether the runner, and its backend, are closed.

        Returns
        -------
        bool
            True once :meth:`close` has run.
        """
        return self._closed

    @property
    def backend(self) -> Backend:
        """Return the backend every run is stepped on.

        Returns
        -------
        solprop.backends.protocol.Backend
            The one backend the runner constructed.
        """
        return self._backend

    def close(self) -> None:
        """Close the backend, releasing its buffers; calling it twice does nothing.

        A run in progress on another thread finishes first: the close waits for it.
        """
        with self._lock:
            self._close_held()

    def _close_held(self) -> None:
        """Close the backend, the runner's lock being held by the caller."""
        if self._closed:
            return
        self._closed = True
        self._backend.close()

    def run(
        self,
        initial: Pulse | SpatialPulse | np.ndarray,
        *,
        output: Output | Path | str | None = None,
        configuration: Mapping[str, object] | None = None,
        noise: NoiseRealisation | Literal["keep"] | None = "keep",
        progress: ProgressReporter | ProgressChoice | None = None,
        progress_period: float = DEFAULT_PROGRESS_PERIOD,
    ) -> Output:
        """Run one input on the runner's backend and return its handler.

        Parameters
        ----------
        initial : Pulse, SpatialPulse or numpy.ndarray
            The input, as :meth:`Simulation.with_input` takes it.
        output : solprop.output.schema.Output or pathlib.Path or str or None, optional
            Where this input's planes go: a handler, a path for an HDF5 file, or
            ``None`` for a new :class:`~solprop.output.memory.MemoryOutput`.
        configuration : collections.abc.Mapping or None, optional
            Caller keys recorded in this input's run record and hash, such as the
            delay of a scan (:meth:`Simulation.with_input`).
        noise : NoiseRealisation, {'keep'} or None, optional
            The noise, as for :meth:`Simulation.with_input`: kept by default, or
            replaced by another realisation of the same model.
        progress : ProgressReporter, {'auto', 'bar', 'log', 'none'} or None, optional
            Where progress goes (:func:`solprop.driver.run.run_problem`).
        progress_period : float, optional
            Seconds between the line reporter's periodic lines, s.

        Returns
        -------
        solprop.output.schema.Output
            This input's handler, closed; a memory handler stays readable.

        Raises
        ------
        DriverError
            If the runner is closed or already running an input; if the derived
            problem is not the runner's but for its input (a different launch
            position, or a shared leaf replaced); or if the noise model differs from
            the runner's, which the backend could not carry.
        MemoryBudgetError
            If this input's in-memory output does not fit beside the backend.
        """
        if not self._lock.acquire(blocking=False):
            msg = (
                "this runner is already running an input: one backend steps one input "
                "at a time, so build one runner per thread or process"
            )
            raise DriverError(msg)
        try:
            if self._closed:
                msg = "this runner is closed; build another with simulation.runner()"
                raise DriverError(msg)
            return self._run(
                initial,
                output=output,
                configuration=configuration,
                noise=noise,
                progress=progress,
                progress_period=progress_period,
            )
        except BaseException:
            # A backend that closed itself cannot run another input.
            if getattr(self._backend, "_closed", False):
                self._close_held()
            raise
        finally:
            self._lock.release()

    def _run(
        self,
        initial: Pulse | SpatialPulse | np.ndarray,
        *,
        output: Output | Path | str | None,
        configuration: Mapping[str, object] | None,
        noise: NoiseRealisation | Literal["keep"] | None,
        progress: ProgressReporter | ProgressChoice | None,
        progress_period: float,
    ) -> Output:
        """Derive, load and run one input; :meth:`run` without its guards.

        Parameters
        ----------
        initial : Pulse, SpatialPulse or numpy.ndarray
            The input.
        output : solprop.output.schema.Output or pathlib.Path or str or None
            Where its planes go.
        configuration : collections.abc.Mapping or None
            Caller keys for its record.
        noise : NoiseRealisation, {'keep'} or None
            Its noise.
        progress : ProgressReporter, {'auto', 'bar', 'log', 'none'} or None
            Where progress goes.
        progress_period : float
            Seconds between periodic lines, s.

        Returns
        -------
        solprop.output.schema.Output
            The handler.
        """
        derived = self._simulation.with_input(
            initial, noise=noise, configuration=configuration
        )
        problem = derived.translate()
        check_derived_problem(self._problem, problem)
        settings = self._simulation.settings
        with _optmat.material_context(derived._material_log):
            metadata = derived.metadata()
            handler, _, _ = _open_output(output, metadata, resume=False)
            retention = field_retention(problem, settings.fields, settings.field_planes)
            admission = admit_memory(
                problem,
                settings.backend,
                self._memory_budget,
                output=handler,
                retention=retention,
            )
            self._backend.set_initial(
                problem.initial.field0,
                None if isinstance(noise, str) else _realisation(problem),
                z=problem.initial.z,
            )
            run_problem(
                problem,
                self._backend,
                handler,
                metadata=metadata,
                stats_spec=problem.stats_spec,
                progress=progress,
                progress_period=progress_period,
                memory=admission,
                resume_history=settings.resume_history,
                fields=retention.fields,
                field_planes=retention.field_planes,
                boundary_monitor=derived._boundary_monitor(),
            )
            derived._peak_fields = _peak_field_reports(
                problem, handler, derived.responses
            )
        if derived._peak_fields and isinstance(handler, PlaneStore):
            handler.annotate(
                PEAK_FIELDS_ANNOTATION,
                [
                    cast("JsonValue", report.to_dict())
                    for report in derived._peak_fields
                ],
            )
        self.runs += 1
        return handler


def _realisation(problem: KernelProblem) -> NoiseRealisation | None:
    """Return a derived problem's noise as the backend loads it, or ``None``.

    Parameters
    ----------
    problem : KernelProblem
        The derived problem, whose ``initial.noise0`` is already in the backend's
        amplitude units.

    Returns
    -------
    solprop.noise.NoiseRealisation or None
        Its realisation, or ``None`` without noise.
    """
    initial = problem.initial
    if initial.noise0 is None or initial.noise_model == "none":
        return None
    return NoiseRealisation(
        spectrum=initial.noise0[0], seed=None, model=initial.noise_model
    )


_SHARED_LEAVES = (
    "grids",
    "linear",
    "transform",
    "responses",
    "stats_spec",
    "tdse",
    "current_numerator",
)
"""The leaves :meth:`Simulation.with_input` shares with the translated problem."""


def check_derived_problem(constructed: KernelProblem, derived: KernelProblem) -> None:
    """Refuse a derived problem that is not the constructed one but for its input.

    What a :class:`SimulationRunner` checks before loading an input into the backend
    it built for `constructed` (SEP 22 section 5.2): the leaves
    :meth:`Simulation.with_input` shares are the same objects, the stepper and the
    flags, which it rebuilds, agree field by field (the save planes elementwise), and
    the launch position and the noise model are the backend's.

    Parameters
    ----------
    constructed : solprop.translate.problem.KernelProblem
        The problem the backend was built for.
    derived : solprop.translate.problem.KernelProblem
        The problem of one input.

    Raises
    ------
    DriverError
        Naming the first leaf or setting that differs.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.driver.simulation import check_derived_problem
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech",
    ... )
    >>> problem = simulation.translate()
    >>> derived = simulation.with_input(problem.initial.field0[0]).translate()
    >>> check_derived_problem(problem, derived) is None
    True
    """
    for name in _SHARED_LEAVES:
        if getattr(derived, name) is not getattr(constructed, name):
            msg = (
                f"the input's problem does not share the runner's {name}: a runner "
                "steps one translated problem and changes only its input"
            )
            raise DriverError(msg)
    if derived.initial.z != constructed.initial.z:
        msg = (
            f"the input is launched at z = {derived.initial.z!r} m and the runner's "
            f"problem at {constructed.initial.z!r} m: the save planes and the "
            "schedule's cursor belong to the problem the backend was built for, so a "
            "different initial_z needs its own simulation"
        )
        raise DriverError(msg)
    if derived.initial.noise_model != constructed.initial.noise_model:
        msg = (
            f"the input carries the noise model {derived.initial.noise_model!r} and "
            f"the runner's problem {constructed.initial.noise_model!r}: the backend "
            "carries the model it was built with, and would drop or misapply another"
        )
        raise DriverError(msg)
    for record, name in ((derived.stepper, "stepper"), (derived.flags, "flags")):
        built = getattr(constructed, name)
        for entry in dataclasses.fields(record):
            if not _same_setting(
                getattr(built, entry.name), getattr(record, entry.name)
            ):
                msg = (
                    f"the input's {name}.{entry.name} differs from the runner's: a "
                    "runner steps one translated problem and changes only its input"
                )
                raise DriverError(msg)


def _simulation_mixture(simulation: Simulation) -> Mixture:
    """Require a material declaration the update can identify by component.

    Parameters
    ----------
    simulation : Simulation
        Existing modal or spatial propagation.

    Returns
    -------
    Mixture
        The material shared by its modes, or carried by its spatial medium.

    Raises
    ------
    DriverError
        If no mixture declaration is present or modes carry different mixtures.
    """
    if simulation.medium is not None and isinstance(
        simulation.medium.material, Mixture
    ):
        return simulation.medium.material
    materials = [getattr(mode, "mixture", None) for mode in simulation.modes]
    if not materials or not isinstance(materials[0], Mixture):
        raise DriverError("with_mixture needs a Mixture in the modes or spatial medium")
    material = materials[0]
    for index, other in enumerate(materials[1:], 1):
        if serialise(other) != serialise(material):
            raise DriverError(f"mode {index} carries a different mixture")
    return material


def _refuse_plasma_settings(
    integration: IntegrationPolicy | None, *, envelope: bool, active: bool, thg: bool
) -> None:
    """Refuse plasma settings that cannot act together.

    Envelope plasma cannot carry the third harmonic, and a running-integral rule
    belongs to the field-grid plasma column: given beside envelope plasma, or with
    the plasma off, nothing would read it.

    Parameters
    ----------
    integration : {'band', 'trapezoid'} or None
        The ``plasma_integration`` keyword.
    envelope : bool
        Whether the plasma is the envelope form.
    active : bool
        Whether any plasma response is selected.
    thg : bool
        Whether the third harmonic is kept.

    Raises
    ------
    DriverError
        If envelope plasma meets the harmonic, or a rule is given beside envelope
        plasma or with the plasma off.
    """
    if envelope and active and thg:
        msg = "envelope plasma cannot be combined with THG"
        raise DriverError(msg)
    if integration is None:
        return
    if envelope:
        msg = (
            f"plasma_integration={integration!r} selects the field-grid plasma "
            "column's running-integral rule; envelope plasma converts its currents "
            "with exact inverse-frequency factors and takes no such rule"
        )
        raise DriverError(msg)
    if not active:
        msg = (
            f"plasma_integration={integration!r} is given but the plasma is off; "
            "turn it on (plasma=True) or drop plasma_integration"
        )
        raise DriverError(msg)


def _refuse_idle_preionisation(mixture: Mixture, responses: Sequence[object]) -> None:
    """Refuse a declared initial ionised fraction that no plasma channel carries.

    A component's preionisation reaches the run only through its plasma or
    envelope-plasma response. Without one the declaration would be recorded in the
    run's metadata and ignored by the propagation, so it is refused by name.

    Parameters
    ----------
    mixture : Mixture
        The ordered populations with their declared fractions.
    responses : sequence of object
        The responses the run will compile.

    Raises
    ------
    DriverError
        Naming the first component whose preionisation is a path or a nonzero number
        and which has no plasma response.
    """
    carried = {
        _population_label(one, mixture)
        for one in responses
        if isinstance(one, PlasmaResponse | EnvelopePlasmaResponse)
    }
    for component in mixture.components:
        value = component.preionisation
        declared = isinstance(value, ZProfile) or value != 0
        if declared and component.label not in carried:
            msg = (
                f"preionisation is declared for {component.label!r}, which has no "
                "plasma response, so no channel would carry it; turn its plasma on "
                "(plasma=True or plasma={label: True}) or drop the preionisation"
            )
            raise DriverError(msg)


def _population_label(
    response: PlasmaResponse | EnvelopePlasmaResponse, mixture: Mixture
) -> str:
    """Resolve an existing plasma declaration to its already-validated population.

    Parameters
    ----------
    response : PlasmaResponse or EnvelopePlasmaResponse
        Explicit label or single-species rate declaration.
    mixture : Mixture
        Ordered component labels and species.

    Returns
    -------
    str
        The unique matching label.
    """
    if response.species is not None:
        return response.species
    if len(mixture.components) == 1:
        return mixture.labels[0]
    name = getattr(response.gas, "name", str(response.gas))
    return next(
        component.label
        for component in mixture.components
        if component.species.name == name
    )


def _same_operator_function(
    source: Simulation,
    updated: Simulation,
    old: Mixture,
    new: Mixture,
    expansion: ParameterExpansion | None,
) -> ParameterExpansion | None:
    """Keep terms only when every dependency outside their axes is unchanged.

    Parameters
    ----------
    source, updated : Simulation
        Geometry-preserving simulations on the same grid and frame.
    old, new : Mixture
        Species-equivalent material declarations.
    expansion : ParameterExpansion or None
        Previously sampled coefficients and axis names.

    Returns
    -------
    ParameterExpansion or None
        The old coefficients when the function is identical, else None with a log.
    """
    if source.settings.prefactor_index != updated.settings.prefactor_index:
        logger.info("Mixture update: expansion reuse refused; prefactor_index changed")
        return None
    if expansion is None:
        return None
    if len(old.components) > 1 and (
        (old.shared is None) != (new.shared is None)
        or (
            old.shared is not None
            and (old.fractions != new.fractions or old.temperature != new.temperature)
        )
    ):
        logger.info(
            "Mixture update: rebuild expansion; fill fractions, EOS temperature "
            "or parameterisation changed"
        )
        return None
    positions = np.asarray(expansion.z_span)
    if source.medium is not None and updated.medium is not None:
        paths = [(source.medium.parameter_paths, updated.medium.parameter_paths)]
    else:
        paths = [
            (before.parameter_paths, after.parameter_paths)
            for before, after in zip(source.modes, updated.modes, strict=True)
            if isinstance(before, ParameterisedMode)
            and isinstance(after, ParameterisedMode)
        ]
    for before_paths, after_paths in paths:
        for name, path in before_paths.items():
            if name in expansion.parameters:
                continue
            other = after_paths.get(name)
            if (
                other is None
                or not other.is_constant
                or not np.array_equal(path(positions), other(positions))
            ):
                logger.info(
                    "Mixture update: rebuild expansion; fixed parameter %s changed",
                    name,
                )
                return None
    return expansion


def _mixture_reuse(
    source: Simulation,
    updated: Simulation,
    old: Mixture,
    new: Mixture,
    problem: KernelProblem,
) -> TranslationReuse:
    """Collect eligible immutable translation data after a material-path update.

    Parameters
    ----------
    source, updated : Simulation
        Geometry-preserving simulations with identical grids and noise.
    old, new : Mixture
        Species-equivalent material declarations.
    problem : KernelProblem
        Source translation whose immutable arrays may be shared.

    Returns
    -------
    TranslationReuse
        Transverse transform, eligible expansion terms and unchanged Raman samples.
    """
    before = {
        response.species or old.labels[0]: response
        for response in source.responses
        if isinstance(response, RamanResponse)
    }
    after = {
        response.species or new.labels[0]: response
        for response in updated.responses
        if isinstance(response, RamanResponse)
    }
    kernels = {}
    for leaf in problem.responses.raman_terms:
        if _undamped_model_key(before[leaf.label]) == _undamped_model_key(
            after[leaf.label]
        ):
            kernels[leaf.label] = leaf.undamped
        else:
            logger.info("Mixture update: rebuild %s undamped Raman model", leaf.label)
    return TranslationReuse(
        problem.transform,
        problem.initial,
        _same_operator_function(
            source, updated, old, new, problem.linear.phi_expansion
        ),
        _same_operator_function(
            source, updated, old, new, problem.responses.prefactor_expansion
        ),
        kernels,
        problem.transverse,
    )


def _undamped_model_key(response: RamanResponse) -> object:
    """Describe only the inputs to the undamped Raman samples.

    Parameters
    ----------
    response : RamanResponse
        The component response whose kind and label the update preserves.

    Returns
    -------
    object
        Serialised model data, excluding the separate dephasing field of the
        built-in oscillator and rotational-line models. Other models keep all data.
    """
    models = []
    for term in response.tensor_terms:
        model = term.model
        record = serialise(model)
        if isinstance(model, (DampedOscillator, RotationalLines)):
            assert isinstance(record, dict)
            record.pop("dephasing", None)
        models.append((term.contraction, record))
    return models


_CF4_BACKEND: Final = "numpy_ref"
"""The one backend that executes a ``"cf4"`` section of the flow.

CF4 is not yet implemented on the compiled backends, which refuse such a section.
"""


def _check_wall_declaration(
    wall: CoherentWall | None,
    *,
    backend: str,
    model: MarcatiliModel,
    loss: LossSpec,
    lattice: TubeLattice | None,
    taper_coupling: bool = True,
) -> None:
    """Refuse, at the keyword the caller typed, a wall this run cannot carry.

    The highest of the coherent wall's doors, and the only one that can name the
    argument rather than the object it became: the backend selection, the loss
    setting the block would replace, the capillary model it is not derived for and
    the structured cladding it does not describe. Each is refused again further
    down -- in :func:`solprop.modes.capillary.wall_operator` and at the backend --
    so nothing depends on this one being reached.

    Parameters
    ----------
    wall : solprop.modes.capillary.CoherentWall or None
        What was declared. ``None`` leaves every other argument alone.
    backend : str
        The ``backend=`` keyword, before :func:`solprop.backends.registry.resolve_name`
        turns ``"auto"`` into a concrete choice.
    model : {'full', 'reduced'}
        The ``model=`` keyword.
    loss : bool, float or callable
        The ``loss=`` keyword.
    lattice : solprop.modes.antiresonant.TubeLattice or None
        The ``lattice=`` keyword.
    taper_coupling : bool, optional
        The ``taper_coupling=`` keyword, which a wall cannot decline.

    Raises
    ------
    DriverError
        Naming the keyword to change and why the block cannot carry its value.
    """
    if wall is None:
        return
    if not taper_coupling:
        msg = (
            "taper_coupling=False declines the taper connection, and a coherent wall "
            "block always carries it: its basis is attached to the local radius. "
            "Leave taper_coupling at its default with wall=, or pass wall=None for "
            "the adiabatic route"
        )
        raise DriverError(msg)
    resolved = resolve_name(backend)
    selected = f"{backend!r}" if backend == resolved else f"{backend!r} ({resolved})"
    if wall.method == "cf4" and resolved != _CF4_BACKEND:
        msg = (
            f"a coherent wall with method='cf4' integrates each section by the "
            f"commutator-free exponentials of its whole generator, which the "
            f"{_CF4_BACKEND} backend alone executes, and this run selects "
            f"{selected}: numba_cpu and the native backend execute the default "
            f"method='frozen_residual' and refuse a CF4 section by name: CF4 is not "
            "yet implemented on the compiled backends. Pass "
            f"backend={_CF4_BACKEND!r} with method='cf4', or leave the method at its "
            "default"
        )
        raise DriverError(msg)
    if loss is not True:
        msg = (
            f"a coherent wall block is the loss model, and this run declares "
            f"loss={loss!r}: the condensed Robin boundary condition replaces the "
            "Marcatili attenuation of every channel, so a LossSpec beside it would "
            "be discarded in silence. Pass loss=True with wall=, or leave wall=None "
            "and keep the uncoupled loss you asked for"
        )
        raise DriverError(msg)
    if model != "full":
        msg = (
            f"a coherent wall block needs one capillary model, the full one, and "
            f"this run declares model={model!r}: the condensation inverts the full "
            "model's transverse eigenvalue, and the reduced model's leading-order "
            "carrier beside it would be overridden rather than propagated"
        )
        raise DriverError(msg)
    if lattice is not None:
        msg = (
            "a coherent wall block assumes one uniform dielectric wall at a single "
            "radius, and this run declares a tube lattice: an antiresonant guide's "
            "loss is its tubes' resonances, not a Robin condition on a bare "
            "capillary. Leave wall=None and keep the Vincetti loss, or drop the "
            "lattice"
        )
        raise DriverError(msg)


# Positions along the capillary at which the modes' group delays are read.
_WALKOFF_SAMPLES: Final = 33


def _warn_adiabatic_taper(modes: Sequence[Mode]) -> None:
    """Warn, once per translation, that a moving multimode set omits its connection.

    A mode set whose transverse shape changes along z -- a capillary or a fibre on a
    taper -- couples its channels through the moving basis, ``<psi_m|d_z psi_n>``, and
    the diagonal route propagates each channel adiabatically with that term absent. A
    single channel needs no connection, and a guide whose shape does not move -- a
    constant bore, or a pressure gradient in a capillary, whose modes depend on the
    radius alone -- has none. The coherent wall route carries the connection and never
    reaches here, and neither does a run that declined it (``taper_coupling=False``); a
    capillary's contiguous ``HE_1m`` set carries it by default
    (:func:`solprop.modes.capillary.taper_coupling_block`), so what warns is a moving
    set the coupling is not derived for.

    Parameters
    ----------
    modes : sequence of Mode
        The run's mode set.

    Warns
    -----
    AdiabaticTaperWarning
        When more than one channel is propagated and any channel's shape moves.
    """
    channels = tuple(modes)
    if len(channels) < 2:
        return
    moving = [
        index
        for index, mode in enumerate(channels)
        if not getattr(mode, "radius_is_constant", mode.z_constant)
    ]
    if not moving or taper_coupling_block(channels) is not None:
        return
    warnings.warn(
        f"{len(channels)} channels propagate on a guide whose modes change shape "
        f"along z (channel {moving[0]} first), and this route omits the connection "
        "<psi_m|d_z psi_n> between them: the inter-mode conversion a taper drives is "
        "absent and each channel follows its local mode adiabatically. A "
        "capillary's contiguous HE_11 to HE_1N set carries it by default, with or "
        "without wall=CoherentWall(); this set is not that block "
        "(docs/guide/waveguides/taper-coupling.md)",
        AdiabaticTaperWarning,
        stacklevel=3,
    )


_PACKAGE_FILES: Final = (str(Path(__file__).resolve().parents[1]),)
"""Frames a fill warning skips, so that it names the caller's line, not solprop's."""


def _flow_radius(
    modes: int | Sequence[Mode], pressure: Pressure | None
) -> ZProfile | None:
    """Return the core radius a supplied mode set's gas flows through, if circular.

    Every mode that declares ``core_radius`` -- every ``MarcatiliMode`` subclass: a
    capillary, a Vincetti mode, a tabulated hollow mode and a ``TabulatedMode`` built
    from arrays -- is taken as a circular bore of that declared radius, which sets the
    Poiseuille conductance and the choke (for a tube lattice the gaps between the tubes
    are ignored). A rectangular guide's duct resistance and a slab's flow between plates
    are not modelled: they keep the uniform law, and warn if their widths or gap vary
    under a pressure gradient.

    Parameters
    ----------
    modes : int or sequence of Mode
        The supplied set.
    pressure : float, sequence, ZProfile or None
        The declared pressure, to tell a gradient from a constant fill.

    Returns
    -------
    ZProfile or None
        The first mode's core radius in m, or ``None`` for a guide without one.

    Warns
    -----
    GasFlowWarning
        For a rectangular guide whose widths, or a slab whose gap, vary beside a
        pressure gradient.
    """
    if isinstance(modes, int) or not modes:
        return None
    first = modes[0]
    radius = getattr(first, "core_radius", None)
    if isinstance(radius, ZProfile):
        return radius
    widths = [getattr(first, name, None) for name in ("width_x", "width_y")]
    varying = any(isinstance(w, ZProfile) and not w.is_constant for w in widths)
    if varying and isinstance(pressure, Sequence):
        # A slab declares only width_x, its half-gap: flow between plates goes as
        # a^3 per unit width, which is not modelled either.
        message = (
            "the slab's gap varies along z, and its pressure gradient follows the "
            "uniform law: flow between plates is not modelled, so the fill ignores "
            "the gap"
            if widths[1] is None
            else "the rectangular guide's widths vary along z, and its pressure "
            "gradient follows the uniform law: a rectangular duct's hydraulic "
            "resistance is not modelled, so the fill ignores the widths"
        )
        warnings.warn(
            message,
            GasFlowWarning,
            stacklevel=2,
            skip_file_prefixes=_PACKAGE_FILES,
        )
    return None


def _warn_boreless_gradient(mixture: Mixture, radius: ZProfile | None) -> None:
    """Warn if a gradient built without the bore fills a guide whose radius varies.

    A :class:`~solprop.profiles.PressureGradient` supplied directly, or inside a
    Mixture, was built by its caller; if it does not know the radius, its ``p^2`` is
    linear in ``z`` although a taper's is linear in ``int a^-4 dz``.

    Parameters
    ----------
    mixture : Mixture
        The resolved fill.
    radius : ZProfile or None
        The guide's core radius in m, if it has one.

    Warns
    -----
    GasFlowWarning
        When the shared fill is a varying gradient without a radius and the core
        radius varies.
    """
    shared = mixture.shared
    if (
        radius is None
        or radius.is_constant
        or not isinstance(shared, PressureGradient)
        or shared.is_constant
        or shared.radius is not None
    ):
        return
    warnings.warn(
        "the pressure gradient was built without the bore, and the guide's core "
        "radius varies along z: its p^2 is linear in z, whereas a taper's steady flow "
        "makes p^2 linear in the hydraulic resistance int a^-4 dz, so a narrow waist "
        "carries most of the drop. Build it with PressureGradient.from_stages(..., "
        "radius=) or solprop.gasflow.steady_fill, or pass the end pressures as a pair "
        "and let the constructor build the fill",
        GasFlowWarning,
        stacklevel=2,
        skip_file_prefixes=_PACKAGE_FILES,
    )


def _warn_capillary_walkoff(
    modes: Sequence[Mode], grid: TimeGrid, boundary: Absorber
) -> None:
    """Warn if the mode set's group-delay walk-off would survive the temporal collar.

    A mode set shares one co-moving frame, so the group delay between its modes is
    physics the operator carries: mode ``p`` drifts against the first by the integral of
    ``beta_1,p - beta_1,1`` along the fibre, 43 fs per metre between HE_11 and HE_12 of
    a 125 um capillary. The span of the set, integrated over the length with the modes'
    own paths, is what :func:`solprop.boundaries.warn_walkoff` needs. It is read at the
    reference frequency alone: a band-wide span would be dominated by band edges
    carrying nothing.

    Parameters
    ----------
    modes : sequence of solprop.modes.protocol.Mode
        The mode set, the first mode being the one the frame follows.
    grid : solprop.grid.TimeGrid
        The grid, for its length, its reference frequency and its temporal collar.
    boundary : solprop.boundaries.Absorber
        The collar the run will use, for the length that sets its temporal strength.
        A spectral-only collar has no temporal part to check.
    """
    if len(modes) < 2 or not np.any(boundary.temporal_rate):
        return
    frame = Frame.comoving(grid)
    positions = np.linspace(0.0, grid.length, _WALKOFF_SAMPLES)
    delays = np.array(
        [
            [frame_quantities(mode, frame, float(z))[1] for mode in modes]
            for z in positions
        ]
    )
    spread = np.max(delays, axis=1) - np.min(delays, axis=1)
    span = float(np.trapezoid(spread, positions))
    warn_walkoff(grid, span, boundary_length=boundary.in_stage_length)


def _check_initial_position(
    simulation: Simulation, settings: SimulationSettings
) -> None:
    """Validate an absolute launch before translation can normalize its field.

    Parameters
    ----------
    simulation : Simulation
        Declared optical input, grid and absolute launch position in m.
    settings : SimulationSettings
        Candidate save schedule and stepping choices.
    """
    z = simulation.initial_z
    if (
        not np.isfinite(z)
        or z < 0
        or z >= simulation.grid.length
        or np.any(np.asarray(settings.save_planes) <= z)
    ):
        raise DriverError(
            "initial_z must be finite, nonnegative and before every save plane"
        )
    if z == 0:
        return
    if (
        not isinstance(simulation.initial, np.ndarray)
        or simulation.spatial_grid is not None
    ):
        raise DriverError("an interior launch requires an explicit modal spectrum")
    if (
        simulation.noise is not None and simulation.noise.model == "modified"
    ) or settings.schedule is not None:
        raise DriverError(
            "an interior launch refuses the modified noise model and a fixed schedule; "
            "input noise takes its draw at the launch (SEP 37)"
        )


def _validate_resume(
    problem: KernelProblem, state: ResumeState, execution: TDSEExecution | None
) -> int:
    """Validate restart position and budgets before preparing atomic resources.

    Parameters
    ----------
    problem : KernelProblem
        Translated scientific problem and absolute save positions in m.
    state : ResumeState
        Saved field, accepted position and cumulative accounting.
    execution : TDSEExecution or None
        Current atomic budget, if this is a TDSE run.

    Returns
    -------
    int
        Index of the next scheduled save strictly beyond the accepted position.
    """
    next_plane = int(
        np.searchsorted(problem.stepper.save_planes, state.plane.z, side="right")
    )
    if state.checkpoint is not None and state.checkpoint.next_save_index != next_plane:
        raise DriverError(
            "checkpoint next_save_index disagrees with its absolute position"
        )
    if not problem.initial.z <= state.plane.z <= problem.stepper.save_planes[-1]:
        raise DriverError(
            "resume position lies outside the declared propagation interval"
        )
    if state.plane.field is None:
        if _work_left(problem, state, next_plane):
            msg = (
                f"the run's last saved plane, at z = {state.plane.z:.6e} m, kept no "
                "field (it was saved with fields='none' or 'planes') and the file "
                "holds no terminal checkpoint, which always carries the field, so "
                "there is no state to continue from: a run stopped by an error or a "
                "signal writes one, a run killed outright does not. Rerun it, or keep "
                "the field on its last plane"
            )
            raise DriverError(msg)
    elif not np.all(np.isfinite(state.plane.field)):
        raise DriverError("resume requires a finite stored field")
    outcome = state.outcome
    if problem.tdse is not None and (
        outcome is None or not outcome.logical_calls or not outcome.atomic_calls
    ):
        raise DriverError("a TDSE resume requires persisted cumulative call ledgers")
    method = problem.stepper.integrator
    if outcome is not None and outcome.logical_calls and isinstance(method, Multistep):
        # Under another integrator the charged ledger stays in the earlier execution
        # records as provenance, and is not installed (SEP 42 section 5.6).
        calls = CallLedger.from_record(outcome.logical_calls)
        if method.max_calls is not None and calls.total > method.max_calls:
            raise DriverError("logical call budget is below previously charged work")
    if outcome is not None and outcome.atomic_calls:
        atomic = AtomicCallLedger.from_record(outcome.atomic_calls)
        if problem.tdse is None or execution is None:
            raise DriverError(
                "an atomic call ledger requires an explicit TDSE execution"
            )
        if (
            execution.max_atomic_calls is not None
            and atomic.attempted > execution.max_atomic_calls
        ):
            raise DriverError("atomic call budget is below previously attempted work")
    return next_plane


def _tdse_memory_execution(
    problem: KernelProblem,
    execution: TDSEExecution,
    admission: MemoryAdmission,
    *,
    archive: Literal["full", "required"] | None = None,
) -> TDSEExecution:
    """Apply the run's budget to combined optical and atomic persistent storage.

    Parameters
    ----------
    problem : KernelProblem
        Translated problem supplying the atomic descriptor and node count.
    execution : TDSEExecution
        Atomic device choices and memory cap in bytes.
    admission : MemoryAdmission
        Optical host estimate and declared run budget in bytes.
    archive : {'full', 'required'} or None, optional
        The raw archive's output selection, whose writer buffers and widened
        provider histories are admitted too; None without an archive.

    Returns
    -------
    TDSEExecution
        Resource choices with the combined hard memory limit enforced.
    """
    descriptor = problem.tdse
    if descriptor is None:
        raise DriverError("atomic memory admission requires a TDSE descriptor")
    budget = admission.budget
    limits = [execution.memory_limit]
    if budget.action == "refuse":
        limits.append(budget.limit)
    finite_limits = [value for value in limits if value is not None]
    effective = dataclasses.replace(
        execution, memory_limit=min(finite_limits) if finite_limits else None
    )
    atomic_bytes = sum(
        tdse_runtime.estimate_binding_memory(
            descriptor, effective, problem.transform.n_nodes, archive=archive
        ).values()
    )
    combined = admission.estimate.total + atomic_bytes
    if effective.memory_limit is not None and combined > effective.memory_limit:
        raise MemoryBudgetError(
            f"combined optical and TDSE estimate requires {combined} bytes, "
            f"exceeding {effective.memory_limit} bytes"
        )
    if budget.action == "warn" and budget.limit is not None and combined > budget.limit:
        warnings.warn(
            f"combined optical and TDSE estimate requires {combined} bytes, "
            f"exceeding the advisory run budget of {budget.limit} bytes "
            f"({budget.source})",
            MemoryBudgetWarning,
            stacklevel=3,
        )
    return effective


def _check_tdse_archive(
    archive: object, problem: KernelProblem, handler: Output
) -> None:
    """Refuse an archive request this run cannot honour, before any work.

    Parameters
    ----------
    archive : object
        The request, checked here to be a ``TDSEArchive`` or None.
    problem : KernelProblem
        The translated problem.
    handler : Output
        The optical output, not yet opened.

    Raises
    ------
    DriverError
        If the request is not a :class:`~solprop.output.TDSEArchive`, the run has
        no TDSE response, or the archive's path is the optical file's.
    """
    if archive is None:
        return
    if not isinstance(archive, TDSEArchive):
        raise DriverError("tdse_archive must be a TDSEArchive or None")
    if problem.tdse is None:
        raise DriverError(
            "tdse_archive archives a TDSE response's provider evaluations; this run "
            "has no TDSE response"
        )
    if isinstance(handler, HDF5Output) and _same_file(archive.path, handler.path):
        raise DriverError(
            f"the TDSE archive {archive.path} must differ from the optical HDF5 file"
        )


def _same_file(first: Path, second: Path) -> bool:
    """Return whether two paths name the same file, existing or not.

    Parameters
    ----------
    first, second : pathlib.Path
        The paths.

    Returns
    -------
    bool
        True when they resolve to one location.
    """
    if first.exists() and second.exists():
        return first.samefile(second)
    return first.resolve() == second.resolve()


def _recorded_archive_uuid(state: ResumeState) -> str | None:
    """Return the archive run UUID a resumed optical file records, if any.

    The latest execution's record wins, then the closing records of the earlier
    executions, latest first: an archive first requested by a resume is recorded
    in that execution's outcome, not in the file's original record. A TDSE resume
    always has an outcome (:func:`_validate_resume`).

    Parameters
    ----------
    state : ResumeState
        What the file holds.

    Returns
    -------
    str or None
        The UUID, or None when no execution of the run archived.
    """
    outcome = state.outcome
    # `_validate_resume` has refused a TDSE file without its ledgers and outcome.
    assert outcome is not None
    records: list[object] = [outcome.execution]
    records.extend(
        previous.get("execution") for previous in reversed(outcome.previous_executions)
    )
    for record in records:
        tdse = cast("Mapping[str, object]", record).get("tdse")
        archive = tdse.get("archive") if isinstance(tdse, Mapping) else None
        value = (
            cast("Mapping[str, object]", archive).get("run_uuid")
            if isinstance(archive, Mapping)
            else None
        )
        if isinstance(value, str):
            return value
    return None


def _open_tdse_archive(
    archive: TDSEArchive | None,
    problem: KernelProblem,
    handler: Output,
    state: ResumeState | None,
    metadata: RunMetadata,
    execution: TDSEExecution,
) -> TDSEArchiveWriter | None:
    """Open the run's archive session, continuing its own run's archive on a resume.

    Parameters
    ----------
    archive : TDSEArchive or None
        The request, already checked by :func:`_check_tdse_archive`.
    problem : KernelProblem
        The translated problem, with its TDSE descriptor.
    handler : Output
        The optical output.
    state : ResumeState or None
        What a resumed file holds, or None for a fresh run.
    metadata : RunMetadata
        The run's record.
    execution : TDSEExecution
        The atomic resources.

    Returns
    -------
    TDSEArchiveWriter or None
        The open session, or None without a request.
    """
    descriptor = problem.tdse
    if archive is None or descriptor is None:
        return None
    recorded = None if state is None else _recorded_archive_uuid(state)
    weights = problem.transform.node_weights
    context: dict[str, object] = {
        "descriptor": json.loads(descriptor.configuration_json),
        "descriptor_identity": descriptor.identity,
        "model": descriptor.model,
        "n_nodes": problem.transform.n_nodes,
        "node_weights_m2": None if weights is None else weights.tolist(),
        "quadrature": metadata.quadrature,
        "execution": dataclasses.asdict(execution),
        "provider_pin": tdse_runtime.pinned_revision(),
        # The file's own record on a resume, which the session holds and a snapshot
        # compares with (SEP 42 section 5.3).
        "optical_configuration_hash": (
            metadata if state is None else state.metadata
        ).configuration_hash(),
        "versions": dict(metadata.versions),
    }
    return TDSEArchiveWriter.open(
        archive,
        run_uuid=str(uuid.uuid4()) if recorded is None else recorded,
        kind="run" if state is None else "resume",
        optical=handler.path if isinstance(handler, HDF5Output) else None,
        context=context,
        resume=recorded is not None,
    )


def _close_run(
    backend: Backend | None,
    binding: TDSEBinding | None,
    archive: TDSEArchiveWriter | None,
    handler: Output,
    failure: BaseException | None,
) -> None:
    """Close a run's resources in order, each even when an earlier close fails.

    The backend before the atomic binding, the binding before the archive that
    records it, and the optical output last.

    Parameters
    ----------
    backend : Backend or None
        The optical backend.
    binding : TDSEBinding or None
        The atomic binding.
    archive : TDSEArchiveWriter or None
        The archive session.
    handler : Output
        The optical output.
    failure : BaseException or None
        The error the run is raising, if any.
    """
    closing = failure
    for resource in (backend, binding, archive, handler):
        if resource is None:
            continue
        try:
            resource.close()
        except BaseException as error:
            if closing is None:
                closing = error
            else:
                closing.add_note(f"a run resource could not be closed: {error}")
    if failure is None and closing is not None:
        raise closing


def _close_tdse_archive(
    archive: TDSEArchiveWriter | None, failure: BaseException | None
) -> None:
    """Close the archive session, never masking the run's own error.

    Parameters
    ----------
    archive : TDSEArchiveWriter or None
        The session.
    failure : BaseException or None
        The error the run is raising, if any.

    Raises
    ------
    TDSEArchiveError
        If closing fails and the run is not already raising.
    """
    if archive is None:
        return
    try:
        archive.close()
    except TDSEArchiveError as error:
        if failure is None:
            raise
        failure.add_note(f"the TDSE archive could not be closed: {error}")
        logger.exception("could not close the TDSE archive")


def _resolve_tdse_grid(
    grid: TimeGrid, responses: Sequence[object], settings: SimulationSettings
) -> tuple[TimeGrid, SimulationSettings]:
    """Resolve the response lattice once, before building the optical input.

    Parameters
    ----------
    grid : TimeGrid
        Original optical grid, before response refinement.
    responses : sequence of object
        Declared physical responses.
    settings : SimulationSettings
        Original collar and execution settings.

    Returns
    -------
    tuple
        Resolved grid and settings with a matching fine temporal collar.
    """
    atomic = [one for one in responses if isinstance(one, TDSEGasResponse)]
    if not atomic:
        if settings.tdse_execution is not None:
            raise DriverError("tdse_execution requires a TDSEGasResponse")
        return grid, settings
    if len(atomic) != 1 or len(responses) != 1:
        raise DriverError("TDSE owns the complete electronic response")
    if settings.tdse_execution is None:
        raise DriverError("TDSE requires explicit tdse_execution settings")
    refined = refine_response_grid(grid, atomic[0].sampling)
    return refined, _refined_tdse_collar(settings, grid, refined)


def _check_tdse_execution(simulation: Simulation, settings: SimulationSettings) -> None:
    """Refuse unsupported execution choices before any atomic preparation.

    Parameters
    ----------
    simulation : Simulation
        Declared response, geometry and optical input.
    settings : SimulationSettings
        Candidate execution and integration choices.
    """
    atomic = any(isinstance(one, TDSEGasResponse) for one in simulation.responses)
    if not atomic:
        if settings.tdse_execution is not None:
            raise DriverError("tdse_execution requires a TDSEGasResponse")
        return
    if (
        settings.tdse_execution is None
        or settings.backend != "numpy_ref"
        or not isinstance(settings.integrator, Multistep)
    ):
        raise DriverError(
            "TDSE requires explicit NumPy, Multistep and tdse_execution settings"
        )
    if simulation.noise is not None and simulation.noise.model == "modified":
        raise DriverError("TDSE refuses noise='modified'; use 'input' or None")
    if settings.schedule is not None or settings.interpolate_planes:
        raise DriverError("TDSE requires adaptive Adams with landed save planes")
    if _scheme(settings) != "nonlinear":
        raise DriverError("TDSE requires the in-stage temporal collar")
    if (
        simulation.spatial_grid is not None
        or simulation.general is not None
        or simulation.wall is not None
    ):
        raise DriverError("TDSE requires the uncoupled modal capillary route")


def _refined_tdse_collar(
    settings: SimulationSettings, parent: TimeGrid, refined: TimeGrid
) -> SimulationSettings:
    """Resample only a recognized analytic collar on the finer response lattice.

    Each rate is rebuilt from its own length (SEP 22 section 5.4): the spectral rate
    and the band from the reference length, the temporal rates from the collar's
    in-stage length, or as zeros for a spectral-only collar.

    Parameters
    ----------
    settings : SimulationSettings
        Original absorber definition and its physical lengths in m.
    parent : TimeGrid
        Grid on which an inherited standard collar was defined.
    refined : TimeGrid
        Resolved response grid with unchanged coarse coordinates and period.

    Returns
    -------
    SimulationSettings
        Settings carrying rates sampled on the resolved fine lattice.
    """
    boundary = settings.absorber
    if boundary is None or boundary.temporal_rate_fine.shape == (refined.n_to,):
        return settings
    original = collar(parent, boundary_length=boundary.reference_length)
    length = boundary.in_stage_length

    def temporal(grid: TimeGrid, axis: np.ndarray) -> np.ndarray:
        """Return the collar's temporal rate on a time axis of a grid, 1/m.

        Parameters
        ----------
        grid : TimeGrid
            The grid whose temporal collar the rate follows.
        axis : numpy.ndarray
            Its coarse or fine time axis, s.

        Returns
        -------
        numpy.ndarray
            The bounded profile at the in-stage length, or zeros for a
            spectral-only collar.
        """
        if not boundary.temporal:
            return np.zeros(axis.size)
        return temporal_collar_rate(axis, grid.t_collar, length)

    expected = {
        "spectral_rate": original.spectral_rate,
        "band_mask": original.band_mask,
        "temporal_rate": temporal(parent, parent.t),
        "temporal_rate_fine": temporal(parent, parent.t_fine),
    }
    if not all(
        np.array_equal(getattr(boundary, name), value)
        for name, value in expected.items()
    ):
        raise DriverError(
            "TDSE refinement requires a standard analytic collar or a custom "
            "absorber already sampled on the refined response grid"
        )
    return dataclasses.replace(
        settings,
        absorber=dataclasses.replace(
            boundary, temporal_rate_fine=temporal(refined, refined.t_fine)
        ),
    )


def _built(
    initial: Pulse | np.ndarray, grid: TimeGrid
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return the initial spectrum, building a pulse against the grid when given one.

    Parameters
    ----------
    initial : solprop.pulses.Pulse or numpy.ndarray
        A pulse, or an already-built spectrum.
    grid : solprop.grid.TimeGrid
        The grid to build against.

    Returns
    -------
    numpy.ndarray
        The spectrum, complex128, shape ``(n_modes, n_pol, n_omega)``.

    Raises
    ------
    DriverError
        If `initial` is neither an array nor a pulse.
    """
    if isinstance(initial, np.ndarray):
        return np.asarray(initial, dtype=np.complex128)
    # A runtime-checkable protocol is satisfied by the presence of `build`, not by its
    # being callable, so the guard below is still the one that names the mistake.
    build = getattr(initial, "build", None)
    if not callable(build):
        msg = (
            f"the input must be a pulse with a build(grid) method or a built spectrum, "
            f"got {type(initial).__name__}"
        )
        raise DriverError(msg)
    return np.asarray(build(grid), dtype=np.complex128)


def _interpolated(state: ResumeState) -> bool:
    """Return whether the plane a file resumes from was interpolated inside a step.

    A landed plane is the end of an accepted step of the run; an interpolated one lies
    inside a step. The entrance is the initial field.

    Parameters
    ----------
    state : solprop.output.hdf5.ResumeState
        What the file holds.

    Returns
    -------
    bool
        Whether the plane is not a state the march passed through.
    """
    if state.checkpoint is not None:
        return not state.checkpoint.landed
    if state.n_saved <= 1:
        return False
    history = state.history
    # The history records where each attempt started and how long it was.
    ends = (history.z + history.h)[history.accepted]
    return not bool(np.any(np.isclose(ends, state.plane.z, rtol=1e-12, atol=0.0)))


def _open_output(
    output: Output | Path | str | None,
    metadata: RunMetadata,
    *,
    resume: _ResumeChoice,
) -> tuple[Output, ResumeState | None, RecordComparison | None]:
    """Choose the handler a run writes to, and read what a resumed file holds.

    The resume gate (SEP 42): the file's record and `metadata` are compared path by
    path. A layout difference is refused, and a compared one unless the resume is
    forced; numerics may differ. What depends on the file's state -- whether work is
    left, whether a session forced it -- is decided by :func:`_resume_decision`.

    Parameters
    ----------
    output : solprop.output.schema.Output or pathlib.Path or str or None
        The caller's choice.
    metadata : solprop.output.schema.RunMetadata
        The record of the run about to start, which a file is compared with.
    resume : bool or {'interpolated', 'force'}
        Whether to continue an existing file; ``"interpolated"`` also from a plane
        interpolated inside a step, and ``"force"`` across compared differences too.

    Returns
    -------
    tuple
        The handler, the state to resume from or ``None``, and the comparison of the
        two records or ``None``.

    Raises
    ------
    DriverError
        If a resume is asked for and `output` is not a path.
    OutputError
        If the file exists and its layout differs, or a compared value differs and
        the resume is not forced.
    """
    if resume and metadata.configuration.get("_benchmark_plasma_source"):
        raise DriverError("gated plasma benchmark output cannot be resumed")
    if isinstance(output, Path | str):
        path = Path(output)
        found = _compare_cache(path, metadata) if resume else None
        if found is not None:
            stored, comparison = found
            if comparison.layout or (comparison.refused and resume != "force"):
                raise OutputError(_resume_refusal(path, stored, comparison))
            state = resume_state(path)
            return HDF5Output(path, append=True, gated=True), state, comparison
        return HDF5Output(path), None, None
    if resume:
        msg = (
            "only a path can be resumed: the file has to be opened in append mode, "
            "which is the driver's to decide; pass output=<path> with resume=True"
        )
        raise DriverError(msg)
    return (MemoryOutput() if output is None else output), None, None


@dataclasses.dataclass(frozen=True)
class _Continuation:
    """How a resumed session continues its file (SEP 42 section 4).

    Attributes
    ----------
    entry : dict or None
        The ``resume`` entry its execution records carry, or ``None`` when nothing
        but rounding separates the records.
    forced : bool
        Whether it continues across compared differences, under ``resume="force"``.
    warning : str
        What to warn once the session is sure to continue: after a forced one's
        statistics are found to fit, so that a refusal is not preceded by it.
    """

    entry: dict[str, object] | None
    forced: bool = False
    warning: str = ""


_FORCED_NOTE = (
    "From here the file's record does not describe its planes: stats/* hold, and "
    "anything rebuilt from the record does not."
)


def _session_planes(
    state: ResumeState | None, continuation: _Continuation | None
) -> int:
    """Return the first plane a post-run report reads.

    Parameters
    ----------
    state : ResumeState or None
        What a resumed file held, or ``None`` for a fresh run.
    continuation : _Continuation or None
        How the session continued it.

    Returns
    -------
    int
        The first plane this session wrote when the file was forced, by this
        session or an earlier one, since the planes before were computed under other
        physics (SEP 42 section 5.5); otherwise 0, every plane.
    """
    if state is None:
        return 0
    forced = (continuation is not None and continuation.forced) or (
        state.outcome is not None and bool(state.outcome.forced_positions)
    )
    return state.n_saved if forced else 0


def _check_resume_choice(resume: object) -> None:
    """Refuse a ``resume`` value that is not one of its four.

    Parameters
    ----------
    resume : object
        The caller's value.

    Raises
    ------
    ValueError
        If it is not ``False``, ``True``, ``"interpolated"`` or ``"force"``: a
        mistyped word would otherwise read as ``True``.
    """
    if resume is True or resume is False or resume in _RESUME_CHOICES[2:]:
        return
    msg = f"resume must be False, True, 'interpolated' or 'force', got {resume!r}"
    raise ValueError(msg)


def _work_left(problem: KernelProblem, state: ResumeState, next_plane: int) -> bool:
    """Return whether a file has anything left to compute, by position.

    Parameters
    ----------
    problem : KernelProblem
        The translated problem, for its save planes.
    state : ResumeState
        What the file holds.
    next_plane : int
        The next save plane beyond the file's position.

    Returns
    -------
    bool
        True while a save plane lies ahead or a terminal checkpoint is still owed
        its plane. Decided by position, not status: a file killed after its last
        plane has no work left, though its status reads ``running``.
    """
    return state.checkpoint is not None or next_plane < len(problem.stepper.save_planes)


def _resume_entry(
    state: ResumeState, comparison: RecordComparison, *, forced: bool
) -> dict[str, object] | None:
    """Return the ``resume`` entry of a session's execution records (SEP 42 §5.3).

    Parameters
    ----------
    state : ResumeState
        What the file holds.
    comparison : RecordComparison
        What separates the file's record from the requested one.
    forced : bool
        Whether the session continues across compared differences.

    Returns
    -------
    dict or None
        Where the session continues from, whether it was forced, the numerics it
        changed, the compared values as a grouped text, and the paths read as
        neutral; ``None`` when there is nothing to record.
    """
    if not (comparison.numerics or comparison.neutral or forced):
        return None
    return {
        "from_z": float(state.plane.z),
        "from_plane": state.n_saved - 1,
        "forced": forced,
        "numerics": comparison.numerics_record(),
        # The grouped text, not every path: the record is rewritten at every plane.
        "compared": comparison.compared_summary() if forced else "",
        "neutral": [dotted(path) for path in comparison.neutral],
    }


def _confirm_continuation(
    continuation: _Continuation | None,
    backend: Backend,
    problem: KernelProblem,
    output: Output | Path | str | None,
) -> None:
    """Check a primed continuation can go ahead, then warn what it does.

    Parameters
    ----------
    continuation : _Continuation or None
        How the session continues its file, or ``None`` for a fresh run.
    backend : Backend
        The primed backend.
    problem : KernelProblem
        The translated problem.
    output : pathlib.Path or str
        The file.
    """
    if continuation is None:
        return
    if continuation.forced and problem.tdse is None:
        _check_forced_shapes(backend, problem.stats_spec, output)
    if continuation.warning:
        warnings.warn(continuation.warning, SolpropWarning, stacklevel=3)


def _check_forced_shapes(
    backend: Backend, spec: StatsSpec, output: Output | Path | str | None
) -> None:
    """Refuse a forced resume whose statistics would not fit the file's arrays.

    A statistic's names and shape follow the responses and the species, so a forced
    change of the number of species, or of plasma on or off, would compute a whole
    segment and then fail at its first save, leaving a checkpoint from the other
    physics. Checked after the backend is primed, before any step or write
    (SEP 42 section 5.5).

    Parameters
    ----------
    backend : Backend
        The primed backend.
    spec : StatsSpec
        The statistics the run collects.
    output : pathlib.Path or str
        The file being continued.

    Raises
    ------
    OutputError
        If a statistic is missing from either side or its shape differs.
    """
    path = Path(cast("Path | str", output))
    stored = _stored_stat_shapes(path)
    requested = {
        name: tuple(np.shape(value)[1:]) for name, value in backend.stats(spec).items()
    }
    changed = sorted(
        f"{name} (stored {stored.get(name)}, now {requested.get(name)})"
        for name in set(stored) | set(requested)
        if stored.get(name) != requested.get(name)
    )
    if changed:
        msg = (
            f"{path} cannot be continued with resume='force': its statistics do not "
            f"fit this run's, which changes {', '.join(changed)}; nothing was "
            "computed or written. Launch a new segment from a saved plane instead"
        )
        raise OutputError(msg)


def _refuse_interpolated(path: Path, state: ResumeState, resume: _ResumeChoice) -> None:
    """Refuse to continue from a plane interpolated inside a step, unless accepted.

    An interpolated plane is not a state the march passed through, so a run restarted
    there is not the run it continues (SEP 14 section 5); ``resume="interpolated"``
    or ``"force"`` accepts that. Tested on the file, not on the requested settings,
    since ``interpolate_planes`` may change on a resume (SEP 42 section 5.6).

    Parameters
    ----------
    path : pathlib.Path
        The file.
    state : ResumeState
        What it holds.
    resume : bool or {'interpolated', 'force'}
        The caller's request.

    Raises
    ------
    DriverError
        If the file ends on an interpolated plane and the request does not accept it.
    """
    if resume in ("interpolated", "force") or not _interpolated(state):
        return
    msg = (
        f"{path} ends on a plane interpolated inside a step at z = "
        f"{state.plane.z:.6e} m, not a state the march passed through; "
        "pass resume='interpolated' to continue afresh from it, or rerun"
    )
    raise DriverError(msg)


def _resume_decision(
    path: Path,
    problem: KernelProblem,
    state: ResumeState,
    next_plane: int,
    comparison: RecordComparison,
    resume: _ResumeChoice,
) -> _Continuation | None:
    """Decide what a resume of a file does, once the records' layouts agree.

    The verdict of SEP 42 section 4: a file with work left continues, recording
    changed numerics, and continues across compared differences only when forced; a
    file with nothing left is returned unless the request's numerics differ from any
    session's; a file a session forced continues only when forced again.

    Parameters
    ----------
    path : pathlib.Path
        The file.
    problem : KernelProblem
        The translated problem.
    state : ResumeState
        What the file holds.
    next_plane : int
        The next save plane beyond the file's position.
    comparison : RecordComparison
        What separates the file's record from the requested one.
    resume : bool or {'interpolated', 'force'}
        The caller's request.

    Returns
    -------
    _Continuation or None
        How the session continues, or ``None`` when there is nothing to do.

    Raises
    ------
    OutputError
        If a session forced the file and this one does not, while work is left; if
        a forced resume has nothing to continue; or if a finished file was computed
        under numerics other than the requested ones.
    """
    work_left = _work_left(problem, state, next_plane)
    outcome = state.outcome
    forced_at = () if outcome is None else outcome.forced_positions
    where = ", ".join(f"{z:.6g}" for z in forced_at)
    if forced_at and resume != "force":
        if not work_left:
            warnings.warn(
                f"{path} was continued with resume='force' at z = {where} m. "
                + _FORCED_NOTE,
                SolpropWarning,
                stacklevel=3,
            )
            return None
        msg = (
            f"{path} was continued with resume='force' at z = {where} m, so its "
            "record no longer describes the run it holds; continue it with "
            "resume='force', or write a new file"
        )
        raise OutputError(msg)
    if comparison.refused:
        if not work_left:
            msg = (
                f"{path} holds a finished run, and resume='force' continues only a "
                f"file with work left; {comparison.summary()}"
            )
            raise OutputError(msg)
        launch = any(
            change.path[:2]
            in (("configuration", "pulse"), ("configuration", "initial"))
            for change in comparison.refused
        )
        warning = (
            f"forcing a resume of {path} at z = {state.plane.z:.6g} m; "
            f"{comparison.summary()}\n"
            + (
                "The launch differs, which has no effect: the stored field continues.\n"
                if launch
                else ""
            )
            + "The Adams history starts afresh. "
            + _FORCED_NOTE
        )
        return _Continuation(
            _resume_entry(state, comparison, forced=True), forced=True, warning=warning
        )
    if not work_left:
        earlier = () if outcome is None else outcome.resumes
        if comparison.numerics or any(entry.get("numerics") for entry in earlier):
            # A tightened tolerance must not hand back the loose result, and a file
            # finished under other numerics than it started with matches neither.
            changed = comparison.summary() or "an earlier session changed its numerics"
            msg = (
                f"{path} holds a finished run that was not computed throughout under "
                f"the requested numerics, so nothing would be computed; {changed}. "
                "Write a new file to compute it under these"
            )
            raise OutputError(msg)
        return None
    _refuse_interpolated(path, state, resume)
    warning = (
        f"resuming {path} at z = {state.plane.z:.6g} m with changed numerics; "
        f"{comparison.summary()}"
        if comparison.numerics
        else ""
    )
    return _Continuation(
        _resume_entry(state, comparison, forced=False), warning=warning
    )


def _gnlse_grid(
    grid: TimeGrid | None,
    length: float,
    wavelength: float,
    wavelength_limits: tuple[float, float] | None,
    trange: float | None,
) -> TimeGrid:
    """Return the envelope grid a GNLSE run propagates on.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid or None
        A grid given by the caller, or ``None`` to build one.
    length : float
        Fibre length, m.
    wavelength : float
        Reference wavelength, m.
    wavelength_limits : tuple of float or None
        Window limits, m.
    trange : float or None
        Time window, s.

    Returns
    -------
    solprop.grid.TimeGrid
        The envelope grid.

    Raises
    ------
    DriverError
        If a field grid is given, or a grid can be neither taken nor built.
    """
    if grid is not None:
        if grid.kind != "envelope":
            msg = (
                f"prop_gnlse propagates an envelope grid and this one is a "
                f"{grid.kind} grid; the GNLSE mapping is the envelope Kerr response "
                "(docs/guide/waveguides/gnlse.md), and a field-resolved run is "
                "prop_capillary's "
                "path"
            )
            raise DriverError(msg)
        return grid
    if wavelength_limits is None or trange is None:
        msg = (
            "give wavelength_limits and trange so a grid can be built, or a grid of "
            "your own"
        )
        raise DriverError(msg)
    return EnvelopeGrid.from_limits(length, wavelength, wavelength_limits, trange)


def _gnlse_fibre(  # noqa: PLR0913 - the coefficients of the mapping
    gamma: float | None,
    betas: Sequence[float] | np.ndarray | None,
    beta: object,
    grid: TimeGrid,
    *,
    wavelength: float,
    loss: float | None,
    shock: bool,
    fibre: Mode | IndexTable | None = None,
    n2: float | str | None = None,
    raman: bool = False,
    raman_fraction: float | None = None,
    selection: MaterialSelection | None = None,
) -> GnlseProblem:
    """Map one dispersion source and one nonlinear coefficient onto the GNLSE.

    Parameters
    ----------
    gamma : float or None
        Nonlinear coefficient in 1/(W m), absent beside `n2`.
    betas : array_like or None
        Taylor coefficients about the reference frequency, s^n/m.
    beta : object
        A pair of frequency and beta arrays, a callable, or None.
    grid : solprop.grid.TimeGrid
        Envelope grid whose active band samples a callable dispersion.
    wavelength : float
        Positive reference wavelength, m.
    loss : float or None
        Power loss in dB/m; None means omitted. Refused beside `fibre`.
    shock : bool
        Keep the prefactor's frequency dependence.
    fibre : Mode or IndexTable or None, optional
        Whole dispersion and loss, including a positive propagation area for `n2`.
    n2 : float, str or None, optional
        Nonlinear index in m^2/W, or ``"recommended"`` under a `selection`;
        requiring `fibre`.
    raman : bool, optional
        Include the silica Raman response.
    raman_fraction : float or None, optional
        Share of the nonlinearity held back from the Kerr term, between zero and one;
        None resolves as in :func:`solprop.nonlinear.gnlse.problem_from`.
    selection : MaterialSelection or None, optional
        The run's material selection, which takes silica's response from optmat.

    Returns
    -------
    solprop.nonlinear.gnlse.GnlseProblem
        Resolved mode, nonlinear responses and their coefficient mapping.

    Raises
    ------
    DriverError
        If dispersion, nonlinearity, area, loss or z dependence is incompatible.
    """
    if sum(value is not None for value in (betas, beta, fibre)) != 1:
        raise DriverError("give exactly one of betas, beta and fibre")
    if (gamma is None) == (n2 is None):
        raise DriverError("give exactly one of gamma and n2")
    if n2 is not None and fibre is None:
        raise DriverError("n2 requires fibre with an effective area")
    if fibre is not None and loss is not None:
        raise DriverError("the fibre carries its loss; scale it on the mode")
    omega_ref = 2.0 * np.pi * C / wavelength
    mode = (
        None
        if fibre is None
        else _gnlse_mode(fibre, wavelength, needs_area=n2 is not None)
    )
    area = 1.0 if mode is None else mode.aeff()
    # The GNLSE mapping of ADR-026 (docs/dev/decisions.md) is unchanged. An externally
    # specified fibre replaces the placeholder mode, so its dispersion never enters the
    # unit prefactor.
    mapping = problem_from(
        np.asarray([0.0] if betas is None else betas, dtype=np.float64),
        omega_ref=omega_ref,
        gamma=gamma,
        n2=n2,
        aeff=area,
        loss_db_per_m=0.0 if loss is None else loss,
        shock=shock,
        raman=raman,
        raman_fraction=raman_fraction,
        selection=selection,
    )
    if mode is not None:
        return dataclasses.replace(mapping, mode=mode)
    if betas is not None:
        return mapping
    nodes, values = _dispersion_table(beta, grid)
    return dataclasses.replace(
        mapping,
        mode=SpecifiedMode.from_table(
            nodes,
            values,
            loss=attenuation_from_db(0.0 if loss is None else loss),
            aeff=mapping.aeff,
        ),
    )


def _gnlse_mode(
    fibre: Mode | IndexTable, wavelength: float, *, needs_area: bool
) -> Mode:
    """Resolve a supplied table or mode with its propagation area at the pump.

    Parameters
    ----------
    fibre : Mode or IndexTable
        Whole linear response.
    wavelength : float
        Positive reference wavelength, m: where a table's area column and a
        provider channel's surrogate area are evaluated.
    needs_area : bool
        Whether the caller supplied n2 rather than gamma.

    Returns
    -------
    Mode
        Mode with its area selected at the reference wavelength, or carrying the
        declared dispersion and area paths of a fibre whose geometry varies along z.

    Raises
    ------
    DriverError
        If n2 has no table area, or the fibre's entrance area is not usable.
    """
    if isinstance(fibre, IndexTable):
        if needs_area and fibre.area is None:
            raise DriverError("n2 requires a fibre table with an effective-area column")
        area = (
            1.0
            if fibre.area is None
            else float(
                fibre.effective_area(np.array([2.0 * np.pi * C / wavelength]))[0]
            )
        )
        return SpecifiedMode.from_index_table(fibre, aeff=area)
    if not math.isfinite(fibre.aeff(0.0)) or fibre.aeff(0.0) <= 0:
        raise DriverError("fibre effective area must be positive and finite")
    # aeff=None hands over the fibre's own declared area path when it has one and
    # samples the source's own area along z when its whole geometry moves; the
    # frozen callables forward the position, so the dispersion and the area move
    # together and a fibre moving along z needs no refusal (ADR-026,
    # docs/dev/decisions.md). The wavelength is where a provider channel's area is
    # solved, as a table's is read at it above; every other mode's area is not a
    # function of frequency and ignores it.
    return SpecifiedMode.from_mode(fibre, wavelength=wavelength)


def _dispersion_table(
    beta: object, grid: TimeGrid
) -> tuple[Float[np.ndarray, " n_nodes"], Float[np.ndarray, " n_beta_nodes"]]:
    """Return the ``(omega, beta)`` table a caller's dispersion describes.

    A callable is sampled on the grid's **active band**, sorted and made unique, because
    that is exactly where the linear operator evaluates a mode
    (:func:`solprop.linear.dispersion_operator`) and a spline refuses to extrapolate
    beyond its data.

    Parameters
    ----------
    beta : object
        A pair of arrays, or a callable of ``omega``.
    grid : solprop.grid.TimeGrid
        The grid supplying the sampling points.

    Returns
    -------
    tuple
        The frequencies in rad/s and the propagation constant in 1/m.

    Raises
    ------
    DriverError
        If `beta` is neither a pair of arrays nor a callable.
    """
    if callable(beta):
        nodes = np.unique(grid.omega[grid.active_mask])
        return nodes, np.asarray(beta(nodes), dtype=np.float64)
    if isinstance(beta, tuple) and len(beta) == 2:
        nodes, values = beta
        return (
            np.asarray(nodes, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
    msg = (
        f"beta must be a (omega, beta) pair of arrays or a callable of omega, got "
        f"{type(beta).__name__}"
    )
    raise DriverError(msg)


def _shaped_pulse(  # noqa: PLR0913 - the pulse's keyword surface
    shape: str,
    *,
    wavelength: float,
    fwhm: float | None,
    width: float | None,
    energy: float | None,
    peak_power: float | None,
    phases: Sequence[float] | np.ndarray,
    pulses: Sequence[Pulse] | None = None,
) -> GaussianPulse | SechPulse | PulseSum:
    """Return the shaped input pulse a keyword interface starts from.

    Parameters
    ----------
    shape : {'gauss', 'sech'}
        Shape of the input.
    wavelength : float
        Centre wavelength, m, and the reference of the spectral phase.
    fwhm : float or None
        Intensity full width at half maximum, s.
    width : float or None
        Natural width of a ``sech**2``, s.
    energy : float or None
        Pulse energy, J.
    peak_power : float or None
        Peak power, W.
    phases : array_like
        Taylor coefficients of the optics-convention spectral phase.
    pulses : sequence of solprop.pulses.Pulse or None, optional
        Several pulses launched together, in place of the single-pulse keywords.

    Returns
    -------
    solprop.pulses.GaussianPulse or solprop.pulses.SechPulse or solprop.pulses.PulseSum
        The pulse specification.

    Raises
    ------
    DriverError
        If the shape is unknown, a Gaussian is asked for by a natural width, which
        solprop specifies by its full width at half maximum only, or `pulses` is given
        beside a single-pulse keyword.
    """
    if pulses is not None:
        given = [
            name
            for name, value in (
                ("fwhm", fwhm),
                ("width", width),
                ("energy", energy),
                ("peak_power", peak_power),
            )
            if value is not None
        ]
        if len(phases):
            given.append("phases")
        if shape != "gauss":
            given.append("shape")
        if given:
            msg = (
                f"pulses= describes every input pulse itself, and this call also sets "
                f"{', '.join(given)}: give each pulse its own duration, energy and "
                "phase, and leave the single-pulse keywords unset"
            )
            raise DriverError(msg)
        return PulseSum(tuple(pulses))
    if shape == "sech":
        return SechPulse(
            wavelength=wavelength,
            fwhm=fwhm,
            width=width,
            energy=energy,
            peak_power=peak_power,
            phase=phases,
        )
    if shape != "gauss":
        msg = f"shape must be 'gauss' or 'sech', got {shape!r}"
        raise DriverError(msg)
    if width is not None or fwhm is None:
        msg = (
            "a Gaussian input is specified by its intensity fwhm; give fwhm= rather "
            "than width=, which is the natural width of a sech**2 pulse"
        )
        raise DriverError(msg)
    return GaussianPulse(
        wavelength=wavelength,
        fwhm=fwhm,
        energy=energy,
        peak_power=peak_power,
        phase=phases,
    )


def _save_planes(
    length: float,
    save_planes: Sequence[float] | np.ndarray | None,
    n_save: int | None,
) -> Float[np.ndarray, " n_z"]:
    """Return the save planes a keyword interface asks for.

    Parameters
    ----------
    length : float
        Fibre length, m.
    save_planes : array_like or None
        Positions given by the caller, m.
    n_save : int or None
        Planes to store, counting the one at ``z = 0``.

    Returns
    -------
    numpy.ndarray
        The planes after ``z = 0``, m, shape ``(n_z,)``.

    Raises
    ------
    DriverError
        If both or neither way of asking is usable, or the count is below two.
    """
    if save_planes is not None:
        if n_save is not None:
            msg = "give save_planes or n_save, not both"
            raise DriverError(msg)
        return np.ascontiguousarray(save_planes, dtype=np.float64).reshape(-1)
    count = DEFAULT_N_SAVE if n_save is None else n_save
    if count < 2:
        msg = (
            f"n_save counts the plane at z = 0, so it must be at least 2, got {count}; "
            "a run always stores the entrance"
        )
        raise DriverError(msg)
    return np.linspace(0.0, length, count)[1:]


def _scheme(settings: SimulationSettings) -> AbsorberScheme:
    """Return the collar scheme the settings resolve to, ``"nonlinear"`` if unnamed.

    Parameters
    ----------
    settings : SimulationSettings
        The run's settings.

    Returns
    -------
    {'nonlinear', 'split_step'}
        The scheme.
    """
    return (
        DEFAULT_ABSORBER_SCHEME
        if settings.absorber_scheme is None
        else settings.absorber_scheme
    )


def _spatial_settings(settings: SimulationSettings, route: str) -> SimulationSettings:
    """Fill an omitted controller with the spatial interface's default (SEP 37).

    Parameters
    ----------
    settings : SimulationSettings
        The caller's settings.
    route : {'radial', 'free', 'planar'}
        The interface they are for.

    Returns
    -------
    SimulationSettings
        `settings`, with the interface's default controller where it gave none
        (:data:`SPATIAL_RTOL` for radial and planar runs, :data:`FREE_RTOL` for
        Cartesian and line ones, each as :func:`_default_rtol` gives it in the run's
        precision), for a free-space run that named no integrator,
        ``RungeKutta()``, and, where no collar scheme is named, ``"split_step"``
        unless planes are interpolated (SEP 21).
    """
    if settings.absorber_scheme is None and not settings.interpolate_planes:
        # The in-stage collar cost the spatial cases 1.2 to 4.6 times the steps of the
        # split scheme, at equal or worse accuracy (SEP 21 Part II).
        settings = dataclasses.replace(settings, absorber_scheme="split_step")
    if route == "free" and settings._route_integrator is None:
        # The Adams history's memory rules the multistep method out for a large
        # transverse state: a free-space run takes the Runge-Kutta march (SEP 37), as
        # its default, so a copy in either precision keeps it and a named one wins.
        settings = dataclasses.replace(settings, _route_integrator=RungeKutta())
    if settings.controller is not None:
        return settings
    rtol, atol = (
        (FREE_RTOL, FREE_ATOL) if route == "free" else (SPATIAL_RTOL, SPATIAL_ATOL)
    )
    controller = PIController(rtol=_default_rtol(rtol, settings.precision), atol=atol)
    return dataclasses.replace(settings, controller=controller)


def _with_boundary(
    settings: SimulationSettings,
    grid: TimeGrid,
    absorber: Absorber | Literal[False] | None,
    scheme: AbsorberScheme | None,
) -> SimulationSettings:
    """Apply a spatial interface's absorber keywords to its settings (SEP 21 section 5).

    Parameters
    ----------
    settings : SimulationSettings
        The settings given.
    grid : TimeGrid
        The run's temporal grid.
    absorber : Absorber, False or None
        The collar given; ``False`` zeroes the temporal and spectral rates of the
        settings' absorber and keeps its lengths (without one it builds the hard mask
        alone); ``None`` leaves the settings' own. The transverse boundary is
        ``spatial_absorber``'s (SEP 40).
    scheme : {'nonlinear', 'split_step'} or None
        The temporal scheme given; ``None`` leaves the settings' own.

    Returns
    -------
    SimulationSettings
        The settings with the keywords applied.

    Raises
    ------
    DriverError
        If a keyword conflicts with what the settings already declare, or a scheme is
        given beside ``absorber=False``.
    """
    changes: dict[str, object] = {}
    if absorber is False:
        if scheme is not None:
            msg = (
                "absorber=False switches the temporal collar off, so there is nothing "
                f"for absorber_scheme={scheme!r} to apply; give one or the other"
            )
            raise DriverError(msg)
        current = settings.absorber
        changes["absorber"] = (
            no_absorber(grid)
            if current is None
            else dataclasses.replace(
                current,
                spectral_rate=np.zeros_like(current.spectral_rate),
                temporal_rate=np.zeros(grid.n_t),
                temporal_rate_fine=np.zeros(grid.n_to),
                temporal=False,
            )
        )
    elif absorber is not None:
        if settings.absorber is not None:
            msg = (
                "the settings already declare an absorber; give it there or as the "
                "absorber keyword, not both"
            )
            raise DriverError(msg)
        changes["absorber"] = absorber
    if scheme is not None:
        if settings.absorber_scheme not in (None, scheme):
            msg = (
                f"the settings already declare absorber_scheme="
                f"{settings.absorber_scheme!r}; give it there or as the keyword"
            )
            raise DriverError(msg)
        changes["absorber_scheme"] = scheme
    return dataclasses.replace(settings, **changes) if changes else settings


def _watched(
    settings: SimulationSettings, check: BoundaryCheck | None
) -> SimulationSettings:
    """Apply a spatial constructor's ``boundary_check`` keyword to its settings.

    Parameters
    ----------
    settings : SimulationSettings
        The caller's settings.
    check : solprop.monitor.BoundaryCheck or None
        The keyword's value; the default leaves the settings' own.

    Returns
    -------
    SimulationSettings
        `settings`, watching the edges as the keyword asks.

    Raises
    ------
    DriverError
        If the keyword and the settings name two different checks.
    """
    if check is DEFAULT_BOUNDARY_CHECK or check == settings.boundary_check:
        return settings
    if settings.boundary_check != DEFAULT_BOUNDARY_CHECK:
        msg = (
            "the settings already declare a boundary_check; give it there or as the "
            "boundary_check keyword, not both"
        )
        raise DriverError(msg)
    return dataclasses.replace(settings, boundary_check=check)


def _rechecked(
    simulation: Simulation, original: KernelProblem, derived: KernelProblem
) -> SpatialAbsorberDesign | None:
    """Return a derived run's transverse design, its check rerun on the new launch.

    SEP 40 section 5.2: a derived run keeps its parent's design, which its shared
    operator and transform were built with, but not its verdict, which belongs to a
    launch. Only a scalar multiple of the checked launch reuses it, because the check
    is linear and normalised by the launch's energy. A new launch whose rms
    wavenumber lies above the one the band was derived from warns as well.

    Parameters
    ----------
    simulation : Simulation
        The parent, for its grids and medium.
    original : solprop.translate.problem.KernelProblem
        The parent's translated problem.
    derived : solprop.translate.problem.KernelProblem
        The derived problem, with the new launch.

    Returns
    -------
    solprop.spatial_absorber.SpatialAbsorberDesign or None
        The design with the new launch's check.

    Raises
    ------
    DriverError
        If the check refuses the design for the new launch.
    """
    design = original.transverse
    if design is None or not design.has_collar:
        return design
    space = simulation.spatial_grid
    medium = simulation.medium
    assert isinstance(space, RadialGrid | FreeGrid | PlanarGrid)
    assert medium is not None
    old = np.asarray(original.initial.field0[:1])
    new = np.asarray(derived.initial.field0[:1])
    reference = float(np.vdot(old, old).real)
    if reference > 0.0:
        scale = np.vdot(old, new) / reference
        residual = float(np.linalg.norm(new - scale * old))
        if residual <= 1e-12 * float(np.linalg.norm(new)):
            return design
    planes = derived.stepper.save_planes
    length = float(np.max(planes)) - float(derived.initial.z)
    dispersion = SpatialDispersion(
        medium, simulation.grid, Frame.comoving(simulation.grid)
    )
    if design.specification.band_source == "launch":
        before = resolve_transverse(
            design.specification, space, dispersion, old, length
        )
        after_kappa = _launch_band(design.specification, space, dispersion, new, length)
        if after_kappa is not None and after_kappa > max(before.kappa_p) * (1 + 1e-9):
            warnings.warn(
                "the new launch spreads beyond the band its run's collar was designed "
                f"for ({after_kappa:.3g} > {max(before.kappa_p):.3g} rad/m): what it "
                "carries above that band is absorbed only partly; translate afresh",
                BoundaryWarning,
                stacklevel=3,
            )
    try:
        return checked_design(
            dataclasses.replace(design, check={}), space, dispersion, new, length
        )
    except TranslationError as error:
        msg = (
            f"{error}. The grid is shared with the parent run, so translate a new "
            "simulation with that collar_fraction for this launch"
        )
        raise DriverError(msg) from error


def _launch_band(
    specification: SpatialAbsorber,
    space: RadialGrid | FreeGrid | PlanarGrid,
    dispersion: SpatialDispersion,
    launch: np.ndarray,
    length: float,
) -> float | None:
    """Return the widest protected band a launch would derive, or ``None``.

    Parameters
    ----------
    specification : solprop.boundaries.SpatialAbsorber
        A launch-derived specification.
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The grid.
    dispersion : solprop.spatial.SpatialDispersion
        The medium and the temporal grid.
    launch : numpy.ndarray
        The launch, canonical layout.
    length : float
        The run's length, m.

    Returns
    -------
    float or None
        The widest axis's band edge, rad/m; ``None`` when it cannot be resolved.
    """
    try:
        return max(
            resolve_transverse(specification, space, dispersion, launch, length).kappa_p
        )
    except TranslationError:
        return None


def _refuse_pending_check(problem: KernelProblem) -> None:
    """Refuse to start a run whose collar's design check never saw a launch.

    A coupled child is translated before its launch exists; its check runs when
    :meth:`Simulation.with_input` gives it one (SEP 40 section 5.2).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The run's problem.

    Raises
    ------
    DriverError
        If its transverse design's check is pending.
    """
    design = problem.transverse
    if design is not None and design.check.get("status") == "pending":
        msg = (
            "this run's collar has not been checked against a launch: give it one "
            "with with_input before running it (SEP 40 section 5.2)"
        )
        raise DriverError(msg)


def _bounded(
    settings: SimulationSettings,
    spatial_absorber: SpatialAbsorber | Literal[False] | None,
) -> SimulationSettings:
    """Apply a spatial constructor's ``spatial_absorber`` keyword to its settings.

    Parameters
    ----------
    settings : SimulationSettings
        The caller's settings.
    spatial_absorber : solprop.boundaries.SpatialAbsorber, False or None
        The keyword's value; ``None`` leaves the settings' own.

    Returns
    -------
    SimulationSettings
        `settings`, with the transverse boundary the keyword asks for.

    Raises
    ------
    DriverError
        If the keyword and the settings name two different boundaries.
    """
    if spatial_absorber is None or spatial_absorber == settings.spatial_absorber:
        return settings
    if settings.spatial_absorber is not None:
        msg = (
            "the settings already declare a spatial_absorber; give it there or as "
            "the spatial_absorber keyword, not both"
        )
        raise DriverError(msg)
    return dataclasses.replace(settings, spatial_absorber=spatial_absorber)


def _resolve_noise(
    noise: str | None,
    seed: int | None,
    precision: str,
    *,
    tdse: bool = False,
    quiet: str | None = None,
    unmeasured: str | None = None,
) -> tuple[NoiseModel | None, NoiseResolution]:
    """Resolve a constructor's ``noise`` keyword into the model a run carries (SEP 37).

    ``"default"`` resolves to ``"input"``, the one-photon-per-mode seed added to the
    input field, except under single precision, whose binary32 rounding swamps the seed:
    there it resolves to none and says why. A seed beside a default that resolves to
    none is refused rather than silently dropped, and so is a seed beside
    ``noise=None``. Any other value is passed through, and an unknown one is refused
    where it is drawn.

    The arguments are plain strings rather than :data:`NoiseSetting` for the reason
    :func:`_shot_noise` gives.

    Parameters
    ----------
    noise : {'default', None, 'input', 'modified'}
        The keyword as given.
    seed : int or None
        The keyword ``seed`` as given.
    precision : {'double', 'single'}
        The run's arithmetic.
    tdse : bool, optional
        Whether the run carries the TDSE response, which takes no noise by default
        and warns when noise is asked for: its runs are expensive, rarely repeated as
        ensembles, and checked coherently.
    quiet : str or None, optional
        Why this interface takes no noise by default, as the free-space one does:
        ``"default"`` then resolves to none, recorded with this reason.
    unmeasured : str or None, optional
        Why this run can carry no noise at all, as a slab mode set cannot, whose
        amplitudes have no declared photon measure: ``"default"`` resolves to none
        with this reason, and an explicit model is refused.

    Returns
    -------
    tuple
        The model to draw, or ``None``, and the record of how it was chosen.

    Warns
    -----
    SolpropWarning
        When a TDSE run asks for a noise model explicitly.

    Raises
    ------
    DriverError
        If a seed is given and the run resolves to no noise, or a model is asked
        for where `unmeasured` says none can be drawn.

    Examples
    --------
    >>> from solprop.driver.simulation import _resolve_noise
    >>> _resolve_noise("default", None, "double")
    ('input', NoiseResolution(requested='default', resolved='input', reason=None))
    >>> _resolve_noise("default", None, "single")[0] is None
    True
    """
    if unmeasured is not None and noise not in ("default", None):
        msg = f"noise={noise!r}: {unmeasured}; pass noise=None"
        raise DriverError(msg)
    if noise == "default":
        fallback = (
            SINGLE_PRECISION_NOISE
            if precision == "single"
            else TDSE_NOISE
            if tdse
            else unmeasured or quiet
        )
        if fallback is not None:
            if seed is not None:
                msg = (
                    f"a noise seed was given, but this run takes no noise by default "
                    f"({fallback}): pass a noise model with the seed, or drop the seed"
                )
                raise DriverError(msg)
            return None, NoiseResolution("default", None, fallback)
        return "input", NoiseResolution("default", "input")
    if noise is None and seed is not None:
        msg = "a noise seed requires a noise model"
        raise DriverError(msg)
    if tdse and noise is not None:
        msg = (
            f"noise={noise!r} beside the TDSE response: the atomic source is expensive "
            "and a TDSE run is rarely repeated as a noise ensemble; pass noise=None "
            "unless the noise is what this run is about"
        )
        warnings.warn(msg, SolpropWarning, stacklevel=4)
    # An unknown model passes through to be refused where it is drawn.
    return cast("NoiseModel | None", noise), NoiseResolution(noise, noise)


def _shot_noise(
    noise: str | None, grid: TimeGrid, seed: int | None, *, n_modes: int = 1
) -> NoiseRealisation | None:
    """Draw the shot-noise realisation a run carries.

    The model is a plain string rather than :data:`NoiseModel` for the reason
    :func:`solprop.fourier._check_kind` gives: this is the run-time guard for callers
    that ignore the annotation, and the test session's ``beartype`` hook would otherwise
    reject the bad value before the check below is reached.

    Parameters
    ----------
    noise : {None, 'input', 'modified'}
        Which model to use, or ``None`` for a noiseless run.
    grid : solprop.grid.TimeGrid
        The grid to draw on.
    seed : int or None
        Entropy of the generator; ``None`` draws from the operating system, and the
        entropy it drew is recorded on the realisation either way.
    n_modes : int, optional
        How many modes the state carries, so that every mode gets its own realisation.

    Returns
    -------
    solprop.noise.NoiseRealisation or None
        The realisation.

    Raises
    ------
    DriverError
        If the model is not one this driver knows.
    """
    if noise is None:
        return None
    if noise not in ("input", "modified"):
        msg = f"noise must be 'default', None, 'input' or 'modified', got {noise!r}"
        raise DriverError(msg)
    generator = np.random.default_rng(seed)
    draw = input_noise if noise == "input" else modified_noise
    return draw(grid, generator, n_modes=n_modes)


def _general_noise(
    noise: str | None,
    grid: TimeGrid,
    seed: int | None,
    guide: ModalGuide,
) -> NoiseRealisation | None:
    """Draw a general modal run's shot noise, in the basis it is defined in.

    The shot-noise draw (``docs/guide/media/noise.md``) is one photon per channel per
    bin, which is an energy-normalised true-power amplitude only in a basis orthonormal
    in true power. A guide in the provider's own coordinates **is** that basis, so
    this is :func:`_shot_noise` unchanged. A unitary change of basis draws
    **once, in the reference basis, with the same seed and the same stream**, and
    carries that one realisation into the run's coordinates through the inverse
    transform (:func:`solprop.noise.transformed_noise`): a second draw in the declared
    basis would be a different stochastic model, because a fixed-modulus random-phase
    draw is not invariant in distribution under unitary mixing.

    Parameters
    ----------
    noise : {None, 'input', 'modified'}
        Which model to use, or ``None`` for a noiseless run.
    grid : solprop.grid.TimeGrid
        The grid to draw on.
    seed : int or None
        Entropy of the generator.
    guide : solprop.modes.guide.ModalGuide
        The guide, whose sampling carries any declared change of basis.

    Returns
    -------
    solprop.noise.NoiseRealisation or None
        The realisation in the run's own channel coordinates.
    """
    drawn = _shot_noise(noise, grid, seed, n_modes=len(guide.mode_collection))
    sampling = guide.sampling
    if drawn is None or sampling is None or sampling.basis is None:
        return drawn
    return transformed_noise(drawn, sampling.basis.transform)


def _spatial_shot_noise(
    noise: NoiseModel, flux: SpatialFlux, n_pol: int, seed: int | None
) -> NoiseRealisation:
    """Draw a radial or Cartesian run's shot noise at its entrance plane.

    Parameters
    ----------
    noise : {'input', 'modified'}
        Which model to use.
    flux : solprop.spatial.SpatialFlux
        The entrance plane's measure.
    n_pol : int
        Independent polarisation components.
    seed : int or None
        Entropy of the generator; ``None`` draws from the operating system.

    Returns
    -------
    solprop.noise.NoiseRealisation
        The realisation (:func:`solprop.noise.spatial_noise`).
    """
    return spatial_noise(flux, np.random.default_rng(seed), model=noise, n_pol=n_pol)


def _spatial_draw(
    noise: NoiseModel, simulation: Simulation, medium: SpatialMedium
) -> Callable[[int | None], NoiseRealisation | None]:
    """Return the draw of a spatial run's shot noise, measured at its entrance plane.

    Parameters
    ----------
    noise : {'input', 'modified'}
        Which model to use.
    simulation : Simulation
        The run whose grids, entrance and polarisation the draw is on.
    medium : solprop.spatial.SpatialMedium
        The medium whose flux measures the draw.

    Returns
    -------
    collections.abc.Callable
        Draws the realisation from a seed (:func:`_spatial_shot_noise`).
    """
    flux = SpatialFlux.resolve(
        simulation._base_grid,
        cast("RadialGrid | FreeGrid | PlanarGrid", simulation.spatial_grid),
        medium,
        z=simulation.initial_z,
    )
    return functools.partial(
        _spatial_shot_noise, noise, flux, simulation._field0.shape[1]
    )


def _with_noise(
    problem: KernelProblem, noise: NoiseRealisation | None
) -> KernelProblem:
    """Return `problem` carrying `noise`, every other leaf shared by identity.

    A realisation reaches a translation only through the initial record and the
    ``has_noise`` flag; what else consults it is a refusal keyed on the model, and a
    replacement that changes the model is checked by its caller.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    noise : solprop.noise.NoiseRealisation or None
        The realisation in true sqrt(W) on a modal route, or ``None`` for no noise.

    Returns
    -------
    solprop.translate.problem.KernelProblem
        The problem with the realisation in its internal, vacuum-normalised amplitude.
    """
    noise0 = None if noise is None else noise_spectrum(noise, problem.shape)
    if noise0 is not None and problem.transform.geometry is None:
        noise0 = power_amplitude(
            noise0, power_weight_at(problem, problem.initial.z), inverse=True
        )
    return dataclasses.replace(
        problem,
        initial=dataclasses.replace(
            problem.initial,
            noise0=noise0,
            noise_model="none" if noise is None else noise.model,
        ),
        flags=dataclasses.replace(problem.flags, has_noise=noise0 is not None),
    )


def _redrawable(
    simulation: Simulation,
    draw: Callable[[int | None], NoiseRealisation | None],
    seed: int | None,
) -> Simulation:
    """Let a realisation drawn without a seed be drawn again from a chosen one.

    A call with ``seed=None`` records the operating-system entropy it drew, so the
    same call made twice describes two runs. Keeping the draw lets a resume take the
    seed its file recorded and a scan point take its own (:meth:`Simulation._reseed`);
    a seeded realisation keeps the one it was given.

    Parameters
    ----------
    simulation : Simulation
        The run a constructor built, carrying `draw`'s realisation.
    draw : collections.abc.Callable
        Draws the realisation from a seed.
    seed : int or None
        The seed the caller gave.

    Returns
    -------
    Simulation
        `simulation`, able to redraw its noise when `seed` was ``None``.
    """
    if seed is None and simulation.noise is not None:
        simulation._noise_redraw = draw
    return simulation


def default_radial_order(n_modes: int) -> int:
    """Return the radial order of the fixed rule a capillary run uses by default.

    The rule has to integrate the *quartic* of the mode profiles, because the Kerr
    polarisation is cubic in the field and its projection multiplies by one more
    profile. Those are Bessel functions of ``r/a``, so a Gauss rule converges spectrally
    and the only question is where. Two measurements set the number: the quartic overlap
    tensor of four HE_1m modes is converged to 2e-14 at ``n_r = 32``
    (``docs/guide/waveguides/mode-sets.md``), and the embedded error estimate for three
    modes falls 1.6e-1, 2.1e-2, 4.1e-5, 7.2e-9, 1.2e-15 at ``n_r = 8, 16, 24, 32, 48``.
    Thus 32 clears the alarm by two orders on three modes and is the reference's
    rounding on four. Eight nodes per mode carries that scaling to larger sets,
    since the highest radial order in the quartic grows with the set.

    A flat, larger order whatever the mode count would be safer and slower; 32 nodes on
    the sets a capillary run actually uses, with the alarm watching, is the trade this
    makes. The number rests on these two measurements and has not been calibrated
    further.

    Parameters
    ----------
    n_modes : int
        How many modes the set carries, >= 1.

    Returns
    -------
    int
        Radial nodes asked of
        :func:`solprop.modes.quadrature.transverse_quadrature`, which with
        ``radial_rule="gauss_kronrod"`` rounds an even count up by one so that the
        rule is the Kronrod extension of the half-order Gauss rule.

    Examples
    --------
    >>> from solprop.driver.simulation import default_radial_order
    >>> default_radial_order(1), default_radial_order(4), default_radial_order(8)
    (32, 32, 64)
    """
    return max(_MINIMUM_RADIAL_ORDER, _RADIAL_NODES_PER_MODE * n_modes)


def default_field_range(peak_field: float) -> tuple[float, float]:
    """Return the field amplitudes an ionisation table covers, from the input's peak.

    The table is built in ``log`` field and its knots double until it is converged
    (:func:`solprop.translate.ionisation.build_rate_table`), so a generous range costs
    little; getting it wrong is what costs. Below ``field_min`` the rate is taken as
    zero, which is right because every rate this package carries is exactly zero below
    its threshold, and the threshold is where the lower limit ends up anyway. Above
    ``field_max`` the table *holds its last value*, so an upper limit that is too low
    is a silent clip of the rate exactly where ionisation matters most.

    Hence three decades below the input peak and thirty times above it. The lower
    factor only has to reach under the rate's threshold, which for the noble gases at
    the intensities a capillary reaches sits one to two decades below the peak. The
    upper factor has to clear the self-compressed peak: a capillary compressing a pump
    to a few optical cycles raises the on-axis field by a small single-digit factor
    before ionisation-induced defocusing and loss clamp it, so thirty is roughly an
    order of margin on that -- a thousand times in intensity.

    Parameters
    ----------
    peak_field : float
        Peak field amplitude of the input, V/m, > 0: on a capillary the largest
        over the axis and the transverse nodes.

    Returns
    -------
    tuple of float
        Lower and upper field amplitude in V/m.

    Raises
    ------
    DriverError
        If the peak field is not positive, which means the input is empty.

    Examples
    --------
    >>> from solprop.driver.simulation import default_field_range
    >>> low, high = default_field_range(1e10)
    >>> f"{low:.3g} {high:.3g}"
    '1e+07 3e+11'
    """
    if not peak_field > 0.0:
        msg = (
            f"the input carries no field (peak {peak_field:.3e} V/m), so "
            "the ionisation table has no range to cover; give "
            "ionisation_field_range explicitly"
        )
        raise DriverError(msg)
    return (peak_field * _FIELD_RANGE_BELOW, peak_field * _FIELD_RANGE_ABOVE)


def _capillary_grid(  # noqa: PLR0913 - the three grid keywords and the two switches
    grid: TimeGrid | None,
    length: float,
    wavelength: float,
    wavelength_limits: tuple[float, float] | None,
    trange: float | None,
    *,
    envelope: bool,
    thg: bool,
) -> TimeGrid:
    """Return the grid a capillary run propagates on.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid or None
        A grid of one's own, which wins over the three keywords.
    length : float
        Capillary length, m.
    wavelength : float
        Reference wavelength, m.
    wavelength_limits : tuple of float or None
        Window limits, m.
    trange : float or None
        Requested time window, s.
    envelope : bool
        Whether to build an envelope grid rather than a field grid.
    thg : bool
        Whether the responses reach the third harmonic, which sets an envelope grid's
        oversampling.

    Returns
    -------
    solprop.grid.TimeGrid
        The grid.

    Raises
    ------
    DriverError
        If a grid is given beside the keywords it replaces, or neither is complete.
    """
    if grid is not None:
        if wavelength_limits is not None or trange is not None:
            msg = (
                "give a grid or the wavelength_limits and trange to build one, not "
                "both: the keywords would be ignored"
            )
            raise DriverError(msg)
        return grid
    if wavelength_limits is None or trange is None:
        msg = (
            "a capillary run needs wavelength_limits and trange to build its grid, "
            "or a grid of your own"
        )
        raise DriverError(msg)
    if envelope:
        return EnvelopeGrid.from_limits(
            length, wavelength, wavelength_limits, trange, thg=thg
        )
    return FieldGrid.from_limits(length, wavelength, wavelength_limits, trange)


def _capillary_modes(  # noqa: PLR0913 - capillary geometry and material keywords
    modes: int | Sequence[Mode],
    radius: float | ZProfile,
    mixture: Mixture,
    *,
    length: float,
    lattice: TubeLattice | None = None,
    model: MarcatiliModel,
    loss: LossSpec,
    cladding: GlassIndex | complex | None,
) -> ModeCollection:
    """Return the mode set under the declared capillary material.

    Parameters
    ----------
    modes : int or sequence of Mode
        Number of HE_1m modes, or a supplied set of capillary geometries.
    radius : float or ZProfile
        Core radius in m, used when constructing a numbered mode set.
    mixture : Mixture
        Ordered species and their density paths.
    length : float
        The run's length in m, over which a supplied set's radius is compared.
    lattice : TubeLattice or None, optional
        Tube geometry selecting Vincetti modes for an integer count.
    model : {'full', 'reduced'}
        Marcatili model for a numbered mode set.
    loss : bool, float or callable
        Total loss policy for a numbered mode set.
    cladding : GlassIndex, complex or None
        Cladding index for a numbered mode set.

    Returns
    -------
    ModeCollection
        Validated geometries with the same material declaration.

    Raises
    ------
    DriverError
        If the mode count is invalid, a mode cannot carry a mixture, a supplied set
        is not orthonormal or disagrees with a declared radius, model, loss or
        cladding, or a lattice is given beside supplied modes or the reduced model.
    """
    if isinstance(modes, int) and modes < 1:
        raise DriverError(f"a capillary mode set needs at least one mode, got {modes}")
    if lattice is not None:
        if not isinstance(modes, int):
            raise DriverError(
                "lattice is refused: the given modes carry their own geometry"
            )
        if model == "reduced":
            raise DriverError('lattice is incompatible with model="reduced"')
        return ModeCollection.vincetti(
            radius,
            count=modes,
            mixture=mixture,
            lattice=lattice,
            loss=loss,
            cladding=cladding,
        )
    if isinstance(modes, int):
        return ModeCollection(
            tuple(
                MarcatiliMode(
                    radius,
                    mixture=mixture,
                    m=index + 1,
                    model=model,
                    loss=loss,
                    cladding=cladding,
                )
                for index in range(modes)
            )
        )
    if not modes:
        # The capillary wording, which the int branch above already uses: the
        # waveguide entry has its own.
        raise DriverError("a capillary mode set needs at least one mode")
    given = _given_gas_modes(modes, mixture)
    _refuse_ignored_geometry(
        given, radius, length=length, model=model, loss=loss, cladding=cladding
    )
    return given


# Two radius paths agree when they agree to this relative tolerance at every sampled
# position: a supplied set and a declared radius built from the same number or the
# same profile agree exactly, so this only forgives the last bits of a radius
# computed two ways, and a real disagreement is a percent or more.
_RADIUS_RTOL = 1e-12


def _refuse_ignored_geometry(
    modes: Sequence[Mode],
    radius: float | ZProfile,
    *,
    length: float,
    model: MarcatiliModel,
    loss: LossSpec,
    cladding: GlassIndex | complex | None,
) -> None:
    """Refuse, by name, a declared capillary argument a supplied mode set ignores.

    A supplied set is the run's geometry: its modes keep their own radius, model, loss
    and cladding and are only rebound to the declared fill, so an argument that
    disagrees with them would be silently discarded. The radius is always declared -- it
    is positional -- and is compared with each mode's own ``core_radius`` path over the
    run, breakpoints included, so a taper beside a constant bore, or a different bore,
    is named. `model`, `loss` and `cladding` are compared only when they differ from
    their defaults, which is the only way this layer can tell a declaration from an
    omission: a default beside a mode that says otherwise was not asked for.

    Parameters
    ----------
    modes : sequence of Mode
        The supplied set, non-empty.
    radius : float or ZProfile
        The declared core radius, m.
    length : float
        The run's length, m, over which the radius paths are compared.
    model : {'full', 'reduced'}
        The declared Marcatili model.
    loss : bool, float or callable
        The declared loss policy.
    cladding : GlassIndex, complex or None
        The declared cladding index.

    Raises
    ------
    DriverError
        Naming the argument and the first mode it disagrees with.
    """
    declared = as_profile(radius)
    for index, mode in enumerate(modes):
        own = getattr(mode, "core_radius", None)
        if own is None:
            msg = (
                "from_capillary's radius is the capillary's bore, and supplied mode "
                f"{index} ({type(mode).__name__}) declares no core radius to compare "
                "it with, so the radius would be ignored: propagate a mode set that "
                "is not a capillary's with from_waveguide or prop_waveguide"
            )
            raise DriverError(msg)
        if not _same_radius(declared, own, length):
            msg = (
                f"from_capillary's radius disagrees with supplied mode {index} "
                f"({mode.channel_label.display_label}): a supplied mode set is the "
                "run's geometry and keeps its own core radius, so the declared one "
                "would be ignored. Declare the modes' own radius (the same number or "
                "profile) or build the modes on the radius wanted"
            )
            raise DriverError(msg)
        _refuse_ignored_setting(index, mode, "model", model, default="full")
        _refuse_ignored_setting(index, mode, "loss", loss, default=True)
        _refuse_ignored_setting(index, mode, "cladding", cladding, default=None)


def _refuse_ignored_setting(
    index: int, mode: Mode, name: str, declared: object, *, default: object
) -> None:
    """Refuse a non-default capillary setting that a supplied mode does not carry.

    Parameters
    ----------
    index : int
        The mode's position in the supplied set.
    mode : Mode
        The supplied mode.
    name : str
        The keyword, which is also the mode's attribute: ``"model"``, ``"loss"`` or
        ``"cladding"``.
    declared : object
        The value the call declared.
    default : object
        The keyword's default, which is never refused.

    Raises
    ------
    DriverError
        If `declared` is not the default and is not the mode's own value.
    """
    if declared is default or declared == default:
        return
    own = getattr(mode, name, None)
    same = own is declared or (
        not callable(own) and not callable(declared) and own == declared
    )
    if not same:
        msg = (
            f"from_capillary's {name}={declared!r} disagrees with supplied mode "
            f"{index}, which carries {name}={own!r}: a supplied mode set keeps its "
            f"own {name}, so the declaration would be ignored. Declare it on the "
            f"modes, or leave {name} at its default beside them"
        )
        raise DriverError(msg)


def _same_radius(declared: ZProfile, own: ZProfile, length: float) -> bool:
    """Return whether two radius paths agree over a run of `length`.

    Parameters
    ----------
    declared : ZProfile
        The declared core radius, m.
    own : ZProfile
        A supplied mode's core radius, m.
    length : float
        The run's length, m.

    Returns
    -------
    bool
        Whether they agree to ``_RADIUS_RTOL`` at 65 equally spaced positions, both
        paths' breakpoints and the midpoints between all of them.
    """
    if declared is own:
        return True
    start = max(0.0, declared.z_span[0], own.z_span[0])
    end = min(length, declared.z_span[1], own.z_span[1])
    edges = np.unique(
        np.concatenate(
            [np.linspace(start, end, 65), declared.breakpoints, own.breakpoints]
        ).clip(start, end)
    )
    positions = np.unique(np.concatenate([edges, 0.5 * (edges[:-1] + edges[1:])]))
    return bool(
        np.allclose(own(positions), declared(positions), rtol=_RADIUS_RTOL, atol=0.0)
    )


def _given_gas_modes(modes: Sequence[Mode], mixture: Mixture) -> ModeCollection:
    """Bind supplied modes and retain the existing orthonormality gate.

    Parameters
    ----------
    modes : sequence of Mode
        Supplied geometry, in propagation row order.
    mixture : Mixture
        New ordered material populations.

    Returns
    -------
    ModeCollection
        Rebound orthonormal modes.
    """
    if not modes:
        raise DriverError("a waveguide mode set needs at least one mode")
    given = ModeCollection(tuple(_rebind_mode(mode, mixture) for mode in modes))
    if len(given) > 1:
        overlaps = given.orthonormality()
        worst = float(np.max(np.abs(overlaps - np.eye(len(given)))))
        if worst > ORTHONORMALITY_TOLERANCE:
            raise DriverError(
                f"the given modes are not an orthonormal set: the largest deviation "
                f"of the overlap matrix from the identity is {worst:.3e}, above "
                f"{ORTHONORMALITY_TOLERANCE:.0e}. One synthesis matrix and its dual "
                "represent an orthonormal basis and nothing else"
            )
    return given


_NODE_MAXIMUM_PLASMA = (
    "electron_density",
    "peak_ionisation_rate",
    "species_electron_density",
    "species_peak_ionisation_rate",
)
"""The plasma statistics read at a sampled node rather than on the axis."""

_SAMPLED_PEAK_THRESHOLD = 0.999
"""Innermost field scale below which a node maximum warns (SEP 19, Q8).

At 0.999 the innermost node sees 99.8% of the axial intensity, so a rate of order 10
there is within 2% of its axial value; every Gauss-Kronrod and Gauss rule in r/a of
eight or more nodes is closer still."""


def _warn_sampled_peak(rule: TransverseQuadrature | None, spec: StatsSpec) -> None:
    """Warn once per run when a node maximum is read far from the axis (SEP 19, Q8).

    A field-scale Gauss rule of few nodes integrates the source but has no node near
    the axis, so a node-maximum plasma statistic samples the axial peak at its
    innermost node, which a steep rate under-reports. The value is unchanged; the
    warning names the node and the on-axis statistics that read the axis.

    Parameters
    ----------
    rule : solprop.modes.quadrature.TransverseQuadrature or None
        The run's transverse rule.
    spec : solprop.stats.StatsSpec
        The statistics the run records.

    Warns
    -----
    SampledPeakWarning
        If a node-maximum plasma statistic is requested and the rule's innermost
        node has a field scale below :data:`_SAMPLED_PEAK_THRESHOLD`.
    """
    if rule is None or rule.radial_rule != "field_scale":
        return
    asked = sorted(set(_NODE_MAXIMUM_PLASMA) & {*spec.names, *spec.periodic})
    if not asked:
        return
    eta = float(j0(jn_zeros(0, 1)[0] * float(np.min(rule.xi_r))))
    if eta < _SAMPLED_PEAK_THRESHOLD:
        warnings.warn(
            f"the field-scale rule's innermost node is at eta = {eta:.3f}, "
            f"{eta**2:.1%} of the axial intensity, so {asked} are node maxima "
            f"there: a rate of order K reads about eta^(2K) of its axial value. "
            "Request the on_axis_* statistics for the axis, or use the Radau rule, "
            "whose first node is on it",
            SampledPeakWarning,
            stacklevel=3,
        )


def _capillary_quadrature(
    modes: ModeCollection,
    n_r: int | None,
    n_theta: int | None,
    *,
    radial_rule: RadialRule,
) -> TransverseQuadrature:
    """Return the fixed transverse rule of a capillary run.

    Parameters
    ----------
    modes : solprop.modes.collection.ModeCollection
        The mode set, whose azimuthal order decides whether the angle is resolved.
    n_r : int or None
        Radial nodes; ``None`` takes the module's
        :func:`~solprop.driver.simulation.default_radial_order`.
    n_theta : int or None
        Azimuthal nodes; ``None`` takes the rule's own default where one is needed.
    radial_rule : str
        The radial rule family, one of :data:`solprop.modes.quadrature.RadialRule`.

    Returns
    -------
    solprop.modes.quadrature.TransverseQuadrature
        The rule.

    Raises
    ------
    DriverError
        If azimuthal nodes are asked for by a set that has no angular structure to
        resolve, where they would multiply the cost and change nothing, or y nodes
        by a slab set, which is uniform in y.
    """
    if modes.modes[0].dimlimits(0.0)[0] == "slab":
        if n_theta is not None:
            msg = (
                f"n_theta={n_theta} was given for a slab set: the planar guide is "
                "uniform in y, so its rule has one y node of unit width and n_r "
                "alone is the x node count"
            )
            raise DriverError(msg)
        # The measured bound of the x rule's quartic products is the Cartesian
        # per-axis one; the default takes whichever is larger.
        needed = cartesian_nodes(max(mode.azimuthal_order for mode in modes.modes))
        order = max(default_radial_order(len(modes)), needed) if n_r is None else n_r
        return transverse_quadrature(modes.modes, n_r=order, radial_rule=radial_rule)
    if n_theta is not None and not modes.needs_full_rule:
        msg = (
            f"n_theta={n_theta} was given for a set whose modes all have azimuthal "
            "order zero: |J_0(u r/a)| does not depend on the angle, so the rule is "
            "radial and every azimuthal node is a copy of the same integrand"
        )
        raise DriverError(msg)
    order = default_radial_order(len(modes)) if n_r is None else n_r
    if modes.modes[0].dimlimits(0.0)[0] == "cartesian":
        # For a Cartesian set n_r and n_theta are the x and y node counts, so
        # both defaults take the measured aliasing bound of the set's highest
        # transverse order rather than the radial and azimuthal defaults, which
        # under-resolve y from order two upwards.
        needed = cartesian_nodes(max(mode.azimuthal_order for mode in modes.modes))
        return transverse_quadrature(
            modes.modes,
            n_r=max(order, needed) if n_r is None else n_r,
            n_theta=needed if n_theta is None else n_theta,
            radial_rule=radial_rule,
        )
    if n_theta is None:
        return transverse_quadrature(modes.modes, n_r=order, radial_rule=radial_rule)
    return transverse_quadrature(
        modes.modes, n_r=order, n_theta=n_theta, radial_rule=radial_rule
    )


def _no_quadrature(
    modes: ModeCollection,
    n_r: int | None,
    n_theta: int | None,
    *,
    radial_rule: RadialRule,
) -> None:
    """Check a mode-averaged hollow run and return its absent transverse rule.

    A mode-averaged run carries one mode as a uniform disc of its effective area, the
    one-node transform (:func:`solprop.translate.modal.mode_averaged_transform`), so it
    has no rule for the keywords that shape one to act on.

    Parameters
    ----------
    modes : solprop.modes.collection.ModeCollection
        The mode set, which must hold exactly one mode.
    n_r : int or None
        Radial nodes, which must not be given.
    n_theta : int or None
        Azimuthal nodes, which must not be given.
    radial_rule : str
        The radial rule family, which must be the default.

    Raises
    ------
    DriverError
        If the set has more than one mode or a rule keyword is given.
    """
    if len(modes) != 1:
        msg = (
            f"mode_average=True needs exactly one mode, and the set has {len(modes)}: "
            "energy moving between modes is what a mode-averaged run cannot carry "
            "(docs/guide/waveguides/mode-averaging.md)"
        )
        raise DriverError(msg)
    given = [
        name
        for name, value, unset in (
            ("n_r", n_r, None),
            ("n_theta", n_theta, None),
            ("radial_rule", radial_rule, "gauss_kronrod"),
        )
        if value != unset
    ]
    if given:
        msg = (
            f"{', '.join(given)} given beside mode_average=True: they shape a "
            "transverse rule, and a mode-averaged run has none, its mode being a "
            "uniform disc of the effective area"
        )
        raise DriverError(msg)


def _capillary_responses(  # noqa: PLR0913 - the switches of the responses, one each
    gas: Gas | str,
    material: Species,
    *,
    kerr: bool,
    plasma: bool | IonisationRate,
    preionisation: float | ZProfile,
    integration: IntegrationPolicy | None,
    wavelength: float,
    envelope: bool,
    electronic: KerrResponse | DispersiveKerrResponse | None,
    selection: MaterialSelection,
) -> list[object]:
    """Return the Kerr and plasma responses of a single-gas capillary run.

    The Raman response, optmat's, is the fill's (:func:`_optmat_raman`).

    Parameters
    ----------
    gas : solprop.physdata.Gas or str
        Filling species, by name, for the rate.
    material : solprop.physdata.species.Species
        The fill's species as resolved under the run's selection, which the plasma
        response carries: it is never rebuilt from the name with the default
        selection, so the material manifest lists only the models that acted (Q45).
    kerr : bool
        Include the Kerr response.
    plasma : bool or IonisationRate
        Include photoionisation, and with which rate.
    preionisation : float or ZProfile
        Fraction already ionised before the pulse, a number or a path along z in m.
    integration : {'band', 'trapezoid'} or None
        The field plasma column's running-integral rule; ``None`` takes the
        response's default.
    wavelength : float
        Reference wavelength, m, the rate's carrier unless the selection names one.
    envelope : bool
        Select cycle-averaged envelope plasma instead of real-field plasma.
    electronic : KerrResponse or DispersiveKerrResponse or None
        The gas's electronic response under the selection's policy
        (:func:`solprop.nonlinear.mixtures.electronic_response`); ``None`` without
        one.
    selection : MaterialSelection
        The run's material selection, under which ``plasma=True`` takes optmat's
        rate.

    Returns
    -------
    list
        The responses, Kerr first.

    Raises
    ------
    DriverError
        If a run is asked for with no nonlinearity at all.
    """
    responses: list[object] = []
    if kerr:
        # Per particle: the density path supplies the rest (ADR-026,
        # docs/dev/decisions.md).
        assert electronic is not None
        responses.append(electronic)
    if plasma is not False:
        rate = _ionisation_rate(gas, plasma, wavelength, selection)
        if envelope:
            responses.append(
                EnvelopePlasmaResponse(material, rate, preionisation=preionisation)
            )
        else:
            chosen: dict[str, IntegrationPolicy] = (
                {} if integration is None else {"integration": integration}
            )
            responses.append(
                PlasmaResponse(material, rate, preionisation=preionisation, **chosen)
            )
    if not responses:
        msg = (
            "a capillary run with neither kerr nor plasma has no nonlinearity; use "
            "the linear propagator instead of a solver"
        )
        raise DriverError(msg)
    return responses


def _default_raman(mixture: Mixture, *, provisional: bool = False) -> bool:
    """Decide what ``raman=None`` means for a fill.

    On when any component carries a Raman response: a built-in gas whose optmat
    response is curated (:func:`solprop.nonlinear.optmat_raman.raman_status`), or a
    custom species carrying a model with at least one term. An atom contributes
    nothing and says nothing. A molecule whose optmat response is provisional (N2O's
    bands, SEP 3, D13) is off and warns rather than silently going without: a user
    expecting every molecule's Raman response would otherwise not learn that nitrous
    oxide is the one curated only provisionally. Under
    ``MaterialSelection(raman="provisional")`` it takes that response instead (Q17),
    which warns once per run with its status and is recorded.

    Parameters
    ----------
    mixture : solprop.physdata.mixtures.Mixture
        The resolved fill.
    provisional : bool, optional
        Whether the run's selection admits optmat's provisional Raman responses.

    Returns
    -------
    bool
        Whether the run carries a Raman response by default.
    """
    on = False
    for component in mixture.components:
        material = component.species
        if not built_in_population(material):
            model = material.raman
            on = on or (
                model is not None
                and (model.rotational is not None or model.vibrational is not None)
            )
            continue
        status = raman_status(material.name)
        if status == "validated" or (status == "provisional" and provisional):
            on = True
        elif status == "provisional":
            warnings.warn(
                f"{component.label} ({material.name}) is a molecule whose Raman "
                "response optmat has only provisionally, so it is off; pass "
                "raman=False to say so, or MaterialSelection(raman='provisional') "
                "to admit it",
                SolpropWarning,
                stacklevel=3,
            )
    return on


def _optmat_raman(
    gas: Gas | str,
    mixture: Mixture,
    selection: MaterialSelection,
    thg: bool,
    wavelength: float,
    length: float,
) -> list[object]:
    """Return a single-gas fill's Raman response from optmat (SEP 3).

    Parameters
    ----------
    gas : solprop.physdata.Gas or str
        Filling species, for the message.
    mixture : Mixture
        The fill, one optmat-resolved population.
    selection : MaterialSelection
        The run's selection.
    thg : bool
        Whether the field grid keeps the third harmonic.
    wavelength : float
        The grid reference wavelength in m, the carrier unless the selection names one.
    length : float
        The run's length in m.

    Returns
    -------
    list
        The response, or nothing for a fill that is vacuum along the whole run.

    Raises
    ------
    MaterialError
        If the species has no molecular Raman response (a noble gas has no
        Raman-active mode: call with ``raman=False``), or optmat refuses it.
    """
    logger.debug("optmat Raman response of the %s fill", gas)
    response = mixture_raman(
        mixture,
        True,
        selection=selection,
        thg=thg,
        wavelength=wavelength,
        length=length,
        explicit=True,
    )
    return [] if response is None else [response]


def _ionisation_rate(
    gas: Gas | str,
    plasma: bool | IonisationRate,
    wavelength: float,
    selection: MaterialSelection,
) -> IonisationRate:
    """Return the ionisation rate a capillary run uses.

    ``plasma=True`` takes optmat's rate model
    (:func:`solprop.nonlinear.optmat_rates.selected_rate`, SEP 3, D1); an explicit
    rate object passes through unchanged. The models' physical limits are discussed
    in ``docs/guide/media/ionisation.md``.

    Parameters
    ----------
    gas : solprop.physdata.Gas or str
        Filling species.
    plasma : bool or IonisationRate
        ``True`` for the default rate of the species, or the rate itself.
    wavelength : float
        Reference wavelength, m, the rate's carrier unless the selection names one.
    selection : MaterialSelection
        The run's material selection.

    Returns
    -------
    solprop.nonlinear.ionisation.IonisationRate
        The rate.
    """
    if isinstance(plasma, bool):
        return selected_rate(species(gas, selection=selection), selection, wavelength)
    return plasma


def _with_integration_statistics(
    spec: StatsSpec, responses: Sequence[object]
) -> StatsSpec:
    """Add the plasma integration diagnostics to a default spec under the band rule.

    The undershoot, work defect and boundary level of SEP 18 are recorded by default
    wherever the field plasma column integrates by the band rule, so the run's warning
    can read them back; a run on the plain trapezoid records none of them.

    Parameters
    ----------
    spec : solprop.stats.StatsSpec
        The route's default statistics.
    responses : sequence of object
        The run's responses.

    Returns
    -------
    solprop.stats.StatsSpec
        `spec`, with the three names appended under the band rule.
    """
    if not any(
        isinstance(one, PlasmaResponse) and one.integration == "band"
        for one in responses
    ):
        return spec
    extra = tuple(
        name for name in PLASMA_INTEGRATION_STATISTICS if name not in spec.names
    )
    return dataclasses.replace(spec, names=(*spec.names, *extra))


@dataclasses.dataclass(frozen=True)
class _TDSEEntry:
    """Hold a TDSE capillary run's resolved response, resources and defaults.

    Attributes
    ----------
    response : TDSEGasResponse
        The declared response, its atomic window's duration resolved.
    execution : TDSEExecution
        Atomic runtime resources.
    backend : str
        The resolved backend, ``numpy_ref``.
    step_planes : int
        Resolved natural-step save interval.
    n_save : int or None
        Resolved requested-plane count; 2 (the entrance and the end) when neither
        `save_planes` nor `n_save` was given.
    record : dict
        The resolved defaults, recorded in the run's configuration.
    """

    response: TDSEGasResponse
    execution: TDSEExecution
    backend: str
    step_planes: int
    n_save: int | None
    record: dict[str, JsonValue]


def _tdse_entry(  # noqa: PLR0913 - the entry point's TDSE keywords together
    tdse: TDSEGasResponse | None,
    execution: TDSEExecution | None,
    *,
    trange: float | None,
    kerr: bool,
    mode_average: bool,
    unset: Mapping[str, object],
    backend: str,
    step_planes: int | None,
    save_planes: Sequence[float] | np.ndarray | None,
    n_save: int | None,
) -> _TDSEEntry | None:
    """Resolve the ``tdse=`` capillary keywords (SEP 20 sections 5.2 and 5.8).

    Parameters
    ----------
    tdse : TDSEGasResponse or None
        The declared response, or None for a rate-based run.
    execution : TDSEExecution or None
        Its resources.
    trange : float or None
        Requested time window in s, the default atomic-window duration.
    kerr : bool
        The Kerr switch, which must stay true beside a TDSE response.
    mode_average : bool
        The mode-averaged switch, which must stay false beside a TDSE response: its
        translation needs the resolved radial quadrature.
    unset : mapping
        Electronic-response keywords that must be left unset beside it.
    backend : str
        Requested backend; ``auto`` resolves to ``numpy_ref``.
    step_planes : int or None
        Requested natural-step interval.
    save_planes : sequence of float or None
        Requested save positions in m.
    n_save : int or None
        Requested plane count.

    Returns
    -------
    _TDSEEntry or None
        The resolution, or None without `tdse`.

    Raises
    ------
    DriverError
        For `tdse_execution` without `tdse`, an electronic keyword or `mode_average`
        beside it, an
        unresolvable window duration, or a non-TDSE record.
    """
    if tdse is None:
        if execution is not None:
            raise DriverError("tdse_execution requires a TDSEGasResponse")
        return None
    if not isinstance(tdse, TDSEGasResponse):
        raise DriverError("tdse must be a TDSEGasResponse")
    given = [name for name, value in unset.items() if value is not None]
    if not kerr:
        given.append("kerr=False")
    if mode_average:
        given.append("mode_average=True")
    if given:
        raise DriverError(
            f"the TDSE response owns the electronic response; leave "
            f"{', '.join(given)} unset beside tdse="
        )
    response = tdse
    window = tdse.atomic_window
    duration_source = None
    if isinstance(window, TDSEAtomicWindow):
        duration_source = "explicit"
        if window.duration is None:
            if trange is None:
                raise DriverError(
                    "the atomic window's duration is unresolved: give trange or "
                    "TDSEAtomicWindow(duration=...)"
                )
            response = dataclasses.replace(
                tdse, atomic_window=dataclasses.replace(window, duration=trange)
            )
            duration_source = "trange"
    resolved_backend = "numpy_ref" if backend == "auto" else backend
    if backend == "auto":
        logger.info("backend='auto' resolves to numpy_ref for a TDSE response")
    final_only = save_planes is None and n_save is None
    resolved_steps = 1 if step_planes is None else step_planes
    record: dict[str, JsonValue] = {
        "model": tdse.model,
        "backend": {"requested": backend, "resolved": resolved_backend},
        "step_planes": {"requested": step_planes, "resolved": resolved_steps},
        "requested_planes": "final_only" if final_only else "explicit",
        "atomic_window_duration": duration_source,
        "execution": "default" if execution is None else "explicit",
    }
    return _TDSEEntry(
        response=response,
        execution=TDSEExecution() if execution is None else execution,
        backend=resolved_backend,
        step_planes=resolved_steps,
        n_save=2 if final_only else n_save,
        record=record,
    )


def _entry_overrides[P, R](
    atomic: _TDSEEntry | None,
    plasma: P,
    raman: R,
    backend: str,
    n_save: int | None,
    step_planes: int | None,
) -> tuple[_TDSEEntry | None, P, R, str, int | None, int]:
    """Return a TDSE entry with the capillary settings it overrides, or the originals.

    Parameters
    ----------
    atomic : _TDSEEntry or None
        The resolved TDSE entry.
    plasma, raman : object
        The rate-based switches; off beside a TDSE response, which owns them.
    backend : str
        Requested backend.
    n_save : int or None
        Requested plane count.
    step_planes : int or None
        Requested natural-step interval; ``None`` is 0 without TDSE.

    Returns
    -------
    tuple
        The entry, the plasma and Raman switches, the backend, the plane count and
        the step interval.
    """
    if atomic is None:
        steps = 0 if step_planes is None else step_planes
        return None, plasma, raman, backend, n_save, steps
    # False is a member of both switches' declared types.
    off = cast("P", False), cast("R", False)
    return atomic, *off, atomic.backend, atomic.n_save, atomic.step_planes


def _tdse_stats(spec: StatsSpec) -> StatsSpec:
    """Remove the statistics a TDSE response cannot supply from a default spec.

    Parameters
    ----------
    spec : solprop.stats.StatsSpec
        The route's default statistics.

    Returns
    -------
    solprop.stats.StatsSpec
        `spec` without :data:`TDSE_UNAVAILABLE_STATISTICS`.
    """
    return dataclasses.replace(
        spec,
        names=tuple(
            name for name in spec.names if name not in TDSE_UNAVAILABLE_STATISTICS
        ),
    )


def _warn_tdse_cost(quadrature: TransverseQuadrature | None) -> None:
    """Warn once when a TDSE run integrates over a many-node radial rule.

    Parameters
    ----------
    quadrature : TransverseQuadrature or None
        The run's transverse rule.
    """
    if quadrature is None or quadrature.radial_rule in (
        "field_scale",
        "field_scale_radau",
    ):
        return
    warnings.warn(
        f"the TDSE response runs on the {quadrature.radial_rule!r} radial rule with "
        f"{quadrature.n_nodes} nodes; each node is one atom in every atomic call, so "
        "the atomic cost is proportional to it. A field-scale rule "
        "(radial_rule='field_scale', a few nodes) integrates the HE11 response at a "
        "fraction of the cost (docs/guide/waveguides/radial-quadrature.md)",
        TDSECostWarning,
        stacklevel=3,
    )


def _capillary_stats(
    *, with_plasma: bool, mixture: Mixture, averaged: bool = False
) -> StatsSpec:
    """Return the statistics a capillary run records unless told otherwise.

    The multimode defaults with the z-profiles, plus every statistic only a backend can
    compute: it holds the nodes the plasma was evaluated at, the mode set the axis
    needs, and the embedded quadrature estimate a later pass cannot reconstruct.

    Parameters
    ----------
    with_plasma : bool
        Whether the run carries a plasma response, whose two statistics the backend
        refuses to compute without one.
    mixture : Mixture
        Ordered populations whose species statistics are included. Independent
        density paths have no total-pressure statistic.
    averaged : bool, optional
        Whether the run is mode-averaged: the mode-averaged defaults then replace the
        multimode ones, and the quadrature estimate, which has no rule to read, is
        left out.

    Returns
    -------
    solprop.stats.StatsSpec
        The spec, with no energy windows; add those for the bands of interest.
    """
    wanted = [
        name
        for name in BACKEND_STATISTICS
        if (with_plasma or name not in PLASMA_STATISTICS)
        and not (averaged and name == "quadrature_error")
    ]
    base = default_spec(
        "mode_averaged" if averaged else "multimode", with_profiles=True
    )
    wanted.extend(
        (
            "species_ionisation",
            "species_electron_density",
            "species_peak_ionisation_rate",
            "species_density",
            "species_ionisation_energy",
        )
    )
    if mixture.shared is None:
        base = dataclasses.replace(
            base, names=tuple(name for name in base.names if name != "pressure")
        )
    return dataclasses.replace(base, names=(*base.names, *wanted))


_GAS_KEYWORD_DEFAULTS: Mapping[str, object] = {
    "gas": None,
    "pressure": None,
    "pressure_floor": None,
    "vacuum_end": None,
    "susceptibility": None,
    "temperature": None,
    "kerr": True,
    "thg": None,
    "plasma": None,
    "preionisation": None,
    "plasma_integration": None,
    "raman": None,
    "mode_average": False,
    "n_r": None,
    "n_theta": None,
    "radial_rule": "gauss_kronrod",
    "ionisation_field_range": None,
    "polarisation": None,
    "envelope": False,
    "node_budget": DEFAULT_NODE_BUDGET,
    "taper_coupling": True,
}
"""Every gas-filled `from_waveguide` keyword, with the value that means "not given".

A general modal guide owns its materials, its responses, its transverse rule and
its grid kind, so each of these is refused by name when it departs from the value
here; a keyword silently ignored would let a caller believe a fill, a response or
a resolution was applied.
"""

_GAS_KEYWORD_REASONS: Mapping[str, str] = {
    "gas": "a guide's materials are its regions' own laws, not a fill",
    "pressure": "a guide's materials are its regions' own laws, not a fill",
    "pressure_floor": "a guide has no fill and so no pressure path",
    "vacuum_end": "a guide has no fill and so no gas flow to choke",
    "susceptibility": "a guide's dispersion is its provider's, solved from the "
    "regions' permittivity laws",
    "temperature": "a material region's law is a declared state, and a guide "
    "carries no gas whose density a temperature would set",
    "kerr": "a guide's Kerr response is its regions'; declare a region with no "
    "MaterialKerr to make it linear",
    "thg": "the general route's third harmonic is set by kerr_form=: the real "
    "cubic carries it and the analytic-signal form does not",
    "plasma": "a general guide carries the instantaneous Kerr term alone; "
    "ionisation is not implemented on the general route",
    "preionisation": "a general guide carries no plasma channel",
    "plasma_integration": "a general guide carries no plasma column to integrate",
    "raman": "a guide's delayed response is its material regions' own, the silica "
    "tensor Raman model",
    "mode_average": "a general guide's channels are resolved on its own node rule; "
    "mode averaging is the gas-filled route's",
    "n_r": "the transverse rule is the guide's own node rule, declared when the "
    "guide was built",
    "n_theta": "the transverse rule is the guide's own node rule, declared when "
    "the guide was built",
    "radial_rule": "the transverse rule is the guide's own node rule, declared when "
    "the guide was built",
    "taper_coupling": "a guide that moves along z carries its own connection in the "
    "general route's flow; the opt-out is the capillary route's",
    "ionisation_field_range": "a general guide carries no ionisation table",
    "polarisation": "a general guide puts every physical channel, including each "
    "parity partner, in its own row; launch into them by name with channel=, or "
    "build the launch yourself and pass it to Simulation.with_input",
    "envelope": "the general route synthesises an analytic field from a "
    "carrier-resolved spectrum and propagates on a field grid",
    "node_budget": "the general route sizes its node chunk from its own memory "
    "ledger, which counts different buffers",
}
"""Why each gas-filled keyword cannot be carried by a general modal guide."""


def _means_not_given(value: object, default: object) -> bool:
    """Return whether a supplied keyword carries the value that means "not given".

    Parameters
    ----------
    value : object
        What the caller passed.
    default : object
        The table's entry: the signature's own default.

    Returns
    -------
    bool
        True when the keyword was, in effect, not given.
    """
    if value is default:
        return True
    if isinstance(value, np.ndarray) and value.ndim > 0:
        # `array == default` is elementwise and its truth value raises; a keyword
        # whose "not given" value is a scalar or None is never an array anyway, so
        # an array is always something the caller meant.
        return False
    return bool(value == default)


def _refuse_gas_keywords(supplied: Mapping[str, object]) -> None:
    """Refuse, by name, every gas-filled keyword given beside a provider.

    Parameters
    ----------
    supplied : collections.abc.Mapping
        The keywords :meth:`Simulation.from_waveguide` collected, by name.

    Raises
    ------
    DriverError
        Naming the first keyword that departs from its "not given" value, and why
        a general modal guide cannot carry it.
    """
    for name, value in supplied.items():
        # Indexed, not `.get`: a keyword this table does not know is a keyword
        # `from_waveguide` would silently ignore on this route, and
        # `test_from_waveguide_refuses_or_names_every_keyword_it_takes` locks the
        # two together so that adding one cannot pass unnoticed.
        default = _GAS_KEYWORD_DEFAULTS[name]
        if _means_not_given(value, default):
            continue
        msg = (
            f"{name}={value!r} beside a provider=: {_GAS_KEYWORD_REASONS[name]}. "
            "The general modal route takes its physics from the guide "
            "(docs/guide/waveguides/general-modal.md)"
        )
        raise DriverError(msg)


def _general_stats() -> StatsSpec:
    """Return the statistics a general modal run records by default.

    The route's **own** names, not the legacy ones: ``physical_energy`` is the
    metric-weighted energy through the plane and ``analytic_flux`` the peak total
    analytic Poynting flux, both scalars, because a coordinate energy and a
    coordinate power are not additive on a basis that is not orthonormal
    (``docs/guide/results/statistics.md``). ``energy``, ``peak_power``,
    ``peak_power_total`` and ``peak_intensity`` are refused by name on this basis
    rather than redefined. ``omega0``,
    ``fwhm_t`` and ``fwhm_t_outer`` stay: they are functionals of the
    same-frequency flux ``a.H W a`` and take the basis metric
    (:data:`solprop.stats.METRIC_WEIGHTED_STATISTICS`), so they are exact on a
    declared basis as well as on the identity one.
    ``quadrature_error`` is not here either: a provider's node rule has
    no embedded coarse pair to difference.

    Returns
    -------
    solprop.stats.StatsSpec
        The default specification.
    """
    return StatsSpec(
        names=("omega0", "physical_energy", "analytic_flux", "fwhm_t", "fwhm_t_outer")
    )


def _in_channel(
    built: Complex[np.ndarray, "n_modes_in n_pol n_omega"],
    guide: ModalGuide,
    channel: int | str | None,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Place a one-channel pulse in a named or indexed channel of a guide.

    **The amplitude is placed unchanged.** A provider normalises every channel to
    unit longitudinal Poynting flux, so a channel amplitude is already a true-power
    amplitude and the launched energy is the pulse's; there is no ``sqrt(Re n_eff)``
    boundary on this route and nothing here crosses one.

    Parameters
    ----------
    built : numpy.ndarray
        The pulse, shape ``(1, n_pol, n_omega)`` in the square root of watts.
    guide : solprop.modes.guide.ModalGuide
        The guide, whose channel labels name the rows.
    channel : int or str or None
        Row index, display label, or ``None`` for the first channel.

    Returns
    -------
    numpy.ndarray
        The state, shape ``(n_modes, n_pol, n_omega)``, zero in every other row.

    Raises
    ------
    DriverError
        If the pulse is not one channel wide, or the channel is not one of the
        guide's.
    """
    labels = [label.display_label for label in guide.mode_collection.channel_labels]
    if built.shape[0] != 1:
        msg = (
            f"a general modal launch places one pulse in one channel and this "
            f"input carries {built.shape[0]}; build the state yourself and pass it "
            "to Simulation.with_input to launch into several at once"
        )
        raise DriverError(msg)
    if channel is None:
        index = 0
    elif isinstance(channel, str):
        if channel not in labels:
            msg = f"channel={channel!r} is not one of this guide's: {labels}"
            raise DriverError(msg)
        index = labels.index(channel)
    else:
        index = int(channel)
        if not 0 <= index < len(labels):
            msg = (
                f"channel={channel!r} is outside this guide's {len(labels)} "
                f"channel(s): {labels}"
            )
            raise DriverError(msg)
    state = np.zeros((len(labels), *built.shape[1:]), dtype=np.complex128)
    state[index] = built[0]
    logger.debug("general modal launch into channel %d (%s)", index, labels[index])
    return state


def _general_pulse(  # noqa: PLR0913 - the pulse's keyword surface and the launch
    incident: SpatialPulse | None,
    channel: int | str | None,
    shape: str,
    *,
    wavelength: float,
    fwhm: float | None,
    width: float | None,
    energy: float | None,
    peak_power: float | None,
    phases: Sequence[float] | np.ndarray,
    pulses: Sequence[Pulse] | None,
) -> GaussianPulse | SechPulse | PulseSum | SpatialPulse:
    """Return a general modal run's input: a shaped pulse, or an incident beam.

    Parameters
    ----------
    incident : solprop.pulses.SpatialPulse or None
        The incident beam, which carries its own temporal pulse; ``None`` builds the
        pulse from the keywords.
    channel : int or str or None
        The named channel, refused beside `incident`.
    shape, wavelength, fwhm, width, energy, peak_power, phases, pulses : object
        The pulse keywords, as :func:`_shaped_pulse` takes them.

    Returns
    -------
    solprop.pulses.GaussianPulse or SechPulse or PulseSum or SpatialPulse
        The input.

    Raises
    ------
    DriverError
        If `incident` is given beside `channel` or a pulse keyword, is a planar
        beam, or is launched away from its waist.
    """
    if incident is None:
        return _shaped_pulse(
            shape,
            wavelength=wavelength,
            fwhm=fwhm,
            width=width,
            energy=energy,
            peak_power=peak_power,
            phases=phases,
            pulses=pulses,
        )
    if channel is not None:
        msg = (
            "channel= and incident= are two ways to say what is launched and this "
            "call gives both: name a channel and the pulse enters it unchanged, or "
            "declare an incident beam and the channels capture what they overlap"
        )
        raise DriverError(msg)
    given = [
        name
        for name, value in (
            ("fwhm", fwhm),
            ("width", width),
            ("energy", energy),
            ("peak_power", peak_power),
            ("pulses", pulses),
        )
        if value is not None
    ]
    if len(phases):
        given.append("phases")
    if shape != "gauss":
        given.append("shape")
    if given:
        msg = (
            f"incident= carries its own temporal pulse, and this call also sets "
            f"{', '.join(given)}: give the beam's pulse its duration, energy and "
            "phase, and leave the pulse keywords unset"
        )
        raise DriverError(msg)
    if isinstance(incident, PlanarPulse):
        msg = (
            "a PlanarPulse launches into a planar guide's slab modes; the incident "
            "beam of a general modal guide is a SpatialPulse"
        )
        raise DriverError(msg)
    if incident.propz != 0:
        msg = (
            f"incident= is launched {incident.propz!r} m from its waist; carrying a "
            "beam to the guide's entrance is a transport step this launch does not "
            "model, so the beam is launched at its waist (propz=0)"
        )
        raise DriverError(msg)
    return incident


def _core_index(
    regions: Sequence[MaterialRegion], grid: TimeGrid
) -> Float[np.ndarray, " n_omega"]:
    """Return the real phase index of a guide's core at each of a grid's frequencies.

    The incident medium of a beam launched into the guide: the innermost material
    region, whose law does not move with z, so it is the entrance's.

    Parameters
    ----------
    regions : sequence of solprop.modes.material.MaterialRegion
        The guide's ordered material regions.
    grid : solprop.grid.TimeGrid
        The run's grid.

    Returns
    -------
    numpy.ndarray
        ``Re sqrt(eps)`` on the active positive band and zero elsewhere, shape
        ``(n_omega,)``.

    Raises
    ------
    DriverError
        If the guide's first region does not start on the axis.
    """
    core = regions[0]
    if not isinstance(core.inner_radius, float | int) or core.inner_radius != 0:
        msg = (
            f"an incident beam enters a guide through its core, the region on the "
            f"axis, and this guide's first region {core.name!r} does not start there"
        )
        raise DriverError(msg)
    live = grid.active_mask & (grid.omega > 0)
    index = np.zeros(grid.n_omega)
    index[live] = np.real(np.sqrt(core.material.permittivity(grid.omega[live])))
    return index


_BEAM_EXTENT = 60.0
"""Field exponent at the edge of an incident beam's radial plane.

The plane extends to where a super-Gaussian of order N has fallen to
``exp(-_BEAM_EXTENT)`` in field, ``r = waist * _BEAM_EXTENT**(1/(2N))``, times the
margin :data:`_BEAM_MARGIN`; its intensity there is below 1e-52 of the peak.
"""

_BEAM_MARGIN = 1.5
"""Factor on the incident beam's plane radius beyond its decay radius."""

_BEAM_NODES = 256
"""Radial nodes of an incident beam's plane.

With the plane radius above, the largest transverse wavenumber is about
``pi * 256 / (1.5 * 60**(1/2N) * waist)``, 69/w for a Gaussian, where its spectrum
is below exp(-1000): the beam is resolved to rounding.
"""


def _beam_grid(pulse: SpatialPulse) -> RadialGrid:
    """Return the radial plane an incident beam is sampled on before coupling.

    Parameters
    ----------
    pulse : solprop.pulses.SpatialPulse
        The beam.

    Returns
    -------
    solprop.grid.RadialGrid
        A grid that resolves the beam and holds it to rounding.
    """
    radius = _BEAM_MARGIN * pulse.waist * _BEAM_EXTENT ** (1.0 / (2.0 * pulse.order))
    return RadialGrid(radius, _BEAM_NODES)


def _incident_launch(
    simulation: "Simulation",  # noqa: UP037 - defined above, referenced while it loads
    pulse: SpatialPulse,
    model: CouplingModel,
) -> "Simulation":  # noqa: UP037 - as above
    """Couple an incident beam into a general modal simulation and install it.

    The beam becomes a radial :class:`~solprop.coupling.plane.FieldPlane` in the
    guide's core index (:meth:`~solprop.coupling.plane.FieldPlane.from_pulse`, its
    request met over its whole plane in the paraxial measure), and
    :func:`~solprop.coupling.coupler.couple` recovers its coordinates in the guide's
    channels under `model`. **Nothing is rescaled to the requested energy**: a
    badly captured beam launches the little the channels take, and a beam they
    cannot see is a valid zero launch. The launch is installed through
    :meth:`Simulation.with_input`, which records its provenance as
    ``configuration["coupling"]``; the coupling's report is recorded as
    ``configuration["launch"]`` and the beam as ``configuration["pulse"]``.

    Parameters
    ----------
    simulation : Simulation
        The general modal simulation, built with a zero input.
    pulse : solprop.pulses.SpatialPulse
        The incident beam and its temporal pulse.
    model : solprop.coupling.models.CouplingModel
        The coupling model.

    Returns
    -------
    Simulation
        The simulation launching the coupled amplitudes, sharing the translation.

    Raises
    ------
    DriverError
        If the beam cannot be built or the coupling refuses it.
    """
    port = simulation.input_port()
    index = (
        np.zeros(simulation.grid.n_omega)
        if port.reference_index is None
        else (np.asarray(port.reference_index))
    )
    try:
        plane = FieldPlane.from_pulse(
            pulse,
            simulation.grid,
            PlaneGeometry(port.geometry.z),
            space=_beam_grid(pulse),
            index=index,
        )
        result = couple(plane, port, model=model)
    except CouplingError as error:
        raise DriverError(str(error)) from error
    report = result.report
    logger.info(
        "incident beam coupled into %d channel(s) under %s: %.4g J incident, "
        "%.4g J launched",
        port.shape[0],
        model.kind,
        report.incident_energy,
        report.launched_energy,
    )
    launched = simulation.with_input(
        result.launch,
        noise=simulation.noise,
        configuration={"pulse": pulse, "launch": report},
    )
    # The launch carries the constructor's realisation unchanged, so a draw made
    # without a seed may still be drawn again from a recorded one.
    launched._noise_redraw = simulation._noise_redraw
    return launched


def _in_lowest_mode(
    built: Complex[np.ndarray, "n_modes_in n_pol n_omega"], n_modes: int
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return a one-mode input placed in the lowest mode of an `n_modes` set.

    A single pulse enters the fundamental and the higher modes start empty, filling
    through the nonlinearity.

    Parameters
    ----------
    built : numpy.ndarray
        The pulse, shape ``(1, n_pol, n_omega)``.
    n_modes : int
        How many modes the set carries.

    Returns
    -------
    numpy.ndarray
        The state, shape ``(n_modes, n_pol, n_omega)``.

    Raises
    ------
    DriverError
        If the input already carries more modes than the set does.
    """
    if built.shape[0] > n_modes:
        msg = (
            f"the input carries {built.shape[0]} modes and the set has {n_modes}; a "
            "pulse cannot be placed in a mode that is not propagated"
        )
        raise DriverError(msg)
    if built.shape[0] == n_modes:
        return built
    state = np.zeros((n_modes, *built.shape[1:]), dtype=np.complex128)
    state[: built.shape[0]] = built
    return state


def _hollow_components(*, supplied: bool) -> tuple[str, ...] | None:
    """Return the transverse components a hollow-guide run resolves.

    The capillary route's own set (a mode count) is unrotated HE_1m, which points along
    x (``docs/dev/conventions.md`` section 5), and keeps ``("x",)`` bitwise. A supplied
    set -- through either entry, with or without a coherent wall -- resolves the
    components its modes need (``None``, inferred by
    :func:`solprop.translate.problem.assemble`), so a TE_0m, TM_0m, EH or rotated set
    keeps its E_y. A wall's coupled flow contracts the channel axis of the one-component
    state and never reads the synthesis, so the backends admit it beside two synthesised
    components. ``from_waveguide`` refuses a mode count before this, so only the
    capillary entry reaches the first branch.

    Parameters
    ----------
    supplied : bool
        Whether the set was supplied rather than built from a mode count.

    Returns
    -------
    tuple of str or None
        ``("x",)``, or ``None`` to infer the components from the modes.
    """
    return None if supplied else DEFAULT_COMPONENTS


def _capillary_input_peak(
    field0: Complex[np.ndarray, "n_modes n_pol n_omega"],
    modes: ModeCollection,
    grid: TimeGrid,
    rule: TransverseQuadrature | None,
) -> tuple[float, float]:
    """Return the input's peak field, which sets the ionisation table's range.

    The peak is the largest field amplitude of the analytic signal over the axis and the
    run's own transverse nodes, which are where the plasma response is evaluated. On the
    axis it is the quantity :data:`solprop.stats.CATALOGUE`'s ``peak_intensity`` is the
    square of, and for an ``HE_1m`` launch the axis is the peak, so the range is
    anchored to a number the run itself reports. A ``TE_0m`` or ``TM_0m`` field vanishes
    on the axis (azimuthal order zero, but an azimuthal or radial vector field), so for
    those the nodes carry the peak. The same peak is the reference field of an optmat
    rate's constant energy per event (SEP 3, D8).

    Parameters
    ----------
    field0 : numpy.ndarray
        The input state, shape ``(n_modes, n_pol, n_omega)``.
    modes : solprop.modes.collection.ModeCollection
        The mode set, whose profiles are evaluated on the axis and at the nodes.
    grid : solprop.grid.TimeGrid
        The grid the state is stored on.
    rule : solprop.modes.quadrature.TransverseQuadrature or None
        The run's transverse quadrature; ``None`` for a mode-averaged run, whose
        table the axis alone bounds: the area-mean field it samples lies below it.

    Returns
    -------
    tuple of float
        The peak field amplitude in V/m, and the circular share ``|S3| / S0`` of the
        field's transverse components where it peaks, zero for a linear field.
    """
    analytic = analytic_field(grid, field0)
    limits = modes.dimlimits()[0]
    peak_power = 0.0
    circular = 0.0
    for where in (
        (on_axis_rule(limits),) if rule is None else (on_axis_rule(limits), rule)
    ):
        matrices = mode_matrix(modes.modes, where, 0.0, components=("x", "y"))
        rows = matrices.synthesis.reshape(matrices.n_nodes, matrices.n_pol, -1)
        # One node at a time: the field at every node and time together would be
        # n_nodes times the state's size, which a full angular rule makes large.
        for node in rows:
            columns = np.tensordot(node, analytic, axes=([1], [0]))
            power = np.sum(np.abs(columns) ** 2, axis=tuple(range(columns.ndim - 1)))
            index = int(np.argmax(power))
            if float(power[index]) > peak_power:
                peak_power = float(power[index])
                # The Stokes parameter of the x and y components at the peak sample.
                x, y = columns.reshape(columns.shape[0], -1, columns.shape[-1])[
                    :, 0, index
                ]
                circular = abs(2.0 * float(np.imag(x * np.conj(y)))) / peak_power
    peak = math.sqrt(peak_power)
    logger.debug("input peak field over the axis and the nodes %.4e V/m", peak)
    return peak, circular


@dataclasses.dataclass(frozen=True)
class _InputPeak:
    """What the input's peak field is found from, evaluated only when it is needed.

    Attributes
    ----------
    field0 : numpy.ndarray
        The input state, shape ``(n_modes, n_pol, n_omega)``.
    modes : solprop.modes.collection.ModeCollection
        The mode set.
    grid : solprop.grid.TimeGrid
        The grid the state is stored on.
    rule : solprop.modes.quadrature.TransverseQuadrature or None
        The run's transverse quadrature; ``None`` for a mode-averaged run.
    """

    field0: Complex[np.ndarray, "n_modes n_pol n_omega"]
    modes: ModeCollection
    grid: TimeGrid
    rule: TransverseQuadrature | None

    def evaluate(self) -> tuple[float, float]:
        """Return the peak field in V/m and its circular share.

        Returns
        -------
        tuple of float
            :func:`_capillary_input_peak` of the input.
        """
        return _capillary_input_peak(self.field0, self.modes, self.grid, self.rule)


def _ionisation_inputs(
    responses: Sequence[object],
    field_range: tuple[float, float] | None,
    peak: _InputPeak | None,
) -> tuple[list[object], tuple[float, float] | None, float | None]:
    """Return a capillary run's ionisation table range and its bound rates.

    The input's peak field is found once, when the table's range is the default one
    (:func:`default_field_range`) or an optmat rate waits for its reference field.

    Parameters
    ----------
    responses : sequence of object
        The run's responses.
    field_range : tuple of float or None
        The caller's table range in V/m, or ``None`` for the default.
    peak : _InputPeak or None
        Where the input's peak is found from; ``None`` for a run without plasma.

    Returns
    -------
    tuple
        The responses (optmat rates bound to the peak), the table range, and the
        reference field in V/m the rates were bound to, or ``None``.
    """
    unbound = _unbound_rates(responses)
    if peak is None or not (field_range is None or unbound):
        return list(responses), field_range, None
    found = peak.evaluate()
    if field_range is None:
        field_range = default_field_range(found[0])
    if not unbound:
        return list(responses), field_range, None
    return _bound_rates(responses, found, field_range), field_range, found[0]


def _peak_field_reports(
    problem: KernelProblem,
    output: Output,
    responses: Sequence[object],
    first_plane: int = 0,
) -> tuple[PeakFieldReport, ...]:
    """Report the largest field each optmat-rate population saw, after a run.

    The populations' peak rates at the save planes are read back from the run's
    output -- ``species_peak_ionisation_rate``, or ``peak_ionisation_rate`` for one
    population -- and inverted through each rate
    (:func:`solprop.nonlinear.optmat_rates.peak_field_report`).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem, whose populations name the statistics' rows.
    output : solprop.output.schema.Output
        The handler the run wrote; a closed file is reopened for reading.
    responses : sequence of object
        The run's responses.
    first_plane : int, optional
        The first plane to read: a session that continued a file across a forced
        resume reports only the planes it wrote, since the earlier ones were
        computed under other physics (SEP 42 section 5.5).

    Returns
    -------
    tuple of PeakFieldReport
        One per population with an optmat rate; none when the run carries no such
        rate or recorded no peak rate, which is logged.

    Warns
    -----
    MaterialRangeWarning
        Once per run, for a population that reached its rate's maximum field, above
        which the rate is held.
    """
    labels = problem.responses.labels
    rates = [
        (
            one.species
            or (labels[0] if len(labels) == 1 else str(getattr(one.gas, "name", ""))),
            one.rate,
        )
        for one in responses
        if isinstance(one, PlasmaResponse | EnvelopePlasmaResponse)
        and isinstance(one.rate, OptmatRate)
    ]
    if not rates:
        return ()
    if isinstance(output, HDF5Output) and output.closed:
        # An internal reader: not the public open_file, which warns of a forced file.
        with HDF5Output(output.path, readonly=True) as reader:
            return _peak_field_reports(problem, reader, responses, first_plane)
    if "species_peak_ionisation_rate" in output.stat_names:
        rows = np.asarray(output["stats/species_peak_ionisation_rate"])[first_plane:]
        peaks = np.max(rows, axis=0)
    elif "peak_ionisation_rate" in output.stat_names and len(labels) == 1:
        rows = np.asarray(output["stats/peak_ionisation_rate"])[first_plane:]
        peaks = np.array([np.max(rows)])
    else:
        logger.info(
            "the run recorded no peak ionisation rate, so the largest field its "
            "optmat rates saw is not reported"
        )
        return ()
    envelope = problem.responses.envelope_plasma
    cycles = (
        {}
        if envelope is None
        else {channel.label: channel.cycle for channel in envelope.channels}
    )
    reports = []
    for label, rate in rates:
        report = peak_field_report(
            rate,
            float(peaks[labels.index(label)]),
            label=label,
            # An envelope's statistic is its own cycle average of its table, at
            # linear polarisation the cosine moment alone.
            average=None
            if label not in cycles
            else functools.partial(_linear_moment, cycles[label]),
        )
        if report.reached_maximum:
            _optmat.warn_once(
                ("rate-reached-maximum", rate.material_records[0].digest),
                f"{label}: the run reached the maximum field "
                f"{report.maximum_field:.4g} V/m of optmat's {report.model}, above "
                "which its rate is held at its value there; the largest field is at "
                "least that",
                MaterialRangeWarning,
            )
        logger.info(
            "%s: largest field %s V/m (maximum %.4g V/m); rate-weighted energy there "
            "%s against the constant %.6g J",
            label,
            "none" if report.largest_field is None else f"{report.largest_field:.4g}",
            report.maximum_field,
            "none"
            if report.departure is None
            else f"departs by {report.departure:+.3e}",
            report.constant_energy,
        )
        reports.append(report)
    return tuple(reports)


def _check_rate_inputs(
    simulation: Simulation, initial: Pulse | SpatialPulse | np.ndarray
) -> None:
    """Check a run's input against each optmat rate it carries, once per run.

    The reference field and the table range stay those the run was built with; the
    input is checked against each rate's maximum field and, when it is not linearly
    polarised, the Keldysh parameter at its peak
    (:meth:`solprop.nonlinear.optmat_rates.OptmatRate.check_input`). On a mode-set
    route the peak and the polarisation are the field's over the axis and the nodes
    (:func:`_capillary_input_peak`); on a free-space route the peak is the physical
    intensity's maximum and the polarisation the pulse's Jones vector, which a
    spectrum given as an array does not declare, so it is taken as linear there.

    Parameters
    ----------
    simulation : Simulation
        The run, its input already built.
    initial : Pulse, SpatialPulse or numpy.ndarray
        The input as given.

    Warns
    -----
    MaterialRangeWarning
        As :meth:`~solprop.nonlinear.optmat_rates.OptmatRate.check_input` does.
    """
    rates = [
        one.rate
        for one in simulation.responses
        if isinstance(one, PlasmaResponse | EnvelopePlasmaResponse)
        and isinstance(one.rate, OptmatRate)
    ]
    if not rates:
        return
    if simulation.spatial_grid is not None:
        peak, circular = _spatial_input_peak(simulation, initial)
    elif simulation.general is None and simulation.settings.quadrature is not None:
        peak, circular = _capillary_input_peak(
            simulation._field0,
            ModeCollection(list(simulation.modes)),
            simulation.grid,
            simulation.settings.quadrature,
        )
    else:
        return
    for rate in rates:
        rate.check_input(peak, circular=circular)


def _bind_spatial_rates(
    simulation: Simulation, initial: Pulse | SpatialPulse | np.ndarray
) -> None:
    """Bind a free-space or planar run's optmat rates to its input, and check it.

    The spatial routes -- radial, Cartesian, line and planar -- bind an optmat rate's
    constant energy per event to the input's peak field, as the capillary route does
    (SEP 3, Q14); binding also checks the input against each rate. A run whose rates
    are all bound is only checked.

    Parameters
    ----------
    simulation : Simulation
        A spatial run, its input already built; its responses are replaced.
    initial : Pulse, SpatialPulse or numpy.ndarray
        The input as given.

    Warns
    -----
    MaterialRangeWarning
        As :func:`_bound_rates` and :func:`_check_rate_inputs` do.
    """
    if _unbound_rates(simulation.responses):
        simulation.responses = tuple(
            _bound_rates(
                simulation.responses,
                _spatial_input_peak(simulation, initial),
                simulation.settings.ionisation_field_range,
            )
        )
    else:
        _check_rate_inputs(simulation, initial)


def _spatial_input_peak(
    simulation: Simulation, initial: Pulse | SpatialPulse | np.ndarray
) -> tuple[float, float]:
    """Return a free-space or planar run's peak input field and its circular share.

    Parameters
    ----------
    simulation : Simulation
        A free-space or planar run, its input already built.
    initial : Pulse, SpatialPulse or numpy.ndarray
        The input as given.

    Returns
    -------
    tuple of float
        The peak physical field amplitude in V/m, and
        the circular share ``|S3|/S0`` of the pulse's Jones vector; a spectrum given
        as an array declares none and is taken as linear.
    """
    space = cast("RadialGrid | FreeGrid | PlanarGrid", simulation.spatial_grid)
    # Ionisation is driven by the physical field, never by a signed axial flux.
    peak = peak_field(simulation.grid, space, simulation._field0)
    jones = (
        np.asarray(initial.jones, dtype=np.complex128)
        if isinstance(initial, SpatialPulse)
        else np.ones(1, dtype=np.complex128)
    )
    circular = (
        abs(2.0 * float(np.imag(jones[0] * np.conj(jones[1]))))
        / float(np.sum(np.abs(jones) ** 2))
        if jones.size == 2
        else 0.0
    )
    return peak, circular


def _linear_moment(cycle: CycleIntegralData, amplitude: float) -> float:
    """Return an envelope run's cycle-averaged rate at a linear amplitude.

    Parameters
    ----------
    cycle : solprop.translate.envelope_plasma.CycleIntegralData
        The run's cycle integrals of one population.
    amplitude : float
        Envelope amplitude in V/m.

    Returns
    -------
    float
        The rate in s^-1 its statistic records there.
    """
    return float(cycle_moments(amplitude, 0.0, cycle)[0])


def _unbound_rates(responses: Sequence[object]) -> bool:
    """Return whether a plasma response carries an optmat rate without a reference.

    Parameters
    ----------
    responses : sequence of object
        The run's responses.

    Returns
    -------
    bool
        True when a rate's constant energy per event still waits for the input's peak.
    """
    return any(
        isinstance(one, PlasmaResponse | EnvelopePlasmaResponse)
        and isinstance(one.rate, OptmatRate)
        and one.rate.reference_field is None
        for one in responses
    )


def _bound_rates(
    responses: Sequence[object],
    peak: tuple[float, float],
    field_range: tuple[float, float] | None,
) -> list[object]:
    """Bind a run's optmat rates to the input's peak field, and check the input.

    Parameters
    ----------
    responses : sequence of object
        The run's responses.
    peak : tuple of float
        The input's peak field in V/m and its circular share there.
    field_range : tuple of float or None
        The field amplitudes in V/m the run tabulates its rates over.

    Returns
    -------
    list
        The responses, each unbound optmat rate taking its constant energy at the
        peak (:func:`solprop.nonlinear.optmat_rates.bind_reference_field`).

    Warns
    -----
    MaterialRangeWarning
        Once per run, when the peak exceeds a rate's maximum field, or the input is
        not linearly polarised at a Keldysh parameter above one
        (:meth:`solprop.nonlinear.optmat_rates.OptmatRate.check_input`).
    """
    field, circular = peak
    bound = list(bind_reference_field(responses, field, field_range))
    for one in bound:
        if isinstance(one, PlasmaResponse | EnvelopePlasmaResponse) and isinstance(
            one.rate, OptmatRate
        ):
            one.rate.check_input(field, circular=circular)
    return bound


def _rebind_mode(mode: Mode, mixture: Mixture) -> Mode:
    """Bind a gas fill through the mode's explicit material interface.

    The mode's own gas, pressure and temperature (a temperature path included) are
    replaced by the mixture's; only its geometry is kept.

    Parameters
    ----------
    mode : Mode
        Supplied gas-filled geometry.
    mixture : Mixture
        New ordered material populations.

    Returns
    -------
    Mode
        The rebound geometry.
    """
    rebind: Callable[[Mixture], Mode] | None = getattr(mode, "with_mixture", None)
    if not callable(rebind):
        # The hint belongs to the mode that has a route: a step-index fibre is a
        # mode-averaged GNLSE run. Anything else gets the plain refusal.
        message = f"{type(mode).__name__} cannot replace its material with a Mixture"
        if isinstance(mode, StepIndexMode):
            message += "; use prop_gnlse(fibre=...) instead"
        raise DriverError(message)
    return rebind(mixture)


def _hollow_stats(
    *,
    with_plasma: bool,
    mixture: Mixture,
    profiles: Mapping[str, ZProfile],
    rule: TransverseQuadrature | None,
) -> StatsSpec:
    """Keep capillary statistics where the supplied geometry defines them.

    Parameters
    ----------
    with_plasma : bool
        Whether plasma statistics are available.
    mixture : Mixture
        Ordered material populations.
    profiles : mapping of str to ZProfile
        Available physical paths.
    rule : solprop.modes.quadrature.TransverseQuadrature or None
        The run's fixed transverse rule, whose embedded coarse rule the
        quadrature alarm needs; ``None`` for a mode-averaged run.

    Returns
    -------
    StatsSpec
        Statistics with unsupported geometry paths omitted.
    """
    base = _capillary_stats(
        with_plasma=with_plasma, mixture=mixture, averaged=rule is None
    )
    dropped = {name for name in ("core_radius",) if name not in profiles}
    if rule is not None and not rule.kronrod:
        # Without an embedded coarse rule the estimate is NaN at every plane, so
        # the statistic is dropped rather than recorded empty. A Cartesian
        # product rule has no Kronrod extension at all (quadrature_rule stores
        # kronrod only for a polar rule), so the alarm is off for every
        # rectangular run, and the log below says so rather than leaving it silent.
        dropped.add("quadrature_error")
        logger.info(
            "the %s rule has no embedded coarse rule, so the quadrature alarm is "
            "off and quadrature_error is not recorded",
            rule.kind,
        )
    return dataclasses.replace(
        base, names=tuple(name for name in base.names if name not in dropped)
    )


def _refuse_dispersive_policy(materials: MaterialSelection | None, route: str) -> None:
    """Refuse a dispersive electronic policy on a route whose response is constant.

    Parameters
    ----------
    materials : MaterialSelection or None
        The run's material selection.
    route : str
        What carries the route's constant electronic coefficient, for the message.

    Raises
    ------
    DriverError
        If the selection names a dispersive electronic policy (SEP 6).
    """
    if materials is None or materials.electronic not in DISPERSIVE_POLICIES:
        return
    msg = (
        f"the electronic policy {materials.electronic!r} evaluates optmat's "
        f"dispersive hyperpolarizability model, and {route} is a constant coefficient "
        "this route keeps: select 'carrier-kerr' (or 'recommended') here"
    )
    raise DriverError(msg)


def _check_electronic_policy(
    materials: MaterialSelection | None, responses: Sequence[object]
) -> None:
    """Refuse a constant Kerr response under a dispersive electronic policy.

    A caller's own responses are taken as they are, but a constant
    :class:`~solprop.nonlinear.responses.KerrResponse` beside a selection that names a
    dispersive policy would record one law and propagate another.

    Parameters
    ----------
    materials : MaterialSelection or None
        The run's material selection.
    responses : sequence of object
        The run's declared responses.

    Raises
    ------
    DriverError
        If the selection is dispersive and a constant Kerr response is declared.
    """
    if materials is None or materials.electronic not in DISPERSIVE_POLICIES:
        return
    if any(isinstance(response, KerrResponse) for response in responses):
        msg = (
            f"the selection's electronic policy is {materials.electronic!r} and the "
            "run declares a constant KerrResponse; build the electronic response as "
            "a DispersiveKerrResponse (solprop.nonlinear.mixtures."
            "electronic_response), or select 'carrier-kerr'"
        )
        raise DriverError(msg)


def _check_selection_carrier(
    materials: MaterialSelection | None,
    reference: float,
    wavelength: float | None = None,
) -> None:
    """Refuse a carrier that is not the run's: one carrier sets the Kerr coefficient.

    Parameters
    ----------
    materials : MaterialSelection or None
        The run's material selection.
    reference : float
        The grid's reference wavelength in m, the run's carrier.
    wavelength : float or None, optional
        The driver's `wavelength` in m, at which the responses are built.

    Raises
    ------
    DriverError
        If the selection's ``carrier_wavelength`` or `wavelength` is not the grid's
        reference wavelength: the electronic coefficient, the Raman response and the
        rate are all evaluated at the carrier.
    """
    if materials is None:
        return
    for name, value in (
        ("the selection's carrier_wavelength", materials.carrier_wavelength),
        ("wavelength", wavelength),
    ):
        if value is not None and not carriers_agree(float(value), reference):
            msg = (
                f"{name} is {value:g} m and the grid's reference wavelength "
                f"{reference:g} m: under a material selection one carrier sets the "
                "Kerr coefficient of every optmat species (SEP 3, section 4.3), and "
                "it is the grid's. Build the grid at that wavelength, or leave the "
                "selection's carrier_wavelength unset"
            )
            raise DriverError(msg)


def _default_cladding(cladding: GlassIndex | complex | None) -> GlassIndex | complex:
    """Return the cladding a numbered capillary set takes.

    Parameters
    ----------
    cladding : GlassIndex, complex or None
        The cladding given; ``None`` is the package default.

    Returns
    -------
    GlassIndex or complex
        A given cladding as it is; the default as optmat's recommended silica.
    """
    return fused_silica() if cladding is None else cladding


def _selected_medium(
    medium: SpatialMedium, materials: MaterialSelection
) -> SpatialMedium:
    """Resolve a bulk medium's built-in gas under a material selection.

    A medium that names a built-in gas meant the built-in gas: it becomes that gas's
    one-population mixture with the selection's models, at the same pressure path and
    temperature, so that it is recorded; a supplied susceptibility is kept beside
    optmat's equation of state, as for a capillary fill. Any other medium is taken as
    it is.

    Parameters
    ----------
    medium : solprop.spatial.SpatialMedium
        The medium given.
    materials : MaterialSelection
        The run's material selection.

    Returns
    -------
    solprop.spatial.SpatialMedium
        The medium to propagate in.
    """
    if not isinstance(medium.material, Gas):
        return medium
    gas = medium.material
    selected = species(gas, selection=materials)
    if medium.susceptibility != default_species(gas).susceptibility:
        selected = dataclasses.replace(selected, susceptibility=medium.susceptibility)
    # A constructed medium holds its resolved temperature, a number.
    fill = Mixture.fill(
        {selected: 1.0}, medium.pressure, temperature=cast("float", medium.temperature)
    )
    return SpatialMedium(fill)


def _selected_guide(guide: ModalGuide, materials: MaterialSelection) -> ModalGuide:
    """Resolve a guide's default delayed silica response under a material selection.

    A region's :class:`~solprop.nonlinear.raman.SilicaTensorResponse` with the
    default parameters, optmat's recommended tensor bundle, meant the built-in silica
    tensor response; under the selection it becomes the tensor bundle its ``raman``
    names (:meth:`~solprop.physdata.glasses.SilicaBundle.resolve` with
    ``tensor=True``). A response declared with any other parameters, and a provider
    of another kind, are taken as they are.

    Parameters
    ----------
    guide : solprop.modes.guide.ModalGuide
        The guide given.
    materials : MaterialSelection
        The run's material selection.

    Returns
    -------
    solprop.modes.guide.ModalGuide
        The guide to propagate on: the one given when nothing resolves, otherwise
        the same provider, channels and sampling with the regions' delayed
        responses resolved.
    """
    default = SilicaTensorResponse()
    if not any(region.raman == default for region in guide.regions):
        return guide
    provider = guide.field_provider
    if not isinstance(provider, StepIndexProvider | GradedIndexProvider):
        return guide
    resolved = SilicaTensorResponse(
        parameters=SilicaBundle.resolve(materials.raman, tensor=True)
    )

    def selected(region: MaterialRegion) -> MaterialRegion:
        """Return a region with its default delayed response resolved.

        Parameters
        ----------
        region : solprop.modes.material.MaterialRegion
            The region given.

        Returns
        -------
        solprop.modes.material.MaterialRegion
            The region, its default response replaced by the bundle's.
        """
        return dataclasses.replace(
            region,
            responses=tuple(
                resolved if one == default else one for one in region.responses
            ),
        )

    rebuilt = (
        dataclasses.replace(
            provider, regions=tuple(selected(one) for one in provider.regions)
        )
        if isinstance(provider, StepIndexProvider)
        else dataclasses.replace(
            provider,
            core_responses=selected(provider.regions[0]).responses,
        )
    )
    return dataclasses.replace(
        guide, field_provider=rebuilt, regions=tuple(rebuilt.regions)
    )


def _selected_planar(space: PlanarGrid, materials: MaterialSelection) -> PlanarGrid:
    """Resolve a planar grid's built-in materials under a material selection.

    The slab modes' fill of a built-in gas with its default susceptibility meant the
    built-in gas: it takes the selection's single-species susceptibility
    (:func:`solprop.physdata.species.species`), and keeps its pressure, temperature
    and optmat density path, so no mixture -- which the hybrid transform does not
    admit -- is needed (SEP 3, Q5). A slab declared without a cladding takes optmat's
    recommended silica. A supplied susceptibility and a declared cladding are the
    caller's.

    Parameters
    ----------
    space : solprop.modes.slab.PlanarGrid
        The planar grid given.
    materials : MaterialSelection
        The run's material selection.

    Returns
    -------
    solprop.modes.slab.PlanarGrid
        The grid given when nothing resolves; otherwise the same grid rebuilt on the
        resolved slab modes.
    """
    modes = cast("tuple[SlabMode, ...]", tuple(space.modes))
    first = modes[0]
    changes: dict[str, object] = {}
    if first.mixture is None and first.gas is not None:
        gas = as_gas(first.gas)
        if first.susceptibility == default_species(gas).susceptibility:
            # The modes share one fill (`PlanarGrid` refuses any other), so one
            # resolved susceptibility serves them all and the medium is checked
            # against them by identity.
            resolved = species(gas, selection=materials).susceptibility
            if resolved != first.susceptibility:
                changes["susceptibility"] = resolved
    cladding = fused_silica()
    rebuilt = tuple(
        dataclasses.replace(
            mode,
            **changes,
            **({"cladding": cladding} if mode.cladding is None else {}),
        )
        if changes or mode.cladding is None
        else mode
        for mode in modes
    )
    if all(one is two for one, two in zip(rebuilt, modes, strict=True)):
        return space
    with warnings.catch_warnings():
        # The caller's own grid already said so if its n_x is below the bound.
        warnings.filterwarnings(
            "ignore", message=r"PlanarGrid n_x = ", category=UserWarning
        )
        return dataclasses.replace(space, modes=rebuilt)


def _planar_medium(space: PlanarGrid) -> SpatialMedium:
    """Return the fill of a planar grid's modes as a constant bulk medium.

    Parameters
    ----------
    space : solprop.modes.slab.PlanarGrid
        The planar grid, whose modes share one fill.

    Returns
    -------
    solprop.spatial.SpatialMedium
        The gas at the modes' pressure, temperature and susceptibility, or vacuum.

    Raises
    ------
    DriverError
        If the fill is a mixture or varies along z: the hybrid transform carries
        constant media only, as Cartesian free space does, and mixtures are not yet
        admitted.
    """
    mode = cast("SlabMode", space.modes[0])
    if mode.mixture is not None:
        msg = (
            "a planar run takes a single-gas fill: a mixture on the hybrid planar "
            "transform is not admitted yet; fill the slab with gas= and pressure="
        )
        raise DriverError(msg)
    if not mode.density.is_constant:
        msg = (
            "the hybrid planar transform requires constant media: a gas path along z "
            "(a varying pressure or temperature) is refused, as for Cartesian free "
            "space, whose seam evaluates a constant operator exactly and has no "
            "varying-medium expansion"
        )
        raise DriverError(msg)
    if mode.gas is None:
        return SpatialMedium()
    return SpatialMedium(
        mode.gas,
        cast("float | ZProfile", mode.pressure),
        temperature=cast("float", mode.temperature),
        susceptibility=mode.susceptibility,
    )


def _boundary(
    absorber: Absorber | Literal[False] | None,
    scheme: AbsorberScheme | None,
    grid: TimeGrid,
    *,
    default: Callable[[TimeGrid], Absorber] | None = None,
) -> tuple[Absorber | None, AbsorberScheme]:
    """Resolve a builder's absorber keywords (SEP 21 section 5).

    Parameters
    ----------
    absorber : Absorber, False or None
        The collar given; ``False`` switches both soft collars off and keeps the
        hard band mask; ``None`` takes `default`.
    scheme : {'nonlinear', 'split_step'} or None
        The temporal scheme given; ``None`` is the default scheme.
    grid : TimeGrid
        The run's grid.
    default : callable or None, optional
        The route's collar when none is given; ``None`` is
        :func:`default_collar`.

    Returns
    -------
    tuple
        The absorber (``None`` for none) and the scheme.

    Raises
    ------
    DriverError
        If a scheme is given beside ``absorber=False``, which has no temporal
        collar for it to apply.
    """
    if absorber is False:
        if scheme is not None:
            msg = (
                "absorber=False switches the temporal collar off, so there is nothing "
                f"for absorber_scheme={scheme!r} to apply; give one or the other"
            )
            raise DriverError(msg)
        return no_absorber(grid), DEFAULT_ABSORBER_SCHEME
    if absorber is None:
        absorber = default_collar(grid) if default is None else default(grid)
    return absorber, DEFAULT_ABSORBER_SCHEME if scheme is None else scheme


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
    >>> from solprop.driver.simulation import default_collar
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

    Parameters
    ----------
    settings : SimulationSettings
        The run's settings.
    grid : TimeGrid
        The run's temporal grid.

    Returns
    -------
    SimulationSettings
        The settings, with an absorber.
    """
    if settings.absorber is not None:
        return settings
    return dataclasses.replace(settings, absorber=default_collar(grid))


def _transverse_specification(
    setting: SpatialAbsorber | Literal[False] | None,
    spatial_grid: object,
) -> SpatialAbsorber | None:
    """Return the transverse boundary a run's translation resolves (SEP 40).

    Parameters
    ----------
    setting : solprop.boundaries.SpatialAbsorber, False or None
        :attr:`SimulationSettings.spatial_absorber`.
    spatial_grid : object
        The run's transverse grid, ``None`` for a modal run.

    Returns
    -------
    solprop.boundaries.SpatialAbsorber or None
        The specification; ``None`` for no ramp and no collar.

    Raises
    ------
    DriverError
        If a modal run is given a specification.
    """
    if spatial_grid is None:
        if isinstance(setting, SpatialAbsorber):
            msg = (
                "spatial_absorber sets a free-space run's transverse boundary; a "
                "modal route has no transverse window"
            )
            raise DriverError(msg)
        return None
    if setting is False:
        return None
    return SpatialAbsorber(real_space=False) if setting is None else setting
