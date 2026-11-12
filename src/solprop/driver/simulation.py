# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The simulation object: physics objects in, a written, resumable run out.

A :class:`Simulation` holds what a run is made of -- the grid, the modes, the responses,
the input, the noise and the numerical settings -- translates it **once** into a
:class:`solprop.translate.problem.PropagationProblem`, and can then be run as often as
one likes, into memory or into a file. It owns three things the loop of
:mod:`solprop.driver.run` does not: the translation and its cache, the
:class:`solprop.output.schema.RunMetadata` that says what produced a file, and the
resume decision.

**Resume.** Given a path, a run checks the file that is already there with
:func:`solprop.output.hdf5.check_cache`: the digest covers the grid, the modes, the
whole serialised configuration, the noise and the stepper settings, so a file written
for a different run is refused by name rather than continued from someone else's
state. If it passes, the last plane comes back through
:func:`solprop.output.hdf5.resume`, the backend is loaded at that plane's position, and
the run carries on into the same file with ``HDF5Output(path, append=True)``.

Layer D of ``docs/dev/architecture.md`` section 3: this module imports the physics, the
translation, the backends and the output layer, and nothing imports it. The manual page
is ``docs/guide/setup/running.md``.
"""

import contextlib
import copy
import dataclasses
import functools
import json
import logging
import math
import threading
import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, SupportsIndex, cast

import numpy as np
from jaxtyping import Complex, Float, Shaped
from scipy.special import j0, jn_zeros

import solprop.driver.tdse as tdse_runtime
from solprop._display import Displayed, engineering, line, summarise
from solprop._ownership import borrow, frozen_copy, hold, hold_view
from solprop.adiabatic import AdiabaticAccuracy
from solprop.backends.ledger import CallLedger
from solprop.backends.protocol import Backend, _all_finite
from solprop.backends.registry import resolve_name, select
from solprop.backends.session import ExecutionOptions, Start, positive_count
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
from solprop.driver._launch import launch_pulse, refuse_beside
from solprop.driver.defaults import (
    CAPILLARY_RTOL,
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    FREE_ATOL,
    FREE_NOISE,
    FREE_RTOL,
    INTERIOR_LAUNCH_NOISE,
    PLANAR_NOISE,
    SLAB_NOISE,
    SPATIAL_ATOL,
    SPATIAL_RTOL,
    TDSE_NOISE,
    WAVEGUIDE_ATOL,
    WAVEGUIDE_RTOL,
    NoiseModel,
    default_controller,
    default_integrator,
    default_rtol,
    default_scheme,
    resolve_noise,
)
from solprop.driver.material_output import (
    material_manifest,
    material_record,
    processing_record,
)
from solprop.driver.mixtures import (
    ResponseSelection,
    capillary_fill,
    is_air,
    mixture_profiles,
)
from solprop.driver.run import (
    RESTART_RECORD,
    check_snapshot_request,
    field_retention,
    open_run_record,
    retention_at,
    run_problem,
)
from solprop.exceptions import (
    AdiabaticTaperWarning,
    BackendError,
    BoundaryWarning,
    CouplingError,
    DriverError,
    GasFlowWarning,
    GrazingIncidenceWarning,
    GridError,
    MaterialRangeWarning,
    MaterialStatusWarning,
    ParameterPath,
    SampledPeakWarning,
    SolpropError,
    SolpropWarning,
    StopRequested,
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
from solprop.log import (
    DEFAULT_PROGRESS_PERIOD,
    ProgressChoice,
    ProgressReporter,
    StopFlag,
)
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
from solprop.output.hdf5 import (
    HDF5Output,
    ResumeState,
    _stored_metadata,
    check_cache,
)
from solprop.output.hdf5 import resume as resume_state
from solprop.output.memory import MemoryOutput
from solprop.output.schema import (
    INTERPOLATED_RESUMES_ANNOTATION,
    PEAK_FIELDS_ANNOTATION,
    CheckpointOutput,
    DefaultsResolution,
    FieldRetention,
    FieldsChoice,
    GridRecord,
    MarchSnapshot,
    ModeRecord,
    NoiseRecord,
    NoiseResolution,
    Output,
    PlaneStore,
    RecipeRecord,
    RunMetadata,
    StatedInput,
    StepLimits,
    StepperRecord,
    _recorded_inline,
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
    SpatialPulseSum,
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
    metric_at,
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
    AbsorberScheme,
    noise_spectrum,
)
from solprop.translate.plasma_integration import (
    integration_record,
)
from solprop.translate.plasma_source import _BENCHMARK_GATED, plasma_source_record
from solprop.translate.problem import (
    DEFAULT_COMPONENTS,
    PropagationProblem,
    PropagatorMode,
    StepperSettings,
    TranslationReuse,
    _held_block,
    _held_state,
    _state,
    assemble,
)
from solprop.translate.responses import KerrForm, declares_harmonic_carrier
from solprop.translate.routes import (
    SpatialRoute,
    flux_of,
    tdse_descriptor,
    tdse_of,
    transverse_of,
)
from solprop.translate.run_request import RunRequest
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
    "SLAB_NOISE",
    "SPATIAL_ATOL",
    "SPATIAL_RTOL",
    "TDSE_NOISE",
    "WAVEGUIDE_ATOL",
    "WAVEGUIDE_RTOL",
    "ModalLaunch",
    "NoiseModel",
    "NoiseSetting",
    "RecipeSource",
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

type _ResumeChoice = bool | Literal["interpolated"]
"""What ``resume`` takes: no, yes, or from a plane interpolated inside a step."""


LOOSE_ADAMS_RTOL = 5e-2
"""The ``rtol`` from which an adaptive Adams run is warned on.

An error of five per cent or more allowed per step compounds through the method's
history, and its predictor-corrector estimate is unreliable there
(``docs/guide/numerics/choosing-a-stepper.md``).
"""

DEFAULT_N_SAVE = 201
"""Planes a GNLSE run stores, counting the one at ``z = 0``."""


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
recording which in the run's :class:`~solprop.output.schema.NoiseResolution`.
"""

ModalLaunch = (
    Mapping[str | int, float | Mapping[str, float]]
    | Sequence[Mapping[str | int, float | Mapping[str, float]]]
)
"""What a modal constructor's ``launch`` keyword takes (SEP 41, S5).

A mapping from a mode's display label or index to its share, or to
``{"share": s, "phase": phi}`` with the phase in radians; or a list of such mappings,
one for each input pulse (``docs/guide/setup/pulses.md``).
"""


class _Defaulted(str):
    """A setting's value that a default rule chose rather than the caller.

    A copy made with :func:`dataclasses.replace` carries the marker with the value, so
    the copy resolves the rule again for its own route or planes (the collar scheme),
    while a value the caller names, even an equal one, is a plain string and stays as
    given.
    """

    __slots__ = ()


if TYPE_CHECKING:
    from solprop.calibration.refine import RefinementReport
    from solprop.driver.estimate import RunEstimate


class RecipeSource(Protocol):
    """What built a simulation from a recipe: it writes the record its output keeps.

    :class:`solprop.recipe.core.Recipe` is one (SEP 41, S2); the driver needs only this.
    """

    def record(self, simulation: "Simulation") -> RecipeRecord:
        """Return the authored and resolved recipe of a run about to start.

        Parameters
        ----------
        simulation : Simulation
            The simulation, translated, with its settings at run start.

        Returns
        -------
        solprop.output.schema.RecipeRecord
            The record its output keeps.
        """
        ...


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
        :data:`~solprop.driver.defaults.SPATIAL_ATOL` of :meth:`Simulation.from_radial`
        and :meth:`Simulation.from_planar`, the
        :data:`~solprop.driver.defaults.FREE_RTOL` and
        :data:`~solprop.driver.defaults.FREE_ATOL` of :meth:`Simulation.from_free`.
        An explicit controller always wins, and settings that already belong to a
        route keep its default. The other fields are keyword-only.
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
        interface's builder sets it (:meth:`Simulation.from_gnlse`,
        :meth:`Simulation.from_waveguide` and :meth:`Simulation.from_free`;
        ``docs/guide/numerics/choosing-a-stepper.md``); the stored value is the
        resolved method. ``Multistep()`` is the low-call exponential Adams
        method, which
        ``numpy_ref`` and ``native`` execute for supported
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
        :func:`solprop.translate.problem.assemble`'s: the taper costs a systematic
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
        most :data:`solprop.driver.run.MARCH_SNAPSHOT_AUTO_BYTES`; it is outside the
        resume key. Every backend's fused and exponential Adams engines, the native
        backend's resident one included, export and install their history.
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
            If the planes are empty, not increasing, or not positive.
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
        solprop.output.schema.DefaultsResolution
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


def _admit_tdse_execution(settings: SimulationSettings) -> None:
    """Refuse TDSE execution resources beside settings a TDSE run cannot step with.

    Each refusal names the setting at fault, in the order a caller would fix them:
    the backend, then the integrator.
    """
    if not isinstance(settings.tdse_execution, TDSEExecution):
        msg = (
            "tdse_execution must be a TDSEExecution, got "
            f"{type(settings.tdse_execution).__name__}"
        )
        raise DriverError(msg, parameter="tdse_execution")
    if settings.backend != "numpy_ref":
        msg = (
            f"tdse_execution requires an explicit backend='numpy_ref', the one backend "
            f"that runs the TDSE response; got backend={settings.backend!r}"
        )
        raise DriverError(
            msg, parameter=("tdse_execution", "backend"), rule="tdse-numpy-ref"
        )
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


def _keyword_controller(
    route: str, rtol: float | None, atol: float
) -> tuple[PIController, PIController | None]:
    """Return a keyword interface's controller, and it again if its rtol is default.

    - ``route``: the row of :data:`~solprop.driver.defaults.ROUTE_DEFAULTS` the
      interface reads.
    - ``rtol``: the caller's relative tolerance, which is never raised silently;
      ``None`` takes the route's (:func:`~solprop.driver.defaults.default_rtol`).
    - Returns: the controller, and the same object where `rtol` was the route's default
      so that the settings record the default, else ``None``.
    """
    if rtol is not None:
        return PIController(rtol=rtol, atol=atol), None
    controller = PIController(rtol=default_rtol(route), atol=atol)
    return controller, controller


def _stepper_record(settings: StepperSettings) -> StepperRecord:
    """Return the run's stepper record, naming an Adams integrator where one steps it.

    The method is recorded in full, and the interval cap and start-up rule with the
    limits, so that a run and its resume hash the stepping they were taken with.
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


def metadata_for(
    problem: PropagationProblem,
    request: RunRequest,
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
    defaults: DefaultsResolution | None = None,
) -> RunMetadata:
    """Build the record that says what produced a run, and whether it may be resumed.

    Everything that describes *what is computed* goes into the digest
    :meth:`solprop.output.schema.RunMetadata.configuration_hash` takes -- the grid, the
    modes, the configuration, the noise and the stepper settings, plus the save planes
    and the statistics, which are read off the run's request here -- and nothing that
    describes *when* or *with which build*, so a resume survives an unrelated commit or
    a NumPy upgrade.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem the run will step.
    request : solprop.translate.run_request.RunRequest
        The request translation returned with it: its stepper settings, save planes
        and statistics enter the hash exactly as they did when the problem carried
        them.
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
    defaults : solprop.output.schema.DefaultsResolution or None, optional
        Which stepping choices took the route's default. Outside the hash.

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
    >>> problem, request = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-14),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> metadata = metadata_for(problem, request, grid, modes=[gnlse.mode])
    >>> metadata.simulation_kind, len(metadata.configuration_hash())
    ('mode_averaged', 64)
    """
    settings = request.stepper
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
    # The planes and the statistics come from the request rather than from the caller:
    # a resume continues at a plane index, so two runs that differ in either are not the
    # same run, and neither is carried by StepperRecord.
    record["save_planes"] = np.asarray(settings.save_planes, dtype=np.float64)
    record["statistics"] = request.observation.stats
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
            if name in request.observation.stats.names
            or name in request.observation.stats.periodic
        }
        record["envelope_plasma"] = assembly_record(
            problem.responses.envelope_plasma, problem.grids.omega
        )
    if problem.responses.plasma_integral.policy != "trapezoid":
        # The band rule's taps and policies (SEP 18, Q9): a resume that would
        # integrate the plasma differently is refused through the hash.
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
        record["mixture_processing"] = processing_record(
            problem, request.observation.save_planes, grid, material, modes
        )
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
            "flux_plan": flux_record(flux_of(problem.route)),
        }
    descriptor = tdse_descriptor(problem.route)
    tdse_record = None if descriptor is None else dataclasses.asdict(descriptor)
    design = transverse_of(problem.route)
    return RunMetadata(
        reference=_plane_reference_record(
            problem, request, grid, record, frame=frame, modes=modes, modal=modal
        ),
        transverse=None if design is None else design.record(),
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
        defaults=defaults,
        spatial=spatial_record,
        quadrature=None
        if quadrature is None or not modes
        else quadrature.record(modes[0].dimlimits(0.0)),
        modal=None if modal is None else dict(modal),
        materials=MaterialManifest() if materials is None else materials,
        initial_z=problem.initial.z,
        tdse=tdse_record,
    )


def _frame_follows_the_electronic_rule(
    responses: Sequence[object], grid: TimeGrid
) -> bool:
    """Return whether a run's co-moving frame changed when it stopped reading Raman.

    Before the frame followed
    :func:`solprop.translate.responses.declares_harmonic_carrier`, any response's
    ``thg`` kept ``beta_0`` on an envelope grid. The two rules disagree exactly when a
    response other than the electronic ones declares ``thg=True`` and no electronic one
    does.

    - Returns: True when the earlier rule would have kept ``beta_0`` and this one
      removes it.
    """
    return (
        grid.kind == "envelope"
        and any(bool(getattr(one, "thg", False)) for one in responses)
        and not declares_harmonic_carrier(responses, grid)
    )


def _plane_reference_record(
    problem: PropagationProblem,
    request: RunRequest,
    grid: TimeGrid,
    configuration: Mapping[str, object],
    *,
    frame: Frame | None,
    modes: Sequence[Mode],
    modal: Mapping[str, JsonValue] | None,
) -> dict[str, object]:
    """Return the plane-reference record of a run (SEP 31).

    - ``configuration``: the run's configuration; a ``coupling`` entry supplies the
      inherited origin and the parent's provenance, and a ``spatial_grid`` the
      transverse basis.
    - ``frame``: the run's frame, or ``None`` for the co-moving one.
    - Returns: ``carrier``, ``origin``, ``parent``, ``frame`` and ``basis``, as
      :attr:`solprop.output.schema.RunMetadata.reference` documents them.
    """
    # The co-moving frame keeps the carrier phase exactly when a harmonic term is
    # present, and then measures its delay from zero frequency (`frame_origin`).
    # `assemble` chose it from the declared responses and checked that choice against
    # the compiled carrier, so reading the compiled one here records the same frame.
    if frame is None:
        frame = Frame.comoving(grid, thg=problem.responses.harmonic_carrier)
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
        "basis": _plane_bases(problem, request, grid, configuration, modes, modal),
    }


def _plane_bases(  # noqa: PLR0917 - called positionally
    problem: PropagationProblem,
    request: RunRequest,
    grid: TimeGrid,
    configuration: Mapping[str, object],
    modes: Sequence[Mode],
    modal: Mapping[str, JsonValue] | None,
) -> dict[str, object]:
    """Return a run's route and its basis identity at each planned plane.

    A spatial basis is its transverse grid and a general one its checked reference,
    the same at every plane; an ordinary mode set's basis is evaluated at each plane
    (:func:`solprop.coupling.plane.modal_basis_identity`), because a guide may vary,
    and once for a set whose every mode is ``z_constant``.

    - ``configuration``: the run's configuration, carrying a spatial run's
      ``spatial_grid``.
    - Returns: ``route``, ``z`` (the launch plane first, then the save planes, m) and
      ``identity`` (one digest per plane, ``None`` on a planar run).
    """
    planes = [
        float(problem.initial.z),
        *map(float, request.observation.save_planes),
    ]
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
    elif all(mode.z_constant for mode in modes):
        # A guide that is the same at every plane has one basis there: evaluated once,
        # since a solved mode (a step-index fibre's) costs a mode solve per evaluation.
        route = "modal"
        identities = [modal_basis_identity(modes, grid, planes[0])] * len(planes)
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
    """

    @functools.wraps(build)
    def inside(*args: P.args, **kwargs: P.kwargs) -> R:
        """Build the simulation with the run's material context open."""
        with _optmat.material_context():
            return build(*args, **kwargs)

    return inside


@contextlib.contextmanager
def _about(
    parameter: ParameterPath | None = None,
    renamed: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """Name the keyword a refusal raised in the block concerns (SEP 41, S10).

    An object built from a keyword's value refuses it by its own argument's name, or by
    none; the interface knows which keyword it passed, so it names that one here. The
    exception keeps its object, class, message and traceback.

    - ``parameter``: the path given to a :class:`~solprop.exceptions.SolpropError` that
      names no parameter; ``None`` leaves it unnamed.
    - ``renamed``: the inner call's argument names that differ from the keywords that
      supplied them, each mapped to its keyword: a path the error names already is
      renamed through it (each path of a conflict), and kept where it has no entry.
    """
    try:
        yield
    except SolpropError as error:
        if error.parameter is None:
            error.parameter = parameter
        elif renamed:
            error.parameter = (
                renamed.get(error.parameter, error.parameter)
                if isinstance(error.parameter, str)
                else tuple(renamed.get(path, path) for path in error.parameter)
            )
        raise


_REISSUED: Final[dict[str | tuple[str, type[Warning], int], int]] = {}
"""The registry of the warnings :func:`_warnings_about` issues again.

It stands in for the issuing module's own, so that a ``"default"`` filter still shows
a warning once per location and text rather than once per run.
"""


@contextlib.contextmanager
def _warnings_about(
    parameter: ParameterPath, categories: tuple[type[SolpropWarning], ...]
) -> Iterator[None]:
    """Name the keyword the value warnings issued in the block concern (SEP 41, S10).

    A material model warns deep inside the physics, where the keyword that chose the
    material is not known. The block's warnings are recorded and, when it ends, issued
    again in order at their own locations and through the caller's filters, so that an
    ``"error"`` filter still raises and an ``"ignore"`` one still hides them: a warning
    of one of `categories` that names no parameter is given `parameter` first, and
    every other warning is issued as it was.
    """
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
    finally:
        for record in caught:
            # A recorded warning is an instance of its category, never a string.
            message = cast("Warning", record.message)
            if isinstance(message, categories) and message.parameter is None:
                message.parameter = parameter
            warnings.warn_explicit(
                message,
                record.category,
                record.filename,
                record.lineno,
                registry=_REISSUED,
                source=record.source,
            )


_MATERIAL_WARNINGS: Final = (MaterialRangeWarning, MaterialStatusWarning)
"""The warnings a material model issues about the conditions it is evaluated at."""

_GRID_KEYWORDS: Final[Mapping[str, str]] = {"reference_wavelength": "wavelength"}
"""The grid constructors' argument names that a keyword route spells otherwise."""

_PULSE_KEYWORDS: Final[Mapping[str, str]] = {"phase": "phases"}
"""The pulse constructors' argument names that a keyword route spells otherwise."""


def _path(*names: str) -> ParameterPath:
    """Return the path of one keyword, or the tuple of several in conflict.

    - ``*names``: the keywords, at least one.
    """
    return names[0] if len(names) == 1 else names


def _unset(**keywords: object) -> ParameterPath:
    """Return the path of the keywords among `keywords` left unset.

    - ``**keywords``: the keywords a refusal needs, by name; at least one is ``None``.
    """
    return _path(*(name for name, value in keywords.items() if value is None))


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
    """

    handler: Output
    state: ResumeState | None
    metadata: RunMetadata


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
        metadata as itself. An array is copied here, so that changing the caller's
        array afterwards changes neither the run nor its record: once in complex128,
        which runs, and, for another dtype, once more as given, which is recorded;
        ``Simulation.initial`` is a read-only view of the copy as given. A broadcast
        view (:func:`numpy.broadcast_to`) copies only the block it repeats. On a
        free-space or spatial route the translated problem holds the same complex128
        copy rather than another; on a modal route it holds the launch converted to
        its internal amplitude beside the simulation's copy, two states in all.
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
        ``linear.coupled_flow``, which ``numpy_ref`` and ``native`` execute.
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
    (2, (1, 1, 1024))
    """

    # How a constructor that drew the noise without a seed draws it again from a given
    # one: set per instance by `_redrawable` and carried to a run derived with its
    # noise kept (`_rebound_noise`), read by `_reseed`.
    _noise_redraw: Callable[[int | None], NoiseRealisation | None] | None = None
    # The recipe the simulation was built from (SEP 41, S2), set by
    # solprop.recipe.Recipe.build and dropped by every change it would not describe.
    _recipe: RecipeSource | None = None

    def __init__(
        self,
        grid: TimeGrid,
        modes: Mode | Sequence[Mode],
        responses: Sequence[object],
        initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray,
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
        self._problem: PropagationProblem | None = None
        # The request translation returned with the problem: every policy of a fresh
        # run of it (SEP 47 section 5.2).
        self._request: RunRequest | None = None
        # The transverse edge's smallest band wavenumber by run length, fixed with the
        # medium and grids: a planar run's costs a slab-mode evaluation per mode, which
        # every run repeated (``_transverse_edge``).
        self._edge_k_min: dict[float, float] = {}
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
        self._input_identity: str | None = None
        self._record_keys: set[str] | None = None
        self._field0 = self._built_input(initial, grid, initial_z)
        if isinstance(initial, np.ndarray):
            # What was passed is what runs and what is recorded: a read-only view of
            # the simulation's own copy, as the caller gave it (`_given_input`).
            self.initial = _given_input(initial, self._field0)

    def _built_input(
        self,
        initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray,
        grid: TimeGrid,
        initial_z: float,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the input as the simulation holds it, building a pulse on its grids.

        A spatial pulse is built on the spatial grid in the medium, in the simulation's
        frame, which it sets to the comoving frame when none was given.

        - ``initial_z``: the launch position, m.
        - Returns: the spectrum, complex128, read-only and the simulation's own
          (`_built`).
        """
        if not isinstance(initial, SpatialPulse | SpatialPulseSum):
            return _built(initial, grid)
        if self.spatial_grid is None or self.medium is None:
            msg = "a SpatialPulse requires spatial_grid and medium"
            raise DriverError(msg)
        if self.frame is None:
            self.frame = Frame.comoving(
                grid, thg=declares_harmonic_carrier(self.responses, grid)
            )
        return _built_state(
            initial,
            initial.build(
                grid,
                self.spatial_grid,
                medium=self.medium,
                z=initial_z,
                frame=self.frame,
            ),
        )

    @property
    def settings(self) -> SimulationSettings:  # numpydoc ignore=RT01
        """How the run steps, records and executes; fixed once translated.

        Assigning a new :class:`SimulationSettings` before the first :meth:`translate`
        (or :meth:`run`) takes effect. After it the translated problem is cached, so an
        assignment that changes anything the translation consumed is refused, because
        it could no longer change the run; the ``backend`` and ``tdse_execution``
        resources, which are read when the run executes, may still change.

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
        if self._recipe is not None:
            dropped = [
                one.name
                for one in dataclasses.fields(SimulationSettings)
                if one.name not in _RECIPE_EXECUTION
                and not one.name.startswith("_")
                and not _same_setting(
                    getattr(self._settings, one.name), getattr(settings, one.name)
                )
            ]
            if dropped:
                self._drop_recipe(f"its {', '.join(dropped)} changed")
        if hasattr(self, "noise"):
            _check_initial_position(self, settings)
            _check_tdse_execution(self, settings)
            if _quiet_collar(settings) != _quiet_collar(self._settings):
                # The opt-in is part of the resume key: the record is built again.
                self._metadata = None
        self._settings = settings

    def _kind(self) -> str:
        """Return what kind of run this is, for printing."""
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

        - Returns: the title and ``(key, value)`` rows.
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
                    (
                        "none"
                        if self.noise_resolution.reason is None
                        else f"none ({self.noise_resolution.reason})"
                    )
                    if self.noise is None
                    else f"{self.noise.model} model, seed {self.noise.seed}",
                ),
                ("settings", str(self.settings)),
            ]
        )
        return f"Simulation: {self._kind()}", rows

    def translate(self) -> PropagationProblem:
        """Assemble the kernel problem, once, and return it.

        Everything expensive about a run that does not depend on the state happens here
        (``docs/dev/architecture.md`` section 7): the cumulative linear operator, the
        masks, the modal transform and the compiled responses. The result is cached, so
        two runs of the same simulation translate once.

        Returns
        -------
        solprop.translate.problem.PropagationProblem
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
        ((1, 1, 768), True)
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
                self._problem, self._request = assemble(
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
                )
        return self._problem

    def _run_request(self) -> RunRequest:
        """Return the request translation returned with the problem, translating once.

        Every policy of a fresh run of :meth:`translate`'s problem: the method, the
        error control, the itinerary, the planes and the statistics (SEP 47 section
        5.2). A run adapts it to its start, retention, monitor and history
        (:meth:`~solprop.translate.run_request.RunRequest.for_run`).
        """
        self.translate()
        assert self._request is not None  # set with the problem
        return self._request

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
        initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray | CouplingLaunch,
        *,
        noise: NoiseRealisation | Literal["keep"] | None = "keep",
        initial_z: float | None = None,
        configuration: Mapping[str, object] | None = None,
        input_identity: str | None = None,
    ) -> "Simulation":
        """Replace the input while sharing the translated propagation problem.

        Only what changed is validated (SEP 22 section 5.2): the replaced input's
        shape, dtype and one finiteness pass, and the replaced noise's, with every
        other leaf of the translated problem shared by identity rather than scanned
        again. A new `initial_z`, which changes the save planes, also keeps the
        stepper's checks: the launch before the first plane, and a replayed schedule
        containing every plane. An array input above
        :data:`~solprop.output.schema.INLINE_ARRAY_LIMIT` elements is recorded by its
        digest (:class:`~solprop.output.schema.ArrayDigest`), not inline, unless its
        identity is stated (`input_identity`).

        Parameters
        ----------
        initial : Pulse, SpatialPulse, numpy.ndarray or solprop.coupling.CouplingLaunch
            Pulse or internal-convention spectrum on the existing grid. The built
            shape must match the original input, including modes and polarisation.
            Modal amplitudes are in sqrt(W); spatial units follow the spatial grid.
            An array is copied when the derived simulation is built, as by the
            constructor.
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
        input_identity : str or None, optional
            What an array input is, stated by the caller who already knows: the record
            then holds this string, the dtype and the shape
            (:class:`~solprop.output.schema.StatedInput`) in place of the array's own
            record, and its bytes are not digested. **It is not checked**: two
            different arrays stated alike have one configuration hash, so a resume or
            a scan's cache accepts one for the other. ``None``, the default, records
            the array as itself or by its digest. Refused for anything but an array.

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
            `configuration` key is one the run records itself, a launch is bound
            to another port or given beside an `initial_z`, or `input_identity` is
            empty or names anything but an array input.

        Warns
        -----
        MaterialRangeWarning
            Once per run, when the new input's peak exceeds an optmat rate's maximum
            field, or the input is not linearly polarised at a Keldysh parameter
            above one there (SEP 3, D12).
        """
        if isinstance(noise, str) and noise != "keep":
            raise DriverError("noise must be keep, None, or a NoiseRealisation")
        _check_input_identity(input_identity, initial)
        coupling = None
        if isinstance(initial, CouplingLaunch):
            launch = initial
            initial, initial_z, noise = self._launched(launch, initial_z, noise)
            coupling = launch.record()
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
            if isinstance(noise, str) or noise is self.noise
            else NoiseResolution(
                None if noise is None else noise.model,
                None if noise is None else noise.model,
            )
        )
        if (
            isinstance(noise, str)
            and launch_z > 0
            and self.noise_resolution.requested == "default"
            and self.noise is not None
        ):
            # SEP 37: a default draw is not added at an interior launch, whose given
            # spectrum may already carry the realisation of an earlier run.
            replacement_noise = None
            resolution = NoiseResolution("default", None, INTERIOR_LAUNCH_NOISE)
        # An array is refused before the derived simulation copies it.
        self._check_replaced_array(
            initial, None if isinstance(noise, str) else replacement_noise
        )
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
            spatial_grid=self.spatial_grid,
            medium=self.medium,
            frame=self.frame,
            general=self.general,
            guide=self.guide,
            wall=self.wall,
            taper_coupling=self.taper_coupling,
            materials=self.materials,
        )
        # Recorded from what the derived run holds, not from the caller's input.
        updated.configuration = self._derived_configuration(
            updated.initial, configuration, coupling
        )
        updated._input_identity = input_identity
        # A derived run starts from what its parent's run has warned and recorded,
        # and keeps its own from here: its siblings never see each other's.
        updated._material_log = self._material_log.copy()
        if isinstance(noise, str) and replacement_noise is not None:
            updated._noise_redraw = self._noise_redraw
        if self._recipe is not None:
            self._warn_recipe_dropped("with_input")
        if not isinstance(initial, np.ndarray):
            self._check_replaced_input(
                updated._field0, None if isinstance(noise, str) else replacement_noise
            )
        record = dataclasses.replace(
            original.initial,
            z=launch_z,
            field0=_replaced_launch(original, updated._field0, launch_z),
        )
        updated._problem = _with_noise(
            dataclasses.replace(original, initial=record), replacement_noise
        )
        request = self._run_request()
        updated._request = request.with_stepper(
            dataclasses.replace(request.stepper, save_planes=settings.save_planes)
        )
        route = updated._problem.route
        if isinstance(route, SpatialRoute):
            updated._problem = dataclasses.replace(
                updated._problem,
                route=dataclasses.replace(
                    route,
                    transverse=_rechecked(
                        self,
                        original,
                        updated._problem,
                        updated._request.observation.save_planes,
                    ),
                ),
            )
        if initial_z is not None:
            # A new launch changes the save planes: the launch must precede them, and
            # a replayed schedule contain them (SEP 22 section 5.2).
            updated._problem.validate_launch(updated._request)
        updated._mixture_selection = self._mixture_selection
        with _optmat.material_context(updated._material_log):
            _check_rate_inputs(updated, updated.initial)
        return updated

    def _launched(
        self,
        launch: CouplingLaunch,
        initial_z: float | None,
        noise: NoiseRealisation | Literal["keep"] | None,
    ) -> tuple[np.ndarray, float | None, NoiseRealisation | None]:
        """Return what a coupling launch means for :meth:`with_input`, or refuse it.

        - ``noise``: the noise asked for: a launch carries its source's realised field,
          so ``'keep'`` draws no second floor.
        - Returns: the launch's spectrum, its interior position (``None`` at this run's
          own entrance) and its noise.
        """
        if initial_z is not None:
            msg = "a launch carries its own entrance: give no initial_z beside it"
            raise DriverError(msg)
        entrance = launch.port.geometry.z
        interior = None if entrance == self.initial_z else entrance
        refusal = self.input_port(interior).accepts(launch.port)
        if refusal is not None:
            msg = f"this launch was computed for another input port: {refusal}"
            raise DriverError(msg)
        return launch.internal(), interior, None if isinstance(noise, str) else noise

    def _derived_configuration(
        self,
        initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray,
        configuration: Mapping[str, object] | None,
        coupling: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Return a derived run's extra configuration: this run's, the caller's keys.

        - ``coupling``: the provenance of a coupling launch, recorded as ``coupling``.
        """
        recorded = dict(self.configuration)
        # Nor does the constructor's incident-beam account describe it, nor a
        # coupling that launched this run.
        recorded.pop("launch", None)
        recorded.pop("coupling", None)
        # Nor does a modal launch describe a replacement input, which is placed as
        # given.
        recorded.pop("modal_launch", None)
        # A constructor's pulse recipe must never describe a replacement spectrum.
        # Arrays carry their exact input in metadata, without an inferred recipe.
        if "pulse" in recorded:
            recorded.pop("pulse")
            if not isinstance(initial, np.ndarray):
                recorded["pulse"] = initial
        if configuration:
            clash = sorted(set(configuration) & ({"coupling"} | self._recorded_keys()))
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

    def _check_replaced_array(
        self,
        initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray,
        noise: NoiseRealisation | None,
    ) -> None:
        """Validate a replaced array input before it is copied; a pulse is built first.

        - ``noise``: a replacement realisation, or ``None`` when the noise is kept or
          removed.
        """
        if isinstance(initial, np.ndarray):
            self._check_replaced_input(initial, noise)

    def _check_replaced_input(
        self,
        field0: Shaped[np.ndarray, "*shape"],
        noise: NoiseRealisation | None,
    ) -> None:
        """Validate a replaced input alone: its shape and one finiteness pass.

        Every other leaf of the derived problem is this one's, shared by identity and
        validated when it was translated (SEP 22 section 5.2); the dtype is fixed by
        the build.

        - ``noise``: a replacement realisation, or ``None`` when the noise is kept or
          removed.
        """
        if field0.shape != self._field0.shape:
            raise DriverError("with_input requires the existing mode and grid shape")
        if not _all_finite(field0):
            raise DriverError("with_input requires a finite initial field")
        if noise is None:
            return
        if noise.spectrum.shape != self._field0.shape:
            raise DriverError("replacement noise must match the input field shape")
        if not _all_finite(noise.spectrum):
            raise DriverError("with_input requires a finite noise realisation")

    def runner(self) -> "SimulationRunner":
        """Return a runner that keeps this translation and one backend for many inputs.

        Translates (or reuses the cached translation) and constructs the backend once;
        each :meth:`SimulationRunner.run` then runs one input on it (SEP 22 section
        5.2). Close it, or use it in a ``with`` block.

        Returns
        -------
        SimulationRunner
            The open runner.

        Raises
        ------
        DriverError
            If the run has a TDSE response.
        """
        return SimulationRunner(self)

    def _drop_recipe(self, reason: str) -> None:
        """Forget the recipe, which no longer describes this simulation, and warn."""
        self._recipe = None
        msg = (
            f"this Simulation was built from a recipe, but {reason}, which the recipe "
            "does not describe: its output will record no recipe, and resuming it "
            "needs this script"
        )
        warnings.warn(msg, SolpropWarning, stacklevel=3)

    def _warn_recipe_dropped(self, mutator: str) -> None:
        """Warn that a derived simulation does not carry this one's recipe."""
        msg = (
            f"{mutator} derives a simulation the recipe this one was built from does "
            "not describe: its output will record no recipe, and resuming it needs "
            "the script"
        )
        warnings.warn(msg, SolpropWarning, stacklevel=3)

    def with_mixture(self, mixture: Mixture) -> "Simulation":
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
        if self._recipe is not None:
            self._warn_recipe_dropped("with_mixture")
        updated.initial = self.initial
        updated._input_identity = self._input_identity
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
            self._metadata = self._record(self._recorded_input())
        return self._metadata

    def _recorded_input(self) -> object:
        """Return the input as this run's record holds it, digesting a large array."""
        if self._input_identity is not None:
            return StatedInput.of(
                cast("np.ndarray", self.initial), self._input_identity
            )
        # A gigabyte array is recorded by its digest (SEP 22, Q4).
        return recorded_input(self.initial)

    def _recorded_keys(self) -> set[str]:
        """Return the keys of this run's configuration record, digesting nothing.

        What a derived run's caller keys are checked against, which a parent that is
        only a placeholder for its inputs should not pay a digest of itself for.
        """
        if self._record_keys is None:
            record = self._metadata or self._record(None)
            self._record_keys = set(record.configuration)
        return self._record_keys

    def _record(self, initial: object) -> RunMetadata:
        """Build this run's record around what it says of its input."""
        configuration = dict(self.configuration)
        configuration.update(
            {
                "responses": list(self.responses),
                "initial": initial,
                "profiles": dict(self.profiles),
                "density": self.settings.density,
                "shock": self.settings.shock,
                "prefactor_index": self.settings.prefactor_index,
                "absorber": self.settings.absorber,
                "absorber_scheme": _scheme(self.settings),
                "propagator_mode": self.settings.propagator_mode,
            }
        )
        if self.frame is not None:
            configuration["frame"] = self.frame
        elif _frame_follows_the_electronic_rule(self.responses, self.grid):
            # A Raman thg flag kept beta_0 on an envelope grid before the frame
            # followed the electronic responses alone (issue #77). Recording the
            # co-moving frame here, and only here, refuses a resume of such an
            # earlier run by name, which would otherwise continue under another
            # frame with the same digest; every other run keeps its digest.
            configuration["frame"] = Frame.comoving(self.grid)
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
            return metadata_for(
                self.translate(),
                self._run_request(),
                self.grid,
                modes=self.modes,
                configuration=configuration,
                noise=self.noise,
                modal=self._modal_identity(),
                materials=self.material_manifest(),
                quadrature=self.settings.quadrature,
                frame=self.frame,
                noise_resolution=self.noise_resolution,
                defaults=self.settings.defaults_resolution(),
            )

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
            vector = problem.transform.n_pol * problem.shape[1] > 1
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

        - Returns: the ``modal`` record, or ``None``.
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
        return wall_identity(
            flow,
            limits,
            channels=[mode.channel_label for mode in self.modes],
            impedance=self.wall.impedance,
            scored=self.translate().grids.inband_mask,
            n_components=self.translate().transform.n_pol,
            has_noise=self.noise is not None,
        )

    def _boundary_monitor(self) -> BoundaryMonitor | None:
        """Return the boundary monitor of a fresh run, or ``None`` without a check."""
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
        """Return the transverse edge a free-space run's monitor reads (SEP 40 5.3)."""
        problem = self.translate()
        planes = self._run_request().observation.save_planes
        length = float(np.max(planes)) - float(problem.initial.z)
        k_min = self._edge_k_min.get(length)
        if k_min is None:
            dispersion = SpatialDispersion(medium, self.grid, Frame.comoving(self.grid))
            k, _ = band_wavenumbers(space, dispersion, max(length, 0.0))
            k_min = self._edge_k_min[length] = float(np.min(k))
        grid = self.grid
        return TransverseEdge(
            space,
            transverse_of(problem.route),
            check.edge_budget,
            check.stop_factor,
            length,
            np.nonzero(grid.inband_mask & grid.active_mask)[0],
            k_min,
        )

    def run(
        self,
        output: Output | Path | str | None = None,
        *,
        progress: ProgressReporter | ProgressChoice | None = None,
        progress_period: float = DEFAULT_PROGRESS_PERIOD,
        resume: _ResumeChoice = False,
        tdse_archive: TDSEArchive | None = None,
        snapshots: SnapshotController | None = None,
        stop: StopFlag | None = None,
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
        resume : bool or {'interpolated'}, optional
            Continue the file at `output` instead of replacing it, when its record and
            this run's have one configuration hash (``docs/guide/results/resuming.md``,
            ADR-066); otherwise it is refused, naming the fields that differ. A finished
            file is returned. Only a path can be resumed: the driver has to open the
            file in append mode itself. Noise a constructor drew without a seed is drawn
            again from the seed the file recorded, so re-running the same call continues
            its file. A file whose last plane was interpolated inside a step is refused
            unless this is ``"interpolated"``: it then continues afresh from the
            interpolated field, and the position is recorded in the annotation
            :data:`~solprop.output.schema.INTERPOLATED_RESUMES_ANNOTATION`.
        tdse_archive : solprop.output.TDSEArchive or None, optional
            Archive every raw TDSE provider evaluation of a run with a TDSE
            response in this separate HDF5 file (SEP 20 section 5.8); ``None``
            archives nothing. It is opened and validated before any atomic work, a
            resume appends a new session to its own run's archive, and the optical
            output records the archive's run UUID and relative path. It changes
            neither the propagation nor the call ledger.
        snapshots : solprop.output.SnapshotController or None, optional
            Publish immutable local checkpoints without rebuilding the backend or
            atomic provider, each at the next accepted endpoint, after every
            interpolated plane its step crossed. Requires HDF5 and ``fields='all'``.
            Adams history cannot be explicitly disabled. A configured archive is
            copied together with the optical file.
        stop : object or None, optional
            This run's own stop flag: any object with an ``is_set()`` method, such as
            a :class:`threading.Event` (SEP 41, S17). It is polled before the first
            step and before every trial step, so the run answers within one step or
            trial; on a native ``Multistep()`` march, whose trials run in native calls
            that cannot read it, before every call, so the run answers within 256
            attempts or 0.1 s and the attempt in flight, whichever comes first. When it
            is set the run writes its state as for SIGTERM, tells the
            reporter through ``interrupted``, and returns the handler: a requested
            stop is not an error. The outcome's message says the caller's flag
            stopped it, and ``resume=True`` continues the file. A front end sets it
            from any thread on any platform, without signals; each run polls its own.

        Returns
        -------
        solprop.output.schema.Output
            The handler the run was written to, complete or, after `stop` was set,
            holding the state where it stopped.

        Raises
        ------
        ValueError
            If `resume` is not one of its three values.
        DriverError
            If a resume is asked for and `output` is not a path, or the file's last
            plane was interpolated and `resume` is not ``"interpolated"``; or
            `tdse_archive` is given without a TDSE response or at the
            path of the optical file.
        TDSEArchiveError
            If the archive's path holds another file, or a write to it fails; the
            run then stops at its last optical checkpoint.
        OutputError
            If a file is there and its configuration hash differs; the message names
            what changed.

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
        return self._run(
            output,
            progress=progress,
            progress_period=progress_period,
            resume=resume,
            tdse_archive=tdse_archive,
            snapshots=snapshots,
            stop=stop,
        )

    def _run(  # noqa: PLR0915 - run's controls, and the backend it may be handed
        self,
        output: Output | Path | str | None = None,
        *,
        progress: ProgressReporter | ProgressChoice | None = None,
        progress_period: float = DEFAULT_PROGRESS_PERIOD,
        resume: _ResumeChoice = False,
        tdse_archive: TDSEArchive | None = None,
        snapshots: SnapshotController | None = None,
        stop: StopFlag | None = None,
        kept: Backend | None = None,
    ) -> Output:
        """Execute the run, on a new backend or on one a caller keeps between runs.

        - ``progress_period``: as for :meth:`run`, s.
        - ``kept``: a backend built for this problem but for its input, which is loaded
          and left open for the caller; ``None`` constructs one and closes it. It serves
          a fresh run only, with no TDSE response.
        """
        _check_resume_choice(resume)
        with _optmat.material_context(self._material_log):
            if resume:
                self._adopt_recorded_seed(output)
            problem = self.translate()
            request = self._run_request()
            _refuse_pending_check(problem)
            metadata = self.metadata()
            handler, state = _open_output(output, metadata, resume=resume)
            if state is not None:
                next_plane = _validate_resume(
                    problem, request, state, self.settings.tdse_execution
                )
                if not _continues(
                    Path(cast("Path | str", output)),
                    request,
                    state,
                    next_plane,
                    resume,
                ):
                    handler.open(state.metadata)
                    handler.close()
                    return handler
            _warn_sampled_peak(self.settings.quadrature, request.observation.stats)
            retention = field_retention(
                request, self.settings.fields, self.settings.field_planes
            )
            if tdse_descriptor(problem.route) is not None and not isinstance(
                handler, CheckpointOutput
            ):
                raise DriverError(
                    "TDSE output must support terminal checkpoints and call ledgers"
                )
            _check_tdse_archive(tdse_archive, problem, handler)
            check_snapshot_request(
                request,
                handler,
                snapshots,
                self.settings.resume_history,
                retention.fields,
            )
            if (
                snapshots is not None
                and tdse_descriptor(problem.route) is not None
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
                if tdse_descriptor(problem.route) is not None:
                    binding, archive = self._prepare_atomic(
                        problem,
                        _AtomicContext(handler, state, metadata),
                        tdse_archive,
                        stop,
                    )
                backend = kept or select(
                    self.settings.backend,
                    problem,
                    request,
                    grid=self.grid,
                    modes=self.modes,
                    profiles=self.profiles,
                    tdse_binding=binding,
                    options=self.settings.execution_options(),
                )
                restoration: dict[str, object] = {}
                start = self._start(
                    backend,
                    problem,
                    state,
                    kept=kept is not None,
                    restoration=restoration,
                )
                self._restore_atomic(binding, start)
                self._consistency_gate(binding, problem, handler, metadata, backend)
                run_problem(
                    problem,
                    backend,
                    handler,
                    start=start,
                    metadata=metadata,
                    stats_spec=request.observation.stats,
                    progress=progress,
                    progress_period=progress_period,
                    tdse_binding=binding,
                    resume_history=self.settings.resume_history,
                    fields=retention.fields,
                    field_planes=retention.field_planes,
                    snapshots=snapshots,
                    tdse_archive=archive,
                    boundary_monitor=self._boundary_monitor(),
                    stop=stop,
                    recipe=None if self._recipe is None else self._recipe.record(self),
                    restoration=restoration or None,
                )
            except StopRequested as error:
                failure = error
                # A stop raised by another run's flag is that run's to catch.
                if error.flag is not stop:
                    raise
            except BaseException as error:
                failure = error
                raise
            finally:
                _close_run(
                    None if kept is not None else backend,
                    binding,
                    archive,
                    handler,
                    failure,
                )
            self._peak_fields = _peak_field_reports(problem, handler, self.responses)
        if (
            state is not None
            and _interpolated(state)
            and isinstance(handler, PlaneStore)
        ):
            # Allowed only by resume="interpolated"; the record says where.
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

    def estimate(self) -> "RunEstimate":
        """Estimate what the run would cost, without running it (SEP 41, S11).

        The grid, the persistent memory the constructed backend holds, the output's
        size, the cost of one right-hand-side evaluation on the backend, timed after an
        untimed warm-up that takes compilation and planning, and the integrator's calls
        per step (:mod:`solprop.driver.estimate`). It constructs the backend, so a run
        that doesn't fit in memory fails here as the run would.

        Returns
        -------
        solprop.driver.estimate.RunEstimate
            The estimate.
        """
        from solprop.driver.estimate import estimate  # noqa: PLC0415 - imports this

        return estimate(self)

    def calibrate(self, budget: float = 60.0) -> "RefinementReport":
        """Refine the native backend's settings for this simulation (SEP 46 stage 2).

        Fresh native workspaces of this simulation's translated problem are built in
        this process under candidate settings and timed by trial steps along its
        length; the settings that win, stay accurate and validate together are stored
        under its execution descriptor, and every later run or scan point of the same
        execution applies them. The simulation itself is not changed. Running
        ``solprop calibrate`` first gives the machine profile this refines
        (:mod:`solprop.calibration`).

        Parameters
        ----------
        budget : float, optional
            The deadline, in seconds.

        Returns
        -------
        solprop.calibration.refine.RefinementReport
            What was kept and whether it was stored.
        """
        from solprop.calibration.refine import refine  # noqa: PLC0415 - imports native

        return refine(self, budget)

    def output_bytes(self) -> int:
        """Return the size of the run's output, without constructing a backend.

        The cheap part of :meth:`estimate`, for asking as the setup changes: it times
        nothing and allocates no run state.

        Returns
        -------
        int
            Bytes, as an in-memory store holds them (an HDF5 file adds little).

        Examples
        --------
        >>> from solprop.recipe import Recipe
        >>> simulation = Recipe.example("soliton").build()
        >>> simulation.output_bytes() == simulation.estimate().output_bytes
        True
        """
        from solprop.driver.estimate import output_bytes  # noqa: PLC0415 - imports this

        return output_bytes(self)

    def _prepare_atomic(
        self,
        problem: PropagationProblem,
        context: _AtomicContext,
        tdse_archive: TDSEArchive | None,
        stop: StopFlag | None = None,
    ) -> tuple[TDSEBinding, TDSEArchiveWriter | None]:
        """Open a TDSE run's archive and prepare its binding.

        The binding's waits on its provider worker poll `stop` (SEP 49).

        - Returns: the binding and the open archive session, or None without a request.
          A failure closes the archive before it propagates.
        """
        descriptor = tdse_descriptor(problem.route)
        assert descriptor is not None
        _warn_tdse_cost(self.settings.quadrature)
        execution = self.settings.tdse_execution
        if execution is None:
            raise DriverError("TDSE requires explicit tdse_execution settings")
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
                defer_plans=descriptor.model == "hacc",
                archive=archive,
                stop=stop,
            )
        except BaseException as error:
            _close_tdse_archive(archive, error)
            raise
        return binding, archive

    @staticmethod
    def _restore_atomic(binding: TDSEBinding | None, start: Start) -> None:
        """Restore a resumed run's cumulative atomic accounting into its binding.

        Before anything can charge the binding: a haCC run's consistency gate makes a
        diagnostic provider call when its local cache has no entry, and a budget that
        has begun work refuses to be restored. The start keeps only the logical
        ledger, which the session restores when it loads the start.

        - ``binding``: the run's open atomic binding; None for a run without one.
        """
        ledgers = start.ledgers
        if binding is None or not ledgers or not ledgers.get("atomic"):
            return
        atomic = cast("Mapping[str, object]", ledgers["atomic"])
        binding.restore_ledger(AtomicCallLedger.from_record(atomic))
        start.ledgers = {"logical": ledgers.get("logical", {})}

    def _consistency_gate(
        self,
        binding: TDSEBinding | None,
        problem: PropagationProblem,
        handler: Output,
        metadata: RunMetadata,
        backend: Backend,
    ) -> None:
        """Check a haCC run's weak-field linear response before propagating.

        The cached or computed result is attached to the binding's provenance, under
        ``tdse.consistency``. The binding was prepared with deferred plans, so the
        check's temporary one-row plan is closed before the propagation plans are
        built here. A refused or invalid result is written to the requested
        output, as the opened run record without any plane or checkpoint, before
        :class:`~solprop.exceptions.TDSEConsistencyError` is raised (SEP 20, 5.4).

        - ``binding``: the run's open atomic binding; None for a run without one.
        """
        descriptor = tdse_descriptor(problem.route)
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
                open_run_record(handler, metadata, backend, binding)
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
        """
        if self._noise_redraw is None or self.noise is None or self.noise.seed == seed:
            return False
        self.noise = self._noise_redraw(seed)
        if self._problem is not None:
            self._problem = _with_noise(self._problem, self.noise)
        self._metadata = None
        return True

    def _reseeded(self, seed: int) -> "Simulation":
        """Return this run with unseeded noise drawn from `seed`, leaving it as it is.

        A scan's builder may return one object for every point, and a batch keeps each
        member by reference, so a point's realisation must not land on the object the
        other points share. Translated first, so that they share its translation.

        - Returns: a copy carrying the new realisation and sharing the translation's
          other leaves, with its own material log as any derived run has; or this run,
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

        - ``medium``: the updated run's medium; ``None`` on a modal route.
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

    def _start(
        self,
        backend: Backend,
        problem: PropagationProblem,
        state: ResumeState | None,
        *,
        kept: bool = False,
        restoration: dict[str, object] | None = None,
    ) -> Start:
        """Return where the run starts: the entrance, or the state a file resumes.

        The run's session loads it (SEP 47): the field, the noise, a resume's call
        ledgers and the method history this run continues.

        - ``backend``: the backend about to step, asked only whether its march can carry
          a method history.
        - ``state``: what a file the run is continuing holds, or ``None`` to start at
          the entrance.
        - ``kept``: the backend was built for another input, so this one's noise has to
          be handed to it.
        """
        if state is None:
            return Start.entrance(
                problem, noise=_realisation(problem) if kept else None
            )
        request = self._run_request()
        next_plane = _validate_resume(
            problem, request, state, self.settings.tdse_execution
        )
        logger.info(
            "resuming at z = %.6e m, plane %d of %d",
            state.plane.z,
            state.n_saved,
            request.observation.save_planes.size + 1,
        )
        # A field-free plane resumes only a finished run, which never gets here.
        assert state.plane.field is not None
        field = state.plane.field
        if problem.transform.geometry is None:
            field = power_amplitude(
                field[np.newaxis], power_weight_at(problem, state.plane.z), inverse=True
            )[0]
        integrator = request.integration.method
        record, march = self._continue_march(
            backend,
            state,
            adams=isinstance(integrator, Multistep),
            low_storage=isinstance(integrator, LowStorageRungeKutta),
        )
        if restoration is not None:
            restoration.update(record)
        return Start(
            field=field,
            z=float(state.plane.z),
            plane_index=next_plane,
            march=march,
            ledgers=None
            if state.outcome is None
            else {
                "logical": state.outcome.logical_calls,
                "atomic": state.outcome.atomic_calls,
            },
            resume=True,
        )

    def _continue_march(
        self,
        backend: Backend,
        state: ResumeState,
        *,
        adams: bool = True,
        low_storage: bool = False,
    ) -> tuple[dict[str, object], MarchSnapshot | None]:
        """Decide whether the Adams history continues from the file's snapshot.

        Says what is restored, for the resumed execution record (SEP 47 section 5.5),
        and warns once when an Adams history the run expected is not restored:
        omitted by the automatic size limit, or kept where this backend's march
        cannot carry it. A history set aside on request (``resume_history=False``) is
        recorded without a warning. A history handed on is installed by the run's
        session, which warns and records a reset if it cannot
        (:func:`solprop.driver.run.run_problem`).

        - Returns ``record``: ``"state"`` (``"restored"``, ``"reset"`` or
          ``"not_applicable"``), ``"reason"`` and ``"z"``, m; for an Adams method also
          ``"retention"``, what the file states it kept there, ``{"state": "unknown"}``
          where it states nothing.
        - Returns ``march``: the history the run's session is to install, or ``None``.
        """
        z = state.plane.z
        if low_storage:
            record = {"state": "not_applicable", "reason": "no_method_history", "z": z}
            return record, None
        if not adams:
            # The proposal and the norm memory start afresh (issue #596).
            return {"state": "reset", "reason": "unsupported_method", "z": z}, None
        execution = None if state.outcome is None else state.outcome.execution
        recorded = retention_at(execution, z)
        retention: dict[str, object] = (
            dict(recorded)
            if recorded is not None
            else {
                "state": "unknown",
                # A file from before the restart record, or one that states nothing
                # at this position: never an invented reason.
                "reason": "legacy"
                if execution is None or RESTART_RECORD not in execution
                else "not_recorded",
            }
        )

        def reset(reason: str) -> dict[str, object]:
            """Return the restoration record of a history that starts afresh.

            - Returns: the record's ``"state"``, ``"reason"``, ``"z"`` and
              ``"retention"``.
            """
            return {"state": "reset", "reason": reason, "z": z, "retention": retention}

        if self.settings.resume_history is False:
            if state.march is not None:
                logger.info("the Adams history starts afresh: resume_history=False")
            return reset("explicit_policy"), None
        # A march that cannot carry its history restarts it without having been asked
        # to: say so once, as a warning (SEP 29 section 6.6).
        refusal = backend.march_export_refusal()
        if refusal is not None:
            warnings.warn(
                f"the Adams history starts afresh: {refusal}",
                SolpropWarning,
                stacklevel=2,
            )
            return reset("incompatible_representation"), None
        if state.march is None:
            return self._missing_history(retention, reset), None
        restored: dict[str, object] = {
            "state": "restored",
            "reason": "retained_history",
            "z": z,
            "retention": retention,
        }
        return restored, state.march

    @staticmethod
    def _missing_history(
        retention: Mapping[str, object],
        reset: Callable[[str], dict[str, object]],
    ) -> dict[str, object]:
        """Record, and warn about, an Adams resume whose file holds no history."""
        kept, why = retention.get("state"), retention.get("reason")
        if kept == "omitted" and why == "automatic_size_limit":
            size = retention.get("nbytes")
            limit = retention.get("limit")
            warnings.warn(
                "the Adams history starts afresh: the run did not keep it for a "
                "resume, "
                + ("" if size is None else f"{size} bytes being ")
                + "over the "
                + ("automatic limit" if limit is None else f"{limit} bytes")
                + " kept by default (resume_history=True keeps it)",
                SolpropWarning,
                stacklevel=3,
            )
            return reset("automatic_size_limit")
        lost: tuple[str, str] | None = None
        if kept == "retained":
            lost = ("kept it here, but it cannot be read back", "unavailable_history")
        elif kept == "omitted" and why == "write_failed":
            lost = ("was keeping it here when the write failed", "unavailable_history")
        elif kept == "omitted" and why == "incompatible_representation":
            lost = (
                "could not keep it: the run's engine does not export it",
                "incompatible_representation",
            )
        if lost is not None:
            warnings.warn(
                f"the Adams history starts afresh: the file {lost[0]}",
                SolpropWarning,
                stacklevel=3,
            )
            return reset(lost[1])
        if kept == "omitted" and why == "explicit_policy":
            return reset("explicit_policy")
        # Nothing to continue (a position the march held no history at), or a file
        # that does not say: as before, the history starts afresh without a warning.
        return reset("unavailable_history")

    @classmethod
    @_one_material_context
    def from_radial(
        cls,
        grid: TimeGrid,
        space: RadialGrid,
        medium: SpatialMedium,
        responses: Sequence[object],
        initial: SpatialPulse | SpatialPulseSum | np.ndarray,
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
    ) -> "Simulation":
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
        initial : SpatialPulse, SpatialPulseSum or numpy.ndarray
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
    def from_free(
        cls,
        grid: TimeGrid,
        space: FreeGrid,
        medium: SpatialMedium,
        responses: Sequence[object],
        initial: SpatialPulse | SpatialPulseSum | np.ndarray,
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
    ) -> "Simulation":
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
        initial : SpatialPulse, SpatialPulseSum or numpy.ndarray
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
            grid and records why (:data:`~solprop.driver.defaults.FREE_NOISE`, SEP
            37); pass ``"input"`` or ``"modified"`` for a noise model.
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
    def from_planar(
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
    ) -> "Simulation":
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
                grid, thg=declares_harmonic_carrier(responses, grid)
            )
        if isinstance(initial, np.ndarray):
            # Refused before the simulation copies it.
            _check_planar_input(initial.shape, space.n_columns, grid.n_omega)
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
        _check_planar_input(simulation._field0.shape, space.n_columns, grid.n_omega)
        _bind_spatial_rates(simulation, initial)
        return simulation

    @classmethod
    def _from_spatial(
        cls,
        grid: TimeGrid,
        space: RadialGrid | FreeGrid,
        medium: SpatialMedium,
        responses: Sequence[object],
        initial: SpatialPulse | SpatialPulseSum | np.ndarray,
        *,
        settings: SimulationSettings,
        frame: Frame | None = None,
        noise: NoiseSetting = "default",
        seed: int | None = None,
        materials: MaterialSelection | None = None,
    ) -> "Simulation":
        """Build a radial or Cartesian propagation using physical fields.

        - ``space``: radial nodes, outer radius and spatial collar, or the Cartesian
          axes and their collar.
        - ``initial``: beam specification or internal spectrum, a Hankel spectrum in V m
          or a raw Cartesian DFT spectrum in V/m, with shape ``(n_columns, n_pol,
          n_omega)``.
        - ``settings``: step tolerances, positive save positions in m, statistics and
          backend. Absolute tolerance has the input spectrum's units.
        - ``frame``: moving frame. None uses the declared responses' THG choice.
        - ``seed``: random generator seed; None draws and records system entropy.
        """
        chosen_frame = frame
        if chosen_frame is None:
            chosen_frame = Frame.comoving(
                grid, thg=declares_harmonic_carrier(responses, grid)
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
        if isinstance(initial, np.ndarray):
            # Refused before the simulation copies it.
            _check_spatial_input(initial.shape, space.n_columns, grid.n_omega)
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
        _check_spatial_input(simulation._field0.shape, space.n_columns, grid.n_omega)
        noise_model, resolution = resolve_noise(
            "radial" if isinstance(space, RadialGrid) else "free",
            noise,
            seed,
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
    def from_gnlse(
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
        step_planes: int | None = None,
        save_planes: Sequence[float] | np.ndarray | None = None,
        n_save: int | None = None,
        stats: StatsSpec | None = None,
        polarisation_taper: bool = False,
        absorber: Absorber | Literal[False] | None = None,
        absorber_scheme: AbsorberScheme | None = None,
        boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK,
        backend: str = "auto",
        materials: MaterialSelection | None = None,
    ) -> "Simulation":
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
        step_planes : int or None, optional
            Also save the end of every `step_planes`-th accepted step, the
            march's own positions, beside the requested planes (SEP 14);
            ``None``, the default, and ``0`` save the requested planes only.
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
            Registry name, or ``"auto"`` (the default) for ``native``, which refuses if
            it isn't installed or this processor is below its baseline. There is no
            fallback, and import or construction errors stay visible.
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
        controller, default = _keyword_controller("gnlse", rtol, atol)
        settings = SimulationSettings(
            controller=controller,
            _default_controller=default,
            _route="gnlse",
            save_planes=_save_planes(length, save_planes, n_save),
            # Relative defaults, so that the first step and the cap scale with the
            # fibre.
            dz_max=length / 2 if dz_max is None else dz_max,
            dz_init=length * 1e-4 if dz_init is None else dz_init,
            auto_init=dz_init is None,
            schedule=schedule,
            integrator=integrator,
            interpolate_planes=interpolate_planes,
            step_planes=0 if step_planes is None else step_planes,
            density=1.0,
            shock=shock,
            prefactor_index=mapping.prefactor_index,
            stats=default_spec("mode_averaged") if stats is None else stats,
            absorber=boundary,
            absorber_scheme=scheme,
            boundary_check=boundary_check,
            polarisation_taper=polarisation_taper,
            backend=backend,
        )
        noise_model, resolution = resolve_noise("gnlse", noise, seed)
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
    def from_capillary(
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
        launch: ModalLaunch | None = None,
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
        polarisation_taper: bool = False,
        node_budget: int | None = None,
        backend: str = "auto",
        polarisation: str | float | None = None,
        materials: MaterialSelection | None = None,
        tdse: TDSEGasResponse | None = None,
        tdse_execution: TDSEExecution | None = None,
    ) -> "Simulation":
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
            Refused beside a constant pressure or a pressure profile, which have
            no end to apply it to.
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
        launch : mapping, sequence of mapping, or None, optional
            Launch the input into chosen modes (SEP 41, S5): a mapping from a mode's
            display label or index to its share, or to ``{"share": s, "phase": phi}``
            with the phase in radians (optics convention), as
            ``{"HE1,1": 0.9, "HE1,2": {"share": 0.1, "phase": 1.57}}``; or a list of
            such mappings, one for each pulse in `pulses`. A label may leave out a
            parity the set carries once (``"HE1,1"`` for ``"HE1,1:even"``). The shares
            are energy fractions, normalised to one for each pulse, and the launch is
            recorded as ``configuration["modal_launch"]``
            (``docs/guide/setup/pulses.md``). ``None``, the default, puts the input in
            the first mode.
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
            and refuses ``"modified"``. ``numpy_ref`` and ``native`` execute it;
            every other mode set or geometry is refused by name. ``None``, the
            default, carries no wall: the modes keep their own Marcatili loss, and a
            multimode taper its taper coupling (below).
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
            Relative step tolerance; the default is
            :data:`~solprop.driver.defaults.CAPILLARY_RTOL`, the capillary default.
            Calibrate it for a production run (``docs/dev/stepper.md`` section 3).
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
        polarisation_taper : bool, optional
            Whether the spectral collar tapers the nonlinear polarisation as well as
            absorbing the field through the linear operator. Default ``False``: the
            taper costs a systematic error of the size of the field's content in the
            collar, and the rate absorber acts either way
            (``docs/guide/numerics/boundaries.md``).
        node_budget : int or None, optional
            Bytes of node-chunk buffers the response pass may hold, an execution
            setting; ``None``, the default, leaves the backend's automatic choice.
        backend : str, optional
            Registry name, or ``"auto"`` (the default) for ``native``, which refuses if
            it isn't installed or this processor is below its baseline. There is no
            fallback, and import or construction errors stay visible.
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
            launch=launch,
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
            rtol=rtol,
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
            polarisation_taper=polarisation_taper,
            node_budget=node_budget,
            backend=backend,
            polarisation=polarisation,
            materials=MaterialSelection() if materials is None else materials,
            tdse=tdse,
            tdse_execution=tdse_execution,
        )

    @classmethod
    @_one_material_context
    def from_waveguide(
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
        launch: ModalLaunch | None = None,
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
        interpolate_planes: bool = False,
        step_planes: int | None = None,
        save_planes: Sequence[float] | np.ndarray | None = None,
        n_save: int | None = None,
        stats: StatsSpec | None = None,
        polarisation_taper: bool = False,
        node_budget: int | None = None,
        backend: str = "auto",
        polarisation: str | float | None = None,
        taper_coupling: bool = True,
        materials: MaterialSelection | None = None,
    ) -> "Simulation":
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
        true-power amplitude by construction. It is qualified on ``numpy_ref`` and the
        native CPU backend, a declared split block's coupled
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
            Refused beside a constant pressure or a pressure profile, which have
            no end to apply it to.
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
        launch : mapping, sequence of mapping, or None, optional
            Launch the input into chosen modes or channels (SEP 41, S5): a mapping
            from a display label or index, resolved as `channel` resolves it, to a
            share, or to ``{"share": s, "phase": phi}`` with the phase in radians
            (optics convention), as ``{"HE1,1": 0.9, "HE1,2": {"share": 0.1,
            "phase": 1.57}}``; or a list of such mappings, one for each pulse in
            `pulses`. A label may leave out a parity the set carries once, and one
            that leaves a parity open between two rows is refused. On a basis whose
            flux metric is the identity the shares are energy fractions, normalised to
            one for each pulse; on a guide that declares a change of basis they are
            relative amplitudes squared, scaled so that the flux-metric energy is the
            pulse's, a coordinate split. The launch is recorded as
            ``configuration["modal_launch"]`` (``docs/guide/setup/pulses.md``).
            Refused beside `channel`, `incident` and `coupling`. ``None``, the
            default, puts the input in the first mode or in `channel`.
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
            Relative step tolerance; the default is
            :data:`~solprop.driver.defaults.WAVEGUIDE_RTOL`, the modal capillary
            default. Calibrate it for a production run (``docs/dev/stepper.md``
            section 3).
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
        interpolate_planes : bool, optional
            Observe the save planes inside accepted steps instead of landing on them
            (SEP 14): through the Adams march's own polynomial, or DOPRI5's
            continuous extension with ``RungeKutta()``. Default ``False``.
        step_planes : int or None, optional
            Also save the end of every `step_planes`-th accepted step, the
            march's own positions, beside the requested planes (SEP 14); ``None``,
            the default, and ``0`` save the requested planes only.
        save_planes : sequence of float or None, optional
            Positions to save at, m. Give this or `n_save`.
        n_save : int or None, optional
            How many planes to save, counting ``z = 0``.
        stats : solprop.stats.StatsSpec or None, optional
            Which statistics to record; ``None`` takes
            :func:`solprop.stats.default_spec` for a multimode run with the profiles,
            or for a mode-averaged one under `mode_average`, plus every statistic
            only a backend can compute.
        polarisation_taper : bool, optional
            Whether the spectral collar tapers the nonlinear polarisation as well as
            absorbing the field through the linear operator. Default ``False``: the
            taper costs a systematic error of the size of the field's content in the
            collar, and the rate absorber acts either way
            (``docs/guide/numerics/boundaries.md``).
        node_budget : int or None, optional
            Bytes of node-chunk buffers the response pass may hold, an execution
            setting; ``None``, the default, leaves the backend's automatic choice.
        backend : str, optional
            Registry name, or ``"auto"`` (the default) for ``native``, which refuses if
            it isn't installed or this processor is below its baseline. There is no
            fallback, and import or construction errors stay visible.
        polarisation : str, float or None, optional
            Refused; this path is scalar.
        taper_coupling : bool, optional
            Carry the taper coupling of a supplied capillary set on a varying radius, as
            :meth:`from_capillary` does; ``False`` is the opt-out to the adiabatic
            route. Refused as ``False`` beside a `provider`, whose moving guide carries
            its own connection.
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
                launch=launch,
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
                interpolate_planes=interpolate_planes,
                step_planes=step_planes,
                save_planes=save_planes,
                n_save=n_save,
                stats=stats,
                polarisation_taper=polarisation_taper,
                node_budget=node_budget,
                backend=backend,
                polarisation=polarisation,
                taper_coupling=taper_coupling,
                materials=MaterialSelection() if materials is None else materials,
            )
        for name, value in (
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
            launch=launch,
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
            rtol=rtol,
            atol=atol,
            dz_max=dz_max,
            dz_init=dz_init,
            schedule=schedule,
            integrator=integrator,
            interpolate_planes=interpolate_planes,
            step_planes=step_planes,
            route="waveguide",
            save_planes=save_planes,
            n_save=n_save,
            stats=stats,
            polarisation_taper=polarisation_taper,
            node_budget=node_budget,
            backend=backend,
            polarisation=polarisation,
            taper_coupling=taper_coupling,
            materials=MaterialSelection() if materials is None else materials,
        )

    @classmethod
    def _from_general(
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
        materials: MaterialSelection | None = None,
        integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None,
        launch: ModalLaunch | None = None,
        interpolate_planes: bool = False,
        step_planes: int | None = None,
        polarisation_taper: bool = False,
        **refused: object,
    ) -> "Simulation":
        """Build a general modal run from a guide that solves its own fields.

        The provider half of :meth:`from_waveguide`, kept apart because it shares no
        physics with the gas-filled one: no fill, no mixture, no static transverse
        quadrature, no declared response list, and a launch that crosses no
        normalisation boundary. Every keyword of the gas-filled path arrives in
        `refused` and is refused by name if it was set.

        - ``length``: guide length, m.
        - ``channel``: which channel the pulse enters; ``None`` takes the first.
        - ``observable_rtol``, ``observable_atol``: accepted only as ``None``, which
          means the solver's own tolerances.
        - ``fwhm``, ``width``, ``shape``, ``energy``, ``peak_power``, ``phases``,
          ``pulses``: the input pulse, as :func:`_shaped_pulse` takes it.
        - ``absorber``: the collar rates; ``None`` builds the grid's default collar, and
          ``False`` switches both soft collars off and keeps the hard band mask.
        - ``absorber_scheme``: how the temporal collar is applied; ``None`` is the
          default scheme.
        - ``stats``: which statistics to record; ``None`` takes :func:`_general_stats`.
        - ``integrator``: the method that steps the run; ``None`` takes ``Multistep()``,
          and ``LowStorageRungeKutta()`` replays `schedule` in three registers.
        - ``step_planes``: also save the end of every `step_planes`-th accepted step;
          ``None`` is ``0``.
        - ``**refused``: every gas-filled keyword; each is refused by name when it is
          set.
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
        refuse_beside(launch, channel=channel, incident=incident, coupling=coupling)
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
        launched: dict[str, object] | None = None
        if beam is not None:
            # Checked before the port is built: a guide whose first region is off the
            # axis has no core for a beam to enter through.
            _core_index(provider.regions, working)
            # The launch is the coupling's, installed through `with_input` below.
            field0 = np.zeros((len(channels), 1, working.n_omega), dtype=np.complex128)
        elif launch is None:
            field0 = _in_channel(built, provider, channel)
        else:
            # A declared change of basis carries a dense flux metric, under which the
            # shares are a coordinate split; in the provider's own coordinates the
            # metric is the identity and they are energy fractions (SEP 41, S5).
            field0, launched = launch_pulse(
                # No beam, so the input is the shaped pulse.
                cast("Pulse", pulse),
                launch,
                working,
                provider.mode_collection.channel_labels,
                metric=metric_at(route.metric_tables, 0.0),
            )
        boundary, scheme = _boundary(absorber, absorber_scheme, working)
        controller, default = _keyword_controller("waveguide", rtol, atol)
        settings = SimulationSettings(
            controller=controller,
            _default_controller=default,
            _route="waveguide",
            save_planes=planes,
            dz_max=length / 2 if dz_max is None else dz_max,
            dz_init=length * 1e-4 if dz_init is None else dz_init,
            auto_init=dz_init is None,
            schedule=schedule,
            integrator=integrator,
            interpolate_planes=interpolate_planes,
            step_planes=0 if step_planes is None else step_planes,
            absorber=boundary,
            absorber_scheme=scheme,
            boundary_check=boundary_check,
            # The route carries its nodes, its materials and its normalisation in
            # its own leaves: the static transverse quadrature, the component
            # selection, the density path and the index override are all refused
            # by the translation layer and are not settings of this run.
            stats=_general_stats() if stats is None else stats,
            polarisation_taper=polarisation_taper,
            backend=backend,
        )
        noise_model, resolution = resolve_noise("waveguide", noise, seed)
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
                **({} if launched is None else {"modal_launch": launched}),
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
    def _from_hollow(  # noqa: PLR0915 - the interface's keyword surface
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
        launch: ModalLaunch | None = None,
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
        route: Literal["capillary", "waveguide"] = "capillary",
        interpolate_planes: bool = False,
        step_planes: int | None = None,
        save_planes: Sequence[float] | np.ndarray | None = None,
        n_save: int | None = None,
        stats: StatsSpec | None = None,
        polarisation_taper: bool = False,
        node_budget: int | None = None,
        backend: str = "auto",
        polarisation: str | float | None = None,
        materials: MaterialSelection | None = None,
        tdse: TDSEGasResponse | None = None,
        tdse_execution: TDSEExecution | None = None,
    ) -> "Simulation":
        """Build a simulation of a pulse in a gas-filled hollow waveguide.

        The shared body of :meth:`from_capillary` and :meth:`from_waveguide`. The mode
        set is the first `modes` HE_1m modes of the capillary unless a set is given; the
        transverse integral is a fixed quadrature rule
        (``docs/guide/waveguides/mode-sets.md``); the
        gas density along z is the pressure path through the equation of state, and the
        same path sets the modes' core index, so a gradient fill disperses and broadens
        consistently.

        See ``docs/guide/setup/running.md`` for the discussion and a worked example.

        - ``radius``: core radius, m, constant or along z.
        - ``length``: capillary length, m.
        - ``gas``: Built-in gas, mole fractions keyed by species name, or ordered
          populations carrying their own density paths. A mapping uses total pressure.
        - ``pressure``: None uses one bar for a gas or fractions. Omit pressure beside a
          Mixture; every explicit pressure is refused because it already carries its
          paths. Fill pressure in **bar**: a number for a constant fill, a pair ``(p_in,
          p_out)`` for the steady-flow gradient of
          :meth:`solprop.profiles.PressureGradient.two_point`, or ``(z, p)`` with the
          positions and the pressures there for
          :meth:`solprop.profiles.PressureGradient.from_stages`.
        - ``pressure_floor``: lowest pressure the fill is allowed to reach, bar;
          ``None`` takes :data:`solprop.profiles.DEFAULT_PRESSURE_FLOOR_FRACTION` of the
          highest stage pressure. With ``vacuum_end="floor"``, pass ``0.0`` for a true
          vacuum end, which warns, because ``dp/dz`` diverges there. Refused beside a
          constant pressure or a pressure profile, which have no end to apply it to.
        - ``vacuum_end``: how a pressure gradient's outer end below its choked pressure
          is treated. ``None`` (the default) means ``"choked"`` for a circular hollow
          guide: the fill follows the bore (Pennetta et al. 2019, eq. 6) and an end
          exhausting into vacuum takes the choked pressure of the flow
          (:func:`solprop.gasflow.steady_fill`), with a warning if the flow is not
          laminar. ``"floor"`` concerns the ends only: it keeps the declared end
          pressures, raised only to `pressure_floor`, for a comparison with a model that
          imposes them, while a taper's fill still follows the bore (on a uniform bore,
          bitwise the uniform law). Refused beside a constant pressure, a pressure
          profile or a Mixture, which have no end to choke, and on the general route.
          Every mode declaring ``core_radius`` -- every ``MarcatiliMode`` subclass, a
          ``TabulatedMode`` built from arrays included -- is taken as a circular bore of
          that radius; a rectangular guide keeps the uniform law.
        - ``wavelength``: reference wavelength, m, and the pulse's centre wavelength.
        - ``wavelength_limits``: shortest and longest wavelength of the frequency
          window, m. Required unless `grid` is given.
        - ``trange``: requested width of the time window, s. Required unless `grid` is
          given.
        - ``grid``: a grid of one's own, instead of the three keywords above; the only
          way to reach ``dt``, ``ffac`` or the transform size.
        - ``fwhm``: pulse duration at half maximum, s.
        - ``width``: natural width of a sech pulse, s; give this or `fwhm`.
        - ``energy``: pulse energy, J. Give this or `peak_power`.
        - ``peak_power``: peak power, W, realised after the spectral phases.
        - ``phases``: Taylor spectral phases about `wavelength`, optics convention: CEP
          in rad, group delay in s, GDD in s^2, and so on.
        - ``pulses``: several pulses launched together, for two-colour and multi-colour
          inputs: each a :mod:`solprop.pulses` object with its own wavelength, duration,
          energy or peak power, phase and delay; the input is the sum of their fields
          (:class:`~solprop.pulses.PulseSum`). Replaces the single-pulse keywords
          `fwhm`, `width`, `shape`, `energy`, `peak_power` and `phases`, which must then
          be left unset; `wavelength` still sets the grid's reference wavelength.
        - ``launch``: the modes each pulse enters, with their shares and phases, as
          :meth:`from_capillary` documents it; ``None`` puts the input in the first
          mode.
        - ``mode_average``: propagate the one mode **mode-averaged**: a uniform disc of
          the mode's effective area instead of a transverse rule
          (``docs/guide/waveguides/mode-averaging.md``). The plasma then sees the
          area-mean field, below the on-axis one (``HE_11``'s axis intensity is 1.768
          times ``P/A_eff``), while ``peak_intensity`` stays on the axis. ``False``, the
          default, resolves the mode on the radial rule. Refused beside more than one
          mode and beside `n_r`, `n_theta` or a `radial_rule`.
        - ``lattice``: tube geometry for an integer mode count, selecting Vincetti
          modes. Refused with supplied modes, which carry their own geometry, and with
          ``model="reduced"``. None selects ordinary capillary modes.
        - ``wall``: declare the scalar **coherent** wall block: the outgoing Robin
          boundary condition condensed onto the retained basis, which couples every
          ``HE_1m`` channel to every other with a definite phase instead of giving each
          its own Marcatili attenuation. It needs the contiguous set ``HE_11`` to
          ``HE_1N`` -- or, with ``impedance="te_tm"``, one azimuthal order -- on a
          constant or a varying guide, and a scalar run; it carries the default
          ``"input"`` noise and refuses ``"modified"``. ``numpy_ref`` and ``native``
          execute it; every other mode set or geometry is refused by name. ``None``,
          the default, carries no wall: the modes keep their own Marcatili loss, and a
          multimode taper its taper coupling (below).
        - ``taper_coupling``: on a multimode capillary whose radius varies along z,
          carry the **taper coupling** -- the connection ``<psi_m|d_z psi_n>`` that
          moves power between the ``HE_1m`` channels as the bore changes -- lossless or
          not. ``True``, the default; ``False`` is the explicit opt-out to the
          **adiabatic** route, on which each channel follows its local mode and
          exchanges no power. It is refused beside `wall`, which always carries the
          connection, and has no effect on one channel, a constant bore (a pressure
          gradient included, which does not change the modes' shape) or a set that is
          not the contiguous ``HE_11`` to ``HE_1N`` block
          (``docs/guide/waveguides/capillaries.md``, "Tapers").
        - ``model``: which Marcatili model the modes use; beside a supplied set, refused
          unless it is the default or the modes' own.
        - ``loss``: propagation loss of the modes; beside a supplied set, refused unless
          it is the default or the modes' own.
        - ``cladding``: cladding index; ``None`` is the package's cited silica. Beside a
          supplied set, refused unless it is the default or the modes' own.
        - ``temperature``: gas temperature, K, or a path of it along z in m. Sets the
          density the pressure path maps to, so it changes both the dispersion and the
          nonlinearity. A path also weights a steady-flow gradient by the gas's ``eta(T)
          T`` (docs/guide/waveguides/profiles.md), makes a shared mixture's populations
          independent parameters, and holds the rotational Raman populations at one
          reference temperature with a
          :class:`~solprop.exceptions.RamanTemperatureWarning`
          (:func:`solprop.nonlinear.mixtures.raman_temperature`). A constant path is the
          number, bit for bit. ``None`` (the default) is 293.15 K. A Mixture owns its
          temperature: beside one, a temperature must be omitted or be the mixture's own
          (the same number or path object), and any other is refused by name.
        - ``thg``: include third-harmonic generation in the Kerr term; ``None`` takes
          ``not envelope``.
        - ``plasma``: include photoionisation. ``None`` takes ``not envelope``. ``True``
          selects optmat's recommended rate model of each built-in gas (SEP 3, D1;
          :class:`~solprop.nonlinear.optmat_rates.OptmatRate`, or the model the
          selection's ``ionisation`` names), for a single gas and for every component of
          a mixture; a custom species needs its own rate. A rate object is used as
          given; a mapping selects a rate or switch per component label, and omitted
          labels are off.
        - ``preionisation``: initial ionised fraction in [0, 1], globally or per label:
          a number, or a path along z in m giving the fraction ionised before the pulse
          arrives at each position. None and omitted mapping labels retain the
          component's existing declaration.
        - ``plasma_integration``: how the field-grid plasma forms its population and
          free-current integrals (SEP 18): ``'band'`` filters each integrand with taps
          designed for the grid, exact to a certified 1e-5 across the active band;
          ``'trapezoid'`` is the plain cumulative trapezoid, short by up to 11 % per
          integral at the top of the default active band, which the frozen witness
          comparisons use. ``None`` takes
          :class:`~solprop.nonlinear.responses.PlasmaResponse`'s default. Refused with
          envelope plasma, whose currents are converted exactly, and when the plasma is
          off.
        - ``raman``: ``None`` (the default) turns the Raman response on for every
          species whose optmat response is curated -- N2, O2, H2, D2, CH4 and SF6 --
          leaves an atom without one, and warns
          (:class:`solprop.exceptions.SolpropWarning`) for a molecule whose response is
          provisional (N2O, unless the selection's ``raman`` is ``"provisional"``)
          rather than leaving it out silently; with ``kerr=False`` the default is off.
          ``False`` disables it; ``True`` selects the available models of a mixture and
          is refused by name for a single gas without data; explicit labels or
          :class:`~solprop.physdata.raman_types.MolecularOptions` select components and
          terms, and a named unavailable model fails explicitly.
        - ``n_r``: radial nodes of the fixed rule. ``None`` takes
          :func:`default_radial_order` of the mode count.
        - ``ionisation_field_range``: field amplitudes in V/m the rate table must cover.
          ``None`` derives them from the input's peak field over the axis and the
          transverse nodes; see :func:`solprop.driver.simulation.default_field_range`.
        - ``absorber``: the collar rates; ``None`` builds the default collar of the
          grid, and ``False`` switches both soft collars off and keeps the hard band
          mask (:func:`solprop.boundaries.no_absorber`; SEP 21).
        - ``boundary_check``: how the run watches its time window's edges (SEP 21): the
          default warns, then stops, when light above the shot-noise floor reaches them;
          ``None`` watches nothing, for a window kept narrow on purpose.
        - ``rtol``: relative step tolerance; ``None`` takes the route's default
          (:func:`~solprop.driver.defaults.default_rtol`). Calibrate it for a production
          run (``docs/dev/stepper.md`` section 3).
        - ``dz_max``: cap on the step, m; ``None`` takes half the capillary.
        - ``dz_init``: first step, m; ``None`` takes ``length * 1e-4``.
        - ``integrator``: the method that steps the run
          (``docs/guide/numerics/choosing-a-stepper.md``). ``None``, the default, takes
          ``Multistep()``: the low-call exponential Adams method, one response
          evaluation per step. ``RungeKutta()`` is the interaction-picture DOPRI5 march,
          for a cheap response or a state too large for the Adams history.
          ``LowStorageRungeKutta()`` replays `schedule` in three registers (SEP 24).
        - ``route``: the interface building the run, whose row of
          :data:`~solprop.driver.defaults.ROUTE_DEFAULTS` gives what the caller left:
          ``"capillary"`` (its TDSE variant beside a TDSE response) or ``"waveguide"``
          (its slab variant for a slab mode set).
        - ``step_planes``: also save the end of every `step_planes`-th accepted step
          (SEP 14); ``None`` is ``0``, or ``1`` with `tdse` (SEP 20 section 5.8).
        - ``save_planes``: positions to save at, m. Give this or `n_save`.
        - ``stats``: which statistics to record; ``None`` takes
          :func:`solprop.stats.default_spec` for a multimode run with the profiles, or
          for a mode-averaged one under `mode_average`, plus every statistic only a
          backend can compute.
        - ``node_budget``: bytes of node-chunk buffers the response pass may hold, an
          execution setting; ``None``, the default, leaves the backend's automatic
          choice.
        - ``backend``: registry name, or ``"auto"`` (the default) for ``native``, which
          refuses if it isn't installed or this processor is below its baseline. There
          is no fallback, and import or construction errors stay visible.
        - ``materials``: take the run's built-in materials from optmat under this
          selection: a gas or fraction mapping of built-in gases, ``"air"`` and
          ``Gas.Air`` resolve through :func:`solprop.physdata.species` and
          :meth:`solprop.physdata.mixtures.Mixture.air` with it, and the default
          cladding (``cladding=None`` with a numbered mode set) is optmat's recommended
          silica. A supplied mixture, mode set, cladding or susceptibility is taken as
          it is. The run records its material manifest. ``None`` (the default) is
          ``MaterialSelection()``.
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
            raise DriverError(msg, parameter=("raman", "kerr"), rule="raman-needs-kerr")
        if polarisation is not None:
            msg = (
                f"polarisation={polarisation!r}: this path is scalar and propagates "
                "one transverse component per mode row. Declare the polarisation "
                "in the modes instead -- a rectangular or slab guide's own "
                "polarisation, each partner a row of its own -- or, for a vector "
                "capillary run, a polarisation-resolved mode set, which is not "
                "validated end to end yet"
            )
            raise DriverError(
                msg, parameter="polarisation", rule="scalar-capillary-polarisation"
            )
        _check_wall_declaration(
            wall,
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
        # The fill's own models warn about the conditions they are evaluated at,
        # which the gas keyword chose.
        with _warnings_about("gas", _MATERIAL_WARNINGS):
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
                raise DriverError(
                    "from_waveguide requires supplied gas-filled modes",
                    parameter="modes",
                )
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
            with _warnings_about("gas", _MATERIAL_WARNINGS):
                responses = _capillary_responses(
                    gas,
                    mixture.components[0].species,
                    kerr=kerr,
                    plasma=plasma,
                    preionisation=0.0 if preionisation is None else preionisation,
                    integration=plasma_integration,
                    wavelength=wavelength,
                    envelope=envelope,
                    # The single gas's electronic response is optmat's, at the
                    # carrier or dispersive, as the selection's electronic policy
                    # says (SEP 6).
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
            # optmat's Raman response is the raman keyword's choice.
            with _about("raman"), _warnings_about("raman", _MATERIAL_WARNINGS):
                responses += (
                    _optmat_raman(gas, mixture, materials, thg, wavelength, length)
                    if raman
                    else []
                )
        else:
            with _warnings_about("gas", _MATERIAL_WARNINGS):
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
        with _about(("preionisation", "plasma")):
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
        field0, launched = _hollow_input(pulse, launch, working, mode_set.modes)
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
        row = (
            "waveguide slab"
            if any(isinstance(mode, SlabMode) for mode in mode_set.modes)
            else "capillary tdse"
            if atomic is not None
            else route
        )
        controller, default = _keyword_controller(row, rtol, atol)
        settings = SimulationSettings(
            controller=controller,
            _default_controller=default,
            _route=row,
            save_planes=_save_planes(length, save_planes, n_save),
            # Relative defaults, so that the first step and the cap scale with the
            # capillary.
            dz_max=length / 2 if dz_max is None else dz_max,
            dz_init=length * 1e-4 if dz_init is None else dz_init,
            auto_init=dz_init is None,
            schedule=schedule,
            integrator=integrator,
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
            polarisation_taper=polarisation_taper,
            backend=backend,
            tdse_execution=None if atomic is None else atomic.execution,
        )
        noise_model, resolution = resolve_noise(row, noise, seed)
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
                **({} if launched is None else {"modal_launch": launched}),
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


@dataclasses.dataclass(frozen=True)
class _LoadedRun:
    """What a runner's march needs, its input held by its start alone.

    Attributes
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The derived problem, without its input (:func:`_without_input`).
    start : solprop.backends.session.Start
        The input's start. The run's session takes its field, so that nothing holds
        the input through the march (issue #447).
    handler : solprop.output.schema.Output
        Where the planes go.
    metadata : solprop.output.schema.RunMetadata
        The run's record.
    retention : solprop.output.schema.FieldRetention
        Which planes keep their field.
    boundary_monitor : solprop.monitor.BoundaryMonitor or None
        The run's monitor of the window's edges.
    material_log : solprop.physdata.selection.MaterialLog
        The derived run's material log, which the march goes on with.
    """

    problem: PropagationProblem
    start: Start
    handler: Output
    metadata: RunMetadata
    retention: FieldRetention
    boundary_monitor: BoundaryMonitor | None
    material_log: MaterialLog


class SimulationRunner(Displayed):
    """One translated problem and one backend, run for many inputs (SEP 22 section 5.2).

    Built by :meth:`Simulation.runner`: the problem is translated (or the cached
    translation reused) and one backend is constructed, once. Each
    :meth:`run` then derives the input with :meth:`Simulation.with_input`, loads it
    into the same backend and runs the save-plane loop, returning that input's own
    handler. A large problem so pays its backend's allocation and transform planning
    once rather than for every input.

    A run on the reused backend repeats a fresh backend's run, to the rounding that
    separates any two runs: loading an input restores everything the march reads
    (the reference's :meth:`solprop.backends.host.driver.HostDriver.set_initial`, the
    native backend's own load). Each run's record is
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

    Raises
    ------
    DriverError
        If the simulation has a TDSE response, whose atomic binding has its own
        lifecycle and is never reused.

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

    def __init__(self, simulation: Simulation) -> None:
        """Translate, admit and construct the backend, once."""
        problem = simulation.translate()
        if tdse_descriptor(problem.route) is not None:
            msg = (
                "a TDSE response binds an atomic runtime with its own lifecycle, which "
                "a runner cannot reuse across inputs; run each input with "
                "simulation.with_input(...).run()"
            )
            raise DriverError(msg)
        settings = simulation.settings
        request = simulation._run_request()
        self._simulation = simulation
        self._problem = problem
        self._request = request
        self._lock = threading.Lock()
        self._closed = False
        self.runs = 0
        with _optmat.material_context(simulation._material_log):
            _warn_sampled_peak(settings.quadrature, request.observation.stats)
            self._backend = select(
                settings.backend,
                problem,
                request,
                grid=simulation.grid,
                modes=simulation.modes,
                profiles=simulation.profiles,
                options=settings.execution_options(),
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

        - Returns: the title and ``(key, value)`` rows.
        """
        return "SimulationRunner", [
            ("simulation", repr(self._simulation)),
            ("backend", self._simulation.settings.backend),
            ("runs", self.runs),
            ("state", "closed" if self._closed else "open"),
        ]

    def __enter__(self) -> "SimulationRunner":
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
    def closed(self) -> bool:  # numpydoc ignore=RT01
        """Whether the runner, and its backend, are closed."""
        return self._closed

    @property
    def backend(self) -> Backend:  # numpydoc ignore=RT01
        """Backend every run is stepped on."""
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
        initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray,
        *,
        output: Output | Path | str | None = None,
        configuration: Mapping[str, object] | None = None,
        noise: NoiseRealisation | Literal["keep"] | None = "keep",
        progress: ProgressReporter | ProgressChoice | None = None,
        progress_period: float = DEFAULT_PROGRESS_PERIOD,
        input_identity: str | None = None,
    ) -> Output:
        """Run one input on the runner's backend and return its handler.

        Parameters
        ----------
        initial : Pulse, SpatialPulse or numpy.ndarray
            The input, as :meth:`Simulation.with_input` takes it. An array the record
            keeps by digest (:func:`~solprop.output.schema.recorded_input`) is not
            copied: it is read while this call loads it, and the runner holds no
            reference to it, nor to any copy, once it is loaded, so a caller that
            keeps none itself has it freed before the first step.
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
        input_identity : str or None, optional
            What an array input is, recorded in its place and not checked, so that its
            bytes are not digested (:meth:`Simulation.with_input`, whose warning
            applies: a wrong identity passes a resume and a scan's cache).

        Returns
        -------
        solprop.output.schema.Output
            This input's handler, closed; a memory handler stays readable.

        Raises
        ------
        DriverError
            If the runner is closed or already running an input; if the derived
            problem is not the runner's but for its input (a different launch
            position, or a shared leaf replaced); if the noise model differs from
            the runner's, which the backend could not carry; or if `input_identity`
            is empty or names anything but an array input.
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
            loaded = self._load(
                initial,
                output=output,
                configuration=configuration,
                noise=noise,
                input_identity=input_identity,
            )
            # The start holds the input, and every frame that saw it has returned but
            # this one; the session takes it from the start, so the march holds no
            # volume for it (issue #447).
            del initial
            return self._march(
                loaded, progress=progress, progress_period=progress_period
            )
        except BaseException:
            # A backend that closed itself cannot run another input.
            if self._backend.closed:
                self._close_held()
            raise
        finally:
            self._lock.release()

    def _takes(self, simulation: Simulation) -> bool:
        """Return whether a simulation is this runner's but for its input."""
        try:
            check_derived_problem(
                self._problem,
                simulation.translate(),
                self._request,
                simulation._run_request(),
            )
        except DriverError as error:
            logger.debug("a new backend for this run: %s", error)
            return False
        return True

    def _run_point(
        self,
        simulation: Simulation,
        output: Output,
        *,
        progress: ProgressReporter | ProgressChoice | None,
        stop: StopFlag | None,
    ) -> None:
        """Run a simulation this runner :meth:`_takes`, as its own ``run`` would.

        For a scan's worker, which has built the point's simulation and holds its
        record already. Unlike :meth:`run` it honours `stop`, and the handler's record
        is the simulation's own.
        """
        with self._lock:
            simulation._run(output, progress=progress, stop=stop, kept=self._backend)
            self.runs += 1

    def _load(
        self,
        initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray,
        *,
        output: Output | Path | str | None,
        configuration: Mapping[str, object] | None,
        noise: NoiseRealisation | Literal["keep"] | None,
        input_identity: str | None,
    ) -> _LoadedRun:
        """Derive one input's run, open its handler and build the input's start.

        The derived simulation and its problem are this frame's alone: they go when it
        returns, and the start alone holds the converted launch of a modal route or the
        view of the input, until the run's session takes it.
        """
        derived = self._simulation.with_input(
            _borrowed_input(initial, stated=input_identity is not None),
            noise=noise,
            configuration=configuration,
            input_identity=input_identity,
        )
        problem = derived.translate()
        request = derived._run_request()
        check_derived_problem(self._problem, problem, self._request, request)
        settings = self._simulation.settings
        with _optmat.material_context(derived._material_log):
            metadata = derived.metadata()
            handler, _ = _open_output(output, metadata, resume=False)
            retention = field_retention(request, settings.fields, settings.field_planes)
            monitor = derived._boundary_monitor()
            start = Start.entrance(
                problem, noise=None if isinstance(noise, str) else _realisation(problem)
            )
        return _LoadedRun(
            _without_input(problem),
            start,
            handler,
            metadata,
            retention,
            monitor,
            derived._material_log,
        )

    def _march(
        self,
        loaded: _LoadedRun,
        *,
        progress: ProgressReporter | ProgressChoice | None,
        progress_period: float,
    ) -> Output:
        """Run the loaded input's save-plane loop and finish its handler.

        - ``progress_period``: seconds between periodic lines, s.
        """
        settings = self._simulation.settings
        problem, handler = loaded.problem, loaded.handler
        with _optmat.material_context(loaded.material_log):
            run_problem(
                problem,
                self._backend,
                handler,
                start=loaded.start,
                metadata=loaded.metadata,
                stats_spec=self._request.observation.stats,
                progress=progress,
                progress_period=progress_period,
                resume_history=settings.resume_history,
                fields=loaded.retention.fields,
                field_planes=loaded.retention.field_planes,
                boundary_monitor=loaded.boundary_monitor,
            )
            reports = _peak_field_reports(problem, handler, self._simulation.responses)
        if reports and isinstance(handler, PlaneStore):
            handler.annotate(
                PEAK_FIELDS_ANNOTATION,
                [cast("JsonValue", report.to_dict()) for report in reports],
            )
        self.runs += 1
        return handler


def _borrowed_input(
    initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray, *, stated: bool
) -> Pulse | SpatialPulse | SpatialPulseSum | np.ndarray:
    """Return a runner's input as its private derived simulation may take it, uncopied.

    A derived simulation copies an array so that its record and its run outlive the
    caller's changes. A runner's lives only until the load, on the caller's thread and
    inside the call, so an array needs no copy unless the record outlives it: one
    recorded inline stays a copy, in the handler's record.

    - Returns: a read-only view of a complex128 array the record does not hold, borrowed
      (:func:`solprop._ownership.borrow`) so that nothing copies it; `initial` itself
      otherwise. The caller's own flags are not touched.
    """
    if (
        isinstance(initial, np.ndarray)
        and initial.dtype == np.complex128
        and (stated or not _recorded_inline(initial))
    ):
        return borrow(np.asarray(initial).view())
    return initial


def _without_input(problem: PropagationProblem) -> PropagationProblem:
    """Return a loaded problem without its input, for the march that follows the load.

    The save-plane loop reads the launch position and never the launch, so the state
    is replaced by a broadcast of one zero, which holds no volume (:class:`~solprop.
    translate.problem.Initial`). The problem is good for the march alone.
    """
    # A broadcast of a held zero is itself held, so Initial takes it uncopied.
    placeholder = np.broadcast_to(
        hold(np.zeros(1, dtype=np.complex128)), problem.initial.field0.shape
    )
    return dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=placeholder)
    )


def _realisation(problem: PropagationProblem) -> NoiseRealisation | None:
    """Return a derived problem's noise as the backend loads it, or ``None``.

    - Returns: its realisation, or ``None`` without noise.
    """
    initial = problem.initial
    if initial.noise0 is None or initial.noise_model == "none":
        return None
    return NoiseRealisation(
        spectrum=initial.noise0, seed=None, model=initial.noise_model
    )


_SHARED_LEAVES = (
    "grids",
    "linear",
    "transform",
    "responses",
)
"""The leaves :meth:`Simulation.with_input` shares with the translated problem.

The route is shared but for a spatial route's transverse design, whose check
:meth:`Simulation.with_input` reruns on the new launch: its variant and its other
leaves are compared beside these (:func:`check_derived_problem`). The statistics are
the request's, shared too."""


def check_derived_problem(
    constructed: PropagationProblem,
    derived: PropagationProblem,
    constructed_request: RunRequest,
    derived_request: RunRequest,
) -> None:
    """Refuse a derived problem that is not the constructed one but for its input.

    What a :class:`SimulationRunner` checks before loading an input into the backend
    it built for `constructed` (SEP 22 section 5.2): the leaves
    :meth:`Simulation.with_input` shares are the same objects, the statistics too, the
    stepper and the flags, which it rebuilds, agree field by field (the save planes
    elementwise), and the launch position and the noise model are the backend's.

    Parameters
    ----------
    constructed : solprop.translate.problem.PropagationProblem
        The problem the backend was built for.
    derived : solprop.translate.problem.PropagationProblem
        The problem of one input.
    constructed_request : solprop.translate.run_request.RunRequest
        The request the backend was built for.
    derived_request : solprop.translate.run_request.RunRequest
        The request of the input's run.

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
    >>> derived = simulation.with_input(problem.initial.field0)
    >>> check_derived_problem(
    ...     problem, derived.translate(), simulation._run_request(),
    ...     derived._run_request(),
    ... ) is None
    True
    """
    for name in _SHARED_LEAVES:
        if getattr(derived, name) is not getattr(constructed, name):
            msg = (
                f"the input's problem does not share the runner's {name}: a runner "
                "steps one translated problem and changes only its input"
            )
            raise DriverError(msg)
    if derived_request.observation.stats is not constructed_request.observation.stats:
        msg = (
            "the input's problem does not share the runner's stats_spec: a runner "
            "steps one translated problem and changes only its input"
        )
        raise DriverError(msg)
    if (
        type(derived.route) is not type(constructed.route)
        or tdse_of(derived.route) is not tdse_of(constructed.route)
        or flux_of(derived.route) is not flux_of(constructed.route)
    ):
        msg = (
            "the input's problem does not share the runner's route: a runner steps "
            "one translated problem and changes only its input"
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
    for record, built, name in (
        (derived_request.stepper, constructed_request.stepper, "stepper"),
        (derived.flags, constructed.flags, "flags"),
    ):
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
    """Require a material declaration the update can identify by component."""
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
    """
    if envelope and active and thg:
        msg = "envelope plasma cannot be combined with THG"
        raise DriverError(
            msg, parameter=("envelope", "plasma", "thg"), rule="envelope-plasma-no-thg"
        )
    if integration is None:
        return
    if envelope:
        msg = (
            f"plasma_integration={integration!r} selects the field-grid plasma "
            "column's running-integral rule; envelope plasma converts its currents "
            "with exact inverse-frequency factors and takes no such rule"
        )
        raise DriverError(
            msg,
            parameter=("plasma_integration", "envelope"),
            rule="plasma-integration-field-grid",
        )
    if not active:
        msg = (
            f"plasma_integration={integration!r} is given but the plasma is off; "
            "turn it on (plasma=True) or drop plasma_integration"
        )
        raise DriverError(
            msg,
            parameter=("plasma_integration", "plasma"),
            rule="plasma-integration-needs-plasma",
        )


def _refuse_idle_preionisation(mixture: Mixture, responses: Sequence[object]) -> None:
    """Refuse a declared initial ionised fraction that no plasma channel carries.

    A component's preionisation reaches the run only through its plasma or
    envelope-plasma response. Without one the declaration would be recorded in the
    run's metadata and ignored by the propagation, so it is refused by name.
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
    """Resolve an existing plasma declaration to its already-validated population."""
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

    - ``expansion``: previously sampled coefficients and axis names.
    - Returns: the old coefficients when the function is identical, else None with a
      log.
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
    problem: PropagationProblem,
) -> TranslationReuse:
    """Collect eligible immutable translation data after a material-path update."""
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
        transverse_of(problem.route),
    )


def _undamped_model_key(response: RamanResponse) -> object:
    """Describe only the inputs to the undamped Raman samples."""
    models = []
    for term in response.tensor_terms:
        model = term.model
        record = serialise(model)
        if isinstance(model, (DampedOscillator, RotationalLines)):
            assert isinstance(record, dict)
            record.pop("dephasing", None)
        models.append((term.contraction, record))
    return models


def _check_wall_declaration(
    wall: CoherentWall | None,
    *,
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

    - ``wall``: what was declared. ``None`` leaves every other argument alone.
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
        raise DriverError(
            msg, parameter=("wall", "taper_coupling"), rule="wall-keeps-taper-coupling"
        )
    if loss is not True:
        msg = (
            f"a coherent wall block is the loss model, and this run declares "
            f"loss={loss!r}: the condensed Robin boundary condition replaces the "
            "Marcatili attenuation of every channel, so a LossSpec beside it would "
            "be discarded in silence. Pass loss=True with wall=, or leave wall=None "
            "and keep the uncoupled loss you asked for"
        )
        raise DriverError(msg, parameter=("wall", "loss"), rule="wall-is-the-loss")
    if model != "full":
        msg = (
            f"a coherent wall block needs one capillary model, the full one, and "
            f"this run declares model={model!r}: the condensation inverts the full "
            "model's transverse eigenvalue, and the reduced model's leading-order "
            "carrier beside it would be overridden rather than propagated"
        )
        raise DriverError(msg, parameter=("wall", "model"), rule="wall-full-model")
    if lattice is not None:
        msg = (
            "a coherent wall block assumes one uniform dielectric wall at a single "
            "radius, and this run declares a tube lattice: an antiresonant guide's "
            "loss is its tubes' resonances, not a Robin condition on a bare "
            "capillary. Leave wall=None and keep the Vincetti loss, or drop the "
            "lattice"
        )
        raise DriverError(msg, parameter=("wall", "lattice"), rule="wall-no-lattice")


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

    - Returns: the first mode's core radius in m, or ``None`` for a guide without one.
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

    - ``radius``: the guide's core radius in m, if it has one.
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

    - ``boundary``: the collar the run will use, for the length that sets its temporal
      strength. A spectral-only collar has no temporal part to check.
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

    - ``simulation``: declared optical input, grid and absolute launch position in m.
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
    problem: PropagationProblem,
    request: RunRequest,
    state: ResumeState,
    execution: TDSEExecution | None,
) -> int:
    """Validate restart position and budgets before preparing atomic resources.

    - ``request``: its run's request, with the absolute save positions in m.
    - ``execution``: current atomic budget, if this is a TDSE run.
    - Returns: index of the next scheduled save strictly beyond the accepted position.
    """
    planes = request.observation.save_planes
    next_plane = int(np.searchsorted(planes, state.plane.z, side="right"))
    if state.checkpoint is not None and state.checkpoint.next_save_index != next_plane:
        raise DriverError(
            "checkpoint next_save_index disagrees with its absolute position"
        )
    if not problem.initial.z <= state.plane.z <= planes[-1]:
        raise DriverError(
            "resume position lies outside the declared propagation interval"
        )
    if state.plane.field is None:
        if _work_left(request, state, next_plane):
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
    if tdse_descriptor(problem.route) is not None and (
        outcome is None or not outcome.logical_calls or not outcome.atomic_calls
    ):
        raise DriverError("a TDSE resume requires persisted cumulative call ledgers")
    method = request.integration.method
    if outcome is not None and outcome.logical_calls and isinstance(method, Multistep):
        calls = CallLedger.from_record(outcome.logical_calls)
        if method.max_calls is not None and calls.total > method.max_calls:
            raise DriverError("logical call budget is below previously charged work")
    if outcome is not None and outcome.atomic_calls:
        atomic = AtomicCallLedger.from_record(outcome.atomic_calls)
        if tdse_descriptor(problem.route) is None or execution is None:
            raise DriverError(
                "an atomic call ledger requires an explicit TDSE execution"
            )
        if (
            execution.max_atomic_calls is not None
            and atomic.attempted > execution.max_atomic_calls
        ):
            raise DriverError("atomic call budget is below previously attempted work")
    return next_plane


def _check_tdse_archive(
    archive: object, problem: PropagationProblem, handler: Output
) -> None:
    """Refuse an archive request this run cannot honour, before any work.

    - ``archive``: the request, checked here to be a ``TDSEArchive`` or None.
    """
    if archive is None:
        return
    if not isinstance(archive, TDSEArchive):
        raise DriverError("tdse_archive must be a TDSEArchive or None")
    if tdse_descriptor(problem.route) is None:
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

    - Returns: True when they resolve to one location.
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

    - Returns: the UUID, or None when no execution of the run archived.
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


def _open_tdse_archive(  # noqa: PLR0917 - called positionally
    archive: TDSEArchive | None,
    problem: PropagationProblem,
    handler: Output,
    state: ResumeState | None,
    metadata: RunMetadata,
    execution: TDSEExecution,
) -> TDSEArchiveWriter | None:
    """Open the run's archive session, continuing its own run's archive on a resume.

    - ``state``: what a resumed file holds, or None for a fresh run.
    - Returns: the open session, or None without a request.
    """
    descriptor = tdse_descriptor(problem.route)
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
        # compares with.
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

    - ``failure``: the error the run is raising, if any.
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

    - ``failure``: the error the run is raising, if any.
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

    - Returns: resolved grid and settings with a matching fine temporal collar.
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
    """Refuse unsupported execution choices before any atomic preparation."""
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

    - ``settings``: original absorber definition and its physical lengths in m.
    """
    boundary = settings.absorber
    if boundary is None or boundary.temporal_rate_fine.shape == (refined.n_to,):
        return settings
    original = collar(parent, boundary_length=boundary.reference_length)
    length = boundary.in_stage_length

    def temporal(grid: TimeGrid, axis: np.ndarray) -> np.ndarray:
        """Return the collar's temporal rate on a time axis of a grid, 1/m.

        - ``axis``: its coarse or fine time axis, s.
        - Returns: the bounded profile at the in-stage length, or zeros for a
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


def _replaced_launch(
    original: PropagationProblem,
    field0: Complex[np.ndarray, "n_modes n_pol n_omega"],
    launch_z: float,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return a replaced input as the derived problem's launch, without a further copy.

    On a free-space or spatial route the launch is a view of the derived simulation's
    own copy; on a modal route it is the power conversion of that copy, a second array
    beside it.

    - ``field0``: the replacement input, in true sqrt(W) on a modal route.
    - ``launch_z``: where it is launched, m.
    """
    if original.transform.geometry is not None:
        return _state(field0, original.shape[-1])
    # The conversion is a new array, held as it is rather than copied again.
    converted = power_amplitude(
        field0, power_weight_at(original, launch_z), inverse=True
    )
    return hold(np.asarray(converted, dtype=np.complex128))


def _check_input_identity(
    identity: str | None,
    initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray | CouplingLaunch,
) -> None:
    """Refuse a stated identity that is empty, or of anything but an array input."""
    if identity is None:
        return
    if not identity:
        raise DriverError("input_identity must say what the input is: it is empty")
    if not isinstance(initial, np.ndarray):
        msg = (
            "input_identity states an array input's identity; "
            f"a {type(initial).__name__} is recorded as itself"
        )
        raise DriverError(msg)


def _given_input(
    given: np.ndarray, held: Complex[np.ndarray, "n_modes n_pol n_omega"]
) -> np.ndarray:
    """Return an array input as the caller gave it, for the record: a read-only view.

    The record keeps the caller's dtype and values, so that a file written from such
    an input before the simulation copied it still resumes; a view of a read-only owner
    cannot be made writeable, so the record and the run cannot be changed through it.

    - Returns: a read-only view of `held` when `given` is complex128, otherwise of a
      held copy of `given` in its own dtype; :class:`~solprop.translate.problem.Initial`
      takes either as held.
    """
    # The whole dtype, byte order included: the record serialises its exact string.
    source = held if given.dtype == held.dtype else _held_block(given, given.dtype)
    return hold(source.view())


def _check_planar_input(shape: tuple[int, ...], n_columns: int, n_omega: int) -> None:
    """Refuse a planar input whose shape is not ``(n_p*n_y, 1, n_omega)``.

    - ``shape``: the input's shape.
    """
    if shape != (n_columns, 1, n_omega):
        msg = (
            "a planar input has shape (n_p*n_y, 1, n_omega) = "
            f"{(n_columns, 1, n_omega)}, got {shape}"
        )
        raise DriverError(msg)


def _check_spatial_input(shape: tuple[int, ...], n_columns: int, n_omega: int) -> None:
    """Refuse a spatial input whose shape is not ``(n_columns, n_pol, n_omega)``.

    - ``shape``: the input's shape.
    """
    if len(shape) != 3 or shape[0] != n_columns or shape[2] != n_omega:
        msg = "spatial input must have shape (n_columns, n_pol, n_omega)"
        raise DriverError(msg)


def _built(
    initial: Pulse | np.ndarray, grid: TimeGrid
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return the initial spectrum, building a pulse against the grid when given one.

    - Returns: the spectrum, complex128, shape ``(n_modes, n_pol, n_omega)``, read-only
      and the simulation's own: a copy of an array (of its repeated block for a
      broadcast), or a solprop pulse's new build.
    """
    if isinstance(initial, np.ndarray):
        # Copied now, so that changing the caller's array later changes nothing, as
        # the problem holds it: a broadcast copies its block alone, and an array a
        # simulation already holds (its own, passed back) is taken as it is.
        return _held_state(initial)
    # A runtime-checkable protocol is satisfied by the presence of `build`, not by its
    # being callable, so the guard below is still the one that names the mistake.
    build = getattr(initial, "build", None)
    if not callable(build):
        msg = (
            f"the input must be a pulse with a build(grid) method or a built spectrum, "
            f"got {type(initial).__name__}"
        )
        raise DriverError(msg)
    return _built_state(initial, build(grid))


def _built_state(
    pulse: object, built: object
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return a pulse's built spectrum as the problem holds it, copied only if foreign.

    A build of solprop's own pulse classes is a new array solprop made, so it is held
    as it is; any other pulse's build may be an array the pulse keeps, and is copied
    (:class:`~solprop.translate.problem.Initial`).

    - Returns: the spectrum, complex128, read-only and held.
    """
    spectrum = np.asarray(built)
    build = getattr(type(pulse), "build", None)
    own = getattr(build, "__module__", None) == _PULSE_MODULE
    if own and spectrum.dtype == np.complex128:
        # A build may give a new array its leading axes by indexing: the memory
        # beneath is what is held.
        return hold_view(spectrum)
    return _held_state(spectrum)


_PULSE_MODULE: Final = SpatialPulse.__module__
"""The module of solprop's pulse classes, whose ``build`` always makes a new array."""


def _interpolated(state: ResumeState) -> bool:
    """Return whether the plane a file resumes from was interpolated inside a step.

    A landed plane is the end of an accepted step of the run; an interpolated one lies
    inside a step. The entrance is the initial field.
    """
    return state.interpolated


def _open_output(
    output: Output | Path | str | None,
    metadata: RunMetadata,
    *,
    resume: _ResumeChoice,
) -> tuple[Output, ResumeState | None]:
    """Choose the handler a run writes to, and read what a resumed file holds.

    The resume gate (ADR-066): the file's configuration hash must equal `metadata`'s.
    Whether work is left is decided by :func:`_continues`.
    """
    if resume and metadata.configuration.get("_benchmark_plasma_source"):
        raise DriverError("gated plasma benchmark output cannot be resumed")
    if isinstance(output, Path | str):
        path = Path(output)
        if resume and check_cache(path, metadata):
            return HDF5Output(path, append=True), resume_state(path)
        return HDF5Output(path), None
    if resume:
        msg = (
            "only a path can be resumed: the file has to be opened in append mode, "
            "which is the driver's to decide; pass output=<path> with resume=True"
        )
        raise DriverError(msg)
    return (MemoryOutput() if output is None else output), None


def _check_resume_choice(resume: object) -> None:
    """Refuse a ``resume`` value that is not one of its three.

    A mistyped word would otherwise read as ``True``.
    """
    if resume is True or resume is False or resume == "interpolated":
        return
    msg = f"resume must be False, True or 'interpolated', got {resume!r}"
    raise ValueError(msg)


def _work_left(request: RunRequest, state: ResumeState, next_plane: int) -> bool:
    """Return whether a file has anything left to compute, by position.

    Decided by position, not status: a file killed after its last plane has no work
    left, though its status reads ``running``; a terminal checkpoint still owes its
    plane.
    """
    return (
        state.checkpoint is not None
        or next_plane < request.observation.save_planes.size
    )


def _refuse_interpolated(path: Path, state: ResumeState, resume: _ResumeChoice) -> None:
    """Refuse to continue from a plane interpolated inside a step, unless accepted.

    An interpolated plane is not a state the march passed through, so a run restarted
    there is not the run it continues (SEP 14 section 5); ``resume="interpolated"``
    accepts that.
    """
    if resume == "interpolated" or not _interpolated(state):
        return
    msg = (
        f"{path} ends on a plane interpolated inside a step at z = "
        f"{state.plane.z:.6e} m, not a state the march passed through; "
        "pass resume='interpolated' to continue afresh from it, or rerun"
    )
    raise DriverError(msg)


def _continues(
    path: Path,
    request: RunRequest,
    state: ResumeState,
    next_plane: int,
    resume: _ResumeChoice,
) -> bool:
    """Say whether a resume of a file computes anything, once its hash has passed.

    - Returns: ``True`` when work is left, and ``False`` for a finished file.
    """
    if not _work_left(request, state, next_plane):
        return False
    _refuse_interpolated(path, state, resume)
    return True


def _gnlse_grid(
    grid: TimeGrid | None,
    length: float,
    wavelength: float,
    wavelength_limits: tuple[float, float] | None,
    trange: float | None,
) -> TimeGrid:
    """Return the envelope grid a GNLSE run propagates on.

    - ``grid``: a grid given by the caller, or ``None`` to build one.
    - ``length``: fibre length, m.
    - ``wavelength``: reference wavelength, m.
    - ``wavelength_limits``: window limits, m.
    - ``trange``: time window, s.
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
            raise DriverError(msg, parameter="grid", rule="gnlse-envelope-grid")
        return grid
    if wavelength_limits is None or trange is None:
        msg = (
            "give wavelength_limits and trange so a grid can be built, or a grid of "
            "your own"
        )
        raise DriverError(
            msg,
            parameter=_unset(wavelength_limits=wavelength_limits, trange=trange),
            rule="grid-or-limits",
        )
    with _about(renamed=_GRID_KEYWORDS):
        return EnvelopeGrid.from_limits(length, wavelength, wavelength_limits, trange)


def _gnlse_fibre(
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

    - ``gamma``: nonlinear coefficient in 1/(W m), absent beside `n2`.
    - ``betas``: Taylor coefficients about the reference frequency, s^n/m.
    - ``beta``: a pair of frequency and beta arrays, a callable, or None.
    - ``wavelength``: positive reference wavelength, m.
    - ``loss``: power loss in dB/m; None means omitted. Refused beside `fibre`.
    - ``fibre``: whole dispersion and loss, including a positive propagation area for
      `n2`.
    - ``n2``: nonlinear index in m^2/W, or ``"recommended"`` under a `selection`;
      requiring `fibre`.
    - ``raman_fraction``: share of the nonlinearity held back from the Kerr term,
      between zero and one; None resolves as in
      :func:`solprop.nonlinear.gnlse.problem_from`.
    """
    if sum(value is not None for value in (betas, beta, fibre)) != 1:
        raise DriverError(
            "give exactly one of betas, beta and fibre",
            parameter=("betas", "beta", "fibre"),
            rule="betas-beta-or-fibre",
        )
    if (gamma is None) == (n2 is None):
        raise DriverError(
            "give exactly one of gamma and n2",
            parameter=("gamma", "n2"),
            rule="gamma-or-n2",
        )
    if n2 is not None and fibre is None:
        raise DriverError(
            "n2 requires fibre with an effective area",
            parameter=("n2", "fibre"),
            rule="n2-needs-fibre-area",
        )
    if fibre is not None and loss is not None:
        raise DriverError(
            "the fibre carries its loss; scale it on the mode",
            parameter=("fibre", "loss_db_per_m"),
            rule="fibre-carries-loss",
        )
    omega_ref = 2.0 * np.pi * C / wavelength
    mode = (
        None
        if fibre is None
        else _gnlse_mode(fibre, wavelength, needs_area=n2 is not None)
    )
    area = 1.0 if mode is None else mode.aeff()
    # The GNLSE mapping of ADR-026 (docs/dev/decisions.md) is unchanged. An externally
    # specified fibre replaces the placeholder mode, so its dispersion never enters the
    # unit prefactor. Its own names are the keywords' but for the area, the fibre's.
    with _about(renamed={"aeff": "fibre"}):
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
    with _about("beta"):
        table = SpecifiedMode.from_table(
            nodes,
            values,
            loss=attenuation_from_db(0.0 if loss is None else loss),
            aeff=mapping.aeff,
        )
    return dataclasses.replace(mapping, mode=table)


def _gnlse_mode(
    fibre: Mode | IndexTable, wavelength: float, *, needs_area: bool
) -> Mode:
    """Resolve a supplied table or mode with its propagation area at the pump.

    - ``wavelength``: positive reference wavelength, m: where a table's area column and
      a provider channel's surrogate area are evaluated.
    """
    if isinstance(fibre, IndexTable):
        if needs_area and fibre.area is None:
            raise DriverError(
                "n2 requires a fibre table with an effective-area column",
                parameter=("n2", "fibre"),
                rule="n2-needs-fibre-area",
            )
        area = (
            1.0
            if fibre.area is None
            else float(
                fibre.effective_area(np.array([2.0 * np.pi * C / wavelength]))[0]
            )
        )
        with _about("fibre"):
            return SpecifiedMode.from_index_table(fibre, aeff=area)
    if not math.isfinite(fibre.aeff(0.0)) or fibre.aeff(0.0) <= 0:
        raise DriverError(
            "fibre effective area must be positive and finite", parameter="fibre"
        )
    # aeff=None hands over the fibre's own declared area path when it has one and
    # samples the source's own area along z when its whole geometry moves; the
    # frozen callables forward the position, so the dispersion and the area move
    # together and a fibre moving along z needs no refusal (ADR-026,
    # docs/dev/decisions.md). The wavelength is where a provider channel's area is
    # solved, as a table's is read at it above; every other mode's area is not a
    # function of frequency and ignores it.
    with _about("fibre"):
        return SpecifiedMode.from_mode(fibre, wavelength=wavelength)


def _dispersion_table(
    beta: object, grid: TimeGrid
) -> tuple[Float[np.ndarray, " n_nodes"], Float[np.ndarray, " n_beta_nodes"]]:
    """Return the ``(omega, beta)`` table a caller's dispersion describes.

    A callable is sampled on the grid's **active band**, sorted and made unique, because
    that is exactly where the linear operator evaluates a mode
    (:func:`solprop.linear.dispersion_operator`) and a spline refuses to extrapolate
    beyond its data.

    - Returns: the frequencies in rad/s and the propagation constant in 1/m.
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
    raise DriverError(msg, parameter="beta")


def _shaped_pulse(
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

    - ``wavelength``: centre wavelength, m, and the reference of the spectral phase.
    - ``fwhm``: intensity full width at half maximum, s.
    - ``width``: natural width of a ``sech**2``, s.
    - ``energy``: pulse energy, J.
    - ``peak_power``: peak power, W.
    - ``phases``: Taylor coefficients of the optics-convention spectral phase.
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
            raise DriverError(
                msg, parameter=("pulses", *given), rule="pulses-or-pulse-keywords"
            )
        with _about("pulses"):
            return PulseSum(tuple(pulses))
    if shape == "sech":
        # The pulse names its own arguments, which are the keywords but for phase.
        with _about(renamed=_PULSE_KEYWORDS):
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
        raise DriverError(msg, parameter="shape")
    if width is not None or fwhm is None:
        msg = (
            "a Gaussian input is specified by its intensity fwhm; give fwhm= rather "
            "than width=, which is the natural width of a sech**2 pulse"
        )
        raise DriverError(
            msg,
            parameter="fwhm" if width is None else ("width", "shape"),
            rule="gauss-fwhm",
        )
    with _about(renamed=_PULSE_KEYWORDS):
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


def _with_boundary(
    settings: SimulationSettings,
    grid: TimeGrid,
    absorber: Absorber | Literal[False] | None,
    scheme: AbsorberScheme | None,
) -> SimulationSettings:
    """Apply a spatial interface's absorber keywords to its settings (SEP 21 section 5).

    - ``absorber``: the collar given; ``False`` zeroes the temporal and spectral rates
      of the settings' absorber and keeps its lengths (without one it builds the hard
      mask alone); ``None`` leaves the settings' own. The transverse boundary is
      ``spatial_absorber``'s (SEP 40).
    - ``scheme``: the temporal scheme given; ``None`` leaves the settings' own.
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
                user_rate=None,
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
        if settings.absorber_scheme_named and settings.absorber_scheme != scheme:
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
    """Apply a spatial constructor's ``boundary_check`` keyword to its settings."""
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
    simulation: Simulation,
    original: PropagationProblem,
    derived: PropagationProblem,
    planes: Float[np.ndarray, " n_z"],
) -> SpatialAbsorberDesign | None:
    """Return a derived run's transverse design, its check rerun on the new launch.

    SEP 40 section 5.2: a derived run keeps its parent's design, which its shared
    operator and transform were built with, but not its verdict, which belongs to a
    launch. Only a scalar multiple of the checked launch reuses it, because the check
    is linear and normalised by the launch's energy. A new launch whose rms
    wavenumber lies above the one the band was derived from warns as well.

    - ``planes``: the derived run's save planes, m.
    """
    design = transverse_of(original.route)
    if design is None or not design.has_collar:
        return design
    space = simulation.spatial_grid
    medium = simulation.medium
    assert isinstance(space, RadialGrid | FreeGrid | PlanarGrid)
    assert medium is not None
    old = np.asarray(original.initial.field0)
    new = np.asarray(derived.initial.field0)
    reference = float(np.vdot(old, old).real)
    if reference > 0.0:
        scale = np.vdot(old, new) / reference
        residual = float(np.linalg.norm(new - scale * old))
        if residual <= 1e-12 * float(np.linalg.norm(new)):
            return design
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

    - ``launch``: the launch, canonical layout.
    - ``length``: the run's length, m.
    - Returns: the widest axis's band edge, rad/m; ``None`` when it cannot be resolved.
    """
    try:
        return max(
            resolve_transverse(specification, space, dispersion, launch, length).kappa_p
        )
    except TranslationError:
        return None


def _refuse_pending_check(problem: PropagationProblem) -> None:
    """Refuse to start a run whose collar's design check never saw a launch.

    A coupled child is translated before its launch exists; its check runs when
    :meth:`Simulation.with_input` gives it one (SEP 40 section 5.2).
    """
    design = transverse_of(problem.route)
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

    - ``spatial_absorber``: the keyword's value; ``None`` leaves the settings' own.
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


def _shot_noise(
    noise: str | None, grid: TimeGrid, seed: int | None, *, n_modes: int = 1
) -> NoiseRealisation | None:
    """Draw the shot-noise realisation a run carries.

    The model is a plain string rather than :data:`NoiseModel` for the reason
    :func:`solprop.fourier._check_kind` gives: this is the run-time guard for callers
    that ignore the annotation, and the test session's ``beartype`` hook would otherwise
    reject the bad value before the check below is reached.

    - ``noise``: which model to use, or ``None`` for a noiseless run.
    - ``seed``: entropy of the generator; ``None`` draws from the operating system, and
      the entropy it drew is recorded on the realisation either way.
    """
    if noise is None:
        return None
    if noise not in ("input", "modified"):
        msg = f"noise must be 'default', None, 'input' or 'modified', got {noise!r}"
        raise DriverError(msg, parameter="noise")
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

    - ``noise``: which model to use, or ``None`` for a noiseless run.
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

    - ``seed``: entropy of the generator; ``None`` draws from the operating system.
    """
    return spatial_noise(flux, np.random.default_rng(seed), model=noise, n_pol=n_pol)


def _spatial_draw(
    noise: NoiseModel, simulation: Simulation, medium: SpatialMedium
) -> Callable[[int | None], NoiseRealisation | None]:
    """Return the draw of a spatial run's shot noise, measured at its entrance plane."""
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
    problem: PropagationProblem, noise: NoiseRealisation | None
) -> PropagationProblem:
    """Return `problem` carrying `noise`, every other leaf shared by identity.

    A realisation reaches a translation only through the initial record and the
    ``has_noise`` flag; what else consults it is a refusal keyed on the model, and a
    replacement that changes the model is checked by its caller.

    - ``noise``: the realisation in true sqrt(W) on a modal route, or ``None`` for no
      noise.
    - Returns: the problem with the realisation in its internal, vacuum-normalised
      amplitude.
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

    - Returns: `simulation`, able to redraw its noise when `seed` was ``None``.
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


def _capillary_grid(
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

    - ``length``: capillary length, m.
    - ``wavelength``: reference wavelength, m.
    - ``wavelength_limits``: window limits, m.
    - ``trange``: requested time window, s.
    """
    if grid is not None:
        if wavelength_limits is not None or trange is not None:
            msg = (
                "give a grid or the wavelength_limits and trange to build one, not "
                "both: the keywords would be ignored"
            )
            given = {"wavelength_limits": wavelength_limits, "trange": trange}
            raise DriverError(
                msg,
                parameter=(
                    "grid",
                    *(name for name, value in given.items() if value is not None),
                ),
                rule="grid-or-limits",
            )
        return grid
    if wavelength_limits is None or trange is None:
        msg = (
            "a capillary run needs wavelength_limits and trange to build its grid, "
            "or a grid of your own"
        )
        raise DriverError(
            msg,
            parameter=_unset(wavelength_limits=wavelength_limits, trange=trange),
            rule="grid-or-limits",
        )
    with _about(renamed=_GRID_KEYWORDS):
        if envelope:
            return EnvelopeGrid.from_limits(
                length,
                wavelength,
                wavelength_limits,
                trange,
                thg=thg,
            )
        return FieldGrid.from_limits(length, wavelength, wavelength_limits, trange)


def _capillary_modes(
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

    - ``radius``: core radius in m, used when constructing a numbered mode set.
    - ``length``: the run's length in m, over which a supplied set's radius is compared.
    """
    if isinstance(modes, int) and modes < 1:
        raise DriverError(
            f"a capillary mode set needs at least one mode, got {modes}",
            parameter="modes",
        )
    if lattice is not None:
        if not isinstance(modes, int):
            raise DriverError(
                "lattice is refused: the given modes carry their own geometry",
                parameter=("lattice", "modes"),
                rule="lattice-builds-modes",
            )
        if model == "reduced":
            raise DriverError(
                'lattice is incompatible with model="reduced"',
                parameter=("lattice", "model"),
                rule="lattice-full-model",
            )
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
        raise DriverError(
            "a capillary mode set needs at least one mode", parameter="modes"
        )
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

    - ``modes``: the supplied set, non-empty.
    - ``radius``: the declared core radius, m.
    - ``length``: the run's length, m, over which the radius paths are compared.
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
            raise DriverError(msg, parameter=("radius", f"modes[{index}]"))
        if not _same_radius(declared, own, length):
            msg = (
                f"from_capillary's radius disagrees with supplied mode {index} "
                f"({mode.channel_label.display_label}): a supplied mode set is the "
                "run's geometry and keeps its own core radius, so the declared one "
                "would be ignored. Declare the modes' own radius (the same number or "
                "profile) or build the modes on the radius wanted"
            )
            raise DriverError(msg, parameter=("radius", f"modes[{index}]"))
        _refuse_ignored_setting(index, mode, "model", model, default="full")
        _refuse_ignored_setting(index, mode, "loss", loss, default=True)
        _refuse_ignored_setting(index, mode, "cladding", cladding, default=None)


def _refuse_ignored_setting(
    index: int, mode: Mode, name: str, declared: object, *, default: object
) -> None:
    """Refuse a non-default capillary setting that a supplied mode does not carry.

    - ``name``: the keyword, which is also the mode's attribute: ``"model"``, ``"loss"``
      or ``"cladding"``.
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
        raise DriverError(msg, parameter=(name, f"modes[{index}]"))


def _same_radius(declared: ZProfile, own: ZProfile, length: float) -> bool:
    """Return whether two radius paths agree over a run of `length`.

    - ``declared``: the declared core radius, m.
    - ``own``: a supplied mode's core radius, m.
    - ``length``: the run's length, m.
    - Returns: whether they agree to ``_RADIUS_RTOL`` at 65 equally spaced positions,
      both paths' breakpoints and the midpoints between all of them.
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
    """Bind supplied modes and retain the existing orthonormality gate."""
    if not modes:
        raise DriverError(
            "a waveguide mode set needs at least one mode", parameter="modes"
        )
    rebound: list[Mode] = []
    for index, mode in enumerate(modes):
        with _about(f"modes[{index}]"):
            rebound.append(_rebind_mode(mode, mixture))
    given = ModeCollection(tuple(rebound))
    if len(given) > 1:
        overlaps = given.orthonormality()
        worst = float(np.max(np.abs(overlaps - np.eye(len(given)))))
        if worst > ORTHONORMALITY_TOLERANCE:
            raise DriverError(
                f"the given modes are not an orthonormal set: the largest deviation "
                f"of the overlap matrix from the identity is {worst:.3e}, above "
                f"{ORTHONORMALITY_TOLERANCE:.0e}. One synthesis matrix and its dual "
                "represent an orthonormal basis and nothing else",
                parameter="modes",
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

    - ``n_r``: radial nodes; ``None`` takes the module's
      :func:`~solprop.driver.simulation.default_radial_order`.
    - ``n_theta``: azimuthal nodes; ``None`` takes the rule's own default where one is
      needed.
    """
    if modes.modes[0].dimlimits(0.0)[0] == "slab":
        if n_theta is not None:
            msg = (
                f"n_theta={n_theta} was given for a slab set: the planar guide is "
                "uniform in y, so its rule has one y node of unit width and n_r "
                "alone is the x node count"
            )
            raise DriverError(msg, parameter="n_theta")
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
        raise DriverError(msg, parameter="n_theta")
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

    - ``modes``: the mode set, which must hold exactly one mode.
    - ``n_r``: radial nodes, which must not be given.
    - ``n_theta``: azimuthal nodes, which must not be given.
    - ``radial_rule``: the radial rule family, which must be the default.
    """
    if len(modes) != 1:
        msg = (
            f"mode_average=True needs exactly one mode, and the set has {len(modes)}: "
            "energy moving between modes is what a mode-averaged run cannot carry "
            "(docs/guide/waveguides/mode-averaging.md)"
        )
        raise DriverError(
            msg, parameter=("mode_average", "modes"), rule="mode-average-one-mode"
        )
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
        raise DriverError(
            msg, parameter=("mode_average", *given), rule="mode-average-no-quadrature"
        )


def _capillary_responses(
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

    - ``material``: the fill's species as resolved under the run's selection, which the
      plasma response carries: it is never rebuilt from the name with the default
      selection, so the material manifest lists only the models that acted (Q45).
    - ``preionisation``: fraction already ionised before the pulse, a number or a path
      along z in m.
    - ``integration``: the field plasma column's running-integral rule; ``None`` takes
      the response's default.
    - ``wavelength``: reference wavelength, m, the rate's carrier unless the selection
      names one.
    - ``electronic``: the gas's electronic response under the selection's policy
      (:func:`solprop.nonlinear.mixtures.electronic_response`); ``None`` without one.
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
        raise DriverError(
            msg, parameter=("kerr", "plasma"), rule="capillary-needs-nonlinearity"
        )
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
                SolpropWarning(
                    f"{component.label} ({material.name}) is a molecule whose Raman "
                    "response optmat has only provisionally, so it is off; pass "
                    "raman=False to say so, or MaterialSelection(raman='provisional') "
                    "to admit it",
                    parameter="raman",
                ),
                stacklevel=3,
            )
    return on


def _optmat_raman(  # noqa: PLR0917 - called positionally
    gas: Gas | str,
    mixture: Mixture,
    selection: MaterialSelection,
    thg: bool,
    wavelength: float,
    length: float,
) -> list[object]:
    """Return a single-gas fill's Raman response from optmat (SEP 3).

    - ``wavelength``: the grid reference wavelength in m, the carrier unless the
      selection names one.
    - ``length``: the run's length in m.
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

    - ``wavelength``: reference wavelength, m, the rate's carrier unless the selection
      names one.
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


def _tdse_entry(
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

    - ``tdse``: the declared response, or None for a rate-based run.
    - ``trange``: requested time window in s, the default atomic-window duration.
    - ``kerr``: the Kerr switch, which must stay true beside a TDSE response.
    - ``mode_average``: the mode-averaged switch, which must stay false beside a TDSE
      response: its translation needs the resolved radial quadrature.
    - ``unset``: Electronic-response keywords that must be left unset beside it.
    - ``save_planes``: requested save positions in m.
    - Returns: the resolution, or None without `tdse`.
    """
    if tdse is None:
        if execution is not None:
            raise DriverError(
                "tdse_execution requires a TDSEGasResponse",
                parameter=("tdse_execution", "tdse"),
                rule="tdse-execution-needs-tdse",
            )
        return None
    if not isinstance(tdse, TDSEGasResponse):
        raise DriverError("tdse must be a TDSEGasResponse", parameter="tdse")
    given = [name for name, value in unset.items() if value is not None]
    if not kerr:
        given.append("kerr=False")
    if mode_average:
        given.append("mode_average=True")
    if given:
        raise DriverError(
            f"the TDSE response owns the electronic response; leave "
            f"{', '.join(given)} unset beside tdse=",
            parameter=("tdse", *(name.removesuffix("=False") for name in given)),
            rule="tdse-owns-electronic",
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
                    "TDSEAtomicWindow(duration=...)",
                    parameter=("tdse", "trange"),
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


def _entry_overrides[P, R](  # noqa: PLR0917 - called positionally
    atomic: _TDSEEntry | None,
    plasma: P,
    raman: R,
    backend: str,
    n_save: int | None,
    step_planes: int | None,
) -> tuple[_TDSEEntry | None, P, R, str, int | None, int]:
    """Return a TDSE entry with the capillary settings it overrides, or the originals.

    - ``step_planes``: requested natural-step interval; ``None`` is 0 without TDSE.
    - Returns: the entry, the plasma and Raman switches, the backend, the plane count
      and the step interval.
    """
    if atomic is None:
        steps = 0 if step_planes is None else step_planes
        return None, plasma, raman, backend, n_save, steps
    # False is a member of both switches' declared types.
    off = cast("P", False), cast("R", False)
    return atomic, *off, atomic.backend, atomic.n_save, atomic.step_planes


def _tdse_stats(spec: StatsSpec) -> StatsSpec:
    """Remove the statistics a TDSE response cannot supply from a default spec."""
    return dataclasses.replace(
        spec,
        names=tuple(
            name for name in spec.names if name not in TDSE_UNAVAILABLE_STATISTICS
        ),
    )


def _warn_tdse_cost(quadrature: TransverseQuadrature | None) -> None:
    """Warn once when a TDSE run integrates over a many-node radial rule."""
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

    - ``averaged``: whether the run is mode-averaged: the mode-averaged defaults then
      replace the multimode ones, and the quadrature estimate, which has no rule to
      read, is left out.
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
    "node_budget": None,
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

    - Returns: True when the keyword was, in effect, not given.
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
    """Refuse, by name, every gas-filled keyword given beside a provider."""
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

    - ``channel``: row index, display label, or ``None`` for the first channel.
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


def _general_pulse(
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

    - ``incident``: the incident beam, which carries its own temporal pulse; ``None``
      builds the pulse from the keywords.
    - ``shape``, ``wavelength``, ``fwhm``, ``width``, ``energy``, ``peak_power``,
      ``phases``, ``pulses``: the pulse keywords, as :func:`_shaped_pulse` takes them.
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

    - Returns: ``Re sqrt(eps)`` on the active positive band and zero elsewhere, shape
      ``(n_omega,)``.
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
    """Return the radial plane an incident beam is sampled on before coupling."""
    radius = _BEAM_MARGIN * pulse.waist * _BEAM_EXTENT ** (1.0 / (2.0 * pulse.order))
    return RadialGrid(radius, _BEAM_NODES)


def _incident_launch(
    simulation: "Simulation",
    pulse: SpatialPulse,
    model: CouplingModel,
) -> "Simulation":
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


def _hollow_input(
    pulse: Pulse,
    launch: ModalLaunch | None,
    grid: TimeGrid,
    modes: Sequence[Mode],
) -> tuple[Complex[np.ndarray, "n_modes n_pol n_omega"], dict[str, object] | None]:
    """Return a hollow guide's initial state, and its launch record if it has one.

    - ``launch``: the ``launch`` keyword; ``None`` puts the input in the first mode.
    - Returns: ``configuration["modal_launch"]``, or ``None`` without a launch.
    """
    if launch is None:
        return _in_lowest_mode(pulse.build(grid), len(modes)), None
    # A hollow guide's modes are orthonormal in true power, so the shares are energy
    # fractions (SEP 41, S5).
    return launch_pulse(
        pulse, launch, grid, tuple(mode.channel_label for mode in modes)
    )


def _in_lowest_mode(
    built: Complex[np.ndarray, "n_modes_in n_pol n_omega"], n_modes: int
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return a one-mode input placed in the lowest mode of an `n_modes` set.

    A single pulse enters the fundamental and the higher modes start empty, filling
    through the nonlinearity.
    """
    if built.shape[0] > n_modes:
        msg = (
            f"the input carries {built.shape[0]} modes and the set has {n_modes}; a "
            "pulse cannot be placed in a mode that is not propagated"
        )
        raise DriverError(msg, parameter=("pulses", "modes"))
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

    - Returns: ``("x",)``, or ``None`` to infer the components from the modes.
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

    - ``modes``: the mode set, whose profiles are evaluated on the axis and at the
      nodes.
    - ``rule``: the run's transverse quadrature; ``None`` for a mode-averaged run, whose
      table the axis alone bounds: the area-mean field it samples lies below it.
    - Returns: the peak field amplitude in V/m, and the circular share ``|S3| / S0`` of
      the field's transverse components where it peaks, zero for a linear field.
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
        """Return the peak field in V/m and its circular share."""
        return _capillary_input_peak(self.field0, self.modes, self.grid, self.rule)


def _ionisation_inputs(
    responses: Sequence[object],
    field_range: tuple[float, float] | None,
    peak: _InputPeak | None,
) -> tuple[list[object], tuple[float, float] | None, float | None]:
    """Return a capillary run's ionisation table range and its bound rates.

    The input's peak field is found once, when the table's range is the default one
    (:func:`default_field_range`) or an optmat rate waits for its reference field.

    - ``field_range``: the caller's table range in V/m, or ``None`` for the default.
    - ``peak``: where the input's peak is found from; ``None`` for a run without plasma.
    - Returns: the responses (optmat rates bound to the peak), the table range, and the
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
    problem: PropagationProblem,
    output: Output,
    responses: Sequence[object],
) -> tuple[PeakFieldReport, ...]:
    """Report the largest field each optmat-rate population saw, after a run.

    The populations' peak rates at the save planes are read back from the run's
    output -- ``species_peak_ionisation_rate``, or ``peak_ionisation_rate`` for one
    population -- and inverted through each rate
    (:func:`solprop.nonlinear.optmat_rates.peak_field_report`).

    - Returns: one per population with an optmat rate; none when the run carries no such
      rate or recorded no peak rate, which is logged.
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
        with HDF5Output(output.path, readonly=True) as reader:
            return _peak_field_reports(problem, reader, responses)
    if "species_peak_ionisation_rate" in output.stat_names:
        rows = np.asarray(output["stats/species_peak_ionisation_rate"])
        peaks = np.max(rows, axis=0)
    elif "peak_ionisation_rate" in output.stat_names and len(labels) == 1:
        rows = np.asarray(output["stats/peak_ionisation_rate"])
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
    simulation: Simulation, initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray
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
    simulation: Simulation, initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray
) -> None:
    """Bind a free-space or planar run's optmat rates to its input, and check it.

    The spatial routes -- radial, Cartesian, line and planar -- bind an optmat rate's
    constant energy per event to the input's peak field, as the capillary route does
    (SEP 3, Q14); binding also checks the input against each rate. A run whose rates
    are all bound is only checked.
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
    simulation: Simulation, initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray
) -> tuple[float, float]:
    """Return a free-space or planar run's peak input field and its circular share.

    - Returns: the peak physical field amplitude in V/m, and the circular share
      ``|S3|/S0`` of the pulse's Jones vector; a spectrum given as an array declares
      none and is taken as linear.
    """
    space = cast("RadialGrid | FreeGrid | PlanarGrid", simulation.spatial_grid)
    # Ionisation is driven by the physical field, never by a signed axial flux.
    peak = peak_field(simulation.grid, space, simulation._field0)
    jones = (
        np.asarray(initial.jones, dtype=np.complex128)
        if isinstance(initial, SpatialPulse | SpatialPulseSum)
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

    - ``amplitude``: envelope amplitude in V/m.
    - Returns: the rate in s^-1 its statistic records there.
    """
    return float(cycle_moments(amplitude, 0.0, cycle)[0])


def _unbound_rates(responses: Sequence[object]) -> bool:
    """Return whether a plasma response carries an optmat rate without a reference.

    - Returns: True when a rate's constant energy per event still waits for the input's
      peak.
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

    - ``peak``: the input's peak field in V/m and its circular share there.
    - ``field_range``: the field amplitudes in V/m the run tabulates its rates over.
    - Warns once per run when the peak exceeds a rate's maximum field, or the input is
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

    - ``rule``: the run's fixed transverse rule, whose embedded coarse rule the
      quadrature alarm needs; ``None`` for a mode-averaged run.
    - Returns: statistics with unsupported geometry paths omitted.
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
    """Refuse a dispersive electronic policy on a route whose response is constant."""
    if materials is None or materials.electronic not in DISPERSIVE_POLICIES:
        return
    msg = (
        f"the electronic policy {materials.electronic!r} evaluates optmat's "
        f"dispersive hyperpolarizability model, and {route} is a constant coefficient "
        "this route keeps: select 'carrier-kerr' (or 'recommended') here"
    )
    raise DriverError(msg, parameter="materials.electronic", rule="gnlse-carrier-kerr")


def _check_electronic_policy(
    materials: MaterialSelection | None, responses: Sequence[object]
) -> None:
    """Refuse a constant Kerr response under a dispersive electronic policy.

    A caller's own responses are taken as they are, but a constant
    :class:`~solprop.nonlinear.responses.KerrResponse` beside a selection that names a
    dispersive policy would record one law and propagate another.
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

    The electronic coefficient, the Raman response and the rate are all evaluated at
    the carrier.

    - ``reference``: the grid's reference wavelength in m, the run's carrier.
    - ``wavelength``: the driver's `wavelength` in m, at which the responses are built.
    """
    if materials is None:
        return
    # Without the driver's wavelength the reference is a grid given whole.
    carrier = (
        "materials.carrier_wavelength",
        "grid" if wavelength is None else "wavelength",
    )
    for name, value, parameter in (
        ("the selection's carrier_wavelength", materials.carrier_wavelength, carrier),
        ("wavelength", wavelength, ("wavelength", "grid")),
    ):
        if value is not None and not carriers_agree(float(value), reference):
            msg = (
                f"{name} is {value:g} m and the grid's reference wavelength "
                f"{reference:g} m: under a material selection one carrier sets the "
                "Kerr coefficient of every optmat species (SEP 3, section 4.3), and "
                "it is the grid's. Build the grid at that wavelength, or leave the "
                "selection's carrier_wavelength unset"
            )
            raise DriverError(msg, parameter=parameter)


def _default_cladding(cladding: GlassIndex | complex | None) -> GlassIndex | complex:
    """Return the cladding a numbered capillary set takes.

    - ``cladding``: the cladding given; ``None`` is the package default.
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

    - Returns: the guide to propagate on: the one given when nothing resolves, otherwise
      the same provider, channels and sampling with the regions' delayed responses
      resolved.
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
        """Return a region with its default delayed response resolved."""
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

    - Returns: the grid given when nothing resolves; otherwise the same grid rebuilt on
      the resolved slab modes.
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
    """Return the fill of a planar grid's modes as a constant bulk medium."""
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
    """
    if settings.absorber is not None:
        return settings
    return dataclasses.replace(settings, absorber=default_collar(grid))


def _transverse_specification(
    setting: SpatialAbsorber | Literal[False] | None,
    spatial_grid: object,
) -> SpatialAbsorber | None:
    """Return the transverse boundary a run's translation resolves (SEP 40).

    - ``spatial_grid``: the run's transverse grid, ``None`` for a modal run.
    - Returns: the specification; ``None`` for no ramp and no collar.
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
