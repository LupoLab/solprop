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

import dataclasses
import functools
import json
import logging
import math
import threading
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
from solprop.backends.registry import resolve_name, select
from solprop.boundaries import (
    Absorber,
    TransverseCollar,
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
    CouplingError,
    DriverError,
    GasFlowWarning,
    GrazingIncidenceWarning,
    MaterialRangeWarning,
    MemoryBudgetError,
    MemoryBudgetWarning,
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
from solprop.linear import Frame, frame_quantities
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
from solprop.output.hdf5 import HDF5Output, ResumeState, check_cache, open_file
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
from solprop.spatial import SpatialFlux, SpatialMedium, peak_field
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
from solprop.translate.masks import DEFAULT_ABSORBER_SCHEME, AbsorberScheme
from solprop.translate.modal import DEFAULT_NODE_BUDGET
from solprop.translate.plasma_integration import (
    integration_record,
)
from solprop.translate.plasma_source import _BENCHMARK_GATED, plasma_source_record
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

__all__ = [
    "BACKEND_STATISTICS",
    "DEFAULT_ATOL",
    "DEFAULT_N_SAVE",
    "DEFAULT_RTOL",
    "LOOSE_ADAMS_RTOL",
    "LOW_STORAGE_BACKENDS",
    "ORTHONORMALITY_TOLERANCE",
    "PLASMA_STATISTICS",
    "WAVEGUIDE_ATOL",
    "WAVEGUIDE_RTOL",
    "NoiseModel",
    "Simulation",
    "SimulationRunner",
    "SimulationSettings",
    "check_derived_problem",
    "default_field_range",
    "default_radial_order",
    "metadata_for",
]

logger = logging.getLogger(__name__)


LOOSE_ADAMS_RTOL = 5e-2
"""The ``rtol`` from which an adaptive Adams run is warned on.

An error of five per cent or more allowed per step compounds through the method's
history, and its predictor-corrector estimate is unreliable there
(``docs/guide/numerics/choosing-a-stepper.md``).
"""

DEFAULT_RTOL = 1e-8
"""Relative tolerance a GNLSE run asks for unless told otherwise."""

DEFAULT_ATOL = 1e-14
"""Absolute tolerance, in the square root of watts, of the same default."""

DEFAULT_N_SAVE = 201
"""Planes a GNLSE run stores, counting the one at ``z = 0``."""

WAVEGUIDE_RTOL = 1e-6
"""Relative tolerance a modal waveguide run asks for unless told otherwise.

The default of :meth:`Simulation.from_capillary`, :meth:`Simulation.from_waveguide` and
the keyword interfaces built on them. It is looser than :data:`DEFAULT_RTOL`, the GNLSE
default, because each step of a modal waveguide run -- a transverse transform and a
response pass over the quadrature nodes at every stage -- costs far more than a GNLSE
step. It is a starting point, not a guarantee: the map from this local tolerance to the
error at the output is problem dependent, so a production run should calibrate it rather
than assume (``docs/dev/stepper.md`` section 3).
"""

WAVEGUIDE_ATOL = 1e-10
"""Absolute tolerance of a modal waveguide run, in sqrt(W).

The companion of :data:`WAVEGUIDE_RTOL`, looser than :data:`DEFAULT_ATOL` for the same
reason.
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
)
"""The settings read when a run executes, not when it is translated."""

NoiseModel = Literal["input", "modified"]
"""Which shot-noise model a run carries; ``None`` carries none.

The two models are described in ``docs/guide/media/noise.md``.
"""


@dataclass(frozen=True, eq=False)
class SimulationSettings(Displayed):
    """How a run is stepped, what it records and which backend executes it.

    Everything that is *not* a physics object, in one group, so that a simulation can be
    constructed from the physics and this. Equality is by identity (``eq=False``)
    because ``save_planes`` is an array.

    Attributes
    ----------
    controller : solprop.stepper.controller.PIController
        Tolerances, safety and gains; the tolerances and the norm choice live here and
        nowhere else (``docs/dev/kernel-contract.md`` section 7).
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
    integrator : Multistep, RungeKutta or LowStorageRungeKutta
        The method that steps the run (``docs/guide/numerics/choosing-a-stepper.md``).
        The default, ``Multistep()``, is the low-call exponential Adams method, which
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
    absorber_scheme : {'nonlinear', 'split_step'}
        How the temporal collar is applied (``docs/guide/numerics/boundaries.md``).
        ``"nonlinear"``, the default, adds ``-alpha(t) E(t) / 2`` inside every stage;
        ``"split_step"`` multiplies the accepted state by ``exp(-alpha(t) h / 2)`` once
        per step, which is a Lie splitting and first order wherever the collar does any
        work. The second stays reachable so the comparison can be re-measured and older
        results reproduced.
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
        resolves to ``"native"`` when installed, otherwise ``"numba_cpu"``, at
        settings construction. The stored name is concrete, including in scan workers.
        An installed backend's import or construction errors never trigger a fallback.
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
    transverse_absorber : solprop.boundaries.TransverseCollar or None
        A transverse-wavevector safety ramp on a Cartesian or line grid (SEP 22
        section 4.3), folded into the operator as an exact diagonal rate. ``None``,
        the default, carries none. It damps content reaching the grid's outer
        wavenumbers; it does not replace sizing the grid by the one-half rule.
        Refused on the radial and hybrid planar kinds and on modal runs.
    adiabatic_accuracy : solprop.adiabatic.AdiabaticAccuracy or None
        The forward-field and reflected-energy budgets a radial medium varying
        along z must meet (SEP 34 section 4.5). ``None``, the default, is
        ``AdiabaticAccuracy()``, both 1e-4; a path exceeding them is refused.
    tdse_execution : solprop.nonlinear.tdse.TDSEExecution or None
        Explicit atomic runtime resources for a TDSE response. Requires an explicit
        ``backend="numpy_ref"`` and a Multistep integrator. Execution resources are
        outside the scientific resume identity; attempted-call budgets are cumulative.

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

    controller: PIController
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
    absorber_scheme: AbsorberScheme = DEFAULT_ABSORBER_SCHEME
    quadrature: TransverseQuadrature | None = None
    components: Sequence[str] | None = None
    ionisation_field_range: tuple[float, float] | None = None
    node_budget: int | None = None
    stats: StatsSpec | None = None
    polarisation_taper: bool = False
    backend: str = "auto"
    integrator: Multistep | RungeKutta | LowStorageRungeKutta = field(
        default_factory=Multistep
    )
    resume_history: bool | None = None
    tdse_execution: TDSEExecution | None = None
    fields: FieldsChoice = "all"
    field_planes: Sequence[int] = ()
    evanescent_rate: float | None = None
    transverse_absorber: TransverseCollar | None = None
    adiabatic_accuracy: AdiabaticAccuracy | None = None

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
                ("rtol", f"{self.controller.rtol:g}"),
                ("atol", f"{self.controller.atol:g}"),
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
                f"rtol {self.controller.rtol:g}, atol {self.controller.atol:g}",
            ),
            ("steps", f"{schedule}, dz_max {engineering(self.dz_max, 'm')}"),
            ("shock term", self.shock),
            (
                "absorber",
                f"{self.absorber_scheme}" if self.absorber is not None else "none",
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
        if self.fields == "planes" and self.step_planes:
            msg = (
                "fields='planes' names saved planes by index, and step_planes saves "
                "the march's own steps as well, so the plane count is not known in "
                "advance; keep fields='all' or 'none' beside step_planes"
            )
            raise DriverError(msg)
        if self.tdse_execution is not None and (
            not isinstance(self.tdse_execution, TDSEExecution)
            or self.backend != "numpy_ref"
            or not isinstance(self.integrator, Multistep)
        ):
            raise DriverError(
                "tdse_execution requires explicit numpy_ref and Multistep settings"
            )
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
        # An Adams integrator resolves "auto" as the Runge-Kutta march does: every
        # backend steps the exponential formulation on every route, and the
        # interaction one on every route but a coupled flow, which it refuses by name
        # (SEP 16 phases 3 and 5).
        object.__setattr__(self, "backend", resolve_name(self.backend))


def _stepper_record(settings: StepperSettings) -> StepperRecord:
    """Return the run's stepper record, naming an Adams integrator where one steps it.

    The method is recorded in full (output schema 9), and the interval cap and start-up
    rule with the limits (schema 10), so that a run and its resume hash the stepping
    they were taken with.

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
        The general modal identity of output schema 7, as
        :func:`solprop.translate.general.modal_identity` writes it: the channel
        labels, the table, metric, response and guide identities, the frame, the
        active band and the accessors the basis admits. ``None`` for every other
        route. It is part of the schema-7 resume key.
    materials : solprop.physdata.selection.MaterialManifest or None, optional
        The run's material manifest of output schema 8
        (:func:`solprop.driver.material_output.material_manifest`); ``None`` is the
        empty manifest of a run of solprop's own materials. Its identities and
        policies are part of the schema-8 resume key.
    quadrature : solprop.modes.quadrature.TransverseQuadrature or None, optional
        The transverse rule of a resolved modal run, recorded with its nodes and
        physical weights on the first mode's cross-section and part of the schema-17
        resume key (SEP 19). ``None`` for every other route.
    frame : solprop.linear.Frame or None, optional
        The run's frame, whose reference frequency is the origin of the linear frame
        term on an envelope grid; ``None`` is the grid's co-moving frame. It enters
        the schema-24 plane-reference record, not the hash.

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
        # A schema-4 Cartesian checkpoint must not resume with changed cutoff bins.
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
    """Return the schema-24 plane-reference record of a run (SEP 31).

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
    reference = grid.omega0 if frame is None else frame.omega0
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
        "frame": frame_record(
            problem.linear, reference if grid.kind == "envelope" else 0.0
        ),
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
        manifest (output schema 8) with every optmat-backed object the run carries.
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
    >>> grid = EnvelopeGrid.from_limits(0.01, 835e-9, (500e-9, 2e-6), 1e-12)
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
        self.noise = noise
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
            ``field_planes`` or ``tdse_execution``, naming them.
        """
        if self._problem is not None:
            changed = [
                one.name
                for one in dataclasses.fields(SimulationSettings)
                if one.name not in _EXECUTION_SETTINGS
                and not _same_setting(
                    getattr(self._settings, one.name), getattr(settings, one.name)
                )
            ]
            if changed:
                msg = (
                    f"this Simulation is already translated, so a new "
                    f"{', '.join(changed)} would be ignored; set the settings before "
                    "the first translate() or run(), or build a new Simulation (only "
                    "the backend, resume_history, fields, field_planes and TDSE "
                    "execution resources may change after translation)"
                )
                raise DriverError(msg)
        if hasattr(self, "noise"):
            _check_initial_position(self, settings)
            _check_tdse_execution(self, settings)
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
                ("noise", "none" if self.noise is None else type(self.noise).__name__),
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
        ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], lambda0=835e-9,
        ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
        ...     width=100e-15, peak_power=9.8, pulseshape="sech", n_save=3,
        ... )
        >>> problem = simulation.translate()
        >>> problem.shape, simulation.translate() is problem
        ((1, 1, 1, 1024), True)
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
                    controller=settings.controller,
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
                    absorber_scheme=settings.absorber_scheme,
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
                    transverse_absorber=settings.transverse_absorber,
                    adiabatic_accuracy=settings.adiabatic_accuracy,
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
        recorded = self._derived_configuration(initial, configuration, coupling)
        updated = Simulation(
            self._base_grid,
            self.modes,
            self.responses,
            initial,
            settings=settings,
            initial_z=launch_z,
            noise=replacement_noise,
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
        self._check_replaced_input(
            updated._field0, None if isinstance(noise, str) else replacement_noise
        )
        record = dataclasses.replace(
            original.initial,
            z=launch_z,
            field0=np.asarray(updated._field0[np.newaxis], dtype=np.complex128),
            noise0=None
            if replacement_noise is None
            else np.asarray(
                replacement_noise.spectrum[np.newaxis], dtype=np.complex128
            ),
            noise_model="none"
            if replacement_noise is None
            else replacement_noise.model,
        )
        if original.transform.geometry is None:
            weight = power_weight_at(original, launch_z)
            record = dataclasses.replace(
                record,
                field0=power_amplitude(record.field0, weight, inverse=True),
                noise0=None
                if record.noise0 is None
                else power_amplitude(record.noise0, weight, inverse=True),
            )
        updated._problem = dataclasses.replace(
            original,
            initial=record,
            stepper=dataclasses.replace(
                original.stepper, save_planes=settings.save_planes
            ),
            flags=dataclasses.replace(
                original.flags, has_noise=record.noise0 is not None
            ),
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
            numerical settings and transverse geometry with this one.

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
        ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], lambda0=835e-9,
        ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
        ...     width=100e-15, peak_power=9.8, pulseshape="sech", n_save=3,
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
                    "absorber_scheme": self.settings.absorber_scheme,
                    "propagator_mode": self.settings.propagator_mode,
                }
            )
            if self.frame is not None:
                configuration["frame"] = self.frame
            # The operator settings of SEP 22 phase 2, recorded only where set, so
            # every earlier run keeps its digest; a collar as its two numbers.
            if self.settings.evanescent_rate is not None:
                configuration["evanescent_rate"] = self.settings.evanescent_rate
            if self.settings.transverse_absorber is not None:
                configuration["transverse_absorber"] = self.settings.transverse_absorber
            if self.settings.adiabatic_accuracy is not None:
                configuration["adiabatic_accuracy"] = self.settings.adiabatic_accuracy
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
        ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], lambda0=835e-9,
        ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12, raman=True,
        ...     width=100e-15, peak_power=9.8, pulseshape="sech", n_save=3,
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
        """Return the schema-7 modal identity of this run, or none at all.

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
        )

    def run(  # noqa: PLR0913 - independent execution and output controls
        self,
        output: Output | Path | str | None = None,
        *,
        progress: ProgressReporter | ProgressChoice | None = None,
        progress_period: float = DEFAULT_PROGRESS_PERIOD,
        resume: bool | Literal["interpolated"] = False,
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
        resume : bool or {'interpolated'}, optional
            Continue the file at `output` if it holds this same run, instead of
            replacing it. Only a path can be resumed: the driver has to open the file in
            append mode itself. A run interpolating its planes
            (``interpolate_planes``) whose last plane was interpolated inside a step is
            refused unless this is ``"interpolated"``: it then continues afresh from
            the interpolated field, and the position is recorded in the annotation
            :data:`~solprop.output.schema.INTERPOLATED_RESUMES_ANNOTATION`.
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
        DriverError
            If a resume is asked for and `output` is not a path, or the last plane
            of a dense run was interpolated and `resume` is not ``"interpolated"``;
            or `tdse_archive` is given without a TDSE response or at the path of
            the optical file.
        TDSEArchiveError
            If the archive's path holds another file, or a write to it fails; the
            run then stops at its last optical checkpoint.
        MemoryBudgetError
            If the run's estimated persistent memory exceeds its budget; nothing has
            been allocated or written.
        OutputError
            If a file is there and was written for a different run; the message names
            the fields that changed.

        Examples
        --------
        >>> from solprop.driver import Simulation
        >>> simulation = Simulation.from_gnlse(
        ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], lambda0=835e-9,
        ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
        ...     width=100e-15, peak_power=9.8, pulseshape="sech", n_save=3, rtol=1e-6,
        ... )
        >>> handler = simulation.run()
        >>> handler.n_saved, handler["z"].tolist()
        (3, [0.0, 0.005, 0.01])
        """
        with _optmat.material_context(self._material_log):
            problem = self.translate()
            metadata = self.metadata()
            handler, state = _open_output(
                output, metadata, resume=resume, dense=self._observes_densely()
            )
            if state is not None:
                next_plane = _validate_resume(
                    problem, state, self.settings.tdse_execution
                )
                if _completed(problem, state, next_plane):
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
                start_plane = self._prime(backend, problem, state)
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
                )
            except BaseException as error:
                failure = error
                raise
            finally:
                _close_run(backend, binding, archive, handler, failure)
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

    def _observes_densely(self) -> bool:
        """Return whether the run observes save planes inside steps (SEP 14).

        Returns
        -------
        bool
            Whether save planes may interpolate between accepted step endpoints.
        """
        return self.settings.interpolate_planes

    def _prime(
        self, backend: Backend, problem: KernelProblem, state: ResumeState | None
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
        backend.set_initial(field, None, z=state.plane.z)
        if state.outcome is not None:
            backend.restore_call_ledgers(
                state.outcome.logical_calls, state.outcome.atomic_calls
            )
        self._continue_march(backend, state)
        return next_plane

    def _continue_march(self, backend: Backend, state: ResumeState) -> None:
        """Continue the Adams history from the file's snapshot, where one fits.

        Parameters
        ----------
        backend : solprop.backends.protocol.Backend
            The backend, loaded at the resume position.
        state : solprop.output.hdf5.ResumeState
            What the file holds.
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
        noise: NoiseModel | None = None,
        seed: int | None = None,
        materials: MaterialSelection | None = None,
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
        noise : {None, 'input', 'modified'}, optional
            One photon per transverse column, polarisation and active bin.
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
            carrier). The run records its material manifest (output schema 8).
            ``None`` (the default) is ``MaterialSelection()``.

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
            settings=settings,
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
        noise: NoiseModel | None = None,
        seed: int | None = None,
        materials: MaterialSelection | None = None,
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
        noise : {None, 'input', 'modified'}, optional
            One photon per transverse column, polarisation and active bin.
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
            carrier). The run records its material manifest (output schema 8).
            ``None`` (the default) is ``MaterialSelection()``.

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
            settings=settings,
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
            and record it in the run's manifest (output schema 8): a fill of a
            built-in gas with its default susceptibility takes the selection's
            single-species susceptibility (no mixture is needed), and a slab without
            a declared cladding takes optmat's recommended fused silica (SEP 3, Q5).
            A supplied susceptibility or cladding is taken as it is. An optmat
            rate's energy per event is bound to the input's peak field (SEP 3, Q14),
            and the input is checked against each rate. ``None`` (the default) is
            ``MaterialSelection()``.

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
        simulation = cls(
            grid,
            (),
            responses,
            initial,
            settings=settings,
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
        noise: NoiseModel | None = None,
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
        noise : {None, 'input', 'modified'}, optional
            One photon per transverse column, polarisation and active bin.
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
        simulation = cls(
            grid,
            (),
            responses,
            initial,
            settings=settings,
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
        if noise is None and seed is not None:
            msg = "a noise seed requires a noise model"
            raise DriverError(msg)
        draw = (
            None
            if noise is None
            else spatial_noise(
                SpatialFlux.resolve(grid, space, medium, z=simulation.initial_z),
                np.random.default_rng(seed),
                model=noise,
                n_pol=field.shape[1],
            )
        )
        simulation.noise = draw
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
        lambda0: float,
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
        pulseshape: str = "gauss",
        energy: float | None = None,
        peak_power: float | None = None,
        phases: Sequence[float] | np.ndarray = (),
        pulses: Sequence[Pulse] | None = None,
        loss_db_per_m: float | None = None,
        shock: bool = True,
        raman: bool = False,
        raman_fraction: float | None = None,
        noise: NoiseModel | None = None,
        seed: int | None = None,
        rtol: float = DEFAULT_RTOL,
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
        backend: str = "auto",
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
            Taylor coefficients of ``beta`` about `lambda0`, ``beta_0, beta_1, ...`` in
            s^n/m. Give exactly one of this, `beta` or `fibre`.
        lambda0 : float
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
            `lambda0`, and so is a provider channel's, by a fresh solve there (along
            the path of a moving guide); without an area column a table requires
            `gamma` and uses the 1 m^2 convention.
        n2 : float or str, optional
            Nonlinear index in m^2/W, requiring `fibre` and ``gamma=None``.
            Exactly one of `gamma` and `n2` is required. The mapping is
            ``gamma = n2 omega0 / (c A_eff)``, with unit prefactor index. Under
            `materials`, ``"recommended"`` takes optmat's silica bundle's total
            ``n2`` at `lambda0`.
        fwhm : float, optional
            Intensity full width at half maximum of the input, s.
        width : float, optional
            Natural width of a ``sech**2`` input, s. Give this or `fwhm`.
        pulseshape : {'gauss', 'sech'}, optional
            Shape of the input.
        energy : float, optional
            Pulse energy, J. Give this or `peak_power`.
        peak_power : float, optional
            Peak power, W.
        phases : array_like, optional
            Optics-convention spectral phase about `lambda0`: CEP in rad, then group
            delay, GDD, TOD, ... in s^n.
        pulses : sequence of solprop.pulses.Pulse or None, optional
            Several pulses launched together, for two-colour and multi-colour
            inputs: each a :mod:`solprop.pulses` object with its own wavelength,
            duration, energy or peak power, phase and delay; the input is the sum of
            their fields (:class:`~solprop.pulses.PulseSum`). Replaces the
            single-pulse keywords `fwhm`, `width`, `pulseshape`, `energy`,
            `peak_power` and `phases`, which must then be left unset; `lambda0` still
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
        noise : {None, 'input', 'modified'}, optional
            Which shot-noise model to carry (``docs/guide/media/noise.md``).
        seed : int, optional
            Entropy of the generator the realisation is drawn from, recorded in the
            output so a run is reproducible.
        rtol : float, optional
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
            ``None``, the default, takes ``Multistep()``: the low-call exponential
            Adams method, one response evaluation per step. ``RungeKutta()`` is the
            interaction-picture DOPRI5 march, for a cheap response or a state too
            large for the Adams history. ``LowStorageRungeKutta()`` replays
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
        backend : str, optional
            Registry name, or ``"auto"`` (the default) to prefer ``native`` if installed
            and otherwise use Numba CPU. Import or construction errors do not fall back.
        materials : solprop.physdata.selection.MaterialSelection or None, optional
            Take fused silica's nonlinear response from optmat's silica bundle
            (:class:`~solprop.physdata.glasses.SilicaBundle`): with `raman` on, an
            omitted fraction is the bundle's ``f_R`` and the response its sampled
            shape; ``n2="recommended"`` is its total ``n2``. The fibre's dispersion,
            area and a given ``gamma`` or ``n2`` are the caller's. The run records
            its manifest (output schema 8), the held-back fraction and its origin
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
        ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], lambda0=835e-9,
        ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
        ...     width=100e-15, peak_power=9.8, pulseshape="sech", n_save=3,
        ... )
        >>> simulation.settings.save_planes.tolist()
        [0.005, 0.01]
        """
        materials = MaterialSelection() if materials is None else materials
        _refuse_dispersive_policy(materials, "a GNLSE run's fitted gamma")
        working = _gnlse_grid(grid, length, lambda0, wavelength_limits, trange)
        _check_selection_carrier(materials, working.reference_wavelength, lambda0)
        mapping = _gnlse_fibre(
            gamma,
            betas,
            beta,
            working,
            lambda0=lambda0,
            loss=loss_db_per_m,
            fibre=fibre,
            n2=n2,
            shock=shock,
            raman=raman,
            raman_fraction=raman_fraction,
            selection=materials,
        )
        pulse = _shaped_pulse(
            pulseshape,
            lambda0=lambda0,
            fwhm=fwhm,
            width=width,
            energy=energy,
            peak_power=peak_power,
            phases=phases,
            pulses=pulses,
        )
        settings = SimulationSettings(
            controller=PIController(rtol=rtol, atol=atol),
            save_planes=_save_planes(length, save_planes, n_save),
            # Relative defaults, so that the first step and the cap scale with the
            # fibre.
            dz_max=length / 2 if dz_max is None else dz_max,
            dz_init=length * 1e-4 if dz_init is None else dz_init,
            auto_init=dz_init is None,
            schedule=schedule,
            integrator=Multistep() if integrator is None else integrator,
            interpolate_planes=interpolate_planes,
            step_planes=step_planes,
            density=1.0,
            shock=shock,
            prefactor_index=mapping.prefactor_index,
            stats=default_spec("mode_averaged") if stats is None else stats,
            polarisation_taper=polarisation_taper,
            backend=backend,
        )
        return cls(
            working,
            mapping.mode,
            list(mapping.responses),
            pulse,
            settings=settings,
            noise=_shot_noise(noise, working, seed),
            configuration={
                "gnlse": mapping,
                "loss_db_per_m": 0.0 if loss_db_per_m is None else loss_db_per_m,
            },
            materials=materials,
        )

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
        lambda0: float,
        wavelength_limits: tuple[float, float] | None = None,
        trange: float | None = None,
        grid: TimeGrid | None = None,
        envelope: bool = False,
        fwhm: float | None = None,
        width: float | None = None,
        pulseshape: str = "gauss",
        energy: float | None = None,
        peak_power: float | None = None,
        phases: Sequence[float] | np.ndarray = (),
        pulses: Sequence[Pulse] | None = None,
        modes: int | Sequence[Mode] = 1,
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
        absorber: Absorber | None = None,
        absorber_scheme: AbsorberScheme | None = None,
        noise: NoiseModel | None = None,
        seed: int | None = None,
        rtol: float = WAVEGUIDE_RTOL,
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
        lambda0 : float
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
        pulseshape : {'gauss', 'sech'}, optional
            Transform-limited shape of the input.
        energy : float or None, optional
            Pulse energy, J. Give this or `peak_power`.
        peak_power : float or None, optional
            Peak power, W, realised after the spectral phases.
        phases : sequence of float, optional
            Taylor spectral phases about `lambda0`, optics convention: CEP in rad,
            group delay in s, GDD in s^2, and so on.
        pulses : sequence of solprop.pulses.Pulse or None, optional
            Several pulses launched together, for two-colour and multi-colour
            inputs: each a :mod:`solprop.pulses` object with its own wavelength,
            duration, energy or peak power, phase and delay; the input is the sum of
            their fields (:class:`~solprop.pulses.PulseSum`). Replaces the
            single-pulse keywords `fwhm`, `width`, `pulseshape`, `energy`,
            `peak_power` and `phases`, which must then be left unset; `lambda0` still
            sets the grid's reference wavelength.
        modes : int or sequence of Mode, optional
            How many HE_1m modes to propagate, or the mode set itself. A set is checked
            for orthonormality. Supplied sets may include TabulatedHollowMode or
            TabulatedMode with their own tables. A supplied set **carries its own
            geometry**: `radius` must agree with every mode's core radius along the run,
            and a `model`, `loss` or `cladding` declared away from its default must be
            every mode's own, or the call is refused by name rather than the argument
            ignored.
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
            varying guide, a scalar run and no noise model, and all three backends
            execute it (a ``method="cf4"`` wall ``numpy_ref`` alone); every other mode
            set or geometry is refused by name. ``None``, the default, carries no wall:
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
        absorber : solprop.boundaries.Absorber or None, optional
            The collar rates; ``None`` builds the default collar of the grid.
        absorber_scheme : {'nonlinear', 'split_step'} or None, optional
            How the temporal collar is applied (ADR-025): inside every stage, the
            default, or once per accepted step. The low-call Adams methods of
            SEP 13 step much further with ``'split_step'``, whose first-order
            splitting error stays inside the collar.
        noise : {None, 'input', 'modified'}, optional
            Which shot-noise model to carry.
        seed : int or None, optional
            Entropy of the noise generator.
        rtol : float, optional
            Relative step tolerance; the default is :data:`WAVEGUIDE_RTOL`, the modal
            waveguide default. Calibrate it for a production run
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
            plus every statistic only a backend can compute.
        node_budget : int, optional
            Bytes of node-chunk buffers the response pass may hold.
        backend : str, optional
            Registry name, or ``"auto"`` (the default) to prefer ``native`` if installed
            and otherwise use Numba CPU. Import or construction errors do not fall back.
        polarisation : str, float or None, optional
            Refused; this path is scalar.
        materials : solprop.physdata.selection.MaterialSelection or None, optional
            Take the run's built-in materials from optmat under this selection: a gas or
            fraction mapping of built-in gases, ``"air"`` and ``Gas.Air`` resolve
            through :func:`solprop.physdata.species` and
            :meth:`solprop.physdata.mixtures.Mixture.air` with it, and the default
            cladding (``cladding=None`` with a numbered mode set) is optmat's
            recommended silica. A supplied mixture, mode set, cladding or susceptibility
            is taken as it is. The run records its material manifest (output schema 8).
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
        ...     125e-6, 0.1, "He", (0.8, 0.1), lambda0=800e-9,
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
            lambda0=lambda0,
            wavelength_limits=wavelength_limits,
            trange=trange,
            grid=grid,
            envelope=envelope,
            fwhm=fwhm,
            width=width,
            pulseshape=pulseshape,
            energy=energy,
            peak_power=peak_power,
            phases=phases,
            pulses=pulses,
            modes=modes,
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
            noise=noise,
            seed=seed,
            rtol=rtol,
            atol=atol,
            dz_max=dz_max,
            dz_init=dz_init,
            schedule=schedule,
            integrator=Multistep() if integrator is None else integrator,
            interpolate_planes=interpolate_planes,
            step_planes=step_planes,
            save_planes=save_planes,
            n_save=n_save,
            stats=stats,
            node_budget=node_budget,
            backend=backend,
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
        lambda0: float,
        wavelength_limits: tuple[float, float] | None = None,
        trange: float | None = None,
        grid: TimeGrid | None = None,
        envelope: bool = False,
        fwhm: float | None = None,
        width: float | None = None,
        pulseshape: str = "gauss",
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
        n_r: int | None = None,
        n_theta: int | None = None,
        radial_rule: RadialRule = "gauss_kronrod",
        ionisation_field_range: tuple[float, float] | None = None,
        absorber: Absorber | None = None,
        noise: NoiseModel | None = None,
        seed: int | None = None,
        rtol: float = WAVEGUIDE_RTOL,
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
            `pulseshape`) are refused beside it, because the beam carries its own
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
        lambda0 : float
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
        pulseshape : {'gauss', 'sech'}, optional
            Transform-limited shape of the input.
        energy : float or None, optional
            Pulse energy, J. Give this or `peak_power`.
        peak_power : float or None, optional
            Peak power, W, realised after the spectral phases.
        phases : sequence of float, optional
            Taylor spectral phases about `lambda0`, optics convention: CEP in rad,
            group delay in s, GDD in s^2, and so on.
        pulses : sequence of solprop.pulses.Pulse or None, optional
            Several pulses launched together, for two-colour and multi-colour
            inputs: each a :mod:`solprop.pulses` object with its own wavelength,
            duration, energy or peak power, phase and delay; the input is the sum of
            their fields (:class:`~solprop.pulses.PulseSum`). Replaces the
            single-pulse keywords `fwhm`, `width`, `pulseshape`, `energy`,
            `peak_power` and `phases`, which must then be left unset; `lambda0` still
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
        absorber : solprop.boundaries.Absorber or None, optional
            The collar rates; ``None`` builds the default collar of the grid.
        noise : {None, 'input', 'modified'}, optional
            Which shot-noise model to carry.
        seed : int or None, optional
            Entropy of the noise generator.
        rtol : float, optional
            Relative step tolerance; the default is :data:`WAVEGUIDE_RTOL`, the modal
            waveguide default. Calibrate it for a production run
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
            ``None``, the default, takes ``Multistep()``: the low-call exponential
            Adams method. ``RungeKutta()`` is the interaction-picture DOPRI5 march.
            ``LowStorageRungeKutta()`` replays `schedule` in three registers.
        save_planes : sequence of float or None, optional
            Positions to save at, m. Give this or `n_save`.
        n_save : int or None, optional
            How many planes to save, counting ``z = 0``.
        stats : solprop.stats.StatsSpec or None, optional
            Which statistics to record; ``None`` takes
            :func:`solprop.stats.default_spec` for a multimode run with the profiles,
            plus every statistic only a backend can compute.
        node_budget : int, optional
            Bytes of node-chunk buffers the response pass may hold.
        backend : str, optional
            Registry name, or ``"auto"`` (the default) to prefer ``native`` if installed
            and otherwise use Numba CPU. Import or construction errors do not fall back.
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
            manifest (output schema 8). ``None`` (the default) is
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
                lambda0=lambda0,
                wavelength_limits=wavelength_limits,
                trange=trange,
                grid=grid,
                envelope=envelope,
                fwhm=fwhm,
                width=width,
                pulseshape=pulseshape,
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
            waveguide=True,
            length=length,
            gas=gas,
            pressure=pressure,
            pressure_floor=pressure_floor,
            vacuum_end=vacuum_end,
            lambda0=lambda0,
            wavelength_limits=wavelength_limits,
            trange=trange,
            grid=grid,
            envelope=envelope,
            fwhm=fwhm,
            width=width,
            pulseshape=pulseshape,
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
        lambda0: float,
        wavelength_limits: tuple[float, float] | None,
        trange: float | None,
        grid: TimeGrid | None,
        fwhm: float | None,
        width: float | None,
        pulseshape: str,
        energy: float | None,
        peak_power: float | None,
        phases: Sequence[float] | np.ndarray,
        pulses: Sequence[Pulse] | None,
        absorber: Absorber | None,
        noise: NoiseModel | None,
        seed: int | None,
        rtol: float,
        atol: float,
        dz_max: float | None,
        dz_init: float | None,
        schedule: FixedSchedule | None,
        save_planes: Sequence[float] | np.ndarray | None,
        n_save: int | None,
        stats: StatsSpec | None,
        backend: str,
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
        lambda0, wavelength_limits, trange, grid : object
            The grid, as :func:`_capillary_grid` takes them; a **field** grid.
        fwhm, width, pulseshape, energy, peak_power, phases, pulses : object
            The input pulse, as :func:`_shaped_pulse` takes it.
        absorber : solprop.boundaries.Absorber or None
            The collar rates; ``None`` builds the grid's default collar.
        noise : {None, 'input', 'modified'}
            Which shot-noise model to carry (``docs/guide/media/noise.md``).
        seed : int or None
            Entropy of the noise generator.
        rtol, atol, dz_max, dz_init, schedule, save_planes, n_save : object
            The step controller and the schedule, as the other entry points take them.
        stats : solprop.stats.StatsSpec or None
            Which statistics to record; ``None`` takes :func:`_general_stats`.
        backend : str
            Registry name, or ``"auto"``.
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
            grid, length, lambda0, wavelength_limits, trange, envelope=False, thg=False
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
            pulseshape,
            lambda0=lambda0,
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
        settings = SimulationSettings(
            controller=PIController(rtol=rtol, atol=atol),
            save_planes=planes,
            dz_max=length / 2 if dz_max is None else dz_max,
            dz_init=length * 1e-4 if dz_init is None else dz_init,
            auto_init=dz_init is None,
            schedule=schedule,
            integrator=Multistep() if integrator is None else integrator,
            absorber=collar(working) if absorber is None else absorber,
            # The route carries its nodes, its materials and its normalisation in
            # its own leaves: the static transverse quadrature, the component
            # selection, the density path and the index override are all refused
            # by the translation layer and are not settings of this run.
            stats=_general_stats() if stats is None else stats,
            backend=backend,
        )
        simulation = cls(
            working,
            channels,
            [],
            field0,
            settings=settings,
            noise=_general_noise(noise, working, seed, provider),
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
        if beam is None:
            return simulation
        return _incident_launch(simulation, beam, cast("CouplingModel", coupling))

    @classmethod
    def _from_hollow(  # noqa: PLR0913 - the interface's keyword surface
        cls,
        radius: float | ZProfile | None,
        length: float,
        gas: Gas | str | Mapping[str, float] | Mixture,
        pressure: Pressure | None = None,
        *,
        waveguide: bool = False,
        pressure_floor: float | None = None,
        vacuum_end: VacuumEnd | None = None,
        lambda0: float,
        wavelength_limits: tuple[float, float] | None = None,
        trange: float | None = None,
        grid: TimeGrid | None = None,
        envelope: bool = False,
        fwhm: float | None = None,
        width: float | None = None,
        pulseshape: str = "gauss",
        energy: float | None = None,
        peak_power: float | None = None,
        phases: Sequence[float] | np.ndarray = (),
        pulses: Sequence[Pulse] | None = None,
        modes: int | Sequence[Mode] = 1,
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
        absorber: Absorber | None = None,
        absorber_scheme: AbsorberScheme | None = None,
        noise: NoiseModel | None = None,
        seed: int | None = None,
        rtol: float = WAVEGUIDE_RTOL,
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
        lambda0 : float
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
        pulseshape : {'gauss', 'sech'}, optional
            Transform-limited shape of the input.
        energy : float or None, optional
            Pulse energy, J. Give this or `peak_power`.
        peak_power : float or None, optional
            Peak power, W, realised after the spectral phases.
        phases : sequence of float, optional
            Taylor spectral phases about `lambda0`, optics convention: CEP in rad,
            group delay in s, GDD in s^2, and so on.
        pulses : sequence of solprop.pulses.Pulse or None, optional
            Several pulses launched together, for two-colour and multi-colour
            inputs: each a :mod:`solprop.pulses` object with its own wavelength,
            duration, energy or peak power, phase and delay; the input is the sum of
            their fields (:class:`~solprop.pulses.PulseSum`). Replaces the
            single-pulse keywords `fwhm`, `width`, `pulseshape`, `energy`,
            `peak_power` and `phases`, which must then be left unset; `lambda0` still
            sets the grid's reference wavelength.
        modes : int or sequence of Mode, optional
            How many HE_1m modes to propagate, or the mode set itself. A set is checked
            for orthonormality. The own HE_1m set is synthesised along x; a supplied set
            resolves the transverse components its modes need -- E_x and E_y for TE_0m,
            TM_0m, EH, higher HE or rotated modes -- with or without a coherent wall.
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
            varying guide, a scalar run and no noise model, and all three backends
            execute it (a ``method="cf4"`` wall ``numpy_ref`` alone); every other mode
            set or geometry is refused by name. ``None``, the default, carries no wall:
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
        absorber : solprop.boundaries.Absorber or None, optional
            The collar rates; ``None`` builds the default collar of the grid.
        absorber_scheme : {'nonlinear', 'split_step'} or None, optional
            How the temporal collar is applied (ADR-025): inside every stage, the
            default, or once per accepted step. The low-call Adams methods of
            SEP 13 step much further with ``'split_step'``, whose first-order
            splitting error stays inside the collar.
        noise : {None, 'input', 'modified'}, optional
            Which shot-noise model to carry.
        seed : int or None, optional
            Entropy of the noise generator.
        rtol : float, optional
            Relative step tolerance; the default is :data:`WAVEGUIDE_RTOL`, the modal
            waveguide default. Calibrate it for a production run
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
            plus every statistic only a backend can compute.
        node_budget : int, optional
            Bytes of node-chunk buffers the response pass may hold.
        backend : str, optional
            Registry name, or ``"auto"`` (the default) to prefer ``native`` if installed
            and otherwise use Numba CPU. Import or construction errors do not fall back.
        polarisation : str, float or None, optional
            Refused; this path is scalar.
        materials : solprop.physdata.selection.MaterialSelection or None, optional
            Take the run's built-in materials from optmat under this selection: a gas or
            fraction mapping of built-in gases, ``"air"`` and ``Gas.Air`` resolve
            through :func:`solprop.physdata.species` and
            :meth:`solprop.physdata.mixtures.Mixture.air` with it, and the default
            cladding (``cladding=None`` with a numbered mode set) is optmat's
            recommended silica. A supplied mixture, mode set, cladding or susceptibility
            is taken as it is. The run records its material manifest (output schema 8).
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
            grid, length, lambda0, wavelength_limits, trange, envelope=envelope, thg=thg
        )
        materials = MaterialSelection() if materials is None else materials
        _check_selection_carrier(materials, working.reference_wavelength, lambda0)
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
        rule = _capillary_quadrature(mode_set, n_r, n_theta, radial_rule=radial_rule)
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
                lambda0=lambda0,
                envelope=envelope,
                # The single gas's electronic response is optmat's, at the carrier
                # or dispersive, as the selection's electronic policy says (SEP 6).
                electronic=electronic_response(
                    mixture.components[0].species,
                    selection=materials,
                    wavelength=lambda0,
                    thg=thg,
                )
                if kerr
                else None,
                selection=materials,
            )
            responses += (
                _optmat_raman(gas, mixture, materials, thg, lambda0, length)
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
                    lambda0=lambda0,
                    envelope=envelope,
                    length=length,
                    selection=materials,
                    plasma_integration=plasma_integration,
                )
            )
        _refuse_idle_preionisation(mixture, responses)
        pulse = _shaped_pulse(
            pulseshape,
            lambda0=lambda0,
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
        boundary = collar(working) if absorber is None else absorber
        _warn_capillary_walkoff(mode_set.modes, working, boundary)
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
            integrator=Multistep() if integrator is None else integrator,
            interpolate_planes=interpolate_planes,
            step_planes=step_planes,
            density=None,
            absorber=boundary,
            absorber_scheme=(
                DEFAULT_ABSORBER_SCHEME if absorber_scheme is None else absorber_scheme
            ),
            quadrature=rule,
            components=_hollow_components(supplied=not isinstance(modes, int)),
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
                    else _capillary_stats(with_plasma=ionising, mixture=mixture),
                    responses,
                )
                if stats is None
                else stats
            )
            if atomic is None or stats is not None
            else _tdse_stats(_capillary_stats(with_plasma=False, mixture=mixture)),
            backend=backend,
            tdse_execution=None if atomic is None else atomic.execution,
        )
        simulation = cls(
            working,
            mode_set.modes,
            responses,
            field0,
            settings=settings,
            noise=_shot_noise(noise, working, seed, n_modes=len(mode_set)),
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
                lambda0,
                envelope,
                length,
                selection=materials,
                reference_field=reference,
                field_range=field_range,
                plasma_integration=plasma_integration,
            )
        )
        return simulation


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
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], lambda0=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, pulseshape="sech", n_save=3, rtol=1e-6,
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
            handler, _ = _open_output(output, metadata, resume=False)
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
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], lambda0=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, pulseshape="sech",
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
    if simulation.noise is not None or settings.schedule is not None:
        raise DriverError(
            "an interior launch requires noise=None and no fixed schedule"
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
        if not _completed(problem, state, next_plane):
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
    if outcome is not None and outcome.logical_calls:
        calls = CallLedger.from_record(outcome.logical_calls)
        method = problem.stepper.integrator
        if not isinstance(method, Multistep):
            raise DriverError("a logical call ledger requires Adams integration")
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


def _completed(problem: KernelProblem, state: ResumeState, next_plane: int) -> bool:
    """Return whether a file holds this run already finished.

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
        True when the file's outcome is complete, it holds no checkpoint, and no save
        plane is left: a resume has nothing to do.
    """
    return (
        state.outcome is not None
        and state.outcome.status == "complete"
        and state.checkpoint is None
        and next_plane == len(problem.stepper.save_planes)
    )


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
        "optical_configuration_hash": metadata.configuration_hash(),
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
    if simulation.noise is not None:
        raise DriverError("TDSE requires noise=None")
    if settings.schedule is not None or settings.interpolate_planes:
        raise DriverError("TDSE requires adaptive Adams with landed save planes")
    if settings.absorber_scheme != "nonlinear":
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
    resume: bool | Literal["interpolated"],
    dense: bool = False,
) -> tuple[Output, ResumeState | None]:
    """Choose the handler a run writes to, and read what a resumed file holds.

    Parameters
    ----------
    output : solprop.output.schema.Output or pathlib.Path or str or None
        The caller's choice.
    metadata : solprop.output.schema.RunMetadata
        The record of the run about to start, which a file must match to be continued.
    resume : bool or {'interpolated'}
        Whether to continue an existing file; ``"interpolated"`` also from a plane
        interpolated inside a step.
    dense : bool
        Whether the run observes its planes densely.

    Returns
    -------
    tuple
        The handler, and the state to resume from or ``None``.

    Raises
    ------
    DriverError
        If a resume is asked for and `output` is not a path, or a dense run would
        resume from an interpolated plane without ``resume="interpolated"``.
    OutputError
        If the file exists and holds a different run.
    """
    if resume and metadata.configuration.get("_benchmark_plasma_source"):
        raise DriverError("gated plasma benchmark output cannot be resumed")
    if isinstance(output, Path | str):
        path = Path(output)
        if resume and check_cache(path, metadata):
            state = resume_state(path)
            if dense and resume != "interpolated" and _interpolated(state):
                # An interpolated plane is not a state the march passed through, so
                # a run restarted there is not the run it continues (SEP 14 section
                # 5); a caller may accept that.
                msg = (
                    f"{path} ends on a plane interpolated inside a step at z = "
                    f"{state.plane.z:.6e} m, not a state the march passed through; "
                    "pass resume='interpolated' to continue afresh from it, or rerun"
                )
                raise DriverError(msg)
            return HDF5Output(path, append=True), state
        return HDF5Output(path), None
    if resume:
        msg = (
            "only a path can be resumed: the file has to be opened in append mode, "
            "which is the driver's to decide; pass output=<path> with resume=True"
        )
        raise DriverError(msg)
    return (MemoryOutput() if output is None else output), None


def _gnlse_grid(
    grid: TimeGrid | None,
    length: float,
    lambda0: float,
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
    lambda0 : float
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
    return EnvelopeGrid.from_limits(length, lambda0, wavelength_limits, trange)


def _gnlse_fibre(  # noqa: PLR0913 - the coefficients of the mapping
    gamma: float | None,
    betas: Sequence[float] | np.ndarray | None,
    beta: object,
    grid: TimeGrid,
    *,
    lambda0: float,
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
    lambda0 : float
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
    omega_ref = 2.0 * np.pi * C / lambda0
    mode = (
        None
        if fibre is None
        else _gnlse_mode(fibre, lambda0, needs_area=n2 is not None)
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


def _gnlse_mode(fibre: Mode | IndexTable, lambda0: float, *, needs_area: bool) -> Mode:
    """Resolve a supplied table or mode with its propagation area at the pump.

    Parameters
    ----------
    fibre : Mode or IndexTable
        Whole linear response.
    lambda0 : float
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
            else float(fibre.effective_area(np.array([2.0 * np.pi * C / lambda0]))[0])
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
    return SpecifiedMode.from_mode(fibre, wavelength=lambda0)


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
    pulseshape: str,
    *,
    lambda0: float,
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
    pulseshape : {'gauss', 'sech'}
        Shape of the input.
    lambda0 : float
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
        if pulseshape != "gauss":
            given.append("pulseshape")
        if given:
            msg = (
                f"pulses= describes every input pulse itself, and this call also sets "
                f"{', '.join(given)}: give each pulse its own duration, energy and "
                "phase, and leave the single-pulse keywords unset"
            )
            raise DriverError(msg)
        return PulseSum(tuple(pulses))
    if pulseshape == "sech":
        return SechPulse(
            wavelength=lambda0,
            fwhm=fwhm,
            width=width,
            energy=energy,
            peak_power=peak_power,
            phase=phases,
        )
    if pulseshape != "gauss":
        msg = f"pulseshape must be 'gauss' or 'sech', got {pulseshape!r}"
        raise DriverError(msg)
    if width is not None or fwhm is None:
        msg = (
            "a Gaussian input is specified by its intensity fwhm; give fwhm= rather "
            "than width=, which is the natural width of a sech**2 pulse"
        )
        raise DriverError(msg)
    return GaussianPulse(
        wavelength=lambda0,
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
        msg = f"noise must be None, 'input' or 'modified', got {noise!r}"
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
    lambda0: float,
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
    lambda0 : float
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
            length, lambda0, wavelength_limits, trange, thg=thg
        )
    return FieldGrid.from_limits(length, lambda0, wavelength_limits, trange)


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


def _capillary_responses(  # noqa: PLR0913 - the switches of the responses, one each
    gas: Gas | str,
    material: Species,
    *,
    kerr: bool,
    plasma: bool | IonisationRate,
    preionisation: float | ZProfile,
    integration: IntegrationPolicy | None,
    lambda0: float,
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
    lambda0 : float
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
        rate = _ionisation_rate(gas, plasma, lambda0, selection)
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
    lambda0: float,
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
    lambda0 : float
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
        wavelength=lambda0,
        length=length,
        explicit=True,
    )
    return [] if response is None else [response]


def _ionisation_rate(
    gas: Gas | str,
    plasma: bool | IonisationRate,
    lambda0: float,
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
    lambda0 : float
        Reference wavelength, m, the rate's carrier unless the selection names one.
    selection : MaterialSelection
        The run's material selection.

    Returns
    -------
    solprop.nonlinear.ionisation.IonisationRate
        The rate.
    """
    if isinstance(plasma, bool):
        return selected_rate(species(gas, selection=selection), selection, lambda0)
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
        For `tdse_execution` without `tdse`, an electronic keyword beside it, an
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


def _capillary_stats(*, with_plasma: bool, mixture: Mixture) -> StatsSpec:
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

    Returns
    -------
    solprop.stats.StatsSpec
        The spec, with no energy windows; add those for the bands of interest.
    """
    wanted = [
        name
        for name in BACKEND_STATISTICS
        if with_plasma or name not in PLASMA_STATISTICS
    ]
    base = default_spec("multimode", with_profiles=True)
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
    pulseshape: str,
    *,
    lambda0: float,
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
    pulseshape, lambda0, fwhm, width, energy, peak_power, phases, pulses : object
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
            pulseshape,
            lambda0=lambda0,
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
    if pulseshape != "gauss":
        given.append("pulseshape")
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
    return simulation.with_input(
        result.launch,
        noise=simulation.noise,
        configuration={"pulse": pulse, "launch": report},
    )


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
    rule: TransverseQuadrature,
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
    rule : solprop.modes.quadrature.TransverseQuadrature
        The run's transverse quadrature.

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
    for where in (on_axis_rule(limits), rule):
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
    rule : solprop.modes.quadrature.TransverseQuadrature
        The run's transverse quadrature.
    """

    field0: Complex[np.ndarray, "n_modes n_pol n_omega"]
    modes: ModeCollection
    grid: TimeGrid
    rule: TransverseQuadrature

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
    problem: KernelProblem, output: Output, responses: Sequence[object]
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
        with open_file(output.path) as reader:
            return _peak_field_reports(problem, reader, responses)
    if "species_peak_ionisation_rate" in output.stat_names:
        peaks = np.max(np.asarray(output["stats/species_peak_ionisation_rate"]), axis=0)
    elif "peak_ionisation_rate" in output.stat_names and len(labels) == 1:
        peaks = np.array([np.max(np.asarray(output["stats/peak_ionisation_rate"]))])
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
    rule: TransverseQuadrature,
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
    rule : solprop.modes.quadrature.TransverseQuadrature
        The run's fixed transverse rule, whose embedded coarse rule the
        quadrature alarm needs.

    Returns
    -------
    StatsSpec
        Statistics with unsupported geometry paths omitted.
    """
    base = _capillary_stats(with_plasma=with_plasma, mixture=mixture)
    dropped = {name for name in ("core_radius",) if name not in profiles}
    if not rule.kronrod:
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
    lambda0: float | None = None,
) -> None:
    """Refuse a carrier that is not the run's: one carrier sets the Kerr coefficient.

    Parameters
    ----------
    materials : MaterialSelection or None
        The run's material selection.
    reference : float
        The grid's reference wavelength in m, the run's carrier.
    lambda0 : float or None, optional
        The driver's `lambda0` in m, at which the responses are built.

    Raises
    ------
    DriverError
        If the selection's ``carrier_wavelength`` or `lambda0` is not the grid's
        reference wavelength: the electronic coefficient, the Raman response and the
        rate are all evaluated at the carrier.
    """
    if materials is None:
        return
    for name, value in (
        ("the selection's carrier_wavelength", materials.carrier_wavelength),
        ("lambda0", lambda0),
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
