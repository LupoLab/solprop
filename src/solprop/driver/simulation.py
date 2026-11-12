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

import copy
import dataclasses
import functools
import logging
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from jaxtyping import Complex, Float, Shaped

from solprop._display import Displayed, line, summarise
from solprop.backends.protocol import Backend, _all_finite
from solprop.backends.registry import select
from solprop.boundaries import Absorber, SpatialAbsorber
from solprop.coupling import (
    CouplingLaunch,
    CouplingModel,
    InputPort,
    PlaneBasis,
    modal_basis_identity,
)
from solprop.driver.defaults import (
    DEFAULT_ATOL,
    INTERIOR_LAUNCH_NOISE,
    PLANAR_NOISE,
    WAVEGUIDE_ATOL,
    resolve_noise,
)
from solprop.driver.inputs import (
    ModalLaunch,
    _bind_spatial_rates,
    _built,
    _built_state,
    _check_initial_position,
    _check_input_identity,
    _check_planar_input,
    _check_rate_inputs,
    _given_input,
    _redrawable,
    _replaced_launch,
    _shaped_pulse,
    _shot_noise,
    _spatial_draw,
    _with_noise,
)
from solprop.driver.material_output import material_manifest
from solprop.driver.metadata import (
    RecipeSource,
    _frame_follows_the_electronic_rule,
    metadata_for,
)
from solprop.driver.mixtures import ResponseSelection, mixture_profiles
from solprop.driver.resume import (
    _adopt_recorded_seed,
    _check_resume_choice,
    _continues,
    _interpolated,
    _open_output,
    _ResumeChoice,
    _start,
    _validate_resume,
)
from solprop.driver.routes.general import _from_general
from solprop.driver.routes.gnlse import _gnlse_fibre, _gnlse_grid
from solprop.driver.routes.hollow import (
    _flow_radius,
    _from_hollow,
    _population_label,
    _rebind_mode,
    _refuse_idle_preionisation,
    _warn_boreless_gradient,
    _warn_sampled_peak,
)
from solprop.driver.routes.materials import (
    _check_electronic_policy,
    _check_selection_carrier,
    _refuse_dispersive_policy,
)
from solprop.driver.routes.spatial import (
    _from_spatial,
    _planar_medium,
    _rechecked,
    _refuse_pending_check,
    _selected_medium,
    _selected_planar,
    _transverse_specification,
)
from solprop.driver.run import check_snapshot_request, field_retention, run_problem
from solprop.driver.runner import SimulationRunner, _peak_field_reports
from solprop.driver.settings import (
    _EXECUTION_SETTINGS,
    _RECIPE_EXECUTION,
    NoiseSetting,
    SimulationSettings,
    _boundary,
    _bounded,
    _keyword_controller,
    _quiet_collar,
    _same_setting,
    _save_planes,
    _scheme,
    _spatial_default,
    _spatial_settings,
    _watched,
    _with_boundary,
)
from solprop.driver.tdse_run import (
    _AtomicContext,
    _check_tdse_archive,
    _check_tdse_execution,
    _close_run,
    _consistency_gate,
    _prepare_atomic,
    _recorded_archive_uuid,
    _resolve_tdse_grid,
    _restore_atomic,
)
from solprop.exceptions import (
    AdiabaticTaperWarning,
    DriverError,
    GrazingIncidenceWarning,
    MaterialRangeWarning,
    SolpropWarning,
    StopRequested,
)
from solprop.gasflow import VacuumEnd
from solprop.grid import FreeGrid, LineGrid, RadialGrid, TimeGrid
from solprop.linear import Frame
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
from solprop.modes.guide import ModalGuide
from solprop.modes.material import JsonValue
from solprop.modes.protocol import Mode, ParameterisedMode
from solprop.modes.quadrature import RadialRule
from solprop.modes.slab import PlanarGrid
from solprop.modes.tabulated import IndexTable
from solprop.monitor import DEFAULT_BOUNDARY_CHECK, BoundaryCheck, BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.ionisation import IntegrationPolicy, IonisationRate
from solprop.nonlinear.optmat_rates import PeakFieldReport
from solprop.nonlinear.raman import (
    DampedOscillator,
    MolecularOptions,
    RamanResponse,
    RotationalLines,
)
from solprop.nonlinear.responses import DispersiveKerrResponse, PlasmaResponse
from solprop.nonlinear.tdse import TDSEExecution, TDSEGasResponse
from solprop.output.schema import (
    INTERPOLATED_RESUMES_ANNOTATION,
    PEAK_FIELDS_ANNOTATION,
    CheckpointOutput,
    ModeRecord,
    NoiseResolution,
    Output,
    PlaneStore,
    RunMetadata,
    StatedInput,
    recorded_input,
    serialise,
)
from solprop.output.snapshots import SnapshotController
from solprop.output.tdse_archive import TDSEArchive, TDSEArchiveWriter
from solprop.physdata import Gas, Mixture, _optmat
from solprop.physdata.mixtures import Pressure
from solprop.physdata.selection import MaterialLog, MaterialManifest, MaterialSelection
from solprop.physdata.sellmeier import GasSusceptibility
from solprop.profiles import ZProfile
from solprop.pulses import PlanarPulse, Pulse, SpatialPulse, SpatialPulseSum
from solprop.spatial import SpatialDispersion, SpatialMedium
from solprop.stats import StatsSpec, default_spec
from solprop.stepper.controller import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.stepper.tableau import DOPRI5
from solprop.translate.coupled import taper_identity, wall_identity
from solprop.translate.coupling import input_port as fill_port
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.general import GeneralModalRoute, modal_identity
from solprop.translate.masks import AbsorberScheme
from solprop.translate.problem import PropagationProblem, TranslationReuse, assemble
from solprop.translate.responses import KerrForm, declares_harmonic_carrier
from solprop.translate.routes import SpatialRoute, tdse_descriptor, transverse_of
from solprop.translate.run_request import RunRequest
from solprop.translate.tdse import TDSEBinding
from solprop.translate.transverse import band_wavenumbers
from solprop.transverse_monitor import TransverseEdge

if TYPE_CHECKING:
    from solprop.calibration.refine import RefinementReport
    from solprop.driver.estimate import RunEstimate

__all__ = ["Simulation"]

logger = logging.getLogger(__name__)

_PRESSURE_FORM_ENTRIES = 2
"""Entries in either of the two pair forms of the ``pressure`` keyword."""


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


class Simulation(Displayed):
    """A run that can be translated once and executed many times.

    Construct it from the physics objects and a
    :class:`~solprop.driver.settings.SimulationSettings`, or through
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

        Assigning a new :class:`~solprop.driver.settings.SimulationSettings` before the
        first :meth:`translate` (or :meth:`run`) takes effect. After it the translated
        problem is cached, so an assignment that changes anything the translation
        consumed is refused, because it could no longer change the run; the ``backend``
        and ``tdse_execution`` resources, which are read when the run executes, may
        still change.

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
        each :meth:`~solprop.driver.runner.SimulationRunner.run` then runs one input on
        it (SEP 22 section 5.2). Close it, or use it in a ``with`` block.

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
                _adopt_recorded_seed(self, output)
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
                    binding, archive = _prepare_atomic(
                        self,
                        problem,
                        _AtomicContext(handler, state, metadata),
                        tdse_archive,
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
                start = _start(
                    self,
                    backend,
                    problem,
                    state,
                    kept=kept is not None,
                    restoration=restoration,
                )
                _restore_atomic(binding, start)
                _consistency_gate(self, binding, problem, handler, metadata, backend)
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
        return _from_spatial(
            cls,
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
        simulation = _from_spatial(
            cls,
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
            (:func:`~solprop.driver.settings.default_collar`), and ``False`` switches
            both soft collars off and keeps the hard band mask
            (:func:`solprop.boundaries.no_absorber`).
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
            :func:`~solprop.driver.routes.hollow.default_radial_order` of the mode
            count.
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
            :func:`solprop.driver.inputs.default_field_range`.
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
        return _from_hollow(
            cls,
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
            Radial nodes of the fixed rule, or the x-axis nodes of a Cartesian product
            or slab rule. ``None`` takes
            :func:`~solprop.driver.routes.hollow.default_radial_order` of the mode
            count, or :func:`~solprop.modes.quadrature.cartesian_nodes` of the set's
            highest transverse order where that is larger.
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
            :func:`solprop.driver.inputs.default_field_range`.
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
            return _from_general(
                cls,
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
        return _from_hollow(
            cls,
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
