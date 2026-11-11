# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lower one request's scalar modal observations without entering a backend.

Julia evaluates each observation from the selected dense or committed field.
This module only prepares numeric inputs: the catalogue's packed output layout,
represented power and material profiles, a separate coarse on-axis map, and the
shared temporal monitor policy. It retains no Python physics object or callback.
Source population work, quadrature readings, noise floors and resumed monitor state
belong to the session's separately owned numerical preparation.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

import numpy as np
from jaxtyping import Bool, Complex, Float, Int

from solprop import fourier
from solprop.backends.julia._descriptors import (
    CoefficientTransfer,
    TableTransfer,
    lower_profile,
    lower_table,
    owned,
)
from solprop.backends.julia._profiles import ProfileLowering, lower_statistic_profile
from solprop.backends.julia._spatial import SeparableTransfer, lower_separable
from solprop.backends.julia._transverse import TransverseTransfer, lower_transverse
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.modes.quadrature import mode_matrix, on_axis_rule
from solprop.monitor import (
    _FLOOR,  # pyright: ignore[reportPrivateUsage] - shared monitor's launch-floor policy
    FLOOR_MARGIN,
    BoundaryMonitor,
)
from solprop.profiles import ZProfile
from solprop.stats import CATALOGUE, ON_AXIS_PLASMA_STATISTICS
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import ModalRoute, SpatialRoute
from solprop.translate.run_request import MethodKind, RunRequest

__all__ = [
    "AxisPopulationObservationTransfer",
    "ObservationProfilesTransfer",
    "ObservationTransfer",
    "PopulationMeasureTransfer",
    "PopulationObservationTransfer",
    "SpatialObservationTransfer",
    "SpatialWidthTransfer",
    "StatisticTransfer",
    "StatisticsGridTransfer",
    "TemporalObservationTransfer",
    "TemporalPolicyTransfer",
    "lower_observations",
]

_STATISTICS = frozenset(
    (
        "omega0",
        "energy",
        "energy_window",
        "peak_power",
        "peak_power_window",
        "peak_power_total",
        "physical_energy",
        "analytic_flux",
        "spectral_flux",
        "fwhm_t",
        "fwhm_t_outer",
        "density",
        "pressure",
        "core_radius",
        "species_ionisation",
        "species_electron_density",
        "electron_density",
        "peak_ionisation_rate",
        "species_density",
        "species_ionisation_energy",
        "species_peak_ionisation_rate",
        "plasma_undershoot",
        "plasma_work_defect",
        "plasma_boundary_level",
        "peak_intensity",
        "quadrature_error",
        *ON_AXIS_PLASMA_STATISTICS,
    )
)
_SPATIAL_STATISTICS = frozenset(
    (
        "energy",
        "density",
        "pressure",
        "core_radius",
        "species_ionisation",
        "species_electron_density",
        "electron_density",
        "peak_ionisation_rate",
        "species_density",
        "species_ionisation_energy",
        "species_peak_ionisation_rate",
        "plasma_undershoot",
        "plasma_work_defect",
        "plasma_boundary_level",
        "peak_intensity",
        "on_axis_intensity",
        "beam_radius",
        "fwhm_r",
        "on_axis_spectrum",
        "quadrature_error",
    )
)
_PROFILES: tuple[Literal["density", "pressure", "core_radius"], ...] = (
    "density",
    "pressure",
    "core_radius",
)
_POPULATIONS = frozenset(
    (
        "species_ionisation",
        "species_electron_density",
        "electron_density",
        "peak_ionisation_rate",
        "species_density",
        "species_ionisation_energy",
        "species_peak_ionisation_rate",
        "plasma_undershoot",
        "plasma_work_defect",
        "plasma_boundary_level",
    )
)
_PLASMA_DIAGNOSTICS = frozenset(
    ("plasma_undershoot", "plasma_work_defect", "plasma_boundary_level")
)
_UNIT_FLUX = frozenset(("physical_energy", "analytic_flux", "spectral_flux"))
_AXIS_POPULATIONS = frozenset(ON_AXIS_PLASMA_STATISTICS)


@dataclass(frozen=True, eq=False)
class StatisticTransfer:
    """Describe one packed output in the public catalogue's order and units.

    Attributes
    ----------
    name : str
        Catalogue name, also the Julia statistic symbol.
    shape : tuple of int
        Public shape excluding batch and plane axes; ``()`` denotes a scalar.
    offset, count : int
        One-based packed start and positive element count. Multidimensional values
        are packed in C order, independently of Julia's internal field layout.
    unit : str
        The catalogue's unchanged unit string, including bar for pressure.
    """

    name: str
    shape: tuple[int, ...]
    offset: int
    count: int
    unit: str


@dataclass(frozen=True, eq=False)
class StatisticsGridTransfer:
    """Own the coarse lattice and analytic-signal integration weights.

    Attributes
    ----------
    omega, time : numpy.ndarray
        Coarse angular frequencies in rad/s and time samples in s, shapes
        ``(n_omega,)`` and ``(n_t,)``. Frequency storage retains FFT order.
    parseval : numpy.ndarray
        Analytic-signal energy weights in s, shape ``(n_omega,)``: ``dt/n_t``
        times ``(1,4,...,4,1)`` for a field, and ones for an envelope.
    windows : numpy.ndarray
        Dense unpacked Boolean half-open frequency gates, shape
        ``(n_omega,n_windows)``. Every array is owned and read-only.
    spectral_density : numpy.ndarray or None
        Independent per-angular-frequency weights in seconds squared for spatial
        axis spectra or modal spectral flux. Absent when neither is requested.
    d_omega : float
        Angular-frequency spacing in rad/s for spatial widths; NaN otherwise.
    """

    omega: Float[np.ndarray, " n_omega"]
    time: Float[np.ndarray, " n_t"]
    parseval: Float[np.ndarray, " n_omega"]
    windows: Bool[np.ndarray, "n_omega n_windows"]
    spectral_density: Float[np.ndarray, " n_omega"] | None = None
    d_omega: float = math.nan


@dataclass(frozen=True, eq=False)
class ObservationProfilesTransfer:
    """Keep each requested scalar profile and its own measured fitting record.

    Attributes
    ----------
    density, pressure, core_radius : ProfileLowering or None
        Independent represented values in m^-3, catalogue bar and m, respectively.
        An unrequested profile is absent, even if supplied in the context.
    """

    density: ProfileLowering | None
    pressure: ProfileLowering | None
    core_radius: ProfileLowering | None

    def record(self) -> dict[str, dict[str, str | float | int]]:
        """Return fresh JSON-compatible ``statistics_tables`` measurements.

        Returns
        -------
        dict
            One independent metadata record per requested profile, without any
            originating profile, problem or numerical array.
        """
        return {
            name: value.record()
            for name in _PROFILES
            if (value := getattr(self, name)) is not None
        }


@dataclass(frozen=True, eq=False)
class PopulationMeasureTransfer:
    """Own the physical transverse measure for final ionised inventory.

    Attributes
    ----------
    weights : numpy.ndarray
        Nonnegative reference node weights, shape ``(n_nodes,)``; square metres
        for resolved geometry, or one dimensionless node for mode averaging.
    area : float or TableTransfer
        Positive represented multiplier, in square metres for mode averaging and
        dimensionless one otherwise.
    taper : float or TableTransfer
        Positive dimensionless represented scale. Effective weights are
        ``weights * area / taper**2`` at the observation coordinate.
    """

    weights: Float[np.ndarray, " n_nodes"]
    area: float | TableTransfer
    taper: float | TableTransfer


@dataclass(frozen=True, eq=False)
class PopulationObservationTransfer:
    """Map fresh source populations into the complete material species order.

    Attributes
    ----------
    species_labels : tuple of str
        Complete translated mixture order, including non-ionising species.
    species_densities : tuple of float or TableTransfer
        Independent represented number-density profiles, in m^-3.
    channel_species : tuple of int
        One-based species indices in the source population channel order.
    peak_species : int
        One-based sole-channel selector for the legacy peak-rate statistic on both
        formulations; zero uses the mixture-wide peak rule. Source buffers and fine
        reconstruction inputs remain in the separately lowered source preparation.
    species_potentials : numpy.ndarray
        Complete species order of nonnegative ionisation potentials in J, shape
        ``(n_species,)``; zero for passive species.
    measure : PopulationMeasureTransfer or None
        Independent physical measure, present only for ionised inventory energy.
    """

    species_labels: tuple[str, ...]
    species_densities: tuple[float | TableTransfer, ...]
    channel_species: tuple[int, ...]
    peak_species: int
    species_potentials: Float[np.ndarray, " n_species"]
    measure: PopulationMeasureTransfer | None


@dataclass(frozen=True, eq=False)
class AxisPopulationObservationTransfer:
    """Own an independent fine-grid on-axis population reconstruction.

    Attributes
    ----------
    synthesis : numpy.ndarray
        Real physical axis map, shape ``(n_components,n_modes)`` with one or two
        Cartesian components, including the fine temporal padding factor once.
    species_densities : tuple of float or TableTransfer
        Complete species order of independently represented densities, in m^-3.
    channel_species : tuple of int
        One-based species indices for the source's population channels.
    taper : float or TableTransfer
        Independent dimensionless mode taper evaluated at the observation plane.
    """

    synthesis: Float[np.ndarray, "n_components n_modes"]
    species_densities: tuple[float | TableTransfer, ...]
    channel_species: tuple[int, ...]
    taper: float | TableTransfer


@dataclass(frozen=True, eq=False)
class TemporalPolicyTransfer:
    """Carry the shared monitor's scalar policy without measured run state.

    Attributes
    ----------
    warn, stop, activity, quiet_share : float
        Dimensionless shared warning, stopping, activity and quiet thresholds.
    quiet_checks, cadence : int
        Required quiet readings and busy accepted-step reading cadence.
    quiet_collar : bool
        Whether the user permits an eligible in-stage collar to rest.
    interval : float
        Maximum unchecked quiet distance in m; positive infinity disables it.
    launch_floor_ratio, floor_margin : float
        Dimensionless shared launch-energy floor and guard/collar floor multiplier.
    split : bool
        Whether removal is measured from an accepted split step.
    crossing_share : float
        Dimensionless occupancy share excluded before finding a crossing cap.
    bucket_limits : numpy.ndarray
        Owned read-only positive step limits in m, shape ``(n_buckets,)``; infinity
        denotes zero walk-off. Empty when no translated crossing rule applies.
    """

    warn: float
    stop: float
    activity: float
    quiet_share: float
    quiet_checks: int
    quiet_collar: bool
    cadence: int
    interval: float
    launch_floor_ratio: float
    floor_margin: float
    split: bool
    crossing_share: float
    bucket_limits: Float[np.ndarray, " n_buckets"]


@dataclass(frozen=True, eq=False)
class TemporalObservationTransfer:
    """Own one fresh temporal monitor's numerical policy and reduction weights.

    Attributes
    ----------
    policy : TemporalPolicyTransfer
        The shared prediction and decision policy, without mutable readings.
    regions : numpy.ndarray
        Owned read-only binary64 weights, shape ``(n_t,4)``: low guard, high guard,
        collar share and temporal power-loss rate. The first three are dimensionless;
        the last is in m^-1.
    columns : numpy.ndarray
        Owned nonnegative binary64 reduction weights, shape ``(n_modes,)``; modal
        defaults are one, independently of the statistics' varying power weight.
    buckets : numpy.ndarray or None
        Owned one-based Int64 crossing buckets, shape ``(n_omega,n_modes)``.
        Absent without a translated crossing rule.
    """

    policy: TemporalPolicyTransfer
    regions: Float[np.ndarray, "n_t 4"]
    columns: Float[np.ndarray, " n_modes"]
    buckets: Int[np.ndarray, "n_omega n_modes"] | None


@dataclass(frozen=True, eq=False)
class SpatialWidthTransfer:
    """Own the physical axes and radial measure for spatial fluence widths.

    Attributes
    ----------
    kind : str
        Radial, Cartesian or line geometry.
    x, y : numpy.ndarray
        Physical coordinates in metres; y is empty for a radial grid.
    weights : numpy.ndarray
        Physical integration measure in square metres (metres on a line).
    """

    kind: str
    x: Float[np.ndarray, " n_x"]
    y: Float[np.ndarray, " n_y"]
    weights: Float[np.ndarray, " n_nodes"]


@dataclass(frozen=True, eq=False)
class SpatialObservationTransfer:
    """Own the independent spatial plane measure and coarse reconstruction metadata.

    Attributes
    ----------
    flux : SeparableTransfer
        Protected constant wave-number inputs to the axial admittance.
    material : TableTransfer or None
        Independently represented complex material square, inverse square metres.
    light_speed : float
        The shared speed of light, metres per second.
    spectral_weights : numpy.ndarray
        Transverse Parseval measure, shape ``(n_modes,)``, in the geometry's units.
    on_axis : numpy.ndarray
        Independent coarse axial synthesis, shape ``(n_modes,)``.
    pad_scale : float
        Fine-grid temporal scaling removed once from radial coarse synthesis.
    widths : SpatialWidthTransfer or None
        Physical axes retained only when a spatial width is requested.
    """

    flux: SeparableTransfer
    material: TableTransfer | None
    light_speed: float
    spectral_weights: Float[np.ndarray, " n_modes"]
    on_axis: Complex[np.ndarray, " n_modes"]
    pad_scale: float
    widths: SpatialWidthTransfer | None


@dataclass(frozen=True, eq=False)
class ObservationTransfer:
    """Own the numerical observations selected by one immutable run request.

    Attributes
    ----------
    kind : str
        Scalar ``field`` or ``envelope`` formulation.
    n_modes : int
        Number of scalar modal columns.
    grid : StatisticsGridTransfer
        Coarse lattice, energy weights and requested frequency gates.
    slots : tuple of StatisticTransfer
        Plane statistics in the exact requested order.
    periodic_indices : tuple of int
        One-based slot indices in the requested periodic order.
    width, period : int
        Total packed row width and positive periodic accepted-step cadence.
    power_weight : CoefficientTransfer or None
        Direct real modal power weights; absent for a spatial route.
    spatial : SpatialObservationTransfer or None
        Independent axial-flux preparation for a spatial route; absent otherwise.
    profiles : ObservationProfilesTransfer
        Requested independent scalar profile statistics and their measurements.
    axis_synthesis : numpy.ndarray or None
        Separate coarse on-axis synthesis, shape ``(n_components,n_modes)``, with
        no fine-grid padding factor. Absent unless peak intensity is requested.
    taper : float or TableTransfer
        The transform's own dimensionless scalar taper, applied at observation z.
    intensity_factor : float
        ``epsilon_0*c/2`` in SI, converting the analytic electric-field pair to W/m^2.
    populations : PopulationObservationTransfer or None
        Species mapping for requested population statistics, otherwise absent.
    temporal : TemporalObservationTransfer or None
        Fresh temporal monitor configuration, otherwise absent.
    transverse : TransverseTransfer or None
        Owned scalar spatial edge configuration, otherwise absent.
    split_rate : numpy.ndarray or None
        Coarse temporal power-loss rate in m^-1, shape ``(n_t,)``, under the split
        scheme, including unmonitored runs. Its application remains Julia-owned.
    can_rest : bool
        Whether the shared route/method rule permits this attached monitor to rest
        an in-stage collar. No Python objects or mutable lifecycle state are retained.
    axis_populations : AxisPopulationObservationTransfer or None
        Separate fine-grid axis reconstruction for the four on-axis plasma rows.
    """

    kind: str
    n_modes: int
    grid: StatisticsGridTransfer
    slots: tuple[StatisticTransfer, ...]
    periodic_indices: tuple[int, ...]
    width: int
    period: int
    power_weight: CoefficientTransfer | None
    profiles: ObservationProfilesTransfer
    axis_synthesis: Float[np.ndarray, "n_components n_modes"] | None
    taper: float | TableTransfer
    intensity_factor: float
    populations: PopulationObservationTransfer | None
    temporal: TemporalObservationTransfer | None
    split_rate: Float[np.ndarray, " n_t"] | None
    can_rest: bool
    spatial: SpatialObservationTransfer | None = None
    transverse: TransverseTransfer | None = None
    axis_populations: AxisPopulationObservationTransfer | None = None


def _admit(
    problem: PropagationProblem,
    request: RunRequest,
    grid: TimeGrid | None,
    modes: Sequence[Mode] | None,
    profiles: Mapping[str, ZProfile] | None,
) -> None:
    """Refuse unsupported names and missing context before numerical preparation.

    Parameters
    ----------
    problem, request : PropagationProblem, RunRequest
        Resolved scientific descriptor and this run's observation request.
    grid, modes, profiles : TimeGrid, sequence, mapping or None
        Optional preparation context, never retained.

    Raises
    ------
    BackendError
        Naming an unsupported route, statistic or missing context.
    """
    names = request.observation.stats.names
    spatial = isinstance(problem.route, SpatialRoute)
    catalogue = _SPATIAL_STATISTICS if spatial else _STATISTICS
    unsupported = tuple(name for name in names if name not in catalogue)
    if unsupported:
        raise BackendError(f"julia scalar statistics do not evaluate {unsupported!r}")
    checks = (
        (
            problem.precision.single and problem.initial.noise_model != "none",
            "binary32 input noise",
        ),
        (problem.shape[1] != 1, "vector polarisation"),
        (
            not isinstance(problem.route, (ModalRoute, SpatialRoute)),
            "unsupported routes",
        ),
        (
            isinstance(problem.route, SpatialRoute) and problem.route.kind == "planar",
            "planar routes",
        ),
        (problem.transform.modal_tables is not None, "general modal transforms"),
        (problem.linear.modal_metric is not None, "general modal power metrics"),
    )
    for refused, capability in checks:
        if refused:
            raise BackendError(f"julia observations refuse {capability}")
    if {"fwhm_t", "fwhm_t_outer"}.intersection(names) and grid is None:
        raise BackendError("julia half-maximum widths need the original time grid")
    if (
        not spatial
        and "peak_intensity" in names
        and (modes is None or len(modes) != problem.shape[0])
    ):
        raise BackendError("julia peak_intensity needs the matching mode set")
    if {"energy_window", "peak_power_window"}.intersection(
        names
    ) and request.observation.stats.n_windows == 0:
        raise BackendError("julia window statistics need at least one energy window")
    for name in _PROFILES:
        if name in names and (profiles is None or name not in profiles):
            raise BackendError(f"julia {name} statistic needs its own {name!r} profile")
    _admit_population_statistics(problem, names, modes)
    if _UNIT_FLUX.intersection(names):
        weight = problem.linear.power_weight_constant
        if weight is None or not np.all(weight == 1.0):
            raise BackendError(
                "julia physical_energy, analytic_flux and spectral_flux require "
                "represented constant unit modal power weights"
            )


def _admit_population_statistics(
    problem: PropagationProblem, names: Sequence[str], modes: Sequence[Mode] | None
) -> None:
    """Check population-family context before constructing numerical observations.

    Parameters
    ----------
    problem : PropagationProblem
        Translated route and compiled population channels.
    names : sequence of str
        Requested public statistic names.
    modes : sequence of Mode or None
        Original mode set required only for independent axis reconstruction.

    Raises
    ------
    BackendError
        For a family without its required route, channels or physical axis context.
    """
    if {"electron_density", "peak_ionisation_rate"}.intersection(
        names
    ) and not problem.responses.population_channels:
        raise BackendError(
            "julia electron_density and peak_ionisation_rate need plasma"
        )
    if _PLASMA_DIAGNOSTICS.intersection(names) and (
        problem.grid_kind != "field" or not problem.responses.plasma_channels
    ):
        raise BackendError("julia plasma diagnostics require field plasma channels")
    if _AXIS_POPULATIONS.intersection(names):
        if (
            problem.transform.kind != "modal"
            or not problem.responses.population_channels
        ):
            raise BackendError(
                "julia on-axis plasma statistics require resolved modal plasma"
            )
        if modes is None or len(modes) != problem.shape[0]:
            raise BackendError(
                "julia on-axis plasma statistics need the matching mode set"
            )


def _grid(
    problem: PropagationProblem,
    request: RunRequest,
    grid: TimeGrid | None,
) -> StatisticsGridTransfer:
    """Copy the coarse grid and form the convention's analytic energy weights.

    Parameters
    ----------
    problem, request : PropagationProblem, RunRequest
        Resolved grid data and half-open angular-frequency windows.
    grid : TimeGrid or None
        Original coarse axis, or None for the convention's centred lattice.

    Returns
    -------
    StatisticsGridTransfer
        Owned coarse arrays with unchanged FFT order.

    Raises
    ------
    BackendError
        If a supplied grid does not match the translated coarse lattice.
    """
    data = problem.grids
    if grid is not None and (
        grid.kind != problem.grid_kind
        or grid.n_t != data.n_t
        or grid.dt != data.dt
        or not np.array_equal(grid.omega, data.omega)
    ):
        raise BackendError("julia observation time grid does not match the problem")
    time = (np.arange(data.n_t) - data.n_t // 2) * data.dt if grid is None else grid.t
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0):
        raise BackendError("julia observation time grid must be finite and increasing")
    parseval = np.full(data.omega.size, data.dt / data.n_t, dtype=np.float64)
    if problem.grid_kind == "field":
        parseval[1:-1] *= 4.0
    edges = request.observation.stats.energy_windows
    windows = (data.omega[:, None] >= edges[:, 0]) & (data.omega[:, None] < edges[:, 1])
    return StatisticsGridTransfer(
        owned(data.omega),
        owned(time),
        owned(parseval),
        owned(windows),
        owned(
            fourier.spectral_density(
                np.ones(data.omega.size, dtype=complex), data.dt, kind=problem.grid_kind
            )
        )
        if isinstance(problem.route, SpatialRoute)
        or "spectral_flux" in request.observation.stats.names
        else None,
        float(2 * np.pi / (data.n_to * data.dt_fine))
        if isinstance(problem.route, SpatialRoute)
        else math.nan,
    )


def _slots(
    problem: PropagationProblem,
    request: RunRequest,
) -> tuple[StatisticTransfer, ...]:
    """Pack the selected public shapes without batch or plane axes.

    Parameters
    ----------
    problem, request : PropagationProblem, RunRequest
        Validated scalar modal extents and selected catalogue order.

    Returns
    -------
    tuple of StatisticTransfer
        Contiguous one-based slots, including scalar count one.
    """
    n_modes = problem.shape[0]
    n_windows = request.observation.stats.n_windows
    n_species = problem.responses.n_species
    shapes = {
        "omega0": (n_modes, 1),
        "energy": (n_modes, 1),
        "energy_window": (n_windows, n_modes, 1),
        "peak_power": (n_modes,),
        "peak_power_window": (n_windows, n_modes),
        "species_ionisation": (n_species, problem.transform.n_nodes),
        "species_electron_density": (n_species,),
        "species_density": (n_species,),
        "species_ionisation_energy": (n_species,),
        "species_peak_ionisation_rate": (n_species,),
        "on_axis_species_electron_density": (n_species,),
        "on_axis_species_peak_ionisation_rate": (n_species,),
        "spectral_flux": (problem.shape[-1],),
    }
    if isinstance(problem.route, SpatialRoute):
        shapes.update(
            energy=(), beam_radius=(2,), on_axis_spectrum=(problem.shape[-1],)
        )
    result: list[StatisticTransfer] = []
    offset = 1
    for name in request.observation.stats.names:
        shape = shapes.get(name, ())
        count = math.prod(shape)
        result.append(
            StatisticTransfer(name, shape, offset, count, CATALOGUE[name].units)
        )
        offset += count
    return tuple(result)


def _power_weight(problem: PropagationProblem) -> CoefficientTransfer:
    """Copy the exact direct weight representation independently of source admission.

    Parameters
    ----------
    problem : PropagationProblem
        Scalar modal descriptor with one translated real weight representation.

    Returns
    -------
    CoefficientTransfer
        Constant, direct table, or fixed terms and direct scalar weights.

    Raises
    ------
    BackendError
        If a weight is complex, nonfinite or structurally invalid.
    """
    linear, shape = problem.linear, problem.shape
    constant, expansion, table = (
        linear.power_weight_constant,
        linear.power_weight_expansion,
        linear.power_weight,
    )
    try:
        if constant is not None:
            if (
                np.iscomplexobj(constant)
                or not np.all(np.isfinite(constant))
                or np.any(constant <= 0)
            ):
                raise BackendError(
                    "julia constant power weights must be real positive finite"
                )
            matrix = np.broadcast_to(constant, shape).reshape(-1, shape[-1]).T
            return CoefficientTransfer(owned(matrix), None, None)
        if expansion is not None:
            terms = expansion.terms
            if np.iscomplexobj(terms) or not np.all(np.isfinite(terms)):
                raise BackendError(
                    "julia power-weight expansion terms must be real finite"
                )
            missing = len(shape) - (terms.ndim - 1)
            expanded = terms.reshape(
                (terms.shape[0], *((1,) * missing), *terms.shape[1:])
            )
            broadcast = np.broadcast_to(expanded, (terms.shape[0], *shape))
            matrix = broadcast.reshape(terms.shape[0], -1, shape[-1]).transpose(2, 1, 0)
            table = expansion.weights
            result = CoefficientTransfer(owned(matrix), None, lower_table(table))
        elif table is not None:
            result = CoefficientTransfer(None, lower_table(table, shape=shape), None)
        else:
            raise BackendError(
                "julia observations need a represented modal power weight"
            )
        if np.iscomplexobj(table.coefficients) or not np.all(
            np.isfinite(table.coefficients)
        ):
            raise BackendError("julia power-weight tables must be real finite")
        return result
    except ValueError as exc:
        raise BackendError(
            "julia power-weight value axes do not match the modal state"
        ) from exc


def _profiles(
    problem: PropagationProblem,
    request: RunRequest,
    profiles: Mapping[str, ZProfile] | None,
) -> ObservationProfilesTransfer:
    """Fit only the requested scalar profiles using the shared measured policy.

    Parameters
    ----------
    problem, request : PropagationProblem, RunRequest
        Scientific tables and the current run's selected names and endpoint.
    profiles : mapping or None
        Original profile context, read only during preparation.

    Returns
    -------
    ObservationProfilesTransfer
        Independent numeric values and measured metadata, with no source references.
    """
    names = request.observation.stats.names
    selected: list[ProfileLowering | None] = []
    for name in _PROFILES:
        if name in names:
            assert profiles is not None
            selected.append(
                lower_statistic_profile(
                    problem,
                    name,
                    profiles[name],
                    float(request.observation.save_planes[-1]),
                )
            )
        else:
            selected.append(None)
    return ObservationProfilesTransfer(*selected)


def _populations(
    problem: PropagationProblem, *, inventory: bool = False
) -> PopulationObservationTransfer:
    """Preserve the source's channel order and complete mixture species mapping.

    Parameters
    ----------
    problem : PropagationProblem
        Translated material labels, independent densities and population channels.
    inventory : bool, optional
        Include the independently represented transverse physical measure.

    Returns
    -------
    PopulationObservationTransfer
        Owned number-density tables and one-based channel indices.

    Raises
    ------
    BackendError
        For missing, duplicate or inconsistent material labels.
    """
    responses = problem.responses
    labels = tuple(responses.labels)
    channels = responses.population_channels
    if (
        not labels
        or len(set(labels)) != len(labels)
        or len(responses.species_densities) != len(labels)
    ):
        raise BackendError(
            "julia population statistics need complete unique species labels"
        )
    if any(channel.label not in labels for channel in channels):
        raise BackendError("julia population channel is absent from the species labels")
    mapping = tuple(labels.index(channel.label) + 1 for channel in channels)
    if len(set(mapping)) != len(mapping):
        raise BackendError("julia population channels must map to distinct species")
    potentials = np.zeros(len(labels))
    for species, channel in zip(mapping, channels, strict=True):
        potentials[species - 1] = channel.ionisation_potential
    return PopulationObservationTransfer(
        labels,
        tuple(lower_profile(table) for table in responses.species_densities),
        mapping,
        mapping[0] if len(channels) == 1 else 0,
        owned(potentials),
        _population_measure(problem) if inventory else None,
    )


def _population_measure(problem: PropagationProblem) -> PopulationMeasureTransfer:
    """Lower the shared species-inventory measure without evaluating a population.

    Parameters
    ----------
    problem : PropagationProblem
        Physical geometry or modal transverse quadrature and independent profiles.

    Returns
    -------
    PopulationMeasureTransfer
        Owned weights and represented scale factors.

    Raises
    ------
    BackendError
        If the route lacks its physical transverse measure.
    """
    transform = problem.transform
    if transform.geometry is not None:
        return PopulationMeasureTransfer(
            owned(transform.geometry.spatial_weights), 1.0, 1.0
        )
    if transform.mode_averaged:
        if transform.aeff is None:
            raise BackendError("julia species inventory requires effective area")
        return PopulationMeasureTransfer(
            owned(np.ones(1)), lower_profile(transform.aeff), 1.0
        )
    if transform.node_weights is None:
        raise BackendError("julia species inventory requires physical node weights")
    return PopulationMeasureTransfer(
        owned(transform.node_weights), 1.0, lower_profile(transform.taper_scale)
    )


def _axis(modes: Sequence[Mode]) -> Float[np.ndarray, "n_components n_modes"]:
    """Build the separately normalised coarse on-axis map at the reference position.

    Parameters
    ----------
    modes : sequence of Mode
        Original modes, evaluated once at z=0 m and never retained.

    Returns
    -------
    numpy.ndarray
        Owned real map in V/m per vacuum modal amplitude, without any pad scale.

    Raises
    ------
    BackendError
        If the map is complex, nonfinite or not a scalar transverse mode map.
    """
    rule = on_axis_rule(modes[0].dimlimits(0.0)[0])
    matrix = mode_matrix(modes, rule, 0.0, components=("x", "y")).synthesis
    if (
        np.iscomplexobj(matrix)
        or not np.all(np.isfinite(matrix))
        or matrix.shape
        not in (
            (1, len(modes)),
            (2, len(modes)),
        )
    ):
        raise BackendError(
            "julia peak_intensity needs a finite real scalar on-axis map"
        )
    return owned(matrix)


def _axis_populations(
    problem: PropagationProblem, modes: Sequence[Mode]
) -> AxisPopulationObservationTransfer:
    """Copy the independent fine on-axis map and complete species ordering.

    Parameters
    ----------
    problem : PropagationProblem
        Resolved modal descriptor with compiled population channels.
    modes : sequence of Mode
        Matching modes, used during preparation and never retained.

    Returns
    -------
    AxisPopulationObservationTransfer
        Owned synthesis with the fine temporal scale applied once.
    """
    population = _populations(problem)
    return AxisPopulationObservationTransfer(
        owned(_axis(modes) * problem.grids.pad_scale),
        population.species_densities,
        population.channel_species,
        lower_profile(problem.transform.taper_scale),
    )


def _temporal(
    problem: PropagationProblem,
    monitor: BoundaryMonitor,
    split_rate: Float[np.ndarray, " n_t"] | None,
) -> TemporalObservationTransfer:
    """Copy a fresh monitor configuration and the problem's authoritative crossing rule.

    Parameters
    ----------
    problem : PropagationProblem
        Modal dimensions and translated split-crossing buckets.
    monitor : BoundaryMonitor
        Requested policy and weights; any existing measured state is ignored.
    split_rate : numpy.ndarray or None
        Coarse split power rate in m^-1, present exactly for an active split collar.

    Returns
    -------
    TemporalObservationTransfer
        Owned weights and exact shared prediction constants.

    Raises
    ------
    BackendError
        For unsupported transverse data, inconsistent schemes or invalid weights.
    """
    if monitor.kind != problem.grid_kind or monitor.split != (split_rate is not None):
        raise BackendError(
            "julia temporal monitor formulation or split scheme is inconsistent"
        )
    regions = monitor.weights
    if (
        regions.shape != (4, problem.grids.n_t)
        or np.iscomplexobj(regions)
        or not np.all(np.isfinite(regions))
        or np.any(regions < 0)
    ):
        raise BackendError(
            "julia temporal monitor needs finite nonnegative (4,n_t) weights"
        )
    if split_rate is not None and not np.array_equal(regions[3], split_rate):
        raise BackendError(
            "julia split monitor rate must equal the translated split rate"
        )
    columns = np.ones(problem.shape[0]) if monitor.columns is None else monitor.columns
    if (
        columns.shape != (problem.shape[0],)
        or np.iscomplexobj(columns)
        or not np.all(np.isfinite(columns))
        or np.any(columns < 0)
    ):
        raise BackendError(
            "julia temporal monitor column weights must match its modal columns"
        )
    crossing = problem.split_crossing if split_rate is not None else None
    try:
        policy = replace(monitor, crossing=crossing).watch_policy()
    except ArithmeticError as exc:
        raise BackendError("julia temporal monitor policy overflowed") from exc
    numeric = ("warn", "stop", "activity", "quiet_share", "share")
    if not all(math.isfinite(cast("float", policy[name])) for name in numeric):
        raise BackendError("julia temporal monitor thresholds must be finite")
    if not 0.0 <= cast("float", policy["share"]) < 1.0:
        raise BackendError("julia split crossing share must lie in [0,1)")
    limits = np.asarray(cast("list[float]", policy["bucket_limits"]), dtype=np.float64)
    if not np.all(limits > 0) or math.isnan(monitor.interval) or monitor.interval <= 0:
        raise BackendError(
            "julia temporal monitor intervals and bucket limits must be positive"
        )
    buckets = None
    if crossing is not None:
        if crossing.buckets.shape != (problem.shape[0], problem.shape[-1]) or np.any(
            crossing.buckets >= len(limits)
        ):
            raise BackendError(
                "julia split crossing buckets do not match the modal state"
            )
        buckets = owned(np.asarray(crossing.buckets.T, dtype=np.int64) + 1)
    prepared_policy = TemporalPolicyTransfer(
        cast("float", policy["warn"]),
        cast("float", policy["stop"]),
        cast("float", policy["activity"]),
        cast("float", policy["quiet_share"]),
        cast("int", policy["quiet_checks"]),
        cast("bool", policy["quiet_collar"]),
        cast("int", policy["cadence"]),
        cast("float", policy["interval"]),
        _FLOOR,
        FLOOR_MARGIN,
        monitor.split,
        cast("float", policy["share"]),
        owned(limits),
    )
    # Monitor context does not pass the problem's binary64 seam normalisation.
    # Widen binary32 represented weights before Julia construction, as the shared
    # monitor's binary64 power products do; reject any overflowing conversion.
    try:
        with np.errstate(over="raise", invalid="raise"):
            prepared_regions = owned(np.asarray(regions.T, dtype=np.float64))
            prepared_columns = owned(np.asarray(columns, dtype=np.float64))
    except FloatingPointError as exc:
        raise BackendError("julia temporal monitor weights exceed binary64") from exc
    return TemporalObservationTransfer(
        prepared_policy, prepared_regions, prepared_columns, buckets
    )


def _spatial_observation(
    problem: PropagationProblem, request: RunRequest
) -> SpatialObservationTransfer | None:
    """Copy scalar spatial flux and selected width geometry without modal weights.

    Parameters
    ----------
    problem, request : PropagationProblem, RunRequest
        Admitted numerical descriptor and selected spatial statistics.

    Returns
    -------
    SpatialObservationTransfer or None
        Independent represented flux and geometry, or None for a modal route.
    """
    route, geometry = problem.route, problem.transform.geometry
    if not isinstance(route, SpatialRoute):
        return None
    assert geometry is not None
    widths = None
    if {"beam_radius", "fwhm_r"}.intersection(request.observation.stats.names):
        widths = SpatialWidthTransfer(
            route.kind,
            owned(geometry.axes[0]),
            owned(np.empty(0) if route.kind == "radial" else geometry.axes[1]),
            owned(geometry.spatial_weights),
        )
    return SpatialObservationTransfer(
        lower_separable(route.flux.base),
        None if route.flux.material is None else lower_table(route.flux.material),
        C,
        owned(geometry.spectral_weights),
        owned(geometry.on_axis),
        problem.grids.pad_scale,
        widths,
    )


def lower_observations(
    problem: PropagationProblem,
    request: RunRequest,
    *,
    grid: TimeGrid | None = None,
    modes: Sequence[Mode] | None = None,
    profiles: Mapping[str, ZProfile] | None = None,
    trace_chunk: int = 1,
) -> ObservationTransfer:
    """Prepare one request's scalar statistics and temporal monitor numerically.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Validated scalar modal or spatial descriptor.
    request : solprop.translate.run_request.RunRequest
        This run's immutable statistic selection, periodic cadence and monitor.
    grid : solprop.grid.TimeGrid or None, optional
        Matching original grid, required for half-maximum widths. Otherwise the
        documented centred coarse time lattice is sufficient.
    modes : sequence of Mode or None, optional
        Matching original mode set for on-axis intensity and plasma statistics.
    profiles : mapping of str to ZProfile or None, optional
        Independent density, pressure and core-radius profiles when requested.
        Pressure is already in catalogue bar; no conversion occurs here.
    trace_chunk : int, optional
        Positive execution bound for transverse collar transforms.

    Returns
    -------
    ObservationTransfer
        Frozen request-specific numerical records with owned read-only arrays.
        No source objects, callbacks, Julia handles or mutable monitor state remain.

    Raises
    ------
    solprop.exceptions.BackendError
        Naming an unsupported statistic/route, missing context or inconsistent input.
        Names and context are checked before fitting represented profiles.

    Notes
    -----
    Fourier and amplitude normalisations follow ``docs/dev/conventions.md``.
    The field analytic signal doubles interior positive-frequency bins, giving the
    factor four in Parseval's energy weights. The coarse on-axis map is evaluated
    without the fine synthesis's padding factor. Population source evaluation,
    quadrature alarms and all lifecycle decisions remain Julia-owned.
    """
    _admit(problem, request, grid, modes, profiles)
    slots = _slots(problem, request)
    prepared_grid = _grid(problem, request, grid)
    responses = problem.responses
    split_rate = (
        owned(responses.temporal_absorber)
        if responses.absorber_scheme == "split_step"
        and responses.temporal_absorber is not None
        else None
    )
    if split_rate is not None and (
        split_rate.shape != (problem.grids.n_t,)
        or not np.all(np.isfinite(split_rate))
        or np.any(split_rate < 0)
    ):
        raise BackendError(
            "julia split rate must be a finite nonnegative coarse-time vector"
        )
    monitor = request.boundary.monitor
    temporal = None if monitor is None else _temporal(problem, monitor, split_rate)
    names = request.observation.stats.names
    spatial = _spatial_observation(problem, request)
    geometry = problem.transform.geometry
    resting_geometry = geometry is None or (
        geometry.kind == "radial" and not np.any(geometry.absorber_rate)
    )
    return ObservationTransfer(
        problem.grid_kind,
        problem.shape[0],
        prepared_grid,
        slots,
        tuple(names.index(name) + 1 for name in request.observation.stats.periodic),
        sum(slot.count for slot in slots),
        request.observation.stats.period,
        _power_weight(problem) if spatial is None else None,
        _profiles(problem, request, profiles),
        _axis(modes)
        if spatial is None and "peak_intensity" in names and modes is not None
        else None,
        lower_profile(problem.transform.taper_scale),
        0.5 * EPS0 * C,
        _populations(problem, inventory="species_ionisation_energy" in names)
        if _POPULATIONS.intersection(names)
        else None,
        temporal,
        split_rate,
        monitor is not None
        and request.integration.kind is not MethodKind.LOW_STORAGE
        and responses.temporal_absorber is not None
        and responses.absorber_scheme == "nonlinear"
        and problem.linear.coupled_flow is None
        and resting_geometry,
        spatial,
        None
        if monitor is None or monitor.transverse is None
        else lower_transverse(problem, monitor.transverse, trace_chunk),
        axis_populations=_axis_populations(problem, modes)
        if _AXIS_POPULATIONS.intersection(names) and modes is not None
        else None,
    )
