# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Statistics lowered once for native evaluation (solprop SEP 47 section 6.4).

:func:`lower` turns a set of statistic names into a native
:class:`~solprop_native._core.StatisticsPlan` (``native/include/solprop_native/
statistics_plan.hpp``): which statistics, in which order, and every z-dependent input
as the representation the propagation itself carries -- solprop's piecewise Chebyshev
tables of the plasma density, each channel's preionisation, the species densities, the
taper, the effective area and the modal power weight (:meth:`~solprop.translate.
chebyshev.PiecewiseChebyshev.packed`), read by the native Clenshaw recursion bit for bit
as solprop evaluates them. A general modal basis's statistics read the square root of
its flux metric (:func:`solprop.stats._whitened`), formed here from the metric's own
values by the same ``eigh`` on each interval where the metric is constant.

A profile statistic (``density``, ``pressure``, ``core_radius``) reads its path from
the representation the propagation already carries when one of the problem's scalar
tables reproduces the profile at rounding level; otherwise the profile itself is
lowered at preparation over the whole run (it holds its end value outside its own
span), with its declared breakpoints as mandatory edges, refined until its measured
error on a dense check set is at rounding level
(:data:`PROFILE_TOLERANCE`), or refused by name. Either way the measured error is
recorded (:attr:`StatisticsLowering.tables`) and reaches the run's execution record
under ``statistics_tables``: it is what separates a native profile statistic from the
profile's own value, planes and rows alike. The metric statistics of a basis whose
metric varies inside one of its intervals stay with the host.

The native evaluation finishes every statistic (the widths, the species reduction, the
integration ratios); :meth:`StatisticsLowering.unpack` only names and shapes the packed
values. A native refusal is translated into the exception the reference raises for the
same condition (:func:`translated`).
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError, OutputError, TranslationError
from solprop.modes.quadrature import mode_matrix, on_axis_rule
from solprop.spatial import ABSORBED_CHANNELS
from solprop.stats import FAR_FIELD_STATISTICS
from solprop.translate.chebyshev import PiecewiseChebyshev

from . import _core

if TYPE_CHECKING:
    from solprop.exceptions import SolpropError
    from solprop.grid import TimeGrid
    from solprop.modes.protocol import Mode
    from solprop.profiles import ZProfile
    from solprop.stats import StatsSpec
    from solprop.translate.problem import KernelProblem

__all__ = [
    "ADMITTED_GENERAL_STATISTICS",
    "ADMITTED_SPATIAL_STATISTICS",
    "ADMITTED_STATISTICS",
    "NODE_POPULATIONS",
    "PLANE_ONLY",
    "PROFILE_TOLERANCE",
    "STATISTIC_IDS",
    "StatisticsLowering",
    "lower",
    "translated",
    "uncovered",
]

NODE_POPULATIONS = "_node_populations"
"""The plan's per-node channel populations, ``(n_nodes, n_channels, 2)``: each node's
final fraction and peak rate, which a host on-axis statistic reuses on a Radau rule
(SEP 19). Not a catalogue statistic."""

STATISTIC_IDS: Mapping[str, int] = {
    name: index
    for index, name in enumerate(
        (
            "omega0",
            "energy",
            "energy_window",
            "peak_power",
            "peak_power_window",
            "peak_power_total",
            "fwhm_t",
            "fwhm_t_outer",
            "electron_density",
            "peak_ionisation_rate",
            "peak_intensity",
            "quadrature_error",
            "species_ionisation",
            "species_electron_density",
            "species_peak_ionisation_rate",
            "species_density",
            "species_ionisation_energy",
            "plasma_undershoot",
            "plasma_work_defect",
            "plasma_boundary_level",
            "physical_energy",
            "spectral_flux",
            "analytic_flux",
            "on_axis_intensity",
            "on_axis_spectrum",
            "absorbed_power",
            "beam_spectrum",
            "region_spectrum",
            "aperture_spectrum",
            "aperture_on_axis_spectrum",
            "density",
            "pressure",
            "core_radius",
            NODE_POPULATIONS,
        )
    )
}
"""Each output the native plan evaluates, by its id (``statistics::Stat``)."""

PLANE_ONLY = (
    "zdw",
    "beam_radius",
    "fwhm_r",
    "on_axis_electron_density",
    "on_axis_peak_ionisation_rate",
    "on_axis_species_electron_density",
    "on_axis_species_peak_ionisation_rate",
)
"""The plane-only statistics the plan does not evaluate.

``zdw`` is a root search over the Python mode objects; ``beam_radius`` and ``fwhm_r``
finish a fluence profile's widths on the host
(:func:`solprop.backends.spatial.signed_fluence_widths`,
:func:`~solprop.backends.spatial.plane_fluence_widths`); the four on-axis plasma
statistics scan the axis row in time on the host
(:func:`solprop.backends.populations.axis_row_populations`). None is
periodic-capable (``StatDefinition.between_planes`` false).
"""

_RESPONSE = (
    "electron_density",
    "peak_ionisation_rate",
    "quadrature_error",
    "species_ionisation",
    "species_electron_density",
    "species_peak_ionisation_rate",
    "species_density",
    "species_ionisation_energy",
    "plasma_undershoot",
    "plasma_work_defect",
    "plasma_boundary_level",
    "density",
    "pressure",
    "core_radius",
)

PROFILE_TOLERANCE = 1e-13
"""The largest error a profile statistic's table may carry, relative to the profile's
largest magnitude on the run, measured on the check set."""

_CHECK_POINTS = 4097
"""Uniform positions of a profile table's check set, beside its knots and midpoints."""

_FIT_TOLERANCES = (1e-12, 1e-13, 1e-14)
"""The builder tolerances tried, in turn, for a profile no problem table reproduces."""

_PROFILE_NAMES = ("density", "pressure", "core_radius")

ADMITTED_STATISTICS = (
    "omega0",
    "energy",
    "energy_window",
    "peak_power",
    "peak_power_window",
    "peak_power_total",
    "fwhm_t",
    "fwhm_t_outer",
    "peak_intensity",
    *_RESPONSE,
)
"""The statistics the native plan evaluates on a modal or mode-averaged route."""

ADMITTED_SPATIAL_STATISTICS = (
    "energy",
    "peak_intensity",
    "on_axis_intensity",
    "on_axis_spectrum",
    "absorbed_power",
    "beam_spectrum",
    "region_spectrum",
    "aperture_spectrum",
    "aperture_on_axis_spectrum",
    *_RESPONSE,
)
"""The statistics it evaluates on a spatial route: the medium's axial flux, the far
field and the responses."""

ADMITTED_GENERAL_STATISTICS = (
    "omega0",
    "physical_energy",
    "spectral_flux",
    "analytic_flux",
    "fwhm_t",
    "fwhm_t_outer",
)
"""The statistics it evaluates on a general modal basis: functionals of ``a.H W a``,
where the metric is constant on each of its intervals."""

_ROUTES = {"modal": 0, "spatial": 1, "general": 2}
_ADMITTED = {
    "modal": frozenset(ADMITTED_STATISTICS),
    "spatial": frozenset(ADMITTED_SPATIAL_STATISTICS),
    "general": frozenset(ADMITTED_GENERAL_STATISTICS),
}
_WEIGHTED = frozenset(
    {
        "omega0",
        "energy",
        "energy_window",
        "peak_power",
        "peak_power_window",
        "peak_power_total",
        "fwhm_t",
        "fwhm_t_outer",
        "peak_intensity",
    }
)
_PLASMA = frozenset(
    {
        "electron_density",
        "peak_ionisation_rate",
        "plasma_undershoot",
        "plasma_work_defect",
        "plasma_boundary_level",
    }
)
_SPECIES = frozenset(
    {
        "species_ionisation",
        "species_electron_density",
        "species_peak_ionisation_rate",
        "species_density",
        "species_ionisation_energy",
    }
)

_TRANSLATED: tuple[tuple[str, type[SolpropError]], ...] = (
    ("species statistics require", OutputError),
    ("is outside a statistics table's span", TranslationError),
    ("a metric-weighted statistic needs", OutputError),
)
"""Native refusals and the exceptions the reference raises for the same condition:
the species reduction's domain (``species_statistics``), a position outside a table
(``PiecewiseChebyshev.interval_containing``) and a metric with no root."""


def translated(error: BaseException) -> BaseException:
    """Return the reference's exception for a native statistics refusal.

    Parameters
    ----------
    error : BaseException
        What the native evaluation raised, or a failure's message as an exception.

    Returns
    -------
    BaseException
        The :class:`~solprop.exceptions.SolpropError` the reference raises for the
        same condition, or `error` itself when the refusal is not one of them.
    """
    message = str(error)
    for marker, kind in _TRANSLATED:
        if marker in message:
            return kind(message)
    return error


def _route(problem: KernelProblem) -> str:
    """Return the route a problem's statistics read.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.

    Returns
    -------
    str
        ``"general"``, ``"spatial"`` or ``"modal"``.
    """
    if problem.transform.modal_tables is not None:
        return "general"
    if problem.transform.geometry is not None:
        return "spatial"
    return "modal"


def _metric_varies(problem: KernelProblem) -> bool:
    """Return whether a basis's metric varies inside one of its intervals.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.

    Returns
    -------
    bool
        True when the dense metric has terms beyond degree zero.
    """
    metric = problem.linear.modal_metric
    if metric is None or metric.kind == "identity" or metric.dense is None:
        return False
    return metric.dense.shape[1] > 1


def uncovered(problem: KernelProblem, names: Sequence[str]) -> tuple[str, ...]:
    """Return the names the native plan does not evaluate on this problem's route.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    names : sequence of str
        Requested statistics.

    Returns
    -------
    tuple of str
        The names left to the host, in request order: the plane-only ones and, on a
        basis whose metric varies inside an interval, its metric statistics.
    """
    route = _route(problem)
    admitted = _ADMITTED[route]
    if route == "general" and _metric_varies(problem):
        admitted = frozenset()
    return tuple(name for name in names if name not in admitted)


@dataclasses.dataclass(frozen=True)
class StatisticsLowering:
    """A statistics evaluation lowered for the native plan.

    Attributes
    ----------
    names : tuple of str
        The outputs, in packed order.
    plan : solprop_native._core.StatisticsPlan or None
        The native plan; ``None`` for no names.
    shapes : dict
        Each output's shape, without a batch axis.
    values : dict
        The plan's lowered inputs, from which :meth:`subset` forms another plan.
    tables : dict
        Each profile statistic's table: its source (the problem table it reads, or
        ``"lowered profile"``), the largest absolute and relative errors measured
        against the profile and the number of check positions; the record of
        ``statistics_tables``.
    """

    names: tuple[str, ...]
    plan: _core.StatisticsPlan | None
    shapes: dict[str, tuple[int, ...]]
    values: dict[str, object]
    tables: dict[str, dict[str, object]] = dataclasses.field(default_factory=dict)

    def subset(self, names: Sequence[str]) -> StatisticsLowering:
        """Return the plan of some of the names, from the same lowered inputs.

        Parameters
        ----------
        names : sequence of str
            Outputs of this lowering.

        Returns
        -------
        StatisticsLowering
            Their plan, reading the same tables.
        """
        names = tuple(names)
        values = dict(self.values, outputs=[STATISTIC_IDS[name] for name in names])
        return StatisticsLowering(
            names,
            _core.StatisticsPlan(values) if names else None,
            {name: self.shapes[name] for name in names},
            values,
            {name: record for name, record in self.tables.items() if name in names},
        )

    def unpack(self, packed: np.ndarray) -> dict[str, np.ndarray]:
        """Name the packed values and give each its catalogue shape.

        Parameters
        ----------
        packed : numpy.ndarray
            The native evaluation, in :attr:`names` order.

        Returns
        -------
        dict
            The outputs by name, without a batch axis; arithmetic-free.

        Raises
        ------
        solprop.exceptions.BackendError
            If the evaluation's size is not what the names take.
        """
        values = np.asarray(packed, dtype=np.float64).reshape(-1)
        out: dict[str, np.ndarray] = {}
        offset = 0
        for name in self.names:
            shape = self.shapes[name]
            size = math.prod(shape)
            out[name] = values[offset : offset + size].reshape(shape)
            offset += size
        if offset != values.size:
            msg = f"native statistics returned {values.size} values; consumed {offset}"
            raise BackendError(msg)
        return out


def _packed(
    edges: np.ndarray, lower: float, upper: float, coefficients: np.ndarray, dtype: type
) -> dict[str, object]:
    """Return a :meth:`PiecewiseChebyshev.packed` table as the native plan reads it.

    Parameters
    ----------
    edges : numpy.ndarray
        The breakpoints, m.
    lower, upper : float
        The span widened by its slack, m.
    coefficients : numpy.ndarray
        ``(n_intervals, n_terms, n_values)``.
    dtype : type
        The slot's type, float64 or complex128.

    Returns
    -------
    dict
        The plan's table.
    """
    return {
        "edges": np.ascontiguousarray(edges, dtype=np.float64),
        "coefficients": np.ascontiguousarray(coefficients, dtype=dtype).reshape(-1),
        "n_terms": int(coefficients.shape[1]),
        "n_values": int(coefficients.shape[2]),
        "lower": float(lower),
        "upper": float(upper),
    }


def _table(table: PiecewiseChebyshev | None) -> dict[str, object] | None:
    """Return a problem's real table as the native plan reads it.

    Parameters
    ----------
    table : solprop.translate.chebyshev.PiecewiseChebyshev or None
        The table.

    Returns
    -------
    dict or None
        Its :meth:`~solprop.translate.chebyshev.PiecewiseChebyshev.packed` form.

    Raises
    ------
    solprop.exceptions.BackendError
        If it is complex.
    """
    if table is None:
        return None
    edges, lower, upper, coefficients = table.packed()
    if np.iscomplexobj(coefficients):
        msg = "a real statistics input is a complex table"
        raise BackendError(msg)
    return _packed(edges, lower, upper, coefficients, np.float64)


def _root(metric: np.ndarray) -> np.ndarray:
    """Return the metric's square root as :func:`solprop.stats._whitened` forms it.

    Parameters
    ----------
    metric : numpy.ndarray
        ``W``, ``(n_modes, n_modes, n_omega)``.

    Returns
    -------
    numpy.ndarray
        ``W^(1/2)``, ``(n_omega, n_modes, n_modes)``.

    Raises
    ------
    solprop.exceptions.OutputError
        If the metric is not positive definite at some frequency, as the reference.
    """
    values, vectors = np.linalg.eigh(np.moveaxis(metric, (0, 1), (-2, -1)))
    if np.any(values <= 0.0):
        msg = (
            "a metric-weighted statistic needs a positive-definite flux metric; the "
            f"smallest eigenvalue is {float(np.min(values)):.6g}"
        )
        raise OutputError(msg)
    return (vectors * np.sqrt(values)[..., None, :]) @ np.conj(
        np.swapaxes(vectors, -1, -2)
    )


def _root_table(problem: KernelProblem) -> dict[str, object] | None:
    """Lower a general modal basis's metric root, or ``None`` for the identity.

    The metric is constant on each interval (:func:`uncovered` leaves any other to
    the host), so its root there is the root of the very values the reference reads
    at every position of the interval: exact, as a table of degree zero over the
    reference's own breakpoints and slack.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The general modal problem.

    Returns
    -------
    dict or None
        The table as the native plan reads it.

    Raises
    ------
    solprop.exceptions.BackendError
        For a metric that is not dense and constant on each interval.
    """
    from solprop.translate.coupled import metric_at

    metric = problem.linear.modal_metric
    if metric is None or metric.kind == "identity":
        return None
    block = metric.dense
    if block is None or block.shape[1] != 1:
        msg = "the native plan reads a dense metric constant on each interval"
        raise BackendError(msg)
    # The table `metric_at` evaluates, for its breakpoints and slack.
    edges, lower, upper, _ = PiecewiseChebyshev(
        block,
        metric.z_edges,
        np.zeros(block.shape[0], dtype=np.int64),
        np.empty(0),
    ).packed()
    roots = np.stack(
        [
            _root(np.asarray(metric_at(metric, float(0.5 * (a + b)))))
            for a, b in itertools.pairwise(edges)
        ]
    )
    return _packed(
        edges, lower, upper, roots.reshape(roots.shape[0], 1, -1), np.complex128
    )


def _measured(
    table: PiecewiseChebyshev, profile: ZProfile, check: np.ndarray
) -> tuple[float, float]:
    """Return a table's largest error against a profile on a check set.

    Parameters
    ----------
    table : solprop.translate.chebyshev.PiecewiseChebyshev
        A scalar table.
    profile : solprop.profiles.ZProfile
        The path it stands for.
    check : numpy.ndarray
        Positions, m.

    Returns
    -------
    tuple of float
        The largest absolute error, in the profile's units, and the largest
        magnitude of the profile on the check set.
    """
    exact = np.asarray(profile(check), dtype=np.float64)
    tabulated = np.asarray(table.at_positions(check), dtype=np.float64).reshape(-1)
    scale = float(np.max(np.abs(exact), initial=0.0))
    return float(np.max(np.abs(tabulated - exact), initial=0.0)), scale


def _check_set(
    profile: ZProfile, start: float, end: float, edges: np.ndarray
) -> np.ndarray:
    """Return the positions a profile table is checked on.

    A uniform grid over the run, every declared knot of the profile inside it, and
    every edge of the table and every interval's midpoint.

    Parameters
    ----------
    profile : solprop.profiles.ZProfile
        The path.
    start, end : float
        The run's span, m.
    edges : numpy.ndarray
        The table's breakpoints, m.

    Returns
    -------
    numpy.ndarray
        The positions, increasing.
    """
    knots = np.asarray(profile.breakpoints, dtype=np.float64)
    inside = edges[(edges >= start) & (edges <= end)]
    middles = 0.5 * (edges[1:] + edges[:-1])
    return np.unique(
        np.concatenate(
            (
                np.linspace(start, end, _CHECK_POINTS),
                knots[(knots >= start) & (knots <= end)],
                inside,
                middles[(middles >= start) & (middles <= end)],
            )
        )
    )


def _covers(table: PiecewiseChebyshev, start: float, end: float) -> bool:
    """Return whether a table may be evaluated over the whole run."""
    _, lower, upper, coefficients = table.packed()
    return (
        coefficients.shape[2] == 1
        and not np.iscomplexobj(coefficients)
        and lower <= start
        and end <= upper
    )


def _profile_table(
    problem: KernelProblem, name: str, profile: ZProfile
) -> tuple[PiecewiseChebyshev, dict[str, object]]:
    """Return a profile statistic's table and the record of its measured error.

    The propagation's own table of the path when one of the problem's scalar tables
    reproduces the profile at rounding level over the run; otherwise the profile
    lowered by the propagation's own tabulation (:func:`solprop.translate.linop.
    profile_table`), at its declared breakpoints, with the builder's tolerance
    tightened until the measured error is at rounding level.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    name : str
        The statistic.
    profile : solprop.profiles.ZProfile
        The path it reads.

    Returns
    -------
    tuple
        The table, and its record for ``statistics_tables``.

    Raises
    ------
    solprop.exceptions.BackendError
        Naming the statistic when no table reaches :data:`PROFILE_TOLERANCE`.
    """
    from solprop.translate.linop import profile_table
    from solprop.translate.problem import _walk_tables

    start = float(problem.initial.z)
    end = float(problem.stepper.z_end)
    if not end > start:
        end = start + max(abs(start), 1.0) * 1e-12

    def record(source: str, error: float, scale: float, count: int) -> dict:
        return {
            "source": source,
            "max_error": error,
            "relative_error": error / scale if scale > 0.0 else 0.0,
            "check_points": count,
        }

    def acceptable(error: float, scale: float) -> bool:
        return error <= PROFILE_TOLERANCE * scale

    for source, table in _walk_tables(problem):  # pyright: ignore[reportPrivateUsage]
        if not _covers(table, start, end):
            continue
        check = _check_set(profile, start, end, np.asarray(table.breakpoints))
        error, scale = _measured(table, profile, check)
        if acceptable(error, scale):
            return table, record(source, error, scale, check.size)
    best = math.inf
    for tolerance in _FIT_TOLERANCES:
        table = profile_table(profile, start, end, tol=tolerance)
        check = _check_set(profile, start, end, np.asarray(table.breakpoints))
        error, scale = _measured(table, profile, check)
        if acceptable(error, scale):
            return table, record("lowered profile", error, scale, check.size)
        best = min(best, error / scale if scale > 0.0 else math.inf)
    msg = (
        f"the {name} statistic: its profile cannot be tabulated to rounding level "
        f"over the run (best measured relative error {best:.3g}, above "
        f"{PROFILE_TOLERANCE:g})"
    )
    raise BackendError(msg)


def _shape(problem: KernelProblem, name: str, n_windows: int, plan: dict) -> tuple:
    """Return an output's shape on this problem, without a batch axis.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    name : str
        The output.
    n_windows : int
        Its spec's energy windows.
    plan : dict
        The plan's extents.

    Returns
    -------
    tuple of int
        The shape.
    """
    modes, pol = problem.shape[1], problem.shape[2]
    n_omega = problem.shape[-1]
    apertures = np.asarray(plan["apertures"]).shape[0]
    shapes = {
        "omega0": (modes, pol),
        "energy": () if plan["route"] == _ROUTES["spatial"] else (modes, pol),
        "energy_window": (n_windows, modes, pol),
        "peak_power": (modes,),
        "peak_power_window": (n_windows, modes),
        "species_ionisation": (plan["n_species"], plan["n_nodes"]),
        "species_electron_density": (plan["n_species"],),
        "species_peak_ionisation_rate": (plan["n_species"],),
        "species_density": (plan["n_species"],),
        "species_ionisation_energy": (plan["n_species"],),
        "spectral_flux": (n_omega,),
        "on_axis_spectrum": (n_omega,),
        "beam_spectrum": (n_omega,),
        "absorbed_power": (plan["n_absorbed"],),
        "region_spectrum": (np.asarray(plan["regions"]).shape[0], n_omega),
        "aperture_spectrum": (apertures, n_omega),
        "aperture_on_axis_spectrum": (apertures, n_omega),
        NODE_POPULATIONS: (plan["n_nodes"], plan["n_channels"], 2),
    }
    return shapes.get(name, ())


def lower(
    problem: KernelProblem,
    spec: StatsSpec,
    names: Sequence[str],
    *,
    grid: TimeGrid | None,
    modes: Sequence[Mode] | None = None,
    profiles: Mapping[str, ZProfile] | None = None,
    apertures: tuple[np.ndarray, np.ndarray, int] | None = None,
    on_axis: np.ndarray | None = None,
) -> StatisticsLowering:
    """Lower statistics for native evaluation, once.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    spec : solprop.stats.StatsSpec
        The windows the names read.
    names : sequence of str
        The outputs, each one the plan evaluates on this route (:func:`uncovered`),
        and :data:`NODE_POPULATIONS`.
    grid : solprop.grid.TimeGrid or None
        The time grid; the widths and the general spectral flux read it.
    modes : sequence of Mode or None, optional
        The mode set; ``peak_intensity`` on a modal route reads its axis synthesis.
    profiles : mapping or None, optional
        The z profiles the profile statistics read (:func:`_profile_table`).
    apertures : tuple or None, optional
        A spatial route's far-field tables packed for the native pass (the adapter's
        ``_aperture_payload``): the apertures, the regions and the exponent.
    on_axis : numpy.ndarray or None, optional
        The mode set's synthesis on the axis at the reference position, when the
        caller already built it (``StepDriver._on_axis``); built here otherwise.

    Returns
    -------
    StatisticsLowering
        The plan and the shapes.

    Raises
    ------
    solprop.exceptions.BackendError
        Naming a statistic the plan does not evaluate here, or an input it lacks.
    """
    names = tuple(names)
    left = uncovered(problem, [name for name in names if name != NODE_POPULATIONS])
    if left:
        msg = f"the native statistics plan does not evaluate {left!r} on this route"
        raise BackendError(msg)
    asked = set(names)
    route = _route(problem)
    transform, responses, linear = problem.transform, problem.responses, problem.linear
    channels = responses.population_channels
    plan: dict[str, object] = {
        "route": _ROUTES[route],
        "outputs": [STATISTIC_IDS[name] for name in names],
        "n_modes": int(problem.shape[1]),
        "n_pol": int(problem.shape[2]),
        "n_t": int(problem.grids.n_t),
        "n_omega": int(problem.shape[-1]),
        "n_nodes": int(transform.n_nodes),
        "n_channels": len(channels),
        "n_species": int(responses.n_species),
        "windows": np.ascontiguousarray(spec.energy_windows, dtype=np.float64).reshape(
            -1
        ),
        "intensity_factor": 0.5 * EPS0 * C,
        "t": np.empty(0),
        "weight_kind": 0,
        "weight": np.empty(0),
        "weight_table": None,
        "on_axis": np.empty(0),
        "taper": _table(transform.taper_scale),
        "plasma_density": None,
        "plasma_preionisation": None,
        "channel_preionisation": [],
        "envelope_plasma": responses.envelope_plasma is not None,
        "species_density": [_table(table) for table in responses.species_densities],
        "channel_species": [responses.labels.index(ch.label) for ch in channels],
        "potentials": np.asarray(
            [float(ch.ionisation_potential) for ch in channels], dtype=np.float64
        ),
        "transverse": 0,
        "node_weights": np.empty(0),
        "aeff": None,
        "far_field_axis": False,
        "apertures": np.empty((0, 0)),
        "exponent": 2,
        "regions": np.empty((0, 4)),
        "n_absorbed": len(ABSORBED_CHANNELS),
        "root": None,
        "spectral_prefactor": 0.0,
        "profiles": [None, None, None],
    }
    tables: dict[str, dict[str, object]] = {}
    for index, name in enumerate(_PROFILE_NAMES):
        if name not in asked:
            continue
        if profiles is None or name not in profiles:
            msg = f"the {name} statistic needs the {name!r} profile"
            raise BackendError(msg)
        table, tables[name] = _profile_table(problem, name, profiles[name])
        slots = plan["profiles"]
        assert isinstance(slots, list)
        slots[index] = _table(table)
    if asked & {"fwhm_t", "fwhm_t_outer"}:
        if grid is None:
            msg = "the half-maximum widths need the time grid at construction"
            raise BackendError(msg)
        plan["t"] = np.ascontiguousarray(grid.t, dtype=np.float64)
    if route == "modal" and asked & _WEIGHTED:
        if linear.power_weight_constant is not None:
            plan["weight_kind"] = 1
            plan["weight"] = np.ascontiguousarray(
                linear.power_weight_constant, dtype=np.float64
            ).reshape(-1)
        elif linear.power_weight_expansion is not None:
            expansion = linear.power_weight_expansion
            terms = np.asarray(expansion.terms)
            if np.iscomplexobj(terms):
                msg = "the modal power weight's expansion is complex"
                raise BackendError(msg)
            plan["weight_kind"] = 2
            plan["weight"] = np.ascontiguousarray(terms, dtype=np.float64).reshape(-1)
            plan["weight_table"] = _table(expansion.weights)
        elif linear.power_weight is not None:
            plan["weight_kind"] = 3
            plan["weight_table"] = _table(linear.power_weight)
    if route == "modal" and "peak_intensity" in asked:
        if modes is None:
            msg = "the peak on-axis intensity needs the mode set at construction"
            raise BackendError(msg)
        if on_axis is None:
            rule = on_axis_rule(modes[0].dimlimits(0.0)[0])
            on_axis = mode_matrix(modes, rule, 0.0, components=("x", "y")).synthesis
        plan["on_axis"] = np.ascontiguousarray(on_axis, dtype=np.float64).reshape(-1)
    if asked & (_PLASMA | _SPECIES | {NODE_POPULATIONS}):
        if len(channels) == 1:
            plan["plasma_density"] = _table(channels[0].density)
            plan["plasma_preionisation"] = _table(channels[0].preionisation)
        plan["channel_preionisation"] = [_table(ch.preionisation) for ch in channels]
    if asked & _SPECIES or (
        asked & {"electron_density", "peak_ionisation_rate"}
        and (len(channels) > 1 or responses.envelope_plasma is not None)
    ):
        if transform.geometry is not None:
            plan["transverse"] = 1
            plan["node_weights"] = np.ascontiguousarray(
                transform.geometry.spatial_weights, dtype=np.float64
            )
        elif transform.mode_averaged:
            plan["transverse"] = 2
            plan["aeff"] = _table(transform.aeff)
        elif transform.node_weights is not None:
            plan["transverse"] = 3
            plan["node_weights"] = np.ascontiguousarray(
                transform.node_weights, dtype=np.float64
            )
    if route == "spatial":
        geometry = transform.geometry
        assert geometry is not None
        plan["far_field_axis"] = geometry.kind == "free"
        if asked & set(FAR_FIELD_STATISTICS) or (
            "on_axis_spectrum" in asked and geometry.kind == "free"
        ):
            if apertures is None:
                msg = "the far-field statistics need the tables formed at construction"
                raise BackendError(msg)
            plan["apertures"], plan["regions"], plan["exponent"] = apertures
    if route == "general":
        plan["root"] = _root_table(problem)
        if "spectral_flux" in asked:
            if grid is None:
                msg = "the spectral flux needs the time grid at construction"
                raise BackendError(msg)
            dt = float(grid.dt)
            plan["spectral_prefactor"] = (
                dt**2 / (2.0 * np.pi)
                if grid.kind == "envelope"
                else 2.0 * dt**2 / np.pi
            )
    for key in ("apertures", "regions"):
        plan[key] = np.ascontiguousarray(plan[key], dtype=np.float64)
    shapes = {name: _shape(problem, name, spec.n_windows, plan) for name in names}
    native = _core.StatisticsPlan(plan) if names else None
    return StatisticsLowering(names, native, shapes, plan, tables)
