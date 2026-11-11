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
lowered at preparation over the propagation's span, the whole run and the tables'
margin past its end (it holds its end value outside its own span), with its declared
breakpoints as mandatory edges, refined until its measured error on a dense check set
is at rounding level (``1e-13`` of its scale, or the closest table within
:data:`PROFILE_TOLERANCE` when the platform's libm leaves none that close), or refused
by name. Either way the
measured error is recorded (:attr:`StatisticsLowering.tables`) and reaches the run's
execution record under ``statistics_tables``: it is what separates a native profile
statistic from the profile's own value, planes and rows alike. ``zdw`` is read the
same way from a table of each mode's root along z (:func:`_zdw_tables`). The metric
statistics of a basis whose metric varies inside one of its intervals are refused by
name (:func:`metric_refusal`).

The native evaluation finishes every statistic (the half-maximum widths, a spatial
route's fluence widths, the species reduction, the integration ratios, the on-axis
plasma statistics' scan of the axis row);
:meth:`StatisticsLowering.unpack` only names and shapes the packed values. A native
refusal is translated into the exception the reference raises for the same condition
(:func:`translated`).
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import weakref
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
from solprop.backends.populations import axis_node, axis_refusal
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError, GridError, OutputError, TranslationError
from solprop.modes.quadrature import mode_matrix, on_axis_rule
from solprop.spatial import (
    ABSORBED_CHANNELS,
    _ring_angles,  # pyright: ignore[reportPrivateUsage] - the host widths' own rule
)
from solprop.stats import (
    FAR_FIELD_STATISTICS,
    METRIC_WEIGHTED_STATISTICS,
    ON_AXIS_PLASMA_STATISTICS,
    inband_wavelengths,
)
from solprop.translate.chebyshev import PiecewiseChebyshev

from . import _core
from ._helpers import NATIVE

if TYPE_CHECKING:
    from solprop.exceptions import SolpropError
    from solprop.grid import TimeGrid
    from solprop.modes.protocol import Mode
    from solprop.profiles import ZProfile
    from solprop.stats import StatsSpec
    from solprop.translate.problem import PropagationProblem

__all__ = [
    "ADMITTED_GENERAL_STATISTICS",
    "ADMITTED_SPATIAL_STATISTICS",
    "ADMITTED_STATISTICS",
    "NODE_POPULATIONS",
    "PROFILE_TOLERANCE",
    "STATISTIC_IDS",
    "ZDW_TOLERANCE",
    "StatisticsLowering",
    "lower",
    "metric_refusal",
    "translated",
    "uncovered",
]

NODE_POPULATIONS = "_node_populations"
"""The plan's per-node channel populations, ``(n_nodes, n_channels, 2)``: each node's
final fraction and peak rate, which the transitional session's host on-axis statistics
reuse on a Radau rule (SEP 19). Not a catalogue statistic."""

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
            "beam_radius",
            "fwhm_r",
            "zdw",
            *ON_AXIS_PLASMA_STATISTICS,
        )
    )
}
"""Each output the native plan evaluates, by its id (``statistics::Stat``)."""

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

PROFILE_TOLERANCE = 1e-12
"""The largest error a profile statistic's table may carry, relative to the profile's
largest magnitude on the run, measured on the check set: a few thousand units of
roundoff of that scale. The measurement compares against the profile evaluated through
the platform's libm, whose last bits differ between platforms; a tolerance of ``1e-13``
held on one and refused the same capillary's density at ``1.14e-13`` on another."""

_CHECK_POINTS = 4097
"""Uniform positions of a profile table's check set, beside its knots and midpoints."""

_FIT_TOLERANCES = (1e-12, 1e-13, 1e-14)
"""The builder tolerances tried, in turn, for a profile no problem table reproduces."""

_FIT_TARGET = 1e-13
"""The relative error at which a lowered table is taken at once; failing it on every
builder tolerance, the closest table is taken if within :data:`PROFILE_TOLERANCE`."""

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
    "zdw",
    *_RESPONSE,
    *ON_AXIS_PLASMA_STATISTICS,
)
"""The statistics the native plan evaluates on a modal or mode-averaged route.

The four on-axis plasma statistics (SEP 19) are a resolved modal route's: the
mode-averaged route refuses them by name
(:func:`solprop.backends.populations.axis_refusal`). The plan scans the axis row
natively, the node scan's own pass on the row the mode profiles synthesise on the axis,
or reads a Radau rule's axis node when the evaluation scans the nodes anyway."""

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
    "beam_radius",
    "fwhm_r",
    *_RESPONSE,
)
"""The statistics it evaluates on a spatial route: the medium's axial flux, the fluence
widths, the far field and the responses."""

ADMITTED_GENERAL_STATISTICS = (
    "omega0",
    "physical_energy",
    "spectral_flux",
    "analytic_flux",
    "fwhm_t",
    "fwhm_t_outer",
    "zdw",
)
"""The statistics it evaluates on a general modal basis: functionals of ``a.H W a``,
where the metric is constant on each of its intervals, and the channels'
zero-dispersion wavelength."""

_ROUTES = {"modal": 0, "spatial": 1, "general": 2}
_ADMITTED: dict[str, frozenset[str]] = {
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
    ("need a nonnegative fluence profile", BackendError),
    ("fluence widths require", GridError),
    ("fluence axes must be uniformly increasing", GridError),
    ("fluence moments are not representable", GridError),
    ("on-axis statistics require", BackendError),
)
"""Native refusals and the exceptions the reference raises for the same condition:
the species reduction's domain (``species_statistics``), a position outside a table
(``PiecewiseChebyshev.interval_containing``), a metric with no root, and the fluence
widths' refusals (:func:`solprop.backends.spatial.signed_fluence_widths`,
:func:`solprop.spatial.radial_fluence_widths`, ``cartesian_fluence_widths``) and the
axis populations' domain (:func:`solprop.backends.populations.axis_statistics`)."""


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


def _route(problem: PropagationProblem) -> str:
    """Return the route a problem's statistics read.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
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


def _metric_varies(problem: PropagationProblem) -> bool:
    """Return whether a basis's metric varies inside one of its intervals.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
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


def uncovered(problem: PropagationProblem, names: Sequence[str]) -> tuple[str, ...]:
    """Return the names the native plan does not evaluate on this problem's route.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    names : sequence of str
        Requested statistics.

    Returns
    -------
    tuple of str
        The names the plan leaves, in request order: a statistic of another route
        and, on a basis whose metric varies inside an interval, its metric
        statistics (:func:`metric_refusal`).
    """
    route = _route(problem)
    admitted = _ADMITTED[route]
    if route == "general" and _metric_varies(problem):
        admitted = admitted - METRIC_WEIGHTED_STATISTICS
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
    tables: dict[str, dict[str, object]] = dataclasses.field(
        default_factory=dict[str, dict[str, object]]
    )

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


def _root_table(problem: PropagationProblem) -> dict[str, object] | None:
    """Lower a general modal basis's metric root, or ``None`` for the identity.

    The metric is constant on each interval (:func:`uncovered` leaves any other to
    the host), so its root there is the root of the very values the reference reads
    at every position of the interval: exact, as a table of degree zero over the
    reference's own breakpoints and slack.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
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
    # The compiled Clenshaw pass: the same operations in the same order as the plain
    # one (solprop.translate.chebyshev), thousands of positions at C speed.
    tabulated = np.asarray(
        table.at_positions(check, helpers=NATIVE), dtype=np.float64
    ).reshape(-1)
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


_PROFILE_TABLES: weakref.WeakKeyDictionary[
    PropagationProblem,
    dict[tuple[str, float], tuple[ZProfile, PiecewiseChebyshev, dict[str, object]]],
] = weakref.WeakKeyDictionary()
"""Each problem's profile tables and their records, by statistic and last plane, while
it lives.

A table and its measured error depend only on the problem, the profile and the run's
last plane, so a run repeated on one problem (a runner, ``with_input``, a scan's reused
translation) lowers them once.
"""


def _profile_table(
    problem: PropagationProblem, name: str, profile: ZProfile, z_end: float
) -> tuple[PiecewiseChebyshev, dict[str, object]]:
    """Return a profile statistic's table and its record, lowered once per problem.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    name : str
        The statistic.
    profile : solprop.profiles.ZProfile
        The path it reads.
    z_end : float
        The run's last plane, m (:attr:`solprop.translate.problem.StepperSettings.
        z_end`).

    Returns
    -------
    tuple
        The table, and its record for ``statistics_tables`` (a copy).
    """
    held = _PROFILE_TABLES.setdefault(problem, {})
    cached = held.get((name, z_end))
    if cached is None or cached[0] is not profile:
        table, record = _lowered_profile_table(problem, name, profile, z_end)
        cached = held[name, z_end] = (profile, table, record)
    return cached[1], dict(cached[2])


def _lowered_profile_table(
    problem: PropagationProblem, name: str, profile: ZProfile, z_end: float
) -> tuple[PiecewiseChebyshev, dict[str, object]]:
    """Return a profile statistic's table and the record of its measured error.

    The propagation's own table of the path when one of the problem's scalar tables
    reproduces the profile at rounding level over the run; otherwise the profile
    lowered by the propagation's own tabulation (:func:`solprop.translate.linop.
    profile_table`), at its declared breakpoints and over the propagation's span (the
    run and the tables' margin beyond its end, :data:`solprop.translate.linop.
    DEFAULT_MARGIN`), with the builder's tolerance tightened until the measured error
    is at rounding level. The margin keeps the run's end inside the table, as it is
    in the propagation's: a table that ends there reads its last interval at a sample
    inset one ulp, and a path that leaves a profile's end steeply (a pressure
    gradient's square root near its floor) moves by more than rounding level of its
    scale over that ulp.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    name : str
        The statistic.
    profile : solprop.profiles.ZProfile
        The path it reads.
    z_end : float
        The run's last plane, m.

    Returns
    -------
    tuple
        The table, and its record for ``statistics_tables``.

    Raises
    ------
    solprop.exceptions.BackendError
        Naming the statistic when no table reaches :data:`PROFILE_TOLERANCE`.
    """
    from solprop.translate.linop import DEFAULT_MARGIN, profile_table
    from solprop.translate.problem import (
        _walk_tables,  # pyright: ignore[reportPrivateUsage]
    )

    start = float(problem.initial.z)
    end = float(z_end)
    if not end > start:
        end = start + max(abs(start), 1.0) * 1e-12
    # The propagation's span: from the fibre's entrance to the margin past its end.
    lower = min(start, 0.0)
    upper = max(end, end + abs(end) * DEFAULT_MARGIN)

    def record(
        source: str, error: float, scale: float, count: int
    ) -> dict[str, object]:
        return {
            "source": source,
            "max_error": error,
            "relative_error": error / scale if scale > 0.0 else 0.0,
            "check_points": count,
        }

    def acceptable(error: float, scale: float) -> bool:
        return error <= _FIT_TARGET * scale

    for source, table in _walk_tables(problem):  # pyright: ignore[reportPrivateUsage]
        if not _covers(table, start, end):
            continue
        check = _check_set(profile, start, end, np.asarray(table.breakpoints))
        error, scale = _measured(table, profile, check)
        if acceptable(error, scale):
            return table, record(source, error, scale, check.size)
    best = math.inf
    closest: tuple[PiecewiseChebyshev, dict[str, object]] | None = None
    for tolerance in _FIT_TOLERANCES:
        table = profile_table(profile, lower, upper, tol=tolerance)
        check = _check_set(profile, start, end, np.asarray(table.breakpoints))
        error, scale = _measured(table, profile, check)
        relative = error / scale if scale > 0.0 else math.inf
        if error <= _FIT_TARGET * scale:
            return table, record("lowered profile", error, scale, check.size)
        if relative < best:
            best = relative
            closest = (table, record("lowered profile", error, scale, check.size))
    if closest is not None and best <= PROFILE_TOLERANCE:
        return closest
    msg = (
        f"the {name} statistic: its profile cannot be tabulated to rounding level "
        f"over the run (best measured relative error {best:.3g}, above "
        f"{PROFILE_TOLERANCE:g})"
    )
    raise BackendError(msg)


def _shape(
    problem: PropagationProblem, name: str, n_windows: int, plan: Mapping[str, object]
) -> tuple[int, ...]:
    """Return an output's shape on this problem, without a batch axis.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
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
    modes, pol, n_omega = problem.shape
    apertures = int(np.asarray(plan["apertures"]).shape[0])
    regions = int(np.asarray(plan["regions"]).shape[0])
    species = int(cast("int", plan["n_species"]))
    nodes = int(cast("int", plan["n_nodes"]))
    absorbed = int(cast("int", plan["n_absorbed"]))
    channels = int(cast("int", plan["n_channels"]))
    shapes: dict[str, tuple[int, ...]] = {
        "omega0": (modes, pol),
        "energy": () if plan["route"] == _ROUTES["spatial"] else (modes, pol),
        "energy_window": (n_windows, modes, pol),
        "peak_power": (modes,),
        "zdw": (modes,),
        "peak_power_window": (n_windows, modes),
        "beam_radius": (2,),
        "species_ionisation": (species, nodes),
        "species_electron_density": (species,),
        "species_peak_ionisation_rate": (species,),
        "on_axis_species_electron_density": (species,),
        "on_axis_species_peak_ionisation_rate": (species,),
        "species_density": (species,),
        "species_ionisation_energy": (species,),
        "spectral_flux": (n_omega,),
        "on_axis_spectrum": (n_omega,),
        "beam_spectrum": (n_omega,),
        "absorbed_power": (absorbed,),
        "region_spectrum": (regions, n_omega),
        "aperture_spectrum": (apertures, n_omega),
        "aperture_on_axis_spectrum": (apertures, n_omega),
        NODE_POPULATIONS: (nodes, channels, 2),
    }
    return shapes.get(name, ())


def _on_axis(modes: Sequence[Mode]) -> np.ndarray:
    """Return the mode set's synthesis on the axis at the reference position.

    ``HostDriver._on_axis_synthesis`` before its taper: the single node ``r = 0`` of
    :func:`~solprop.modes.quadrature.on_axis_rule`, rows x and y.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set.

    Returns
    -------
    numpy.ndarray
        ``(2, n_modes)``.
    """
    rule = on_axis_rule(modes[0].dimlimits(0.0)[0])
    return mode_matrix(modes, rule, 0.0, components=("x", "y")).synthesis


def metric_refusal(problem: PropagationProblem, names: Sequence[str]) -> str | None:
    """Return why native refuses a basis's metric statistics, naming them, or ``None``.

    The plan whitens a general modal basis's columns with the root of its flux metric,
    formed exactly from the metric's own values where the metric is constant on each of
    its intervals (:func:`_root_table`). A metric that varies inside an interval has no
    such root, so native refuses its metric statistics by name, before anything is
    allocated; ``numpy_ref`` computes them (maintainer's decision, SEP 47 phase 4). No
    built-in route builds such a metric (:func:`solprop.translate.coupled.
    dense_metric` is constant on each interval).

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    names : sequence of str
        The statistics asked for.

    Returns
    -------
    str or None
        The refusal, or ``None``.
    """
    if _route(problem) != "general" or not _metric_varies(problem):
        return None
    asked = tuple(name for name in names if name in METRIC_WEIGHTED_STATISTICS)
    if not asked:
        return None
    return (
        f"the metric statistics {asked!r} of a general modal basis whose flux metric "
        "varies inside one of its intervals: the native plan whitens the columns "
        "with the metric's root, exact only where the metric is constant on each "
        "interval; use backend='numpy_ref'"
    )


ZDW_TOLERANCE = 1e-6
"""The largest error a ``zdw`` table may carry, relative to the mode's
zero-dispersion wavelength on the run, measured on its check set against the root
search itself (:func:`solprop.modes.base.zdw`): a picometre at a micrometre.

Not rounding level, because the reference is not: its root scatters by up to a few
times 1e-8 of the wavelength between neighbouring positions on a gas capillary
(``beta_2`` is a finite difference of the dispersion), which no table of z follows,
and a pressure gradient over a decade is resolved to about 5e-7 within the fit's
budget (:data:`_ZDW_FIT`)."""


class _FitOptions(TypedDict):
    """Builder options for :func:`solprop.translate.chebyshev.build`."""

    tol: float
    n_max: int
    max_intervals: int


_ZDW_FIT: _FitOptions = {"tol": 1e-7, "n_max": 16, "max_intervals": 16}
"""The ``zdw`` fit's builder options (:func:`solprop.translate.chebyshev.build`): each
sample is a root search of about 0.2 s, so the degree and the intervals are capped
(at most a few hundred searches) rather than refined to a rounding level the root
search does not have."""

_ZDW_CHECK_POINTS = 17
"""Uniform positions of a ``zdw`` table's check set, beside its knots, edges and
midpoints: each is a root search, 0.2 to 0.6 s on a gas capillary, so far fewer than
a profile's."""

_ZDW_PLANE_LIMIT = 2 * _ZDW_CHECK_POINTS
"""The most positions a ``zdw`` is searched at exactly, in place of its fitted table:
the fewest searches a fit costs, its check set and one interval of degree 16."""

_ZDW_TABLES: weakref.WeakKeyDictionary[
    PropagationProblem,
    dict[
        tuple[float, tuple[float, ...] | None],
        tuple[tuple[int, ...], list[PiecewiseChebyshev], dict[str, object]],
    ],
] = weakref.WeakKeyDictionary()
"""Each problem's ``zdw`` tables and their record, by last plane and exact positions,
with the modes' ids they were lowered for, while the problem lives."""


def _zdw_tables(
    problem: PropagationProblem,
    modes: Sequence[Mode],
    grid: TimeGrid,
    z_end: float,
    planes: Sequence[float] | None = None,
) -> tuple[list[PiecewiseChebyshev], dict[str, object]]:
    """Return each mode's ``zdw`` table and their record, lowered once per problem.

    Where every position the statistic is taken at is known (`planes`) and there are
    no more than :data:`_ZDW_PLANE_LIMIT`, a z-varying mode's root is searched at
    exactly those positions, as the reference searches it there, and held in a table
    constant around each (:func:`_plane_tables`); otherwise it is fitted
    (:func:`_lowered_zdw_tables`).

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    modes : sequence of Mode
        The propagated modes, in state order.
    grid : solprop.grid.TimeGrid
        The time grid, whose band the roots are searched in.
    z_end : float
        The run's last plane, m.
    planes : sequence of float or None, optional
        Every position the statistic will be taken at, or ``None`` when they are not
        all known before the run (step planes, interpolated planes, a caller's own).

    Returns
    -------
    tuple
        The tables, and their record for ``statistics_tables`` (a copy).
    """
    held = _ZDW_TABLES.setdefault(problem, {})
    identity = tuple(id(mode) for mode in modes)
    exact = None
    if planes is not None:
        positions = tuple(sorted({float(z) for z in planes}))
        if positions and len(positions) <= _ZDW_PLANE_LIMIT:
            exact = positions
    key = (z_end, exact)
    cached = held.get(key)
    if cached is None or cached[0] != identity:
        if exact is not None and not all(mode.z_constant for mode in modes):
            tables, record = _plane_tables(modes, grid, exact, z_end)
        else:
            tables, record = _lowered_zdw_tables(problem, modes, grid, z_end)
        cached = held[key] = (identity, tables, record)
    return cached[1], dict(cached[2])


def _plane_tables(
    modes: Sequence[Mode], grid: TimeGrid, positions: Sequence[float], z_end: float
) -> tuple[list[PiecewiseChebyshev], dict[str, object]]:
    """Return each mode's ``zdw`` searched at exactly the positions it is taken at.

    Each table is constant on an interval about each position, its edges halfway
    between neighbours, so that it holds the reference's own root at each, to the
    bit: the run's searches, made once before it rather than at each plane.

    Parameters
    ----------
    modes : sequence of Mode
        The propagated modes.
    grid : solprop.grid.TimeGrid
        The time grid.
    positions : sequence of float
        The positions, increasing and distinct, m.
    z_end : float
        The run's last plane, m.

    Returns
    -------
    tuple
        The tables, one per mode, and their record (error zero).
    """
    import warnings

    from solprop.exceptions import NoZeroDispersionError, SolpropWarning
    from solprop.modes.base import zdw
    from solprop.translate.linop import DEFAULT_MARGIN

    window = inband_wavelengths(grid)
    points = np.asarray(positions, dtype=np.float64)
    end = max(float(z_end), float(points[-1]))
    lower = min(float(points[0]), 0.0)
    upper = max(end, end + abs(end) * DEFAULT_MARGIN)
    edges = np.concatenate(([lower], 0.5 * (points[1:] + points[:-1]), [upper]))
    tables: list[PiecewiseChebyshev] = []
    missing: list[int] = []
    for index, mode in enumerate(modes):
        values = np.empty(points.size)
        for slot, z in enumerate(points):
            try:
                values[slot] = float(zdw(mode, float(z), wavelength_limits=window))
            except NoZeroDispersionError:
                values[slot] = math.nan
        if np.any(np.isnan(values)):
            missing.append(index)
        tables.append(
            PiecewiseChebyshev(
                values[:, None],
                edges,
                np.zeros(points.size, dtype=np.int64),
                np.empty(0),
            )
        )
    if missing:
        warnings.warn(
            f"the 'zdw' statistic records nan for modes {missing}: beta_2 has one "
            f"sign at both ends of the grid's {window[0] * 1e9:.4g}-"
            f"{window[1] * 1e9:.4g} nm band at a plane",
            SolpropWarning,
            stacklevel=4,
        )
    record: dict[str, object] = {
        "source": "save planes",
        "max_error": 0.0,
        "relative_error": 0.0,
        "check_points": int(points.size),
        "tolerance": ZDW_TOLERANCE,
    }
    return tables, record


def _constant_table(value: float, lower: float, upper: float) -> PiecewiseChebyshev:
    """Return a scalar table holding one value exactly: degree zero, one interval."""
    return PiecewiseChebyshev(
        np.array([[value]], dtype=np.float64),
        np.array([lower, upper], dtype=np.float64),
        np.zeros(1, dtype=np.int64),
        np.empty(0),
    )


def _lowered_zdw_tables(
    problem: PropagationProblem, modes: Sequence[Mode], grid: TimeGrid, z_end: float
) -> tuple[list[PiecewiseChebyshev], dict[str, object]]:
    """Lower each mode's zero-dispersion wavelength along z to a table.

    The root search of :func:`solprop.stats.collect` (``beta_2``'s zero in the grid's
    band, :func:`solprop.stats.inband_wavelengths`), NaN where the band holds none, as
    the reference records it. A z-independent mode's table holds its one root, which is
    the reference's at every plane, to the bit. Any other mode's root is tabulated over
    the propagation's span with the problem's own table edges as knots
    (:func:`solprop.translate.chebyshev.build` with :data:`_ZDW_FIT`), and its
    measured error on a check set must be within :data:`ZDW_TOLERANCE`, or the
    statistic is refused by name; so is a mode whose band holds a root over part of
    the run only.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    modes : sequence of Mode
        The propagated modes.
    grid : solprop.grid.TimeGrid
        The time grid.
    z_end : float
        The run's last plane, m.

    Returns
    -------
    tuple
        The tables, one per mode, and the record of the largest measured error.

    Raises
    ------
    solprop.exceptions.BackendError
        Naming the mode whose root no table represents within :data:`ZDW_TOLERANCE`.

    Warns
    -----
    solprop.exceptions.SolpropWarning
        Once, if a mode's band holds no zero at the run's start.
    """
    import warnings

    from solprop.exceptions import (
        NoZeroDispersionError,
        SolpropWarning,
    )
    from solprop.modes.base import zdw
    from solprop.translate.chebyshev import build
    from solprop.translate.linop import DEFAULT_MARGIN
    from solprop.translate.problem import (
        _walk_tables,  # pyright: ignore[reportPrivateUsage]
    )

    window = inband_wavelengths(grid)
    start = float(problem.initial.z)
    end = float(z_end)
    if not end > start:
        end = start + max(abs(start), 1.0) * 1e-12
    lower = min(start, 0.0)
    upper = max(end, end + abs(end) * DEFAULT_MARGIN)

    def root(mode: Mode, z: float) -> float:
        try:
            return float(zdw(mode, z, wavelength_limits=window))
        except NoZeroDispersionError:
            return math.nan

    first = [root(mode, start) for mode in modes]
    missing = [index for index, value in enumerate(first) if math.isnan(value)]
    if missing:
        warnings.warn(
            f"the 'zdw' statistic records nan for modes {missing}: beta_2 has one "
            f"sign at both ends of the grid's {window[0] * 1e9:.4g}-"
            f"{window[1] * 1e9:.4g} nm band",
            SolpropWarning,
            stacklevel=4,
        )
    knots = sorted(
        {
            float(edge)
            for _, table in _walk_tables(problem)  # pyright: ignore[reportPrivateUsage]
            for edge in np.asarray(table.breakpoints)
        }
    )
    uniform = np.linspace(start, end, _ZDW_CHECK_POINTS)
    tables: list[PiecewiseChebyshev] = []
    worst_error, worst_relative, checked = 0.0, 0.0, 0
    for index, (mode, value) in enumerate(zip(modes, first, strict=True)):
        if mode.z_constant:
            tables.append(_constant_table(value, lower, upper))
            continue

        def path(z: np.ndarray, mode: Mode = mode) -> np.ndarray:
            return np.asarray([root(mode, float(one)) for one in np.atleast_1d(z)])

        with warnings.catch_warnings():
            # Each position repeats the start's material warnings, given once above.
            warnings.simplefilter("ignore")
            check = np.unique(
                np.concatenate(
                    (uniform, [knot for knot in knots if start <= knot <= end])
                )
            )
            exact = path(check)
            if np.all(np.isnan(exact)):
                tables.append(_constant_table(math.nan, lower, upper))
                continue
            if np.any(np.isnan(exact)):
                msg = (
                    f"the zdw statistic: mode {index}'s band holds a zero-dispersion "
                    "wavelength over part of the run only, which no table of z "
                    "represents"
                )
                raise BackendError(msg)
            scale = float(np.max(np.abs(exact)))
            inside = [knot for knot in knots if lower < knot < upper]
            table = build(path, lower, upper, mandatory_breakpoints=inside, **_ZDW_FIT)
            edges = np.asarray(table.breakpoints)
            middles = 0.5 * (edges[1:] + edges[:-1])
            extra = np.concatenate((edges, middles))
            extra = extra[(extra >= start) & (extra <= end)]
            positions = np.concatenate((check, extra))
            wanted = np.concatenate((exact, path(extra)))
            tabulated = np.asarray(
                table.at_positions(positions, helpers=NATIVE), dtype=np.float64
            ).reshape(-1)
            error = float(np.max(np.abs(tabulated - wanted)))
            count = positions.size
            if not error <= ZDW_TOLERANCE * scale:
                msg = (
                    f"the zdw statistic: mode {index}'s zero-dispersion wavelength "
                    "cannot be tabulated along z within its tolerance (measured "
                    f"relative error {error / scale:.3g}, above {ZDW_TOLERANCE:g})"
                )
                raise BackendError(msg)
        tables.append(table)
        worst_error = max(worst_error, error)
        worst_relative = max(worst_relative, error / scale)
        checked += int(count)
    varying = any(not mode.z_constant for mode in modes)
    record: dict[str, object] = {
        "source": "lowered zdw" if varying else "z-independent modes",
        "max_error": worst_error,
        "relative_error": worst_relative,
        "check_points": checked,
        "tolerance": ZDW_TOLERANCE,
    }
    return tables, record


def _width_rule(problem: PropagationProblem) -> dict[str, object]:
    """Return a spatial route's fluence-width rule as the native plan reads it.

    The geometry's rule (:func:`solprop.backends.spatial.signed_fluence_widths`,
    :func:`~solprop.backends.spatial.plane_fluence_widths`): a radial profile with its
    radii and weights, a slab's profile in y with the weights it integrates over x, a
    line's x, or a plane's axes and the angles its rings sample, which its axes alone
    fix. The fluence's term count fixes its rounding budget
    (:func:`~solprop.backends.spatial.fluence_budget`).

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The spatial problem.

    Returns
    -------
    dict
        The plan's width entries.
    """
    geometry = problem.transform.geometry
    assert geometry is not None
    x = np.ascontiguousarray(geometry.axes[0], dtype=np.float64)
    weights = np.ascontiguousarray(geometry.spatial_weights, dtype=np.float64)
    rule: dict[str, object] = {
        "width_terms": int(problem.shape[1] * problem.grids.n_t)
        + int(problem.transform.n_modes)
    }
    if geometry.kind == "radial":
        rule.update(widths=1, width_x=x, width_weights=weights)
    elif geometry.kind == "slab":
        n_x, n_y = geometry.shape
        rule.update(
            widths=2,
            width_x=np.ascontiguousarray(geometry.axes[1], dtype=np.float64),
            width_weights=weights,
            width_n_x=int(n_x),
            width_n_y=int(n_y),
        )
    elif geometry.axes[1].size > 1:
        y = np.ascontiguousarray(geometry.axes[1], dtype=np.float64)
        cosines, sines = _ring_angles(x, y)
        rule.update(
            widths=4,
            width_x=x,
            width_y=y,
            ring_cosines=np.ascontiguousarray(cosines, dtype=np.float64),
            ring_sines=np.ascontiguousarray(sines, dtype=np.float64),
        )
    else:
        rule.update(widths=3, width_x=x)
    return rule


def lower(
    problem: PropagationProblem,
    spec: StatsSpec,
    names: Sequence[str],
    *,
    grid: TimeGrid | None,
    modes: Sequence[Mode] | None = None,
    profiles: Mapping[str, ZProfile] | None = None,
    apertures: tuple[np.ndarray, np.ndarray, int] | None = None,
    on_axis: np.ndarray | None = None,
    z_end: float,
    planes: Sequence[float] | None = None,
) -> StatisticsLowering:
    """Lower statistics for native evaluation, once.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    spec : solprop.stats.StatsSpec
        The windows the names read.
    names : sequence of str
        The outputs, each one the plan evaluates on this route (:func:`uncovered`),
        and :data:`NODE_POPULATIONS`.
    grid : solprop.grid.TimeGrid or None
        The time grid; the widths and the general spectral flux read it.
    modes : sequence of Mode or None, optional
        The mode set; ``peak_intensity`` and the on-axis plasma statistics on a modal
        route read its axis synthesis.
    profiles : mapping or None, optional
        The z profiles the profile statistics read (:func:`_profile_table`).
    apertures : tuple or None, optional
        A spatial route's far-field tables packed for the native pass (the adapter's
        ``_aperture_payload``): the apertures, the regions and the exponent.
    on_axis : numpy.ndarray or None, optional
        The mode set's synthesis on the axis at the reference position, when the
        caller already built it (``HostDriver._on_axis``); built here otherwise.
    z_end : float
        The run's last plane, m, which a profile statistic's table must reach.
    planes : sequence of float or None, optional
        Every position the plane statistics will be taken at, when the run fixes
        them all before it starts; ``zdw`` is then searched exactly there
        (:func:`_zdw_tables`).

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
        "n_modes": int(problem.shape[0]),
        "n_pol": int(problem.shape[1]),
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
        "pad_scale": float(problem.grids.pad_scale),
        "axis_node": -1,
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
        "widths": 0,
        "width_x": np.empty(0),
        "width_y": np.empty(0),
        "width_weights": np.empty(0),
        "width_n_x": 0,
        "width_n_y": 0,
        "width_terms": 0,
        "ring_cosines": np.empty(0),
        "ring_sines": np.empty(0),
        "e2_fraction": math.exp(-2),
        "zdw": [],
    }
    tables: dict[str, dict[str, object]] = {}
    for index, name in enumerate(_PROFILE_NAMES):
        if name not in asked:
            continue
        if profiles is None or name not in profiles:
            msg = f"the {name} statistic needs the {name!r} profile"
            raise BackendError(msg)
        table, tables[name] = _profile_table(problem, name, profiles[name], z_end)
        slots = plan["profiles"]
        assert isinstance(slots, list)
        slots[index] = _table(table)
    if "zdw" in asked:
        if modes is None or grid is None:
            msg = "the zdw statistic needs the mode set and the time grid"
            raise BackendError(msg)
        zdw_tables, tables["zdw"] = _zdw_tables(problem, modes, grid, z_end, planes)
        plan["zdw"] = [_table(table) for table in zdw_tables]
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
    axis = asked & set(ON_AXIS_PLASMA_STATISTICS)
    if axis:
        refused = axis_refusal(problem, sorted(axis))
        if refused is not None:
            raise BackendError(refused)
    if route == "modal" and ("peak_intensity" in asked or axis):
        if modes is None:
            msg = (
                "the on-axis plasma statistics need the mode set at construction"
                if axis
                else "the peak on-axis intensity needs the mode set at construction"
            )
            raise BackendError(msg)
        if on_axis is None:
            on_axis = _on_axis(modes)
        plan["on_axis"] = np.ascontiguousarray(on_axis, dtype=np.float64).reshape(-1)
        if axis:
            node = axis_node(problem, on_axis)
            plan["axis_node"] = -1 if node is None else node
    if asked & (_PLASMA | _SPECIES | {NODE_POPULATIONS}) or axis:
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
        if asked & {"beam_radius", "fwhm_r"}:
            plan.update(_width_rule(problem))
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
