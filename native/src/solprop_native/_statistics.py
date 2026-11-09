# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Statistics lowered once for native evaluation (solprop SEP 47 section 6.4).

:func:`lower` turns a set of statistic names into a native
:class:`~solprop_native._core.StatisticsPlan` (``native/include/solprop_native/
statistics_plan.hpp``): which statistics, in which order, and every z-dependent input
as the representation the propagation itself carries -- solprop's piecewise Chebyshev
tables of the plasma density, each channel's preionisation, the species densities, the
taper, the effective area and the modal power weight, read by the native Clenshaw
recursion bit for bit as solprop evaluates them. Two inputs carry no table, and are
lowered here into one with a recorded error:

- a profile a profile statistic reads (``density``, ``pressure``, ``core_radius``), a
  Python :class:`~solprop.profiles.ZProfile`, is fitted by solprop's own adaptive
  Chebyshev builder (:func:`solprop.translate.chebyshev.build`) at its declared
  breakpoints over the run's span;
- a general modal basis's flux metric ``W`` is a table, but the statistics read its
  square root (:func:`solprop.stats._whitened`): on a metric constant over each of its
  intervals (every route today) the root is formed from the same values by the same
  ``eigh`` and is exact; a metric that varies inside an interval is fitted like a
  profile.

Each fit's error is measured against the Python object on a dense check set and
recorded in :attr:`StatisticsLowering.fit_errors`, at least the builder's own
tolerance times the value scale: a native value differs from the Python one by no
more than it. The native evaluation finishes every statistic (the widths, the species
reduction, the integration ratios); :meth:`StatisticsLowering.unpack` only names and
shapes the packed values.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError
from solprop.modes.quadrature import mode_matrix, on_axis_rule
from solprop.spatial import ABSORBED_CHANNELS
from solprop.stats import (
    FAR_FIELD_STATISTICS,
    METRIC_WEIGHTED_STATISTICS,
    PROFILE_STATISTICS,
)
from solprop.translate import chebyshev

from . import _core

if TYPE_CHECKING:
    from solprop.grid import TimeGrid
    from solprop.modes.protocol import Mode
    from solprop.profiles import ZProfile
    from solprop.spatial import ApertureTables
    from solprop.stats import StatsSpec
    from solprop.translate.chebyshev import PiecewiseChebyshev
    from solprop.translate.problem import KernelProblem

__all__ = [
    "ADMITTED_GENERAL_STATISTICS",
    "ADMITTED_SPATIAL_STATISTICS",
    "ADMITTED_STATISTICS",
    "PLANE_ONLY",
    "STATISTIC_IDS",
    "StatisticsLowering",
    "lower",
    "uncovered",
]

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
            "density",
            "pressure",
            "core_radius",
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
        )
    )
}
"""Each statistic the native plan evaluates, by its id (``statistics::Stat``)."""

PLANE_ONLY = (
    "zdw",
    "beam_radius",
    "fwhm_r",
    "on_axis_electron_density",
    "on_axis_peak_ionisation_rate",
    "on_axis_species_electron_density",
    "on_axis_species_peak_ionisation_rate",
)
"""The statistics the plan does not evaluate, none periodic-capable.

``zdw`` is a root search over the Python mode objects; ``beam_radius`` and ``fwhm_r``
finish a fluence profile's widths on the host
(:func:`solprop.backends.spatial.signed_fluence_widths`,
:func:`~solprop.backends.spatial.plane_fluence_widths`); the four on-axis plasma
statistics scan the axis row in time on the host
(:func:`solprop.backends.populations.axis_row_populations`). Each is a plane
statistic only (``StatDefinition.between_planes`` false).
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
"""The statistics it evaluates on a general modal basis: functionals of ``a.H W a``."""

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
_EDGE_SLACK = 1e-10
"""solprop's edge slack (``chebyshev._EDGE_SLACK_FRACTION``), of a table's span."""

_CHECK_POINTS = 4097
"""Uniform positions of the dense check set a fit's error is measured on."""


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
        The names a plan leaves to the host, in request order.
    """
    admitted = _ADMITTED[_route(problem)]
    return tuple(name for name in names if name not in admitted)


@dataclasses.dataclass(frozen=True)
class StatisticsLowering:
    """A statistics evaluation lowered for the native plan.

    Attributes
    ----------
    names : tuple of str
        The statistics, in packed order.
    plan : solprop_native._core.StatisticsPlan or None
        The native plan; ``None`` for no names.
    shapes : dict
        Each statistic's shape, without a batch axis.
    fit_errors : dict
        The recorded error of each input lowered to a fitted table rather than read
        from one the problem carries: by profile statistic, and ``"metric_root"``, in
        the input's own units; a native value differs from the Python one by no more.
    """

    names: tuple[str, ...]
    plan: _core.StatisticsPlan | None
    shapes: dict[str, tuple[int, ...]]
    fit_errors: dict[str, float]

    def unpack(self, packed: np.ndarray) -> dict[str, np.ndarray]:
        """Name the packed values and give each its catalogue shape.

        Parameters
        ----------
        packed : numpy.ndarray
            The native evaluation, in :attr:`names` order.

        Returns
        -------
        dict
            The statistics by name, without a batch axis; arithmetic-free.

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


def _table(table: PiecewiseChebyshev | None, *, real: bool = True) -> dict | None:
    """Return a problem's table as the native plan reads it.

    Parameters
    ----------
    table : solprop.translate.chebyshev.PiecewiseChebyshev or None
        The table.
    real : bool, optional
        Whether the slot is real.

    Returns
    -------
    dict or None
        The breakpoints, the flattened coefficients and the span with its slack.

    Raises
    ------
    solprop.exceptions.BackendError
        If a real slot holds a complex table.
    """
    if table is None:
        return None
    coefficients = np.asarray(table.coefficients)
    if real and np.iscomplexobj(coefficients):
        msg = "a real statistics input is a complex table"
        raise BackendError(msg)
    dtype = np.float64 if real else np.complex128
    lo, hi = (float(value) for value in table.z_span)
    slack = _EDGE_SLACK * (hi - lo)
    return {
        "edges": np.ascontiguousarray(table.breakpoints, dtype=np.float64),
        "coefficients": np.ascontiguousarray(coefficients, dtype=dtype).reshape(-1),
        "n_terms": int(coefficients.shape[1]),
        "n_values": math.prod(coefficients.shape[2:]),
        "lower": lo - slack,
        "upper": hi + slack,
    }


def _fit(
    values: Callable[[np.ndarray], np.ndarray],
    start: float,
    end: float,
    breakpoints: Sequence[float],
    jumps: Sequence[float] = (),
) -> tuple[PiecewiseChebyshev, float]:
    """Fit a function of z with solprop's builder and measure the fit's error.

    Parameters
    ----------
    values : callable
        Positions ``(n_z,)`` to values ``(n_z, *value)``.
    start, end : float
        The span, m.
    breakpoints : sequence of float
        Declared kinks and jumps inside the span, m: mandatory edges.
    jumps : sequence of float, optional
        Declared jumps, m, where the table takes the value of the interval starting
        there; excluded from the check set.

    Returns
    -------
    tuple
        The table, and the recorded error: the largest difference on the check set
        (uniform positions, every edge and every interval's midpoint), at least the
        builder's tolerance times the value scale.
    """
    inner = sorted({float(b) for b in breakpoints if start < float(b) < end})
    table = chebyshev.build(values, start, end, mandatory_breakpoints=inner)
    edges = np.asarray(table.breakpoints)
    check = np.unique(
        np.concatenate(
            (
                np.linspace(start, end, _CHECK_POINTS),
                edges,
                0.5 * (edges[1:] + edges[:-1]),
            )
        )
    )
    check = check[~np.isin(check, np.asarray(jumps, dtype=np.float64))]
    exact = np.asarray(values(check)).reshape(check.size, -1)
    fitted = np.stack([np.asarray(table(float(z))).reshape(-1) for z in check])
    scale = float(np.max(np.abs(exact), initial=0.0))
    error = float(np.max(np.abs(fitted - exact), initial=0.0))
    return table, max(error, chebyshev.DEFAULT_TOL * scale)


def _profile_table(
    profile: ZProfile, start: float, end: float
) -> tuple[PiecewiseChebyshev, float]:
    """Lower a profile a profile statistic reads into a table, with its error.

    Parameters
    ----------
    profile : solprop.profiles.ZProfile
        The path.
    start, end : float
        The run's span, m.

    Returns
    -------
    tuple
        The table and its recorded error, in the profile's units.
    """
    low, high = (float(value) for value in profile.z_span)
    lo, hi = max(start, low), min(end, high)
    if not hi > lo:
        lo, hi = start, end
    return _fit(
        lambda z: np.asarray(
            profile(np.asarray(z, dtype=np.float64)), dtype=np.float64
        ),
        lo,
        hi,
        np.asarray(profile.breakpoints, dtype=np.float64).tolist(),
        np.asarray(profile.jumps, dtype=np.float64).tolist(),
    )


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
    solprop.exceptions.BackendError
        If the metric is not positive definite at some frequency.
    """
    values, vectors = np.linalg.eigh(np.moveaxis(metric, (0, 1), (-2, -1)))
    if np.any(values <= 0.0):
        msg = (
            "a metric-weighted statistic needs a positive-definite flux metric; the "
            f"smallest eigenvalue is {float(np.min(values)):.6g}"
        )
        raise BackendError(msg)
    return (vectors * np.sqrt(values)[..., None, :]) @ np.conj(
        np.swapaxes(vectors, -1, -2)
    )


def _root_table(
    problem: KernelProblem, start: float, end: float
) -> tuple[dict | None, float | None]:
    """Lower a general modal basis's metric root, or ``None`` for the identity.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The general modal problem.
    start, end : float
        The run's span, m.

    Returns
    -------
    tuple
        The table as the native plan reads it, and its recorded error (zero when the
        root is formed exactly, interval by interval).
    """
    from solprop.translate.coupled import metric_at

    metric = problem.linear.modal_metric
    if metric is None or metric.kind == "identity":
        return None, None
    block = metric.dense
    if block is None:
        msg = "a diagonal modal metric has no executed statistics path"
        raise BackendError(msg)
    edges = np.ascontiguousarray(metric.z_edges, dtype=np.float64)
    lo, hi = float(edges[0]), float(edges[-1])
    slack = _EDGE_SLACK * (hi - lo)
    if block.shape[1] == 1:
        # Constant on each interval: the root of the very values the reference reads.
        roots = np.stack(
            [
                _root(np.asarray(metric_at(metric, float(0.5 * (a + b)))))
                for a, b in itertools.pairwise(edges)
            ]
        )
        n_values = math.prod(roots.shape[1:])
        return {
            "edges": edges,
            "coefficients": np.ascontiguousarray(roots, dtype=np.complex128).reshape(
                -1
            ),
            "n_terms": 1,
            "n_values": n_values,
            "lower": lo - slack,
            "upper": hi + slack,
        }, 0.0
    table, error = _fit(
        lambda z: np.stack(
            [_root(np.asarray(metric_at(metric, float(p)))).reshape(-1) for p in z]
        ),
        max(start, lo),
        min(end, hi),
        edges[1:-1].tolist(),
    )
    return _table(table, real=False), error


def _shape(problem: KernelProblem, name: str, n_windows: int, plan: dict) -> tuple:
    """Return a statistic's catalogue shape on this problem, without a batch axis.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    name : str
        The statistic.
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
        "region_spectrum": (plan["regions"].shape[0], n_omega),
        "aperture_spectrum": (plan["apertures"].shape[0], n_omega),
        "aperture_on_axis_spectrum": (plan["apertures"].shape[0], n_omega),
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
    tables: ApertureTables | None = None,
) -> StatisticsLowering:
    """Lower statistics for native evaluation, once.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    spec : solprop.stats.StatsSpec
        The windows the names read.
    names : sequence of str
        The statistics, each one the plan evaluates on this route (:func:`uncovered`).
    grid : solprop.grid.TimeGrid or None
        The time grid; the widths and the general spectral flux read it.
    modes : sequence of Mode or None, optional
        The mode set; ``peak_intensity`` on a modal route reads its axis synthesis.
    profiles : mapping or None, optional
        The z profiles the profile statistics read.
    apertures : tuple or None, optional
        The far-field tables packed for the native pass (the adapter's
        ``_aperture_payload``): the apertures, the regions and the exponent.
    tables : solprop.spatial.ApertureTables or None, optional
        Unused beside `apertures`; kept for the caller's record.

    Returns
    -------
    StatisticsLowering
        The plan, the shapes and the recorded errors.

    Raises
    ------
    solprop.exceptions.BackendError
        Naming a statistic the plan does not evaluate here, or an input it lacks.
    """
    del tables
    names = tuple(names)
    left = uncovered(problem, names)
    if left:
        msg = f"the native statistics plan does not evaluate {left!r} on this route"
        raise BackendError(msg)
    asked = set(names)
    route = _route(problem)
    transform, responses, linear = problem.transform, problem.responses, problem.linear
    channels = responses.population_channels
    n_omega = int(problem.shape[-1])
    planes = np.asarray(problem.stepper.save_planes, dtype=np.float64)
    start = min(0.0, float(np.min(planes)))
    end = float(problem.stepper.z_end)
    fit_errors: dict[str, float] = {}
    plan: dict[str, object] = {
        "route": _ROUTES[route],
        "outputs": [STATISTIC_IDS[name] for name in names],
        "n_modes": int(problem.shape[1]),
        "n_pol": int(problem.shape[2]),
        "n_t": int(problem.grids.n_t),
        "n_omega": n_omega,
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
        "profiles": [None, None, None],
        "far_field_axis": False,
        "apertures": np.empty((0, 11 + n_omega)),
        "exponent": 2,
        "regions": np.empty((0, 4)),
        "n_absorbed": len(ABSORBED_CHANNELS),
        "root": None,
        "spectral_prefactor": 0.0,
    }
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
        rule = on_axis_rule(modes[0].dimlimits(0.0)[0])
        matrix = mode_matrix(modes, rule, 0.0, components=("x", "y")).synthesis
        plan["on_axis"] = np.ascontiguousarray(matrix, dtype=np.float64).reshape(-1)
    if asked & (_PLASMA | _SPECIES):
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
    for index, name in enumerate(PROFILE_STATISTICS):
        if name not in asked:
            continue
        if profiles is None or name not in profiles:
            msg = f"the {name} statistic needs the {name!r} profile"
            raise BackendError(msg)
        table, error = _profile_table(profiles[name], start, end)
        profile_slots = plan["profiles"]
        assert isinstance(profile_slots, list)
        profile_slots[index] = _table(table)
        fit_errors[name] = error
    if route == "spatial":
        geometry = transform.geometry
        assert geometry is not None
        plan["far_field_axis"] = geometry.kind == "free"
        if asked & set(FAR_FIELD_STATISTICS) or (
            "on_axis_spectrum" in asked and geometry.kind == "free"
        ):
            if apertures is not None:
                plan["apertures"], plan["regions"], plan["exponent"] = apertures
    if route == "general":
        if asked & METRIC_WEIGHTED_STATISTICS:
            root, error = _root_table(problem, start, end)
            plan["root"] = root
            if error is not None:
                fit_errors["metric_root"] = error
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
    return StatisticsLowering(names, native, shapes, fit_errors)
