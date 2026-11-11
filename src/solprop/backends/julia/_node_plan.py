# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plan Julia's retained node-array capacity without starting a numerical runtime.

The named inventory follows the Julia source and observation constructors, rather
than the native/reference backend's historical buffer multipliers. See
``julia/NODE_MEMORY.md`` for the derivation and explicit opaque-storage exclusions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from solprop.backends.julia._observations import (
    _SPATIAL_STATISTICS,
    _STATISTICS,
    _slots,
)
from solprop.backends.node_chunk import NodeChunkPlan, default_node_budget
from solprop.backends.session import ExecutionOptions
from solprop.boundaries import split_bucket_rates
from solprop.exceptions import BackendError
from solprop.translate.routes import ModalRoute, SpatialRoute

if TYPE_CHECKING:
    from solprop.translate.problem import PropagationProblem
    from solprop.translate.run_request import RunRequest

__all__ = ["JuliaNodePlan", "NodeBuffer", "plan_julia_nodes"]

_POPULATIONS = frozenset(
    {
        "species_ionisation",
        "species_electron_density",
        "electron_density",
        "peak_ionisation_rate",
    }
)
_TEMPORAL = frozenset({"peak_power", "peak_power_total", "fwhm_t", "fwhm_t_outer"})
_WIDTHS = frozenset({"beam_radius", "fwhm_r"})
_EXCLUSIONS = (
    "opaque FFTW native plan storage and temporary planning copies",
    "Julia/Python object, view, chunk-descriptor and allocator overhead",
    "immutable preparation arrays, method/state/history, input-noise and export data",
    "bounded transport rows and scalar notice/continuation records",
)


@dataclass(frozen=True)
class NodeBuffer:
    """Name one numerical workspace payload and its capacity dependence.

    Attributes
    ----------
    name : str
        Julia constructor-owned buffer family, unique within one inventory.
    per_node_bytes : int
        Upper bound per selected node or transverse trace, bytes.
    fixed_bytes : int
        Payload independent of the node chunk and complete spatial column count.
    full_volume_bytes : int
        Payload covering every retained mode/node, independent of node chunk.
    """

    name: str
    per_node_bytes: int = 0
    fixed_bytes: int = 0
    full_volume_bytes: int = 0


@dataclass(frozen=True)
class JuliaNodePlan:
    """Keep a request-specific bound on retained Julia node-array data.

    Attributes
    ----------
    plan : solprop.backends.node_chunk.NodeChunkPlan
        Selected chunk, byte budget and authoritative source of that selection.
    n_nodes : int
        Complete positive physical node count.
    per_node_bytes : int
        Sum of source, population and transverse chunk-dependent array bounds.
    fixed_workspace_bytes, full_volume_workspace_bytes : int
        Separately reported array payloads outside the node budget. These describe
        the listed source/observation work, not the complete session or RSS.
    inventory : tuple of NodeBuffer
        Named component bounds, without retaining the originating problem/request.
    opaque_fft_plans : tuple of str
        Planned transform families whose native allocations have no proven bound.
    exclusions : tuple of str
        Other storage explicitly outside the stated node-array budget.
    """

    plan: NodeChunkPlan
    n_nodes: int
    per_node_bytes: int
    fixed_workspace_bytes: int
    full_volume_workspace_bytes: int
    inventory: tuple[NodeBuffer, ...]
    opaque_fft_plans: tuple[str, ...]
    exclusions: tuple[str, ...] = _EXCLUSIONS

    @property
    def trace_chunk(self) -> int:
        """Return the jointly budgeted transverse trace capacity.

        Returns
        -------
        int
            The node chunk; each line additionally caps this at its trace count.
        """
        return self.plan.chunk

    @property
    def chunk_bytes(self) -> int:
        """Return the selected chunk's retained numerical array bound, bytes.

        Returns
        -------
        int
            ``plan.chunk * per_node_bytes``; forced choices may exceed the budget.
        """
        return self.plan.chunk * self.per_node_bytes


@dataclass(frozen=True)
class _Dimensions:
    """Keep scalar extents and element widths used by the owned array inventory.

    Attributes
    ----------
    fine_time, time, frequency, fine_frequency : int
        Fine/coarse temporal and spectral sample counts.
    modes, nodes : int
        Retained spectral columns and physical nodes.
    real, complex, scalar : int
        Real, complex and source-temporal element widths, bytes.
    spatial, cartesian, coarse : bool
        Spatial route, Cartesian/line route and distinct embedded projection flags.
    """

    fine_time: int
    time: int
    frequency: int
    fine_frequency: int
    modes: int
    nodes: int
    real: int
    complex: int
    scalar: int
    spatial: bool
    cartesian: bool
    coarse: bool


def _dimensions(problem: PropagationProblem) -> _Dimensions:
    """Read admitted scalar extents without starting Julia.

    Parameters
    ----------
    problem : PropagationProblem
        Validated numerical descriptor.

    Returns
    -------
    _Dimensions
        Positive extents, element widths and route flags.

    Raises
    ------
    BackendError
        For a route, precision or polarisation outside this inventory.
    """
    route, transform = problem.route, problem.transform
    if (
        not isinstance(route, (ModalRoute, SpatialRoute))
        or transform.modal_tables is not None
    ):
        raise BackendError("julia node planning refuses general modal routes")
    if isinstance(route, SpatialRoute) and route.kind == "planar":
        raise BackendError("julia node planning refuses planar routes")
    if problem.precision.single:
        raise BackendError("julia node planning currently refuses binary32")
    if problem.shape[1] != 1 or transform.n_pol != 1:
        raise BackendError("julia node planning refuses vector polarisation")
    grids = problem.grids
    field = problem.grid_kind == "field"
    real, complex_width = (
        problem.precision.real_dtype.itemsize,
        problem.precision.complex_dtype.itemsize,
    )
    cartesian = isinstance(route, SpatialRoute) and route.kind in {"cartesian", "line"}
    return _Dimensions(
        grids.n_to,
        grids.n_t,
        problem.shape[-1],
        grids.n_to // 2 + 1 if field else grids.n_to,
        problem.shape[0],
        transform.n_nodes,
        real,
        complex_width,
        real if field else complex_width,
        isinstance(route, SpatialRoute),
        cartesian,
        not cartesian
        and not np.array_equal(transform.projection, transform.projection_coarse),
    )


def _source_inventory(problem: PropagationProblem, d: _Dimensions) -> list[NodeBuffer]:
    """Count retained source work by its actual Julia constructor families.

    Parameters
    ----------
    problem : PropagationProblem
        Source response and prefactor descriptions.
    d : _Dimensions
        Admitted extents and element widths.

    Returns
    -------
    list of NodeBuffer
        Chunk-dependent, fixed and full-volume source payload bounds, bytes.
    """
    time, frequency, fine = d.fine_time, d.frequency, d.fine_frequency
    s, c, r = d.scalar, d.complex, d.real
    rows = [NodeBuffer("source.nodes_and_polarisation", 2 * time * s)]
    rows.append(NodeBuffer("source.padding_maps", fixed_bytes=8 * (frequency + fine)))
    if d.cartesian:
        rows.extend(
            (
                NodeBuffer("source.chunk_spectra", (fine + frequency) * c),
                NodeBuffer(
                    "source.spatial_band", full_volume_bytes=frequency * d.nodes * c
                ),
                NodeBuffer("source.failures", full_volume_bytes=8 * d.nodes),
            )
        )
    else:
        rows.extend(
            (
                NodeBuffer(
                    "source.modal_time_and_projection",
                    full_volume_bytes=(3 + d.coarse) * time * d.modes * s,
                ),
                NodeBuffer(
                    "source.modal_spectrum", full_volume_bytes=fine * d.modes * c
                ),
                NodeBuffer(
                    "source.prefactor_and_quadrature",
                    full_volume_bytes=(1 + 2 * d.coarse) * frequency * d.modes * c,
                ),
                NodeBuffer("source.failures", 8, full_volume_bytes=8 * d.modes),
            )
        )
        weights = problem.responses.prefactor_expansion
        rows.append(
            NodeBuffer(
                "source.prefactor_weights",
                fixed_bytes=0 if weights is None else r * weights.n_terms,
            )
        )
        if d.spatial:
            rows.append(
                NodeBuffer(
                    "source.radial_loss",
                    full_volume_bytes=(frequency * c + 8) * d.modes,
                )
            )
    responses = problem.responses
    analytic = problem.grid_kind == "field" and (
        (
            responses.kerr_coefficients is not None
            and responses.kerr_coefficients.analytic_signal
        )
        or any(not term.thg for term in responses.raman_terms)
    )
    if analytic:
        rows.append(
            NodeBuffer(
                "source.analytic",
                (3 if d.cartesian else 1) * time * c,
                full_volume_bytes=0 if d.cartesian else 2 * time * d.modes * c,
            )
        )
    if responses.raman_terms:
        rows.append(NodeBuffer("source.raman_shared", 2 * time * r + (time + 1) * c))
        for index, term in enumerate(responses.raman_terms):
            rows.append(
                NodeBuffer(
                    f"source.raman.{index}.kernel_and_resonances",
                    fixed_bytes=2 * time * r + (time + 1) * c + 2 * term.n_terms * r,
                )
            )
    if responses.population_channels:
        recurrence = 7 * r + 1 if problem.grid_kind == "field" else 4 * r + c + 1
        rows.append(NodeBuffer("source.plasma_recurrence", time * recurrence))
        rows.append(
            NodeBuffer(
                "source.plasma_columns",
                (4 if d.cartesian else 2) * time * s,
                full_volume_bytes=0
                if d.cartesian
                else (2 + 2 * d.coarse) * time * d.modes * s,
            )
        )
        rows.append(
            NodeBuffer(
                "source.plasma_factors",
                fixed_bytes=2 * frequency * c
                + 2 * len(responses.population_channels) * r,
            )
        )
    return rows


def _population_inventory(
    problem: PropagationProblem, d: _Dimensions
) -> list[NodeBuffer]:
    """Count independent observer population reconstruction and recurrence work.

    Parameters
    ----------
    problem : PropagationProblem
        Species and population channel descriptions.
    d : _Dimensions
        Admitted extents and element widths.

    Returns
    -------
    list of NodeBuffer
        Population workspace bounds, bytes.
    """
    time, s, c, r = d.fine_time, d.scalar, d.complex, d.real
    rows = [NodeBuffer("observations.population_nodes", time * s)]
    if d.cartesian:
        rows.extend(
            (
                NodeBuffer(
                    "observations.population_spectra",
                    (d.fine_frequency + d.frequency) * c,
                ),
                NodeBuffer(
                    "observations.population_spatial_band",
                    full_volume_bytes=d.frequency * d.nodes * c,
                ),
            )
        )
    else:
        rows.append(
            NodeBuffer(
                "observations.population_modal",
                full_volume_bytes=(d.fine_frequency * c + time * s) * d.modes,
            )
        )
    if problem.responses.population_channels:
        rows.append(
            NodeBuffer("observations.population_recurrence", time * (5 * r + 1))
        )
    species = problem.responses.n_species
    rows.extend(
        (
            NodeBuffer(
                "observations.population_species",
                fixed_bytes=species * r,
                full_volume_bytes=2 * d.nodes * species * r,
            ),
            NodeBuffer(
                "observations.population_padding",
                fixed_bytes=8 * (d.frequency + d.fine_frequency),
            ),
        )
    )
    return rows


def _observation_inventory(
    problem: PropagationProblem, request: RunRequest, d: _Dimensions
) -> list[NodeBuffer]:
    """Count selected statistics and temporal monitor or split work.

    Parameters
    ----------
    problem : PropagationProblem
        Numerical response, power-weight and split descriptions.
    request : RunRequest
        Selected statistics and boundary services.
    d : _Dimensions
        Admitted extents and element widths.

    Returns
    -------
    list of NodeBuffer
        Observer workspace bounds, bytes; empty when services are absent.
    """
    names = frozenset(request.observation.stats.names)
    catalogue = _SPATIAL_STATISTICS if d.spatial else _STATISTICS
    unsupported = names - catalogue
    if unsupported:
        raise BackendError(
            "julia node planning refuses unsupported statistics: "
            + ", ".join(sorted(unsupported))
        )
    monitor = request.boundary.monitor
    split = (
        problem.responses.absorber_scheme == "split_step"
        and problem.responses.temporal_absorber is not None
    )
    if not names and monitor is None and not split:
        return []
    f, n, m, x, r, c = d.frequency, d.time, d.modes, d.nodes, d.real, d.complex
    population = bool(names & _POPULATIONS)
    rows = [NodeBuffer("observations.spectrum", full_volume_bytes=f * m * c)]
    rows.append(
        NodeBuffer(
            "observations.packed_values",
            fixed_bytes=2 * sum(slot.count for slot in _slots(problem, request)) * r,
        )
    )
    if "quadrature_error" in names:
        rows.append(NodeBuffer("observations.quadrature", full_volume_bytes=5 * m * r))
    rows.append(
        NodeBuffer(
            "observations.statistics_work",
            fixed_bytes=n * r,
            full_volume_bytes=(n * m + 2 * m + 3 * x * (d.spatial or population)) * r,
        )
    )
    if d.spatial:
        rows.extend(_spatial_observation_inventory(problem, names, d))
    else:
        rows.append(
            NodeBuffer("observations.power_weight", full_volume_bytes=f * m * r)
        )
        expansion = problem.linear.power_weight_expansion
        rows.append(
            NodeBuffer(
                "observations.power_weight_work",
                fixed_bytes=0 if expansion is None else expansion.n_terms * r,
            )
        )
        if names & _TEMPORAL or "peak_intensity" in names:
            rows.append(
                NodeBuffer(
                    "observations.analytic_pair", full_volume_bytes=4 * n * m * c
                )
            )
        if "peak_power_window" in names:
            rows.append(
                NodeBuffer(
                    "observations.window_pairs",
                    full_volume_bytes=4
                    * n
                    * m
                    * request.observation.stats.n_windows
                    * c,
                )
            )
        if "peak_intensity" in names:
            # Scalar state polarisation does not imply a one-component physical
            # axis map: the shared lowering admits one or two components.
            rows.append(NodeBuffer("observations.axis_pair", fixed_bytes=4 * n * c))
    if population:
        rows.extend(_population_inventory(problem, d))
    if monitor is not None:
        crossing = problem.split_crossing if split else None
        # The shared policy retains every defined rate bucket, including empty
        # buckets absent from the represented spectral indices.
        buckets = 0 if crossing is None else len(split_bucket_rates(crossing.fastest))
        rows.append(
            NodeBuffer(
                "observations.temporal_monitor",
                fixed_bytes=(n + 2 * buckets) * r,
                full_volume_bytes=(f * c + n * d.scalar) * m,
            )
        )
    if split:
        rows.append(
            NodeBuffer(
                "observations.temporal_split",
                fixed_bytes=n * r,
                full_volume_bytes=(n * d.scalar + f * c) * m
                + (f * m * c if d.cartesian else 0),
            )
        )
    return rows


def _spatial_observation_inventory(
    problem: PropagationProblem, names: frozenset[str], d: _Dimensions
) -> list[NodeBuffer]:
    """Count selected physical-node and on-axis spatial reconstruction work.

    Parameters
    ----------
    problem : PropagationProblem
        Admitted spatial descriptor.
    names : frozenset of str
        Selected statistic names.
    d : _Dimensions
        Admitted extents and element widths.

    Returns
    -------
    list of NodeBuffer
        Fixed/full-volume spatial observation bounds, bytes.
    """
    f, n, m, x, r, c = d.frequency, d.time, d.modes, d.nodes, d.real, d.complex
    widths, local, axis = (
        bool(names & _WIDTHS),
        "peak_intensity" in names,
        "on_axis_intensity" in names,
    )
    physical = widths or local
    axial = widths or axis or "on_axis_spectrum" in names
    rows = []
    if "energy" in names:
        rows.append(
            NodeBuffer("observations.power_weight", full_volume_bytes=f * m * r)
        )
    if physical or axial:
        rows.append(
            NodeBuffer("observations.flux_closure", full_volume_bytes=f * m * c)
        )
    route = problem.route
    assert isinstance(route, SpatialRoute)
    if route.flux.material is not None:
        rows.append(NodeBuffer("observations.material_index", fixed_bytes=f * c))
    if physical:
        rows.append(
            NodeBuffer(
                "observations.physical_pair",
                full_volume_bytes=((2 + d.cartesian) * f + 4 * n * local) * x * c,
            )
        )
    if axial:
        rows.append(
            NodeBuffer(
                "observations.spatial_axis", fixed_bytes=(2 * f + 4 * n * axis) * c
            )
        )
    if widths:
        # Radial samples or a half-pixel radial average never outnumber the
        # complete radial/Cartesian/line physical node set.
        rows.append(NodeBuffer("observations.radial_average", full_volume_bytes=x * r))
    return rows


def _transverse_inventory(request: RunRequest, d: _Dimensions) -> list[NodeBuffer]:
    """Bound every retained transverse line and optional noise reading.

    Parameters
    ----------
    request : RunRequest
        Selected boundary service, without consulting its measured state.
    d : _Dimensions
        Admitted extents and element widths.

    Returns
    -------
    list of NodeBuffer
        Joint trace-chunk and full-volume bounds, bytes.
    """
    monitor = request.boundary.monitor
    if monitor is None or monitor.transverse is None:
        return []
    edge = monitor.transverse
    rows = [
        NodeBuffer(
            "transverse.spatial_band",
            full_volume_bytes=d.frequency * d.nodes * d.complex,
        )
    ]
    # Charge every optional noise/clean reading, including future resumed starts.
    rows.append(
        NodeBuffer(
            "transverse.column_and_node_readings",
            full_volume_bytes=(d.nodes + 5 * d.modes) * d.real,
        )
    )
    lines = edge.reduction.lines
    rows.append(
        NodeBuffer(
            "transverse.small_readings",
            fixed_bytes=(4 * len(lines) + 3 * len(lines)) * d.real,
        )
    )
    for index, line in enumerate(lines):
        rows.append(
            NodeBuffer(
                f"transverse.line.{index}",
                d.complex * (2 * len(line.strip) + 3 * (line.onset is not None)),
            )
        )
    return rows


def _opaque_plans(
    problem: PropagationProblem, request: RunRequest, d: _Dimensions
) -> tuple[str, ...]:
    """Name retained transform families whose native plan allocations are opaque.

    Parameters
    ----------
    problem : PropagationProblem
        Source response and split descriptions.
    request : RunRequest
        Selected observation and boundary services.
    d : _Dimensions
        Admitted route flags.

    Returns
    -------
    tuple of str
        Explicit plan exclusions, never guessed byte estimates.
    """
    names = ["source temporal FFT plans"]
    if d.cartesian:
        names.append("source spatial FFT plans")
    if problem.responses.raman_terms:
        names.append("shared Raman convolution and per-component kernel FFT plans")
    if request.observation.stats.names or request.boundary.monitor is not None:
        names.append("selected observation and monitor FFT plans")
    if (
        problem.responses.temporal_absorber is not None
        and problem.responses.absorber_scheme == "split_step"
    ):
        names.append("accepted temporal split FFT plans")
    if (
        request.boundary.monitor is not None
        and request.boundary.monitor.transverse is not None
    ):
        names.append("transverse spatial and per-line strip FFT plans")
    return tuple(names)


def plan_julia_nodes(
    problem: PropagationProblem,
    request: RunRequest,
    options: ExecutionOptions | None = None,
) -> JuliaNodePlan:
    """Bound Julia-owned node arrays and select a balanced physical-node chunk.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Validated scalar binary64 modal, radial, Cartesian or line description.
    request : solprop.translate.run_request.RunRequest
        Selected statistics and boundary services; no existing measured monitor
        state is consulted or retained.
    options : solprop.backends.session.ExecutionOptions or None, optional
        Authoritative explicit byte budget or forced chunk; otherwise route defaults.

    Returns
    -------
    JuliaNodePlan
        Frozen numerical-array inventory and chunk choice, without runtime entry.

    Raises
    ------
    BackendError
        For an unsupported layout or a budget smaller than one complete node.
        An explicit chunk bypasses the budget and is capped at all physical nodes.

    Notes
    -----
    The budget excludes fixed/full-volume work and opaque FFT native allocations.
    It is not a total-session or RSS bound. See ``julia/NODE_MEMORY.md``.
    """
    options = ExecutionOptions() if options is None else options
    d = _dimensions(problem)
    inventory = tuple(
        _source_inventory(problem, d)
        + _observation_inventory(problem, request, d)
        + _transverse_inventory(request, d)
    )
    per_node = sum(entry.per_node_bytes for entry in inventory)
    budget = (
        default_node_budget(problem.transform)
        if options.budget is None
        else options.budget
    )
    if options.chunk is not None:
        plan = NodeChunkPlan(
            min(options.chunk, d.nodes), budget, "forced", options.chunk
        )
    else:
        if budget < per_node:
            raise BackendError(
                f"julia node_budget cannot hold one node: {per_node} "
                "retained array bytes required"
            )
        limit = min(d.nodes, budget // per_node)
        passes = (d.nodes + limit - 1) // limit
        plan = NodeChunkPlan(
            (d.nodes + passes - 1) // passes,
            budget,
            "default" if options.budget is None else "explicit",
        )
    return JuliaNodePlan(
        plan,
        d.nodes,
        per_node,
        sum(entry.fixed_bytes for entry in inventory),
        sum(entry.full_volume_bytes for entry in inventory),
        inventory,
        _opaque_plans(problem, request, d),
    )
