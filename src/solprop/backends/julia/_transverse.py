# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Own numerical transverse-monitor preparation without a live Python monitor."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from solprop.backends.driver import (
    _declared_discard,  # pyright: ignore[reportPrivateUsage] - shared declaration
)
from solprop.backends.julia._descriptors import owned
from solprop.exceptions import BackendError
from solprop.grid import FreeGrid, RadialGrid
from solprop.spatial_absorber import RAMP_K_FLOOR
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import SpatialRoute
from solprop.transverse_monitor import (
    BACKSTOP_READS,
    NOISE_FACTOR,
    REACH_GROWTH,
    READS,
    TransverseEdge,
)

__all__ = [
    "TransverseFormat",
    "TransverseLineTransfer",
    "TransversePolicyTransfer",
    "TransverseTransfer",
    "lower_transverse",
    "transverse_format",
]


@dataclass(frozen=True)
class TransversePolicyTransfer:
    """Carry the shared edge's thresholds and scheduling constants.

    Attributes
    ----------
    warn, stop : float
        Warning and stop energy shares, formed in the shared policy's order.
    backstop_start, backstop_max : float
        Initial and maximum read spacing in metres.
    reach_growth : float
        Relative reach growth that resets the backstop.
    guard_numerator : float
        Four guard widths, in metres, divided by the measured reach in Julia.
    k_min, k_floor_fraction : float
        Requested minimum wavenumber in rad/m and relative longitudinal floor.
    due_tolerance : float
        Relative read-coordinate comparison slack.
    noise_factor, strip_floor_margin, strip_leakage : float
        Shared noise allowance, strip floor multiplier and Kaiser leakage share.
    has_design, has_collar : bool
        Whether a ramp design and requested real-space collar exist.
    """

    warn: float
    stop: float
    backstop_start: float
    backstop_max: float
    reach_growth: float
    guard_numerator: float
    k_min: float
    k_floor_fraction: float
    due_tolerance: float
    noise_factor: float
    strip_floor_margin: float
    strip_leakage: float
    has_design: bool
    has_collar: bool


@dataclass(frozen=True, eq=False)
class TransverseLineTransfer:
    """Own one collar line's selected nodes and transform conventions.

    Attributes
    ----------
    axis : str
        Physical normal axis, ``r``, ``x`` or ``y``.
    sample_scale : numpy.ndarray
        Axis scaling, shape ``(n_nodes,)``: square-root radius radially, ones otherwise.
    strip, window : numpy.ndarray
        One-based outward node indices and Kaiser weights, shape ``(n_strip,)``.
    spacing, measure : float
        Mean strip spacing in metres and transverse integration measure.
    inward : tuple of int
        One-based inclusive inward FFT bins; ``(1,0)`` denotes no selected bin.
    onset : tuple of int or None
        One-based onset, before and after nodes, or absent without both neighbours.
    onset_step : float
        Outward spacing of the neighbours in metres.
    """

    axis: str
    sample_scale: Float[np.ndarray, " n_nodes"]
    strip: Int[np.ndarray, " n_strip"]
    window: Float[np.ndarray, " n_strip"]
    spacing: float
    measure: float
    inward: tuple[int, int]
    onset: tuple[int, int, int] | None
    onset_step: float


@dataclass(frozen=True, eq=False)
class TransverseTransfer:
    """Own the geometry and numerical policy of one scalar spatial edge.

    Attributes
    ----------
    policy : TransversePolicyTransfer
        Numerical thresholds, without any measured state.
    faces, clip, totals : numpy.ndarray
        Node weights, shape ``(n_nodes,4)``, with unused face columns zero.
    column_weights, kappa, ramp_rates : numpy.ndarray
        Spectral integration weights, radial wavenumbers in rad/m and ramp power
        rates in m^-1, each shape ``(n_columns,)``.
    order : numpy.ndarray
        One-based ascending-wavenumber permutation, shape ``(n_columns,)``.
    face_ranks : tuple of int
        One-based lexical tie ranks in the actual declared face order.
    noise_face_share : float
        First face's share of the total integration weight.
    noise_modes : int
        Positive spatial/frequency count in the shared noise allowance.
    rows : numpy.ndarray
        Ordinary reduction node mask, shape ``(n_nodes,)``.
    declared : bool
        Whether the input declares content above the transverse discard threshold.
    trace_chunk : int
        Positive execution bound for concurrently transformed collar traces.
    lines : tuple of TransverseLineTransfer
        Ordered collar lines, empty without a requested collar.
    """

    policy: TransversePolicyTransfer
    faces: Float[np.ndarray, "n_nodes 4"]
    clip: Float[np.ndarray, "n_nodes 4"]
    totals: Float[np.ndarray, "n_nodes 4"]
    column_weights: Float[np.ndarray, " n_columns"]
    kappa: Float[np.ndarray, " n_columns"]
    ramp_rates: Float[np.ndarray, " n_columns"]
    order: Int[np.ndarray, " n_columns"]
    face_ranks: tuple[int, ...]
    noise_face_share: float
    noise_modes: int
    rows: Bool[np.ndarray, " n_nodes"]
    declared: bool
    trace_chunk: int
    lines: tuple[TransverseLineTransfer, ...]


@dataclass(frozen=True, eq=False)
class TransverseFormat:
    """Retain host-only context for the shared edge's public messages.

    Attributes
    ----------
    space : solprop.grid.RadialGrid or FreeGrid
        Original immutable grid, used only by the shared size-recommendation formatter.
        This context never crosses into Julia and contains no mutable edge monitor.
    faces : tuple of str
        Declared face names in numerical order.
    length, stop : float
        Guide length in metres and the stop energy share used in messages.
    """

    space: RadialGrid | FreeGrid
    faces: tuple[str, ...]
    length: float
    stop: float


def transverse_format(edge: TransverseEdge | None) -> TransverseFormat | None:
    """Detach the descriptive context used to format an already decided edge event.

    Parameters
    ----------
    edge : solprop.transverse_monitor.TransverseEdge or None
        Requested fresh edge, or absent for a modal/unmonitored run.

    Returns
    -------
    TransverseFormat or None
        Host-only immutable formatting context without measured state.

    Raises
    ------
    BackendError
        For the unqualified planar route.
    """
    if edge is None:
        return None
    if not isinstance(edge.space, RadialGrid | FreeGrid):
        raise BackendError("julia transverse monitoring refuses planar geometry")
    faces = edge._faces  # pyright: ignore[reportPrivateUsage] - shared geometry
    return TransverseFormat(
        edge.space,
        tuple(face.name for face in faces),
        float(edge.length),
        (edge.stop_factor * edge.budget) ** 2,
    )


def lower_transverse(
    problem: PropagationProblem, edge: TransverseEdge, trace_chunk: int
) -> TransverseTransfer:
    """Copy an admitted edge's numerical geometry before allocating Julia state.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Validated scalar radial, Cartesian or line descriptor.
    edge : solprop.transverse_monitor.TransverseEdge
        Shared geometry and policy; existing measured state is deliberately ignored.
    trace_chunk : int
        Positive execution bound, independent of scientific identity.

    Returns
    -------
    TransverseTransfer
        Owned, read-only arrays with Julia indexing and the shared expression order.

    Raises
    ------
    BackendError
        For an unsupported route or nonpositive trace bound.

    Notes
    -----
    Thresholds follow SEP 40 section 5.3 and ``TransverseEdge``. The strip floor
    multiplier three and Kaiser leakage share 1e-6 are the shared return policy;
    reductions, scheduling, noise baselines and all decisions remain Julia-owned.
    """
    if not isinstance(problem.route, SpatialRoute) or trace_chunk < 1:
        raise BackendError(
            "julia transverse monitoring needs a spatial route and chunk"
        )
    context = transverse_format(edge)
    assert context is not None
    faces = edge._faces  # pyright: ignore[reportPrivateUsage] - shared geometry
    weights = edge._weights  # pyright: ignore[reportPrivateUsage] - shared weights
    columns = edge._columns  # pyright: ignore[reportPrivateUsage] - shared weights
    kappa = edge._kappa  # pyright: ignore[reportPrivateUsage] - represented wavenumbers
    order = edge._kappa_order  # pyright: ignore[reportPrivateUsage] - shared tie order
    guard = edge._guard  # pyright: ignore[reportPrivateUsage] - shared guard width
    design, space = edge.design, context.space
    policy = TransversePolicyTransfer(
        edge.budget**2,
        context.stop,
        edge.length / READS,
        edge.length / BACKSTOP_READS,
        REACH_GROWTH,
        4.0 * guard,
        float(edge.k_min),
        RAMP_K_FLOOR,
        1e-12,
        NOISE_FACTOR,
        3.0,
        1e-6,
        design is not None,
        design is not None and design.has_collar,
    )
    lines: list[TransverseLineTransfer] = []
    for line in edge.reduction.lines:
        scale = (
            np.sqrt(np.asarray(space.r))
            if isinstance(space, RadialGrid)
            else np.ones(space.n_x if line.axis == "x" else space.n_y)
        )
        lines.append(
            TransverseLineTransfer(
                line.axis,
                owned(np.asarray(scale, dtype=np.float64)),
                owned(np.asarray(line.strip, dtype=np.int64) + 1),
                owned(np.asarray(line.window, dtype=np.float64)),
                float(line.spacing),
                float(line.measure),
                (line.inward[0] + 1, line.inward[1]),
                None
                if line.onset is None
                else (line.onset[0] + 1, line.onset[1] + 1, line.onset[2] + 1),
                float(line.onset_step),
            )
        )
    return TransverseTransfer(
        policy,
        owned(np.asarray(edge.tables.faces.T, dtype=np.float64)),
        owned(np.asarray(edge.tables.clip.T, dtype=np.float64)),
        owned(np.asarray(edge.tables.totals.T, dtype=np.float64)),
        owned(np.asarray(columns, dtype=np.float64)),
        owned(np.asarray(kappa, dtype=np.float64)),
        owned(np.zeros_like(kappa) if design is None else design.ramp_rate),
        owned(np.asarray(order, dtype=np.int64) + 1),
        tuple(sorted(context.faces).index(name) + 1 for name in context.faces),
        float(np.sum(weights[faces[0].edge])) / float(np.sum(weights)),
        max(int(np.count_nonzero(faces[0].edge)), 1) * max(int(edge.band.size), 1),
        owned(np.asarray(edge.tables.rows, dtype=np.bool_)),
        _declared_discard(problem),
        trace_chunk,
        tuple(lines),
    )
