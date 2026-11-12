# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The boundary monitor's transverse edge (SEP 40 section 5.3).

A free-space run has a wall or a wrap at its grid's outer size. With no real-space
collar, whatever reaches the edge comes back, so reaching it is the failure: the
monitor judges, per face, the energy in the outer 5 % of the *outer* size, its largest
reading above its launch value, less a noise allowance, at the levels of
:class:`solprop.monitor.BoundaryCheck`. With a collar, reaching the edge is intended and
what returns is the failure; the outward shares are recorded at INFO and the collar's
removal is credited.

Reads are spaced so that light cannot cross the edge unseen between two of them,
``min(L/16, 4 g / v_q)`` with ``g`` the guard's width and ``v_q`` the reach of the
current spectrum, and the spacing is a ceiling on the step. The safety ramp's removal
is credited per read interval and judged above the realised input noise's own share.
Every run reports a ``transverse_status``; only ``clear`` is a clean verdict.

A backend reduces its committed spectral state where the state lives and hands the edge
an ``_EdgeReading``, numbers and never the field. ``_HostEdgeReduction`` is the
reference, on a host copy of the state, and everything else here is the same on every
backend: the faces, the weights, the spacing, the ledgers and the judgement (SEP 44). A
backend that does not read the edge leaves the run ``unwatched``, which it says before
the first step.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import scipy.fft
from jaxtyping import Bool, Complex, Float, Int

from solprop._process import thread_budget
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.modes.slab import PlanarGrid
from solprop.spatial_absorber import RAMP_K_FLOOR, SpatialAbsorberDesign

__all__ = [
    "BACKSTOP_READS",
    "GUARD",
    "NOISE_FACTOR",
    "REACH_GROWTH",
    "READS",
    "TransverseEdge",
    "TransverseStatus",
]

logger = logging.getLogger(__name__)

GUARD = 0.05
"""The edge's width as a fraction of the outer size on its axis (SEP 40 section 5.3)."""

READS = 16
"""The read spacing's backstop starts at the guide's length over this many."""

BACKSTOP_READS = 4
"""The backstop grows to the guide's length over this many, no further (issue #396)."""

REACH_GROWTH = 0.1
"""The relative growth of the reach that starts the backstop again.

The reach moves by one transverse column between reads on a steady spectrum (5 % on
the ``line`` case's 1-D grid, whose ``kappa_q`` steps between neighbouring columns
read to read); a tenth leaves that jitter alone."""

NOISE_FACTOR = 8.0
"""Standard deviations of one noise reading the allowance takes (section 5.3)."""

_SMALL_TRANSFORM = 2**18
"""Host transforms below this many elements run on one thread."""

TransverseStatus = Literal[
    "clear", "warned", "stopped", "insensitive", "coarse", "unwatched", "off"
]

# What the edge reports, one text for every owner of its decisions: the edge here and a
# backend whose session judges natively (SEP 47 section 6.4).

_UNWATCHED_REASON = "this backend does not read the transverse edge"
"""The ``transverse_reason`` of a run whose backend does not read the edge."""

_UNWATCHED_MESSAGE = (
    "this backend does not read the transverse edge of a free-space run: "
    "the run is unwatched there and cannot report a clean verdict; "
    "solprop's own backends read it (SEP 40 section 5.3)"
)
"""The warning such a run issues before its first step."""


def _ramp_band_message(share: float, *, stop: bool) -> str:
    """Return what a launch's share in the safety ramp's band reports.

    - ``stop``: whether it is the stop's message, else the warning's.
    """
    if stop:
        return (
            f"the launch holds {share:.2e} of its energy in the safety ramp's "
            "band, which the ramp would erase: refine the transverse grid, launch "
            "with less tilt, declare the content with "
            "SpatialAbsorber(discard_above=...), or remove the ramp with "
            "spatial_absorber=False (SEP 40 section 5.3)"
        )
    return (
        f"the launch holds {share:.2e} of its energy in the safety ramp's "
        "band, which the ramp removes as the run starts"
    )


def _clip_message(clipped: float, face: str) -> str:
    """Return the warning of a launch the aperture clips on a face."""
    return (
        f"the launch holds {clipped:.2e} of its energy in the outer "
        f"{GUARD:.0%} of the requested size on its {face} face: the "
        "aperture clips it"
    )


def _insensitive_reason(allowance: float) -> str:
    """Return the ``transverse_reason`` of an edge the input noise leaves insensitive.

    - ``allowance``: the noise's allowance, a share of the launch energy.
    """
    return f"the input noise's allowance {allowance:.2e} exceeds the warning"


def _insensitive_message(allowance: float) -> str:
    """Return the warning of an edge the input noise leaves insensitive."""
    return (
        "the transverse edge cannot be judged below "
        f"{allowance:.2e} of the launch energy under this input "
        "noise (SEP 40 section 5.3)"
    )


def _coarse_message(longest: float, spacing: float) -> str:
    """Return the warning of a fixed schedule whose steps outrun the read spacing.

    - ``longest``: the schedule's longest step, m.
    - ``spacing``: the read spacing at the launch, m.
    """
    return (
        f"the fixed schedule's longest step, {longest:.3e} m, is "
        f"longer than the transverse edge's read spacing, {spacing:.3e} m: "
        "light could cross the edge unseen between two reads; the run is "
        "reported as coarse (SEP 40 section 5.3)"
    )


def _face_message(face: str, z: float, judged: float) -> str:
    """Return the warning of light first judged above the warning on a face.

    - ``z``: where, m.
    """
    return (
        f"light reached the transverse edge on its {face} face at "
        f"z = {z:.4e} m: {judged:.2e} of the launch energy above its "
        "launch value, which the wall or the wrap returns (SEP 40 "
        "section 5.3)"
    )


def _stop_message(
    space: RadialGrid | FreeGrid | PlanarGrid,
    length: float,
    stop: float,
    face: str,
    judged: float,
    *,
    z: float,
    warned_at: float,
) -> str:
    """Return a face's stop message, naming a size and a sample count to try.

    - ``length``: the guide's length, m.
    - ``z``: where it stopped, m.
    - ``warned_at``: where the edge first warned, m; NaN where it did not.
    """
    first = warned_at if math.isfinite(warned_at) else z
    factor = length / max(first, 1e-300)
    if isinstance(space, RadialGrid):
        size = f"a radius of at least {factor * space.outer_radius:.3g} m"
        count = f"n_r of at least {math.ceil(factor * space.n_r)}"
    elif isinstance(space, PlanarGrid):
        outer = 0.5 * space.n_y * float(space.y[1] - space.y[0])
        size = f"a y half-width of at least {factor * outer:.3g} m"
        count = f"n_y of at least {math.ceil(factor * space.n_y)}"
    else:
        outer = 0.5 * space.n_x * float(space.x[1] - space.x[0])
        size = f"an x half-width of at least {factor * outer:.3g} m"
        count = f"n_x of at least {math.ceil(factor * space.n_x)}"
    return (
        f"light reached the transverse edge on its {face} face at z = {z:.4e} m: "
        f"{judged:.2e} of the launch energy above its launch value, past the "
        f"stop of {stop:.0e}; the wall or the "
        f"wrap returns it into the aperture. Try {size}, at the same spacing "
        f"({count}); this is a lower bound, and light generated late may need "
        "more. A user who knows better raises BoundaryCheck.transverse_budget"
    )


def _return_message(returned: float, z: float, stop: float | None) -> str:
    """Return what a requested collar's return reports.

    - ``z``: where, m.
    - ``stop``: the stop, for the stop's message; ``None`` for the warning's.
    """
    if stop is not None:
        return (
            f"the requested collar returns {returned:.2e} of the launch energy at "
            f"z = {z:.4e} m, past the stop of {stop:.0e} (SEP 40 section 5.3): "
            "it is absorbing too little of what reaches it; widen it, or give the "
            "run more room"
        )
    return (
        f"the requested collar returns {returned:.2e} of the launch energy at "
        f"z = {z:.4e} m (SEP 40 section 5.3)"
    )


def _ramp_removal_message(removed: float, stop: float | None) -> str:
    """Return what the safety ramp's removal reports.

    - ``stop``: the stop, for the stop's message; ``None`` for the warning's.
    """
    if stop is not None:
        return (
            f"the safety ramp has removed {removed:.2e} of the launch energy "
            f"above the input noise's share, past the stop of {stop:.0e}: "
            "content reached the grid's outer transverse wavenumbers, which the "
            "grid cannot resolve; refine the transverse grid (SEP 40 section 5.3)"
        )
    return (
        f"the safety ramp has removed {removed:.2e} of the launch energy above "
        "the input noise's share: content reached the grid's outer "
        "wavenumbers; refine the transverse grid (SEP 40 section 5.3)"
    )


def _log_declared(share: float) -> None:
    """Log a declared launch's share in the ramp's band, which is not judged."""
    logger.info("the launch holds %.2e of its energy in the ramp's band", share)


def _log_finish(status: str, largest: float, removed: float, reads: int) -> None:
    """Log the edge's verdict at the end of a run."""
    logger.info(
        "transverse edge: %s; largest judged share %.2e; the ramp removed %.2e; "
        "%d reads",
        status,
        largest,
        removed,
        reads,
    )


@dataclass(frozen=True, eq=False)
class _EdgeTables:
    """The weights a state's node energies are summed with, region by region.

    Each table holds four regions' weights on the nodes (or rows), so that a backend
    needs no faces, masks or geometry of its own: it forms ``table @ nodes``, where
    ``nodes`` is each node's unweighted ``sum |E|^2`` over polarisation and frequency,
    and returns the numbers. A region no face uses is zero.

    Attributes
    ----------
    faces : numpy.ndarray
        Each face's edge, in the outer :data:`GUARD` of the outer size.
    clip : numpy.ndarray
        Each face's clip, in the outer :data:`GUARD` of the requested size: a launch's.
    totals : numpy.ndarray
        The whole grid; a requested collar's rate over it, 1/m; and what lies outside
        the request, for the launch's baseline of the current.
    rows : numpy.ndarray
        Boolean mask of the nodes (or rows) a read's sums can touch.
    """

    faces: Float[np.ndarray, "4 n_nodes"]
    clip: Float[np.ndarray, "4 n_nodes"]
    totals: Float[np.ndarray, "4 n_nodes"]
    rows: Bool[np.ndarray, " n_nodes"]


@dataclass(frozen=True, eq=False)
class _EdgeReading:
    """What a backend reads of its state for the edge: numbers, never the field.

    Attributes
    ----------
    faces : numpy.ndarray or None
        ``_EdgeTables.faces @ nodes``, shape ``(4,)``; ``None`` when only the current
        was asked for.
    clip : numpy.ndarray or None
        ``_EdgeTables.clip @ nodes``, for a launch.
    totals : numpy.ndarray or None
        ``_EdgeTables.totals @ nodes``, for a launch or a collar's read. A collar's read
        uses only the rate, ``[1]``: over the rows a read touches, which hold all of it,
        the other three can be a different sum on different backends.
    columns : numpy.ndarray or None
        Each spectral column's ``sum |E|^2``, unweighted, shape ``(n_columns,)``;
        ``None`` when only the current was asked for.
    strip : numpy.ndarray or None
        A requested collar's return strip, ``(inward, windowed)`` energy per line along
        a face's normal, shape ``(n_lines, 2)``, unweighted by the launch; ``None``
        without a collar, or when only the current was asked for.
    current : float or None
        The rectified inward current across the collar's onset, unweighted by the
        launch; ``None`` when it was not asked for.
    """

    faces: Float[np.ndarray, " 4"] | None = None
    clip: Float[np.ndarray, " 4"] | None = None
    totals: Float[np.ndarray, " 4"] | None = None
    columns: Float[np.ndarray, " n_columns"] | None = None
    strip: Float[np.ndarray, "n_lines 2"] | None = None
    current: float | None = None


@dataclass(frozen=True)
class _Face:
    """One face of the transverse edge.

    Attributes
    ----------
    name : str
        ``"r"``, ``"x-"``, ``"x+"``, ``"y-"`` or ``"y+"``.
    edge : numpy.ndarray
        Nodes (or rows) in the outer :data:`GUARD` of the outer size.
    clip : numpy.ndarray
        Nodes in the outer :data:`GUARD` of the requested size, for the launch.
    """

    name: str
    edge: np.ndarray
    clip: np.ndarray


def _row_energy(rows: np.ndarray) -> Float[np.ndarray, " n_rows"]:
    """Return each row's ``sum |E|^2`` of a two-dimensional complex array.

    Double precision is summed as a real view by ``einsum``, without the temporaries
    of ``abs(E)**2``; a read makes these sums many times a run (section 7.7).
    """
    if rows.dtype != np.complex128:
        return np.sum(np.abs(rows) ** 2, axis=1)
    flat = np.ascontiguousarray(rows).view(np.float64)
    return np.einsum("ij,ij->i", flat, flat)


def _axis_faces(
    name: str, axis: np.ndarray, size: float, flat: np.ndarray
) -> tuple[_Face, _Face]:
    """Return an axis's low and high faces.

    - ``name``: the axis, ``"x"`` or ``"y"``.
    - ``axis``: its centred samples, m.
    - ``size``: its requested half-size, m.
    - ``flat``: each node's coordinate along it, m.
    """
    outer = 0.5 * axis.size * float(axis[1] - axis[0])
    low = _Face(
        f"{name}-",
        flat <= -(1 - GUARD) * outer,
        (flat <= -(1 - GUARD) * size) & (flat >= -size),
    )
    high = _Face(
        f"{name}+",
        flat >= (1 - GUARD) * outer - 0.5 * float(axis[1] - axis[0]),
        (flat >= (1 - GUARD) * size) & (flat <= size),
    )
    return low, high


def _faces(space: RadialGrid | FreeGrid | PlanarGrid) -> tuple[_Face, ...]:
    """Return the faces of a grid's edge on its physical nodes or rows."""
    if isinstance(space, RadialGrid):
        r = np.asarray(space.r)
        return (
            _Face(
                "r",
                r >= (1 - GUARD) * space.outer_radius,
                (r >= (1 - GUARD) * space.radius) & (r <= space.radius),
            ),
        )
    if isinstance(space, PlanarGrid):
        y = np.asarray(space.y)
        return _axis_faces("y", y, space.radius_y, y)
    x = np.tile(np.asarray(space.x), space.n_y)
    faces = _axis_faces("x", np.asarray(space.x), space.radius_x, x)
    if isinstance(space, LineGrid):
        return faces
    y = np.repeat(np.asarray(space.y), space.n_x)
    return faces + _axis_faces("y", np.asarray(space.y), space.radius_y, y)


def _guard_width(space: RadialGrid | FreeGrid | PlanarGrid) -> float:
    """Return the narrowest face's guard width, m."""
    if isinstance(space, RadialGrid):
        return GUARD * space.outer_radius
    if isinstance(space, PlanarGrid):
        return GUARD * 0.5 * space.n_y * float(space.y[1] - space.y[0])
    widths = [0.5 * space.n_x * float(space.x[1] - space.x[0])]
    if not isinstance(space, LineGrid):
        widths.append(0.5 * space.n_y * float(space.y[1] - space.y[0]))
    return GUARD * min(widths)


@dataclass(frozen=True, eq=False)
class _EdgeLine:
    """A face's outward normal, as a requested collar's reductions walk it.

    One description for the host's reduction and native's, so that they differ only in
    their arithmetic. Python chooses every node, window and bin here, and native does no
    tie-breaking of its own.

    Attributes
    ----------
    axis : {"r", "x", "y"}
        The grid axis the line runs along.
    measure : float
        What a row's sum is multiplied by: the cross-section of one row.
    strip : numpy.ndarray
        The axis's nodes in the return strip, by increasing outward coordinate; empty
        when there are too few of them to resolve a direction.
    window : numpy.ndarray
        The Kaiser(8) window on those nodes.
    spacing : float
        Their mean spacing, m.
    inward : tuple of int
        The padded transform's bins ``[first, last)``, inward at ``|q| >= 16 / s``.
    onset : tuple of int or None
        The axis nodes at the requested size and either side of it, in outward order;
        ``None`` when the node has no neighbour on one side.
    onset_step : float
        The outward distance between those two neighbours, m.
    """

    axis: str
    measure: float
    strip: Int[np.ndarray, " m"]
    window: Float[np.ndarray, " m"]
    spacing: float
    inward: tuple[int, int]
    onset: tuple[int, int, int] | None
    onset_step: float


def _requested_size(space: RadialGrid | FreeGrid | PlanarGrid) -> float:
    """Return the requested half-size on the narrowest axis, m."""
    if isinstance(space, RadialGrid):
        return space.radius
    if isinstance(space, PlanarGrid):
        return space.radius_y
    if isinstance(space, LineGrid):
        return space.radius_x
    return min(space.radius_x, space.radius_y)


def _edge_line(
    axis: str,
    s: Float[np.ndarray, " n"],
    measure: float,
    bounds: tuple[float, float],
    size: float,
) -> _EdgeLine:
    """Return one line's description, from its outward coordinate.

    - ``axis``: the grid axis.
    - ``s``: the outward coordinate of each of its nodes, m.
    - ``bounds``: the return strip, ``(a, b)`` in `s`, m.
    - ``size``: the requested half-size, m.
    """
    a, b = bounds
    selected = np.nonzero((s >= a) & (s <= b))[0]
    strip = np.empty(0, dtype=np.int64)
    window = np.empty(0)
    spacing = 0.0
    inward = (0, 0)
    if selected.size >= 4:
        strip = selected[np.argsort(s[selected])]
        coordinate = s[strip]
        spacing = float(np.mean(np.diff(coordinate)))
        u = 2 * (coordinate - a) / (b - a) - 1
        window = np.i0(8.0 * np.sqrt(np.clip(1 - u * u, 0.0, 1.0))) / np.i0(8.0)
        # solprop propagates as exp(-i k_z z) (the operator carries -i(k_z - k)), so
        # a component exp(i q s) travels towards smaller s: inward is q > 0.
        count = 2 * strip.size
        q = 2 * np.pi * np.fft.fftfreq(count, d=spacing)
        bins = np.nonzero(q >= 16.0 / (b - a))[0]
        if bins.size:
            inward = (int(bins[0]), int(bins[-1]) + 1)
    node = int(np.argmin(np.abs(s - size)))
    # A node without a neighbour on each side has no slope to read.
    order = np.argsort(s)
    position = int(np.nonzero(order == node)[0][0])
    onset = None
    onset_step = 0.0
    if 0 < position < s.size - 1:
        before, after = int(order[position - 1]), int(order[position + 1])
        onset = (node, before, after)
        onset_step = float(s[after] - s[before])
    return _EdgeLine(axis, measure, strip, window, spacing, inward, onset, onset_step)


def _edge_lines(
    space: RadialGrid | FreeGrid | PlanarGrid, design: SpatialAbsorberDesign | None
) -> tuple[_EdgeLine, ...]:
    """Return the lines a requested collar's strip and current run along.

    - Returns: one per direction along each axis; none without a collar.
    """
    if design is None or not design.has_collar:
        return ()
    size = _requested_size(space)
    width = min(design.width)
    bounds = (size - min(0.5 * width, 0.25 * size), size + width)
    if isinstance(space, RadialGrid):
        return (_edge_line("r", np.asarray(space.r), 2 * np.pi, bounds, size),)
    if isinstance(space, PlanarGrid):
        y = np.asarray(space.y)
        return (
            _edge_line("y", y, 1.0, bounds, size),
            _edge_line("y", -y, 1.0, bounds, size),
        )
    x = np.asarray(space.x)
    dy = 1.0 if isinstance(space, LineGrid) else float(space.y[1] - space.y[0])
    lines = [
        _edge_line("x", x, dy, bounds, size),
        _edge_line("x", -x, dy, bounds, size),
    ]
    if not isinstance(space, LineGrid):
        y = np.asarray(space.y)
        dx = float(space.x[1] - space.x[0])
        lines += [
            _edge_line("y", y, dx, bounds, size),
            _edge_line("y", -y, dx, bounds, size),
        ]
    return tuple(lines)


_STRIP_BATCH_BYTES = 2**28
"""The most a batch of a strip's padded transforms holds on the state's memory space."""


def _line_rows(
    space: RadialGrid | FreeGrid | PlanarGrid, line: _EdgeLine, n_pol: int, n_omega: int
) -> int:
    """Return how many rows of values a line carries."""
    planes = n_pol * n_omega
    if isinstance(space, RadialGrid):
        return planes
    if isinstance(space, PlanarGrid):
        return planes * space.n_p
    return planes * (space.n_y if line.axis == "x" else space.n_x)


def _strip_batch(line: _EdgeLine, rows: int) -> int:
    """Return how many of a line's rows one batch of its padded transform takes."""
    return min(rows, max(1, _STRIP_BATCH_BYTES // (16 * 2 * line.strip.size)))


def _transverse_edge_bytes(
    space: RadialGrid | FreeGrid | PlanarGrid,
    design: SpatialAbsorberDesign | None,
    n_pol: int,
    n_omega: int,
) -> int:
    """Return what a native read of the edge holds resident, from the grid alone.

    Three tables of four binary64 rows on the nodes and the columns' energies; radially
    the synthesis; and with a requested collar each line's nodes and window, and one
    padded buffer per distinct batch. A run reading no edge holds none of it.

    - Returns: the bytes.
    """
    columns = int(np.asarray(space.spectral_weights).size)
    if isinstance(space, RadialGrid):
        total = 8 * (12 * space.n_r + columns + space.n_r**2)
    else:
        nodes = space.n_y if isinstance(space, PlanarGrid) else space.n_y * space.n_x
        total = 8 * (12 * nodes + columns)
    lines = _edge_lines(space, design)
    if isinstance(space, RadialGrid) and lines:
        total += 8 * space.n_r  # the scale of the radial line's values
    # The strips run in turn, on one buffer as large as the largest batch needs.
    largest = 0
    for line in lines:
        total += 16 * line.strip.size
        if line.strip.size:
            rows = _line_rows(space, line, n_pol, n_omega)
            largest = max(largest, _strip_batch(line, rows) * 2 * line.strip.size)
    return total + 16 * largest


def _radial_inverse(space: RadialGrid) -> Float[np.ndarray, "n_r n_r"]:
    """Return the radial grid's synthesis from its spectral columns to its nodes."""
    hankel = space.hankel
    return (
        hankel.inverse_scale * (hankel.j1[:, None] / hankel.j1[None, :]) * hankel.matrix
    )


@dataclass
class _HostEdgeReduction:
    """The reference reductions of the edge, on a host copy of a state (SEP 44).

    A backend whose state lives elsewhere forms the same numbers there instead
    (``HostDriver._edge_reading``), and is held to these.

    Attributes
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The run's grid.
    tables : solprop.transverse_monitor._EdgeTables
        What a state's node energies are summed with.
    lines : tuple of solprop.transverse_monitor._EdgeLine
        A requested collar's lines; none without one.
    k_min : float
        The smallest wavenumber over the requested band, rad/m.
    """

    space: RadialGrid | FreeGrid | PlanarGrid
    tables: _EdgeTables
    lines: tuple[_EdgeLine, ...]
    k_min: float
    _inverse: np.ndarray | None = field(default=None, init=False)
    # The host's transverse transforms take the process's thread budget (ADR-052).
    _workers: int = field(default_factory=lambda: thread_budget().threads, init=False)

    def reading(
        self,
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
        *,
        faces: bool = True,
        current: bool = False,
        full: bool = False,
    ) -> _EdgeReading:
        """Return what the edge asked of a state.

        - ``full``: whether every node is wanted, as a launch wants, not only the read's
          rows.
        """
        rate = self.current(state) if current else None
        if not faces:
            return _EdgeReading(current=rate)
        nodes = self.node_power(state, full=full)
        collar = bool(self.lines)
        return _EdgeReading(
            faces=self.tables.faces @ nodes,
            clip=self.tables.clip @ nodes if full else None,
            totals=self.tables.totals @ nodes if full or collar else None,
            columns=self.column_power(state),
            strip=self.strip(state) if collar else None,
            current=rate,
        )

    def synthesis(self) -> Float[np.ndarray, "n_r n_r"] | None:
        """Return a radial grid's synthesis from columns to nodes, or ``None``.

        - Returns: the dense matrix; ``None`` on a grid whose transform is an FFT.
        """
        if not isinstance(self.space, RadialGrid):
            return None
        if self._inverse is None:
            self._inverse = _radial_inverse(self.space)
        return self._inverse

    def node_power(
        self,
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
        *,
        full: bool = True,
    ) -> Float[np.ndarray, " n_nodes"]:
        """Return each physical node's (or row's) energy, over pol and frequency.

        - ``full``: whether every node is wanted. Otherwise, radially, the rows a read
          doesn't use are not synthesised and are zero: a read happens many times a
          run, and the full synthesis is an ``n_r``-square product per frequency.
        """
        space = self.space
        flat = state.reshape(state.shape[0], -1)
        if isinstance(space, RadialGrid):
            if self._inverse is None:
                self._inverse = _radial_inverse(space)
            if full:
                return _row_energy(self._inverse @ flat)
            nodes = np.zeros(space.n_r)
            rows = self.tables.rows
            nodes[rows] = _row_energy(self._inverse[rows] @ flat)
            return nodes
        if isinstance(space, PlanarGrid):
            cube = flat.reshape(space.n_y, space.n_p, -1)
            physical = scipy.fft.ifft(cube, axis=0, workers=self._fft_workers(cube))
            physical = physical.reshape(space.n_y, -1)
        else:
            cube = flat.reshape(space.n_y, space.n_x, -1)
            workers = self._fft_workers(cube)
            physical = scipy.fft.ifftn(cube, axes=(0, 1), workers=workers)
            physical = physical.reshape(space.n_y * space.n_x, -1)
        if full:
            return _row_energy(physical)
        nodes = np.zeros(physical.shape[0])
        rows = self.tables.rows
        nodes[rows] = _row_energy(physical[rows])
        return nodes

    def _fft_workers(self, cube: np.ndarray) -> int:
        """Return the threads for one host transform of `cube`.

        A small transform is faster on one thread than on a pool woken beside the
        backend's own threads.
        """
        return self._workers if cube.size >= _SMALL_TRANSFORM else 1

    @staticmethod
    def column_power(
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
    ) -> Float[np.ndarray, " n_columns"]:
        """Return each spectral column's unweighted energy."""
        return _row_energy(state.reshape(state.shape[0], -1))

    def _axis_values(
        self, state: Complex[np.ndarray, "n_modes n_pol n_omega"]
    ) -> dict[str, np.ndarray]:
        """Return the field along each axis's nodes, line by line.

        - Returns: by axis, the field on its nodes, ``(n_nodes, n_rows)`` with one row
          per frequency, polarisation and transverse position (``sqrt(r) E`` radially).
        """
        space = self.space
        flat = state.reshape(state.shape[0], -1)
        if isinstance(space, RadialGrid):
            scale = np.sqrt(np.asarray(space.r))[:, None]
            return {"r": scale * (_radial_inverse(space) @ flat)}
        if isinstance(space, PlanarGrid):
            cube = np.fft.ifft(flat.reshape(space.n_y, space.n_p, -1), axis=0)
            return {"y": cube.reshape(space.n_y, -1)}
        cube = np.fft.ifftn(flat.reshape(space.n_y, space.n_x, -1), axes=(0, 1))
        values = {"x": np.moveaxis(cube, 1, 0).reshape(space.n_x, -1)}
        if not isinstance(space, LineGrid):
            values["y"] = cube.reshape(space.n_y, -1)
        return values

    def strip(
        self, state: Complex[np.ndarray, "n_modes n_pol n_omega"]
    ) -> Float[np.ndarray, "n_lines 2"]:
        """Return each line's resolved inward energy and windowed energy in the strip.

        A Kaiser(8)-windowed transform along each face's normal, padded twice; the
        inward energy is that at ``|q| >= 16 / s`` travelling inward, where the
        window's leakage is about ``1e-6`` (section 5.3).
        """
        values = self._axis_values(state)
        found = []
        for line in self.lines:
            if not line.strip.size:
                found.append((0.0, 0.0))
                continue
            rows = line.window[:, None] * values[line.axis][line.strip]
            count = 2 * line.strip.size
            spectrum = (
                np.abs(np.fft.fft(rows, count, axis=0)) ** 2 * line.spacing / count
            )
            first, last = line.inward
            found.append(
                (
                    float(np.sum(spectrum[first:last])) * line.measure,
                    float(np.sum(spectrum)) * line.measure,
                )
            )
        return np.array(found, dtype=np.float64).reshape(-1, 2)

    def current(self, state: Complex[np.ndarray, "n_modes n_pol n_omega"]) -> float:
        """Return the inward energy current across the collar's onset, rectified.

        ``J = Im(E* dE/dn) / k`` on the requested size's boundary, summed over its
        frequencies and lines, per metre of propagation, counting inward flow only
        (section 5.3).
        """
        values = self._axis_values(state)
        total = 0.0
        for line in self.lines:
            if line.onset is None:
                continue
            node, before, after = line.onset
            along = values[line.axis]
            slope = (along[after] - along[before]) / line.onset_step
            # Under exp(-i k_z z), Im(E* dE/ds) / k is the flux towards smaller s.
            flux = np.imag(np.conj(along[node]) * slope) / self.k_min
            total += float(np.sum(np.maximum(flux, 0.0))) * line.measure
        return total


@dataclass
class TransverseEdge:
    """The transverse edge of one free-space run: geometry, totals and record.

    Attributes
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The run's grid.
    design : solprop.spatial_absorber.SpatialAbsorberDesign or None
        Its transverse boundary; ``None`` for ``spatial_absorber=False``.
    budget : float
        The warning level as a field amplitude: energy shares above its square warn.
    stop_factor : float
        Budgets above which a share stops the run.
    length : float
        The guide's length, m.
    band : numpy.ndarray
        The requested band's bins on the run's frequency axis.
    k_min : float
        The smallest wavenumber over the requested band, rad/m.
    """

    space: RadialGrid | FreeGrid | PlanarGrid
    design: SpatialAbsorberDesign | None
    budget: float
    stop_factor: float
    length: float
    band: Int[np.ndarray, " n_band"]
    k_min: float
    _faces: tuple[_Face, ...] = field(init=False)
    _weights: np.ndarray = field(init=False)
    _columns: np.ndarray = field(init=False)
    _kappa: np.ndarray = field(init=False)
    _guard: float = field(init=False)
    _launch: float = field(default=math.nan, init=False)
    _baseline: list[float] = field(default_factory=list, init=False)
    _peak: list[float] = field(default_factory=list, init=False)
    _peak_raw: list[float] = field(default_factory=list, init=False)
    _allowance: float = field(default=0.0, init=False)
    _noise_columns: np.ndarray | None = field(default=None, init=False)
    _noise_share_in_ramp: float = field(default=0.0, init=False)
    _ramp_raw: float = field(default=0.0, init=False)
    _ramp_noise: float = field(default=0.0, init=False)
    _spatial_removed: float = field(default=0.0, init=False)
    _last_rate: tuple[float, float] | None = field(default=None, init=False)
    _last_columns: np.ndarray | None = field(default=None, init=False)
    _last_z: float = field(default=math.nan, init=False)
    _next: float = field(default=math.inf, init=False)
    _spacing_now: float = field(default=math.inf, init=False)
    _backstop: float = field(default=math.inf, init=False)
    _reach_reference: float = field(default=0.0, init=False)
    _reads: int = field(default=0, init=False)
    _largest_gap: float = field(default=0.0, init=False)
    _warned_at: float = field(default=math.nan, init=False)
    _status: TransverseStatus = field(default="clear", init=False)
    _reason: str = field(default="", init=False)
    _warned: bool = field(default=False, init=False)
    _ramp_warned: bool = field(default=False, init=False)
    _resumed: bool = field(default=False, init=False)
    _declared_share: float = field(default=0.0, init=False)
    _strip_baseline: float = field(default=0.0, init=False)
    _strip_floor: float = field(default=0.0, init=False)
    _strip_peak: float = field(default=0.0, init=False)
    _current: float = field(default=0.0, init=False)
    _current_last: tuple[float, float] | None = field(default=None, init=False)
    _current_baseline: float = field(default=0.0, init=False)
    _returned_peak: float = field(default=0.0, init=False)
    _kappa_order: np.ndarray = field(init=False)
    tables: _EdgeTables = field(init=False)
    reduction: _HostEdgeReduction = field(init=False)

    def __post_init__(self) -> None:
        """Build the faces, the node weights and the column wavenumbers."""
        space = self.space
        self._faces = _faces(space)
        self._guard = _guard_width(space)
        if isinstance(space, RadialGrid):
            self._weights = np.asarray(space.radial_weights, dtype=np.float64)
            self._kappa = np.asarray(space.k_r, dtype=np.float64)
        elif isinstance(space, PlanarGrid):
            self._weights = np.full(space.n_y, float(space.y[1] - space.y[0]))
            self._kappa = np.abs(np.repeat(np.asarray(space.k_y), space.n_p))
        else:
            self._weights = np.asarray(space.spatial_weights, dtype=np.float64)
            self._kappa = np.sqrt(np.asarray(space.kperp_sq, dtype=np.float64))
        self._kappa_order = np.argsort(self._kappa)
        self._columns = np.asarray(space.spectral_weights, dtype=np.float64)
        self._peak = [0.0] * len(self._faces)
        self._peak_raw = [0.0] * len(self._faces)
        self._baseline = [0.0] * len(self._faces)
        self.tables = self._edge_tables()
        self.reduction = _HostEdgeReduction(
            space, self.tables, _edge_lines(space, self.design), self.k_min
        )

    def _edge_tables(self) -> _EdgeTables:
        """Return the weights a state's node energies are summed with."""
        weights = self._weights
        faces, clip = np.zeros((2, 4, weights.size))
        for index, face in enumerate(self._faces):
            faces[index] = weights * face.edge
            clip[index] = weights * face.clip
        totals = np.zeros((4, weights.size))
        totals[0] = weights
        rate = self._node_rate()
        if rate is not None:
            totals[1] = rate * weights
        if self.design is not None and self.design.has_collar:
            outside = np.ones(weights.size, dtype=bool)
            for face in self._faces:
                outside &= ~face.clip
            if isinstance(self.space, RadialGrid):
                outside = np.asarray(self.space.r) > _requested_size(self.space)
            totals[2] = weights * outside
        return _EdgeTables(faces, clip, totals, self._rows_read())

    def _rows_read(self) -> np.ndarray:
        """Return the nodes (or rows) a read uses: the faces' and a collar's."""
        rows = np.zeros(self._weights.size, dtype=bool)
        for face in self._faces:
            rows |= face.edge
        rate = self._node_rate()
        if rate is not None:
            rows |= rate > 0.0
        return rows

    def _node_rate(self) -> Float[np.ndarray, " n_nodes"] | None:
        """Return a requested collar's power rate on the nodes (or rows), 1/m.

        - Returns: the rate; ``None`` without a collar. On a slab it depends on y alone.
        """
        if self.design is None or self.design.collar_rate is None:
            return None
        rate = np.asarray(self.design.collar_rate)
        if isinstance(self.space, PlanarGrid):
            rate = rate.reshape(self.space.n_y, -1)[:, 0]
        return rate

    def _column_power(
        self, columns: Float[np.ndarray, " n_columns"]
    ) -> Float[np.ndarray, " n_columns"]:
        """Return each spectral column's energy, in the node reduction's units."""
        return self._columns * columns

    # ------------------------------------------------------------------ the record
    @property
    def resumed(self) -> bool:  # numpydoc ignore=RT01
        """Whether this edge continues an earlier execution's record.

        True after :meth:`resume` found a record, so that the launch is not judged.
        """
        return self._resumed

    def launch(  # noqa: PLR0912, PLR0915 - the launch and judgements, in order
        self,
        reading: _EdgeReading,
        z: float,
        *,
        noise: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
        clean_columns: Float[np.ndarray, " n_columns"] | None = None,
        steps: Float[np.ndarray, " n_steps"] | None,
        reads: bool,
        declared: bool,
    ) -> None:
        """Record the launch, judge it, and set the first read.

        Parameters
        ----------
        reading : solprop.transverse_monitor._EdgeReading
            The launched state's reading: every node, unless :attr:`resumed`.
        z : float
            Its position, m.
        noise : numpy.ndarray or None
            The input noise it carries, if any, which is always a host array.
        clean_columns : numpy.ndarray or None, optional
            Each column's unweighted energy without that noise, for the safety ramp's
            judgement. ``None`` when there is no noise, or the launch was resumed.
        steps : numpy.ndarray or None
            A fixed schedule's steps, which the read spacing cannot limit.
        reads : bool
            Whether the backend reads the edge.
        declared : bool
            Whether the user declared content above ``discard_above``.

        Raises
        ------
        BoundaryError
            When physical light launched into the safety ramp's band exceeds ten
            budgets (section 5.3).
        """
        assert reading.faces is not None and reading.columns is not None
        columns = reading.columns
        if self._resumed:
            # The launch was judged by the first execution; its baselines and totals
            # are restored, and the reads continue from the resumed state.
            self._last_z = z
            if noise is not None:
                self._noise_columns = self._column_power(
                    self.reduction.column_power(noise)
                )
            self._last_columns = self._column_power(columns)
            self._respace(self._last_columns, launch=True)
            self._next = z + self._spacing_now if reads else math.inf
            self._last_rate = (z, self._collar_rate(reading))
            if reading.current is not None and self._launch > 0.0:
                self._current_last = (z, reading.current / self._launch)
            if not reads:
                self._status = "unwatched"
            return
        assert reading.clip is not None and reading.totals is not None
        self._launch = float(reading.totals[0])
        self._last_z = z
        if not self._launch > 0.0:
            self._next = math.inf
            return
        warn, stop = self.budget**2, (self.stop_factor * self.budget) ** 2
        # About the launch, not the backend: judged whether or not the edge is read.
        self._judge_ramp_band(
            self._column_power(columns if clean_columns is None else clean_columns),
            stop,
            warn,
            declared=declared,
        )
        if not reads:
            self._status = "unwatched"
            self._reason = _UNWATCHED_REASON
            warnings.warn(_UNWATCHED_MESSAGE, BoundaryWarning, stacklevel=4)
            return
        for index, face in enumerate(self._faces):
            self._baseline[index] = float(reading.faces[index]) / self._launch
            clipped = float(reading.clip[index]) / self._launch
            if clipped > warn:
                warnings.warn(
                    _clip_message(clipped, face.name), BoundaryWarning, stacklevel=4
                )
        if noise is not None:
            noise_nodes = self._weights * self.reduction.node_power(noise)
            eta = float(np.sum(noise_nodes)) / self._launch
            share = float(np.sum(self._weights[self._faces[0].edge])) / float(
                np.sum(self._weights)
            )
            modes = max(int(np.count_nonzero(self._faces[0].edge)), 1) * max(
                int(self.band.size), 1
            )
            self._allowance = NOISE_FACTOR * eta * share / math.sqrt(modes)
            self._noise_columns = self._column_power(self.reduction.column_power(noise))
            if self.design is not None:
                ramp = self.design.ramp_rate > 0.0
                self._noise_share_in_ramp = (
                    float(np.sum(self._noise_columns[ramp])) / self._launch
                )
            if self._allowance > warn:
                self._status = "insensitive"
                self._reason = _insensitive_reason(self._allowance)
                warnings.warn(
                    _insensitive_message(self._allowance),
                    BoundaryWarning,
                    stacklevel=4,
                )
        self._last_columns = self._column_power(columns)
        self._respace(self._last_columns, launch=True)
        spacing = self._spacing_now
        self._next = z + spacing
        if steps is not None and steps.size and float(np.max(steps)) > spacing:
            self._status = "coarse" if self._status == "clear" else self._status
            self._largest_gap = float(np.max(steps))
            warnings.warn(
                _coarse_message(float(np.max(steps)), spacing),
                BoundaryWarning,
                stacklevel=4,
            )
        self._last_rate = (z, self._collar_rate(reading))
        if self.every_step:
            self._launch_return(reading, noise)

    def _judge_ramp_band(
        self,
        columns: Float[np.ndarray, " n_columns"],
        stop: float,
        warn: float,
        *,
        declared: bool,
    ) -> None:
        """Judge the noise-free launch's share in the safety ramp's band (5.3)."""
        if self.design is None:
            return
        total = float(np.sum(columns))
        share = (
            float(np.sum(columns[self.design.ramp_rate > 0.0])) / total
            if total > 0.0
            else 0.0
        )
        if declared:
            # Declared above discard_above: not wanted, so its removal is not judged.
            self._declared_share = share
            _log_declared(share)
            return
        if share > stop:
            raise BoundaryError(_ramp_band_message(share, stop=True))
        if share > warn:
            warnings.warn(
                _ramp_band_message(share, stop=False), BoundaryWarning, stacklevel=5
            )

    def _collar_rate(self, reading: _EdgeReading) -> float:
        """Return a requested collar's removal rate, as a share of the launch per m."""
        if self.design is None or self.design.collar_rate is None:
            return 0.0
        assert reading.totals is not None
        return float(reading.totals[1]) / self._launch

    def _reach(self, columns: Float[np.ndarray, " n_columns"]) -> float:
        """Return the reach ``v_q`` of the current spectrum.

        - Returns: the transverse group velocity at ``kappa_q``, the wavenumber above
          which the field holds no more than the warning's share of the launch; zero
          when nothing is above it.
        """
        if self._noise_columns is not None:
            columns = np.maximum(columns - self._noise_columns, 0.0)
        order = self._kappa_order
        tail = np.cumsum(columns[order][::-1])[::-1] / self._launch
        above = np.nonzero(tail > self.budget**2)[0]
        kappa_q = float(self._kappa[order][above[-1]]) if above.size else 0.0
        k = self.k_min
        k_z = math.sqrt(max(k * k - kappa_q * kappa_q, 0.0))
        return kappa_q / max(k_z, RAMP_K_FLOOR * k)

    def _respace(
        self, columns: Float[np.ndarray, " n_columns"], *, launch: bool
    ) -> None:
        """Set the read spacing for the current spectrum, ``min(B, 4 g / v_q)``.

        ``4 g / v_q`` bounds how far the fastest content measured can travel between
        two reads. The backstop ``B`` bounds the gap when the spectrum broadens after a
        read, which no reach measured at the read foresees. It starts at
        ``L / READS``, and doubles at each read whose reach has not grown by
        :data:`REACH_GROWTH` since the backstop last started, up to
        ``L / BACKSTOP_READS``; a read that finds the reach grown, or no reach at
        all, starts it again (solprop issue #396). Growth is measured against the
        reach at that start, not at the last read, so that a creeping spectrum cannot
        double it forever.
        """
        reach = self._reach(columns)
        # A spectrum with no reach (nothing above the warning, so only under a budget
        # of order one) gives no evidence that it holds: the backstop stays at its
        # start.
        grown = reach > self._reach_reference * (1.0 + REACH_GROWTH)
        if launch or grown or reach == 0.0:
            self._backstop = self.length / READS
            self._reach_reference = reach
        else:
            self._backstop = min(2.0 * self._backstop, self.length / BACKSTOP_READS)
        spacing = self._backstop
        if reach > 0.0:
            spacing = min(spacing, 4.0 * self._guard / reach)
        self._spacing_now = spacing

    def ceiling(self, z: float) -> float:
        """Return the longest step the reads allow from `z`: the read spacing.

        A step is not made to land on a read. It is capped at the spacing, and the edge
        is read at the last committed position from which the next step would carry
        past the next read (:meth:`due`), so reads stay at most a spacing apart without
        shortening the steps between them.

        Parameters
        ----------
        z : float
            The current position, m.

        Returns
        -------
        float
            The step, m; infinite when nothing is read.
        """
        del z  # the spacing does not depend on the position between reads
        if not math.isfinite(self._next):
            return math.inf
        return self._spacing_now

    def due(self, z: float, proposal: float = 0.0) -> bool:
        """Return whether the edge is read at the committed position `z`.

        Parameters
        ----------
        z : float
            The committed position, m.
        proposal : float, optional
            The next step proposed from `z`, m: the edge is read now when that step
            would carry past the next read.

        Returns
        -------
        bool
            True to call :meth:`read`.
        """
        return z + proposal >= self._next - 1e-12 * max(abs(z), 1.0)

    def wants_read(self, z: float) -> bool:
        """Return whether a read at `z` would record anything.

        A saved plane reads the edge whether or not the step that reached it already
        did, and a read at the position of the last one records nothing. Asking first
        spares the caller the state it would otherwise hand over for nothing, which on
        a device is a crossing of the whole state.

        Parameters
        ----------
        z : float
            The committed position, m.

        Returns
        -------
        bool
            False when nothing was launched, the edge is unwatched, or it was last
            read at `z`.
        """
        return self._launch > 0.0 and self._status != "unwatched" and z > self._last_z

    def read(self, reading: _EdgeReading, z: float) -> None:
        """Take a reading at a committed position, credit the removals and judge.

        Parameters
        ----------
        reading : solprop.transverse_monitor._EdgeReading
            The state's reading.
        z : float
            Its position, m.

        Raises
        ------
        BoundaryError
            When a face's judged share exceeds ``stop_factor`` budgets.
        """
        if not self.wants_read(z):
            return
        assert reading.faces is not None and reading.columns is not None
        gap = z - self._last_z
        self._reads += 1
        self._largest_gap = max(self._largest_gap, gap)
        # The ramp's removal over the interval, from each column's power at its start.
        if self.design is not None and self._last_columns is not None:
            taken = -np.expm1(-self.design.ramp_rate * gap)
            self._ramp_raw += float(np.sum(self._last_columns * taken)) / self._launch
            if self._noise_columns is not None:
                self._ramp_noise += float(np.sum(self._noise_columns * taken)) / (
                    self._launch
                )
        rate = self._collar_rate(reading)
        # Every launch that leaves the edge read sets the rate's first sample.
        assert self._last_rate is not None
        previous_z, previous = self._last_rate
        self._spatial_removed += 0.5 * (z - previous_z) * (previous + rate)
        self._last_rate = (z, rate)
        self._last_z = z
        self._last_columns = self._column_power(reading.columns)
        self._respace(self._last_columns, launch=False)
        self._next = z + self._spacing_now
        if self.design is not None and self.design.has_collar:
            # With a collar, outward flow is intended: recorded at INFO, never stopped.
            for index in range(len(self._faces)):
                share = float(reading.faces[index]) / self._launch
                self._peak_raw[index] = max(self._peak_raw[index], share)
            self._judge_return(reading, z)
            self._judge_ramp_removal()
            return
        warn, stop = self.budget**2, (self.stop_factor * self.budget) ** 2
        judgements = []
        for index, face in enumerate(self._faces):
            share = float(reading.faces[index]) / self._launch
            self._peak_raw[index] = max(self._peak_raw[index], share)
            judged = share - self._baseline[index] - self._allowance
            self._peak[index] = max(self._peak[index], judged)
            judgements.append((judged, face.name))
        worst, name = max(judgements)
        if worst > stop:
            self._status = "stopped"
            raise BoundaryError(self._stop_message(name, worst, z))
        for judged, face_name in judgements:
            if judged > warn and not self._warned:
                self._warned = True
                self._warned_at = z
                if self._status in ("clear", "coarse"):
                    self._status = "warned"
                warnings.warn(
                    _face_message(face_name, z, judged), BoundaryWarning, stacklevel=4
                )
        self._judge_ramp_removal()

    # ------------------------------------------------------------------ a collar
    def _strip_shares(
        self, strip: Float[np.ndarray, "n_lines 2"]
    ) -> tuple[float, float]:
        """Return the strip's resolved inward energy and its windowed energy.

        The larger line's, as shares of the launch (section 5.3).

        - Returns: the inward share and the windowed share.
        """
        best, windowed = 0.0, 0.0
        for inward, total in strip:
            share = float(inward) / self._launch
            if share > best:
                best, windowed = share, float(total) / self._launch
        return best, windowed

    def step(self, current: float, z: float) -> None:
        """Accumulate a collar's inward current at a committed step (section 5.3).

        Parameters
        ----------
        current : float
            The rectified current across the onset, unweighted by the launch.
        z : float
            Its position, m.
        """
        if (
            self.design is None
            or not self.design.has_collar
            or not self._launch > 0.0
            or self._status == "unwatched"
        ):
            return
        share = current / self._launch
        if self._current_last is not None:
            previous_z, previous = self._current_last
            self._current += 0.5 * (z - previous_z) * (previous + share)
        self._current_last = (z, share)

    @property
    def next_read(self) -> float:  # numpydoc ignore=RT01
        """Where the edge is next read, m; infinite when nothing is read.

        A backend that marches without the host, as the native Adams march does,
        holds its steps to it as :meth:`ceiling` does and hands the state back there.
        """
        return self._next

    @property
    def every_step(self) -> bool:  # numpydoc ignore=RT01
        """Whether the edge accumulates at every committed step (a collar).

        True with a requested collar.
        """
        return self.design is not None and self.design.has_collar

    def _launch_return(
        self,
        reading: _EdgeReading,
        noise: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
    ) -> None:
        """Record a collar's launch baselines and floors (section 5.3)."""
        assert reading.strip is not None
        strip, _ = self._strip_shares(reading.strip)
        self._strip_baseline = strip
        if noise is not None:
            self._strip_floor = 3.0 * self._strip_shares(self.reduction.strip(noise))[0]
        assert reading.totals is not None
        self._current_baseline = float(reading.totals[2]) / self._launch
        self._current_last = None
        self._current = 0.0

    def _judge_return(self, reading: _EdgeReading, z: float) -> None:
        """Judge what a collar returns: the strip and the current (section 5.3).

        - ``z``: its position, m.
        """
        assert reading.strip is not None
        strip, windowed = self._strip_shares(reading.strip)
        self._strip_peak = max(self._strip_peak, strip)
        judged_strip = max(
            strip - self._strip_baseline - self._strip_floor - 1e-6 * windowed, 0.0
        )
        judged_current = max(self._current - self._current_baseline, 0.0)
        returned = max(judged_strip, judged_current)
        self._returned_peak = max(self._returned_peak, returned)
        warn, stop = self.budget**2, (self.stop_factor * self.budget) ** 2
        if returned > stop:
            self._status = "stopped"
            raise BoundaryError(_return_message(returned, z, stop))
        if returned > warn and not self._warned:
            self._warned = True
            self._warned_at = z
            self._status = "warned"
            warnings.warn(
                _return_message(returned, z, None), BoundaryWarning, stacklevel=5
            )

    def _judge_ramp_removal(self) -> None:
        """Warn, or stop, on what the safety ramp has removed (section 5.3, Q8).

        The ramp's removal tracks the true error of an under-resolved run within a few
        times, and the ramp can hide that light from the edge; it is judged at the
        edge's levels (the maintainer's decision on Q8).
        """
        removed = self.ramp_removed
        stop = (self.stop_factor * self.budget) ** 2
        if removed > stop:
            self._status = "stopped"
            raise BoundaryError(_ramp_removal_message(removed, stop))
        if removed > self.budget**2 and not self._ramp_warned:
            self._ramp_warned = True
            # Only a run below the warning throughout is clear (section 5.3).
            if self._status in ("clear", "coarse"):
                self._status = "warned"
            warnings.warn(
                _ramp_removal_message(removed, None), BoundaryWarning, stacklevel=5
            )

    def _stop_message(self, face: str, judged: float, z: float) -> str:
        """Return the stop's message, naming a size and a sample count to try.

        - ``z``: where it stopped, m.
        """
        return _stop_message(
            self.space,
            self.length,
            (self.stop_factor * self.budget) ** 2,
            face,
            judged,
            z=z,
            warned_at=self._warned_at,
        )

    # ------------------------------------------------------------------ results
    @property
    def ramp_removed(self) -> float:  # numpydoc ignore=RT01
        """Safety ramp's removal above noise and declared content.

        A share of the launch energy, signed.
        """
        return (
            self._ramp_raw
            - min(self._ramp_noise, self._noise_share_in_ramp)
            - self._declared_share
        )

    @property
    def status(self) -> TransverseStatus:  # numpydoc ignore=RT01
        """The run's transverse status (section 5.3).

        ``clear``, ``warned``, ``stopped``, ``insensitive``, ``coarse`` or
        ``unwatched``.
        """
        return self._status

    def resume(self, record: dict[str, object]) -> None:
        """Continue an earlier execution's record of the edge (SEP 40 section 5.3).

        The launch baselines and the totals are restored, so a resume judges against
        the original launch; :meth:`launch` then records the resumed state unjudged.

        Parameters
        ----------
        record : dict
            The ``transverse`` entry of the earlier monitor record.
        """

        def _number(name: str, default: float = 0.0) -> float:
            """Return a recorded number, or `default`."""
            value = record.get(name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                return default
            return float(value)

        def _numbers(name: str) -> list[float] | None:
            """Return a recorded per-face list, or ``None`` if it does not fit."""
            value = record.get(name)
            if not isinstance(value, list) or len(value) != len(self._faces):
                return None
            return [float(cast("float", one)) for one in cast("list[object]", value)]

        launch = _number("launch_energy", math.nan)
        baseline = _numbers("edge_baseline")
        if not launch > 0.0 or baseline is None:
            return
        self._resumed = True
        self._launch = launch
        self._baseline = baseline
        self._peak = _numbers("edge_max") or self._peak
        self._peak_raw = _numbers("edge_max_raw") or self._peak_raw
        self._allowance = _number("noise_allowance")
        self._ramp_raw = _number("raw_ramp_removed")
        self._ramp_noise = _number("noise_ramp_removed")
        self._noise_share_in_ramp = _number("noise_share_in_ramp")
        self._declared_share = _number("declared_share_in_ramp")
        self._spatial_removed = _number("spatial_removed")
        self._current = _number("current")
        self._current_baseline = _number("current_baseline")
        self._strip_baseline = _number("strip_baseline")
        self._strip_floor = _number("strip_floor")
        self._strip_peak = _number("strip_max")
        self._returned_peak = _number("returned_max")
        self._reads = int(_number("reads"))
        self._largest_gap = _number("largest_gap")
        self._warned_at = _number("warned_at", math.nan)
        self._warned = math.isfinite(self._warned_at)
        status = record.get("transverse_status")
        if isinstance(status, str) and status in (
            "clear",
            "warned",
            "insensitive",
            "coarse",
        ):
            self._status = cast("TransverseStatus", status)

    def summary(self) -> dict[str, object]:
        """Return the edge's record for the monitor's.

        Returns
        -------
        dict
            JSON-ready values: the status and its reason, the faces, the launch
            energy and each face's launch baseline, largest judged and raw shares,
            the noise allowance, the ramp's removal (judged and raw) and a collar's,
            the reads and the largest gap between them, and where it first warned.
        """
        return {
            "launch_energy": self._launch if math.isfinite(self._launch) else None,
            "noise_ramp_removed": self._ramp_noise,
            "noise_share_in_ramp": self._noise_share_in_ramp,
            "declared_share_in_ramp": self._declared_share,
            "transverse_status": self._status,
            "transverse_reason": self._reason,
            "faces": [face.name for face in self._faces],
            "edge_baseline": list(self._baseline),
            "edge_max": list(self._peak),
            "edge_max_raw": list(self._peak_raw),
            "noise_allowance": self._allowance,
            "ramp_removed": self.ramp_removed,
            "raw_ramp_removed": self._ramp_raw,
            "spatial_removed": self._spatial_removed,
            "reads": self._reads,
            "returned_max": self._returned_peak,
            "strip_max": self._strip_peak,
            "strip_baseline": self._strip_baseline,
            "strip_floor": self._strip_floor,
            "current": self._current,
            "current_baseline": self._current_baseline,
            "largest_gap": self._largest_gap,
            "warned_at": None if math.isnan(self._warned_at) else self._warned_at,
        }

    def finish(self) -> None:
        """Log the edge's verdict at the end of a run."""
        _log_finish(
            self._status, max(self._peak, default=0.0), self.ramp_removed, self._reads
        )
