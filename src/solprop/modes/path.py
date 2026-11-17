# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sample a guide whose geometry moves along z, section by section, in its own gauge.

A solid guide's modes change *shape* along a taper, so the general route's synthesis
``S`` and projection ``Q`` move with z; this module samples that motion.
:func:`path_sections` cuts the path at its declared kinks; :func:`path_rule` and
:func:`path_snapshot` solve one position on the provider's own rule there and in its
own gauge; :func:`sample_modal_path` doubles the Chebyshev-Lobatto degree per section
(from :data:`INITIAL_PATH_DEGREE` to :data:`MAX_PATH_DEGREE`) until the expansion
reproduces held-out solves, returning a :class:`ModalPathTable`: the snapshots beside
the :class:`ModalPath` record of the sampling, which a guide keeps and a run rebuilds
the same snapshots from. What a run reads of it is built in
:mod:`solprop.translate.modal_path`.

Two invariants. The node rule **moves with the geometry**, so node ``q`` stays in one
material region and a map stored there is smooth in z; the connection's Eulerian
derivative keeps the union-split rule of
:func:`solprop.modes.connection.connection_nodes` instead. The gauge is **the
provider's own**, because the connection is measured in it, so a provider whose gauge
is discontinuous along the path is refused (:data:`GAUGE_PHASE_ATOL`). The full account
is ``docs/guide/waveguides/moving-modal.md``, section "A guide whose geometry moves
along z".
"""

import itertools
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Inexact
from numpy.polynomial import chebyshev as chebyshev_basis

from solprop._display import Displayed, engineering, line
from solprop._ownership import frozen_copy
from solprop.exceptions import ModeError
from solprop.modes.material import JsonValue, quantised_measurement
from solprop.modes.quadrature import (
    lobatto_coefficients,
    lobatto_points,
    refinement_points,
)
from solprop.modes.tables import (
    DEFAULT_GRAM_ATOL,
    MIN_OVERLAP_DOMINANCE,
    DenseModalTable,
    apply_channel_basis,
    cross_poynting,
)
from solprop.modes.vector import (
    ChannelBasis,
    ModalNodeRule,
    ModalProvider,
    ModalSample,
    flux_product,
)

__all__ = [
    "GAUGE_PHASE_ATOL",
    "INITIAL_PATH_DEGREE",
    "MAX_PATH_DEGREE",
    "PATH_RTOL_FLOOR",
    "ModalPath",
    "ModalPathTable",
    "path_rule",
    "path_sections",
    "path_snapshot",
    "sample_modal_path",
]

logger = logging.getLogger(__name__)

INITIAL_PATH_DEGREE = 4
"""Chebyshev degree along z a path's accuracy search starts at, per section.

Four is five solves per section before anything is measured. The search doubles from
here, and the Chebyshev-Lobatto points nest bitwise under a doubling, so no solve is
ever repeated (:func:`solprop.modes.quadrature.lobatto_points`).
"""

MAX_PATH_DEGREE = 32
"""Highest Chebyshev degree along z a path's accuracy search tries per section.

The same ceiling :data:`solprop.translate.coupled.MAX_SECTION_DEGREE` puts on a
coupled flow's sections, and for the same reason: a smooth section of a declared
taper resolves at a low degree, and a degree this high means the section spans
structure the provider should declare as a kink, or the caller should divide.
"""

PATH_RTOL_FLOOR = 1e-10
"""Smallest held-out error along z a path's accuracy search is asked to reach.

The default target is the frequency interpolation's own measured error, and a very
well sampled band can measure below what a provider's solves reproduce between two
positions: the fields at a node are a sum over a quadrature and an eigenproblem, and
their relative round-off is of order ``1e-13`` to ``1e-12``. Below this floor the
search would be doubling to resolve round-off.
"""

GAUGE_PHASE_ATOL = 1e-9
"""Largest phase, rad, between two neighbouring samples' aligned overlap.

The connection is measured in the provider's own gauge with its stencil aligned to
the centre, which removes any phase the gauge accumulates along z; the maps are
expanded in the same gauge **without** that alignment. They agree only where the
provider's gauge has no phase rate of its own, and the two qualified providers'
fields are real up to a fixed factor, so their overlaps are real to rounding. A
provider whose gauge drifts would put ``i phi'`` on the maps' diagonal and not on the
connection's, which is refused rather than propagated.
"""


def _count(name: str, value: object) -> int:
    """Return a path's positive integer declaration, refusing anything else by name."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModeError(f"a path's {name} is a positive integer, got {value!r}")
    return value


def _measure(name: str, value: float, *, positive: bool) -> float:
    """Return a path's finite budget or measurement, refusing anything else by name.

    - ``positive``: whether zero is refused too, as it is for a budget.
    """
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or (positive and number == 0.0):
        raise ModeError(
            f"a path's {name} is a {'positive' if positive else 'non-negative'} "
            f"finite number, got {value!r}"
        )
    return number


@dataclass(frozen=True, eq=False)
class ModalPath(Displayed):
    """Record how a moving guide was sampled along z, and what that measured.

    The z half of a guide's sampling (:class:`~solprop.modes.guide.ModalSampling`),
    filled by :meth:`~solprop.modes.guide.ModalGuide.from_provider` for a provider
    whose geometry moves. The path is cut into **sections** -- the provider's declared
    kink intervals, each divided `sections` times (:func:`path_sections`) -- and every
    section is solved at the Chebyshev-Lobatto points of one degree along z, the
    degree the accuracy check demanded: doubled until the expansion reproduced a fresh
    solve at the points a doubling adds to within `rtol` (:func:`sample_modal_path`,
    whose :class:`ModalPathTable` holds the snapshots beside this record). A run
    rebuilds the same positions from this record rather than searching again.

    Attributes
    ----------
    z_edges : numpy.ndarray
        Section edges in m, strictly increasing, shape ``(n_edges_path,)``: the
        provider's z domain split at its interior breakpoints and subdivided.
    degree : int
        The Chebyshev degree along z every section is sampled at, positive; each
        section holds ``degree + 1`` positions, its two ends included.
    sections : int
        How many sections each kink interval was divided into, positive.
    rtol : float
        The budget the degree was searched against, positive: by default the
        frequency interpolation's own measured profile error, so that the sampling in
        z is never the coarser of the two.
    z_error : float
        The measured held-out departure at the degree kept, dimensionless and
        non-negative: the largest relative maximum-norm departure of E, H or the
        generator from a fresh solve between the sampled positions.
    gram_departure : float
        The largest ``norm(W - I, 2)`` of the expanded basis at those same held-out
        positions, dimensionless and non-negative.

    Raises
    ------
    ModeError
        Naming the field that is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.path import ModalPath
    >>> path = ModalPath(np.array([0.0, 1e-3]), 2, 1, 1e-6, 3e-7, 1e-8)
    >>> path.positions.round(6).tolist()
    [[0.0, 0.0005, 0.001]]
    >>> round(float(path.interpolate(np.array([0.0, 1.0, 2.0]), 2.5e-4)), 12)
    0.5
    """

    z_edges: Float[np.ndarray, " n_edges_path"]
    degree: int
    sections: int
    rtol: float
    z_error: float
    gram_departure: float

    def __post_init__(self) -> None:
        """Freeze the edges and check every declaration and measurement.

        Raises
        ------
        ModeError
            Naming the field that is out of range.
        """
        edges = frozen_copy(np.reshape(self.z_edges, -1), np.float64)
        if (
            edges.size < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ModeError(
                "a path's section edges are at least two finite, strictly increasing "
                f"positions, got {edges.tolist()}"
            )
        object.__setattr__(self, "z_edges", edges)
        _count("degree", self.degree)
        _count("sections", self.sections)
        _measure("rtol", self.rtol, positive=True)
        _measure("z_error", self.z_error, positive=False)
        _measure("gram_departure", self.gram_departure, positive=False)

    @property
    def n_intervals(self) -> int:  # numpydoc ignore=RT01
        """How many sections the path is cut into."""
        return int(self.z_edges.size - 1)

    @property
    def positions(
        self,
    ) -> Float[np.ndarray, "n_intervals_path n_deg_path"]:  # numpydoc ignore=RT01
        """Every sampled position, section by section, m.

        Shape ``(n_intervals, degree + 1)``, increasing inside a section, so a position
        on an interior edge appears twice.
        """
        # The same Chebyshev-Lobatto points the path was sampled at, bit for bit,
        # because the run solves exactly these positions again.
        return np.stack(
            [
                lobatto_points(self.degree, float(lo), float(hi))
                for lo, hi in zip(self.z_edges[:-1], self.z_edges[1:], strict=True)
            ]
        )

    def section(self, z: float) -> int:
        """Return the section a position lies in, clamped to the path's ends.

        Parameters
        ----------
        z : float
            Position in m.

        Returns
        -------
        int
            The section index; a position on an interior edge takes the section
            starting there, and one outside the path the nearest end's.
        """
        raw = int(np.searchsorted(self.z_edges, float(z), side="right")) - 1
        return min(max(raw, 0), self.n_intervals - 1)

    def interpolate(
        self,
        values: Inexact[np.ndarray, " n_positions *value"],
        z: float,
    ) -> np.ndarray:
        """Evaluate values sampled at the path's positions at any position.

        The barycentric form of the interpolating polynomial through one section's
        Chebyshev-Lobatto samples -- weights ``(-1)^j``, halved at the two ends -- which
        is the same polynomial the translation layer's Chebyshev coefficients carry,
        evaluated stably. A position outside the path **holds** the nearest end's
        value, as a :class:`~solprop.profiles.ZProfile` holds its own beyond its span:
        a run's tables reach a few per cent past its end, and the geometry is not
        declared there.

        Parameters
        ----------
        values : numpy.ndarray
            One value per position, in :attr:`positions` order flattened, position
            axis first, shape ``(n_intervals * (degree + 1), *value_shape)``.
        z : float
            Position in m.

        Returns
        -------
        numpy.ndarray
            The value at `z`, shape ``value_shape``.

        Raises
        ------
        ModeError
            If `values` does not carry one row per position.
        """
        rows = np.asarray(values)
        width = self.degree + 1
        if rows.shape[0] != self.n_intervals * width:
            raise ModeError(
                f"a path of {self.n_intervals} section(s) at degree {self.degree} "
                f"interpolates one value per position, {self.n_intervals * width} in "
                f"all, and got {rows.shape[0]}"
            )
        index = self.section(z)
        lo, hi = float(self.z_edges[index]), float(self.z_edges[index + 1])
        x = min(max((2.0 * float(z) - lo - hi) / (hi - lo), -1.0), 1.0)
        section = rows[index * width : (index + 1) * width]
        nodes = -np.cos(np.pi * np.arange(width) / self.degree)
        # A position that maps onto a node to within a few ulp -- the mid-section
        # node is -cos(pi/2) = -6e-17, which no position maps to exactly -- is that
        # node, whose value is returned as sampled rather than through a quotient.
        exact = np.flatnonzero(np.abs(nodes - x) <= 4.0 * np.finfo(float).eps)
        if exact.size:
            return np.asarray(section[int(exact[0])])
        weights = (-1.0) ** np.arange(width)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        terms = weights / (x - nodes)
        return np.asarray(np.tensordot(terms, section, axes=1) / float(np.sum(terms)))

    @property
    def identity(self) -> Mapping[str, JsonValue]:  # numpydoc ignore=RT01
        """What a run records of the path, and a resume compares.

        The declared edges, degree, subdivision and budget exactly; the two measured
        departures at :data:`~solprop.modes.material.MEASURED_DIGITS`, as every other
        measured entry of a guide's identity is.

        JSON-serialisable scalars and lists of scalars.
        """
        return {
            "z_edges": [float(one) for one in self.z_edges],
            "degree": int(self.degree),
            "sections": int(self.sections),
            "rtol": float(self.rtol),
            "z_error": quantised_measurement(self.z_error),
            "gram_departure": quantised_measurement(self.gram_departure),
        }

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``ModalPath(1 section, degree 8, 0 to 1 mm)``.
        """
        count = self.n_intervals
        return line(
            "ModalPath",
            [
                ("", f"{count} section{'' if count == 1 else 's'}"),
                ("", f"degree {self.degree}"),
                (
                    "",
                    f"{engineering(float(self.z_edges[0]), 'm')} to "
                    f"{engineering(float(self.z_edges[-1]), 'm')}",
                ),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        return "ModalPath: how a moving guide was sampled along z", [
            ("sections", f"{self.n_intervals} ({self.sections} per kink interval)"),
            (
                "span",
                f"{engineering(float(self.z_edges[0]), 'm')} to "
                f"{engineering(float(self.z_edges[-1]), 'm')}",
            ),
            ("degree", self.degree),
            ("positions", self.n_intervals * (self.degree + 1)),
            ("budget", f"{self.rtol:.3e}"),
            ("held-out error", f"{self.z_error:.3e}"),
            ("gram departure", f"{self.gram_departure:.3e}"),
        ]


def path_sections(
    provider: ModalProvider, sections: int = 1
) -> Float[np.ndarray, " n_edges"]:
    """Return the section edges of a moving provider's path, in m.

    The provider's z domain split at every declared breakpoint strictly inside it --
    each a kink of the geometry, where the connection jumps with the slope -- and each
    of those intervals divided into `sections` equal parts.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        The guide, whose ``z_domain`` and ``z_breakpoints`` are read.
    sections : int, optional
        Sections per kink interval, positive; one by default.

    Returns
    -------
    numpy.ndarray
        The edges in m, strictly increasing, from the domain's start to its end.

    Raises
    ------
    solprop.exceptions.ModeError
        If `sections` is not a positive integer.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.graded_index import (
    ...     GradedIndexProvider, PowerLawPermittivity,
    ... )
    >>> from solprop.modes.material import IndexMaterial
    >>> from solprop.profiles import PiecewiseLinear
    >>> from solprop.modes.path import path_sections
    >>> guide = GradedIndexProvider(
    ...     permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
    ...     exterior=IndexMaterial(1.45 * np.sqrt(1 - 2 * 0.005)),
    ...     radius=PiecewiseLinear(
    ...         np.array([0.0, 4e-4, 1e-3]), np.array([25e-6, 24e-6, 22.5e-6])
    ...     ),
    ...     family="LP", count=2, elements=16, extent=2.0,
    ... )
    >>> [round(one, 6) for one in path_sections(guide, 2).tolist()]
    [0.0, 0.0002, 0.0004, 0.0007, 0.001]
    """
    _count("sections", sections)
    low, high = (float(one) for one in provider.z_domain)
    kinks = sorted(
        {low, high, *(float(one) for one in provider.z_breakpoints if low < one < high)}
    )
    edges = [low]
    for start, end in itertools.pairwise(kinks):
        edges.extend(
            start + (end - start) * index / sections for index in range(1, sections)
        )
        edges.append(end)
    return np.asarray(edges, dtype=np.float64)


def path_rule(provider: ModalProvider, rule: ModalNodeRule, z: float) -> ModalNodeRule:
    """Return the provider's own rule at `z`, asked for exactly as `rule` was.

    The frequency and the resolution are read from `rule`'s own identity, so the rule
    at every position of a path is the one the provider would have built there on the
    same request: its panel edges on that position's interfaces, its tail certified
    there. It must keep `rule`'s structure -- the same node count and the same material
    region at every node -- because node ``q`` is one moving point along the whole
    path and the seam carries one region index per node.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        The guide.
    rule : solprop.modes.vector.ModalNodeRule
        The entrance rule, whose identity names ``omega``, ``n_r`` and ``n_theta``.
    z : float
        Position in m inside the provider's z domain.

    Returns
    -------
    solprop.modes.vector.ModalNodeRule
        The rule at `z`.

    Raises
    ------
    solprop.exceptions.ModeError
        If `rule` does not say how it was asked for, or the rule at `z` changes its
        node count or any node's material region.
    """
    identity = rule.identity
    missing = [
        name
        for name in ("omega", "n_r", "n_theta")
        if not isinstance(identity.get(name), int | float)
        or isinstance(identity.get(name), bool)
    ]
    if missing:
        raise ModeError(
            f"a moving guide asks its provider for the rule again at every position "
            f"of its path, exactly as the entrance rule was asked for, and this rule's "
            f"identity does not name its {missing}: pass nodes=None, or a rule "
            f"{type(provider).__name__}.node_rule built"
        )
    omega = float(identity["omega"])  # type: ignore[arg-type]
    moved = provider.node_rule(
        omega,
        float(z),
        n_r=int(identity["n_r"]),  # type: ignore[arg-type]
        n_theta=int(identity["n_theta"]),  # type: ignore[arg-type]
    )
    if moved.weights.shape != rule.weights.shape or not np.array_equal(
        moved.region_index, rule.region_index
    ):
        raise ModeError(
            f"{type(provider).__name__}'s rule at z = {float(z):.6g} m has "
            f"{moved.weights.size} nodes and {rule.weights.size} at the entrance, or "
            "a node that changes material between the two: a moving guide's node q "
            "is one point moving with the interfaces along the whole path, and the "
            "seam carries one region index per node"
        )
    return moved


def _check_gauge(
    reference: ModalSample, sample: ModalSample, nodes: ModalNodeRule, where: str
) -> None:
    """Refuse a provider whose own gauge is not continuous between two samples."""
    overlap = cross_poynting(reference, sample, nodes)
    diagonal = np.diag(overlap)
    size = np.abs(overlap)
    off = size - np.diag(np.abs(diagonal))
    worst_phase = float(np.max(np.abs(np.angle(diagonal))))
    if (
        np.any(np.real(diagonal) < MIN_OVERLAP_DOMINANCE)
        or np.any(np.max(off, axis=0) >= np.abs(diagonal))
        or worst_phase > GAUGE_PHASE_ATOL
    ):
        raise ModeError(
            f"the provider's own gauge is not continuous {where}: the channels' "
            f"overlaps with their neighbours are {np.round(diagonal, 6).tolist()}, "
            f"largest phase {worst_phase:.3e} rad against {GAUGE_PHASE_ATOL:.1e}. A "
            "moving guide is expanded in the gauge the provider returns, because its "
            "connection is measured in it; a sign that flips, a channel order that "
            "changes or a phase that drifts along the path would put a connection "
            "in the maps the measured one does not carry"
        )


def path_snapshot(
    provider: ModalProvider,
    omega: Float[np.ndarray, " n_omega"],
    z: float,
    nodes: ModalNodeRule,
    *,
    degenerate_blocks: Sequence[Sequence[int]] = (),
    basis: ChannelBasis | None = None,
    gram_atol: float = DEFAULT_GRAM_ATOL,
) -> DenseModalTable:
    """Solve one position of a path over the band, in the provider's own gauge.

    Unlike :func:`solprop.modes.tables.sample_modal_table` the knots are
    **not** aligned to one another: they are kept exactly as the provider returns
    them, with the gauge's continuity across the band checked instead
    (:data:`GAUGE_PHASE_ATOL`), because the connection a moving guide is propagated
    with is measured in that gauge. A declared `basis` is applied after, as it is
    there.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        The guide.
    omega : numpy.ndarray
        At least four increasing positive frequencies in rad/s, shape (n_omega,).
    z : float
        Position in m inside the provider's z domain.
    nodes : solprop.modes.vector.ModalNodeRule
        The rule at `z` (:func:`path_rule`).
    degenerate_blocks : sequence of sequence of int, optional
        The declared degenerate blocks, retained on the table.
    basis : solprop.modes.guide.ChannelBasis or None, optional
        A declared constant unitary change of the channel coordinates.
    gram_atol : float, optional
        The table's construction budget on its sampled Gram defect.

    Returns
    -------
    solprop.modes.tables.DenseModalTable
        A snapshot table at `z`.

    Raises
    ------
    solprop.exceptions.ModeError
        If the provider's gauge is not continuous across the band, or on anything the
        provider or the table refuses.
    """
    raw: list[ModalSample] = []
    for frequency in np.asarray(omega, dtype=float):
        sample = provider.solve(float(frequency), float(z), nodes=nodes)
        if raw:
            _check_gauge(
                raw[-1],
                sample,
                nodes,
                f"between {float(raw[-1].omega):.6e} and {float(frequency):.6e} rad/s "
                f"at z = {float(z):.6g} m",
            )
        raw.append(sample)
    row = tuple(
        raw if basis is None else (apply_channel_basis(one, basis) for one in raw)
    )
    blocks = tuple(tuple(int(one) for one in block) for block in degenerate_blocks)
    return DenseModalTable(
        np.asarray(omega, dtype=float),
        np.array([float(z)]),
        (row,),
        nodes,
        blocks,
        gram_atol,
        basis,
    )


@dataclass(frozen=True, eq=False)
class ModalPathTable(Displayed):
    """Hold a moving guide's snapshots at its path's Chebyshev-Lobatto points.

    What :func:`sample_modal_path` returns: the record of how the path was sampled
    and the snapshot solved at each of its positions.

    Attributes
    ----------
    path : ModalPath
        How the path was sampled and what that measured: its section edges, degree,
        positions and held-out departures.
    tables : tuple of solprop.modes.tables.DenseModalTable
        One snapshot per position of ``path.positions``, section by section and
        increasing inside each, so a position on an edge between two sections appears
        twice, as the end of one and the start of the next.

    Raises
    ------
    solprop.exceptions.ModeError
        If the tables do not sit at the path's positions, in order.
    """

    path: ModalPath
    tables: tuple[DenseModalTable, ...]

    def __post_init__(self) -> None:
        """Check each table sits where the path places it.

        Raises
        ------
        solprop.exceptions.ModeError
            Naming the count expected.
        """
        path = self.path
        expected = path.positions.reshape(-1)
        if len(self.tables) != expected.size or any(
            float(table.z[0]) != float(position)
            for table, position in zip(self.tables, expected, strict=False)
        ):
            raise ModeError(
                f"a path of {path.n_intervals} section(s) at degree {path.degree} "
                f"holds one snapshot at each of its {expected.size} Chebyshev-Lobatto "
                f"positions, in order; got {len(self.tables)}"
            )
        object.__setattr__(self, "tables", tuple(self.tables))

    def section_tables(self, index: int) -> tuple[DenseModalTable, ...]:
        """Return one section's snapshots, increasing in z.

        Parameters
        ----------
        index : int
            The section, in ``[0, path.n_intervals)``.

        Returns
        -------
        tuple of solprop.modes.tables.DenseModalTable
            Its ``path.degree + 1`` snapshots.
        """
        width = self.path.degree + 1
        return self.tables[index * width : (index + 1) * width]

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``ModalPathTable(1 section, degree 8, 9 positions)``.
        """
        count = self.path.n_intervals
        return line(
            "ModalPathTable",
            [
                ("", f"{count} section{'' if count == 1 else 's'}"),
                ("", f"degree {self.path.degree}"),
                ("", f"{len(self.tables)} positions"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        path = self.path
        return "ModalPathTable: a moving guide's snapshots along z", [
            ("sections", path.n_intervals),
            (
                "span",
                f"{engineering(float(path.z_edges[0]), 'm')} to "
                f"{engineering(float(path.z_edges[-1]), 'm')}",
            ),
            ("degree", path.degree),
            ("positions", len(self.tables)),
            ("held-out error", f"{path.z_error:.3e}"),
            ("gram departure", f"{path.gram_departure:.3e}"),
        ]


def _stacked(
    tables: Sequence[DenseModalTable],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack the knots' E, H and generator of several snapshots, position first.

    - Returns: E and H of shape ``(n_positions, n_knots, n_nodes, 3, n_modes)`` and the
      generator of shape ``(n_positions, n_knots, n_modes, n_modes)``.
    """
    return tuple(  # type: ignore[return-value]
        np.array(
            [[getattr(sample, name) for sample in table.samples[0]] for table in tables]
        )
        for name in ("electric", "magnetic", "generator")
    )


def _relative(interpolated: np.ndarray, fresh: np.ndarray) -> float:
    """Return the relative maximum-norm departure of an interpolant from a solve.

    - ``fresh``: the fresh solve's, of the same shape and not identically zero.
    """
    return float(np.max(np.abs(interpolated - fresh)) / np.max(np.abs(fresh)))


def _held_out(
    section: Sequence[DenseModalTable],
    fresh: Sequence[DenseModalTable],
    lo: float,
    hi: float,
) -> tuple[float, float]:
    """Measure a section's expansion against fresh solves at the held-out points.

    - ``lo``, ``hi``: the section's ends, m.
    - Returns: the largest relative departure of E, H and the generator, and the largest
      ``norm(W - I, 2)`` of the expanded basis on the held-out rule.
    """
    degree = len(section) - 1
    coefficients = [lobatto_coefficients(one) for one in _stacked(section)]
    held = _stacked(fresh)
    error = 0.0
    departure = 0.0
    for index, table in enumerate(fresh):
        x = (2.0 * float(table.z[0]) - lo - hi) / (hi - lo)
        weights = chebyshev_basis.chebvander(x, degree)[0]
        electric, magnetic, generator = (
            np.tensordot(weights, one, axes=1) for one in coefficients
        )
        error = max(
            error,
            _relative(electric, held[0][index]),
            _relative(magnetic, held[1][index]),
            _relative(generator, held[2][index]),
        )
        identity = np.eye(electric.shape[-1])
        for knot in range(electric.shape[0]):
            metric = flux_product(electric[knot], magnetic[knot], table.nodes.weights)
            departure = max(departure, float(np.linalg.norm(metric - identity, ord=2)))
    return error, departure


def sample_modal_path(
    provider: ModalProvider,
    omega: Float[np.ndarray, " n_omega"],
    rule: ModalNodeRule,
    *,
    rtol: float | None = None,
    sections: int | None = None,
    path: ModalPath | None = None,
    degenerate_blocks: Sequence[Sequence[int]] = (),
    basis: ChannelBasis | None = None,
    gram_atol: float = DEFAULT_GRAM_ATOL,
    known: Mapping[float, DenseModalTable] | None = None,
) -> ModalPathTable:
    """Sample a moving guide along its sections, at the degree its accuracy demands.

    The path is cut into sections (:func:`path_sections`), and each section is solved
    at the Chebyshev-Lobatto points of a degree, starting at
    :data:`INITIAL_PATH_DEGREE`, on the provider's own rule at each position
    (:func:`path_rule`) and in its own gauge (:func:`path_snapshot`). The expansion
    along z is then checked against **fresh** solves at the points a doubling adds,
    which are the natural held-out points: they lie between every pair of the current
    ones and are exactly the solves the next degree needs, so the check costs nothing
    the doubling would not. The degree is doubled, for every section together because
    the seam's degree axis is rectangular, until the largest relative departure of E,
    H and the generator is within `rtol`; the degree then kept is the one whose error
    was **measured**, not the doubled one, so that the certificate is a measurement of
    what is stored.

    With a qualified `path` given instead, nothing is searched or measured: the
    snapshots are solved at its edges and degree, and it is carried as it stands,
    which is how a run rebuilds the path its guide already qualified.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        The moving guide.
    omega : numpy.ndarray
        The frequency knots in rad/s, shape (n_omega,), at least four.
    rule : solprop.modes.vector.ModalNodeRule
        The entrance rule, which every position's rule is asked for like
        (:func:`path_rule`).
    rtol : float or None, optional
        Budget on the held-out departure, positive and finite, for a search; given
        unless `path` is.
    sections : int or None, optional
        Sections per kink interval of a search, positive; ``None`` is one. Not given
        with `path`.
    path : ModalPath or None, optional
        A path already qualified, which skips the search.
    degenerate_blocks : sequence of sequence of int, optional
        The declared degenerate blocks.
    basis : solprop.modes.guide.ChannelBasis or None, optional
        A declared constant unitary change of the channel coordinates.
    gram_atol : float, optional
        Each snapshot's construction budget on its sampled Gram defect.
    known : mapping or None, optional
        Snapshots already solved, keyed by position, reused rather than solved again.

    Returns
    -------
    ModalPathTable
        The path record and its snapshots.

    Raises
    ------
    solprop.exceptions.ModeError
        If neither or both of a search's budget and a qualified `path` are given,
        `rtol` is not positive and finite, `sections` is not a positive integer, the
        held-out departure is still above `rtol` at :data:`MAX_PATH_DEGREE`, or on
        anything a snapshot refuses.
    """
    if (path is None) == (rtol is None) or (path is not None and sections is not None):
        raise ModeError(
            "a moving guide's path is either searched, within a budget rtol over its "
            "sections, or rebuilt from a path already qualified: give rtol (and "
            f"sections) or path, not both, got rtol={rtol!r}, sections={sections!r} "
            f"and path={path!r}"
        )
    memo: dict[float, DenseModalTable] = dict(known or {})

    def _at(position: float) -> DenseModalTable:
        """Return the snapshot at one position, solving it at most once.

        - ``position``: position in m.
        """
        if position not in memo:
            memo[position] = path_snapshot(
                provider,
                omega,
                position,
                path_rule(provider, rule, position),
                degenerate_blocks=degenerate_blocks,
                basis=basis,
                gram_atol=gram_atol,
            )
        return memo[position]

    if path is None:
        assert rtol is not None  # the budget of a search, checked above
        budget = _measure("rtol", rtol, positive=True)
        count = 1 if sections is None else sections
        edges = path_sections(provider, count)
        spans = list(itertools.pairwise(edges))
        chosen = INITIAL_PATH_DEGREE
        while True:
            error = departure = 0.0
            for lo, hi in spans:
                section = [_at(float(one)) for one in lobatto_points(chosen, lo, hi)]
                fresh = [_at(float(one)) for one in refinement_points(chosen, lo, hi)]
                measured_error, measured_departure = _held_out(section, fresh, lo, hi)
                error = max(error, measured_error)
                departure = max(departure, measured_departure)
            logger.debug(
                "modal path: degree %d over %d section(s), held-out error %.3e against "
                "%.3e, gram departure %.3e",
                chosen,
                len(spans),
                error,
                budget,
                departure,
            )
            if error <= budget:
                break
            if 2 * chosen > MAX_PATH_DEGREE:
                raise ModeError(
                    f"a moving guide's expansion along z does not resolve: at degree "
                    f"{chosen} its held-out departure from a fresh solve is "
                    f"{error:.4g}, above the {budget:.4g} budget, and the next "
                    f"degree would exceed {MAX_PATH_DEGREE}. Divide the path into more "
                    "sections, or declare the structure as a kink of the geometry"
                )
            chosen *= 2
        path = ModalPath(edges, chosen, count, budget, error, departure)
    tables = tuple(
        _at(float(one))
        for lo, hi in itertools.pairwise(path.z_edges)
        for one in lobatto_points(path.degree, float(lo), float(hi))
    )
    # Every neighbouring pair of positions, the ones given in `known` included: the
    # check is an overlap per pair and costs nothing beside the solves.
    for first, second in itertools.pairwise(sorted(memo)):
        _check_gauge(
            memo[first].samples[0][0],
            memo[second].samples[0][0],
            memo[second].nodes,
            f"between z = {first:.6g} and {second:.6g} m",
        )
    return ModalPathTable(path, tables)
