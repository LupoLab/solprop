# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Sample a guide whose geometry moves along z, and expand what a run reads of it.

A capillary's modes are ``J_0(u r/a)/a``, a pure
function of ``r/a``, so a tapered capillary's whole transform is its entrance transform
times one scalar. A **solid** guide's modes change *shape* along a taper -- a
step-index or graded-index mode moves with its normalised frequency -- so the general
route's synthesis ``S`` and projection ``Q`` move with z and every nonlinear overlap
follows them. This module is where that motion is sampled and expanded.

**The node rule moves with the geometry.** Each sampled position is solved on the
provider's **own** rule at that position, asked for at one fixed frequency and one
fixed resolution (:func:`path_rule`): its panel edges sit on the material interfaces
*there*, so every solve is integrated exactly (a rule whose panel edge has left a
moving interface is wrong at first order in the displacement). Node ``q``
of the rule at ``z`` is a point ``x_q(z)`` that moves smoothly with the interfaces, and
its material region does not change, so the field a map stores at node ``q`` is a
smooth function of z inside a section and the per-node region index the seam carries
is the same at every position. That is what lets the maps be expanded along z at all:
on a rule frozen at one position a node the interface sweeps over changes material
half way along, and the field there has a kink or a jump in z that no expansion in z
resolves. The connection's derivative is a different quantity -- the Eulerian
``<psi_m | d_z psi_n>`` at fixed transverse position -- and it keeps the union-split
rule of :func:`solprop.modes.connection.connection_nodes`, which is exactly what that
derivative needs; the two rules describe the same physical modes and differ only in
where they integrate them.

**The sampling is per section, at the Chebyshev-Lobatto points of a degree the
tables' own accuracy check demands.** The sections are the provider's declared
kink intervals, optionally subdivided (:func:`path_sections`); a section never spans a
kink, so everything sampled inside it is smooth. The degree is doubled until the
expansion reproduces a **fresh** solve at the positions the doubling adds -- the
held-out points -- to within the budget, which by default is the frequency
interpolation's own measured error, so that sampling in z is never the coarser of the
two (:func:`sample_modal_path`).

**The gauge is the provider's own.** Each position is solved and kept in the gauge the
provider returns, with no tracking alignment, because the connection
(:func:`solprop.modes.connection.band_connection`) is measured in that gauge: its
stencil is aligned to the provider's own centre solve. A provider whose own gauge is
not continuous along the path -- a sign that flips between two positions or two
frequencies, a phase that drifts -- would put a gauge connection in the maps that the
measured connection does not carry, so it is refused by name
(:func:`path_snapshot`).
"""

import itertools
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float
from numpy.polynomial import chebyshev as chebyshev_basis
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

from solprop._display import Displayed, engineering, line
from solprop.exceptions import ModeError
from solprop.modes.connection import ModalConnection, band_connection
from solprop.modes.guide import ChannelBasis
from solprop.modes.vector import ModalNodeRule, ModalProvider, ModalSample, flux_product
from solprop.translate.chebyshev import (
    _EDGE_SLACK_FRACTION,
    lobatto_coefficients,
    lobatto_points,
    refinement_points,
)
from solprop.translate.modal_tables import (
    DEFAULT_GRAM_ATOL,
    MIN_OVERLAP_DOMINANCE,
    DenseModalTable,
    ModalTransformTables,
    _coordinate_dual,
    _overlap,
    _projection_map,
    apply_channel_basis,
)

__all__ = [
    "GAUGE_PHASE_ATOL",
    "INITIAL_PATH_DEGREE",
    "MAX_PATH_DEGREE",
    "PATH_RTOL_FLOOR",
    "ModalPathTable",
    "in_frequency",
    "path_connections",
    "path_rule",
    "path_sections",
    "path_snapshot",
    "path_transform_tables",
    "sample_modal_path",
    "section_coefficients",
    "transform_section",
    "transform_weights",
]

logger = logging.getLogger(__name__)

INITIAL_PATH_DEGREE = 4
"""Chebyshev degree along z a path's accuracy search starts at, per section.

Four is five solves per section before anything is measured. The search doubles from
here, and the Chebyshev-Lobatto points nest bitwise under a doubling, so no solve is
ever repeated (:func:`solprop.translate.chebyshev.lobatto_points`).
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
    >>> from solprop.translate.modal_path import path_sections
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
    if isinstance(sections, bool) or not isinstance(sections, int) or sections < 1:
        raise ModeError(
            f"a moving guide's sections per kink interval is a positive integer, got "
            f"{sections!r}"
        )
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
    """Refuse a provider whose own gauge is not continuous between two samples.

    Parameters
    ----------
    reference : solprop.modes.vector.ModalSample
        The neighbouring sample already accepted.
    sample : solprop.modes.vector.ModalSample
        The new one, on a rule of the same structure.
    nodes : solprop.modes.vector.ModalNodeRule
        The new sample's rule, whose weights the overlap is taken with.
    where : str
        What separates the two, for the message.

    Raises
    ------
    solprop.exceptions.ModeError
        If a channel's own overlap is not the dominant, positive, real one.
    """
    overlap = _overlap(reference, sample, nodes)
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


def path_snapshot(  # noqa: PLR0913 - a provider, a band, a point, a rule and three declarations
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

    Unlike :func:`solprop.translate.modal_tables.sample_modal_table` the knots are
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
    solprop.translate.modal_tables.DenseModalTable
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
    """Hold a moving guide's snapshots at its sections' Chebyshev-Lobatto points.

    Attributes
    ----------
    z_edges : numpy.ndarray
        Section edges in m, shape ``(n_edges_path,)``, strictly increasing.
    degree : int
        The Chebyshev degree along z every section is sampled at, at least one; each
        section holds ``degree + 1`` positions, its two ends included.
    tables : tuple of solprop.translate.modal_tables.DenseModalTable
        One snapshot per position, section by section and increasing inside each, so
        a position on an edge between two sections appears twice, as the end of one
        and the start of the next.
    z_error : float
        The measured held-out error of the expansion along z, dimensionless and
        non-negative: the largest relative maximum-norm departure of E, H or the
        generator from a fresh solve at the positions a degree doubling adds.
    gram_departure : float
        The largest ``norm(W - I, 2)`` of the expanded basis at those same positions,
        on the rule there, dimensionless and non-negative.

    Raises
    ------
    solprop.exceptions.ModeError
        If the counts, the positions or the measurements are inconsistent.
    """

    z_edges: Float[np.ndarray, " n_edges_path"]
    degree: int
    tables: tuple[DenseModalTable, ...]
    z_error: float
    gram_departure: float

    def __post_init__(self) -> None:
        """Freeze the edges and check each table sits where it should.

        Raises
        ------
        solprop.exceptions.ModeError
            Naming what is inconsistent.
        """
        edges = np.array(self.z_edges, dtype=np.float64, copy=True).reshape(-1)
        if (
            edges.size < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0)
        ):
            raise ModeError(
                "a path's section edges are at least two finite increasing positions"
            )
        edges.setflags(write=False)
        object.__setattr__(self, "z_edges", edges)
        if (
            isinstance(self.degree, bool)
            or not isinstance(self.degree, int)
            or self.degree < 1
        ):
            raise ModeError(
                f"a path's degree is a positive integer, got {self.degree!r}"
            )
        expected = self.positions.reshape(-1)
        if len(self.tables) != expected.size or any(
            float(table.z[0]) != float(position)
            for table, position in zip(self.tables, expected, strict=False)
        ):
            raise ModeError(
                f"a path of {self.n_intervals} section(s) at degree {self.degree} "
                f"holds one snapshot at each of its {expected.size} Chebyshev-Lobatto "
                f"positions, in order; got {len(self.tables)}"
            )
        for name in ("z_error", "gram_departure"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ModeError(
                    f"a path's {name} is a measured non-negative number, got "
                    f"{getattr(self, name)!r}"
                )

    @property
    def n_intervals(self) -> int:
        """Return how many sections the path is cut into.

        Returns
        -------
        int
            The section count.
        """
        return int(self.z_edges.size - 1)

    @property
    def positions(self) -> Float[np.ndarray, "n_intervals_positions n_deg_positions"]:
        """Return every sampled position, section by section.

        Returns
        -------
        numpy.ndarray
            Positions in m, shape ``(n_intervals, degree + 1)``.
        """
        return np.stack(
            [
                lobatto_points(self.degree, float(lo), float(hi))
                for lo, hi in zip(self.z_edges[:-1], self.z_edges[1:], strict=True)
            ]
        )

    def section_tables(self, index: int) -> tuple[DenseModalTable, ...]:
        """Return one section's snapshots, increasing in z.

        Parameters
        ----------
        index : int
            The section, in ``[0, n_intervals)``.

        Returns
        -------
        tuple of solprop.translate.modal_tables.DenseModalTable
            Its ``degree + 1`` snapshots.
        """
        width = self.degree + 1
        return self.tables[index * width : (index + 1) * width]

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``ModalPathTable(1 section, degree 8, 9 positions)``.
        """
        return line(
            "ModalPathTable",
            [
                (
                    "",
                    f"{self.n_intervals} section{'' if self.n_intervals == 1 else 's'}",
                ),
                ("", f"degree {self.degree}"),
                ("", f"{len(self.tables)} positions"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "ModalPathTable: a moving guide's snapshots along z", [
            ("sections", self.n_intervals),
            (
                "span",
                f"{engineering(float(self.z_edges[0]), 'm')} to "
                f"{engineering(float(self.z_edges[-1]), 'm')}",
            ),
            ("degree", self.degree),
            ("positions", len(self.tables)),
            ("held-out error", f"{self.z_error:.3e}"),
            ("gram departure", f"{self.gram_departure:.3e}"),
        ]


def _stacked(
    tables: Sequence[DenseModalTable],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack the knots' E, H and generator of several snapshots, position first.

    Parameters
    ----------
    tables : sequence of solprop.translate.modal_tables.DenseModalTable
        Snapshots on rules of one structure and one band.

    Returns
    -------
    tuple of numpy.ndarray
        E and H of shape ``(n_positions, n_knots, n_nodes, 3, n_modes)`` and the
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

    Parameters
    ----------
    interpolated : numpy.ndarray
        The expansion's value.
    fresh : numpy.ndarray
        The fresh solve's, of the same shape and not identically zero.

    Returns
    -------
    float
        ``max|interpolated - fresh| / max|fresh|``, dimensionless.
    """
    return float(np.max(np.abs(interpolated - fresh)) / np.max(np.abs(fresh)))


def _held_out(
    section: Sequence[DenseModalTable],
    fresh: Sequence[DenseModalTable],
    lo: float,
    hi: float,
) -> tuple[float, float]:
    """Measure a section's expansion against fresh solves at the held-out points.

    Parameters
    ----------
    section : sequence of solprop.translate.modal_tables.DenseModalTable
        The section's snapshots at the Lobatto points of its degree.
    fresh : sequence of solprop.translate.modal_tables.DenseModalTable
        Fresh snapshots at the points a doubling adds.
    lo, hi : float
        The section's ends, m.

    Returns
    -------
    tuple of float
        The largest relative departure of E, H and the generator, and the largest
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


def sample_modal_path(  # noqa: PLR0913 - a provider, a band, a rule, the sections and five declarations
    provider: ModalProvider,
    omega: Float[np.ndarray, " n_omega"],
    rule: ModalNodeRule,
    z_edges: Float[np.ndarray, " n_edges"],
    *,
    rtol: float,
    degree: int | None = None,
    measured: tuple[float, float] = (0.0, 0.0),
    degenerate_blocks: Sequence[Sequence[int]] = (),
    basis: ChannelBasis | None = None,
    gram_atol: float = DEFAULT_GRAM_ATOL,
    known: Mapping[float, DenseModalTable] | None = None,
) -> ModalPathTable:
    """Sample a moving guide along its sections, at the degree its accuracy demands.

    Each section is solved at the Chebyshev-Lobatto points of a degree, starting at
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

    With `degree` given, nothing is searched or measured: the path is sampled at that
    degree and carries the `measured` numbers it was certified with, which is how a
    run rebuilds the path its guide already qualified.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        The moving guide.
    omega : numpy.ndarray
        The frequency knots in rad/s, shape (n_omega,), at least four.
    rule : solprop.modes.vector.ModalNodeRule
        The entrance rule, which every position's rule is asked for like
        (:func:`path_rule`).
    z_edges : numpy.ndarray
        Section edges in m (:func:`path_sections`), shape (n_edges,).
    rtol : float
        Budget on the held-out departure, positive and finite.
    degree : int or None, optional
        A degree already qualified, which skips the search.
    measured : tuple of float, optional
        The held-out error and Gram departure the qualified `degree` was measured
        at; carried, not re-measured.
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
        The snapshots and the measurements.

    Raises
    ------
    solprop.exceptions.ModeError
        If `rtol` is not positive and finite, the held-out departure is still above
        it at :data:`MAX_PATH_DEGREE`, or on anything a snapshot refuses.
    """
    if not math.isfinite(float(rtol)) or float(rtol) <= 0.0:
        raise ModeError(
            f"a path's held-out budget is positive and finite, got {rtol!r}"
        )
    edges = np.asarray(z_edges, dtype=np.float64)
    memo: dict[float, DenseModalTable] = dict(known or {})

    def at(position: float) -> DenseModalTable:
        """Return the snapshot at one position, solving it at most once.

        Parameters
        ----------
        position : float
            Position in m.

        Returns
        -------
        solprop.translate.modal_tables.DenseModalTable
            The snapshot there.
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

    sections = list(itertools.pairwise(edges))
    if degree is not None:
        chosen, (error, departure) = int(degree), measured
    else:
        chosen = INITIAL_PATH_DEGREE
        while True:
            error = departure = 0.0
            for lo, hi in sections:
                section = [at(float(one)) for one in lobatto_points(chosen, lo, hi)]
                fresh = [at(float(one)) for one in refinement_points(chosen, lo, hi)]
                measured_error, measured_departure = _held_out(section, fresh, lo, hi)
                error = max(error, measured_error)
                departure = max(departure, measured_departure)
            logger.debug(
                "modal path: degree %d over %d section(s), held-out error %.3e against "
                "%.3e, gram departure %.3e",
                chosen,
                len(sections),
                error,
                rtol,
                departure,
            )
            if error <= float(rtol):
                break
            if 2 * chosen > MAX_PATH_DEGREE:
                raise ModeError(
                    f"a moving guide's expansion along z does not resolve: at degree "
                    f"{chosen} its held-out departure from a fresh solve is "
                    f"{error:.4g}, above the {float(rtol):.4g} budget, and the next "
                    f"degree would exceed {MAX_PATH_DEGREE}. Divide the path into more "
                    "sections, or declare the structure as a kink of the geometry"
                )
            chosen *= 2
    tables = tuple(
        at(float(one))
        for lo, hi in sections
        for one in lobatto_points(chosen, float(lo), float(hi))
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
    return ModalPathTable(edges, chosen, tables, float(error), float(departure))


def section_coefficients(
    values: Complex[np.ndarray, " n_intervals n_positions *value"],
) -> Complex[np.ndarray, " n_intervals n_deg *value"]:
    """Return the Chebyshev coefficients along z of values sampled per section.

    Parameters
    ----------
    values : numpy.ndarray
        Samples at each section's Chebyshev-Lobatto points, increasing, shape
        ``(n_intervals, degree + 1, *value_shape)``.

    Returns
    -------
    numpy.ndarray
        The coefficients :func:`numpy.polynomial.chebyshev.chebval` evaluates on
        ``x = (2 z - lo - hi)/(hi - lo)``, same shape.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.modal_path import section_coefficients
    >>> section_coefficients(np.array([[[1.0 + 0j], [1.0 + 0j]]])).round(12).tolist()
    [[[(1+0j)], [0j]]]
    """
    return np.stack([lobatto_coefficients(one) for one in np.asarray(values)])


def transform_section(tables: ModalTransformTables, z: float) -> int:
    """Return the section of a record's maps a position lies in.

    The maps are **continuous** across a section edge -- the fields at a node are, even
    at a kink of the geometry, where only their slope jumps -- so which side a
    position on an edge reads changes the value by the representation error alone and
    no one-sided rule is needed, unlike the coupled flow's
    (:func:`solprop.translate.coupled.flow_weights`). A position within ``1e-10`` of
    the span of an interior edge takes the section starting there, as
    :func:`solprop.translate.coupled.flow_section` does, and one a rounding outside the
    span the nearest end's.

    Parameters
    ----------
    tables : solprop.translate.modal_tables.ModalTransformTables
        The record.
    z : float
        Position in m.

    Returns
    -------
    int
        The section index, in ``[0, n_intervals)``.
    """
    edges = tables.z_edges
    slack = _EDGE_SLACK_FRACTION * float(edges[-1] - edges[0])
    raw = int(np.searchsorted(edges, float(z) + slack, side="right")) - 1
    return min(max(raw, 0), edges.size - 2)


def transform_weights(
    tables: ModalTransformTables | None, z: float
) -> Float[np.ndarray, " n_transform_terms"] | None:
    """Return the weights that combine a record's moving maps at one position.

    The stage-bundle row of kernel schema 14
    (:attr:`solprop.translate.stage.StageScalars.transform_weights`): a backend
    evaluates ``S(z) = sum_j w_j S_j`` and ``Q(z) = sum_j w_j Q_j`` over the maps'
    ``(n_intervals, n_deg)`` coefficient axes **flattened**, exactly as it evaluates
    ``dPhi`` over its terms. Every weight outside the section containing `z`
    (:func:`transform_section`) is zero, so no section's expansion is extrapolated.

    ``None`` where there is nothing to combine: no record, a record carrying no maps,
    or maps of **one** coefficient -- one interval of degree zero, a constant
    geometry -- whose one term is the map itself. That keeps the stage bundle of
    every problem whose maps do not move unchanged.

    Parameters
    ----------
    tables : solprop.translate.modal_tables.ModalTransformTables or None
        The record, or ``None``.
    z : float
        Position in m.

    Returns
    -------
    numpy.ndarray or None
        Float64, shape ``(n_intervals * n_deg,)``, or ``None``.

    Examples
    --------
    >>> from solprop.translate.modal_path import transform_weights
    >>> transform_weights(None, 0.5) is None
    True
    """
    if tables is None or tables.synthesis is None:
        return None
    intervals, degrees = (int(one) for one in tables.synthesis.shape[:2])
    if intervals * degrees == 1:
        return None
    index = transform_section(tables, z)
    lo, hi = float(tables.z_edges[index]), float(tables.z_edges[index + 1])
    weights = np.zeros((intervals, degrees), dtype=np.float64)
    weights[index] = chebyshev_basis.chebvander(
        (2.0 * float(z) - lo - hi) / (hi - lo), degrees - 1
    )[0]
    return weights.reshape(-1)


def path_transform_tables(
    path: ModalPathTable,
    omega: Float[np.ndarray, " n_omega"],
    *,
    z_end: float,
    profile_error: ArrayLike | float,
    omega_fine: ArrayLike | None = None,
    policy_id: str = "dense-knots-cubic-frequency-chebyshev-z",
) -> ModalTransformTables:
    """Build the flat seam record of a moving guide, its maps expanded along z.

    The moving counterpart of
    :func:`solprop.translate.modal_tables.modal_transform_tables`. At every sampled
    position of every section the path covers up to `z_end`, the snapshot is evaluated
    on the run's coarse grid exactly as a snapshot record's is -- ``S`` the stored
    electric profile, ``Q = W^-1 conj(e) w / 4`` with **that position's** area
    weights -- and each section's samples become Chebyshev coefficients along z
    (:func:`section_coefficients`), which is the ``(n_intervals, n_deg)`` layout the
    record's maps have always had. The dense reference, its partners, the coordinate
    dual, the node positions and the weights are stored at the section edges, which
    are the record's ``z_edges``; the region index is the path's one per node.

    Outside the band the maps are explicitly zero, as a snapshot's are.

    Parameters
    ----------
    path : ModalPathTable
        The sampled path.
    omega : numpy.ndarray
        Coarse angular frequencies in rad/s, shape (n_omega,), increasing.
    z_end : float
        Where the run ends, m; the record keeps every section up to the one that
        contains it, so it covers the run and nothing is extrapolated.
    profile_error : array_like or float
        Declared non-negative profile error per section and coarse frequency, or one
        value for all.
    omega_fine : array_like or None, optional
        Fine angular frequencies in rad/s; ``None`` uses the coarse grid.
    policy_id : str, optional
        Identity of the numeric policy the record was built under.

    Returns
    -------
    solprop.translate.modal_tables.ModalTransformTables
        The record, with maps of ``n_intervals`` sections and ``degree + 1``
        coefficients each.

    Raises
    ------
    solprop.exceptions.ModeError
        If the path does not reach `z_end`, or the record does not validate.
    """
    end = float(z_end)
    edges = path.z_edges
    if not edges[0] <= 0.0 < end <= edges[-1]:
        raise ModeError(
            f"a moving guide's path covers [{float(edges[0]):.6g}, "
            f"{float(edges[-1]):.6g}] m and the run [0, {end:.6g}] m: the geometry is "
            "declared over the provider's own z domain, and a run past its end would "
            "hold the last position's modes over the rest in silence"
        )
    kept = int(np.searchsorted(edges, end, side="left"))
    kept = max(kept, 1)
    coarse = np.ascontiguousarray(omega, dtype=np.float64)
    fine = coarse if omega_fine is None else np.ascontiguousarray(omega_fine, float)
    first = path.tables[0]
    reference = first.samples[0][0]
    modes = int(reference.electric.shape[2])
    nodes = int(first.nodes.weights.size)
    inside = (first.omega[0] <= coarse) & (coarse <= first.omega[-1])
    inside_fine = (first.omega[0] <= fine) & (fine <= first.omega[-1])
    width = path.degree + 1
    synthesis = np.zeros((kept, width, nodes, 3, modes, coarse.size), np.complex128)
    projection = np.zeros((kept, width, modes, nodes, 3, coarse.size), np.complex128)
    for section in range(kept):
        for position, table in enumerate(path.section_tables(section)):
            weights = table.nodes.weights
            for column in np.flatnonzero(inside):
                sample = table.evaluate(float(coarse[column]), float(table.z[0]))
                synthesis[section, position, :, :, :, column] = sample.electric
                projection[section, position, :, :, :, column] = _projection_map(
                    sample, weights
                )
    synthesis = section_coefficients(synthesis)
    projection = section_coefficients(projection)
    ends = [path.section_tables(index)[0] for index in range(kept)]
    ends.append(path.section_tables(kept - 1)[-1])
    reciprocal = reference.reciprocal_gram is not None
    shape = (len(ends), nodes, 3, modes, coarse.size)
    electric = np.zeros(shape, np.complex128)
    magnetic = np.zeros(shape, np.complex128)
    backward = [
        np.zeros(shape, np.complex128) if reciprocal else None for _ in range(2)
    ]
    gram = (
        np.zeros((len(ends), modes, modes, coarse.size), np.complex128)
        if reciprocal
        else None
    )
    dual = np.zeros((len(ends), modes, nodes, 6, coarse.size), np.complex128)
    for row, table in enumerate(ends):
        for column in np.flatnonzero(inside):
            sample = table.evaluate(float(coarse[column]), float(table.z[0]))
            electric[row, :, :, :, column] = sample.electric
            magnetic[row, :, :, :, column] = sample.magnetic
            dual[row, :, :, :, column] = _coordinate_dual(sample, table.nodes.weights)
            if backward[0] is not None and backward[1] is not None:
                backward[0][row, :, :, :, column] = sample.backward_electric
                backward[1][row, :, :, :, column] = sample.backward_magnetic
            if gram is not None:
                gram[row, :, :, column] = sample.reciprocal_gram
    logger.debug(
        "moving modal transform tables: %d channels on %d nodes, %d section(s) at "
        "degree %d, %d of %d coarse frequencies inside the band",
        modes,
        nodes,
        kept,
        path.degree,
        int(np.count_nonzero(inside)),
        coarse.size,
    )
    return ModalTransformTables(
        coarse,
        fine,
        edges[: kept + 1],
        synthesis,
        projection,
        electric,
        magnetic,
        backward[0],
        backward[1],
        gram,
        dual,
        np.stack([table.nodes.xy for table in ends]),
        np.stack([table.nodes.weights for table in ends]),
        first.nodes.region_index,
        -1j * coarse,
        np.broadcast_to(inside_fine, (len(ends), modes, fine.size)),
        np.broadcast_to(
            np.asarray(profile_error, dtype=np.float64), (kept, coarse.size)
        ),
        None,
        reference.gauge_id,
        reference.physical_id,
        policy_id,
    )


def path_connections(  # noqa: PLR0913 - a path, a provider and four declarations
    path: ModalPathTable,
    provider: ModalProvider,
    *,
    sections: int | None = None,
    degenerate_blocks: Sequence[Sequence[int]] = (),
    basis: ChannelBasis | None = None,
    n_r: int | None = None,
    n_theta: int | None = None,
) -> tuple[tuple[ModalConnection, ...], ...]:
    """Measure the moving-basis connection at every sampled position of a path.

    :func:`solprop.modes.connection.band_connection` at each position and every
    frequency knot, in the provider's own gauge with the declared basis applied, which
    is the gauge the path's snapshots are in (:func:`path_snapshot`). A position on a
    section's **end** is measured one ulp inside it, so that its stencil is taken on
    that section's own side of an edge: at a kink of the geometry the slope jumps and
    the connection with it, and the section that closes there must read its own
    one-sided limit (the rule of the coupled flow's
    :func:`solprop.translate.coupled.flow_weights`, applied where it is sampled).

    Parameters
    ----------
    path : ModalPathTable
        The sampled path.
    provider : solprop.modes.vector.ModalProvider
        The moving guide.
    sections : int or None, optional
        How many of the path's sections to measure, from the first; ``None`` measures
        them all.
    degenerate_blocks : sequence of sequence of int, optional
        The declared degenerate blocks.
    basis : solprop.modes.guide.ChannelBasis or None, optional
        The declared constant unitary change of the channel coordinates.
    n_r, n_theta : int or None, optional
        The resolution each stencil position's rule is asked for.

    Returns
    -------
    tuple
        Per position of the measured sections, section by section, one
        :class:`~solprop.modes.connection.ModalConnection` per frequency knot.
    """
    count = path.n_intervals if sections is None else int(sections)
    width = path.degree + 1
    result: list[tuple[ModalConnection, ...]] = []
    for index in range(count):
        lo, hi = float(path.z_edges[index]), float(path.z_edges[index + 1])
        inset = float(np.spacing(max(abs(lo), abs(hi), hi - lo)))
        for position, table in enumerate(path.section_tables(index)):
            z = float(table.z[0])
            if position == 0:
                z = lo + inset
            elif position == width - 1:
                z = hi - inset
            result.append(
                band_connection(
                    provider,
                    table.omega,
                    z,
                    degenerate_blocks=degenerate_blocks,
                    basis=basis,
                    n_r=n_r,
                    n_theta=n_theta,
                )
            )
    return tuple(result)


def in_frequency(
    knots: Float[np.ndarray, " n_knots"],
    values: Complex[np.ndarray, " n_knots *value"],
    omega: Float[np.ndarray, " n_omega"],
    inside: np.ndarray,
) -> Complex[np.ndarray, "*value n_omega"]:
    """Return knot values interpolated onto a run's bins, zero outside the band.

    The same not-a-knot cubic in frequency the dense tables interpolate their fields
    with (:class:`solprop.translate.modal_tables.DenseModalTable`), so a connection and
    the maps it is propagated beside are carried onto the run's grid alike.

    Parameters
    ----------
    knots : numpy.ndarray
        The knots in rad/s, increasing, shape (n_knots,).
    values : numpy.ndarray
        Values at the knots, knot axis first.
    omega : numpy.ndarray
        The run's coarse bins in rad/s, shape (n_omega,).
    inside : numpy.ndarray
        Which bins lie inside the knots, bool, shape (n_omega,).

    Returns
    -------
    numpy.ndarray
        The values with the frequency axis **last**, zero outside `inside`.
    """
    spline = CubicSpline(knots, values, axis=0, extrapolate=False)
    result = np.zeros((*np.shape(values)[1:], np.size(omega)), dtype=np.complex128)
    if np.any(inside):
        result[..., inside] = np.moveaxis(spline(np.asarray(omega)[inside]), 0, -1)
    return result
