# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Expand what a run reads of a moving guide's sampled path, along z.

A moving guide is sampled in :mod:`solprop.modes.path`: a
:class:`~solprop.modes.path.ModalPathTable` of snapshots at each section's
Chebyshev-Lobatto points, in the provider's own gauge. This module turns that path
into what the seam carries. :func:`path_transform_tables` builds the flat
:class:`~solprop.translate.modal_tables.ModalTransformTables` record whose synthesis
and projection maps are Chebyshev coefficients along z per section
(:func:`section_coefficients`); a stage combines them with :func:`transform_weights`
over the section :func:`transform_section` picks. :func:`path_connections` measures
the moving-basis connection at every sampled position, and :func:`in_frequency`
carries knot values onto a run's frequency bins with the tables' own cubic.
"""

import logging
from collections.abc import Sequence

import numpy as np
from jaxtyping import Complex, Float
from numpy.polynomial import chebyshev as chebyshev_basis
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

from solprop.exceptions import ModeError
from solprop.modes.connection import ModalConnection, band_connection
from solprop.modes.path import ModalPathTable
from solprop.modes.quadrature import lobatto_coefficients
from solprop.modes.vector import ChannelBasis, ModalProvider
from solprop.translate.chebyshev import _EDGE_SLACK_FRACTION
from solprop.translate.modal_tables import (
    ModalTransformTables,
    coordinate_dual,
    projection_map,
)

__all__ = [
    "in_frequency",
    "path_connections",
    "path_transform_tables",
    "section_coefficients",
    "transform_section",
    "transform_weights",
]

logger = logging.getLogger(__name__)


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
                projection[section, position, :, :, :, column] = projection_map(
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
            dual[row, :, :, :, column] = coordinate_dual(sample, table.nodes.weights)
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


def path_connections(
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
    is the gauge the path's snapshots are in
    (:func:`~solprop.modes.path.path_snapshot`). A position on a section's **end** is
    measured one ulp inside it, so that its stencil is taken on that section's own
    side of an edge: at a kink of the geometry the slope jumps and the connection with
    it, and the section that closes there must read its own one-sided limit (the rule
    of the coupled flow's :func:`solprop.translate.coupled.flow_weights`, applied where
    it is sampled).

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
    with (:class:`solprop.modes.tables.DenseModalTable`), so a connection and
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
