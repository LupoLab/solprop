# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The transform between modal amplitudes and the field at quadrature nodes.

A nonlinear response is a statement about the field at a point of the cross-section, in
volts per metre; the propagated state is a set of modal amplitudes in the square root of
watts. Two matrices carry the state between them — synthesis onto the nodes, projection
back onto the modes with the quadrature weights folded in — and a scalar path along z
rescales both when the waveguide tapers. Section 4 of ``docs/dev/kernel-contract.md`` is
the class :class:`Transform`, field for field.

Mode-averaged propagation explicitly selects an effective-area transform. It has
one node, but a resolved quadrature with one node remains a modal transform. In
the mode-averaged case the matrices are one by one and
the conversion factor is ``1 / sqrt(eps0 c A_eff / 2)`` (``conventions.md`` section 4).

**The oversampling pad scale lives here.** The state is zero-padded onto the fine
spectral grid before the inverse transform, and the factor ``n_to / n_t`` that keeps the
two time-domain signals equal at coincident samples (``conventions.md`` section 3) is
folded into the synthesis rather than into a response coefficient. It has to be: a
response coefficient absorbs a field scale only when the response is homogeneous in the
field, which the Kerr term is and an ionisation rate emphatically is not. Folded into
the synthesis, the field every response sees is the physical one in volts per metre. The
crop scale on the way back is folded into the prefactor instead
(:mod:`solprop.translate.responses`), where it meets no such condition.
"""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from jaxtyping import Float

from solprop.exceptions import ModeError, TranslationError
from solprop.grid import TimeGrid
from solprop.modes.protocol import GEOMETRY_PATH_NAMES, Mode
from solprop.modes.quadrature import TransverseQuadrature, mode_matrix, taper_scale
from solprop.nonlinear.normalisation import sqrtw_to_vm
from solprop.translate.chebyshev import DEFAULT_TOL, PiecewiseChebyshev
from solprop.translate.linop import tabulate
from solprop.translate.modal_tables import ModalTransformTables
from solprop.translate.spatial import SpatialGeometry

__all__ = [
    "BUFFERS_PER_NODE",
    "DEFAULT_NODE_BUDGET",
    "DEFAULT_SPATIAL_NODE_BUDGET",
    "Transform",
    "TransformKind",
    "general_modal_transform",
    "mode_averaged_transform",
    "multimode_transform",
]

logger = logging.getLogger(__name__)

DEFAULT_NODE_BUDGET = 64 * 2**20
"""Bytes of node-chunk buffers a problem may hold, before the chunk size is reduced.

Architecture section 13 chooses the whole node set when it fits a per-problem budget and
the largest chunk that fits otherwise. Sixty-four mebibytes reproduces the chunk of
about sixteen nodes that section 8 tabulates for the comb-to-continuum target (32768
time samples, six modes, Raman) while leaving a small problem un-chunked. It is an
estimate until the benchmark suite measures the crossover.

The free-space, line and planar grids default to :data:`DEFAULT_SPATIAL_NODE_BUDGET`.
"""

DEFAULT_SPATIAL_NODE_BUDGET = 2**30
"""Node-chunk bytes a free-space, line or planar problem may hold by default (SEP 28).

A spatial evaluation pays a fixed cost per node chunk: its launches, its temporal
transforms' thread wake-ups, and its gather and scatter. A field volume is gigabytes, so
a larger group is cheap in memory. With the per-chunk copies and clears removed, one
gibibyte was the fastest of 64, 256 and 1024 MiB for the 384- and 768-square Kerr beams
on Zen 5 and on Apple silicon, 2.3 to 2.4 times faster than 64 MiB on the native
backend's march;
on an H200, IP-DOPRI5 was 1.3 times faster at one gibibyte than at 64 MiB.
"""

TransformKind = Literal["mode_averaged", "modal", "radial", "free", "slab"]
"""What a transform represents (kernel schema 26, SEP 19).

``"mode_averaged"`` is the effective-area route of :func:`mode_averaged_transform`, the
only kind whose response prefactor folds in an area conversion. ``"modal"`` is every
resolved modal transform -- a quadrature of any node count, including one, and the
general modal tables. The spatial kinds match their geometry.
"""

BUFFERS_PER_NODE = 3
"""Node-chunk buffers the fused pass holds: ``E``, ``P`` and the Raman drive."""


@dataclass(frozen=True, eq=False)
class Transform:
    """The synthesis and projection matrices and their scaling along z.

    Section 4 of the kernel contract. The matrices are built once, at the reference
    position, and `taper_scale` carries whatever z dependence a scale-invariant
    waveguide has: the normalised mode field goes as ``1 / a`` at nodes placed at
    ``r = a xi`` and the weights as ``a**2``, so multiplying the synthesis by
    ``a(0) / a(z)`` and dividing the projection by it is exact rather than an
    approximation (:meth:`solprop.modes.quadrature.ModeMatrices.rescaled`).

    Attributes
    ----------
    synthesis : numpy.ndarray
        Mode fields at the nodes, float64, shape ``(n_nodes * n_pol, n_modes)``, in
        V/(m sqrt(W)), with the oversampling pad scale ``n_to / n_t`` folded in. The
        column index is ``i * n_pol + p``.
    projection : numpy.ndarray
        The same fields with the quadrature weights folded in, float64, shape
        ``(n_modes, n_nodes * n_pol)``.
    projection_coarse : numpy.ndarray
        The embedded coarse-rule projection, same shape, for the quadrature error
        estimate. It is an alarm, never a controller.
    taper_scale : solprop.translate.chebyshev.PiecewiseChebyshev or None
        The scalar rescaling ``a(0) / a(z)`` as a table along z, or ``None`` when the
        cross-section does not change.
    node_chunk : int
        Nodes the response pass handles at a time, at least 1 and at most ``n_nodes``.
    aeff : solprop.translate.chebyshev.PiecewiseChebyshev
        Effective area in m^2 as a table along z. It is the normalisation itself when
        ``kind == 'mode_averaged'`` and a diagnostic otherwise.
    geometry : solprop.translate.spatial.SpatialGeometry or None
        Spatial data for radial/free/slab propagation; ``None`` for modal
        amplitudes. A free transform carries empty matrices and uses the declared
        FFT shape; a slab transform carries the x half of the map, the slab rule's
        ``(n_x, n_p)`` synthesis and its projections, and an FFT in y.
    node_weights : numpy.ndarray or None
        Physical transverse quadrature weights in m² at z=0, shape (n_nodes,), for
        resolved modal populations. Divide by taper_scale² at later planes. Spatial
        transforms use geometry.spatial_weights; mode-averaged transforms use aeff.
    modal_tables : solprop.translate.modal_tables.ModalTransformTables or None
        Kernel schema 12: the full E/H reference of a general modal guide, with its
        own node rule, separate synthesis and projection maps and optional factors.
        ``None`` for every route that goes through the real static matrices above.
        When it is present the general route reads these tables instead, and the
        static matrices are **declared placeholders**: they must be the single
        zero column ``(1, n_modes)`` and its two ``(n_modes, 1)`` transposes, with
        ``node_chunk = 1``, no taper and no node weights. They are mandatory
        fields of the contract that this route does not use, and a placeholder
        that is checked cannot be mistaken for a second, silently wrong transform.
        The route's own chunking is the ledger of
        :mod:`solprop.backends.numpy_ref.general_modal`, not ``node_chunk``.
    kind : {'mode_averaged', 'modal', 'radial', 'free', 'slab'}
        Kernel schema 26: what the transform represents, stored rather than inferred
        from the node count, so that a one-node quadrature is an ordinary modal
        transform. A spatial kind must equal its geometry's kind; a mode-averaged
        transform has one node, one component and neither tables nor node weights.

    Raises
    ------
    TranslationError
        If the matrix shapes disagree, or ``node_chunk`` is out of range.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import SpecifiedMode
    >>> from solprop.translate.modal import mode_averaged_transform
    >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
    >>> mode = SpecifiedMode.from_taylor(
    ...     [0.0], omega_ref=grid.omega0, aeff=1e-10
    ... )
    >>> transform = mode_averaged_transform(mode, grid, z_max=1.0)
    >>> transform.n_nodes, transform.node_chunk, transform.taper_scale is None
    (1, 1, True)
    >>> transform.kind
    'mode_averaged'
    """

    synthesis: Float[np.ndarray, "n_rows n_modes"]
    projection: Float[np.ndarray, "n_modes_p n_columns"]
    projection_coarse: Float[np.ndarray, "n_modes_c n_columns_c"]
    taper_scale: PiecewiseChebyshev | None
    node_chunk: int
    aeff: PiecewiseChebyshev
    geometry: SpatialGeometry | None = None
    node_weights: Float[np.ndarray, " n_weights"] | None = None
    modal_tables: ModalTransformTables | None = None
    kind: TransformKind = field(kw_only=True)

    def __post_init__(self) -> None:
        """Adopt read-only copies and check the shapes the annotations cannot.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        for name in ("synthesis", "projection", "projection_coarse"):
            values = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.node_weights is not None:
            weights = np.array(self.node_weights, dtype=float, copy=True)
            if (
                weights.shape != (self.n_nodes,)
                or np.any(weights < 0)
                or not np.all(np.isfinite(weights))
            ):
                raise TranslationError(
                    "node_weights must be finite, non-negative and match n_nodes"
                )
            weights.setflags(write=False)
            object.__setattr__(self, "node_weights", weights)
        rows, n_modes = self.synthesis.shape
        for name in ("projection", "projection_coarse"):
            if getattr(self, name).shape != (n_modes, rows):
                msg = (
                    f"{name} has shape {getattr(self, name).shape}, expected "
                    f"{(n_modes, rows)}, the transpose of synthesis"
                )
                raise TranslationError(msg)
        if self.geometry is not None:
            if self._n_pol != 1 or self.taper_scale is not None:
                msg = (
                    "a spatial transform applies one scalar map per state "
                    "polarisation and cannot taper"
                )
                raise TranslationError(msg)
            geometry = self.geometry
            expected = (
                (0, 0)
                if geometry.kind == "free"
                # The slab kind's matrices are the x half alone: the slab rule's
                # nodes against the channels, whose count times n_y is the columns.
                else (geometry.shape[0], geometry.n_columns // geometry.shape[1])
                if geometry.kind == "slab"
                else (self.n_modes, self.n_modes)
            )
            if self.synthesis.shape != expected:
                msg = f"{self.geometry.kind} synthesis must have shape {expected}"
                raise TranslationError(msg)
        limit = self.n_nodes if self.geometry is not None else rows
        if not 1 <= self.node_chunk <= limit:
            msg = (
                f"node_chunk must lie between 1 and the number of columns ({limit}), "
                f"got {self.node_chunk!r}"
            )
            raise TranslationError(msg)
        if self.modal_tables is not None:
            self._check_placeholders()
        self._check_kind()

    def _check_kind(self) -> None:
        """Check the stored kind against what the transform carries.

        Raises
        ------
        TranslationError
            If the kind is unknown or contradicts the geometry, the tables, the node
            count or the node weights.
        """
        kind = self.kind
        if self.geometry is not None:
            if kind != self.geometry.kind:
                msg = (
                    f"a {self.geometry.kind} geometry needs kind "
                    f"{self.geometry.kind!r}, got {kind!r}"
                )
                raise TranslationError(msg)
            return
        if kind == "mode_averaged":
            if (
                self.modal_tables is not None
                or self.n_nodes != 1
                or self._n_pol != 1
                or self.node_weights is not None
            ):
                msg = (
                    "a mode-averaged transform has one node and one component, and "
                    "carries neither modal tables nor node weights"
                )
                raise TranslationError(msg)
        elif kind != "modal":
            msg = (
                f"kind must be 'mode_averaged' or 'modal' without a geometry, "
                f"got {kind!r}"
            )
            raise TranslationError(msg)

    def _check_placeholders(self) -> None:
        """Check the static matrices a general modal guide does not use.

        Raises
        ------
        TranslationError
            If anything the general route replaces carries a value.
        """
        n_modes = self.n_modes
        expected = ((1, n_modes), (n_modes, 1), (n_modes, 1))
        matrices = (self.synthesis, self.projection, self.projection_coarse)
        if any(
            values.shape != shape or np.any(values)
            for values, shape in zip(matrices, expected, strict=True)
        ):
            msg = (
                "transform.modal_tables replaces the static matrices, which must "
                f"then be the zero placeholders of shape {expected}, got "
                f"{tuple(values.shape for values in matrices)}"
            )
            raise TranslationError(msg)
        if (
            self.node_chunk != 1
            or self.taper_scale is not None
            or self.node_weights is not None
            or self.geometry is not None
            or self._n_pol != 1
        ):
            msg = (
                "a general modal transform carries its node rule, its chunking and "
                "its z dependence in transform.modal_tables: node_chunk is 1 and "
                "the taper, the node weights and the geometry are absent"
            )
            raise TranslationError(msg)

    @property
    def n_modes(self) -> int:
        """Return the number of modes the transform carries.

        Returns
        -------
        int
            The width of ``synthesis``.

        Examples
        --------
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> from solprop.translate.modal import mode_averaged_transform
        >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
        >>> mode = SpecifiedMode.from_taylor([0.0], omega_ref=grid.omega0)
        >>> mode_averaged_transform(mode, grid, z_max=1.0).n_modes
        1
        """
        return (
            self.geometry.n_columns
            if self.geometry is not None
            else int(self.synthesis.shape[1])
        )

    @property
    def n_nodes(self) -> int:
        """Return the number of quadrature nodes, one for a mode-averaged run.

        Returns
        -------
        int
            ``synthesis.shape[0] // n_pol``; the rows carry node and polarisation
            together, so this uses the stored chunk arithmetic rather than a guess.

        Examples
        --------
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> from solprop.translate.modal import mode_averaged_transform
        >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
        >>> mode = SpecifiedMode.from_taylor([0.0], omega_ref=grid.omega0)
        >>> mode_averaged_transform(mode, grid, z_max=1.0).n_nodes
        1
        """
        return (
            math.prod(self.geometry.shape)
            if self.geometry is not None
            else int(self.synthesis.shape[0]) // self._n_pol
        )

    @property
    def n_pol(self) -> int:
        """Return the transverse components the transform resolves at each node.

        Returns
        -------
        int
            1 when the rows of `synthesis` are one component per node -- a scalar run,
            or the mode-averaged case -- and 2 when both Cartesian components are
            carried, which is the polarisation-resolved multimode set.

        Examples
        --------
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> from solprop.translate.modal import mode_averaged_transform
        >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
        >>> mode = SpecifiedMode.from_taylor([0.0], omega_ref=grid.omega0)
        >>> mode_averaged_transform(mode, grid, z_max=1.0).n_pol
        1
        """
        return self._n_pol

    @property
    def mode_averaged(self) -> bool:
        """Return whether this is the effective-area case.

        Read from the stored `kind`, never inferred from the node count: a one-node
        quadrature (SEP 19's field-scale rule at ``n_r=1``) is an ordinary modal
        transform, and a general modal guide's placeholder column is not a node.

        Returns
        -------
        bool
            ``True`` when ``kind == "mode_averaged"``.

        Examples
        --------
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> from solprop.translate.modal import mode_averaged_transform
        >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
        >>> mode = SpecifiedMode.from_taylor([0.0], omega_ref=grid.omega0)
        >>> mode_averaged_transform(mode, grid, z_max=1.0).mode_averaged
        True
        """
        return self.kind == "mode_averaged"

    _n_pol: int = 1


def _node_chunk(n_nodes: int, n_pol: int, n_to: int, budget: int) -> int:
    """Return the nodes a response pass may hold at once within a memory budget.

    Parameters
    ----------
    n_nodes : int
        Quadrature nodes.
    n_pol : int
        Polarisation components.
    n_to : int
        Fine time samples.
    budget : int
        Bytes the node-chunk buffers may occupy in total.

    Returns
    -------
    int
        The chunk size, between 1 and `n_nodes`.

    Raises
    ------
    TranslationError
        If `budget` is not positive.
    """
    if budget <= 0:
        msg = f"node_budget must be a positive number of bytes, got {budget!r}"
        raise TranslationError(msg)
    per_node = BUFFERS_PER_NODE * n_pol * n_to * np.dtype(np.complex128).itemsize
    return int(min(n_nodes, max(1, budget // per_node)))


_DECLARED_PATHS = (*GEOMETRY_PATH_NAMES, "dispersion_path")
"""The names under which a mode declares a path it follows along z.

The cross-section names of :data:`solprop.modes.protocol.GEOMETRY_PATH_NAMES` --
which :func:`solprop.nonlinear.gnlse._source_paths` walks for the same modes, from a
module this one may not import -- and one more: a specified mode whose dispersion
moves declares ``dispersion_path`` beside its area.
That one is a :class:`solprop.nonlinear.gnlse.DispersionPath` rather than a
:class:`solprop.profiles.ZProfile`, and only its breakpoints are read here. The
derivative and loss paths are not walked and need not be: a fibre that declares them
declares its area at the union of their positions, so the area path carries their
corners (:meth:`solprop.nonlinear.gnlse.SpecifiedMode._check_paths_agree`).
"""


def _sampled_areas(
    mode: Mode, z: Float[np.ndarray, " n_z"]
) -> Float[np.ndarray, " n_z"]:
    """Sample a mode's effective area and refuse a path that leaves the physics.

    A declared area path is validated at its breakpoints and its ends when the mode
    is built, which is enough for a linear or piecewise-linear one but not for a
    path given as a callable: a cosine is positive at both ends and negative in
    between. This is where the transform asks for every position it tabulates, so
    it is where an impossible area is caught, by name, instead of reaching
    ``sqrt`` as a bare ``ValueError``.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        The mode being tabulated.
    z : numpy.ndarray
        Positions in m, shape (n_z,).

    Returns
    -------
    numpy.ndarray
        Effective areas in m², shape (n_z,).

    Raises
    ------
    ModeError
        If the area is not positive and finite at one of the sampled positions.
    """
    areas = np.array([mode.aeff(float(one)) for one in z], dtype=float)
    bad = ~(np.isfinite(areas) & (areas > 0.0))
    if np.any(bad):
        position = float(np.asarray(z)[bad][0])
        raise ModeError(
            f"the effective area of {type(mode).__name__} is "
            f"{float(areas[bad][0])!r} m^2 at z = {position!r} m: an effective-area "
            "path must stay positive and finite everywhere the transform samples "
            "it, not only at its breakpoints and its ends"
        )
    return areas


def _radius_kinks(mode: Mode) -> tuple[float, ...]:
    """Return the positions where the mode's cross-section kinks.

    The union over every path the mode declares **that moves** -- its transverse
    geometry and, for a specified mode whose dispersion moves, that dispersion path
    -- and over the source mode a specified wrapper carries, so that a fibre built
    from a tapered mode keeps that taper's corners. One mechanism owns these
    breakpoints: the transform's area and taper tables split on them, and
    :func:`solprop.translate.linop.declared_paths` puts the same paths into the
    cumulative operator's hard kinks, which is what the schedule lands on.

    Parameters
    ----------
    mode : Mode
        The mode.

    Returns
    -------
    tuple of float
        The declared breakpoints in m, increasing and without duplicates, empty
        when the mode declares no path.
    """
    values: set[float] = set()
    one: Mode | None = mode
    while one is not None:
        for name in _DECLARED_PATHS:
            path = getattr(one, name, None)
            # A path that reports itself constant has no corner to land on, whatever
            # positions it was declared at: a dispersion path carrying one set at
            # every position would otherwise split the tables where nothing happens.
            if path is not None and not path.is_constant:
                values.update(float(v) for v in np.atleast_1d(path.breakpoints))
        # A specified mode built from another mode keeps that mode in its frozen
        # callables, and its geometry is that source's; the chain is finite because
        # each wrapper holds a mode that was built before it.
        one = getattr(getattr(one, "dispersion", None), "mode", None)
    return tuple(sorted(values))


def mode_averaged_transform(
    mode: Mode,
    grid: TimeGrid,
    *,
    z_max: float,
    margin: float = 0.05,
    tol: float = DEFAULT_TOL,
) -> Transform:
    """Build the one-node transform of a mode-averaged problem.

    The single "node" is the mode itself: the field a response sees is the propagated
    amplitude times ``1 / sqrt(eps0 c A_eff / 2)``
    (:func:`solprop.nonlinear.normalisation.sqrtw_to_vm`), and the polarisation the
    response returns is brought back by the reciprocal, which is already folded into
    :func:`solprop.nonlinear.normalisation.mode_average_prefactor`. So the projection is
    the identity and the synthesis carries the conversion — times the oversampling pad
    scale, for the reason in the module docstring.

    A varying effective area is carried by `taper_scale` rather than by rebuilding
    anything: with ``tau(z) = sqrt(A_eff(0) / A_eff(z))`` the synthesis at ``z`` is
    ``tau(z)`` times the stored one and the projection is the stored one divided by it,
    which reproduces the conversion at every position exactly. For a scale-invariant
    mode that is the ``a(0) / a(z)`` of
    :func:`solprop.modes.quadrature.taper_scale`, so mode-averaged and multimode tapers
    use one mechanism.

    Parameters
    ----------
    mode : Mode
        The mode; only ``aeff`` and its declared ``core_radius`` breakpoints are used.
    grid : solprop.grid.TimeGrid
        The grid; supplies the oversampling pad scale.
    z_max : float
        Length the run covers, m, positive and finite.
    margin : float, optional
        Fraction of `z_max` the tables extend beyond the end, matching the cumulative
        operator's.
    tol : float, optional
        Target bound on the uniform error of the tables relative to their value scale.

    Returns
    -------
    Transform
        Section 4 of the kernel contract, with ``n_nodes == 1``.

    Raises
    ------
    TranslationError
        If `z_max` or `margin` is invalid.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import SpecifiedMode
    >>> from solprop.nonlinear.normalisation import sqrtw_to_vm
    >>> from solprop.translate.modal import mode_averaged_transform
    >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
    >>> mode = SpecifiedMode.from_taylor(
    ...     [0.0], omega_ref=grid.omega0, aeff=1e-10
    ... )
    >>> transform = mode_averaged_transform(mode, grid, z_max=1.0)
    >>> float(transform.synthesis[0, 0]) == sqrtw_to_vm(1e-10) * grid.pad_scale
    True
    """
    span = _checked_span(z_max, margin)
    kinks = _radius_kinks(mode)
    aeff = tabulate(lambda z: _sampled_areas(mode, z), 0.0, span, kinks=kinks, tol=tol)
    reference = _sampled_areas(mode, np.array([0.0]))[0]
    constant = bool(getattr(mode, "z_constant", False)) or bool(
        getattr(mode, "radius_is_constant", False)
    )
    taper = (
        None
        if constant
        else tabulate(
            lambda z: np.sqrt(reference / _sampled_areas(mode, z)),
            0.0,
            span,
            kinks=kinks,
            tol=tol,
        )
    )
    synthesis = np.array([[sqrtw_to_vm(reference) * grid.pad_scale]])
    identity = np.array([[1.0]])
    logger.debug(
        "mode-averaged transform: A_eff(0) = %.6e m^2 over %d intervals, taper %s",
        reference,
        aeff.degrees.size,
        "tabulated" if taper is not None else "absent",
    )
    return Transform(
        synthesis=synthesis,
        projection=identity,
        projection_coarse=identity,
        taper_scale=taper,
        node_chunk=1,
        aeff=aeff,
        _n_pol=1,
        kind="mode_averaged",
    )


def multimode_transform(  # noqa: PLR0913 - the documented signature takes every option
    modes: Sequence[Mode],
    quadrature: TransverseQuadrature,
    grid: TimeGrid,
    *,
    z_max: float,
    components: Sequence[str] = ("x", "y"),
    node_budget: int = DEFAULT_NODE_BUDGET,
    margin: float = 0.05,
    tol: float = DEFAULT_TOL,
) -> Transform:
    """Build the fixed-quadrature transform of a multimode problem.

    The matrices are :func:`solprop.modes.quadrature.mode_matrix`'s, built
    once at the reference position, with the oversampling pad scale folded into the
    synthesis; the projection is the dual basis as ``mode_matrix`` builds it, so the
    pair a backend applies inverts to ``pad_scale`` times the identity on an
    orthonormal set. When the set is scale invariant and its radius varies,
    `taper_scale` tabulates ``a(0) / a(z)`` so the matrices are rescaled rather than
    rebuilt; a set whose cross-section changes shape would need rebuilding and is
    refused here rather than approximated.

    Parameters
    ----------
    modes : sequence of Mode
        The ordered mode set.
    quadrature : solprop.modes.quadrature.TransverseQuadrature
        The fixed rule, with the Kronrod extension when an error estimate is wanted.
    grid : solprop.grid.TimeGrid
        The grid; supplies the pad scale and the fine time length the chunk is sized
        from.
    z_max : float
        Length the run covers, m, positive and finite.
    components : sequence of str, optional
        Transverse components carried: ``("x", "y")`` for a vector run, one of them for
        a scalar run.
    node_budget : int, optional
        Bytes the node-chunk buffers may occupy. Default :data:`DEFAULT_NODE_BUDGET`.
    margin : float, optional
        Fraction of `z_max` the tables extend beyond the end.
    tol : float, optional
        Target bound on the uniform error of the tables relative to their value scale.

    Returns
    -------
    Transform
        Section 4 of the kernel contract.

    Raises
    ------
    TranslationError
        If `z_max`, `margin` or `node_budget` is invalid, or the set varies along z in a
        way the rescaling cannot express.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.modes import MarcatiliMode, transverse_quadrature
    >>> from solprop.translate.modal import multimode_transform
    >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, m=m) for m in (1, 2)]
    >>> rule = transverse_quadrature(modes, n_r=16, radial_rule="gauss_kronrod")
    >>> transform = multimode_transform(modes, rule, grid, z_max=1.0,
    ...                                 components=("x",))
    >>> transform.n_modes, transform.n_nodes
    (2, 17)
    """
    span = _checked_span(z_max, margin)
    if not modes:
        msg = "a multimode transform needs at least one mode"
        raise TranslationError(msg)
    reference = modes[0]
    kinks = _radius_kinks(reference)
    constant = all(bool(getattr(one, "radius_is_constant", False)) for one in modes)
    if not constant and not all(one.scale_invariant for one in modes):
        msg = (
            "a mode set whose cross-section changes along z and is not scale invariant "
            "cannot be carried by a single rescaling; its matrices would have to be "
            "rebuilt, which the kernel contract has no field for"
        )
        raise TranslationError(msg)
    matrices = mode_matrix(modes, quadrature, 0.0, components=components)
    taper = (
        None
        if constant
        else tabulate(
            lambda z: np.array([taper_scale(reference, float(one)) for one in z]),
            0.0,
            span,
            kinks=kinks,
            tol=tol,
        )
    )
    aeff = tabulate(
        lambda z: np.array([reference.aeff(float(one)) for one in z]),
        0.0,
        span,
        kinks=kinks,
        tol=tol,
    )
    chunk = _node_chunk(matrices.n_nodes, matrices.n_pol, grid.n_to, node_budget)
    logger.debug(
        "multimode transform: %d modes on %d nodes x %d polarisations, chunk %d, "
        "taper %s",
        len(modes),
        matrices.n_nodes,
        matrices.n_pol,
        chunk,
        "tabulated" if taper is not None else "absent",
    )
    return Transform(
        synthesis=matrices.synthesis * grid.pad_scale,
        projection=matrices.projection,
        projection_coarse=matrices.projection_coarse,
        taper_scale=taper,
        node_chunk=chunk,
        aeff=aeff,
        _n_pol=matrices.n_pol,
        node_weights=quadrature.weights(reference.dimlimits(0.0)),
        kind="modal",
    )


def general_modal_transform(
    modes: Sequence[Mode],
    tables: ModalTransformTables,
    *,
    z_max: float,
    margin: float = 0.05,
    tol: float = DEFAULT_TOL,
) -> Transform:
    """Build the transform of a general modal guide: the tables and placeholders.

    The general modal route synthesises and projects through the
    frequency-dependent complex maps of `tables`, not through a real static
    matrix, so section 4's three matrices are the checked zero placeholders
    :class:`Transform` describes and every other field of the group is absent. The
    one real entry is ``aeff``, which is a **diagnostic**
    here as it is for any multimode transform: the scalar surrogate area of the
    first channel, which is what the legacy mode-averaged interfaces mean by an
    area and not the nonlinear coefficient of a vector channel.

    Parameters
    ----------
    modes : sequence of Mode
        The guide's channels, in state order; only the first channel's ``aeff``
        is read.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The general guide's E/H reference and execution maps, already tabulated on
        the run's own canonical grids.
    z_max : float
        Length the run covers, m, positive and finite.
    margin : float, optional
        Fraction of `z_max` the tables extend beyond the end.
    tol : float, optional
        Target bound on the uniform error of the area table.

    Returns
    -------
    Transform
        Section 4 of the kernel contract, carrying ``modal_tables``.

    Raises
    ------
    TranslationError
        If `z_max` or `margin` is invalid, or no channel was given.
    """
    span = _checked_span(z_max, margin)
    if not modes:
        msg = "a general modal transform needs the guide's channels in state order"
        raise TranslationError(msg)
    reference = modes[0]
    n_modes = int(tables.electric.shape[3])
    aeff = tabulate(
        lambda z: np.array([reference.aeff(float(one)) for one in z]),
        0.0,
        span,
        kinks=_radius_kinks(reference),
        tol=tol,
    )
    logger.debug(
        "general modal transform: %d channels on %d nodes, %d fine frequencies, "
        "gauge %s",
        n_modes,
        tables.electric.shape[1],
        tables.omega_fine.size,
        tables.gauge_id,
    )
    return Transform(
        synthesis=np.zeros((1, n_modes)),
        projection=np.zeros((n_modes, 1)),
        projection_coarse=np.zeros((n_modes, 1)),
        taper_scale=None,
        node_chunk=1,
        aeff=aeff,
        modal_tables=tables,
        kind="modal",
    )


def _checked_span(z_max: float, margin: float) -> float:
    """Return the end of the tabulated range, validating both arguments.

    Parameters
    ----------
    z_max : float
        Length the run covers, m.
    margin : float
        Fraction of `z_max` to extend beyond it.

    Returns
    -------
    float
        ``z_max (1 + margin)``.

    Raises
    ------
    TranslationError
        If `z_max` is not positive and finite or `margin` is negative or not finite.
    """
    if not (math.isfinite(z_max) and z_max > 0.0):
        msg = f"z_max must be positive and finite, got {z_max!r}"
        raise TranslationError(msg)
    if not (math.isfinite(margin) and margin >= 0.0):
        msg = f"margin must be non-negative and finite, got {margin!r}"
        raise TranslationError(msg)
    return z_max * (1.0 + margin)
