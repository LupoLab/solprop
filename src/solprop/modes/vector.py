# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Represent internal-convention Maxwell samples on physical area quadrature.

The records every vector modal provider and table shares: :class:`ModalNodeRule`
(Cartesian nodes, area weights and material regions; :func:`radial_node_rule` builds
the interface-split polar one), :class:`ModalSample` (one
frequency and position's E/H, generator and flux metric), the
:class:`ModalProvider` protocol that produces them, and :func:`flux_product`, the
Hermitian Poynting product. :class:`ChannelBasis` declares a unitary change of a
guide's degenerate channel coordinates (:func:`circular_pair`, read against
:func:`provider_channels`), unitary to :data:`BASIS_UNITARY_ATOL`. These physics
records do not extend the propagation kernel contract. Bound, lossless modes use
the Hermitian Poynting product and true-power amplitudes.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from jaxtyping import Bool, Complex, Float, Int32

from solprop._display import Displayed, engineering, line
from solprop.exceptions import ModeError
from solprop.modes.material import JsonValue, MaterialRegion
from solprop.modes.protocol import ChannelLabel
from solprop.modes.quadrature import quadrature_rule

__all__ = [
    "BASIS_UNITARY_ATOL",
    "ChannelBasis",
    "ModalNodeRule",
    "ModalProvider",
    "ModalSample",
    "circular_pair",
    "flux_product",
    "provider_channels",
    "radial_node_rule",
]

BASIS_UNITARY_ATOL = 1e-12
"""Largest ``norm(U.H U - I, 2)`` a declared change of basis is still called unitary.

Dimensionless. A declared transform is admitted only where it is a **unitary** change
of coordinates of an exactly degenerate block: the flux metric then transforms as
``W' = U.H W U``, which is the identity wherever ``W`` was, and the generator
``U.H diag(beta) U`` stays diagonal. A nonunitary transform does neither -- it is a
leaky or skewed basis whose metric is genuinely off the identity and whose flow is a
matrix exponential -- and nothing in this package has measured one, so it is refused
here rather than approximated.

**The number.** :func:`circular_pair`'s matrix has entries ``+-1/sqrt(2)`` and
``+-i/sqrt(2)``; ``1/sqrt(2)`` is exact to a half ulp and the products that form
``U.H U`` cancel to the last bit, so its measured defect is at the 1e-16 level.
A budget of 1e-12 is four orders above that -- room for a matrix assembled from
measured numbers rather than from a literal -- and twelve orders below any
departure a reader would call a different basis.

**The room is real, because every later gate honours this budget.** A transform
that is unitary only to ``E`` moves a knot's measured ``norm(W - I, 2)`` by
``norm(E + E.H, 2)``, which is this defect and not the metric's, so the per-knot
invariance check of :func:`solprop.modes.tables.apply_channel_basis`
takes this constant as its floor rather than the rounding of its own product
(2.8e-14 for two channels, which would refuse a declaration that had just passed
here). The four orders are therefore the room a declaration actually has, not the
room it has until the next gate.
"""


@dataclass(frozen=True, eq=False)
class ModalNodeRule(Displayed):
    """Own Cartesian quadrature nodes and their material regions.

    Attributes
    ----------
    xy : numpy.ndarray
        Finite Cartesian positions in m, shape (n_nodes_xy, 2).
    weights : numpy.ndarray
        Positive area weights in m², shape (n_nodes_weights,).
    region_index : numpy.ndarray
        Non-negative int32 region indices, shape (n_nodes_region_index,).
    identity : collections.abc.Mapping
        Serializable construction and tail-certificate metadata.
    """

    xy: Float[np.ndarray, "n_nodes_xy 2"]
    weights: Float[np.ndarray, " n_nodes_weights"]
    region_index: Int32[np.ndarray, " n_nodes_region_index"]
    identity: Mapping[str, JsonValue]

    def __repr__(self) -> str:
        """Return a one-line description: the node and region counts.

        Returns
        -------
        str
            For example ``ModalNodeRule(60 nodes, 2 regions)``.
        """
        return line(
            "ModalNodeRule",
            [
                ("", f"{len(self.weights)} nodes"),
                ("", f"{len(set(self.region_index.tolist()))} regions"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        return "ModalNodeRule: Cartesian nodes with declared material ownership", [
            ("nodes", len(self.weights)),
            ("regions", len(set(self.region_index.tolist()))),
            ("identity", ", ".join(sorted(self.identity)) or "none"),
        ]

    def __post_init__(self) -> None:
        """Copy, validate and freeze the physical node arrays.

        Raises
        ------
        ModeError
            If arrays disagree in shape, contain invalid values or have no nodes.
        """
        xy = np.array(self.xy, dtype=np.float64, copy=True)
        weights = np.array(self.weights, dtype=np.float64, copy=True)
        regions = np.array(self.region_index, copy=True)
        if xy.ndim != 2 or xy.shape[1] != 2 or not xy.shape[0]:
            raise ModeError("xy must contain at least one Cartesian pair")
        if weights.shape != (len(xy),) or regions.shape != weights.shape:
            raise ModeError("node coordinates, weights and regions must agree")
        if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(weights)):
            raise ModeError("node coordinates and weights must be finite")
        if np.any(weights <= 0):
            raise ModeError("area weights must be positive")
        if regions.dtype != np.int32 or np.any(regions < 0):
            raise ModeError("region_index must contain non-negative int32 indices")
        for name, value in (
            ("xy", xy),
            ("weights", weights),
            ("region_index", regions),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "identity", dict(self.identity))


def radial_node_rule(
    interfaces: Sequence[float],
    outer_radius: float,
    *,
    n_r: int,
    n_theta: int,
) -> ModalNodeRule:
    """Build interface-split Cartesian nodes from the existing polar panel rule.

    Parameters
    ----------
    interfaces : sequence of float
        Strictly increasing positive material radii in m, inside outer_radius.
    outer_radius : float
        Positive finite truncation radius in m; tail certification is separate.
    n_r : int
        At least two Gauss nodes in each radial panel.
    n_theta : int
        At least two angular nodes; caller certifies angular resolution.

    Returns
    -------
    ModalNodeRule
        Positive area weights and one int32 material index per node.

    Raises
    ------
    ModeError
        If the radius or interface arrangement is invalid.
    """
    if not np.isfinite(outer_radius) or outer_radius <= 0:
        raise ModeError("outer_radius must be positive and finite")
    rule = quadrature_rule(
        "polar",
        full=True,
        n_r=n_r,
        n_theta=n_theta,
        breaks=tuple(float(r / outer_radius) for r in interfaces),
    )
    limits = ("polar", (0.0, 0.0), (outer_radius, 2 * np.pi))
    polar = rule.nodes(limits)
    xy = np.column_stack(
        (polar[:, 0] * np.cos(polar[:, 1]), polar[:, 0] * np.sin(polar[:, 1]))
    )
    return ModalNodeRule(
        xy,
        rule.weights(limits),
        np.searchsorted(interfaces, polar[:, 0], side="left").astype(np.int32),
        {
            "interfaces": list(interfaces),
            "outer_radius": outer_radius,
            "n_r": n_r,
            "n_theta": n_theta,
        },
    )


@dataclass(frozen=True, eq=False)
class ModalSample(Displayed):
    """Own a full Maxwell mode set in the internal FFT convention.

    Attributes
    ----------
    omega : float
        Positive angular frequency in rad/s.
    z : float
        Finite longitudinal position in m.
    electric : numpy.ndarray
        E in V/m per sqrt(W), shape (n_nodes_electric, 3, n_modes_electric).
    magnetic : numpy.ndarray
        H in A/m per sqrt(W), shape (n_nodes_magnetic, 3, n_modes_magnetic).
    generator : numpy.ndarray
        Internal generator in 1/m, square (n_modes_generator_out, n_modes_generator_in).
        Unframed: it carries the absolute propagation constants, and the co-moving
        frame is subtracted once by the translation layer (see :class:`ModalProvider`).
        It is in the **internal** convention of ``docs/dev/conventions.md`` section 2,
        so a diagonal entry is ``G_cc = -i (omega/c) conj(n_eff)``: its imaginary part
        is ``-beta`` and its real part is ``-alpha/2``. A **passive** channel therefore
        has ``Re G_cc <= 0``, and ``Re G_cc > 0`` is gain. This is the same sign
        convention the linear operator has, ``exp(G z)`` for a forward amplitude. It
        is **not** the optical convention of an ``IndexTable``
        (:mod:`solprop.modes.tabulated`), which stores ``n_eff`` with
        ``Im n_eff > 0`` for loss; the bridge between the two conjugates, as every
        optics/internal boundary does.
    flux_metric : numpy.ndarray
        Dimensionless positive Hermitian flux,
        shape (n_modes_flux_out, n_modes_flux_in).
    boundary_residual : numpy.ndarray
        Non-negative scaled boundary defects, shape (n_modes_boundary_residual,).
    valid : numpy.ndarray
        Validity mask, shape (n_modes_valid,). Every provider in this package
        refuses an unsupported channel by name rather than masking it, so a
        returned sample has this mask all ``True``; tracking refuses a sample
        that does not. The field exists so that a future provider which can
        genuinely return a partial set has somewhere to say so.
    backward_electric : numpy.ndarray or None
        Reciprocal E, shape (n_nodes_backward_electric, 3, n_modes_backward_electric).
    backward_magnetic : numpy.ndarray or None
        Reciprocal H, shape (n_nodes_backward_magnetic, 3, n_modes_backward_magnetic).
    reciprocal_gram : numpy.ndarray or None
        Dimensionless reciprocal flux,
        shape (n_modes_reciprocal_out, n_modes_reciprocal_in).
    gauge_id : str
        Non-empty phase and coordinate convention identifier.
    physical_id : str
        Non-empty physical provider identifier.
    """

    omega: float
    z: float
    electric: Complex[np.ndarray, "n_nodes_electric 3 n_modes_electric"]
    magnetic: Complex[np.ndarray, "n_nodes_magnetic 3 n_modes_magnetic"]
    generator: Complex[np.ndarray, "n_modes_generator_out n_modes_generator_in"]
    flux_metric: Complex[np.ndarray, "n_modes_flux_out n_modes_flux_in"]
    boundary_residual: Float[np.ndarray, " n_modes_boundary_residual"]
    valid: Bool[np.ndarray, " n_modes_valid"]
    backward_electric: (
        Complex[np.ndarray, "n_nodes_backward_electric 3 n_modes_backward_electric"]
        | None
    )
    backward_magnetic: (
        Complex[np.ndarray, "n_nodes_backward_magnetic 3 n_modes_backward_magnetic"]
        | None
    )
    reciprocal_gram: (
        Complex[np.ndarray, "n_modes_reciprocal_out n_modes_reciprocal_in"] | None
    )
    gauge_id: str
    physical_id: str

    def __repr__(self) -> str:
        """Return a one-line description: the extents and the sampled shape.

        Returns
        -------
        str
            For example ``ModalSample(1 channel, 60 nodes, 2.256e+15 rad/s, 0 m)``.
        """
        nodes, _, modes = np.shape(self.electric)
        return line(
            "ModalSample",
            [
                ("", f"{modes} channel{'' if modes == 1 else 's'}"),
                ("", f"{nodes} nodes"),
                ("", f"{float(self.omega):.4e} rad/s"),
                ("", engineering(float(self.z), "m")),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        nodes, _, modes = np.shape(self.electric)
        return "ModalSample: internal-convention E/H on physical nodes", [
            ("omega", f"{float(self.omega):.6e} rad/s"),
            ("z", engineering(float(self.z), "m")),
            ("channels", modes),
            ("nodes", nodes),
            ("gauge", self.gauge_id),
            # The identity is a JSON document; the block's own column cut shows its
            # head, which is where the channels and the model are.
            ("provider", self.physical_id),
            ("residual", f"{float(np.max(self.boundary_residual)):.3e} largest"),
            ("reciprocal", "yes" if self.reciprocal_gram is not None else "no"),
        ]

    def __post_init__(self) -> None:
        """Validate extents, finite data and positive physical flux in production.

        Raises
        ------
        ModeError
            If metadata, shapes, reciprocal data or the flux metric are invalid.
        """
        if not np.isfinite(self.omega) or self.omega <= 0 or not np.isfinite(self.z):
            raise ModeError("omega must be positive and z finite")
        if not self.gauge_id or not self.physical_id:
            raise ModeError("physical_id and gauge_id must be non-empty")
        reciprocal = (
            self.backward_electric,
            self.backward_magnetic,
            self.reciprocal_gram,
        )
        if any(value is None for value in reciprocal) and not all(
            value is None for value in reciprocal
        ):
            raise ModeError(
                "backward fields and reciprocal Gram must be supplied together"
            )
        # Coerce before inspecting, as ModalNodeRule does: a list or a nested
        # sequence has no ndim, and reading one raises AttributeError instead of
        # the ModeError a public boundary owes its caller.
        electric = np.array(self.electric, dtype=np.complex128, copy=True)
        if electric.ndim != 3 or electric.shape[1] != 3:
            raise ModeError("electric must have three Cartesian components")
        nodes, _, modes = electric.shape
        if not nodes or not modes:
            raise ModeError("a modal sample needs nodes and modes")
        shapes = {
            "electric": (nodes, 3, modes),
            "magnetic": (nodes, 3, modes),
            "generator": (modes, modes),
            "flux_metric": (modes, modes),
            "boundary_residual": (modes,),
            "valid": (modes,),
            "backward_electric": (nodes, 3, modes),
            "backward_magnetic": (nodes, 3, modes),
            "reciprocal_gram": (modes, modes),
        }
        if np.asarray(self.valid).dtype != np.bool_:
            raise ModeError("valid must be a boolean mask")
        for name, shape in shapes.items():
            value = electric if name == "electric" else getattr(self, name)
            if value is None:
                continue
            dtype = (
                np.bool_
                if name == "valid"
                else np.float64
                if name == "boundary_residual"
                else np.complex128
            )
            array = np.array(value, dtype=dtype, copy=True)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ModeError(f"{name} must have finite values and shape {shape}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if np.any(self.boundary_residual < 0):
            raise ModeError("boundary residuals must be non-negative")
        self._validate_flux(modes)

    def _validate_flux(self, modes: int) -> None:
        """Require positive Hermitian flux and a nonsingular reciprocal pairing.

        - ``modes``: validated positive channel count, dimensionless.
        """
        tolerance = 64 * modes * np.finfo(float).eps * np.linalg.norm(self.flux_metric)
        if np.linalg.norm(self.flux_metric - self.flux_metric.conj().T) > tolerance:
            raise ModeError("flux_metric must be Hermitian")
        if np.linalg.eigvalsh(self.flux_metric)[0] <= 0:
            raise ModeError("flux_metric must be positive definite")
        if (
            self.reciprocal_gram is not None
            and np.linalg.matrix_rank(self.reciprocal_gram) < modes
        ):
            raise ModeError("reciprocal_gram must be nonsingular")


@runtime_checkable
class ModalProvider(Protocol):
    """Supply internal-convention full-vector modes on common physical nodes.

    **The generator is unframed.** :attr:`ModalSample.generator` carries the absolute
    propagation constants, ``-i diag(beta)`` for a diagonal provider, not a co-moving
    generator: the frame is subtracted exactly once, by the translation layer's one
    shared frame for the whole mode set (a per-mode frame destroys walk-off; see
    docs/dev/decisions.md). A provider that removed a frame of its own
    would have it removed twice.
    """

    def node_rule(
        self,
        omega: float,
        z: float = 0.0,
        *,
        n_r: int | None = None,
        n_theta: int | None = None,
    ) -> ModalNodeRule:
        """Build the physical quadrature this provider's fields are resolved on.

        One signature for every provider: the rule depends on the frequency, because
        a bound field's extent does, and on the position, because the geometry may.
        An implementation that does not resolve its domain in frequency still takes
        the argument and uses it to certify the power its rule discards.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s, inside :attr:`omega_domain`.
        z : float, optional
            Position in m inside :attr:`z_domain`.
        n_r : int or None, optional
            Radial Gauss nodes per panel, at least two; ``None`` asks for the count
            the implementation's own geometry needs, which is the one a caller that
            must not invent a resolution wants. This protocol declares no number of
            its own: it has no geometry to choose one from.
        n_theta : int or None, optional
            Angular nodes; ``None`` lets the implementation choose the count its
            channels' azimuthal orders require.

        Returns
        -------
        ModalNodeRule
            Nodes with declared material ownership and a tail certificate in the
            identity.
        """
        ...

    def solve(self, omega: float, z: float, *, nodes: ModalNodeRule) -> ModalSample:
        """Evaluate the modes at one frequency and longitudinal position.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s within omega_domain.
        z : float
            Longitudinal coordinate in m within z_domain.
        nodes : ModalNodeRule
            Physical quadrature with declared region ownership.

        Returns
        -------
        ModalSample
            Three-component E/H, generator and physical diagnostics. An
            implementation refuses an unsupported channel by name; it does not
            return a sample whose ``valid`` mask is partly ``False``.
        """
        ...

    @property
    def regions(self) -> tuple[MaterialRegion, ...]:  # numpydoc ignore=RT01
        """Ordered material-region records."""
        ...

    @property
    def z_breakpoints(self) -> tuple[float, ...]:  # numpydoc ignore=RT01
        """Physical path breakpoints, m."""
        ...

    @property
    def omega_domain(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Declared angular-frequency interval, rad/s."""
        ...

    @property
    def z_domain(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Declared longitudinal interval, m."""
        ...

    @property
    def identity(self) -> Mapping[str, JsonValue]:  # numpydoc ignore=RT01
        """Physical identity, including ``spectral_convention='internal'``."""
        ...


def flux_product(
    electric: Complex[np.ndarray, "n_nodes 3 n_modes"],
    magnetic: Complex[np.ndarray, "n_nodes 3 n_modes"],
    weights: Float[np.ndarray, " n_nodes"],
) -> Complex[np.ndarray, "n_modes n_modes"]:
    """Integrate the Hermitian Poynting product.

    Parameters
    ----------
    electric : numpy.ndarray
        E in V/m per sqrt(W), shape (n_nodes, 3, n_modes).
    magnetic : numpy.ndarray
        H in A/m per sqrt(W), shape (n_nodes, 3, n_modes).
    weights : numpy.ndarray
        Area weights in m², shape (n_nodes,).

    Returns
    -------
    numpy.ndarray
        Dimensionless Hermitian flux metric, shape (n_modes, n_modes).
    """
    cross = np.einsum("qm,qn,q->mn", magnetic[:, 1].conj(), electric[:, 0], weights)
    cross -= np.einsum("qm,qn,q->mn", magnetic[:, 0].conj(), electric[:, 1], weights)
    return (cross + cross.conj().T) / 4


@dataclass(frozen=True, eq=False)
class ChannelBasis(Displayed):
    """Declare the unitary coordinates a guide's degenerate channels are given in.

    A guide's channels arrive in whatever basis its provider solved them in, which
    for an azimuthally degenerate family is the real parity pair: the ``cos`` and
    ``sin`` patterns :class:`~solprop.modes.protocol.ChannelLabel` defines. Any
    unitary combination of an
    **exactly degenerate** block is as good a basis, and one of them -- the circular
    pair of :func:`circular_pair` -- diagonalises the isotropic Kerr tensor's
    polarisation dependence. This record is the declaration of that choice: the
    matrix, the blocks it acts inside, the labels the transformed channels carry and
    the identity a file records and a resume compares.

    **The transform acts on the reference, not on the state.** With ``e`` the
    provider's nodal field rows and ``U`` this matrix, the declared basis is
    ``e' = e U``, so a physical field whose reference-basis amplitudes are ``c`` has
    declared-basis amplitudes ``d = U.H c``: the columns of ``U`` are the new basis
    vectors written in the old one. Every derived object follows from that one rule
    -- ``W' = U.H W U``, ``G' = U.H G U``, ``Q' = U.H Q``, ``S' = S U`` -- and the
    noise draw made in the reference basis reaches the declared one as ``U.H`` times
    itself.

    **Whether a block is degenerate is decided where the table is built**, not here
    (:func:`solprop.modes.tables.apply_channel_basis`): ``U.H diag(beta) U``
    is diagonal when the block's propagation constants are equal and is a coupled
    matrix flow, ``linear.coupled_flow``, when they are not.

    Attributes
    ----------
    transform : numpy.ndarray
        The unitary ``U``, shape (n_modes, n_modes), dimensionless. It is the
        identity outside `blocks`, and unitary to :data:`BASIS_UNITARY_ATOL`.
    blocks : tuple of tuple of int
        Disjoint channel-index blocks ``U`` acts inside, each with at least two
        members. They must be blocks the guide declared degenerate.
    labels : tuple of ChannelLabel
        One label per channel of the declared basis, in row order: the provider's
        own outside the blocks and the transformed ones inside them.
    gauge_id : str
        Non-empty identity of this declaration, composed into the tables' and the
        metric's ``gauge_id`` so that a run in one basis cannot resume in another.

    Raises
    ------
    solprop.exceptions.ModeError
        If the matrix is not square, finite, unitary or block-diagonal on `blocks`,
        if the blocks are not disjoint valid sets of at least two channels, if the
        label count disagrees with the matrix, or if the identity is empty.

    Examples
    --------
    >>> from solprop.modes.guide import circular_pair
    >>> from solprop.modes.protocol import ChannelLabel
    from solprop.modes.quadrature import quadrature_rule
    >>> pair = (
    ...     ChannelLabel("HE", 1, 1, "even", None, "HE1,1:even"),
    ...     ChannelLabel("HE", 1, 1, "odd", None, "HE1,1:odd"),
    ... )
    >>> basis = circular_pair(pair, (0, 1))
    >>> basis.gauge_id
    'circular(0,1)'
    >>> [label.display_label for label in basis.labels]
    ['HE1,1:left', 'HE1,1:right']
    """

    transform: Complex[np.ndarray, "n_modes_basis_out n_modes_basis_in"]
    blocks: tuple[tuple[int, ...], ...]
    labels: tuple[ChannelLabel, ...]
    gauge_id: str

    def __repr__(self) -> str:
        """Return a one-line description, never the matrix.

        Returns
        -------
        str
            For example ``ChannelBasis(circular(0,1), 2 channels, 1 block)``.
        """
        blocks = len(self.blocks)
        return line(
            "ChannelBasis",
            [
                ("", self.gauge_id),
                (
                    "",
                    f"{len(self.labels)} channel{'' if len(self.labels) == 1 else 's'}",
                ),
                ("", f"{blocks} block{'' if blocks == 1 else 's'}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        return "ChannelBasis: a declared unitary change of a guide's channel basis", [
            ("gauge", self.gauge_id),
            ("channels", ", ".join(label.display_label for label in self.labels)),
            ("blocks", ", ".join(str(list(block)) for block in self.blocks)),
            ("unitary defect", f"{self.unitary_defect:.3e}"),
        ]

    def __post_init__(self) -> None:
        """Copy, validate and freeze the matrix, and check the blocks and labels.

        Raises
        ------
        solprop.exceptions.ModeError
            Naming what is out of range.
        """
        matrix = np.array(self.transform, dtype=np.complex128, order="C", copy=True)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or not matrix.shape[0]
            or not np.all(np.isfinite(matrix))
        ):
            raise ModeError(
                "a channel basis transform must be a finite non-empty square "
                f"matrix, got shape {matrix.shape}"
            )
        matrix.setflags(write=False)
        object.__setattr__(self, "transform", matrix)
        object.__setattr__(
            self,
            "blocks",
            tuple(tuple(int(one) for one in block) for block in self.blocks),
        )
        object.__setattr__(self, "labels", tuple(self.labels))
        if not self.gauge_id:
            raise ModeError("a channel basis needs a non-empty gauge_id")
        count = matrix.shape[0]
        if len(self.labels) != count:
            raise ModeError(
                f"a channel basis needs one label per channel: the transform is "
                f"{count} wide and {len(self.labels)} label(s) were given"
            )
        self._check_blocks(count)
        defect = self.unitary_defect
        if not (defect <= BASIS_UNITARY_ATOL):
            raise ModeError(
                f"a declared change of basis must be unitary: this one is "
                f"{defect:.4g} from it, above the {BASIS_UNITARY_ATOL:.4g} budget. "
                "A nonunitary transform is a leaky or skewed basis whose flux "
                "metric is genuinely off the identity and whose flow is a coupled "
                "matrix exponential; neither has been measured in this package, and "
                "the coupled flow carries a unitary change of basis only"
            )

    def _check_blocks(self, count: int) -> None:
        """Require disjoint valid blocks and a matrix that is the identity outside."""
        seen: set[int] = set()
        membership = np.full(count, -1, dtype=int)
        for position, block in enumerate(self.blocks):
            if (
                len(block) < 2
                or len(set(block)) != len(block)
                or any(not 0 <= one < count for one in block)
                or seen.intersection(block)
            ):
                raise ModeError(
                    "a channel basis acts inside disjoint blocks of at least two "
                    f"distinct channels of the {count} it describes, got "
                    f"{[list(one) for one in self.blocks]}"
                )
            seen.update(block)
            membership[list(block)] = position
        # Outside a declared block the transform must be the identity: that is what
        # makes "this basis mixes only these channels" a checkable statement, and it
        # is what the exact-degeneracy gate on each block is then enough to cover.
        same = membership[:, None] == membership[None, :]
        allowed = same & (membership[:, None] >= 0)
        outside = self.transform - np.eye(count)
        if np.any(outside[~allowed] != 0.0):
            raise ModeError(
                "a channel basis transform must be the identity outside its "
                f"declared blocks {[list(one) for one in self.blocks]}: it mixes "
                "channels this guide has not declared degenerate, and the exact "
                "degeneracy that keeps the generator diagonal is checked per block"
            )

    @property
    def unitary_defect(self) -> float:  # numpydoc ignore=RT01
        """``norm(U.H U - I, 2)``, how far the declaration is from unitary."""
        matrix = self.transform
        product = matrix.conj().T @ matrix
        return float(np.linalg.norm(product - np.eye(len(matrix)), ord=2))

    @property
    def identity(self) -> Mapping[str, JsonValue]:  # numpydoc ignore=RT01
        """What a run records of this declaration, and a resume compares.

        The matrix enters **by value**: two different unitary transforms of the same
        block are two different bases, and the block list alone does not tell them
        apart. Its entries are declared numbers, not measurements, so they enter
        exactly, as a declared ``chi3`` or a knot does
        (:meth:`solprop.modes.guide.ModalSampling.identity`).

        JSON-serialisable scalars and lists of scalars, stable across processes.
        """
        return {
            "gauge_id": self.gauge_id,
            "blocks": [[int(one) for one in block] for block in self.blocks],
            # The two parts by name, as lists of floats: a complex number has no
            # JSON form, and a pair of real matrices is what survives a round trip.
            "transform_real": [
                [float(one) for one in row] for row in np.real(self.transform)
            ],
            "transform_imag": [
                [float(one) for one in row] for row in np.imag(self.transform)
            ],
            "labels": [dict(label.as_mapping()) for label in self.labels],
        }


def provider_channels(provider: ModalProvider) -> tuple[ChannelLabel, ...]:
    """Return a provider's declared channels as labels, in its own order.

    A provider publishes its channels as mappings inside ``identity``, which is what
    survives JSON; this is the one place they are turned back into
    :class:`~solprop.modes.protocol.ChannelLabel` objects for a caller that has to
    name one -- to declare a change of basis over a pair, most of all.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        A provider that names its channels.

    Returns
    -------
    tuple
        One label per channel.

    Raises
    ------
    ModeError
        If the provider names no channels.
    """
    declared = provider.identity.get("channels")
    if not isinstance(declared, list) or not declared:
        raise ModeError(
            "this provider names no channels, so its labels cannot be read; "
            f"{type(provider).__name__} declares none"
        )
    labels = []
    for mapping in declared:
        if not isinstance(mapping, dict):
            raise ModeError("a declared channel must be a label mapping")
        labels.append(ChannelLabel.from_mapping(mapping))
    return tuple(labels)


def circular_pair(labels: Sequence[ChannelLabel], block: Sequence[int]) -> ChannelBasis:
    """Declare the circular partners of one exactly degenerate parity pair.

    The change of basis that turns the real ``cos``/``sin`` pair of an azimuthally
    degenerate family into its two circular partners, which is the basis an
    isotropic Kerr tensor is diagonal in: for a field that is circular at every
    node ``E . E`` vanishes, so ``(E . E) E*`` drops out of
    ``P = eps0 chi3 (2 |E|^2 E + (E . E) E*) / 4`` and the self coefficient falls to
    two thirds of the linear one.

    **The convention, derived here.** The even partner is the ``cos((n-1) theta)``
    pattern about **x**, which for ``n = 1`` is the x-polarised member, and the odd
    one is its quarter-turn partner about **y**
    (:class:`~solprop.modes.protocol.ChannelLabel`). A provider's
    fields are in the **internal** convention of ``docs/dev/conventions.md``
    section 2, whose plane wave is ``exp(i(omega t - beta z))``, so the physical
    transverse field of a channel with nodal rows ``(e_even, e_odd)`` and amplitude
    one is ``Re[(e_even + a e_odd) exp(i omega t)]``. With ``a = -i`` that is
    ``(cos(omega t), sin(omega t))`` in ``(x, y)``: the vector turns from x towards
    y, which an observer looking **towards the source** sees turn anticlockwise.
    That is positive helicity, which optics calls **left** circular (Jackson,
    *Classical Electrodynamics*, section 7.2), and its optics-convention Jones
    vector is the conjugate ``(1, i)/sqrt(2)``, which
    :mod:`solprop.polarisation` records as positive Stokes ``V``. So

    ``e_left  = (e_even - i e_odd)/sqrt(2)``, ``e_right = (e_even + i e_odd)/sqrt(2)``

    and the two columns of the returned matrix are exactly those, in that order:
    ``block[0]`` becomes the **left** channel and ``block[1]`` the **right** one.
    The order is not free -- ``block[0]`` must be the **even** member and
    ``block[1]`` the odd one, and a block given the other way round is refused by
    name. The names *left* and *right* are a convention, fixed by the derivation
    above, not a derived fact.

    Parameters
    ----------
    labels : collections.abc.Sequence
        Every channel's label, in the guide's own row order.
    block : collections.abc.Sequence
        Exactly two channel indices, one labelled ``"even"`` and one ``"odd"``,
        of the same family and quantum numbers.

    Returns
    -------
    ChannelBasis
        The declaration, with ``gauge_id`` ``"circular(i,j)"`` for the block's own
        two indices in ascending order and the two labels relabelled by handedness.

    Raises
    ------
    solprop.exceptions.ModeError
        If the block is not two distinct valid indices, or the two labels are not a
        parity pair of one family.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.guide import circular_pair
    >>> from solprop.modes.protocol import ChannelLabel
    from solprop.modes.quadrature import quadrature_rule
    >>> pair = (
    ...     ChannelLabel("HE", 1, 1, "even", None, "HE1,1:even"),
    ...     ChannelLabel("HE", 1, 1, "odd", None, "HE1,1:odd"),
    ... )
    >>> basis = circular_pair(pair, (0, 1))
    >>> bool(np.allclose(basis.transform[:, 0], [1, -1j] / np.sqrt(2)))
    True
    >>> basis.labels[1].handedness
    'right'
    """
    indices = [int(one) for one in block]
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ModeError(
            f"a circular pair is exactly two distinct channels, got {list(block)}"
        )
    if any(not 0 <= one < len(labels) for one in indices):
        raise ModeError(
            f"circular pair {indices} is outside the {len(labels)} channel(s) given"
        )
    even, odd = (labels[one] for one in indices)
    if (even.parity, odd.parity) != ("even", "odd") or (
        even.family,
        even.azimuthal_order,
        even.radial_order,
    ) != (odd.family, odd.azimuthal_order, odd.radial_order):
        raise ModeError(
            "a circular pair is the even and the odd parity of one family, in that "
            f"order: got {even.display_label!r} ({even.parity}) and "
            f"{odd.display_label!r} ({odd.parity})"
        )
    root = 1.0 / np.sqrt(2.0)
    matrix = np.eye(len(labels), dtype=np.complex128)
    matrix[np.ix_(indices, indices)] = np.array(
        [[root, root], [-1j * root, 1j * root]], dtype=np.complex128
    )
    base = even.display_label.removesuffix(":even")
    turned = [
        ChannelLabel(
            even.family,
            even.azimuthal_order,
            even.radial_order,
            None,
            hand,
            f"{base}:{hand}",
        )
        for hand in ("left", "right")
    ]
    relabelled = list(labels)
    for position, label in zip(indices, turned, strict=True):
        relabelled[position] = label
    low, high = sorted(indices)
    return ChannelBasis(
        matrix,
        ((indices[0], indices[1]),),
        tuple(relabelled),
        f"circular({low},{high})",
    )
