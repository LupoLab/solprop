# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Represent internal-convention Maxwell samples on physical area quadrature.

These physics records do not extend the propagation kernel contract. Bound,
lossless modes use the Hermitian Poynting product and true-power amplitudes.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from jaxtyping import Bool, Complex, Float, Int32

from solprop._display import Displayed, engineering, line
from solprop.exceptions import ModeError
from solprop.modes.material import JsonValue, MaterialRegion

__all__ = ["ModalNodeRule", "ModalProvider", "ModalSample", "flux_product"]


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

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
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

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
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

        Parameters
        ----------
        modes : int
            Validated positive channel count, dimensionless.

        Raises
        ------
        ModeError
            If a physical metric is indefinite or a reciprocal pairing is singular.
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
    def regions(self) -> tuple[MaterialRegion, ...]:
        """Return ordered material-region records."""
        ...

    @property
    def z_breakpoints(self) -> tuple[float, ...]:
        """Return physical path breakpoints in m."""
        ...

    @property
    def omega_domain(self) -> tuple[float, float]:
        """Return the declared angular-frequency interval in rad/s."""
        ...

    @property
    def z_domain(self) -> tuple[float, float]:
        """Return the declared longitudinal interval in m."""
        ...

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Return physical identity including spectral_convention='internal'."""
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
