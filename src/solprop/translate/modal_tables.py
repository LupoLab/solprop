# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Flatten a general guide's sampled tables into the record a run's seam carries.

The provider sampling -- tracking, interpolation and validation of the dense tables
-- lives in :mod:`solprop.modes.tables`. This module takes one of those tables and
builds :class:`ModalTransformTables`, the flat ``transform.modal_tables`` leaf of a
``PropagationProblem``: the dense E/H reference, the synthesis ``S`` and projection
``Q`` maps tabulated on the run's coarse grid (:func:`modal_transform_tables`), with
the per-sample builders :func:`projection_map` and :func:`coordinate_dual` that the
moving-guide record of :mod:`solprop.translate.modal_path` reuses. It also projects a
declared transverse field onto a table's channels (:func:`project_transverse_field`,
returning a :class:`ProjectedLaunch`).
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
from jaxtyping import Bool, Complex, Float, Inexact, Int32
from numpy.typing import ArrayLike

from solprop._display import Displayed, engineering, line
from solprop._ownership import frozen_copy
from solprop.exceptions import ModeError
from solprop.modes.material import JsonValue, quantised_measurement
from solprop.modes.tables import DenseModalTable
from solprop.modes.vector import ModalSample
from solprop.translate.coupled import interval_count

__all__ = [
    "ModalTransformTables",
    "ProjectedLaunch",
    "coordinate_dual",
    "modal_transform_tables",
    "project_transverse_field",
    "projection_map",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class ModalTransformTables(Displayed):
    """Own the full E/H reference of a general guide, flat, as the seam will take it.

    Built from a :class:`~solprop.modes.tables.DenseModalTable` by
    :func:`modal_transform_tables`, or along a moving guide by
    :mod:`solprop.translate.modal_path`. It is flat -- arrays, not a
    nest of samples -- because that is what a backend can be handed, and it is the
    ``transform.modal_tables`` leaf of a ``PropagationProblem``.

    ``factors`` is always ``None``; the dense ``electric`` and ``magnetic``
    references, the ``coordinate_dual`` and the execution maps ``synthesis`` and
    ``projection`` are filled. Frequency is the last axis and is contiguous. **Both
    maps are tabulated on the coarse ``omega``**: the state is a coarse spectrum and
    the fine grid is zero everywhere but ``grids.crop_index``, so a synthesis tabulated
    on ``omega_fine`` would carry columns no run can read. The oversampled grid is
    where the *field* lives, not where the
    map does: the pass synthesises on the coarse grid and places the result in the fine
    spectrum at ``crop_index``.

    Attributes
    ----------
    omega : numpy.ndarray
        Coarse angular frequencies in rad/s, shape (n_omega,), increasing and
        non-negative. On a ``PropagationProblem`` it is ``grids.omega`` itself, whose
        first bin is zero on a field grid, with the inactive bins explicitly zero
        rather than evaluated.
    omega_fine : numpy.ndarray
        Fine angular frequencies in rad/s, shape (n_omega_fine,), increasing and
        non-negative; the problem's fine spectrum, by the same rule.
    z_edges : numpy.ndarray
        Stored positions in m, shape (n_z,), increasing; they are the edges the
        interval-indexed arrays are counted over, one interval per pair of
        positions and one for a single snapshot.
    synthesis, projection : numpy.ndarray or None
        The execution maps, supplied together or not at all, both on ``omega``.
        ``synthesis`` is S in V/(m sqrt(W)), shape ``(n_intervals, n_deg, n_nodes,
        3, n_modes, n_omega)``; ``projection`` is Q in V m/sqrt(W), shape
        ``(n_intervals, n_deg, n_modes, n_nodes, 3, n_omega)``, carrying the
        area weights and the reciprocal normalisation and **excluding** the
        separate ``source_frequency_factor``. The leading pair of axes is a
        Chebyshev expansion along z inside each interval, so one interval carrying
        a single coefficient (degree zero) is a constant map.
    electric, magnetic : numpy.ndarray
        Reference E and H in V/m and A/m per sqrt(W), shape
        (n_z, n_nodes, 3, n_modes, n_omega).
    backward_electric, backward_magnetic, reciprocal_gram : numpy.ndarray or None
        Reciprocal partners and their bilinear pairing, supplied together or not at
        all.
    coordinate_dual : numpy.ndarray or None
        Amplitude recovery from a physical field, shape
        (n_z, n_modes, n_nodes, 6, n_omega): the first three components pair with E
        and the last three with H, in sqrt(W) m/V and sqrt(W) m/A.
    node_xy : numpy.ndarray
        Node positions in m, shape (n_z, n_nodes, 2).
    weights : numpy.ndarray
        Positive area weights in m², shape (n_z, n_nodes).
    region_index : numpy.ndarray
        Non-negative int32 material index per node, shape (n_nodes,).
    source_frequency_factor : numpy.ndarray
        The ``-i omega`` the projected polarisation is multiplied by, shape
        (n_omega,), in rad/s.
    valid : numpy.ndarray
        Fine-grid support mask, shape (n_z, n_modes, n_omega_fine).
    profile_error : numpy.ndarray
        Declared non-negative profile error per interval and coarse frequency,
        shape (n_intervals, n_omega), dimensionless.
    factors : object or None
        A factored representation; not implemented, so refused unless ``None``.
    gauge_id, model_id, policy_id : str
        Non-empty phase/coordinate, physical-model and numeric-policy identities.
    """

    omega: Float[np.ndarray, " n_omega_modal"]
    omega_fine: Float[np.ndarray, " n_omega_fine_modal"]
    z_edges: Float[np.ndarray, " n_edges_modal"]
    synthesis: (
        Complex[
            np.ndarray,
            "n_intervals_synthesis n_deg_synthesis n_nodes_synthesis"
            " n_components_synthesis n_modes_synthesis n_omega_synthesis",
        ]
        | None
    )
    projection: (
        Complex[
            np.ndarray,
            "n_intervals_projection n_deg_projection n_modes_projection"
            " n_nodes_projection n_components_projection n_omega_projection",
        ]
        | None
    )
    electric: Complex[
        np.ndarray, "n_z_electric n_nodes_electric 3 n_modes_electric n_omega_electric"
    ]
    magnetic: Complex[
        np.ndarray, "n_z_magnetic n_nodes_magnetic 3 n_modes_magnetic n_omega_magnetic"
    ]
    backward_electric: (
        Complex[
            np.ndarray,
            "n_z_backward_electric n_nodes_backward_electric 3"
            " n_modes_backward_electric n_omega_backward_electric",
        ]
        | None
    )
    backward_magnetic: (
        Complex[
            np.ndarray,
            "n_z_backward_magnetic n_nodes_backward_magnetic 3"
            " n_modes_backward_magnetic n_omega_backward_magnetic",
        ]
        | None
    )
    reciprocal_gram: (
        Complex[
            np.ndarray,
            "n_z_reciprocal n_modes_reciprocal_out n_modes_reciprocal_in"
            " n_omega_reciprocal",
        ]
        | None
    )
    coordinate_dual: (
        Complex[np.ndarray, "n_z_dual n_modes_dual n_nodes_dual 6 n_omega_dual"] | None
    )
    node_xy: Float[np.ndarray, "n_z_xy n_nodes_xy 2"]
    weights: Float[np.ndarray, "n_z_weights n_nodes_weights"]
    region_index: Int32[np.ndarray, " n_nodes_region_index"]
    source_frequency_factor: Complex[np.ndarray, " n_omega_source_factor"]
    valid: Bool[np.ndarray, "n_z_valid n_modes_valid n_omega_fine_valid"]
    profile_error: Float[np.ndarray, "n_intervals_profile_error n_omega_profile_error"]
    factors: object | None
    gauge_id: str
    model_id: str
    policy_id: str

    def __repr__(self) -> str:
        """Return a one-line description: the extents of the stored reference.

        Returns
        -------
        str
            For example ``ModalTransformTables(2 channels, 60 nodes, 8 frequencies,
            1 z)``.
        """
        _, nodes, _, modes, frequencies = np.shape(self.electric)
        return line(
            "ModalTransformTables",
            [
                ("", f"{modes} channel{'' if modes == 1 else 's'}"),
                ("", f"{nodes} nodes"),
                ("", f"{frequencies} frequencies"),
                ("", f"{len(self.z_edges)} z"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        _, nodes, _, modes, _ = np.shape(self.electric)
        return "ModalTransformTables: the flat E/H reference of a general guide", [
            (
                "frequencies",
                f"{len(self.omega)} coarse, {len(self.omega_fine)} fine, "
                f"{self.omega[0]:.4e} to {self.omega[-1]:.4e} rad/s",
            ),
            (
                "z",
                f"snapshot at {engineering(float(self.z_edges[0]), 'm')}"
                if len(self.z_edges) == 1
                else f"{len(self.z_edges)}, "
                f"{engineering(float(self.z_edges[0]), 'm')} to "
                f"{engineering(float(self.z_edges[-1]), 'm')}",
            ),
            ("channels", modes),
            ("nodes", nodes),
            (
                "maps",
                "dense reference, synthesis and projection"
                if self.synthesis is not None
                else "dense reference only (no execution map)",
            ),
            ("reciprocal", "yes" if self.reciprocal_gram is not None else "no"),
            ("profile error", f"{float(np.max(self.profile_error)):.3e} largest"),
            ("gauge", self.gauge_id),
            ("policy", self.policy_id),
        ]

    @property
    def frozen_rule(self) -> bool:  # numpydoc ignore=RT01
        """Whether one node rule was broadcast across every position.

        A record built over several positions from **one** rule -- the same
        Cartesian nodes and the same area weights at every z -- is the frozen-rule
        case. It is legitimate and it is not free: along a path the panel edge leaves
        the material interface, and the interpolated basis's flux defect grows
        linearly with that displacement (4.9e-7, 9.8e-7 and 1.95e-6 at a quarter, a
        half and the whole of a 1e-5 relative radius change). A taper needs sections
        with their
        own rules (:mod:`solprop.translate.modal_path`); a snapshot is trivially
        frozen.

        True when ``node_xy`` and ``weights`` do not vary along z.
        """
        return bool(
            np.all(self.node_xy == self.node_xy[0])
            and np.all(self.weights == self.weights[0])
        )

    @property
    def identity(self) -> Mapping[str, JsonValue]:  # numpydoc ignore=RT01
        """Scalar identity a run records and a resume compares.

        The record is ``eq=False``: two of them are never compared by value, and a
        resume must not compare megabytes of E and H field by field either. What
        enters identity is this mapping -- the three declared IDs, the six extents,
        the region count, the band and position endpoints, which optional
        representations are present, the frozen-rule flag and the profile-error
        budget. Every entry is a scalar, a string, a boolean or a list
        of scalars; no array is included by value.

        A factored map enters by its declared ``kind`` tag, never by its ``repr``:
        a default ``repr`` carries a memory address, which is not stable across the
        two processes a resume compares.

        The profile-error budget is the caller's declared one, which on the general
        modal route is the guide's **measured** error; it therefore enters at
        :data:`~solprop.modes.material.MEASURED_DIGITS` significant digits, exactly
        as :attr:`solprop.modes.guide.ModalSampling.identity` records the same
        number, so that the two agree and a re-solve on another BLAS does not move
        either of them.

        JSON-serialisable scalars, stable across processes.
        """
        positions, nodes, components, modes, frequencies = np.shape(self.electric)
        factors = self.factors
        # A moving guide's maps add their sections and degree, written only where the
        # maps move, so that a snapshot record's identity carries no section entry.
        synthesis = self.synthesis
        moving: dict[str, JsonValue] = (
            {
                "map_z_edges": [float(one) for one in self.z_edges],
                "map_degree": int(synthesis.shape[1]) - 1,
            }
            if synthesis is not None and synthesis.shape[0] * synthesis.shape[1] > 1
            else {}
        )
        return {
            **moving,
            "gauge_id": self.gauge_id,
            "model_id": self.model_id,
            "policy_id": self.policy_id,
            "n_modes": int(modes),
            "n_nodes": int(nodes),
            "n_components": int(components),
            "n_omega": int(frequencies),
            "n_omega_fine": len(self.omega_fine),
            "n_z": int(positions),
            "n_regions": int(np.max(self.region_index)) + 1,
            "omega_span": [float(self.omega[0]), float(self.omega[-1])],
            "omega_fine_span": [
                float(self.omega_fine[0]),
                float(self.omega_fine[-1]),
            ],
            "z_span": [float(self.z_edges[0]), float(self.z_edges[-1])],
            "has_synthesis": self.synthesis is not None,
            "has_projection": self.projection is not None,
            "has_reciprocal": self.reciprocal_gram is not None,
            "has_coordinate_dual": self.coordinate_dual is not None,
            "factors_kind": getattr(factors, "kind", None),
            "frozen_rule": self.frozen_rule,
            "profile_error_max": quantised_measurement(
                float(np.max(self.profile_error))
            ),
        }

    def __post_init__(self) -> None:
        """Copy, validate and freeze every leaf array.

        Every array is copied into one this record owns, so that a caller keeps its
        own arrays writeable and cannot change the record through a view it already
        held.

        Raises
        ------
        ModeError
            If a grid, an extent, a value or an identity is invalid, if the
            reciprocal arrays are not supplied together, or if a representation
            that is not implemented is not ``None``.
        """
        if self.factors is not None:
            raise ModeError(
                "a factored modal map is not implemented; factors must be None"
            )
        if not (self.gauge_id and self.model_id and self.policy_id):
            raise ModeError("gauge_id, model_id and policy_id must be non-empty")
        # Own every leaf. np.ascontiguousarray returns the caller's own object when
        # it is already contiguous, so freezing it would make the caller's array
        # read-only, and a view the caller still holds could then rewrite the
        # record behind its back. Copy, as every other record here does.
        for name in ("omega", "omega_fine", "z_edges"):
            values = frozen_copy(getattr(self, name), np.float64)
            if (
                values.ndim != 1
                or not values.size
                or not np.all(np.isfinite(values))
                or np.any(np.diff(values) <= 0)
            ):
                raise ModeError(
                    f"{name} must be a finite, non-empty, increasing vector"
                )
            object.__setattr__(self, name, values)
        if self.omega[0] < 0 or self.omega_fine[0] < 0:
            raise ModeError("frequency grids must be non-negative")
        self._validate_arrays()

    def _map_degree(self, intervals: int) -> int:
        """Return the Chebyshev degree count the two execution maps share.

        - Returns: the degrees along z, or zero when the record carries no map at all.
        """
        if (self.synthesis is None) != (self.projection is None):
            raise ModeError(
                "synthesis and projection are the two halves of one execution map "
                "and must be supplied together"
            )
        if self.synthesis is None:
            return 0
        shape = np.shape(self.synthesis)
        if len(shape) != 6 or shape[0] != intervals or not shape[1]:
            raise ModeError(
                f"synthesis must carry {intervals} interval(s) and at least one "
                f"Chebyshev degree along z, got {shape}"
            )
        return int(shape[1])

    def _validate_arrays(self) -> None:
        """Copy, check and freeze every leaf against the extents the record declares."""
        # The component count is the contract's ceiling, which lives beside the
        # state's polarisation ceiling in translate.problem; that module imports this
        # one, so the import is deferred rather than the number copied.
        from solprop.translate.problem import (  # noqa: PLC0415 - import cycle
            MAX_N_COMPONENTS,
        )

        reference = frozen_copy(self.electric, np.complex128)
        if reference.ndim != 5 or reference.shape[2] != MAX_N_COMPONENTS:
            raise ModeError(
                f"electric must have shape (n_z, n_nodes, {MAX_N_COMPONENTS}, "
                f"n_modes, n_omega), got {reference.shape}"
            )
        positions, nodes, _, modes, frequencies = reference.shape
        if (
            positions != len(self.z_edges)
            or frequencies != len(self.omega)
            or not nodes
            or not modes
        ):
            raise ModeError(
                "the stored reference must have one position per z edge and one "
                f"frequency per coarse grid point, got {reference.shape}"
            )
        intervals = interval_count(positions)
        fine = len(self.omega_fine)
        group = (self.backward_electric, self.backward_magnetic, self.reciprocal_gram)
        if any(value is None for value in group) and not all(
            value is None for value in group
        ):
            raise ModeError(
                "backward fields and the reciprocal Gram must be supplied together"
            )
        field = (positions, nodes, MAX_N_COMPONENTS, modes, frequencies)
        shapes: dict[str, tuple[tuple[int, ...], type]] = {
            "electric": (field, np.complex128),
            "magnetic": (field, np.complex128),
            "backward_electric": (field, np.complex128),
            "backward_magnetic": (field, np.complex128),
            "reciprocal_gram": ((positions, modes, modes, frequencies), np.complex128),
            "coordinate_dual": (
                (positions, modes, nodes, 6, frequencies),
                np.complex128,
            ),
            "node_xy": ((positions, nodes, 2), np.float64),
            "weights": ((positions, nodes), np.float64),
            "region_index": ((nodes,), np.int32),
            "source_frequency_factor": ((frequencies,), np.complex128),
            "valid": ((positions, modes, fine), np.bool_),
            "profile_error": ((intervals, frequencies), np.float64),
        }
        degree = self._map_degree(intervals)
        if degree:
            shapes["synthesis"] = (
                (intervals, degree, nodes, MAX_N_COMPONENTS, modes, frequencies),
                np.complex128,
            )
            shapes["projection"] = (
                (intervals, degree, modes, nodes, MAX_N_COMPONENTS, frequencies),
                np.complex128,
            )
        for name, (shape, dtype) in shapes.items():
            value = reference if name == "electric" else getattr(self, name)
            if value is None:
                continue
            array = frozen_copy(value, dtype)
            if array.shape != shape:
                raise ModeError(f"{name} must have shape {shape}, got {array.shape}")
            if dtype is not np.bool_ and not np.all(np.isfinite(array)):
                raise ModeError(f"{name} must be finite")
            object.__setattr__(self, name, array)
        if np.any(self.weights <= 0):
            raise ModeError("area weights must be positive")
        if np.any(self.region_index < 0):
            raise ModeError("region_index must be non-negative")
        if np.any(self.profile_error < 0):
            raise ModeError("profile_error must be non-negative")


def coordinate_dual(
    sample: ModalSample, weights: Float[np.ndarray, " n_nodes"]
) -> Complex[np.ndarray, "n_modes n_nodes 6"]:
    """Build the functional that recovers amplitudes from a physical field.

    The amplitude of channel ``m`` in a field ``(E, H)`` sampled at the nodes is
    ``a = W^-1 B`` with ``B_m = int(E x h_m* + e_m* x H)_z dA / 4``, the same
    Hermitian Poynting pairing the flux metric is built from, so that synthesising
    a declared amplitude vector and contracting this against it returns that vector.
    Only the transverse components enter the longitudinal flux, so the two
    longitudinal columns are zero. It is for amplitude recovery, not for nonlinear
    projection, which carries the separate ``-i omega`` factor.

    Parameters
    ----------
    sample : solprop.modes.vector.ModalSample
        Internal-convention sample at one frequency and position.
    weights : numpy.ndarray
        The rule's positive area weights in m², shape (n_nodes,).

    Returns
    -------
    numpy.ndarray
        Coefficients, shape (n_modes, n_nodes, 6): the first three pair with E in
        sqrt(W) m/V and the last three with H in sqrt(W) m/A.
    """
    inverse = np.linalg.inv(sample.flux_metric)
    electric, magnetic = sample.electric.conj(), sample.magnetic.conj()
    pairing = np.zeros((sample.electric.shape[2], len(weights), 6), dtype=complex)
    pairing[:, :, 0] = np.einsum("mn,qn->mq", inverse, magnetic[:, 1])
    pairing[:, :, 1] = -np.einsum("mn,qn->mq", inverse, magnetic[:, 0])
    pairing[:, :, 3] = -np.einsum("mn,qn->mq", inverse, electric[:, 1])
    pairing[:, :, 4] = np.einsum("mn,qn->mq", inverse, electric[:, 0])
    return pairing * weights[None, :, None] / 4


def projection_map(
    sample: ModalSample, weights: Float[np.ndarray, " n_nodes"]
) -> Complex[np.ndarray, "n_modes n_nodes 3"]:
    """Build the source projection Q of one frequency and position.

    The modal source functional of a bound reciprocal basis is
    ``da_m/dz = -i omega [W^-1]_mn int(e_n* . P) dA / 4``, so the
    map that takes a nodal polarisation to an amplitude rate, **without** the
    separate ``-i omega`` factor, is

    ``Q[m, q, c] = [W^-1]_mn conj(e_n[q, c]) w[q] / 4``,

    the same Hermitian pairing and the same reciprocal normalisation the coefficient
    dual is built from. Source projection and amplitude recovery are different
    operations: this one contracts a polarisation in C/m^2 and has units
    V m/sqrt(W), so ``Q P omega`` is in sqrt(W)/m.

    Parameters
    ----------
    sample : solprop.modes.vector.ModalSample
        Internal-convention sample at one frequency and position.
    weights : numpy.ndarray
        The rule's positive area weights in m², shape (n_nodes,).

    Returns
    -------
    numpy.ndarray
        ``Q`` in V m/sqrt(W), shape (n_modes, n_nodes, 3).
    """
    inverse = np.linalg.inv(sample.flux_metric)
    projected = np.einsum("mn,qcn->mqc", inverse, sample.electric.conj())
    return projected * weights[None, :, None] / 4


def modal_transform_tables(
    table: DenseModalTable,
    omega: Float[np.ndarray, " n_omega"],
    *,
    profile_error: ArrayLike | float,
    omega_fine: ArrayLike | None = None,
    z: ArrayLike | None = None,
    policy_id: str = "dense-knots-cubic-frequency",
) -> ModalTransformTables:
    """Build the flat seam record from a dense reference table.

    The one constructor of :class:`ModalTransformTables`. It evaluates the table on
    the grid a run will use -- which need not be the table's own knots, and in
    general is not -- and stores the fields, the reciprocal partners, the
    coefficient dual, the two execution maps and the node geometry as arrays.

    **Outside the table's band the record is explicitly zero**, and the fine grid's
    ``valid`` mask is false there. A canonical field grid starts
    at zero and runs past any guide's support, so a leaf tabulated on it
    necessarily leaves the band; what must not happen is an *evaluated* profile
    there, which extrapolates a mode past the frequencies it was solved at. A zero
    reference is a zero source and a zero synthesised field: a run whose active
    band leaves the guide's support is refused where the problem is assembled, not
    silently propagated against zeros.

    **The execution maps are built for a snapshot table only.** S and Q are stored
    as one Chebyshev coefficient per interval, which is exact when the map does not
    vary along z and an average when it does, so a table sampled over a path of
    positions gets ``synthesis`` and ``projection`` of ``None`` and the general
    route refuses it: a moving basis needs sections with their own node rules,
    which :mod:`solprop.translate.modal_path` builds. A snapshot is broadcast across
    every requested position, which is the frozen-rule case
    :attr:`ModalTransformTables.frozen_rule` names.

    Parameters
    ----------
    table : DenseModalTable
        Tracked reference knots. Requested frequencies outside its band are stored
        as zeros; requested positions outside a *path* table's own are refused, and
        a snapshot is broadcast to all of them.
    omega : numpy.ndarray
        Coarse angular frequencies in rad/s, shape (n_omega,), increasing.
    profile_error : array_like or float
        Declared non-negative profile error of the representation, per interval and
        coarse frequency or one value for all of them. It is a budget the caller
        has measured, for example with
        :func:`~solprop.modes.tables.validate_modal_table`; nothing here infers it.
    omega_fine : array_like or None, optional
        Fine angular frequencies in rad/s; ``None`` uses the coarse grid, which is
        what a run with no oversampling declares.
    z : array_like or None, optional
        Positions in m to store; ``None`` uses the table's own.
    policy_id : str, optional
        Identity of the numeric policy this record was built under.

    Returns
    -------
    ModalTransformTables
        The validated flat record, with ``factors`` left ``None`` and the
        ``coordinate_dual`` filled.

    Raises
    ------
    ModeError
        If a position leaves a path table's domain, or the record does not validate.
    """
    coarse = np.ascontiguousarray(omega, dtype=np.float64)
    fine = coarse if omega_fine is None else np.ascontiguousarray(omega_fine, float)
    positions = (
        np.asarray(table.z, dtype=np.float64)
        if z is None
        else np.ascontiguousarray(z, dtype=np.float64)
    )
    # A snapshot has one position and no z interpolant at all, so every requested
    # position is that one solve: the geometry does not move (from_provider refuses
    # a provider whose does) and the record says so through `frozen_rule`.
    snapshot = len(table.z) == 1
    weights = table.nodes.weights
    reference = table.samples[0][0]
    modes = int(reference.electric.shape[2])
    nodes = len(weights)
    inside = (table.omega[0] <= coarse) & (coarse <= table.omega[-1])
    inside_fine = (table.omega[0] <= fine) & (fine <= table.omega[-1])
    reciprocal = reference.reciprocal_gram is not None
    shape = (len(positions), nodes, 3, modes, len(coarse))
    stack = np.zeros(shape, dtype=np.complex128)
    magnetic = np.zeros(shape, dtype=np.complex128)
    backward = [
        np.zeros(shape, dtype=np.complex128) if reciprocal else None for _ in range(2)
    ]
    gram = (
        np.zeros((len(positions), modes, modes, len(coarse)), dtype=np.complex128)
        if reciprocal
        else None
    )
    dual = np.zeros((len(positions), modes, nodes, 6, len(coarse)), np.complex128)
    intervals = interval_count(len(positions))
    projection = (
        np.zeros((intervals, 1, modes, nodes, 3, len(coarse)), np.complex128)
        if snapshot
        else None
    )
    synthesis = (
        np.zeros((intervals, 1, nodes, 3, modes, len(coarse)), np.complex128)
        if snapshot
        else None
    )
    # The arrays with a row per position: every row of a snapshot is the one solve at
    # the table's own position, so its first row's numbers are copied to the others
    # rather than interpolated again.
    per_position = cast(
        "list[np.ndarray]",
        [stack, magnetic, dual, *((*backward, gram) if reciprocal else ())],
    )
    for row, position in enumerate(positions):
        if snapshot and row:
            for array in per_position:
                array[row] = array[0]
            continue
        at = float(table.z[0]) if snapshot else float(position)
        for column in np.flatnonzero(inside):
            sample = table.evaluate(float(coarse[column]), at)
            stack[row, :, :, :, column] = sample.electric
            magnetic[row, :, :, :, column] = sample.magnetic
            dual[row, :, :, :, column] = coordinate_dual(sample, weights)
            if backward[0] is not None and backward[1] is not None:
                backward[0][row, :, :, :, column] = sample.backward_electric
                backward[1][row, :, :, :, column] = sample.backward_magnetic
            if gram is not None:
                gram[row, :, :, column] = sample.reciprocal_gram
            if projection is not None and not row:
                projection[:, 0, :, :, :, column] = projection_map(sample, weights)
            # S is the stored electric profile itself, on the same coarse grid the
            # state lives on: the run synthesises there and places the result in its
            # fine spectrum, so a column of S the coarse grid has no bin for could
            # never be read.
            if synthesis is not None and not row:
                synthesis[:, 0, :, :, :, column] = sample.electric
    logger.debug(
        "modal transform tables: %d channels on %d nodes, %d of %d coarse and %d of "
        "%d fine frequencies inside the band, %d positions, maps %s",
        modes,
        nodes,
        int(np.count_nonzero(inside)),
        len(coarse),
        int(np.count_nonzero(inside_fine)),
        len(fine),
        len(positions),
        "built" if snapshot else "absent (a path)",
    )
    return ModalTransformTables(
        coarse,
        fine,
        positions,
        synthesis,
        projection,
        stack,
        magnetic,
        backward[0],
        backward[1],
        gram,
        dual,
        np.broadcast_to(table.nodes.xy, (len(positions), *table.nodes.xy.shape)),
        np.broadcast_to(weights, (len(positions), nodes)),
        table.nodes.region_index,
        -1j * coarse,
        np.broadcast_to(inside_fine, (len(positions), modes, len(fine))),
        np.broadcast_to(
            np.asarray(profile_error, dtype=np.float64),
            (intervals, len(coarse)),
        ),
        None,
        reference.gauge_id,
        reference.physical_id,
        policy_id,
    )


@dataclass(frozen=True, eq=False)
class ProjectedLaunch(Displayed):
    """What a declared transverse field becomes in a guide's channels.

    The result of :func:`project_transverse_field`: the channel amplitudes, the
    power those amplitudes carry, and what the guide's bound channels do **not**
    carry of the declared field. The last is named rather than discarded, because
    a projection that captured a tenth of what was declared and said nothing would
    be a silent change of the initial-value problem.

    Attributes
    ----------
    amplitudes : numpy.ndarray
        The channel amplitudes, complex128, shape ``(n_modes, n_omega)``, in the
        square root of watts -- the true-power amplitude every route uses (see
        ``docs/dev/decisions.md``), because the basis is flux-normalised.
    captured : numpy.ndarray
        The longitudinal Poynting flux the projected field carries, float64, shape
        ``(n_omega,)``, in W. It is the physical
        ``int Re(E x conj(H))/2 . z dA`` of the **projected** field, formed from
        the guide's own stored E and H, which equals ``a.H W a`` and is not
        ``sum_m |a_m|^2`` unless the basis is exactly orthonormal.
    residual : numpy.ndarray
        Share of the declared field the bound channels do not reproduce, float64,
        shape ``(n_omega,)``, dimensionless and non-negative -- and, like
        ``residual_share`` below, **not clipped at one**:
        ``sum_q w_q |E - sum_m a_m e_m|^2 / sum_q w_q |E|^2`` at each bin, in the
        guide's own quadrature norm. It is ``nan`` where the declared field is
        identically zero, because there is no share of nothing.
    residual_share : float
        The same quantity over the whole declared field, weighting each bin by its
        own declared norm rather than equally: one number for the projection. It
        is **not clipped at one**: on a basis that is not orthonormal the flux
        projection can overshoot, leaving a residual larger than the declared
        field itself, and a share above one says exactly that. Clipping it would
        report a projection that missed everything as one that missed all of it,
        which are different statements.

    Raises
    ------
    ModeError
        If the arrays disagree in shape, or the residual share is not finite and
        non-negative.
    """

    amplitudes: Complex[np.ndarray, "n_modes_launch n_omega_launch"]
    captured: Float[np.ndarray, " n_omega_captured"]
    residual: Float[np.ndarray, " n_omega_residual"]
    residual_share: float

    def __post_init__(self) -> None:
        """Check that the three arrays describe one projection.

        Raises
        ------
        ModeError
            Naming what disagrees.
        """
        bins = self.amplitudes.shape[1]
        for name, value in (("captured", self.captured), ("residual", self.residual)):
            if value.shape != (bins,):
                msg = (
                    f"a projected launch's {name} covers its own frequency axis: "
                    f"{bins} bin(s) of amplitudes and {value.shape}"
                )
                raise ModeError(msg)
        if not np.isfinite(self.residual_share) or self.residual_share < 0.0:
            msg = (
                "a projected launch's residual_share is the share of the declared "
                f"field its channels miss, finite and non-negative; got "
                f"{self.residual_share!r}"
            )
            raise ModeError(msg)

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``ProjectedLaunch(2 channels, 1.58 W captured, 3.1e-03
            missed)``.
        """
        modes = int(self.amplitudes.shape[0])
        return line(
            "ProjectedLaunch",
            [
                ("", f"{modes} channel{'' if modes == 1 else 's'}"),
                ("", f"{engineering(float(np.max(self.captured)), 'W')} peak"),
                ("", f"{self.residual_share:.1e} missed"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        return "ProjectedLaunch: a declared field in a guide's channels", [
            ("channels", int(self.amplitudes.shape[0])),
            ("frequencies", int(self.amplitudes.shape[1])),
            ("peak captured", engineering(float(np.max(self.captured)), "W")),
            ("missed share", f"{self.residual_share:.3e}"),
        ]


def project_transverse_field(
    tables: ModalTransformTables,
    electric: Inexact[np.ndarray, "n_nodes n_components n_omega"],
) -> ProjectedLaunch:
    r"""Project a declared transverse electric field onto a guide's channels.

    The launch a caller declares as a **field** rather than as a channel: the
    amplitude of each channel in a declared ``E(x, y, omega)`` sampled at the
    guide's own quadrature nodes, with the power captured and the remainder named.

    **The map is the coordinate dual, not the source projection.** The amplitude
    of a field in a flux-normalised basis is ``a = W^-1 B`` with
    ``B_m = int(E x h_m* + e_m* x H)_z dA / 4``, the same Hermitian Poynting
    pairing the flux metric is built from, and that is what
    ``tables.coordinate_dual`` carries. ``Q``, the record's other map, contracts a
    *polarisation* with the conjugated electric profile and carries the reciprocal
    normalisation for a **source** term; it is a different operation and gives a
    different number.

    **Only E is declared, so the H half is closed by the forward-field
    assumption.** For any superposition of these channels travelling forward, the
    two halves of ``B_m`` are equal -- the Poynting cross matrix of a lossless
    bound basis is Hermitian -- so ``B_m = 2 int(E x h_m*)_z dA / 4`` and the
    declared E alone determines the amplitudes. Measured on the step-index parity
    pair, a field synthesised from known amplitudes comes back through this map to
    **1.412e-15** of them, which is rounding and not the basis's Gram defect
    (1.3e-6), so the closure is exact for this package's providers rather than
    approximate. (Its last digits move with the BLAS thread count, as a 288-node
    reduction's rounding does; the quoted number is one thread everywhere, and
    the bound the test asserts is 1e-13.) **That exactness is a statement about a
    lossless basis**: the Hermitian symmetry of the Poynting cross matrix is what
    makes the two halves equal, and a complex
    :class:`~solprop.modes.material.PermittivityLaw` breaks it, so on such a basis
    the closure would be wrong by an amount only a growing residual share would
    show. No such basis can reach here: both providers this package ships refuse a
    complex permittivity when they solve, by name
    (``"provider material differs from its lossless step-index law"`` and
    ``"radial provider requires positive lossless permittivity"``), so the
    condition is enforced upstream and a future lossy provider must re-derive the
    closure rather than inherit it. A field that is not forward-propagating -- a
    declared backward or evanescent component -- is not what this map returns
    either, and nothing here can see the difference; declare the amplitudes
    directly if that is wanted.

    **What is not captured is named.** The bound channels span what they span: a
    declared field with power in radiation modes, in a channel the guide did not
    solve for, or simply outside the node rule's reach leaves a residual, and
    `ProjectedLaunch.residual` reports it as a share of the declared field in the
    guide's own quadrature norm. Reprojecting a field creates a new initial-value
    problem; it is not a resume path.

    Parameters
    ----------
    tables : ModalTransformTables
        The guide's reference, which must carry its coordinate dual.
    electric : numpy.ndarray
        The declared transverse field at the guide's nodes, real or complex (a
        real profile is the common case and is taken as it is), shape
        ``(n_nodes, n_components, n_omega)``, in V/m, on the record's own coarse
        frequency axis. A profile that does not depend on frequency is broadcast
        to it by the caller.

    Returns
    -------
    ProjectedLaunch
        The amplitudes, the captured flux and the residual.

    Raises
    ------
    ModeError
        If the record carries no coordinate dual, or the declared field does not
        lie on its nodes, components and frequencies.
    """
    dual = tables.coordinate_dual
    if dual is None:
        msg = (
            "these tables carry no coordinate_dual, so a declared field cannot be "
            "projected onto their channels: build the reference with "
            "modal_transform_tables, which fills it"
        )
        raise ModeError(msg)
    reference = tables.electric[0]
    nodes, components, modes, frequencies = reference.shape
    field = np.asarray(electric, dtype=np.complex128)
    if field.shape != (nodes, components, frequencies):
        msg = (
            f"a declared transverse field lies on this guide's nodes, components "
            f"and coarse frequencies, shape ({nodes}, {components}, "
            f"{frequencies}); got {field.shape}"
        )
        raise ModeError(msg)
    # The doubled E half of `a = W^-1 B`: the forward-field closure above. The
    # dual's longitudinal columns are identically zero, so summing over all three
    # components is the transverse pairing whatever the caller put in E_z.
    amplitudes = 2.0 * np.einsum("mqcw,qcw->mw", dual[0, :, :, :components, :], field)
    weights = tables.weights[0]
    projected = np.einsum("qcmw,mw->qcw", reference, amplitudes)
    magnetic = np.einsum("qcmw,mw->qcw", tables.magnetic[0], amplitudes)
    # The physical flux of the projected field, formed from the guide's own stored
    # E and H rather than from a metric: `Re int(E x conj(H))/2 . z dA`, which is
    # `a.H W a` for the same W `flux_product` builds and is the number the route's
    # `analytic_flux` peaks.
    captured = 0.5 * np.real(
        np.einsum(
            "q,qw->w",
            weights,
            projected[:, 0] * np.conj(magnetic[:, 1])
            - projected[:, 1] * np.conj(magnetic[:, 0]),
        )
    )
    declared_norm = np.einsum("q,qcw->w", weights, np.abs(field) ** 2)
    residual_norm = np.einsum("q,qcw->w", weights, np.abs(field - projected) ** 2)
    empty = declared_norm <= 0.0
    residual = np.where(
        empty, np.nan, residual_norm / np.where(empty, 1.0, declared_norm)
    )
    total = float(np.sum(declared_norm))
    share = 0.0 if total <= 0.0 else float(np.sum(residual_norm) / total)
    logger.debug(
        "projected a declared field onto %d channel(s): %.4g W peak captured, "
        "%.3e of the declared norm missed",
        modes,
        float(np.max(captured)) if captured.size else 0.0,
        share,
    )
    return ProjectedLaunch(amplitudes, captured, residual, share)
