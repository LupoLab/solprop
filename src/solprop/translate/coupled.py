# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The metric and the coupled-flow representation of a general modal guide.

Two leaves of section 3 of ``docs/dev/kernel-contract.md``, added at kernel schema
12: :class:`ModalMetricTables`, the explicit physical flux metric the controller, the
linear flow and the output layer all read, and :class:`CoupledFlowTables`, the frozen
or ordered representation of a generator whose channels couple.

:class:`CoupledFlowTables` is the record, its validation and its printing, built by
:func:`constant_schur_flow`, :func:`frozen_residual_flow` and :func:`cf4_flow` for the
routes whose channels couple -- a capillary's coherent wall and taper coupling, and a
general modal guide whose declared basis or motion couples its channels. Everywhere
else it is ``None`` and a guide propagates through the diagonal cumulative flow.
:class:`ModalMetricTables` admits all three kinds: ``"identity"``, ``"diagonal"`` and
the ``"dense"`` kind a declared unitary change of basis produces
(:func:`dense_metric`). Which of them
a **backend** executes is its own admission, and ``numpy_ref`` refuses the diagonal
one, which nothing here builds.

Units are SI throughout (``docs/dev/conventions.md``): ``omega`` in rad/s, ``z_edges``
in m, the generator and every term of it in 1/m, metric entries dimensionless, errors
and bounds dimensionless. Every complex array is in the **internal FFT convention**,
like the rest of the contract; there is no fourth conversion site.
"""

import logging
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Complex, Float
from numpy.polynomial import chebyshev as chebyshev_basis
from scipy.linalg import expm, schur

from solprop._display import Displayed, engineering, line
from solprop.exceptions import SolpropWarning, TranslationError
from solprop.modes.capillary import WallImpedance, WallLimits
from solprop.modes.material import JsonValue, quantised_measurement
from solprop.modes.protocol import ChannelLabel
from solprop.translate.chebyshev import (
    _EDGE_SLACK_FRACTION,
    INITIAL_DEGREE,
    PiecewiseChebyshev,
    lobatto_coefficients,
    lobatto_points,
)

__all__ = [
    "CF4_GROWTH_LIMIT",
    "CF4_NODES",
    "CF4_WEIGHTS",
    "MAX_SECTION_DEGREE",
    "SECTION_TOL",
    "CoupledFlowTables",
    "MetricKind",
    "ModalMetricTables",
    "SectionMethod",
    "advisory_step",
    "cf4_defect",
    "cf4_flow",
    "cf4_growth_bound",
    "constant_schur_flow",
    "dense_metric",
    "eigenbasis_limit",
    "flow_edge_slack",
    "flow_section",
    "flow_weights",
    "frozen_generator",
    "frozen_residual_flow",
    "interval_count",
    "metric_at",
    "orthonormal_metric",
    "residual_truncation",
    "schur_blocks",
    "schur_eigenbasis",
    "section_expansion",
    "taper_identity",
    "wall_identity",
]

logger = logging.getLogger(__name__)

MetricKind = Literal["identity", "diagonal", "dense"]
"""Which representation of the physical flux metric a problem carries.

``"identity"`` is an orthonormal power basis, whose metric owns no coefficient array at
all; ``"diagonal"`` is a per-channel positive weight; ``"dense"`` is the full Hermitian
matrix a nonorthogonal basis needs, which a declared unitary change of basis
produces.
"""

SectionMethod = Literal["diagonal", "constant_schur", "frozen_residual", "cf4"]
"""How one interval of a coupled flow is integrated.

``"diagonal"`` is the existing cumulative commuting flow; ``"constant_schur"`` is the
Schur matrix-function action of a constant coherent block; ``"frozen_residual"`` freezes
the block inside the section and leaves its variation to the explicit source; ``"cf4"``
is the commutator-free fourth-order flow of the whole section generator, admitted by
:func:`cf4_flow` behind its internal-growth and source-feedback guards.
"""


def interval_count(edges: int) -> int:
    """Return how many z intervals a set of stored positions declares.

    Parameters
    ----------
    edges : int
        Number of stored positions, at least one.

    Returns
    -------
    int
        ``edges - 1`` for a path, and 1 for a single snapshot, whose one interval is
        the position it was taken at. A record's interval-indexed
        arrays -- the error rows, the coefficient representations -- use this count,
        while its dense arrays are indexed by the positions themselves.

    Examples
    --------
    >>> from solprop.translate.coupled import interval_count
    >>> interval_count(1), interval_count(4)
    (1, 3)
    """
    return max(edges - 1, 1)


def _vector(values: object, name: str) -> Float[np.ndarray, " n_entries"]:
    """Return a frozen, finite, strictly increasing float64 vector.

    Parameters
    ----------
    values : array_like
        The candidate axis.
    name : str
        Field name, used in the refusal.

    Returns
    -------
    numpy.ndarray
        A read-only C-contiguous copy.

    Raises
    ------
    TranslationError
        If the values are not a finite, non-empty, increasing vector.
    """
    array = np.array(values, dtype=np.float64, order="C", copy=True)
    if (
        array.ndim != 1
        or not array.size
        or not np.all(np.isfinite(array))
        or np.any(np.diff(array) <= 0)
    ):
        msg = f"{name} must be a finite, non-empty, increasing vector"
        raise TranslationError(msg)
    array.setflags(write=False)
    return array


def _owned(
    values: object, name: str, shape: tuple[int, ...], dtype: type
) -> np.ndarray:
    """Return a frozen copy of an array, checked against a declared shape.

    Parameters
    ----------
    values : array_like
        The candidate array.
    name : str
        Field name, used in the refusal.
    shape : tuple of int
        The shape the record's own extents require.
    dtype : type
        ``numpy.float64`` or ``numpy.complex128``.

    Returns
    -------
    numpy.ndarray
        A read-only C-contiguous copy.

    Raises
    ------
    TranslationError
        If the shape disagrees or a value is not finite.
    """
    array = np.array(values, dtype=dtype, order="C", copy=True)
    if array.shape != shape:
        msg = f"{name} must have shape {shape}, got {array.shape}"
        raise TranslationError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite"
        raise TranslationError(msg)
    array.setflags(write=False)
    return array


def _degree_axis(array: np.ndarray, name: str) -> None:
    """Refuse a coefficient block that declares no degree at all.

    Parameters
    ----------
    array : numpy.ndarray
        A coefficient array whose second axis is the Chebyshev degree along z.
    name : str
        Field name, used in the refusal.

    Raises
    ------
    TranslationError
        If the degree axis is empty. A block of ``n_deg = 0`` evaluates to nothing
        at every position, so it is refused here rather than read as a zero.
    """
    if not array.shape[1]:
        msg = (
            f"{name} has an empty degree axis; a coefficient block along z carries "
            "at least one degree, and a constant is degree zero with one coefficient"
        )
        raise TranslationError(msg)


def _square(values: object, name: str, rank: int) -> tuple[np.ndarray, int]:
    """Return a frozen complex array whose two channel axes agree, with its width.

    Parameters
    ----------
    values : array_like
        The candidate array, whose channel axes are the two before the frequency axis.
    name : str
        Field name, used in the refusal.
    rank : int
        Number of axes the array must have.

    Returns
    -------
    tuple
        The read-only array and its channel count.

    Raises
    ------
    TranslationError
        If the rank is wrong, the two channel axes disagree, or a value is not finite.
    """
    array = np.array(values, dtype=np.complex128, order="C", copy=True)
    if array.ndim != rank or array.shape[-3] != array.shape[-2] or not array.shape[-2]:
        msg = (
            f"{name} must have {rank} axes with equal and non-empty output and input "
            f"channel axes, got {array.shape}"
        )
        raise TranslationError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite"
        raise TranslationError(msg)
    array.setflags(write=False)
    return array, int(array.shape[-2])


@dataclass(frozen=True, eq=False)
class ModalMetricTables(Displayed):
    """The explicit physical flux metric of a modal basis.

    The documented leaf of contract section 3 that the step controller's norm, the
    linear flow and every physical-energy statistic read, so that one W is shared by
    identity and never rebuilt. A coordinate change ``c = T d``
    takes it to ``T.H @ W @ T``; interference terms cannot be assigned unique positive
    per-channel energies, so coordinate intensities are not additive in a
    nonorthogonal basis.

    Attributes
    ----------
    kind : {'identity', 'diagonal', 'dense'}
        Which representation is stored. ``"identity"`` owns no coefficient array;
        ``"diagonal"`` owns ``diagonal``; ``"dense"`` owns ``dense``, is Hermitian
        at every interval, degree and frequency, and is what a declared unitary
        change of basis produces (:func:`dense_metric`). Which of them a
        **backend** executes is its own admission: ``numpy_ref`` takes the
        identity and a dense metric that is a unitary transform of one, and
        refuses a diagonal metric, which nothing builds.
    omega : numpy.ndarray
        The canonical coarse angular frequencies in rad/s, shape ``(n_omega,)``,
        increasing and non-negative. A problem's leaf carries ``grids.omega``
        itself, whose first bin is zero on a field grid
        (:meth:`solprop.translate.problem.KernelProblem.validate`).
    z_edges : numpy.ndarray
        Stored positions in m, shape ``(n_edges,)``, increasing. The
        interval-indexed arrays are counted by :func:`interval_count`.
    diagonal : numpy.ndarray or None
        Positive dimensionless per-channel weights, shape
        ``(n_intervals, n_deg, n_modes, n_omega)``; present exactly for
        ``kind="diagonal"``.
    dense : numpy.ndarray or None
        Hermitian dimensionless metric, shape
        ``(n_intervals, n_deg, n_modes, n_modes, n_omega)``; present exactly for
        ``kind="dense"``.
    error : numpy.ndarray
        Non-negative dimensionless bound on the stored representation, shape
        ``(n_intervals, n_omega)``. Declared by the caller; nothing here infers it.
    minimum_eigenvalue, maximum_eigenvalue : numpy.ndarray
        Non-negative dimensionless extreme eigenvalues of W, shape
        ``(n_intervals, n_omega)``, with the minimum never above the maximum. A
        minimum of zero is a semidefinite metric: it may be stored as flux data but
        certifies no positive error or noise norm.
    gauge_id, certificate_id : str
        Non-empty phase/coordinate identity and the identity of the certificate the
        bounds came from. Scalar identities, never arrays, enter resume identity.

    Raises
    ------
    TranslationError
        Naming the field that failed.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import ModalMetricTables
    >>> zeros = np.zeros((1, 3))
    >>> metric = ModalMetricTables(
    ...     "identity", np.linspace(1e15, 2e15, 3), np.array([0.0]), None, None,
    ...     zeros, np.ones((1, 3)), np.ones((1, 3)), "lab", "orthonormal",
    ... )
    >>> metric.kind, metric.n_omega
    ('identity', 3)
    """

    kind: MetricKind
    omega: Float[np.ndarray, " n_omega_metric"]
    z_edges: Float[np.ndarray, " n_edges_metric"]
    diagonal: (
        Float[
            np.ndarray,
            "n_intervals_diagonal n_deg_diagonal n_modes_diagonal n_omega_diagonal",
        ]
        | None
    )
    dense: (
        Complex[
            np.ndarray,
            "n_intervals_dense n_deg_dense n_modes_dense_out n_modes_dense_in"
            " n_omega_dense",
        ]
        | None
    )
    error: Float[np.ndarray, "n_intervals_metric_error n_omega_metric_error"]
    minimum_eigenvalue: Float[np.ndarray, "n_intervals_metric_min n_omega_metric_min"]
    maximum_eigenvalue: Float[np.ndarray, "n_intervals_metric_max n_omega_metric_max"]
    gauge_id: str
    certificate_id: str

    @property
    def n_omega(self) -> int:
        """Return how many coarse frequencies the metric is tabulated on.

        Returns
        -------
        int
            The length of ``omega``.
        """
        return len(self.omega)

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``ModalMetricTables(diagonal, 4 channels, 8 frequencies,
            1 z)``.
        """
        return line(
            "ModalMetricTables",
            [
                ("", self.kind),
                ("", f"{self.n_modes} channel{'' if self.n_modes == 1 else 's'}"),
                ("", f"{self.n_omega} frequencies"),
                ("", f"{len(self.z_edges)} z"),
            ],
        )

    @property
    def n_modes(self) -> int:
        """Return how many channels the metric describes.

        Returns
        -------
        int
            The channel count of whichever coefficient array is stored; 0 for
            ``"identity"``, which owns none and takes its size from the state.
        """
        if self.dense is not None:
            return int(self.dense.shape[2])
        return 0 if self.diagonal is None else int(self.diagonal.shape[2])

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "ModalMetricTables: the physical flux metric of a modal basis", [
            ("kind", self.kind),
            (
                "frequencies",
                f"{self.n_omega}, {self.omega[0]:.4e} to {self.omega[-1]:.4e} rad/s",
            ),
            (
                "z",
                f"snapshot at {engineering(float(self.z_edges[0]), 'm')}"
                if len(self.z_edges) == 1
                else f"{len(self.z_edges)}, "
                f"{engineering(float(self.z_edges[0]), 'm')} to "
                f"{engineering(float(self.z_edges[-1]), 'm')}",
            ),
            ("channels", self.n_modes or "from the state"),
            ("eigenvalues", f"{float(np.min(self.minimum_eigenvalue)):.3e} smallest"),
            ("error", f"{float(np.max(self.error)):.3e} largest"),
            ("gauge", self.gauge_id),
            ("certificate", self.certificate_id),
        ]

    def __post_init__(self) -> None:
        """Copy, validate and freeze every leaf array.

        Raises
        ------
        TranslationError
            If the kind is unknown or unqualified, the present arrays disagree with
            it, an extent is wrong, or a value is out of range.
        """
        if self.kind not in ("identity", "diagonal", "dense"):
            msg = (
                f"modal_metric.kind must be 'identity', 'diagonal' or 'dense', got "
                f"{self.kind!r}"
            )
            raise TranslationError(msg)
        if not (self.gauge_id and self.certificate_id):
            msg = "modal_metric: gauge_id and certificate_id must be non-empty"
            raise TranslationError(msg)
        object.__setattr__(self, "omega", _vector(self.omega, "modal_metric.omega"))
        object.__setattr__(
            self, "z_edges", _vector(self.z_edges, "modal_metric.z_edges")
        )
        if self.omega[0] < 0:
            msg = "modal_metric.omega must be non-negative"
            raise TranslationError(msg)
        self._validate_arrays()

    def _validate_arrays(self) -> None:
        """Check the coefficient array against the kind and the bounds against it.

        Raises
        ------
        TranslationError
            Naming the field that failed.
        """
        if (self.dense is not None) != (self.kind == "dense"):
            msg = (
                f"modal_metric.dense must be present exactly for kind='dense'; "
                f"kind is {self.kind!r} and the array is "
                f"{'absent' if self.dense is None else 'present'}"
            )
            raise TranslationError(msg)
        if (self.diagonal is not None) != (self.kind == "diagonal"):
            msg = (
                f"modal_metric.diagonal must be present exactly for kind='diagonal'; "
                f"kind is {self.kind!r} and the array is "
                f"{'absent' if self.diagonal is None else 'present'}"
            )
            raise TranslationError(msg)
        intervals = interval_count(len(self.z_edges))
        if self.diagonal is not None:
            weights = np.array(self.diagonal, dtype=np.float64, order="C", copy=True)
            if weights.ndim != 4 or weights.shape[0] != intervals:
                msg = (
                    "modal_metric.diagonal must have shape (n_intervals, n_deg, "
                    f"n_modes, n_omega) with {intervals} interval(s), got "
                    f"{weights.shape}"
                )
                raise TranslationError(msg)
            if weights.shape[3] != self.n_omega or not weights.shape[2]:
                msg = (
                    "modal_metric.diagonal must carry one column per coarse frequency "
                    f"and at least one channel, got {weights.shape}"
                )
                raise TranslationError(msg)
            _degree_axis(weights, "modal_metric.diagonal")
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
                msg = "modal_metric.diagonal must be finite and positive"
                raise TranslationError(msg)
            weights.setflags(write=False)
            object.__setattr__(self, "diagonal", weights)
        if self.dense is not None:
            self._validate_dense(intervals)
        shape = (intervals, self.n_omega)
        for name in ("error", "minimum_eigenvalue", "maximum_eigenvalue"):
            values = _owned(
                getattr(self, name), f"modal_metric.{name}", shape, np.float64
            )
            if np.any(values < 0):
                msg = f"modal_metric.{name} must be non-negative"
                raise TranslationError(msg)
            object.__setattr__(self, name, values)
        if np.any(self.minimum_eigenvalue > self.maximum_eigenvalue):
            msg = (
                "modal_metric.minimum_eigenvalue must not exceed "
                "maximum_eigenvalue at any interval and frequency"
            )
            raise TranslationError(msg)

    def _validate_dense(self, intervals: int) -> None:
        """Check the dense coefficient block's extents and its Hermitian symmetry.

        A flux metric is the Hermitian Poynting pairing of a basis with itself, so a
        stored dense one that is not Hermitian is not a metric at all: its quadratic
        form would have an imaginary part and the energies built on it would not be
        real. The budget is ``64 n eps`` times the block's own norm, the same scale
        :meth:`solprop.modes.vector.ModalSample._validate_flux` uses on one sample's
        metric, because that is what a ``U.H W U`` product loses to rounding.

        Parameters
        ----------
        intervals : int
            Interval count from the stored ``z_edges``.

        Raises
        ------
        TranslationError
            If an extent disagrees or the block is not Hermitian.
        """
        block, modes = _square(self.dense, "modal_metric.dense", 5)
        if block.shape[0] != intervals or block.shape[4] != self.n_omega:
            msg = (
                "modal_metric.dense must have shape (n_intervals, n_deg, n_modes, "
                f"n_modes, n_omega) with {intervals} interval(s) and "
                f"{self.n_omega} frequencies, got {block.shape}"
            )
            raise TranslationError(msg)
        _degree_axis(block, "modal_metric.dense")
        moved = np.moveaxis(block, (2, 3), (-2, -1))
        defect = float(np.max(abs(moved - np.conj(np.swapaxes(moved, -1, -2)))))
        tolerance = 64 * modes * float(np.finfo(float).eps) * float(np.max(abs(block)))
        if defect > tolerance:
            msg = (
                f"modal_metric.dense must be Hermitian at every interval, degree "
                f"and frequency; the largest departure is {defect:.4g}, above the "
                f"{tolerance:.4g} rounding of the block's own norm"
            )
            raise TranslationError(msg)
        object.__setattr__(self, "dense", block)


@dataclass(frozen=True, eq=False)
class CoupledFlowTables(Displayed):
    """A generator whose channels couple, frozen or ordered along z.

    The leaf of contract section 3 that carries the coherent wall operator and the
    taper coupling. ``numpy_ref`` executes three combinations. The first is the dense
    generator of a single constant section, ``section_methods ==
    ("constant_schur",)``, no connection and an exactly zero residual, built by
    :func:`constant_schur_flow` from a constant coherent block. The second is the
    dense generator of any number of ``"frozen_residual"`` sections at any degree
    along z, with the taper connection and an explicit residual the Runge-Kutta
    source carries, built by :func:`frozen_residual_flow` from the same block
    evaluated along the declared profile. The third is ``"cf4"`` sections, built by
    :func:`cf4_flow` from the same expansion, whose whole generator the backend
    integrates by the commutator-free fourth-order exponential with no explicit
    residual, mixed with ``"frozen_residual"`` sections only where a declared
    fallback put them. Every other method, representation and mixture is refused by
    name at the backend, and the compiled backends refuse ``"cf4"`` sections.

    The dense ``generator`` is optional. The factored alternative combines fixed
    matrices with scalar weights, rank-one wall terms and connection coefficients
    without a mandatory dense ``(z, omega, modes, modes)`` allocation; exactly one of
    the two complete representations is stored, never an ambiguous mixture.

    Attributes
    ----------
    omega : numpy.ndarray
        The canonical coarse angular frequencies in rad/s, shape ``(n_omega,)``,
        increasing and non-negative. A problem's leaf carries ``grids.omega``
        itself, whose first bin is zero on a field grid
        (:meth:`solprop.translate.problem.KernelProblem.validate`).
    z_edges : numpy.ndarray
        Interval edges in m, shape ``(n_edges,)``, increasing.
    generator : numpy.ndarray or None
        Dense generator in 1/m, shape
        ``(n_intervals, n_deg, n_modes, n_modes, n_omega)``; the dense
        representation, mutually exclusive with the factored one.
    fixed_terms : numpy.ndarray or None
        Fixed matrices in 1/m, shape ``(n_terms, n_modes, n_modes, n_omega)``.
    term_weights : numpy.ndarray or None
        Dimensionless scalar weights of those matrices, shape
        ``(n_intervals, n_deg, n_terms)``; supplied with ``fixed_terms``.
    wall_vectors : numpy.ndarray or None
        Rank-one wall vectors, **dimensionless**, shape
        ``(n_intervals, n_modes, n_omega)``; supplied with ``wall_weights``.
    wall_weights : numpy.ndarray or None
        Their scalar weights in **1/m**, shape ``(n_intervals, n_deg, n_omega)``, so
        that ``w (v x v)`` carries the generator's own 1/m, and
        :func:`constant_schur_flow` checks the reconstruction.
    connection : numpy.ndarray or None
        Taper connection coefficients in 1/m, shape
        ``(n_intervals, n_deg, n_modes, n_modes, n_omega)``.
    generator_error : numpy.ndarray
        Non-negative dimensionless representation error, shape
        ``(n_intervals, n_omega)``.
    residual_norm_bound : numpy.ndarray
        Non-negative bound in 1/m on the explicit residual, same shape; zero on a
        ``"cf4"`` section, which has none.
    inverse_condition_bound : numpy.ndarray
        Dimensionless bound of at least one on the conditioning of the inverse
        action, same shape; on a ``"cf4"`` section, on the product of the four factor
        norms of a forward and inverse pair (:func:`cf4_flow`).
    section_methods : tuple of str
        One of ``"diagonal"``, ``"constant_schur"``, ``"frozen_residual"`` or
        ``"cf4"`` per interval.
    gauge_id, certificate_id : str
        Non-empty identities; scalars, never arrays, enter resume identity.

    Raises
    ------
    TranslationError
        Naming the field that failed.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import CoupledFlowTables
    >>> flow = CoupledFlowTables(
    ...     np.linspace(1e15, 2e15, 2), np.array([0.0, 1.0]),
    ...     np.zeros((1, 1, 2, 2, 2), dtype=complex), None, None, None, None, None,
    ...     np.zeros((1, 2)), np.zeros((1, 2)), np.ones((1, 2)), ("diagonal",),
    ...     "lab", "none",
    ... )
    >>> flow.representation, flow.n_modes
    ('dense', 2)
    """

    omega: Float[np.ndarray, " n_omega_flow"]
    z_edges: Float[np.ndarray, " n_edges_flow"]
    generator: (
        Complex[
            np.ndarray,
            "n_intervals_generator n_deg_generator n_modes_generator_out"
            " n_modes_generator_in n_omega_generator",
        ]
        | None
    )
    fixed_terms: (
        Complex[
            np.ndarray,
            "n_terms_fixed n_modes_fixed_out n_modes_fixed_in n_omega_fixed",
        ]
        | None
    )
    term_weights: (
        Complex[
            np.ndarray,
            "n_intervals_term_weights n_deg_term_weights n_terms_term_weights",
        ]
        | None
    )
    wall_vectors: (
        Complex[
            np.ndarray,
            "n_intervals_wall_vectors n_modes_wall_vectors n_omega_wall_vectors",
        ]
        | None
    )
    wall_weights: (
        Complex[
            np.ndarray,
            "n_intervals_wall_weights n_deg_wall_weights n_omega_wall_weights",
        ]
        | None
    )
    connection: (
        Complex[
            np.ndarray,
            "n_intervals_connection n_deg_connection n_modes_connection_out"
            " n_modes_connection_in n_omega_connection",
        ]
        | None
    )
    generator_error: Float[
        np.ndarray, "n_intervals_generator_error n_omega_generator_error"
    ]
    residual_norm_bound: Float[
        np.ndarray, "n_intervals_residual_bound n_omega_residual_bound"
    ]
    inverse_condition_bound: Float[
        np.ndarray, "n_intervals_inverse_bound n_omega_inverse_bound"
    ]
    section_methods: tuple[SectionMethod, ...]
    gauge_id: str
    certificate_id: str

    @property
    def representation(self) -> Literal["dense", "factored"]:
        """Return which of the two complete representations is stored.

        Returns
        -------
        {'dense', 'factored'}
            ``"dense"`` when the generator array is present, ``"factored"`` when the
            fixed matrices and their weights are.
        """
        return "dense" if self.generator is not None else "factored"

    @property
    def n_modes(self) -> int:
        """Return how many channels the flow couples.

        Returns
        -------
        int
            The channel count of the stored representation.
        """
        terms = self.generator if self.generator is not None else self.fixed_terms
        assert terms is not None
        return int(terms.shape[-2])

    @property
    def n_deg(self) -> int:
        """Return how many Chebyshev coefficients along z each section carries.

        Returns
        -------
        int
            The degree axis of the dense generator, or one for the factored
            representation, whose weights carry the z dependence instead.
        """
        return 1 if self.generator is None else int(self.generator.shape[1])

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``CoupledFlowTables(dense, 4 channels, 8 frequencies,
            2 sections)``.
        """
        return line(
            "CoupledFlowTables",
            [
                ("", self.representation),
                ("", f"{self.n_modes} channel{'' if self.n_modes == 1 else 's'}"),
                ("", f"{len(self.omega)} frequencies"),
                (
                    "",
                    f"{len(self.section_methods)} section"
                    f"{'' if len(self.section_methods) == 1 else 's'}",
                ),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "CoupledFlowTables: a generator whose channels couple", [
            ("representation", self.representation),
            (
                "frequencies",
                f"{len(self.omega)}, {self.omega[0]:.4e} to {self.omega[-1]:.4e} rad/s",
            ),
            (
                "z",
                f"{engineering(float(self.z_edges[0]), 'm')} to "
                f"{engineering(float(self.z_edges[-1]), 'm')}",
            ),
            ("channels", self.n_modes),
            ("sections", ", ".join(self.section_methods)),
            ("wall", "yes" if self.wall_vectors is not None else "no"),
            ("connection", "yes" if self.connection is not None else "no"),
            ("residual", f"{float(np.max(self.residual_norm_bound)):.3e} 1/m largest"),
            ("gauge", self.gauge_id),
            ("certificate", self.certificate_id),
        ]

    def __post_init__(self) -> None:
        """Copy, validate and freeze every leaf array.

        Raises
        ------
        TranslationError
            If the two representations are mixed or both absent, a half of a paired
            group is missing, an extent is wrong, or a bound is out of range.
        """
        if not (self.gauge_id and self.certificate_id):
            msg = "coupled_flow: gauge_id and certificate_id must be non-empty"
            raise TranslationError(msg)
        object.__setattr__(self, "omega", _vector(self.omega, "coupled_flow.omega"))
        object.__setattr__(
            self, "z_edges", _vector(self.z_edges, "coupled_flow.z_edges")
        )
        if self.omega[0] < 0:
            msg = "coupled_flow.omega must be non-negative"
            raise TranslationError(msg)
        self._validate_representation()
        self._validate_optional()
        self._validate_bounds()

    def _validate_representation(self) -> None:
        """Check exactly one complete generator representation is stored.

        Raises
        ------
        TranslationError
            If the dense and factored representations are mixed or both absent.
        """
        factored = (self.fixed_terms is not None, self.term_weights is not None)
        if self.generator is not None and any(factored):
            msg = (
                "coupled_flow carries both a dense generator and factored terms; "
                "exactly one complete representation is permitted"
            )
            raise TranslationError(msg)
        if self.generator is None and not all(factored):
            msg = (
                "coupled_flow needs either a dense generator or both fixed_terms and "
                "term_weights"
            )
            raise TranslationError(msg)
        intervals = interval_count(len(self.z_edges))
        n_omega = len(self.omega)
        if self.generator is not None:
            dense, _ = _square(self.generator, "coupled_flow.generator", 5)
            if dense.shape[0] != intervals or dense.shape[4] != n_omega:
                msg = (
                    "coupled_flow.generator must have shape (n_intervals, n_deg, "
                    f"n_modes, n_modes, n_omega) with {intervals} interval(s) and "
                    f"{n_omega} frequencies, got {dense.shape}"
                )
                raise TranslationError(msg)
            _degree_axis(dense, "coupled_flow.generator")
            object.__setattr__(self, "generator", dense)
            return
        terms, _ = _square(self.fixed_terms, "coupled_flow.fixed_terms", 4)
        if terms.shape[3] != n_omega or not terms.shape[0]:
            msg = (
                "coupled_flow.fixed_terms must have shape (n_terms, n_modes, "
                f"n_modes, n_omega) with at least one term and {n_omega} "
                f"frequencies, got {terms.shape}"
            )
            raise TranslationError(msg)
        weights = np.array(self.term_weights, dtype=np.complex128, order="C", copy=True)
        if (
            weights.ndim != 3
            or weights.shape[0] != intervals
            or weights.shape[2] != terms.shape[0]
        ):
            msg = (
                "coupled_flow.term_weights must have shape (n_intervals, n_deg, "
                f"n_terms) with {intervals} interval(s) and {terms.shape[0]} term(s), "
                f"got {weights.shape}"
            )
            raise TranslationError(msg)
        _degree_axis(weights, "coupled_flow.term_weights")
        if not np.all(np.isfinite(weights)):
            msg = "coupled_flow.term_weights must be finite"
            raise TranslationError(msg)
        weights.setflags(write=False)
        object.__setattr__(self, "fixed_terms", terms)
        object.__setattr__(self, "term_weights", weights)

    def _validate_optional(self) -> None:
        """Check the wall pair and the connection against the record's extents.

        Raises
        ------
        TranslationError
            If one half of the wall pair is missing or an extent disagrees.
        """
        intervals = interval_count(len(self.z_edges))
        n_omega = len(self.omega)
        modes = self.n_modes
        wall = (self.wall_vectors is not None, self.wall_weights is not None)
        if any(wall) and not all(wall):
            msg = "coupled_flow.wall_vectors and wall_weights are supplied together"
            raise TranslationError(msg)
        if self.wall_vectors is not None:
            vectors = _owned(
                self.wall_vectors,
                "coupled_flow.wall_vectors",
                (intervals, modes, n_omega),
                np.complex128,
            )
            object.__setattr__(self, "wall_vectors", vectors)
            weights = np.array(
                self.wall_weights, dtype=np.complex128, order="C", copy=True
            )
            if (
                weights.ndim != 3
                or weights.shape[0] != intervals
                or weights.shape[2] != n_omega
            ):
                msg = (
                    "coupled_flow.wall_weights must have shape (n_intervals, n_deg, "
                    f"n_omega) with {intervals} interval(s) and {n_omega} "
                    f"frequencies, got {weights.shape}"
                )
                raise TranslationError(msg)
            _degree_axis(weights, "coupled_flow.wall_weights")
            if not np.all(np.isfinite(weights)):
                msg = "coupled_flow.wall_weights must be finite"
                raise TranslationError(msg)
            weights.setflags(write=False)
            object.__setattr__(self, "wall_weights", weights)
        if self.connection is not None:
            link, link_modes = _square(self.connection, "coupled_flow.connection", 5)
            if (
                link.shape[0] != intervals
                or link.shape[4] != n_omega
                or link_modes != modes
            ):
                msg = (
                    "coupled_flow.connection must have shape (n_intervals, n_deg, "
                    f"n_modes, n_modes, n_omega) with {intervals} interval(s), "
                    f"{modes} channels and {n_omega} frequencies, got {link.shape}"
                )
                raise TranslationError(msg)
            _degree_axis(link, "coupled_flow.connection")
            object.__setattr__(self, "connection", link)

    def _validate_bounds(self) -> None:
        """Check the three certificate rows and the per-section method names.

        Raises
        ------
        TranslationError
            If a bound is out of range or a section method is unknown.
        """
        intervals = interval_count(len(self.z_edges))
        shape = (intervals, len(self.omega))
        for name in (
            "generator_error",
            "residual_norm_bound",
            "inverse_condition_bound",
        ):
            values = _owned(
                getattr(self, name), f"coupled_flow.{name}", shape, np.float64
            )
            floor = 1.0 if name == "inverse_condition_bound" else 0.0
            if np.any(values < floor):
                msg = f"coupled_flow.{name} must be at least {floor}"
                raise TranslationError(msg)
            object.__setattr__(self, name, values)
        methods = tuple(self.section_methods)
        if len(methods) != intervals:
            msg = (
                f"coupled_flow.section_methods must name one method per interval, "
                f"got {len(methods)} for {intervals}"
            )
            raise TranslationError(msg)
        allowed = ("diagonal", "constant_schur", "frozen_residual", "cf4")
        unknown = [method for method in methods if method not in allowed]
        if unknown:
            msg = (
                f"coupled_flow.section_methods has unknown entries {unknown}; "
                f"the methods are {list(allowed)}"
            )
            raise TranslationError(msg)
        object.__setattr__(self, "section_methods", methods)


def orthonormal_metric(
    omega: Float[np.ndarray, " n_omega"],
    z_edges: Float[np.ndarray, " n_edges"],
    *,
    defect: float,
    gauge_id: str,
    certificate_id: str,
) -> ModalMetricTables:
    """Build the identity metric of a basis certified orthonormal in true power.

    A Poynting-normalised bound basis whose sampled Gram passed the construction
    gate of :class:`solprop.translate.modal_tables.DenseModalTable` has ``W = I`` to
    within that gate's budget, so the metric owns no coefficient array at all and the
    measured defect is what the error row carries. The extreme eigenvalues are one: a
    certified positive metric, which is what the noise draw and the controller norm
    need.

    Parameters
    ----------
    omega : numpy.ndarray
        The canonical coarse angular frequencies in rad/s, increasing.
    z_edges : numpy.ndarray
        Stored positions in m, increasing, covering the run.
    defect : float
        The measured ``norm(W - I, 2)`` of the basis, non-negative and dimensionless:
        the table's ``gram_defect``.
    gauge_id : str
        The basis's phase and coordinate identity.
    certificate_id : str
        Identity of the certificate the defect came from.

    Returns
    -------
    ModalMetricTables
        The validated identity metric.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import orthonormal_metric
    >>> metric = orthonormal_metric(
    ...     np.linspace(0.0, 2e15, 3), np.array([0.0, 1.0]), defect=1e-12,
    ...     gauge_id="lab", certificate_id="sampled-gram",
    ... )
    >>> metric.kind, metric.diagonal is None
    ('identity', True)
    """
    rows = (interval_count(len(np.asarray(z_edges))), len(np.asarray(omega)))
    return ModalMetricTables(
        "identity",
        omega,
        z_edges,
        None,
        None,
        np.full(rows, float(defect)),
        np.ones(rows),
        np.ones(rows),
        gauge_id,
        certificate_id,
    )


def metric_at(
    metric: ModalMetricTables | None, z: float
) -> Complex[np.ndarray, "n_modes_at_out n_modes_at_in n_omega_at"] | None:
    """Evaluate a basis metric at one statistics or observation plane.

    The one arithmetic path from the stored leaf to the ``W`` a statistic is handed,
    so that the run-time reduction and a later check of it read the same numbers.
    ``None`` is the identity and is returned as ``None`` rather than as an array of
    ones: the statistics take ``metric=None`` as their fast, bitwise-unchanged path,
    and building an identity here would make every unweighted run pay for a matrix
    product it does not need.

    Parameters
    ----------
    metric : ModalMetricTables or None
        The leaf, or ``None`` for a problem that carries none.
    z : float
        Position in m inside the metric's own span.

    Returns
    -------
    numpy.ndarray or None
        ``W(omega)``, shape ``(n_modes, n_modes, n_omega)``, dimensionless; ``None``
        for an identity metric or no metric at all.

    Raises
    ------
    TranslationError
        If the metric is diagonal, which no route builds and nothing executes.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import dense_metric, metric_at
    >>> sampled = np.broadcast_to(np.eye(2, dtype=complex)[:, :, None], (2, 2, 3))
    >>> metric = dense_metric(
    ...     np.linspace(1e15, 2e15, 3), np.array([0.0, 1.0]), sampled,
    ...     defect=0.0, gauge_id="lab+circular(0,1)", certificate_id="sampled-gram",
    ... )
    >>> metric_at(metric, 0.5).shape
    (2, 2, 3)
    >>> metric_at(None, 0.5) is None
    True
    """
    if metric is None or metric.kind == "identity":
        return None
    block = metric.dense
    if block is None:
        msg = (
            "a diagonal modal metric has no executed statistics path: nothing in "
            "this package builds one, and a per-channel weight is not the same "
            "object as linear.power_weight"
        )
        raise TranslationError(msg)
    expansion = PiecewiseChebyshev(
        block,
        metric.z_edges,
        np.full(block.shape[0], block.shape[1] - 1, dtype=np.int64),
        np.empty(0),
    )
    return np.asarray(expansion(float(z)), dtype=np.complex128)


def dense_metric(
    omega: Float[np.ndarray, " n_omega"],
    z_edges: Float[np.ndarray, " n_edges"],
    sampled: Complex[
        np.ndarray, "n_modes_sampled_out n_modes_sampled_in n_omega_sampled"
    ],
    *,
    defect: float,
    gauge_id: str,
    certificate_id: str,
) -> ModalMetricTables:
    r"""Build the dense metric of a basis from its own sampled flux Gram.

    The companion of :func:`orthonormal_metric` for a basis whose flux metric is no
    longer the identity in any gauge it could be written in. The caller supplies
    ``W(omega)`` measured on the basis itself -- the Hermitian Poynting pairing
    :func:`solprop.modes.vector.flux_product` integrates -- and this stores it as the
    documented leaf of contract section 3, with the eigenvalue certificate a
    controller norm and a noise draw need.

    **Why it is genuinely dense.** Under a declared unitary change of basis ``U`` a
    reference whose sampled Gram is ``I + delta`` becomes ``I + U.H delta U``. The
    exact identity is invariant, but the *measured* defect is not: the off-diagonal
    of ``U.H delta U`` is not the off-diagonal of ``delta``, and it is the measured
    object a run carries. Storing a literal identity here instead would be a
    decorated identity rather than a declaration of what the basis is.

    **The metric is constant along z** in this constructor: one interval carrying one
    Chebyshev coefficient, which is degree zero, because the general modal route
    propagates constant geometry and its reference is a snapshot.

    Parameters
    ----------
    omega : numpy.ndarray
        The canonical coarse angular frequencies in rad/s, increasing, shape
        ``(n_omega,)``.
    z_edges : numpy.ndarray
        Stored positions in m, increasing, covering the run.
    sampled : numpy.ndarray
        The Hermitian ``W(omega)``, shape ``(n_modes, n_modes, n_omega)``,
        dimensionless and positive definite at every frequency.
    defect : float
        The measured ``norm(W - I, 2)`` of the **reference** basis this one was
        built from, non-negative and dimensionless: the table's ``gram_defect``,
        which a unitary transform leaves unchanged in the spectral norm.
    gauge_id : str
        The basis's composed phase and coordinate identity.
    certificate_id : str
        Identity of the certificate `defect` came from.

    Returns
    -------
    ModalMetricTables
        The validated dense metric, with the extreme eigenvalues of the stored
        block per frequency.

    Raises
    ------
    TranslationError
        If the sampled metric is not a square positive-definite Hermitian block on
        the given frequencies.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import dense_metric
    >>> sampled = np.broadcast_to(np.eye(2, dtype=complex)[:, :, None], (2, 2, 3))
    >>> metric = dense_metric(
    ...     np.linspace(1e15, 2e15, 3), np.array([0.0, 1.0]), sampled,
    ...     defect=1e-12, gauge_id="lab+circular(0,1)", certificate_id="sampled-gram",
    ... )
    >>> metric.kind, metric.n_modes
    ('dense', 2)
    """
    block = np.asarray(sampled, dtype=np.complex128)
    frequencies = len(np.asarray(omega))
    if (
        block.ndim != 3
        or block.shape[0] != block.shape[1]
        or not block.shape[0]
        or block.shape[2] != frequencies
    ):
        msg = (
            "a dense metric's sampled Gram must have shape (n_modes, n_modes, "
            f"n_omega) with {frequencies} frequencies, got {block.shape}"
        )
        raise TranslationError(msg)
    rows = (interval_count(len(np.asarray(z_edges))), frequencies)
    # eigvalsh reads the lower triangle of a Hermitian matrix, so it is the
    # eigenvalue routine a metric is entitled to: a stored block whose Hermitian
    # defect is above rounding is refused by the record's own validation.
    eigenvalues = np.linalg.eigvalsh(np.moveaxis(block, (0, 1), (-2, -1)))
    if np.any(eigenvalues <= 0.0):
        msg = (
            "a dense metric must be positive definite at every frequency; the "
            f"smallest eigenvalue is {float(np.min(eigenvalues)):.6g}"
        )
        raise TranslationError(msg)
    return ModalMetricTables(
        "dense",
        omega,
        z_edges,
        None,
        np.broadcast_to(block[None, None], (rows[0], 1, *block.shape)),
        np.full(rows, float(defect)),
        np.broadcast_to(np.min(eigenvalues, axis=-1)[None, :], rows),
        np.broadcast_to(np.max(eigenvalues, axis=-1)[None, :], rows),
        gauge_id,
        certificate_id,
    )


def schur_blocks(
    generator: Complex[np.ndarray, "n_modes_schur_out n_modes_schur_in n_omega_schur"],
) -> tuple[
    Complex[np.ndarray, "n_omega_z n_modes_z_out n_modes_z_in"],
    Complex[np.ndarray, "n_omega_t n_modes_t_out n_modes_t_in"],
]:
    """Return the complex Schur factors of a generator, one per frequency.

    ``G = Z T Z^H`` with ``Z`` unitary and ``T`` upper triangular, which is the
    decomposition a matrix-function action of a **nonnormal** generator is entitled
    to: an eigenvector basis of a wall block near a degeneracy is arbitrarily
    ill-conditioned, and the Schur factor is not. A frequency whose block
    is exactly zero -- an out-of-band bin the wall formula was never evaluated at --
    is not decomposed at all: its factors are the identity and zero, which is what the
    decomposition would give and costs nothing.

    Parameters
    ----------
    generator : numpy.ndarray
        The generator in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``Z`` and ``T``, each complex of shape ``(n_omega, n_modes, n_modes)``: the
        frequency axis leads, because that is the axis a batched action broadcasts
        over.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import schur_blocks
    >>> block = np.zeros((2, 2, 3), dtype=complex)
    >>> block[0, 1, 1] = 3.0
    >>> vectors, triangular = schur_blocks(block)
    >>> vectors.shape, float(abs(triangular[1]).max())
    ((3, 2, 2), 3.0)
    """
    blocks = np.ascontiguousarray(
        np.moveaxis(np.asarray(generator, dtype=np.complex128), -1, 0)
    )
    modes = blocks.shape[1]
    vectors = np.broadcast_to(np.eye(modes, dtype=np.complex128), blocks.shape).copy()
    triangular = np.zeros_like(blocks)
    for index, block in enumerate(blocks):
        if not np.any(block):
            continue
        upper, unitary = schur(block, output="complex")
        triangular[index] = upper
        vectors[index] = unitary
    return vectors, triangular


def eigenbasis_limit(n_modes: int) -> float:
    """Return the largest eigenbasis condition the compiled flow actions accept.

    The compiled backends form a flow action ``Z exp(l T) Z^H`` from the eigenbasis
    of the Schur factor, ``W exp(l Lambda) Y`` (:func:`schur_eigenbasis`), where its
    rounding is bounded by ``kappa (n + 11 + 2 a) u`` relative to the action, with
    ``kappa = || |W| ||_2 || |Y| ||_2`` and ``a = max |l lambda|``. The reference's
    scaling-and-squaring is bounded by ``16 u + 2 sqrt(2) n (n + 2) u`` at its
    smallest -- no squaring, a vanishing exponent: the sixteen-term series and the
    two products of the back-transform -- and grows with the step from there. The limit
    is the ``kappa`` at which the two routes' vanishing-step rounding terms meet -- a
    heuristic gate, not a bound comparison at every step length (the eigenbasis bound
    also carries the measured ``||WY - I||`` and a drift that grows with the step);
    measured, the eigenbasis route is at least as accurate as the reference's on every
    case tried; a bin beyond it (a nearly degenerate
    pair with a nonnormal coupling, where ``kappa`` grows as the inverse of the
    separation) is formed by the compiled scaling-and-squaring instead. The bounds are
    derived in ``tests/flow_parity.py`` and ``docs/dev/backends.md`` section 3.9.

    Parameters
    ----------
    n_modes : int
        Channels of the flow's block, positive.

    Returns
    -------
    float
        The limit, dimensionless: 2.04 for one channel, rising with ``n``; a
        one-channel block has ``kappa = 1`` exactly.

    Examples
    --------
    >>> from solprop.translate.coupled import eigenbasis_limit
    >>> round(eigenbasis_limit(3), 3)
    4.173
    """
    return (16.0 + 2.0 * math.sqrt(2.0) * n_modes * (n_modes + 2)) / (n_modes + 11.0)


def schur_eigenbasis(
    vectors: Complex[np.ndarray, "n_omega_z n_modes_z_out n_modes_z_in"],
    triangular: Complex[np.ndarray, "n_omega_t n_modes_t_out n_modes_t_in"],
) -> tuple[
    Complex[np.ndarray, "n_omega_l n_modes_l_out n_modes_l_in"],
    Complex[np.ndarray, "n_omega_r n_modes_r_out n_modes_r_in"],
    Complex[np.ndarray, "n_omega_e n_modes_e"],
    Bool[np.ndarray, " n_omega_m"],
]:
    """Return the eigenbasis of each Schur factor, and where it is well conditioned.

    With ``G = Z T Z^H`` and ``T`` upper triangular with distinct diagonal
    ``lambda``, ``T = V Lambda V^-1`` with ``V`` unit upper triangular, so every
    action of the flow over a length ``l`` is ``W exp(l Lambda) Y`` with ``W = Z V``
    and ``Y = V^-1 Z^H``: ``n`` exponentials and one ``n^3`` product, shared by the
    forward and inverse actions and by every stage length of a step. ``V`` is formed
    column by column by back substitution, ``V_ij = sum_k T_ik V_kj / (lambda_j -
    lambda_i)``, and its inverse by the division-free unit triangular recurrence, each
    sum taken from zero in increasing ``k``.

    Where two diagonal entries nearly coincide with a coupling between them, ``V`` is
    ill-conditioned and the route loses accuracy as the inverse of the separation;
    where they coincide exactly it does not exist. A bin is therefore marked for this
    route only where ``V`` is finite and ``|| |W| ||_2 || |Y| ||_2`` is within
    :func:`eigenbasis_limit`; the others are formed by scaling and squaring. Their
    ``W`` and ``Y`` are returned as zeros.

    Parameters
    ----------
    vectors : numpy.ndarray
        The unitary Schur factors ``Z``, complex, shape ``(n_omega, n_modes,
        n_modes)``, as :func:`schur_blocks` returns them.
    triangular : numpy.ndarray
        The triangular factors ``T``, 1/m, of the same shape.

    Returns
    -------
    tuple of numpy.ndarray
        ``W`` and ``Y``, complex of the input's shape; the eigenvalues ``lambda``,
        1/m, shape ``(n_omega, n_modes)``; and the boolean route, shape
        ``(n_omega,)``, true where the eigenbasis is used.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import schur_eigenbasis
    >>> vectors = np.eye(2, dtype=complex)[None]
    >>> triangular = np.array([[[-1.0, 0.1], [0.0, -2.0]]], dtype=complex)
    >>> left, right, eigenvalues, route = schur_eigenbasis(vectors, triangular)
    >>> bool(route[0]), bool(np.allclose(left[0] @ right[0], np.eye(2)))
    (True, True)
    """
    stack = np.asarray(triangular, dtype=np.complex128)
    unitary = np.asarray(vectors, dtype=np.complex128)
    n_bins, n_modes, _ = stack.shape
    eigenvalues = np.ascontiguousarray(np.diagonal(stack, axis1=-2, axis2=-1))
    identity = np.broadcast_to(np.eye(n_modes, dtype=np.complex128), stack.shape)
    right_vectors = identity.copy()
    inverse = identity.copy()
    # An exact coincidence divides by zero, to an infinity or a NaN, and routes the
    # bin to scaling and squaring below; nothing is raised or warned for it.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for column in range(n_modes):
            for row in range(column - 1, -1, -1):
                total = np.zeros(n_bins, dtype=np.complex128)
                for inner in range(row + 1, column + 1):
                    total += stack[:, row, inner] * right_vectors[:, inner, column]
                right_vectors[:, row, column] = total / (
                    eigenvalues[:, column] - eigenvalues[:, row]
                )
        for column in range(n_modes):
            for row in range(column - 1, -1, -1):
                total = np.zeros(n_bins, dtype=np.complex128)
                for inner in range(row + 1, column + 1):
                    total += right_vectors[:, row, inner] * inverse[:, inner, column]
                inverse[:, row, column] = -total
    finite = np.all(np.isfinite(right_vectors), axis=(-2, -1)) & np.all(
        np.isfinite(inverse), axis=(-2, -1)
    )
    right_vectors[~finite] = identity[~finite]
    inverse[~finite] = identity[~finite]
    left = unitary @ right_vectors
    right = inverse @ np.conj(np.swapaxes(unitary, -1, -2))
    condition = np.linalg.norm(abs(left), ord=2, axis=(-2, -1)) * np.linalg.norm(
        abs(right), ord=2, axis=(-2, -1)
    )
    route = finite & (condition <= eigenbasis_limit(n_modes))
    left[~route] = 0.0
    right[~route] = 0.0
    return left, right, eigenvalues, route


def _flow_certificate(
    vectors: Complex[np.ndarray, "n_omega n_modes_c_out n_modes_c_in"],
    triangular: Complex[np.ndarray, "n_omega n_modes_t_out n_modes_t_in"],
    blocks: Complex[np.ndarray, "n_omega n_modes_b_out n_modes_b_in"],
    dz_max: float,
) -> tuple[Float[np.ndarray, " n_omega_e"], Float[np.ndarray, " n_omega_k"]]:
    """Return the representation error and the inverse-conditioning bound per frequency.

    Both rows are **declared** from measurements of the stored decomposition, because
    a small Schur reconstruction residual alone is insufficient.

    The representation error is the relative reconstruction residual
    ``norm(Z T Z^H - G)/norm(G)`` plus the unitary defect ``norm(Z^H Z - I)``,
    amplified by ``1 + nu/|lambda|_max`` with ``nu`` Henrici's departure from
    normality ``norm(T - diag(T))_F``. That factor is the branch conditioning of a
    matrix function of the triangular factor: a normal block carries none, and a
    strongly nonnormal one carries the amplification its function inherits. It is a
    measured diagnostic, not a proven forward-error bound on the exponential, which
    would need that function's own Frechet derivative.

    The conditioning bound is ``exp(dz_max (mu_max - mu_min))`` with ``mu`` the extreme
    eigenvalues of the Hermitian part of the generator -- its logarithmic norms -- so
    it bounds ``norm(U) norm(U^-1)`` of the flow over any step the controller may
    propose. Local re-anchoring is what makes it a per-step quantity: the
    interaction picture is re-anchored every step, so no inverse action is ever taken
    over more than one of them.

    Parameters
    ----------
    vectors, triangular : numpy.ndarray
        The Schur factors, shape ``(n_omega, n_modes, n_modes)``.
    blocks : numpy.ndarray
        The generator itself in the same layout, 1/m.
    dz_max : float
        Largest step the run may take, m, positive.

    Returns
    -------
    tuple of numpy.ndarray
        The dimensionless representation error and the conditioning bound, each of
        shape ``(n_omega,)``.
    """
    adjoint = np.conj(np.swapaxes(vectors, -1, -2))
    residual = np.linalg.norm(vectors @ triangular @ adjoint - blocks, axis=(-2, -1))
    scale = np.linalg.norm(blocks, axis=(-2, -1))
    identity = np.eye(blocks.shape[1], dtype=np.complex128)
    defect = np.linalg.norm(adjoint @ vectors - identity, axis=(-2, -1))
    eigenvalues = np.diagonal(triangular, axis1=-2, axis2=-1)
    strict = np.linalg.norm(triangular, axis=(-2, -1)) ** 2 - np.sum(
        abs(eigenvalues) ** 2, axis=-1
    )
    departure = np.sqrt(np.maximum(strict, 0.0))
    largest = np.max(abs(eigenvalues), axis=-1)
    amplification = 1.0 + np.divide(
        departure, largest, out=np.zeros_like(departure), where=largest > 0.0
    )
    relative = np.divide(
        residual, scale, out=np.zeros_like(residual), where=scale > 0.0
    )
    hermitian = 0.5 * (blocks + np.conj(np.swapaxes(blocks, -1, -2)))
    logarithmic = np.linalg.eigvalsh(hermitian)
    spread = logarithmic[:, -1] - logarithmic[:, 0]
    return (relative + defect) * amplification, np.exp(float(dz_max) * spread)


def constant_schur_flow(  # noqa: PLR0913 - the leaf's own fields plus its two limits
    omega: Float[np.ndarray, " n_omega"],
    z_edges: Float[np.ndarray, " n_edges"],
    generator: Complex[np.ndarray, "n_modes_flow_out n_modes_flow_in n_omega_flow"],
    *,
    dz_max: float,
    condition_limit: float,
    wall_vectors: Float[np.ndarray, " n_modes_pair"] | None = None,
    wall_weights: Complex[np.ndarray, " n_omega_pair"] | None = None,
    gauge_id: str,
    certificate_id: str,
) -> CoupledFlowTables:
    """Build the coupled flow of one constant section, with its certificate.

    The coupled flow at constant geometry: one section
    whose complete generator is frozen -- exactly, because it does not vary -- and
    integrated by a Schur matrix-function action, with the explicit residual
    identically zero. The dense generator is stored, and the rank-one wall pair beside
    it when the caller has one, so that a reader can see which part of the block is
    the wall without factoring it back out.

    The generator is in the **internal** FFT convention, like every other array of the
    contract, and carries only what the diagonal cumulative flow does not: the shared
    scalar carrier of the section stays in ``linear.phi``, where it is integrated
    analytically and commutes with the whole block.

    Parameters
    ----------
    omega : numpy.ndarray
        The canonical coarse angular frequencies in rad/s, increasing.
    z_edges : numpy.ndarray
        The section's two edges in m, covering the run.
    generator : numpy.ndarray
        The constant internal-convention generator in 1/m, complex, shape
        ``(n_modes, n_modes, n_omega)``, exactly zero on every bin the model was not
        evaluated at.
    dz_max : float
        Largest step the run may take, m, positive: what the conditioning bound is
        declared over.
    condition_limit : float
        Largest admitted conditioning, dimensionless and at least one.
    wall_vectors : numpy.ndarray or None, optional
        The rank-one wall vector ``h``, **dimensionless**, shape ``(n_modes,)``.
    wall_weights : numpy.ndarray or None, optional
        Its weight in **1/m**, complex, shape ``(n_omega,)``, supplied with the
        vector. The pair multiplies out to the generator's own 1/m: the vector
        carries no dimension and the weight carries the whole of it.
    gauge_id : str
        The basis's phase and coordinate identity.
    certificate_id : str
        Identity of the certificate the bounds came from.

    Returns
    -------
    CoupledFlowTables
        The validated leaf, ``section_methods == ("constant_schur",)``.

    Raises
    ------
    TranslationError
        If the generator varies in shape from the axes, if the wall pair does not
        reconstruct the generator's off-diagonal, or if the conditioning bound exceeds
        `condition_limit` at any frequency.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import constant_schur_flow
    >>> block = np.zeros((2, 2, 3), dtype=complex)
    >>> block[0, 0] = block[1, 1] = -1.0
    >>> flow = constant_schur_flow(
    ...     np.linspace(1e15, 2e15, 3), np.array([0.0, 1e-3]), block,
    ...     dz_max=1e-4, condition_limit=10.0, gauge_id="lab",
    ...     certificate_id="schur",
    ... )
    >>> flow.section_methods, float(flow.residual_norm_bound.max())
    (('constant_schur',), 0.0)
    """
    frequencies = len(np.asarray(omega))
    block = np.asarray(generator, dtype=np.complex128)
    if (
        block.ndim != 3
        or block.shape[0] != block.shape[1]
        or not block.shape[0]
        or block.shape[2] != frequencies
    ):
        msg = (
            "a constant coupled flow's generator must have shape (n_modes, n_modes, "
            f"n_omega) with {frequencies} frequencies, got {block.shape}"
        )
        raise TranslationError(msg)
    if float(dz_max) <= 0.0 or float(condition_limit) < 1.0:
        msg = (
            "a constant coupled flow needs a positive dz_max and a condition_limit of "
            f"at least one, got {dz_max!r} and {condition_limit!r}"
        )
        raise TranslationError(msg)
    pair = _checked_wall_pair(block, wall_vectors, wall_weights)
    blocks = np.ascontiguousarray(np.moveaxis(block, -1, 0))
    vectors, triangular = schur_blocks(block)
    error, conditioning = _flow_certificate(vectors, triangular, blocks, float(dz_max))
    worst = int(np.argmax(conditioning))
    if conditioning[worst] > float(condition_limit):
        msg = (
            f"a constant coupled flow's inverse conditioning reaches "
            f"{conditioning[worst]:.4g} over a step of {float(dz_max):.4g} m at "
            f"{float(np.asarray(omega)[worst]):.6g} rad/s, above the "
            f"{float(condition_limit):.4g} this flow admits: the interaction "
            "picture's inverse action would amplify the nonlinear source by that "
            "factor. Cap dz_max, retain fewer channels, or raise the limit "
            "deliberately"
        )
        raise TranslationError(msg)
    rows = (1, frequencies)
    return CoupledFlowTables(
        omega,
        z_edges,
        block[None, None],
        None,
        None,
        None if pair is None else pair[0],
        None if pair is None else pair[1],
        None,
        error.reshape(rows),
        np.zeros(rows),
        np.maximum(conditioning, 1.0).reshape(rows),
        ("constant_schur",),
        gauge_id,
        certificate_id,
    )


def _checked_wall_pair(
    block: Complex[np.ndarray, "n_modes_p_out n_modes_p_in n_omega_p"],
    wall_vectors: Float[np.ndarray, " n_modes_pv"] | None,
    wall_weights: Complex[np.ndarray, " n_omega_pw"] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Check a declared rank-one wall pair against the generator it came from.

    Defence in depth on the pair's units: a pair whose weight were in m
    rather than 1/m would reconstruct an off-diagonal wrong by the square of a length,
    which this comparison catches immediately. The **off-diagonal** is what is
    compared, because the dense generator's diagonal also carries the transverse
    term, which the pair does not describe.

    Parameters
    ----------
    block : numpy.ndarray
        The dense generator in 1/m, shape ``(n_modes, n_modes, n_omega)``.
    wall_vectors : numpy.ndarray or None
        The dimensionless rank-one vector, shape ``(n_modes,)``.
    wall_weights : numpy.ndarray or None
        Its weight in 1/m, shape ``(n_omega,)``.

    Returns
    -------
    tuple of numpy.ndarray or None
        The pair in the leaf's own layout -- ``(1, n_modes, n_omega)`` and
        ``(1, 1, n_omega)`` -- or ``None`` when none was supplied.

    Raises
    ------
    TranslationError
        If one half is missing, an extent disagrees, or the reconstruction does not.
    """
    supplied = (wall_vectors is not None, wall_weights is not None)
    if not any(supplied):
        return None
    if not all(supplied):
        msg = "a constant coupled flow's wall vector and weight are supplied together"
        raise TranslationError(msg)
    vector = np.asarray(wall_vectors, dtype=np.complex128)
    weight = np.asarray(wall_weights, dtype=np.complex128)
    if vector.shape != block.shape[:1] or weight.shape != block.shape[2:]:
        msg = (
            f"a constant coupled flow's wall pair must have shapes "
            f"{block.shape[:1]} and {block.shape[2:]}, got {vector.shape} and "
            f"{weight.shape}"
        )
        raise TranslationError(msg)
    rank_one = weight * np.outer(vector, vector)[:, :, None]
    modes = block.shape[0]
    off = ~np.eye(modes, dtype=bool)
    defect = float(np.max(abs(rank_one[off] - block[off]), initial=0.0))
    budget = 64 * modes * float(np.finfo(float).eps) * float(np.max(abs(block)))
    if defect > budget:
        msg = (
            "a constant coupled flow's wall pair does not reconstruct the "
            f"generator's off-diagonal: the largest departure is {defect:.4g} 1/m, "
            f"above the {budget:.4g} 1/m rounding of the block. The vector is "
            "dimensionless and the weight carries the whole 1/m"
        )
        raise TranslationError(msg)
    return (
        np.broadcast_to(vector[None, :, None], (1, *block.shape[::2])).copy(),
        weight[None, None],
    )


SECTION_TOL: float = 1.0e-12
"""Relative coefficient tail a section's z expansion is resolved to.

The same target :data:`solprop.translate.chebyshev.DEFAULT_TOL` sets for the cumulative
operator, and for the same reason: below it the interpolant is resolving the round-off
of its own samples. What is *not* resolved to it is refused rather than accepted with a
larger error, because the residual bound this expansion feeds is the whole certificate
of the frozen split.

The samples' own floor is well below it. The coherent wall's diagonal ``k - beta_11``
is formed from the two refractivities (``(n_core - 1) - (n_eff - 1)``), not as the
difference of two wavenumbers that agree to five or six digits, which carried the
absolute rounding of ``k`` (about 5e-9 1/m, up to 5e-10 of ``k - beta_11`` itself) and
now carries 1e-14 of it at worst. Forced to degree 64 across the transition of the
three- and six-channel qualification tapers and the TE/TM taper, the generator's tail
floors at 4e-14 relative (8e-15 to 4.0e-14), 25 times below this target, so a tail
above it is unresolved structure and never round-off.
"""

MAX_SECTION_DEGREE: int = 32
"""Highest Chebyshev degree along z tried on one section before it is refused.

A section of a declared taper or pressure stage carries a smooth rational function of
``a(z)`` and ``rho(z)``; a degree this high on one section means the section spans
structure the caller should have declared or should divide, which is what
``CoherentWall.sections`` is for. The search starts at
:data:`solprop.translate.chebyshev.INITIAL_DEGREE` and doubles.
"""


def section_expansion(
    sample: Callable[[Float[np.ndarray, " n_z"]], np.ndarray],
    z_edges: Float[np.ndarray, " n_edges"],
    *,
    tol: float = SECTION_TOL,
    max_degree: int = MAX_SECTION_DEGREE,
    name: str = "the section expansion",
) -> tuple[np.ndarray, float]:
    """Expand an array-valued function of z in Chebyshev coefficients per section.

    The fixed-section counterpart of :func:`solprop.translate.chebyshev.build`: the
    edges are **given** -- they are the flow's sections, which already land on every
    declared kink -- so nothing is split and nothing is guessed, and the only search is
    the degree. One degree is resolved for every section, because the leaf's ``n_deg``
    axis is rectangular, and it is the largest any section needed.

    The expansion is what makes the explicit residual a *bounded* quantity rather than a
    sampled one: ``|T_k| <= 1``, so ``sum_k ||c_k||`` encloses the function over the
    whole section including the extrema no sample sees
    (:func:`frozen_residual_flow`).

    Parameters
    ----------
    sample : callable
        Takes positions in m, shape ``(n_z,)``, and returns the values with the position
        axis **first**, shape ``(n_z, *value_shape)``.
    z_edges : numpy.ndarray
        The section edges in m, shape ``(n_intervals + 1,)``, strictly increasing.
    tol : float, optional
        Relative coefficient tail to resolve to. Default :data:`SECTION_TOL`.
    max_degree : float, optional
        Highest degree tried. Default :data:`MAX_SECTION_DEGREE`.
    name : str, optional
        What the refusal calls the quantity.

    Returns
    -------
    tuple
        The coefficients, shape ``(n_intervals, n_deg, *value_shape)``, and the worst
        relative tail actually reached, dimensionless.

    Raises
    ------
    TranslationError
        If a section's tail is still above `tol` at `max_degree`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import section_expansion
    >>> coefficients, tail = section_expansion(
    ...     lambda z: np.exp(-z)[:, None], np.array([0.0, 0.5, 1.0])
    ... )
    >>> coefficients.shape[0], bool(tail < 1e-12)
    (2, True)
    """
    edges = np.asarray(z_edges, dtype=float)
    degree = INITIAL_DEGREE
    while True:
        blocks = []
        worst = 0.0
        for index in range(edges.size - 1):
            lo, hi = float(edges[index]), float(edges[index + 1])
            nodes = lobatto_points(degree, lo, hi)
            # The ends are inset one ulp of the section's own scale, exactly as the
            # adaptive builder insets them: a declared profile written `z >= z_stage`
            # returns the post-kink value *at* the kink, so a section that ends there
            # would sample the wrong side of its own corner and read a jump as
            # unresolvable structure.
            inset = float(np.spacing(max(abs(lo), abs(hi), hi - lo)))
            nodes[0] = lo + inset
            nodes[-1] = hi - inset
            values = np.asarray(sample(nodes))
            coefficients = lobatto_coefficients(values)
            blocks.append(coefficients)
            worst = max(worst, _coefficient_tail(coefficients))
        if worst <= tol:
            return np.stack(blocks), worst
        if degree >= max_degree:
            msg = (
                f"{name} does not resolve on its sections: the relative Chebyshev "
                f"tail is {worst:.4g} at degree {degree}, above the {tol:.4g} target "
                "and far above the round-off floor of its samples. Divide the "
                "sections (CoherentWall.sections, on a coherent wall) or declare the "
                "structure as a kink of the profile"
            )
            raise TranslationError(msg)
        degree *= 2


def _coefficient_tail(coefficients: np.ndarray) -> float:
    """Return the relative size of the last two coefficients of one section.

    Parameters
    ----------
    coefficients : numpy.ndarray
        One section's coefficients, shape ``(n_deg, *value_shape)``.

    Returns
    -------
    float
        The last two coefficients' largest magnitude over the largest magnitude of the
        whole block, or zero for a block that is identically zero.
    """
    scale = float(np.max(abs(coefficients), initial=0.0))
    if scale == 0.0:
        return 0.0
    return float(np.max(abs(coefficients[-2:]), initial=0.0)) / scale


def frozen_generator(
    flow: CoupledFlowTables, index: int
) -> Complex[np.ndarray, "n_modes_frozen_out n_modes_frozen_in n_omega_frozen"]:
    """Return the generator one section of a coupled flow is frozen at.

    The frozen point is the section's **midpoint**, and it is read out of the stored
    expansion rather than carried
    beside it: the translation layer's certificate and the backend's Schur factors are
    then the same matrix to the last bit, because they come from the same coefficients
    through the same evaluation. At the midpoint ``x = 0``, so the weights are
    ``T_k(0)`` -- one, zero, minus one, zero, ... -- and a section of one degree returns
    its only coefficient unchanged, which keeps the constant-Schur flow bitwise the
    stored block.

    Parameters
    ----------
    flow : CoupledFlowTables
        The leaf.
    index : int
        Which section, in ``[0, n_intervals)``.

    Returns
    -------
    numpy.ndarray
        The frozen generator in 1/m, shape ``(n_modes, n_modes, n_omega)``.

    Raises
    ------
    TranslationError
        If the flow carries the factored representation, which has no dense generator
        to freeze.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import constant_schur_flow, frozen_generator
    >>> block = np.zeros((2, 2, 3), dtype=complex)
    >>> block[0, 0] = block[1, 1] = -1.0
    >>> flow = constant_schur_flow(
    ...     np.linspace(1e15, 2e15, 3), np.array([0.0, 1e-3]), block,
    ...     dz_max=1e-4, condition_limit=10.0, gauge_id="lab",
    ...     certificate_id="schur",
    ... )
    >>> bool(np.array_equal(frozen_generator(flow, 0), block))
    True
    """
    if flow.generator is None:
        msg = (
            "a factored coupled flow has no dense generator to freeze; the frozen "
            "section is read out of the dense representation"
        )
        raise TranslationError(msg)
    block = flow.generator[index]
    weights = chebyshev_basis.chebvander(0.0, block.shape[0] - 1)[0]
    return np.tensordot(weights, block, axes=1)


def flow_weights(
    flow: CoupledFlowTables, z: float, section: int | None = None
) -> Float[np.ndarray, " n_flow_terms"]:
    """Return the weights that combine a coupled flow's coefficients at one position.

    The stage-bundle half of the seam's evaluation form, for the coupled flow's own
    ``(n_intervals, n_deg)`` coefficient block: a backend evaluates
    ``G(z) = sum_j w_j terms_j`` over the **flattened** block, exactly as it evaluates
    ``dPhi`` over :attr:`solprop.translate.stage.StageTerms.phi`
    (``docs/dev/kernel-contract.md`` section 1). Every weight outside the section
    containing `z` is exactly zero, so no section's expansion is ever extrapolated, and
    the connection is combined over the same weights because it shares the axes.

    **Which section a position on an edge belongs to is the step's to say.** A section
    edge is a *kink*: the declared taper corner the sections were placed on, where
    ``a'`` jumps and the connection with it. A stage that lands exactly on the end of
    its own step's section must read the closing side of that corner, not the opening
    one, or one stage of that step evaluates a source discontinuously larger than the
    rest and the whole run drops to **first** order -- measured, before the tie was
    broken this way, as a global error of ``h`` times 1.0e2 per metre on the
    qualification taper. `section` is therefore the step's own, and it wins whenever
    the position lies inside its closed interval; a position genuinely outside it --
    which only a step spanning an edge produces, and the driver lands on every hard
    kink -- falls back to its own section rather than extrapolating a Chebyshev
    expansion past its end, which diverges as ``exp(2 n sqrt(eps))``. That fallback
    takes the section the stage **closes** in rather than the one that opens at it, for
    the same reason: a stage that ran past its step's section arrived from below, and
    the residual it is measured against is the one-sided limit on the side it came
    from.

    Parameters
    ----------
    flow : CoupledFlowTables
        The leaf.
    z : float
        Position in m.
    section : int or None, optional
        The step's own section, from
        :attr:`solprop.translate.stage.StageScalars.flow_section`. ``None`` takes the
        section containing `z`, which is what a caller with no step to speak of wants.

    Returns
    -------
    numpy.ndarray
        Float64, shape ``(n_intervals * n_deg,)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import constant_schur_flow, flow_weights
    >>> block = np.zeros((2, 2, 3), dtype=complex)
    >>> block[0, 0] = block[1, 1] = -1.0
    >>> flow = constant_schur_flow(
    ...     np.linspace(1e15, 2e15, 3), np.array([0.0, 1e-3]), block,
    ...     dz_max=1e-4, condition_limit=10.0, gauge_id="lab",
    ...     certificate_id="schur",
    ... )
    >>> flow_weights(flow, 5e-4).tolist()
    [1.0]
    """
    edges = flow.z_edges
    degrees = flow.n_deg
    slack = flow_edge_slack(flow)
    if section is None:
        index = flow_section(flow, z)
    elif edges[section] - slack <= z <= edges[section + 1] + slack:
        index = section
    else:
        # The stage ran past the step's own section, so the section it lands in is
        # the one that **closes** at or after it: a stage arriving at a kink from
        # below reads the side it came from, which is the one-sided limit the frozen
        # block it is measured against belongs to.
        raw = int(np.searchsorted(edges, float(z) - slack, side="right")) - 1
        index = min(max(raw, 0), edges.size - 2)
    lo = float(edges[index])
    hi = float(edges[index + 1])
    weights = np.zeros((edges.size - 1, degrees), dtype=np.float64)
    weights[index] = chebyshev_basis.chebvander(
        (2.0 * float(z) - lo - hi) / (hi - lo), degrees - 1
    )[0]
    return weights.reshape(-1)


def flow_section(flow: CoupledFlowTables, z: float) -> int:
    """Return the index of the section a position lies in.

    Parameters
    ----------
    flow : CoupledFlowTables
        The leaf.
    z : float
        Position in m.

    Returns
    -------
    int
        The section index, clamped to the ends: a position a rounding outside the span
        belongs to the nearest section, as a step landing on the last plane is. A
        position within ``1e-10`` of the span of an interior edge belongs to the section
        **starting** there, because a schedule that lands on a kink arrives a few ulp
        short of it as often as a few ulp past, and the two must not name different
        sections for the same step.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import constant_schur_flow, flow_section
    >>> block = np.zeros((2, 2, 3), dtype=complex)
    >>> block[0, 0] = block[1, 1] = -1.0
    >>> flow = constant_schur_flow(
    ...     np.linspace(1e15, 2e15, 3), np.array([0.0, 1e-3]), block,
    ...     dz_max=1e-4, condition_limit=10.0, gauge_id="lab",
    ...     certificate_id="schur",
    ... )
    >>> flow_section(flow, 2e-3)
    0
    """
    edges = flow.z_edges
    raw = (
        int(np.searchsorted(edges, float(z) + flow_edge_slack(flow), side="right")) - 1
    )
    return min(max(raw, 0), edges.size - 2)


def flow_edge_slack(flow: CoupledFlowTables) -> float:
    """Return the tolerance a position is matched to a section edge within.

    A step schedule lands on a declared kink by summing its steps, so it arrives
    within a few ulp of the total span rather than on it exactly;
    ``chebyshev._EDGE_SLACK_FRACTION`` of the span -- the same fraction
    :class:`solprop.translate.chebyshev.PiecewiseChebyshev` matches its own
    breakpoints to -- makes the section a stage is evaluated on depend on the
    schedule's intent rather than on its last bits.

    It is also what ``KernelProblem._check_schedule`` of
    :mod:`solprop.translate.problem` holds a **fixed** schedule to beside a coupled
    flow. That check's own window is
    wider (``1e-9`` of the run, which is 1e-9 m for anything under a metre), and a
    schedule edge between the two lands a step anchored in the closing section but
    running past the corner, against a frozen block that is not the one its stages
    read: fifth order collapses to first with nothing said. The two windows must
    therefore be one quantity, and a schedule that falls between them is refused
    rather than silently integrated.

    Parameters
    ----------
    flow : CoupledFlowTables
        The leaf.

    Returns
    -------
    float
        The tolerance in m, positive.
    """
    return _EDGE_SLACK_FRACTION * float(flow.z_edges[-1] - flow.z_edges[0])


def advisory_step(
    flow: CoupledFlowTables,
    scored: Bool[np.ndarray, " n_omega"] | None = None,
) -> Float[np.ndarray, "n_intervals n_omega"]:
    """Return the advisory step limit of every section and frequency, in m.

    ``h_advisory = 0.9 / (K ||U_s|| ||U_s^-1||)``, with ``K``
    the interval upper bound on the norm of the **whole** explicit linear residual --
    which is :attr:`CoupledFlowTables.residual_norm_bound` -- and the conditioning
    factor its interaction-picture similarity transform costs, which is
    :attr:`CoupledFlowTables.inverse_condition_bound`. Both rows are already across the
    seam, so this needs no row of its own.

    The factor 0.9 is a warning margin below the production tableau's imaginary
    stability radius 0.997189, **not an approved solver default and not a stability
    proof**: a wall difference can have positive real eigenvalues, representing local
    growth relative to the frozen attenuating frame, and no positive step makes that
    residual alone contractive. In a lossy or nonnormal case this is advisory only and
    a growth certificate is still required
    (:func:`solprop.translate.problem.validate_coupled_flow` says so in the warning).

    A section with an exactly zero residual -- every constant one -- returns infinity,
    because there is no explicit term to resolve. So does a bin outside `scored`: the
    explicit residual is applied only on the bins the step controller's error norm
    scores, and the spectral collar beyond them is carried by the frozen flow alone,
    which has no stability limit.

    Parameters
    ----------
    flow : CoupledFlowTables
        The leaf.
    scored : numpy.ndarray of bool, optional
        The bins the run applies the residual on, shape ``(n_omega,)``: the error
        norm's ``grids.inband_mask``. ``None`` takes every bin.

    Returns
    -------
    numpy.ndarray
        Float64, shape ``(n_intervals, n_omega)``, positive, ``inf`` where the residual
        is exactly zero.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import advisory_step, constant_schur_flow
    >>> block = np.zeros((2, 2, 3), dtype=complex)
    >>> block[0, 0] = block[1, 1] = -1.0
    >>> flow = constant_schur_flow(
    ...     np.linspace(1e15, 2e15, 3), np.array([0.0, 1e-3]), block,
    ...     dz_max=1e-4, condition_limit=10.0, gauge_id="lab",
    ...     certificate_id="schur",
    ... )
    >>> bool(np.all(np.isinf(advisory_step(flow))))
    True
    """
    scale = flow.residual_norm_bound * flow.inverse_condition_bound
    if scored is not None:
        scale = np.where(np.asarray(scored, dtype=bool)[None, :], scale, 0.0)
    return np.divide(
        _STABILITY_MARGIN,
        scale,
        out=np.full(scale.shape, np.inf),
        where=scale > 0.0,
    )


_STABILITY_MARGIN: float = 0.9
"""Warning margin below the DOPRI5 imaginary stability radius 0.9971890086.

A **reported scale**, not a solver default:
nothing here changes a caller's tolerances or schedule on it.
"""


def residual_truncation(x: float) -> float:
    """Return the frozen-residual truncation defect of one step, dimensionless.

    ``||R(h R_s) - exp(h R_s)|| <= x^6/3600 + exp(x) x^7/5040`` with ``x = h K``, valid
    in any matrix norm and without an eigenvector assumption. The
    first term is ``|1/600 - 1/720| x^6``, the difference between the production DOPRI5
    tableau's own sixth-order coefficient ``1/600`` and the exponential's ``1/720``; the
    second bounds the exponential tail from the seventh order.

    It is a **local frozen** defect, not a bound for a time-varying stage sequence, and
    it is charged to the linear share of the error budget, with the Schur action and
    the explicit-linear integration.

    Parameters
    ----------
    x : float
        ``h K``, dimensionless and non-negative.

    Returns
    -------
    float
        The bound, dimensionless; ``inf`` where the exponential tail overflows a
        double, which a step so far above the advisory one reaches (a declared
        ``dz_max`` of half a capillary, for instance) and which the advisory warning
        then reports rather than failing on.

    Raises
    ------
    TranslationError
        If `x` is negative or not finite.

    Examples
    --------
    >>> from solprop.translate.coupled import residual_truncation
    >>> f"{residual_truncation(0.1):.4e}"
    '2.9971e-10'
    >>> residual_truncation(0.0)
    0.0
    >>> residual_truncation(1e3)
    inf
    """
    value = float(x)
    if not math.isfinite(value) or value < 0.0:
        msg = f"the truncation bound takes a finite non-negative h K, got {x!r}"
        raise TranslationError(msg)
    if value > _TRUNCATION_OVERFLOW:
        return math.inf
    return value**6 / 3600.0 + math.exp(value) * value**7 / 5040.0


_TRUNCATION_OVERFLOW: float = 600.0
"""Largest ``h K`` whose truncation bound is formed: ``exp(600) 600^7`` is 1.7e280."""


def frozen_residual_flow(  # noqa: PLR0913 - the leaf's own arrays plus its two limits
    omega: Float[np.ndarray, " n_omega"],
    z_edges: Float[np.ndarray, " n_edges"],
    generator: Complex[
        np.ndarray,
        "n_intervals_fr n_deg_fr n_modes_fr_out n_modes_fr_in n_omega_fr",
    ],
    connection: Complex[
        np.ndarray,
        "n_intervals_cn n_deg_cn n_modes_cn_out n_modes_cn_in n_omega_cn",
    ]
    | None,
    *,
    dz_max: float,
    condition_limit: float,
    representation_error: float,
    gauge_id: str,
    certificate_id: str,
) -> CoupledFlowTables:
    """Build the certified coupled flow of a varying guide, frozen section by section.

    The coupled flow at varying geometry. Each section carries the **complete**
    generator as a Chebyshev expansion along z; the block frozen inside the
    exponential is that expansion at the section midpoint (:func:`frozen_generator`),
    and everything else --

    ``R_s(z) = D(z) - D(z_s) + H(z) - H(z_s) - C(z)``

    -- is the explicit residual the interaction-picture Runge-Kutta source carries.
    **With ``R`` evaluated exactly, freezing is a rearrangement of the original
    equation, not a second-order approximation**; section refinement reduces the
    residual's stiffness and the inverse amplification and is not permission to discard
    it.

    The residual row is a true enclosure over each whole section rather than a sampled
    maximum: with ``R_s = sum_k r_k T_k`` and ``|T_k| <= 1``, ``sum_k ||r_k||_2`` bounds
    it at every position, extrema no sample saw included. The interpolant's own error is
    the `representation_error` the caller measured, which joins the Schur diagnostic in
    ``generator_error``.

    Parameters
    ----------
    omega : numpy.ndarray
        The canonical coarse angular frequencies in rad/s, increasing.
    z_edges : numpy.ndarray
        The section edges in m, shape ``(n_intervals + 1,)``, covering the run.
    generator : numpy.ndarray
        Chebyshev coefficients along z of the complete internal-convention generator in
        1/m, shape ``(n_intervals, n_deg, n_modes, n_modes, n_omega)``, exactly zero on
        every bin the model was not evaluated at.
    connection : numpy.ndarray or None
        The taper connection's coefficients in the same layout and units, or ``None``
        for a guide whose radius does not move.
    dz_max : float
        Largest step the run may take, m, positive: what the conditioning bound is
        declared over.
    condition_limit : float
        Largest admitted conditioning, dimensionless and at least one.
    representation_error : float
        The relative Chebyshev tail of the expansions, dimensionless and non-negative,
        as :func:`section_expansion` measured it.
    gauge_id : str
        The basis's phase and coordinate identity.
    certificate_id : str
        Identity of the certificate the bounds came from.

    Returns
    -------
    CoupledFlowTables
        The validated leaf, ``section_methods == ("frozen_residual",) * n_intervals``.

    Raises
    ------
    TranslationError
        If the arrays disagree with the axes, if `representation_error` is negative, or
        if the conditioning bound exceeds `condition_limit` at any frequency.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import frozen_residual_flow
    >>> block = np.zeros((2, 1, 2, 2, 3), dtype=complex)
    >>> block[:, 0, 0, 0] = block[:, 0, 1, 1] = -1.0
    >>> flow = frozen_residual_flow(
    ...     np.linspace(1e15, 2e15, 3), np.array([0.0, 5e-4, 1e-3]), block, None,
    ...     dz_max=1e-4, condition_limit=10.0, representation_error=0.0,
    ...     gauge_id="lab", certificate_id="frozen-residual",
    ... )
    >>> flow.section_methods, float(flow.residual_norm_bound.max())
    (('frozen_residual', 'frozen_residual'), 0.0)
    """
    frequencies = len(np.asarray(omega))
    edges = np.asarray(z_edges, dtype=float)
    intervals = interval_count(edges.size)
    coefficients = np.asarray(generator, dtype=np.complex128)
    if (
        coefficients.ndim != 5
        or coefficients.shape[0] != intervals
        or coefficients.shape[2] != coefficients.shape[3]
        or not coefficients.shape[2]
        or coefficients.shape[4] != frequencies
    ):
        msg = (
            "a frozen-residual flow's generator must have shape (n_intervals, n_deg, "
            f"n_modes, n_modes, n_omega) with {intervals} section(s) and "
            f"{frequencies} frequencies, got {coefficients.shape}"
        )
        raise TranslationError(msg)
    link = None if connection is None else np.asarray(connection, dtype=np.complex128)
    if link is not None and link.shape != coefficients.shape:
        msg = (
            "a frozen-residual flow's connection must have the generator's own shape "
            f"{coefficients.shape}, got {link.shape}"
        )
        raise TranslationError(msg)
    if float(dz_max) <= 0.0 or float(condition_limit) < 1.0:
        msg = (
            "a frozen-residual flow needs a positive dz_max and a condition_limit of "
            f"at least one, got {dz_max!r} and {condition_limit!r}"
        )
        raise TranslationError(msg)
    if not (float(representation_error) >= 0.0):
        msg = (
            "a frozen-residual flow's representation_error is the non-negative "
            f"relative tail of its expansions, got {representation_error!r}"
        )
        raise TranslationError(msg)
    error = np.empty((intervals, frequencies))
    conditioning = np.empty((intervals, frequencies))
    residual = np.empty((intervals, frequencies))
    midpoint = chebyshev_basis.chebvander(0.0, coefficients.shape[1] - 1)[0]
    for index in range(intervals):
        block = np.tensordot(midpoint, coefficients[index], axes=1)
        blocks = np.ascontiguousarray(np.moveaxis(block, -1, 0))
        vectors, triangular = schur_blocks(block)
        row, bound = _flow_certificate(vectors, triangular, blocks, float(dz_max))
        error[index] = row + float(representation_error)
        conditioning[index] = np.maximum(bound, 1.0)
        residual[index] = _residual_enclosure(coefficients[index], block, link, index)
    worst = int(np.argmax(np.max(conditioning, axis=0)))
    if conditioning[:, worst].max() > float(condition_limit):
        msg = (
            f"a frozen-residual flow's inverse conditioning reaches "
            f"{float(conditioning[:, worst].max()):.4g} over a step of "
            f"{float(dz_max):.4g} m at {float(np.asarray(omega)[worst]):.6g} rad/s, "
            f"above the {float(condition_limit):.4g} this flow admits: the "
            "interaction picture's inverse action would amplify the nonlinear source "
            "by that factor. Cap dz_max, divide the sections, retain fewer channels, "
            "or raise the limit deliberately"
        )
        raise TranslationError(msg)
    return CoupledFlowTables(
        omega,
        edges,
        coefficients,
        None,
        None,
        None,
        None,
        link,
        error,
        residual,
        conditioning,
        ("frozen_residual",) * intervals,
        gauge_id,
        certificate_id,
    )


def _residual_enclosure(
    coefficients: Complex[np.ndarray, "n_deg_r n_modes_r_out n_modes_r_in n_omega_r"],
    frozen: Complex[np.ndarray, "n_modes_z_out n_modes_z_in n_omega_z"],
    connection: Complex[np.ndarray, " n_intervals_c n_deg_c *rest"] | None,
    index: int,
) -> Float[np.ndarray, " n_omega_e"]:
    """Return the spectral-norm enclosure of one section's explicit residual.

    ``R_s(z) = sum_k r_k T_k(x)`` with ``r_0 = c_0 - G_s - l_0`` and ``r_k = c_k - l_k``
    above, so ``|T_k| <= 1`` gives ``||R_s(z)||_2 <= sum_k ||r_k||_2`` at every position
    of the section. That is an enclosure of the stored interpolant, extrema between the
    samples included, which a sampled maximum is not.

    Parameters
    ----------
    coefficients : numpy.ndarray
        One section's generator coefficients, shape
        ``(n_deg, n_modes, n_modes, n_omega)``.
    frozen : numpy.ndarray
        The generator frozen at that section's midpoint, 1/m.
    connection : numpy.ndarray or None
        The whole connection block, or ``None``.
    index : int
        Which section, used to select the connection's own coefficients.

    Returns
    -------
    numpy.ndarray
        The bound in 1/m per frequency, shape ``(n_omega,)``.
    """
    terms = coefficients.copy()
    terms[0] -= frozen
    if connection is not None:
        terms -= connection[index]
    moved = np.moveaxis(terms, (1, 2), (-2, -1))
    return np.sum(np.linalg.norm(moved, ord=2, axis=(-2, -1)), axis=0)


CF4_NODES: tuple[float, float] = (
    0.5 - math.sqrt(3.0) / 6.0,
    0.5 + math.sqrt(3.0) / 6.0,
)
"""The two Gauss-Legendre nodes of a CF4 step, as fractions of its length."""

CF4_WEIGHTS: tuple[float, float] = (
    (3.0 - 2.0 * math.sqrt(3.0)) / 12.0,
    (3.0 + 2.0 * math.sqrt(3.0)) / 12.0,
)
"""``(a, b)`` of the two-exponential CF4: ``a + b = 1/2`` and ``b - a = 1/sqrt(3)``.

Matching the integral of the generator fixes the sum, and matching the cubic Magnus
term ``-h^3 [A_0, A_1]/12`` against the product's BCH commutator
``-h^3 (b - a)[A_0, A_1]/(4 sqrt(3))`` fixes the difference. ``a = -0.0386751`` is
**negative**, and is forced by the two-exponential ansatz: it is why a factor of a
dissipative flow need not be contractive, and why :func:`cf4_flow` guards the growth
of its internal factors.
"""

_CF4_GROWTH_EXPONENT: float = (2.0 * math.sqrt(3.0) - 3.0) / 6.0
"""``2|a| = 0.0773503``: the excess growth rate of the two forward factors' product.

Each factor's exponent is ``h(a A_early + b A_late)`` or ``h(b A_early + a A_late)``
with ``a < 0 < b``. The logarithmic norm is subadditive and ``mu(c A) = c mu_min(A)``
for ``c < 0``, so with every Hermitian-part eigenvalue of the generator in
``[m_lo, m_hi]`` each factor has ``mu <= b m_hi - |a| m_lo``, and the two together
``m_hi + 2|a|(m_hi - m_lo)``, because ``2b = 1 + 2|a|``. The exact action is bounded by
``exp(h m_hi)``; the factors' product can exceed that by at most
``exp(2|a| h (m_hi - m_lo))``.
"""

_CF4_CONDITION_EXPONENT: float = 2.0 / math.sqrt(3.0)
"""``1 + 4|a| = 2/sqrt(3)``: the rate of the factor-norm product of a CF4 pair.

The inverse factors have ``mu <= -b m_lo + |a| m_hi`` each, so the four factor norms of
one forward-and-inverse pair multiply to at most ``exp((1 + 4|a|) h (m_hi - m_lo))``,
against ``exp(h (m_hi - m_lo))`` for the exact flow.
"""

CF4_GROWTH_LIMIT: float = 2.0
"""Default limit on a CF4 section's internal growth bound, dimensionless.

A section whose bound ``exp(2|a| dz_max (m_hi - m_lo))`` exceeds it is refused by name
(or falls back to ``"frozen_residual"`` where that is declared). Two is at most one bit
of the state's scale lost to the product of the internal factors, and corresponds to
``dz_max (m_hi - m_lo) <= 8.96`` -- about the nine e-foldings of the Hermitian spread
that the default ``CoherentWall.condition_limit`` of 1e4 admits. It is a **proposal**
reasoned from that bound, not a threshold established by experiment.
"""


def _hermitian_enclosure(
    coefficients: Complex[
        np.ndarray, "n_intervals_h n_deg_h n_modes_h_out n_modes_h_in n_omega_h"
    ],
    connection: Complex[
        np.ndarray, "n_intervals_k n_deg_k n_modes_k_out n_modes_k_in n_omega_k"
    ]
    | None,
) -> tuple[Float[np.ndarray, "n_intervals_l n_omega_l"], Float[np.ndarray, "n_i n_w"]]:
    """Return an enclosure of the Hermitian-part spectrum of each section's generator.

    The generator a CF4 step exponentiates is ``L(z) = G(z) - C(z)``, the whole of it.
    With ``L = sum_k l_k T_k(x)`` on a section, the Hermitian part is
    ``Herm(l_0) + sum_{k>=1} Herm(l_k) T_k(x)`` and ``|T_k| <= 1``, so by Weyl's
    inequality every eigenvalue of ``Herm L(z)``, at every position of the section, lies
    within ``sum_{k>=1} ||Herm l_k||_2`` of the spectrum of ``Herm l_0``. That is an
    enclosure of the stored interpolant, not a sampled range. The connection is real
    and antisymmetric in the true-power amplitude, so in the kernel's it has a small
    symmetric part, of the order of the channels' weight spread; it is subtracted with
    the rest, because the enclosure is of what is exponentiated.

    Parameters
    ----------
    coefficients : numpy.ndarray
        The generator's Chebyshev coefficients in 1/m.
    connection : numpy.ndarray or None
        The connection's, in the same layout, or ``None``.

    Returns
    -------
    tuple of numpy.ndarray
        The lower and upper ends ``m_lo`` and ``m_hi`` in 1/m, each of shape
        ``(n_intervals, n_omega)``.
    """
    terms = coefficients if connection is None else coefficients - connection
    moved = np.moveaxis(terms, (2, 3), (-2, -1))
    hermitian = 0.5 * (moved + np.conj(np.swapaxes(moved, -1, -2)))
    centre = np.linalg.eigvalsh(hermitian[:, 0])
    radius = np.zeros(centre.shape[:2])
    if hermitian.shape[1] > 1:
        radius = np.sum(np.linalg.norm(hermitian[:, 1:], ord=2, axis=(-2, -1)), axis=1)
    return centre[..., 0] - radius, centre[..., -1] + radius


def _reach_spread(
    edges: Float[np.ndarray, " n_edges_r"],
    lower: Float[np.ndarray, "n_intervals_r n_omega_r"],
    upper: Float[np.ndarray, "n_intervals_r2 n_omega_r2"],
    dz_max: float,
) -> Float[np.ndarray, "n_intervals_s n_omega_s"]:
    """Return the Hermitian spread every step anchored in a section can meet.

    A step anchored in section ``s`` starts inside it and is at most ``dz_max`` long, so
    its Gauss nodes lie in ``[lo_s, hi_s + dz_max]``, which may reach into the sections
    after it: its enclosure is the union of theirs. The spread is taken over that
    union, so that a certificate declared on a section holds for every step the
    controller may anchor there.

    Parameters
    ----------
    edges : numpy.ndarray
        The section edges in m.
    lower, upper : numpy.ndarray
        Each section's enclosure, 1/m, shape ``(n_intervals, n_omega)``.
    dz_max : float
        Largest step, m.

    Returns
    -------
    numpy.ndarray
        ``m_hi - m_lo`` over each section's reach, 1/m, non-negative.
    """
    intervals = lower.shape[0]
    spread = np.empty_like(lower)
    for index in range(intervals):
        reach = int(np.searchsorted(edges, edges[index + 1] + dz_max, side="left"))
        last = min(max(reach, index + 1), intervals)
        spread[index] = np.max(upper[index:last], axis=0) - np.min(
            lower[index:last], axis=0
        )
    return spread


def cf4_flow(  # noqa: PLR0913 - the leaf's own arrays plus its three limits
    omega: Float[np.ndarray, " n_omega"],
    z_edges: Float[np.ndarray, " n_edges"],
    generator: Complex[
        np.ndarray,
        "n_intervals_c4 n_deg_c4 n_modes_c4_out n_modes_c4_in n_omega_c4",
    ],
    connection: Complex[
        np.ndarray,
        "n_intervals_cc n_deg_cc n_modes_cc_out n_modes_cc_in n_omega_cc",
    ]
    | None,
    *,
    dz_max: float,
    condition_limit: float,
    growth_limit: float = CF4_GROWTH_LIMIT,
    fallback: Literal["frozen_residual"] | None = None,
    representation_error: float,
    gauge_id: str,
    certificate_id: str,
) -> CoupledFlowTables:
    """Build the CF4 coupled flow of a varying guide, with its two guards.

    The certified alternative to :func:`frozen_residual_flow`:
    nothing is frozen and nothing is left to an explicit residual. A step of length
    ``h`` from ``z`` applies the fourth-order commutator-free exponential of the whole
    section generator ``L = G - C``,

    ``U(h) = exp(h (a A_1 + b A_2)) exp(h (b A_1 + a A_2))``,

    with ``A_j = L(z + c_j h)`` at the Gauss nodes ``c = 1/2 -+ sqrt(3)/6`` and
    ``a, b = (3 -+ 2 sqrt(3))/12``; the early-weighted factor acts first. Its local
    error is ``O(h^5)`` on a smooth section, so a run with it is at most **fourth**
    order globally, and the embedded Runge-Kutta pair does not see that error at all
    (:func:`cf4_defect`). Every section takes the same Chebyshev expansion
    :func:`frozen_residual_flow` does; what differs is the certificate.

    **The internal-growth guard**, before admission. ``a`` is negative, so a factor of
    a dissipative flow can grow: a block with eigenvalues 0 and -200/m over 1 m
    has internal factors of norm 1898 against an exact action of 0.96. With every
    Hermitian-part eigenvalue of ``L`` over the reach of a step enclosed in
    ``[m_lo, m_hi]`` (Weyl's inequality on the stored expansion), the logarithmic norm
    bounds the product of the two forward factors by ``exp(h m_hi)`` -- the largest the
    exact action can be -- times the **growth bound**
    ``exp((2 sqrt(3) - 3)/6 dz_max (m_hi - m_lo))``, and the four factors of a forward
    and inverse pair by ``exp((2/sqrt(3)) dz_max (m_hi - m_lo))``, which is the
    ``inverse_condition_bound`` a CF4 section declares and the backend checks per step
    against the factors it actually forms. A section whose growth bound exceeds
    `growth_limit`, or whose conditioning exceeds `condition_limit`, is **refused by
    name** -- or, only when `fallback` says so, takes ``"frozen_residual"`` instead,
    with a warning naming it.

    The rows of a CF4 section: ``generator_error`` is the expansion's relative tail
    alone, there being no Schur action to diagnose; ``residual_norm_bound`` is zero,
    there being no explicit residual; ``inverse_condition_bound`` is the pair bound
    above. A fallback section carries exactly the rows :func:`frozen_residual_flow`
    gives it.

    Parameters
    ----------
    omega : numpy.ndarray
        The canonical coarse angular frequencies in rad/s, increasing.
    z_edges : numpy.ndarray
        The section edges in m, shape ``(n_intervals + 1,)``, covering the run.
    generator : numpy.ndarray
        Chebyshev coefficients along z of the complete internal-convention generator in
        1/m, shape ``(n_intervals, n_deg, n_modes, n_modes, n_omega)``.
    connection : numpy.ndarray or None
        The taper connection's coefficients in the same layout, or ``None``.
    dz_max : float
        Largest step the run may take, m, positive.
    condition_limit : float
        Largest admitted conditioning of a step's flow action, at least one.
    growth_limit : float, optional
        Largest admitted growth bound of a CF4 section, at least one. Default
        :data:`CF4_GROWTH_LIMIT`.
    fallback : {None, 'frozen_residual'}, optional
        What a section the guards reject becomes. ``None``, the default, refuses it.
    representation_error : float
        The relative Chebyshev tail of the expansions, non-negative.
    gauge_id : str
        The basis's phase and coordinate identity.
    certificate_id : str
        Identity of the certificate the bounds came from.

    Returns
    -------
    CoupledFlowTables
        The validated leaf, ``"cf4"`` on every admitted section.

    Raises
    ------
    TranslationError
        If a limit or the fallback is out of range, on everything
        :func:`frozen_residual_flow` refuses, if a section fails a guard with no
        fallback declared, or if a fallback section fails the frozen conditioning.

    Warns
    -----
    solprop.exceptions.SolpropWarning
        Naming every section that fell back, the guard that rejected it and its bound.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import cf4_flow
    >>> block = np.zeros((2, 1, 2, 2, 3), dtype=complex)
    >>> block[:, 0, 0, 0] = block[:, 0, 1, 1] = -1.0
    >>> flow = cf4_flow(
    ...     np.linspace(1e15, 2e15, 3), np.array([0.0, 5e-4, 1e-3]), block, None,
    ...     dz_max=1e-4, condition_limit=10.0, representation_error=0.0,
    ...     gauge_id="lab", certificate_id="cf4",
    ... )
    >>> flow.section_methods, float(flow.inverse_condition_bound.max())
    (('cf4', 'cf4'), 1.0)
    """
    if not (math.isfinite(float(growth_limit)) and float(growth_limit) >= 1.0):
        msg = (
            "a CF4 flow's growth_limit bounds the product of its internal factors "
            f"against the exact action's and is at least one, got {growth_limit!r}"
        )
        raise TranslationError(msg)
    if fallback not in (None, "frozen_residual"):
        msg = (
            "a CF4 flow falls back to 'frozen_residual' where that is declared and "
            f"refuses otherwise; got fallback={fallback!r}"
        )
        raise TranslationError(msg)
    # The frozen split supplies the validation of every array and the rows a fallback
    # section keeps. Its own conditioning limit is lifted here and applied below to
    # the sections that actually fall back, because a CF4 section's conditioning is
    # its own pair bound and not the frozen block's.
    base = frozen_residual_flow(
        omega,
        z_edges,
        generator,
        connection,
        dz_max=dz_max,
        condition_limit=math.inf,
        representation_error=representation_error,
        gauge_id=gauge_id,
        certificate_id=certificate_id,
    )
    assert base.generator is not None
    lower, upper = _hermitian_enclosure(base.generator, base.connection)
    spread = _reach_spread(base.z_edges, lower, upper, float(dz_max))
    with np.errstate(over="ignore"):
        conditioning = np.exp(_CF4_CONDITION_EXPONENT * float(dz_max) * spread)
        growth = np.exp(_CF4_GROWTH_EXPONENT * float(dz_max) * spread)
    rejected = [
        index
        for index in range(spread.shape[0])
        if float(np.max(growth[index])) > float(growth_limit)
        or float(np.max(conditioning[index])) > float(condition_limit)
    ]
    reasons = [
        _cf4_rejection(
            index,
            base.omega,
            growth[index],
            conditioning[index],
            growth_limit=float(growth_limit),
            condition_limit=float(condition_limit),
            dz_max=float(dz_max),
        )
        for index in rejected
    ]
    if rejected and fallback is None:
        msg = (
            f"a CF4 coupled flow's guards reject {len(rejected)} of its "
            f"{spread.shape[0]} section(s): {'; '.join(reasons)}. Cap dz_max, divide "
            "the sections, raise the limit deliberately, or declare "
            "fallback='frozen_residual' to integrate those sections with the frozen "
            "split instead"
        )
        raise TranslationError(msg)
    for index in rejected:
        frozen = float(np.max(base.inverse_condition_bound[index]))
        if frozen > float(condition_limit):
            msg = (
                f"a CF4 coupled flow's section {index} is rejected by its guards "
                f"({reasons[rejected.index(index)]}) and cannot fall back either: its "
                f"frozen block's conditioning is {frozen:.4g} over dz_max = "
                f"{float(dz_max):.4g} m, above the {float(condition_limit):.4g} "
                "admitted. Cap dz_max"
            )
            raise TranslationError(msg)
    if rejected:
        warnings.warn(
            f"a CF4 coupled flow's guards reject {len(rejected)} of its "
            f"{spread.shape[0]} section(s), which fall back to 'frozen_residual' as "
            f"declared: {'; '.join(reasons)}",
            SolpropWarning,
            stacklevel=2,
        )
    error = np.array(base.generator_error)
    residual = np.array(base.residual_norm_bound)
    bound = np.array(base.inverse_condition_bound)
    methods: list[SectionMethod] = []
    for index in range(spread.shape[0]):
        if index in rejected:
            methods.append("frozen_residual")
            continue
        methods.append("cf4")
        error[index] = float(representation_error)
        residual[index] = 0.0
        bound[index] = np.maximum(conditioning[index], 1.0)
    return CoupledFlowTables(
        base.omega,
        base.z_edges,
        base.generator,
        None,
        None,
        None,
        None,
        base.connection,
        error,
        residual,
        bound,
        tuple(methods),
        gauge_id,
        certificate_id,
    )


def _cf4_rejection(  # noqa: PLR0913 - one section's two bounds and their two limits
    index: int,
    omega: Float[np.ndarray, " n_omega"],
    growth: Float[np.ndarray, " n_omega_g"],
    conditioning: Float[np.ndarray, " n_omega_c"],
    *,
    growth_limit: float,
    condition_limit: float,
    dz_max: float,
) -> str:
    """Return the clause naming why a CF4 section was rejected.

    Parameters
    ----------
    index : int
        The section.
    omega : numpy.ndarray
        The frequency axis in rad/s.
    growth, conditioning : numpy.ndarray
        The section's two bounds per frequency, dimensionless.
    growth_limit, condition_limit : float
        The limits they were held to.
    dz_max : float
        The step they were declared over, m.

    Returns
    -------
    str
        One clause naming both bounds at the frequency where they are largest -- the
        same frequency for both, each being increasing in the same spread.
    """
    worst = int(np.argmax(growth))
    return (
        f"section {index} has internal growth bound {float(growth[worst]):.4g} "
        f"(limit {growth_limit:.4g}) and pair conditioning "
        f"{float(conditioning[worst]):.4g} (limit {condition_limit:.4g}) over dz_max "
        f"= {dz_max:.4g} m at {float(omega[worst]):.6g} rad/s"
    )


def cf4_growth_bound(
    flow: CoupledFlowTables,
) -> Float[np.ndarray, "n_intervals n_omega"]:
    """Return the internal growth bound of every section, dimensionless.

    ``exp((2 sqrt(3) - 3)/6 dz_max (m_hi - m_lo))`` for a ``"cf4"`` section, read back
    from the conditioning row it declares, ``exp((2/sqrt(3)) dz_max (m_hi - m_lo))``:
    the growth bound is that row to the power ``(2 - sqrt(3))/4``, so it needs no row
    of its own (:func:`cf4_flow`). Every other section returns one, having no internal
    factors to grow.

    Parameters
    ----------
    flow : CoupledFlowTables
        The leaf.

    Returns
    -------
    numpy.ndarray
        Float64, shape ``(n_intervals, n_omega)``, at least one.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import cf4_flow, cf4_growth_bound
    >>> block = np.zeros((1, 1, 2, 2, 1), dtype=complex)
    >>> block[0, 0, 1, 1] = -100.0
    >>> flow = cf4_flow(
    ...     np.array([1e15]), np.array([0.0, 1e-3]), block, None, dz_max=1e-3,
    ...     condition_limit=10.0, representation_error=0.0, gauge_id="lab",
    ...     certificate_id="cf4",
    ... )
    >>> f"{float(cf4_growth_bound(flow)[0, 0]):.6f}"
    '1.007765'
    """
    rows = np.ones_like(flow.inverse_condition_bound)
    power = _CF4_GROWTH_EXPONENT / _CF4_CONDITION_EXPONENT
    for index, method in enumerate(flow.section_methods):
        if method == "cf4":
            rows[index] = flow.inverse_condition_bound[index] ** power
    return rows


def _flow_generator_at(
    flow: CoupledFlowTables,
    terms: Complex[np.ndarray, "n_flow_terms n_modes_g_out n_modes_g_in n_omega_g"],
    z: float,
    section: int,
) -> Complex[np.ndarray, "n_omega_o n_modes_o_out n_modes_o_in"]:
    """Return ``L(z) = G(z) - C(z)`` of a step anchored in `section`, frequency first.

    Parameters
    ----------
    flow : CoupledFlowTables
        The leaf, whose edges and degree the weights are built on.
    terms : numpy.ndarray
        The flattened coefficients of ``G - C`` in 1/m.
    z : float
        Position in m.
    section : int
        The step's section, which :func:`flow_weights` resolves the position against.

    Returns
    -------
    numpy.ndarray
        The generator in 1/m, shape ``(n_omega, n_modes, n_modes)``.
    """
    combined = np.tensordot(flow_weights(flow, z, section), terms, axes=1)
    return np.ascontiguousarray(np.moveaxis(combined, -1, 0))


def _cf4_action(
    flow: CoupledFlowTables,
    terms: Complex[np.ndarray, "n_flow_terms n_modes_a_out n_modes_a_in n_omega_a"],
    z: float,
    length: float,
    section: int,
) -> Complex[np.ndarray, "n_omega_x n_modes_x_out n_modes_x_in"]:
    """Return the CF4 forward action over ``[z, z + length]``, by SciPy's exponential.

    The translation layer's own evaluation, independent of the backend's scaled Taylor
    series: the estimator and the executor are two implementations of one formula.

    Parameters
    ----------
    flow : CoupledFlowTables
        The leaf.
    terms : numpy.ndarray
        The flattened coefficients of ``G - C`` in 1/m.
    z : float
        Where the step starts, m.
    length : float
        Its length, m, positive.
    section : int
        The step's section.

    Returns
    -------
    numpy.ndarray
        The action, shape ``(n_omega, n_modes, n_modes)``.
    """
    early = _flow_generator_at(flow, terms, z + CF4_NODES[0] * length, section)
    late = _flow_generator_at(flow, terms, z + CF4_NODES[1] * length, section)
    small, large = CF4_WEIGHTS
    first = expm(length * (large * early + small * late))
    second = expm(length * (small * early + large * late))
    return second @ first


def cf4_defect(
    flow: CoupledFlowTables,
    dz: float,
    kinks: Sequence[float] | Float[np.ndarray, " n_kinks"] = (),
) -> Float[np.ndarray, "n_intervals n_omega"]:
    """Return the step-doubling estimate of every CF4 section's local flow defect.

    **The source-feedback estimator, and what kind of statement it is.** In the
    interaction picture ``u = U b`` the stage source is ``g = U^-1 N(U b)``, and a CF4
    run uses an approximation ``V`` of every stage's ``U``. With ``||U - V|| <= delta``,
    the source is perturbed by at most ``||V^-1|| delta (||U^-1|| ||N|| + L ||b||)`` --
    ``L`` the source's Lipschitz constant on a ball containing both states -- and the
    step's end is moved by ``delta ||b||`` directly. The direct term is ``O(h^5)`` per
    step; the feedback term enters the state through one more factor ``h`` and is
    ``O(h^6)``. The
    run is therefore **at most fourth order** in the flow's own error, and the embedded
    Runge-Kutta pair, which sees only the difference of two stage combinations, does
    not see it at all. No Lipschitz constant of the material response is available on
    a containing state ball -- Kerr, Raman and plasma each have their own, and none is
    enclosed by this package -- so the rigorous envelope is **not** what is declared.
    What is declared is the **asymptotic estimator**: two half steps have one sixteenth
    of the leading error of a full one, so ``(16/15) ||V(h) - V(h/2)^2||`` estimates the
    full step's local defect, qualified by observable refinement (measured against
    an independent DOP853 solution of the flow alone, and by the run's own order). It
    is neither a bound nor valid near a kink, at the rounding floor, or outside the
    asymptotic regime -- which nothing but this estimate polices: the internal-growth
    guard bounds growth from the Hermitian spread alone, so a
    lossless section admits any step, and far outside the regime the estimate
    **underestimates** (0.19 against a true 0.38 at ``Δβ h = 100``). It is what
    :func:`solprop.translate.problem.validate_coupled_flow` compares with the
    controller's tolerance.

    Each CF4 section is sampled with a step of length `dz` aligned with its start, its
    middle and its end, and the largest relative defect is kept, because the steepest
    part of a taper is where the defect is. A sampled step ends at the first of
    `kinks` after its start, and at the end of the run, because a step the driver
    takes does: adaptive stepping lands on every hard kink and a fixed schedule that
    straddles one is refused, and a CF4 step across a corner of the
    generator is not fourth order at all. Every other section returns zero.

    Parameters
    ----------
    flow : CoupledFlowTables
        The leaf, dense.
    dz : float
        The step to estimate over, m, positive.
    kinks : sequence of float, optional
        The run's hard kinks in m (``linear.hard_kinks``), which no step crosses.

    Returns
    -------
    numpy.ndarray
        The relative defect ``||V(h) - U(h)||_2 / ||U(h)||_2``, estimated, shape
        ``(n_intervals, n_omega)``.

    Raises
    ------
    TranslationError
        If the flow is factored or `dz` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.coupled import cf4_defect, cf4_flow
    >>> block = np.zeros((1, 1, 2, 2, 1), dtype=complex)
    >>> block[0, 0, 1, 1] = -100.0
    >>> flow = cf4_flow(
    ...     np.array([1e15]), np.array([0.0, 1e-3]), block, None, dz_max=1e-3,
    ...     condition_limit=10.0, representation_error=0.0, gauge_id="lab",
    ...     certificate_id="cf4",
    ... )
    >>> bool(cf4_defect(flow, 1e-3).max() < 1e-15)
    True
    """
    if flow.generator is None:
        msg = "a factored coupled flow has no dense generator to take a CF4 step of"
        raise TranslationError(msg)
    if not (float(dz) > 0.0):
        msg = f"a CF4 defect is estimated over a positive step, got {dz!r}"
        raise TranslationError(msg)
    edges = flow.z_edges
    step = float(dz)
    stops = np.unique(np.append(np.asarray(kinks, dtype=float), float(edges[-1])))
    slack = flow_edge_slack(flow)
    terms = (
        flow.generator if flow.connection is None else flow.generator - flow.connection
    )
    terms = terms.reshape(-1, *terms.shape[2:])
    defect = np.zeros(flow.inverse_condition_bound.shape)
    for index, method in enumerate(flow.section_methods):
        if method != "cf4":
            continue
        lo, hi = float(edges[index]), float(edges[index + 1])
        spare = max(hi - lo - step, 0.0)
        for anchor in sorted({lo, lo + 0.5 * spare, lo + spare}):
            following = stops[stops > anchor + slack]
            length = min(step, float(following[0]) - anchor)
            coarse = _cf4_action(flow, terms, anchor, length, index)
            fine = _cf4_action(
                flow, terms, anchor + 0.5 * length, 0.5 * length, index
            ) @ _cf4_action(flow, terms, anchor, 0.5 * length, index)
            relative = (
                16.0
                / 15.0
                * np.linalg.norm(coarse - fine, ord=2, axis=(-2, -1))
                / np.linalg.norm(fine, ord=2, axis=(-2, -1))
            )
            defect[index] = np.maximum(defect[index], relative)
    return defect


def wall_identity(  # noqa: PLR0913 - the flow, its limits and the CF4 declaration
    flow: CoupledFlowTables,
    limits: WallLimits,
    *,
    channels: Sequence[ChannelLabel],
    impedance: WallImpedance = "mean",
    growth_limit: float | None = None,
    fallback: str | None = None,
    dz_max: float | None = None,
    kinks: Sequence[float] | Float[np.ndarray, " n_kinks"] = (),
    scored: Bool[np.ndarray, " n_omega"] | None = None,
    n_components: int = 1,
    has_noise: bool = False,
) -> Mapping[str, JsonValue]:
    """Return the schema-7 modal identity of a capillary carrying a coherent wall.

    A **new key entry, not a new schema**: ``modal.flow`` joins the identity the
    general modal route writes, as ``modal.guide.sampling.basis`` does, without moving
    output schema 7. A file written without a ``flow`` entry keeps the key it was
    stamped with.

    The identity is what a resume compares, so **no array enters it** and every
    measured float is quantised (:func:`solprop.modes.material.quantised_measurement`)
    against the last bits of the BLAS that produced it. What is recorded of the block
    is what makes it the block it is: the section methods and how many of them there
    are, the degree of their expansions along z, whether a taper connection is carried,
    the two identity strings, the three validity ratios, the band they were taken over,
    the three certificate rows reduced to their extremes and the advisory step they
    imply (:func:`advisory_step`). The radius and the ratios are the **worst** section's
    (:func:`solprop.modes.capillary.worst_wall_limits`), which for a constant guide is
    the only one there is, taken over the error norm's scored band as the run's
    admission takes them.

    ``modal.basis`` is ``"capillary HE_1m"``, which is how
    :func:`solprop.output.capability.capability_of` tells this identity from a general
    guide's: the coherent wall does not change the basis, which is the same orthogonal
    ``HE_1m`` set it always was, only the flow through it.

    Parameters
    ----------
    flow : CoupledFlowTables
        The leaf the run carries.
    limits : solprop.modes.capillary.WallLimits
        The validity ratios the block was admitted under.
    channels : sequence of solprop.modes.protocol.ChannelLabel
        The channel labels in state order.
    impedance : str, optional
        Which wall the run declared. A ``"te_tm"`` block writes the model
        ``"te_tm impedance coherent wall"`` and the basis ``"capillary azimuthal
        order"`` rather than ``"capillary HE_1m"``, because its channels
        are two radial families of one azimuthal order and not one ``HE_1m`` set; it
        is still an orthogonal capillary mode set, so
        :func:`solprop.output.capability.capability_of` still reads it as the ordinary
        modal basis and the ordinary accessors stay admitted. A resume across a
        changed wall impedance is refused by this field.
    growth_limit : float or None, optional
        The CF4 growth limit the flow was admitted under; required, with `dz_max`,
        exactly when a section is ``"cf4"``.
    fallback : str or None, optional
        The declared fallback of a CF4 flow, recorded as declared.
    dz_max : float or None, optional
        The largest step, m, over which the CF4 defect is estimated.
    kinks : sequence of float, optional
        The run's hard kinks in m, at which a sampled CF4 step ends
        (:func:`cf4_defect`).
    scored : numpy.ndarray of bool, optional
        The error norm's bins, over which the recorded advisory step is taken, as the
        run's warning takes it (:func:`advisory_step`); ``None`` takes every bin.
    n_components : int, optional
        The transverse components the synthesis resolves, 1 or 2: two for a supplied
        ``TE_0m``, ``TM_0m`` or ``EH`` set, one for the
        capillary's own ``HE_1m`` set.
    has_noise : bool, optional
        Whether the run carries an input-noise draw, which is made per channel in
        the orthonormal true-power basis and recorded as such.

    Returns
    -------
    collections.abc.Mapping
        The ``modal`` record of :class:`solprop.output.schema.RunMetadata`. A flow with
        a ``"cf4"`` section adds a ``flow.cf4`` entry beside ``advisory_step`` -- a
        new key entry, as ``flow`` itself is, and absent from every other flow,
        whose record is therefore unchanged: the largest internal growth bound
        (:func:`cf4_growth_bound`), the limit it was held to, the declared fallback,
        and the largest step-doubling defect estimate at `dz_max`
        (:func:`cf4_defect`), named as the asymptotic estimator it is.

    Raises
    ------
    TranslationError
        If a CF4 flow is recorded without its growth limit or its step.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode, wall_operator
    >>> from solprop.translate.coupled import constant_schur_flow, wall_identity
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, m=index) for index in (1, 2)]
    >>> omega = np.array([2.3e15, 2.4e15])
    >>> block = wall_operator(modes, omega)
    >>> flow = constant_schur_flow(
    ...     omega, np.array([0.0, 1e-3]), np.conj(block.generator), dz_max=1e-5,
    ...     condition_limit=100.0, gauge_id="capillary HE_1m",
    ...     certificate_id="constant-schur",
    ... )
    >>> record = wall_identity(
    ...     flow, block.limits, channels=[mode.channel_label for mode in modes]
    ... )
    >>> record["basis"], record["flow"]["section_methods"]
    ('capillary HE_1m', ['constant_schur'])
    """
    record: dict[str, JsonValue] = {
        "model": f"{impedance} impedance coherent wall",
        "section_methods": list(flow.section_methods),
        "n_sections": len(flow.section_methods),
        "n_deg": flow.n_deg,
        "connection": flow.connection is not None,
        "gauge_id": flow.gauge_id,
        "certificate_id": flow.certificate_id,
        "radius": float(limits.radius),
        "transverse_ratio": quantised_measurement(limits.transverse_ratio),
        "vacuum_ratio": quantised_measurement(limits.vacuum_ratio),
        "quasi_static": quantised_measurement(limits.quasi_static),
        "generator_error": quantised_measurement(float(np.max(flow.generator_error))),
        "residual_norm_bound": quantised_measurement(
            float(np.max(flow.residual_norm_bound))
        ),
        "inverse_condition_bound": quantised_measurement(
            float(np.max(flow.inverse_condition_bound))
        ),
        "advisory_step": quantised_measurement(
            float(np.min(advisory_step(flow, scored)))
        ),
    }
    if "cf4" in flow.section_methods:
        if growth_limit is None or dz_max is None:
            msg = (
                "a CF4 coupled flow's record carries the growth limit it was admitted "
                "under and the step its defect is estimated over; pass growth_limit "
                "and dz_max"
            )
            raise TranslationError(msg)
        record["cf4"] = {
            "growth_bound": quantised_measurement(
                float(np.max(cf4_growth_bound(flow)))
            ),
            "growth_limit": float(growth_limit),
            "fallback": fallback,
            "defect": quantised_measurement(
                float(np.max(cf4_defect(flow, dz_max, kinks)))
            ),
            "defect_estimator": "step doubling, asymptotic",
        }
    return {
        "basis": (
            "capillary HE_1m" if impedance == "mean" else "capillary azimuthal order"
        ),
        # The transverse components the synthesis resolves, as the general route
        # records them: two for a TE_0m, TM_0m or EH set.
        "n_components": int(n_components),
        "n_channels": int(flow.n_modes),
        "channels": [dict(label.as_mapping()) for label in channels],
        "tables": None,
        "metric": None,
        "responses": None,
        "guide": None,
        "frame": "shared co-moving",
        "power_weight": "per channel",
        "active_band": [
            quantised_measurement(limits.band[0]),
            quantised_measurement(limits.band[1]),
        ],
        "noise_reference_basis": "orthonormal true power" if has_noise else None,
        "accessors": [],
        "flow": record,
    }


def taper_identity(
    flow: CoupledFlowTables,
    *,
    channels: Sequence[ChannelLabel],
    active: Bool[np.ndarray, " n_omega"],
    scored: Bool[np.ndarray, " n_omega"] | None = None,
    n_components: int = 1,
    has_noise: bool = False,
) -> Mapping[str, JsonValue]:
    """Return the schema-7 modal identity of a capillary taper's coupling.

    A multimode capillary taper without a wall carries the taper coupling in a
    coupled flow, and a file must say so: a resume against the adiabatic route of
    the same geometry is a different equation. The record is :func:`wall_identity`'s
    with the wall's own entries -- its validity ratios, its radius and its impedance
    -- absent, because no wall block was built; ``flow.model`` is
    ``"taper coupling"`` and the certificate is
    :data:`solprop.translate.problem.TAPER_FLOW_CERTIFICATE`. A new key entry and not
    a new schema, as for :func:`wall_identity`. The basis is still the orthogonal
    ``"capillary HE_1m"`` set, so the ordinary accessors stay admitted.

    Parameters
    ----------
    flow : CoupledFlowTables
        The leaf the run carries.
    channels : sequence of solprop.modes.protocol.ChannelLabel
        The channel labels in state order.
    active : numpy.ndarray of bool
        The run's active band, whose ends are recorded.
    scored : numpy.ndarray of bool, optional
        The error norm's bins, over which the recorded advisory step is taken
        (:func:`advisory_step`); ``None`` takes every bin.
    n_components : int, optional
        The transverse components the synthesis resolves.
    has_noise : bool, optional
        Whether the run carries an input-noise draw, recorded as
        :func:`wall_identity` records it.

    Returns
    -------
    collections.abc.Mapping
        The ``modal`` record of :class:`solprop.output.schema.RunMetadata`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.protocol import ChannelLabel
    >>> from solprop.translate.coupled import frozen_residual_flow, taper_identity
    >>> omega = np.array([2.3e15, 2.4e15])
    >>> block = np.zeros((1, 1, 2, 2, 2), dtype=complex)
    >>> block[0, 0, 1, 1] = -1j * 1e3
    >>> flow = frozen_residual_flow(
    ...     omega, np.array([0.0, 1e-3]), block, None, dz_max=1e-5,
    ...     condition_limit=100.0, representation_error=0.0,
    ...     gauge_id="capillary HE_1m", certificate_id="taper-frozen-residual",
    ... )
    >>> labels = [ChannelLabel("HE", 1, m, None, None, f"HE1{m}") for m in (1, 2)]
    >>> record = taper_identity(flow, channels=labels, active=np.ones(2, bool))
    >>> record["flow"]["model"], record["basis"]
    ('taper coupling', 'capillary HE_1m')
    """
    band = flow.omega[np.asarray(active, dtype=bool)]
    record: dict[str, JsonValue] = {
        "model": "taper coupling",
        "section_methods": list(flow.section_methods),
        "n_sections": len(flow.section_methods),
        "n_deg": flow.n_deg,
        "connection": flow.connection is not None,
        "gauge_id": flow.gauge_id,
        "certificate_id": flow.certificate_id,
        "generator_error": quantised_measurement(float(np.max(flow.generator_error))),
        "residual_norm_bound": quantised_measurement(
            float(np.max(flow.residual_norm_bound))
        ),
        "inverse_condition_bound": quantised_measurement(
            float(np.max(flow.inverse_condition_bound))
        ),
        "advisory_step": quantised_measurement(
            float(np.min(advisory_step(flow, scored)))
        ),
    }
    return {
        "basis": "capillary HE_1m",
        "n_components": int(n_components),
        "n_channels": int(flow.n_modes),
        "channels": [dict(label.as_mapping()) for label in channels],
        "tables": None,
        "metric": None,
        "responses": None,
        "guide": None,
        "frame": "shared co-moving",
        "power_weight": "per channel",
        "active_band": [
            quantised_measurement(float(band[0])),
            quantised_measurement(float(band[-1])),
        ],
        "noise_reference_basis": "orthonormal true power" if has_noise else None,
        "accessors": [],
        "flow": record,
    }
