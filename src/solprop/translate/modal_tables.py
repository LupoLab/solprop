# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Track and interpolate dense Maxwell tables without changing the kernel seam."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from jaxtyping import Bool, Complex, Float, Inexact, Int32
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment

from solprop._display import Displayed, engineering, line
from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.modes.guide import BASIS_UNITARY_ATOL, ChannelBasis
from solprop.modes.material import JsonValue, quantised_measurement
from solprop.modes.tabulated import IndexTable, _SampledIndexTable
from solprop.modes.vector import ModalNodeRule, ModalProvider, ModalSample, flux_product
from solprop.translate.coupled import interval_count

__all__ = [
    "DEFAULT_GRAM_ATOL",
    "MAX_GENERATOR_COUPLING",
    "MIN_OVERLAP_DOMINANCE",
    "MIN_Z_KNOTS",
    "DenseModalTable",
    "ModalTransformTables",
    "ProjectedLaunch",
    "align_modal_sample",
    "apply_channel_basis",
    "channel_area",
    "composed_gauge",
    "generator_is_coupled",
    "modal_transform_tables",
    "project_transverse_field",
    "sample_modal_table",
    "sample_provider_table",
    "solved_in_table_basis",
    "transformed_sample",
    "validate_modal_table",
]

DEFAULT_GRAM_ATOL = 1e-4
"""Largest sampled Gram defect ``norm(W - I, 2)`` a table admits at a knot.

A provider normalises its fields over the whole physical domain and deliberately
does not renormalise a caller's rule, so a rule that truncates the fields, or
resolves them badly, returns a basis whose *sampled* Gram is not the identity --
and every sample is still ``valid``, because the provider refused nothing. The
table is where that has to be caught, before an interpolated basis carries the
defect into a propagation.

The value is the loosest defect a provider in this package can legitimately
produce. The step-index rule certifies the cladding power outside it below 1e-12
of the total, so its defect is quadrature error alone: measured 1.33e-15, 4.68e-15
and 4.76e-15 over six mixed HE/EH/TE/TM channels at radial orders 24, 48 and 96.
The radial provider admits up to
:data:`solprop.modes.graded_index.MAX_EXTERIOR_TAIL_FRACTION` = 1e-4 of a
channel's power beyond its matching radius, and the sampled Gram is short by
exactly the power the rule discards, so that guard is the binding one and this
budget is the same number.

It is a construction budget, not a physical statement: a caller who deliberately
integrates part of a cross-section passes its own, and a taper whose interface
leaves the frozen panel edge needs the held-out ``flux_atol`` of
:func:`validate_modal_table` instead, which measures the interpolated basis.
"""

logger = logging.getLogger(__name__)

MIN_OVERLAP_DOMINANCE = 0.5
"""Smallest aligned physical overlap a tracked channel may keep.

The aligned diagonal of the cross-Poynting matrix is a dimensionless overlap
whose correct value is one, so a dominance criterion is the right test: below a
half the assignment is no longer identifying the same physical channel, whatever
permutation it chose. A bound of ``1e-8`` would be eight orders looser than
the quantity it guards and would accept an almost arbitrary mis-track.

Measured on the 250 nm strand's degenerate HE11 pair, the weakest aligned
overlap against the reference sample is 1.0, 0.99999948, 0.99985 and 0.99872 at
frequency ratios 1.0, 1.01, 1.2 and 2.0 -- so even a doubling of the frequency,
far beyond any table knot spacing, stays three orders above this floor.
"""

MAX_GENERATOR_COUPLING = 1e-9
"""Largest off-diagonal generator a per-channel index table admits, relatively.

A channel's propagation constant is the diagonal of the provider's generator, so
a generator that couples its channels has no such constant: its flow is a matrix
exponential, which is the ``coupled_flow`` leaf and is built by the general modal
route, not by this table. Both providers here are diagonal by construction and a tracked
alignment is a permutation and a phase, so the measured off-diagonal norm is
exactly zero; the budget is there to refuse a provider that is not, not to admit
a small coupling.

**What it is measured against.** ``norm(G - diag G, 'fro')`` is compared with the
**smallest splitting between the channels**, ``min |beta_i - beta_j|`` over
``i != j``, and not with the size of ``beta`` itself. Two channels mix over a
length set by the ratio of their coupling to their splitting, so that ratio is the
physical quantity; comparing with ``|beta| ~ 1e7 1/m`` would have admitted an
off-diagonal of 1e-3 1/m between two channels split by 1 1/m, which mixes them
completely in a few metres. The splitting is floored at ``64 eps |beta|_max``, the
rounding of the propagation constants themselves, so that an exactly degenerate
pair still has a defined scale.

**What this does not detect.** A coupling below 1e-9 of the smallest splitting,
which turns the pair by less than 1e-8 rad over a beat length and is what the
budget is for. A coupling that appears *between* the knots: only the sampled
generators are examined. Anything about the z direction: a table is a snapshot.
And a physically coupled basis that a provider has already diagonalised -- this
refuses a generator that is not diagonal, not a guide whose channels exchange
power by some other mechanism.
"""

MIN_Z_KNOTS = 4
"""Fewest longitudinal knots a table interpolates over, beside a single snapshot.

SciPy's not-a-knot cubic silently degrades to a parabola on three points and to
a straight line on two, so a table built on two or three positions would report
an interpolation order it does not have. One position is a snapshot and is
interpolated in frequency alone.
"""


def _overlap(
    left: ModalSample, right: ModalSample, nodes: ModalNodeRule
) -> Complex[np.ndarray, "n_modes n_modes"]:
    """Form the physical cross-Poynting matrix on common nodes.

    Parameters
    ----------
    left : ModalSample
        Reference sample on common physical nodes.
    right : ModalSample
        Candidate sample on the same physical nodes.
    nodes : ModalNodeRule
        The common physical nodes and their area weights in m².

    Returns
    -------
    numpy.ndarray
        Cross overlaps, shape (n_modes, n_modes).
    """
    e, h, f, g = left.electric, left.magnetic, right.electric, right.magnetic
    return (
        np.einsum("qm,qn,q->mn", h[:, 1].conj(), f[:, 0], nodes.weights)
        - np.einsum("qm,qn,q->mn", h[:, 0].conj(), f[:, 1], nodes.weights)
        + np.einsum("qm,qn,q->mn", e[:, 0].conj(), g[:, 1], nodes.weights)
        - np.einsum("qm,qn,q->mn", e[:, 1].conj(), g[:, 0], nodes.weights)
    ) / 4


def align_modal_sample(
    reference: ModalSample,
    sample: ModalSample,
    nodes: ModalNodeRule,
    *,
    degenerate_blocks: Sequence[Sequence[int]] = (),
) -> ModalSample:
    """Align channels by physical overlap, phase and declared subspaces.

    Parameters
    ----------
    reference : ModalSample
        Previous sample on the same physical Cartesian nodes.
    sample : ModalSample
        New full-vector sample with the same retained channel count.
    nodes : ModalNodeRule
        Common physical nodes and area weights in m².
    degenerate_blocks : sequence of sequence of int, optional
        Disjoint channel-index blocks permitted to rotate. The complete generator
        is rotated too, so a near-degenerate block remains physically equivalent.

    Returns
    -------
    ModalSample
        Aligned E/H, generator, flux and reciprocal data in reference coordinates.

    Raises
    ------
    ModeError
        If the common domain, validity, channel count or overlap is unresolved.
    """
    if (
        reference.electric.shape != sample.electric.shape
        or len(nodes.xy) != sample.electric.shape[0]
    ):
        raise ModeError(
            "tracking requires identical channel counts on common physical nodes"
        )
    if reference.physical_id != sample.physical_id:
        raise ModeError("tracking requires the same physical provider identity")
    if not np.all(reference.valid) or not np.all(sample.valid):
        raise ModeError("tracking refuses unsupported occupied channels")
    overlap = _overlap(reference, sample, nodes)
    _, order = linear_sum_assignment(-abs(overlap))
    transform = np.eye(len(order), dtype=complex)[:, order]
    diagonal = np.diag(overlap[:, order])
    weakest = float(np.min(abs(diagonal)))
    if weakest <= MIN_OVERLAP_DOMINANCE:
        raise ModeError(
            "physical overlap is unresolved: the weakest assigned channel keeps "
            f"only {weakest:.4g} of its unit reference overlap, at or below "
            f"{MIN_OVERLAP_DOMINANCE}; refine the tracking mesh or the knot spacing"
        )
    transform = transform * np.exp(-1j * np.angle(diagonal))
    used: set[int] = set()
    for block in degenerate_blocks:
        indices = np.asarray(block, dtype=int)
        if (
            len(indices) < 2
            or np.any(indices < 0)
            or np.any(indices >= len(order))
            or len(set(block)) != len(block)
            or used.intersection(block)
        ):
            raise ModeError("degenerate blocks must be disjoint valid channel sets")
        used.update(block)
        subspace = (overlap @ transform)[np.ix_(indices, indices)]
        left, singular, right = svd(subspace)
        if singular[-1] < 1e-8 * singular[0]:
            raise ModeError("degenerate physical overlap is rank deficient")
        rotation = np.eye(len(order), dtype=complex)
        rotation[np.ix_(indices, indices)] = right.conj().T @ left.conj().T
        transform = transform @ rotation
    return transformed_sample(sample, transform, gauge_id=reference.gauge_id)


def transformed_sample(
    sample: ModalSample,
    transform: Complex[np.ndarray, "n_modes_out n_modes_in"],
    *,
    gauge_id: str,
) -> ModalSample:
    """Put a sample in new channel coordinates: ``e' = e U``, with every partner.

    The one place a channel transform is applied to a
    :class:`~solprop.modes.vector.ModalSample`, shared
    by the tracking of :func:`align_modal_sample` -- whose transform is a
    permutation, a phase and a degenerate-block rotation -- and by the declared
    change of basis of :class:`solprop.modes.guide.ChannelBasis`. The rules follow
    from ``e' = e U`` alone: the Hermitian objects conjugate as ``U.H X U`` and the
    **bilinear** reciprocal partners, which pair without a conjugate, carry
    ``U.conj()``.

    The boundary residual of a mixed channel is reported as the block's largest.
    A combination ``sum_n c_n e_n`` of unit-norm coefficients has a residual bounded
    by ``sum_n |c_n|`` times the largest, which for a unitary column is at most
    ``sqrt(n_modes)`` times it, so this is a diagnostic and not a certificate; it is
    the rule the tracking has always used and one number for the set is what the
    validation of :class:`~solprop.modes.vector.ModalSample` compares.

    Parameters
    ----------
    sample : solprop.modes.vector.ModalSample
        The sample to re-express.
    transform : numpy.ndarray
        The matrix ``U``, shape (n_modes, n_modes), whose columns are the new basis
        vectors written in the old one.
    gauge_id : str
        Non-empty phase and coordinate identity of the result.

    Returns
    -------
    solprop.modes.vector.ModalSample
        The same physics in the new channel coordinates.
    """
    count = sample.electric.shape[2]
    return replace(
        sample,
        electric=sample.electric @ transform,
        magnetic=sample.magnetic @ transform,
        generator=transform.conj().T @ sample.generator @ transform,
        flux_metric=transform.conj().T @ sample.flux_metric @ transform,
        boundary_residual=np.full(count, float(np.max(sample.boundary_residual))),
        backward_electric=None
        if sample.backward_electric is None
        else sample.backward_electric @ transform.conj(),
        backward_magnetic=None
        if sample.backward_magnetic is None
        else sample.backward_magnetic @ transform.conj(),
        reciprocal_gram=None
        if sample.reciprocal_gram is None
        else transform.conj().T @ sample.reciprocal_gram @ transform,
        gauge_id=gauge_id,
    )


def composed_gauge(gauge_id: str, basis: ChannelBasis | None) -> str:
    """Return the gauge a declared basis leaves on top of a tracking gauge.

    **The order is fixed and it is documented here.** A knot is first put in the
    tracking gauge -- the permutation, the phase and the degenerate-block SVD
    alignment :func:`align_modal_sample` applies -- and the declared basis is
    applied **after** it, so the composed transform is ``T_track U`` and the
    composed identity is ``"<tracking gauge>+<basis gauge>"``. That order is what a
    resume has to reproduce, and it is the one that leaves a run with no declared
    basis bitwise what it was: the tracking is untouched and the declaration is a
    second, independent step on top of it.

    Parameters
    ----------
    gauge_id : str
        The tracking gauge's own identity, the provider's.
    basis : solprop.modes.guide.ChannelBasis or None
        The declared change of basis, or ``None``.

    Returns
    -------
    str
        The composed identity, unchanged when nothing is declared.

    Examples
    --------
    >>> from solprop.translate.modal_tables import composed_gauge
    >>> composed_gauge("step-index-exact", None)
    'step-index-exact'
    """
    return gauge_id if basis is None else f"{gauge_id}+{basis.gauge_id}"


def apply_channel_basis(sample: ModalSample, basis: ChannelBasis) -> ModalSample:
    """Put one tracked sample in a declared channel basis, splitting the two cases.

    The transformed generator is ``U.H diag(beta) U``, which stays diagonal when the
    block's propagation constants are equal and becomes a **coupled matrix flow**
    when they are not. Both are admitted, and they are not the same object:

    * an **exactly degenerate** block leaves the generator diagonal, and the
      generator is carried over rather than multiplied out (below);
    * a **split** block leaves it coupled, and the product is evaluated. That block
      is ``linear.coupled_flow``, executed by ``numpy_ref`` and ``numba_cpu`` and
      built by the general modal route from the transformed generator
      (:func:`solprop.translate.general.coupled_block`). Any refusal is the
      *route's*, where a flow can be built, and this function's job is only to say
      which of the two the declaration produced.

    **The equality is judged at the rounding of the propagation constants**, the same
    floor ``_channel_splitting`` gives a degenerate pair: ``64 eps max|beta|``.
    The two HE11 parities of a step-index strand satisfy it exactly -- measured 0.0
    relative -- and a pair that is degenerate only to a
    part in 1e-9 is a physically split pair whose partners beat, which is the case
    this refuses. The generator must also already **be** diagonal to that same
    rounding, which is what the two conditions together make ``G' = G`` an identity
    rather than a product: ``U`` is the identity outside the blocks and each block
    is ``g_b`` times the identity inside it, so it commutes with ``U``.

    **Where every block is degenerate the generator is carried over exactly, not
    multiplied out.** Evaluating ``U.H G U`` numerically injects rounding into
    quantities that are exactly zero: a lossless provider has ``Re G = 0`` to the
    bit, and the product put a spurious ``Re G = +1.1e-26`` 1/m there -- gain, which
    ``_channel_index_table`` refuses outright, and an ``Im n_eff`` of the wrong
    sign at the 1e-33 level. Taking the exact value is the identity the two gates
    above have just certified -- **to their own floor**, which is
    ``64 eps max|beta|``, 1.55e-7 1/m on a step-index strand and 1.4e-14 of the
    propagation constants themselves. Below that floor the carried-over generator
    and the evaluated product differ, and this keeps the exact one, so a degenerate
    declaration is bitwise independent of the split path.

    **Where a block is split the product is evaluated**, because there is then no
    identity to carry over: the off-diagonal is the physics, not the rounding. The
    diagonal that comes out of it is checked for gain exactly as the carried-over
    one is, so a conjugation error is still caught.

    **The invariance gate is certified to the declaration's own budget.** The
    transform must not change ``norm(W - I, 2)``, which a unitary one cannot; the
    budget is the rounding of the product or
    :data:`~solprop.modes.guide.BASIS_UNITARY_ATOL`, whichever is larger, because a
    transform that is unitary only to that budget moves the measured norm by its own
    unitary defect and the declaration has already admitted it at exactly that size.

    Parameters
    ----------
    sample : solprop.modes.vector.ModalSample
        A tracked sample in the provider's own channel coordinates.
    basis : solprop.modes.guide.ChannelBasis
        The declared change of basis.

    Returns
    -------
    solprop.modes.vector.ModalSample
        The sample in the declared coordinates, with the composed gauge identity.

    Raises
    ------
    ModeError
        If the transform's width disagrees with the sample, the generator is not
        already diagonal, the evaluated product of a split block shows gain, or the
        transform changes the basis's measured departure from orthonormal, which a
        unitary one cannot.
    """
    diagonal = np.diag(sample.generator)
    if len(diagonal) != len(basis.transform):
        raise ModeError(
            f"a declared channel basis describes {len(basis.transform)} channel(s) "
            f"and this sample carries {len(diagonal)}"
        )
    floor = 64 * float(np.finfo(float).eps) * float(np.max(abs(diagonal)))
    coupling = float(np.linalg.norm(sample.generator - np.diag(diagonal)))
    if coupling > floor:
        raise ModeError(
            f"a declared change of basis needs a generator that is already "
            f"diagonal; this one couples its channels by {coupling:.6g} 1/m at "
            f"{sample.omega:.6e} rad/s, above the {floor:.6g} 1/m rounding of its "
            "own propagation constants, so it has no per-channel constant for the "
            "degeneracy gate below to compare"
        )
    split = any(
        float(
            np.max(abs(diagonal[list(block)][:, None] - diagonal[list(block)][None, :]))
        )
        > floor
        for block in basis.blocks
    )
    turned = transformed_sample(
        sample, basis.transform, gauge_id=composed_gauge(sample.gauge_id, basis)
    )
    if split:
        _check_transformed_passivity(sample, turned, basis)
    else:
        turned = replace(turned, generator=sample.generator)
    # The nonunitary refusal, measured on this knot's real flux metric rather than
    # on the declaration alone. The spectral norm is **unitarily invariant**, so
    # `norm(U.H W U - I, 2) = norm(W - I, 2)` exactly when U is unitary -- the
    # transform moves the defect around inside the basis and cannot change its
    # size. A transform that is not unitary changes it, and this catches that on
    # the data even where the declaration's own unitarity budget was met by a
    # matrix that is not the one applied here.
    #
    # The floor is `BASIS_UNITARY_ATOL`, and it is not decoration. A near-unitary
    # `U = V(I + E)` moves the measured norm by `norm(E + E.H, 2)` to first order
    # -- the unitary *defect* itself, whatever `norm(W - I, 2)` is -- so a gate at
    # the rounding of the product alone (2.8e-14 for two channels) refuses a
    # declaration that has just passed its own certificate at 1e-12. This gate
    # therefore certifies the invariance to **the declaration's own budget**: a
    # transform inside `BASIS_UNITARY_ATOL` passes both, one an order above it
    # fails here as well as there.
    count = len(diagonal)
    before = float(np.linalg.norm(sample.flux_metric - np.eye(count), ord=2))
    after = float(np.linalg.norm(turned.flux_metric - np.eye(count), ord=2))
    tolerance = max(
        64
        * count
        * float(np.finfo(float).eps)
        * max(1.0, float(np.linalg.norm(sample.flux_metric, ord=2))),
        BASIS_UNITARY_ATOL,
    )
    if abs(after - before) > tolerance:
        raise ModeError(
            f"a declared change of basis moved this knot's departure from "
            f"orthonormal from {before:.6g} to {after:.6g} at "
            f"{sample.omega:.6e} rad/s, a change of {abs(after - before):.6g} above "
            f"the {tolerance:.6g} rounding of the product: the spectral norm is "
            "unitarily invariant, so a transform that changes it is not unitary, "
            "and a nonunitary basis needs linear.coupled_flow and a measured "
            "transport"
        )
    return turned


def _check_transformed_passivity(
    sample: ModalSample, turned: ModalSample, basis: ChannelBasis
) -> None:
    """Refuse an evaluated ``U.H G U`` whose diagonal shows gain it did not have.

    The gate the exactly degenerate path gets for free by carrying the generator
    over. A unitary transform cannot turn a passive block into an active one -- the
    Hermitian part's eigenvalues are unitarily invariant -- so a diagonal above the
    original's own worst rate by more than the product's rounding is arithmetic that
    went wrong, and ``_channel_index_table`` would otherwise meet it as a channel
    with ``Im n_eff`` of the wrong sign.

    Parameters
    ----------
    sample : solprop.modes.vector.ModalSample
        The sample before the declared transform.
    turned : solprop.modes.vector.ModalSample
        The same sample after it, with the product evaluated.
    basis : solprop.modes.guide.ChannelBasis
        The declaration, named in the refusal.

    Raises
    ------
    ModeError
        If a transformed channel's rate exceeds the block's own worst by more than
        the rounding of the product.
    """
    count = len(np.diag(sample.generator))
    scale = float(np.max(abs(sample.generator)))
    budget = 64.0 * count * float(np.finfo(float).eps) * max(scale, 1.0)
    before = float(np.max(np.real(np.diag(sample.generator))))
    rates = np.real(np.diag(turned.generator))
    worst = int(np.argmax(rates))
    if float(rates[worst]) > before + budget:
        raise ModeError(
            f"the declared change of basis {basis.gauge_id!r} leaves channel "
            f"{worst} with a rate of {float(rates[worst]):.6g} 1/m at "
            f"{sample.omega:.6e} rad/s where the untransformed block's worst is "
            f"{before:.6g} 1/m, a gain of {float(rates[worst]) - before:.6g} above "
            f"the {budget:.6g} rounding of the product: a unitary transform leaves "
            "the Hermitian part's eigenvalues where they were and cannot make a "
            "passive block active"
        )


_INTERPOLATED = (
    "electric",
    "magnetic",
    "generator",
    "backward_electric",
    "backward_magnetic",
)
"""Sample components a dense table interpolates, each with its own knot spline."""


@dataclass(frozen=True, eq=False)
class DenseModalTable(Displayed):
    """Retain full reference samples at every frequency and longitudinal knot.

    Attributes
    ----------
    omega : numpy.ndarray
        Strictly increasing positive frequencies in rad/s, shape (n_omega_table,).
    z : numpy.ndarray
        Strictly increasing positions in m, shape (n_z_table,); one is a snapshot,
        otherwise at least :data:`MIN_Z_KNOTS`.
    samples : tuple of tuple of ModalSample
        Full E/H reference knots indexed by z, then frequency, in one tracked gauge.
    nodes : ModalNodeRule
        Common physical nodes and weights for the entire table.
    degenerate_blocks : tuple of tuple of int
        The blocks the knots were tracked with, retained so that a held-out
        comparison can align a fresh solve in the same declared subspaces. A
        table tracked with a rotating degenerate pair cannot be validated
        without them.
    gram_atol : float
        Positive budget on the sampled Gram defect of every knot, default
        :data:`DEFAULT_GRAM_ATOL`. Construction refuses a basis above it.
    basis : solprop.modes.guide.ChannelBasis or None
        The declared unitary change of basis the knots carry on top of the
        tracking gauge, or ``None`` for the provider's own coordinates. It is
        retained so that a held-out comparison can put a fresh provider solve in
        the same coordinates before aligning it.
    gram_defect : float
        The largest ``norm(W - I, 2)`` over the knots, dimensionless: how far the
        sampled basis is from orthonormal in true power. Reported here so that a
        problem built from this table can carry it into its debug summary.
    """

    omega: Float[np.ndarray, " n_omega_table"]
    z: Float[np.ndarray, " n_z_table"]
    samples: tuple[tuple[ModalSample, ...], ...]
    nodes: ModalNodeRule
    degenerate_blocks: tuple[tuple[int, ...], ...] = ()
    gram_atol: float = DEFAULT_GRAM_ATOL
    basis: ChannelBasis | None = None
    gram_defect: float = field(init=False, default=0.0)
    _omega_splines: Mapping[str, CubicSpline] = field(init=False, repr=False)

    def __repr__(self) -> str:
        """Return a one-line description: the knot counts and the sampled shape.

        Returns
        -------
        str
            For example ``DenseModalTable(9 frequencies, 1 z, 1 channel, 60
            nodes)``.
        """
        _, _, modes = self.samples[0][0].electric.shape
        return line(
            "DenseModalTable",
            [
                ("", f"{len(self.omega)} frequencies"),
                ("", f"{len(self.z)} z"),
                ("", f"{modes} channel{'' if modes == 1 else 's'}"),
                ("", f"{len(self.nodes.weights)} nodes"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        reference = self.samples[0][0]
        _, _, modes = reference.electric.shape
        blocks = (
            ", ".join(str(list(block)) for block in self.degenerate_blocks) or "none"
        )
        return "DenseModalTable: full E/H knots in one tracked gauge", [
            (
                "frequencies",
                f"{len(self.omega)}, {self.omega[0]:.4e} to {self.omega[-1]:.4e} rad/s",
            ),
            (
                "z",
                f"snapshot at {engineering(float(self.z[0]), 'm')}"
                if len(self.z) == 1
                else f"{len(self.z)}, {engineering(float(self.z[0]), 'm')} to "
                f"{engineering(float(self.z[-1]), 'm')}",
            ),
            ("channels", modes),
            ("nodes", len(self.nodes.weights)),
            ("degenerate", blocks),
            ("basis", "the provider's own" if self.basis is None else repr(self.basis)),
            ("gauge", reference.gauge_id),
            ("gram defect", f"{self.gram_defect:.3e} of {self.gram_atol:.3g}"),
        ]

    def __post_init__(self) -> None:
        """Validate and freeze knot arrays and coordinate ownership.

        Raises
        ------
        ModeError
            If knots, samples, gauges or validity disagree.
        """
        for name in ("omega", "z"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if (
                values.ndim != 1
                or not values.size
                or not np.all(np.isfinite(values))
                or np.any(np.diff(values) <= 0)
            ):
                raise ModeError(
                    "table knots must be finite, non-empty and strictly increasing"
                )
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if (
            len(self.omega) < 4
            or self.omega[0] <= 0
            or len(self.samples) != len(self.z)
        ):
            raise ModeError(
                "table requires four positive frequencies and one row per z"
            )
        if 1 < len(self.z) < MIN_Z_KNOTS:
            raise ModeError(
                f"a table needs one z snapshot or at least {MIN_Z_KNOTS} z knots: "
                "the not-a-knot cubic degrades silently to a parabola on three "
                f"positions and to a line on two, got {len(self.z)}"
            )
        if any(len(row) != len(self.omega) for row in self.samples):
            raise ModeError("sample count must match the frequency knots")
        object.__setattr__(
            self,
            "degenerate_blocks",
            tuple(
                tuple(int(index) for index in block) for block in self.degenerate_blocks
            ),
        )
        reference = self.samples[0][0]
        shape = reference.electric.shape
        identity = np.eye(shape[2])
        worst = (0.0, 0.0, 0.0)
        for position, row in zip(self.z, self.samples, strict=True):
            for frequency, sample in zip(self.omega, row, strict=True):
                if (
                    sample.gauge_id != reference.gauge_id
                    or sample.physical_id != reference.physical_id
                    or (sample.reciprocal_gram is None)
                    != (reference.reciprocal_gram is None)
                    or sample.omega != frequency
                    or sample.z != position
                    or sample.electric.shape != shape
                    or shape[0] != len(self.nodes.xy)
                    or not np.all(sample.valid)
                ):
                    raise ModeError(
                        "sample domain, shape or validity disagrees with table"
                    )
                defect = float(np.linalg.norm(sample.flux_metric - identity, ord=2))
                worst = max(worst, (defect, float(frequency), float(position)))
        self._gate_gram(*worst)
        # Stack and fit once, here, rather than in every evaluate: restacking the
        # whole table and refitting two splines per component cost 24 ms for a
        # 17x5 table on 576 nodes, and that cost grew with the table while the
        # answer did not depend on it. The frequency fit is the one that spans the
        # whole table, so caching it removes the n_omega factor from evaluate; the
        # longitudinal fit still runs per call, on the frequency-reduced array,
        # because interpolating z first would change the arithmetic and with it
        # the last bits of every stored result.
        object.__setattr__(
            self,
            "_omega_splines",
            {
                name: CubicSpline(
                    self.omega,
                    np.array(
                        [
                            [getattr(sample, name) for sample in row]
                            for row in self.samples
                        ]
                    ),
                    axis=1,
                    extrapolate=False,
                )
                for name in _INTERPOLATED
                if getattr(reference, name) is not None
            },
        )

    def _gate_gram(self, defect: float, omega: float, z: float) -> None:
        """Record the largest sampled Gram defect and refuse one above the budget.

        Parameters
        ----------
        defect : float
            Largest ``norm(W - I, 2)`` over the knots, dimensionless.
        omega : float
            Angular frequency in rad/s of the knot that reached it.
        z : float
            Position in m of that knot.

        Raises
        ------
        ModeError
            If the budget is not positive and finite, or the defect exceeds it.
        """
        if not np.isfinite(self.gram_atol) or self.gram_atol <= 0:
            raise ModeError("gram_atol must be positive and finite")
        if defect > self.gram_atol:
            raise ModeError(
                f"the sampled basis is {defect:.4g} from orthonormal at omega="
                f"{omega:.6e} rad/s, z={z:.6g} m, above the declared budget "
                f"{self.gram_atol:.4g}: the node rule truncates or under-resolves "
                "these fields, which a provider does not renormalise away; widen "
                "the rule, refine it, or declare a larger gram_atol"
            )
        object.__setattr__(self, "gram_defect", defect)

    def _interpolate(
        self, name: str, omega: float, z: float, derivative: int = 0
    ) -> Complex[np.ndarray, "*shape"]:
        """Evaluate a dense field or generator component inside its knots.

        Parameters
        ----------
        name : str
            Stored component name, one of :data:`_INTERPOLATED`.
        omega : float
            Positive angular frequency in rad/s.
        z : float
            Position in m within the declared path.
        derivative : int, optional
            Non-negative angular-frequency derivative order.

        Returns
        -------
        numpy.ndarray
            Interpolated component, with its stored shape.

        Raises
        ------
        ModeError
            If the requested point lies outside the knots.
        """
        if (
            not self.omega[0] <= omega <= self.omega[-1]
            or not self.z[0] <= z <= self.z[-1]
        ):
            raise ModeError("modal table does not extrapolate frequency or z")
        along_omega = self._omega_splines[name](omega, derivative)
        return np.asarray(
            along_omega[0]
            if len(self.z) == 1
            else CubicSpline(self.z, along_omega, axis=0, extrapolate=False)(z)
        )

    def _bracketing_residual(
        self, omega: float, z: float
    ) -> Float[np.ndarray, " n_modes"]:
        """Return the largest knot boundary residual around a requested point.

        Parameters
        ----------
        omega : float
            Angular frequency in rad/s inside the knot interval.
        z : float
            Position in m inside the knot interval.

        Returns
        -------
        numpy.ndarray
            Per-channel residual in the interpolation cell, shape (n_modes,).
        """
        # Carrying knot (0, 0)'s residual described a corner of the table, not the
        # point asked for. The bracketing knots are the local physical statement.
        cells = []
        for knots, value in ((self.omega, omega), (self.z, z)):
            lower = max(int(np.searchsorted(knots, value, side="right")) - 1, 0)
            cells.append((lower, min(lower + 1, len(knots) - 1)))
        return np.max(
            [
                self.samples[position][frequency].boundary_residual
                for position in cells[1]
                for frequency in cells[0]
            ],
            axis=0,
        )

    def evaluate(self, omega: float, z: float) -> ModalSample:
        """Interpolate full E/H and the generator inside the table domain.

        Parameters
        ----------
        omega : float
            Angular frequency in rad/s inside the knot interval.
        z : float
            Position in m inside the knot interval, or the snapshot position.

        Returns
        -------
        ModalSample
            Dense interpolated sample; its flux is recomputed from interpolated E/H
            and its ``boundary_residual`` is the largest of the bracketing knots'.
            Held-out validation is required before using this approximation.
        """
        e = self._interpolate("electric", omega, z)
        h = self._interpolate("magnetic", omega, z)
        metric = flux_product(e, h, self.nodes.weights)
        reference = self.samples[0][0]
        backward_electric = backward_magnetic = reciprocal = None
        if reference.reciprocal_gram is not None:
            backward_electric = self._interpolate("backward_electric", omega, z)
            backward_magnetic = self._interpolate("backward_magnetic", omega, z)
            # Recompute the bilinear flux of the interpolated fields. Interpolating
            # C separately would violate reciprocity between knots, just as an
            # independently interpolated W would mismeasure physical power.
            cross = np.cross(e[:, :, None, :], backward_magnetic[:, :, :, None], axis=1)
            reverse = np.cross(
                backward_electric[:, :, :, None], h[:, :, None, :], axis=1
            )
            reciprocal = np.einsum(
                "qmn,q->mn", cross[:, 2] - reverse[:, 2], self.nodes.weights
            )
        return replace(
            reference,
            omega=omega,
            z=z,
            electric=e,
            magnetic=h,
            generator=self._interpolate("generator", omega, z),
            flux_metric=metric,
            boundary_residual=self._bracketing_residual(omega, z),
            backward_electric=backward_electric,
            backward_magnetic=backward_magnetic,
            reciprocal_gram=reciprocal,
        )

    def generator_derivative(
        self, omega: float, z: float
    ) -> Complex[np.ndarray, "n_modes n_modes"]:
        """Differentiate the dense generator interpolant in angular frequency.

        Parameters
        ----------
        omega : float
            Angular frequency in rad/s inside the knot interval.
        z : float
            Position in m inside the knot interval.

        Returns
        -------
        numpy.ndarray
            Generator derivative in s/(rad m), shape (n_modes, n_modes).
        """
        return self._interpolate("generator", omega, z, 1)


def sample_modal_table(  # noqa: PLR0913 - a provider, two grids, a rule and three declarations
    provider: ModalProvider,
    omega: Float[np.ndarray, " n_omega"],
    z: Float[np.ndarray, " n_z"],
    nodes: ModalNodeRule,
    *,
    degenerate_blocks: Sequence[Sequence[int]] = (),
    basis: ChannelBasis | None = None,
    gram_atol: float = DEFAULT_GRAM_ATOL,
) -> DenseModalTable:
    """Build dense knots with frequency and longitudinal physical-overlap tracking.

    Parameters
    ----------
    provider : ModalProvider
        Internal-convention bound-mode physics provider.
    omega : numpy.ndarray
        At least four increasing positive frequencies in rad/s, shape (n_omega,).
    z : numpy.ndarray
        Increasing longitudinal positions in m, shape (n_z,); one position, or at
        least :data:`MIN_Z_KNOTS` of them.
    nodes : ModalNodeRule
        Common physical nodes; changing material ownership requires separate tables.
    degenerate_blocks : sequence of sequence of int, optional
        Declared blocks passed to the physical-overlap alignment, and retained on
        the table so that held-out validation aligns a fresh solve the same way.
    basis : solprop.modes.guide.ChannelBasis or None, optional
        A declared unitary change of channel coordinates, applied to every knot
        **after** the tracking gauge and gated on the exact degeneracy of each of
        its blocks (:func:`apply_channel_basis`, :func:`composed_gauge`). Every
        block it acts inside must be one of `degenerate_blocks`: tracking rotates
        freely inside a declared block, so a transform of a block the tracking did
        not know about would be applied on top of an ordering the eigenvalues
        cannot resolve. ``None`` keeps the provider's own coordinates, and then
        this function is bitwise what it was.
    gram_atol : float, optional
        Budget on each knot's sampled Gram defect, default
        :data:`DEFAULT_GRAM_ATOL`; the table refuses a basis above it.

    Returns
    -------
    DenseModalTable
        Full E/H knots, with frequency tracking after each z-row entrance alignment.

    Raises
    ------
    ModeError
        If the provider has an unsupported convention or crosses a physical kink,
        if `basis` acts on a block the tracking was not given or on one whose
        propagation constants differ, or if the sampled basis is further from
        orthonormal than `gram_atol`.
    """
    if (
        len(omega) < 4
        or not len(z)
        or not np.all(np.isfinite(omega))
        or not np.all(np.isfinite(z))
        or omega[0] <= 0
        or np.any(np.diff(omega) <= 0)
        or np.any(np.diff(z) <= 0)
    ):
        raise ModeError(
            "table requires four increasing positive frequencies and non-empty z"
        )
    if 1 < len(z) < MIN_Z_KNOTS:
        raise ModeError(
            f"a table needs one z snapshot or at least {MIN_Z_KNOTS} z knots: "
            "the not-a-knot cubic degrades silently to a parabola on three "
            f"positions and to a line on two, got {len(z)}"
        )
    if provider.identity.get("spectral_convention") != "internal":
        raise ModeError("modal providers must declare the internal spectral convention")
    if any(z[0] < point < z[-1] for point in provider.z_breakpoints):
        raise ModeError("a physical kink requires separate modal table sections")
    blocks = tuple(tuple(int(index) for index in block) for block in degenerate_blocks)
    if basis is not None:
        declared = {frozenset(block) for block in blocks}
        unknown = [
            list(block) for block in basis.blocks if frozenset(block) not in declared
        ]
        if unknown:
            raise ModeError(
                f"the declared basis {basis.gauge_id!r} mixes channels {unknown}, "
                f"which this table's degenerate blocks {[list(b) for b in blocks]} "
                "do not name: the tracking rotates freely inside a declared block "
                "and matches one-to-one outside it, so a transform of an "
                "undeclared block would be applied on top of an ordering the "
                "eigenvalues cannot resolve"
            )
    rows: list[tuple[ModalSample, ...]] = []
    for position in z:
        row: list[ModalSample] = []
        for frequency in omega:
            sample = provider.solve(float(frequency), float(position), nodes=nodes)
            reference = row[-1] if row else rows[-1][0] if rows else None
            if reference is not None:
                sample = align_modal_sample(
                    reference, sample, nodes, degenerate_blocks=degenerate_blocks
                )
            row.append(sample)
        rows.append(tuple(row))
    if basis is not None:
        # After the tracking, never inside it: the composed transform is
        # `T_track U` and a run with no declared basis is bitwise what it was.
        rows = [tuple(apply_channel_basis(one, basis) for one in row) for row in rows]
    return DenseModalTable(omega, z, tuple(rows), nodes, blocks, gram_atol, basis)


def validate_modal_table(
    table: DenseModalTable,
    provider: ModalProvider,
    points: Float[np.ndarray, "n_points 2"],
    *,
    profile_rtol: float,
    generator_atol: float,
    flux_atol: float,
) -> Float[np.ndarray, "n_points 3"]:
    """Compare dense interpolation with held-out physical provider solves.

    The third defect is the flux defect ``norm(W - I, 2)`` of the interpolated
    basis: an interpolated field is no longer an exact mode, and the spectral norm
    of its sampled Gram against the identity says by how much the basis has
    stopped being orthonormal in true power. It falls with the frequency knots
    like the profile error -- measured 2.25e-8, 1.13e-9 and 5.96e-13 for 5, 9 and
    17 knots on the 250 nm strand -- but a held-out *position* also carries a
    provider effect that refinement in frequency does not remove: the node rule is
    frozen at the entrance radius, so on a taper the panel edge no longer sits on
    the material interface and the defect grows linearly with the displacement
    (4.9e-7, 9.8e-7 and 1.95e-6 at z = 0.25, 0.5 and 1.0 m for a 1e-5 relative
    radius change, 7.22e-7 at the report's held-out z). Budget it accordingly, and
    split the table where an interface crosses a node.

    Parameters
    ----------
    table : DenseModalTable
        Reference table on fixed physical nodes. Its declared degenerate blocks
        are reused here, so a table tracked through a rotating degenerate pair is
        compared in the same subspaces it was built in.
    provider : ModalProvider
        Independent fresh mode solves, in the same physical model.
    points : numpy.ndarray
        Held-out (omega, z) pairs in rad/s and m, shape (n_points, 2).
    profile_rtol : float
        Positive relative E/H weighted norm budget, fixed before evaluation.
    generator_atol : float
        Positive absolute generator error budget in 1/m, fixed before evaluation.
    flux_atol : float
        Positive dimensionless budget on ``norm(W - I, 2)``, fixed before
        evaluation.

    Returns
    -------
    numpy.ndarray
        Profile, generator and flux defects, shape (n_points, 3).

    Raises
    ------
    ModeError
        If a budget is invalid or a held-out error exceeds it.
    """
    budgets = (profile_rtol, generator_atol, flux_atol)
    if any(not np.isfinite(value) or value <= 0 for value in budgets):
        raise ModeError("held-out budgets must be positive and finite")
    errors = []
    weight = np.sqrt(table.nodes.weights)[:, None, None]
    for omega, z in points:
        sample = table.evaluate(float(omega), float(z))
        direct = align_modal_sample(
            sample,
            solved_in_table_basis(table, provider, float(omega), float(z)),
            table.nodes,
            degenerate_blocks=table.degenerate_blocks,
        )
        electric = np.linalg.norm(
            weight * (sample.electric - direct.electric)
        ) / np.linalg.norm(weight * direct.electric)
        magnetic = np.linalg.norm(
            weight * (sample.magnetic - direct.magnetic)
        ) / np.linalg.norm(weight * direct.magnetic)
        profile = max(float(electric), float(magnetic))
        generator = float(np.linalg.norm(sample.generator - direct.generator, ord=2))
        flux = float(
            np.linalg.norm(sample.flux_metric - np.eye(len(sample.flux_metric)), ord=2)
        )
        if profile > profile_rtol or generator > generator_atol or flux > flux_atol:
            raise ModeError(
                f"held-out modal error exceeds budget: profile={profile}, "
                f"generator={generator}, flux={flux}"
            )
        errors.append((profile, generator, flux))
    return np.asarray(errors).reshape(-1, 3)


def solved_in_table_basis(
    table: DenseModalTable, provider: ModalProvider, omega: float, z: float
) -> ModalSample:
    """Solve a fresh sample and put it in the table's declared channel basis.

    A held-out comparison asks the provider again and aligns the answer against the
    table's own knots. The provider always answers in **its** coordinates, so a
    table carrying a declared basis must apply the same declaration before the
    alignment; otherwise the degenerate-block SVD would be asked to discover the
    declared transform, which it can only do up to the same block rotation the
    declaration exists to fix.

    Parameters
    ----------
    table : DenseModalTable
        The table whose basis the solve is put in.
    provider : solprop.modes.vector.ModalProvider
        The physics, asked again at one point.
    omega : float
        Positive angular frequency in rad/s.
    z : float
        Position in m.

    Returns
    -------
    solprop.modes.vector.ModalSample
        The fresh solve, in the table's coordinates but not yet aligned to it.
    """
    sample = provider.solve(omega, z, nodes=table.nodes)
    basis = table.basis
    return sample if basis is None else apply_channel_basis(sample, basis)


def channel_area(sample: ModalSample, nodes: ModalNodeRule, channel: int) -> float:
    """Return the scalar effective area of one channel of a vector sample.

    The scalar definition ``A_eff = (int |E|^2 dA)^2 / int |E|^4 dA`` with
    ``|E|^2 = E . E*`` summed over the three Cartesian components, integrated on the
    caller's rule. It is a **surrogate**: it is what the legacy mode-averaged
    interfaces mean by an area, and it is not the nonlinear coefficient of a vector
    channel, which is the full contraction of
    :func:`solprop.nonlinear.normalisation.modal_kerr_coefficient` over the material
    regions that respond.

    Parameters
    ----------
    sample : ModalSample
        Internal-convention sample whose fields are normalised to unit power.
    nodes : ModalNodeRule
        The rule the sample was evaluated on; supplies the area weights in m².
    channel : int
        Channel index on the mode axis.

    Returns
    -------
    float
        Effective area in m².
    """
    intensity = np.sum(abs(sample.electric[:, :, channel]) ** 2, axis=1)
    return float(
        np.sum(nodes.weights * intensity) ** 2 / np.sum(nodes.weights * intensity**2)
    )


def _channel_splitting(
    diagonal: Complex[np.ndarray, " n_modes_diagonal"],
) -> float:
    """Return the scale a generator's off-diagonal is judged against.

    The smallest splitting ``min |beta_i - beta_j|`` between two channels, which is
    what decides whether an off-diagonal mixes them, floored at the rounding of the
    propagation constants themselves so that an exactly degenerate pair -- or a
    single channel, which has no splitting at all -- still has a positive scale.

    Parameters
    ----------
    diagonal : numpy.ndarray
        The generator's diagonal in 1/m, shape (n_modes,).

    Returns
    -------
    float
        Positive scale in 1/m.
    """
    floor = 64 * float(np.finfo(float).eps) * float(np.max(abs(diagonal)))
    if len(diagonal) < 2:
        return floor
    off = ~np.eye(len(diagonal), dtype=bool)
    return max(float(np.min(abs(diagonal[:, None] - diagonal[None, :])[off])), floor)


def _area_knot(omega: Float[np.ndarray, " n_omega"]) -> int:
    """Return the knot a channel's single scalar surrogate area is taken at.

    The **median** knot, the middle of the band the table was sampled over, so that
    a scalar a whole run must use is the band's central value rather than one of its
    ends. Which frequency it is taken at is part of the number: measured on a
    parabolic-index sector guide, the lowest knot's surrogate area is 2.0 % above the
    median knot's over a +-2 % band, 25 % above it over +-20 % and 66 % above it
    over +-40 %, so a scalar taken at the entrance is a different quantity from one
    taken in the middle. For an even knot count this is the upper of the
    two middle knots, so that it is always a frequency the band was solved at.

    Parameters
    ----------
    omega : numpy.ndarray
        Increasing angular frequencies in rad/s, shape (n_omega,).

    Returns
    -------
    int
        Index on the frequency axis.
    """
    return len(omega) // 2


def generator_is_coupled(table: DenseModalTable) -> bool:
    """Return whether a tracked table's generator couples its channels anywhere.

    The one predicate both callers of the split-block route ask: the guide
    constructor, to know whether its per-channel index tables are surrogates
    rather than propagation constants, and the route, to know whether to build
    ``linear.coupled_flow`` (:func:`solprop.translate.general.coupled_block`). It is
    judged at the **rounding of the propagation constants** themselves,
    ``64 eps max|beta|``, the same floor :func:`apply_channel_basis` gates its
    declared blocks at, so that a generator diagonal to the bit is diagonal here too.

    Parameters
    ----------
    table : DenseModalTable
        The tracked reference, in whatever basis was declared on it.

    Returns
    -------
    bool
        True when some knot's generator has an off-diagonal above that floor.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.graded_index import (
    ...     GradedIndexProvider, PowerLawPermittivity,
    ... )
    >>> from solprop.modes.material import IndexMaterial
    >>> from solprop.translate.modal_tables import (
    ...     generator_is_coupled, sample_modal_table,
    ... )
    >>> guide = GradedIndexProvider(
    ...     permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
    ...     exterior=IndexMaterial(1.45 * np.sqrt(1 - 2 * 0.005)),
    ...     radius=25e-6, family="LP", count=2, elements=16, extent=2.0,
    ... )
    >>> omega = np.linspace(1.80e15, 1.86e15, 4)
    >>> table = sample_modal_table(
    ...     guide, omega, np.array([0.0]), guide.node_rule(1.80e15, n_r=2)
    ... )
    >>> generator_is_coupled(table)
    False
    """
    for row in table.samples:
        for sample in row:
            diagonal = np.diag(sample.generator)
            floor = 64.0 * float(np.finfo(float).eps) * float(np.max(abs(diagonal)))
            if float(np.max(abs(sample.generator - np.diag(diagonal)))) > floor:
                return True
    return False


def _channel_index_table(
    table: DenseModalTable, channel: int, *, include_area: bool, coupled: bool = False
) -> IndexTable:
    """Read one channel's dispersion out of a tracked dense table.

    Parameters
    ----------
    table : DenseModalTable
        Tracked knots of one physical guide at one longitudinal position.
    channel : int
        Channel index on the mode axis.
    include_area : bool
        Sample :func:`channel_area` at every frequency knot.
    coupled : bool, optional
        Admit a generator that couples its channels and read ``i G_cc``, the
        channel's **mean** propagation constant, off its diagonal. Default false,
        which refuses one by name. It is not a tuning knob: a coupled basis has no
        per-channel propagation constant, because its channels beat, and a run that
        declares one propagates the whole block through ``linear.coupled_flow``
        instead. What the per-channel table is still
        *for* on such a route is everything the flow does not do -- the surrogate
        effective area, the dimension limits, the labels, and the cumulative
        ``Phi`` of the **one** reference channel the block is measured against -- so
        it is built, from the diagonal, by a caller that has said it knows which of
        the two it is asking for.

    Returns
    -------
    IndexTable
        Index snapshot with the interpolant's slopes at the knots.

    Raises
    ------
    ModeError
        If the channel is not on the mode axis, the table spans more than one
        position, the generator couples its channels and `coupled` is false, or a
        channel declares gain (``Re G > 0`` in the internal convention).
    """
    modes = table.samples[0][0].electric.shape[2]
    if not 0 <= channel < modes:
        raise ModeError(f"channel {channel} is outside the {modes} tracked channels")
    if len(table.z) != 1:
        raise ModeError(
            "a channel index table is a snapshot: build it from a table at one "
            f"position, not {len(table.z)}"
        )
    z = float(table.z[0])
    for sample in table.samples[0]:
        generator = sample.generator
        diagonal = np.diag(generator)
        coupling = float(np.linalg.norm(generator - np.diag(diagonal)))
        splitting = _channel_splitting(diagonal)
        if not coupled and coupling > MAX_GENERATOR_COUPLING * splitting:
            raise ModeError(
                f"a channel index table needs a diagonal generator; this one "
                f"couples its channels by {coupling:.3g} 1/m, which is "
                f"{coupling / splitting:.3g} of their smallest splitting "
                f"({splitting:.3g} 1/m) and above {MAX_GENERATOR_COUPLING}, and a "
                "coupled basis has no per-channel propagation constant"
            )
    # beta is the provider's own generator, i G_cc, and the two derivatives come
    # from the dense interpolant of that generator: a provider declares no analytic
    # frequency derivative.
    beta = np.array(
        [1j * sample.generator[channel, channel] for sample in table.samples[0]]
    )
    slopes = np.array(
        [
            1j * table.generator_derivative(float(omega), z)[channel, channel]
            for omega in table.omega
        ]
    )
    # A provider's generator is internal-convention (conventions.md section 2):
    # G_cc = -i (omega/c) conj(n_eff), so a passive channel has Re G_cc <= 0 and a
    # positive real part is gain. Reading Im beta instead had the sign backwards --
    # it refused loss and admitted gain.
    real_part = np.array(
        [
            float(np.real(sample.generator[channel, channel]))
            for sample in table.samples[0]
        ]
    )
    if np.any(real_part > 0):
        raise ModeError(
            "a channel index table must be passive, and this one declares gain: "
            f"its internal generator reaches Re G = {float(np.max(real_part)):.6g} "
            "1/m at a knot, and a passive channel has Re G <= 0 "
            "(docs/dev/conventions.md section 2)"
        )
    # Internal to optical, in one place. An IndexTable is a P-layer record in the
    # optics convention, where Im n_eff > 0 is loss, and the internal generator has
    # the conjugate; the table-to-operator site undoes this when the run is built,
    # so it is the ordinary two-site round trip of any optical table and not a
    # fourth conjugation site.
    neff = np.conj(C * beta / table.omega)
    area = (
        np.array(
            [channel_area(sample, table.nodes, channel) for sample in table.samples[0]]
        )
        if include_area
        else None
    )
    # A bound lossless provider has an imaginary index of exactly zero, which has
    # no logarithm and which the table's default policy refuses outright.
    return _SampledIndexTable(
        table.omega,
        neff,
        area,
        "linear",
        derivative_values=(C * np.conj(slopes) - neff) / table.omega,
        source_identity={
            "mode_type": "solprop.modes.vector.ModalProvider",
            "mode": table.samples[0][0].physical_id,
            "channel": channel,
            "z": z,
            "omega_domain": (float(table.omega[0]), float(table.omega[-1])),
            "area_model": "sampled-vector-surrogate" if include_area else "none",
            # Which knot a consumer that needs one scalar area must read, so that
            # the frequency the surrogate belongs to is on the record.
            "channel_area_omega": float(table.omega[_area_knot(table.omega)])
            if include_area
            else None,
            "derivative_model": "dense-table-interpolant",
            "gram_defect": table.gram_defect,
        },
    )


def sample_provider_table(
    provider: ModalProvider,
    channel: int,
    omega: Float[np.ndarray, " n_omega"],
    *,
    z: float = 0.0,
    nodes: ModalNodeRule | None = None,
    include_area: bool = True,
) -> IndexTable:
    """Sample one channel of a modal provider into a dense index snapshot.

    The sibling of :func:`solprop.modes.tabulated.sample_index_table` for a provider
    rather than a mode. ``beta`` is the provider's own generator at the knots, read
    off the tracked diagonal, so the channel is followed by physical overlap and not
    by eigenvalue order; ``beta_1`` and ``beta_2`` are **the interpolant's**, the
    dense table's cubic in frequency differentiated, because a provider declares no
    analytic frequency derivative.

    Nothing here certifies those derivatives, and :func:`validate_modal_table` does
    not either: it measures a profile, a generator and a flux defect, and no
    frequency derivative at all. Measured held-out against fresh-solve five-point
    differences on the reference parabolic sector of
    ``tests/modes/test_modal_tables.py``, over a +-2 % band, ``beta_1`` is 4.4e-13
    relative at nine knots while ``beta_2`` is 2.8e-2 -- eleven orders apart,
    because a table's cubic reproduces a nearly constant group delay exactly and
    still has almost no curvature. Treat ``beta_1`` as the interpolant's, and
    ``beta_2`` as a **diagnostic**
    (measured by a development study of the provider-table derivatives;
    :class:`solprop.modes.guide.ProviderChannelMode`).

    Parameters
    ----------
    provider : ModalProvider
        Internal-convention bound-mode provider that names its channels.
    channel : int
        Channel index on the provider's mode axis.
    omega : numpy.ndarray
        At least four increasing positive frequencies in rad/s, shape (n_omega,).
    z : float, optional
        Snapshot position in m inside the provider's z domain.
    nodes : ModalNodeRule or None, optional
        Common physical rule; ``None`` asks the provider for one at the **lowest**
        frequency of the band, where the power a rule discards is largest, so that
        its certified tail is a bound over the band.
    include_area : bool, optional
        Sample the surrogate :func:`channel_area` at every frequency.

    Returns
    -------
    IndexTable
        Sorted, immutable index samples with the interpolant's slopes, carrying
        ``derivative_values`` and a ``source_identity`` that names the provider,
        the channel, the snapshot position, the area model and the derivative
        model.

    Raises
    ------
    ModeError
        If the frequencies, the channel, the rule or the sampled basis is invalid.
    """
    frequencies = np.asarray(omega, dtype=float)
    if nodes is None:
        # The lowest frequency, where a bound field reaches furthest, so that the
        # rule's certified tail bounds the whole band.
        nodes = provider.node_rule(float(np.min(frequencies)), z)
    table = sample_modal_table(provider, frequencies, np.array([float(z)]), nodes)
    return _channel_index_table(table, channel, include_area=include_area)


@dataclass(frozen=True, eq=False)
class ModalTransformTables(Displayed):
    """Own the full E/H reference of a general guide, flat, as the seam will take it.

    Built from a :class:`DenseModalTable` by :func:`modal_transform_tables`, or along a
    moving guide by :mod:`solprop.translate.modal_path`. It is flat -- arrays, not a
    nest of samples -- because that is what a backend can be handed, and it is the
    ``transform.modal_tables`` leaf of a ``KernelProblem``.

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
        non-negative. On a ``KernelProblem`` it is ``grids.omega`` itself, whose
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

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
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
    def frozen_rule(self) -> bool:
        """Return whether one node rule was broadcast across every position.

        A record built over several positions from **one** rule -- the same
        Cartesian nodes and the same area weights at every z -- is the frozen-rule
        case. It is legitimate and it is not free: along a path the panel edge leaves
        the material interface, and the interpolated basis's flux defect grows
        linearly with that displacement (4.9e-7, 9.8e-7 and 1.95e-6 at a quarter, a
        half and the whole of a 1e-5 relative radius change). A taper needs sections
        with their
        own rules (:mod:`solprop.translate.modal_path`); a snapshot is trivially
        frozen.

        Returns
        -------
        bool
            True when ``node_xy`` and ``weights`` do not vary along z.
        """
        return bool(
            np.all(self.node_xy == self.node_xy[0])
            and np.all(self.weights == self.weights[0])
        )

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Return the scalar identity a run records and a resume compares.

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

        Returns
        -------
        mapping
            JSON-serialisable scalars, stable across processes.
        """
        positions, nodes, components, modes, frequencies = np.shape(self.electric)
        factors = self.factors
        # A moving guide's maps add their sections and degree,
        # written only where the maps move so that every snapshot record keeps the
        # identity it was stamped with.
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
            values = np.array(
                getattr(self, name), dtype=np.float64, order="C", copy=True
            )
            if (
                values.ndim != 1
                or not values.size
                or not np.all(np.isfinite(values))
                or np.any(np.diff(values) <= 0)
            ):
                raise ModeError(
                    f"{name} must be a finite, non-empty, increasing vector"
                )
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.omega[0] < 0 or self.omega_fine[0] < 0:
            raise ModeError("frequency grids must be non-negative")
        self._validate_arrays()

    def _map_degree(self, intervals: int) -> int:
        """Return the Chebyshev degree count the two execution maps share.

        Parameters
        ----------
        intervals : int
            Intervals the record's positions declare.

        Returns
        -------
        int
            The degrees along z, or zero when the record carries no map at all.

        Raises
        ------
        ModeError
            If only one half of the map is present, or the synthesis does not
            carry one leading interval axis and at least one degree.
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
        """Copy, check and freeze every leaf against the extents the record declares.

        Raises
        ------
        ModeError
            If an array has the wrong shape or dtype, holds a value it may not, or
            belongs to a group that is only half supplied.
        """
        # The component count is the contract's ceiling, which lives beside the
        # state's polarisation ceiling in translate.problem; that module imports this
        # one, so the import is deferred rather than the number copied.
        from solprop.translate.problem import (  # noqa: PLC0415 - import cycle
            MAX_N_COMPONENTS,
        )

        reference = np.array(self.electric, dtype=np.complex128, order="C", copy=True)
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
            array = np.array(value, dtype=dtype, order="C", copy=True)
            if array.shape != shape:
                raise ModeError(f"{name} must have shape {shape}, got {array.shape}")
            if dtype is not np.bool_ and not np.all(np.isfinite(array)):
                raise ModeError(f"{name} must be finite")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if np.any(self.weights <= 0):
            raise ModeError("area weights must be positive")
        if np.any(self.region_index < 0):
            raise ModeError("region_index must be non-negative")
        if np.any(self.profile_error < 0):
            raise ModeError("profile_error must be non-negative")


def _coordinate_dual(
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
    sample : ModalSample
        Internal-convention sample at one frequency and position.
    weights : numpy.ndarray
        Area weights in m², shape (n_nodes,).

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


def _projection_map(
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
    sample : ModalSample
        Internal-convention sample at one frequency and position.
    weights : numpy.ndarray
        Area weights in m², shape (n_nodes,).

    Returns
    -------
    numpy.ndarray
        The projection, shape (n_modes, n_nodes, 3), in V m/sqrt(W).
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
        has measured, for example with :func:`validate_modal_table`; nothing here
        infers it.
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
    for row, position in enumerate(positions):
        at = float(table.z[0]) if snapshot else float(position)
        for column in np.flatnonzero(inside):
            sample = table.evaluate(float(coarse[column]), at)
            stack[row, :, :, :, column] = sample.electric
            magnetic[row, :, :, :, column] = sample.magnetic
            dual[row, :, :, :, column] = _coordinate_dual(sample, weights)
            if backward[0] is not None and backward[1] is not None:
                backward[0][row, :, :, :, column] = sample.backward_electric
                backward[1][row, :, :, :, column] = sample.backward_magnetic
            if gram is not None:
                gram[row, :, :, column] = sample.reciprocal_gram
            if projection is not None and not row:
                projection[:, 0, :, :, :, column] = _projection_map(sample, weights)
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

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
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
