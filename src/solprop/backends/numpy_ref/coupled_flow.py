# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coherent wall's coupled linear flow, as the reference executes it (SEP 16).

A ``coupled_flow`` leaf mixes the channel axis per frequency bin. :func:`held_flow`
decomposes it once into :class:`HeldFlow`: each section's Schur factors on the bins
that carry a block, the residual's terms and the declared conditioning certificate.
:func:`flow_action` forms a step's forward and inverse actions ``Z exp(+-l T) Z^H``
by scaling, a Taylor series and squaring; :func:`check_flow_conditioning` measures
``norm(U) norm(U^-1)`` and refuses a step above the certificate; :func:`apply_flow`
mixes a state by an action. A ``"frozen_residual"`` flow adds an explicit residual
to each stage's source: :func:`residual_action` forms ``R_s(z)``,
:func:`residual_drive` applies it on the scored bins, and :class:`BlockResidual`
keeps those actions for the Adams history's samples.
"""

import collections
import logging
import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Complex, Float

from solprop.exceptions import BackendError
from solprop.translate.coupled import CoupledFlowFactors, coupled_flow_factors
from solprop.translate.problem import PropagationProblem
from solprop.translate.stage import combine

__all__ = [
    "BlockResidual",
    "HeldFlow",
    "apply_flow",
    "check_flow_conditioning",
    "flow_action",
    "held_flow",
    "residual_action",
    "residual_drive",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeldFlow(CoupledFlowFactors):
    """What ``numpy_ref`` holds to execute one coherent block, frozen per section.

    The shared factors (:class:`~solprop.translate.coupled.CoupledFlowFactors`, which
    native reads too), with the section axis leading and only the bins the generator
    is non-zero at held: the rest of the band has an identity action and the state is
    masked to zero there anyway.

    A ``"constant_schur"`` leaf is the one-section case with no residual at all, and
    goes through exactly the constant-section arithmetic. A ``"frozen_residual"`` leaf
    adds `residual_terms`: the coefficients of ``G(z) - C(z)`` along z, flattened over
    ``(n_intervals, n_deg)`` in that order, so that the stage bundle's
    ``flow_weights`` combine them into ``G(z_1) - C(z_1)`` exactly as ``phi_delta``
    combines the terms of ``dPhi`` (``docs/dev/kernel-contract.md`` section 1). The
    explicit residual is that combination minus the frozen block of the step's own
    section. It is applied only on `residual_bins`, the active bins of the requested
    band; in the spectral collar outside them the frozen flow alone carries the state
    (``docs/guide/waveguides/general-modal.md``).

    Attributes
    ----------
    residual_bins : numpy.ndarray
        Which bins the explicit residual is applied at, bool, shape ``(n_omega,)``:
        `active` and the problem's ``grids.inband_mask``, the requested band: in the
        collar beyond it the wall model is outside its validity, and the residual is
        left out whether or not the error norm scores there.
    n_modes : int
        Channels the block couples.
    """

    residual_bins: Bool[np.ndarray, " n_omega_scored"]
    n_modes: int


def held_flow(problem: PropagationProblem) -> HeldFlow | None:
    """Decompose the coupled flow this backend will execute, once.

    - Returns: the Schur factors and the certificate, or ``None`` for a problem carrying
      no coupled flow.
    """
    flow = problem.linear.coupled_flow
    if flow is None:
        return None
    assert flow.generator is not None
    factors = coupled_flow_factors(flow)
    active, residual = factors.active, factors.residual_terms
    sections = len(flow.section_methods)
    held = HeldFlow(
        **vars(factors),
        residual_bins=active & problem.grids.inband_mask,
        n_modes=flow.n_modes,
    )
    logger.debug(
        "coupled flow: %d channel(s) on %d of %d bins over %d section(s), %s, largest "
        "declared conditioning %.3e, largest declared residual %.3e 1/m, largest "
        "declared representation error %.3e",
        held.n_modes,
        int(np.count_nonzero(active)),
        len(active),
        sections,
        "no explicit residual" if residual is None else f"{residual.shape[0]} terms",
        float(np.max(held.condition_bound, initial=1.0)),
        float(np.max(flow.residual_norm_bound)),
        float(np.max(flow.generator_error)),
    )
    return held


_TAYLOR_ORDER = 16
"""Degree of the Taylor series a scaled triangular exponential is summed to.

Scaling brings the row-sum norm below :data:`_TAYLOR_SCALE`, so the truncated tail is
at most ``_TAYLOR_SCALE**(n+1)/(n+1)!``, which is 2.1e-20 at this degree -- below the
rounding of the squarings that follow it and of the two unitary products around it.
The alternative, a Pade approximant, buys a lower degree at the cost of a solve per
frequency; the oracle takes the series (``docs/dev/architecture.md`` section 3:
physics and reference code carry no performance tricks).
"""


_TAYLOR_SCALE = 0.5
"""Row-sum norm a block is scaled below before the series is summed."""


_MAX_FLOW_SQUARINGS = 30
"""Largest number of squarings a flow action may need before it is refused.

``2**30`` times the scale is a step whose own generator spans 21 orders of magnitude
in the exponent; a problem that needs it has not been certified by anything the
translation layer measured.
"""


def _flow_squarings(largest: float) -> int:
    """Return how many squarings an exponent of this row-sum norm needs, or refuse.

    The count brings the norm below :data:`_TAYLOR_SCALE`; past
    :data:`_MAX_FLOW_SQUARINGS` the step is refused, as native refuses it.
    """
    squarings = 0
    if largest > _TAYLOR_SCALE:
        squarings = math.ceil(math.log2(largest / _TAYLOR_SCALE))
    if squarings > _MAX_FLOW_SQUARINGS:
        msg = (
            "numpy_ref cannot take a coupled flow action whose exponent has row-sum "
            f"norm {largest:.4g}: it would need {squarings} squarings, above the "
            f"{_MAX_FLOW_SQUARINGS} this backend admits. Cap dz_max"
        )
        raise BackendError(msg)
    return squarings


def _triangular_exponential(
    blocks: Complex[np.ndarray, "n_active_e n_modes_e_out n_modes_e_in"],
) -> Complex[np.ndarray, "n_active_x n_modes_x_out n_modes_x_in"]:
    """Return ``exp(A)`` for a stack of blocks, by scaling, a series and squaring.

    - Returns: ``exp(A)`` in the same layout.
    """
    largest = float(np.max(np.sum(abs(blocks), axis=-1), initial=0.0))
    squarings = _flow_squarings(largest)
    scaled = blocks / float(2**squarings)
    identity = np.broadcast_to(
        np.eye(blocks.shape[1], dtype=np.complex128), blocks.shape
    )
    series = identity.copy()
    for term in range(_TAYLOR_ORDER, 0, -1):
        series = identity + scaled @ series / term
    for _ in range(squarings):
        series = series @ series
    return series


def flow_action(
    flow: HeldFlow, section: int, length: float
) -> tuple[
    Complex[np.ndarray, "n_active_f n_modes_f_out n_modes_f_in"],
    Complex[np.ndarray, "n_active_i n_modes_i_out n_modes_i_in"],
]:
    """Return the forward and inverse flow actions of one section over one length.

    ``U = Z exp(l T) Z^H`` and ``U^-1 = Z exp(-l T) Z^H``: the inverse is the matrix
    function of the negated exponent, never a numerical inverse of ``U``, so the
    interaction picture's backward action is as accurate as its forward one and does
    not inherit the conditioning of a solve (``docs/dev/stepper.md`` section 4: no
    unbounded inverse propagation).

    - ``length``: distance in m, positive for the forward action.
    """
    vectors = flow.vectors[section]
    exponent = float(length) * flow.triangular[section]
    adjoint = np.conj(np.swapaxes(vectors, -1, -2))
    forward = vectors @ _triangular_exponential(exponent) @ adjoint
    inverse = vectors @ _triangular_exponential(-exponent) @ adjoint
    return forward, inverse


def check_flow_conditioning(
    flow: HeldFlow,
    section: int,
    pair: tuple[
        Complex[np.ndarray, "n_active_p n_modes_p_out n_modes_p_in"],
        Complex[np.ndarray, "n_active_q n_modes_q_out n_modes_q_in"],
    ],
    dz: float,
) -> None:
    """Refuse a step whose flow action exceeds the leaf's declared conditioning.

    The certificate row the translation layer declared is
    ``exp(dz_max (mu_max - mu_min))``, a bound on ``norm(U)_2 norm(U^-1)_2``; here the
    two spectral norms are **measured**, from the singular values of the actual
    action, and compared with it. That is the point of a certificate: the row is a
    claim the executor checks rather than a number it trusts, and a step longer than
    the ``dz_max`` the row was declared over is exactly what this catches.

    The inverse is what the interaction picture carries the nonlinear source back by,
    so an unbounded one is an unbounded amplification of that source (the
    ``norm(g(b) - V^-1 N(V b))`` inequality). Local re-anchoring is what keeps it
    bounded at all: the picture is re-anchored at every accepted step, so no inverse
    action is ever taken over more than one of them.

    - ``pair``: the forward and inverse actions over the whole step.
    - ``dz``: length of the step, m, used in the refusal.
    """
    forward, inverse = pair
    if not forward.size:
        return
    measured = np.linalg.norm(forward, ord=2, axis=(-2, -1)) * np.linalg.norm(
        inverse, ord=2, axis=(-2, -1)
    )
    declared = flow.condition_bound[section]
    # A bin whose block is normal has a bound of exactly one and a measured product
    # that is one to within the rounding of two spectral norms; the allowance is that
    # rounding and nothing more, because the bound is a true bound on the exact
    # action and any excess above it is arithmetic.
    allowance = 1.0 + 64.0 * flow.n_modes * float(np.finfo(float).eps)
    excess = measured / (declared * allowance)
    worst = int(np.argmax(excess))
    if excess[worst] > 1.0:
        msg = (
            # Enough digits, and the ratio itself: the two quantities differ by as
            # little as 1e-13 at the step that first exceeds the bound, and `.4g`
            # printed both as the same number with no excess to read.
            "numpy_ref measured a coupled flow conditioning of "
            f"{float(measured[worst]):.12g} over a step of {dz:.4g} m, above the "
            f"{float(declared[worst]):.12g} the certificate declares at "
            f"active bin {worst} of section {section} -- exceeding it by a factor "
            f"{float(excess[worst]):.12g} of the rounding allowance: the inverse "
            "action carries the nonlinear source back by the measured factor. Lower "
            "dz_max, which is what the bound was declared over"
        )
        raise BackendError(msg)


def apply_flow(
    state: Complex[np.ndarray, "n_modes n_pol n_omega"],
    action: Complex[np.ndarray, "n_active_a n_modes_a_out n_modes_a_in"],
    active: Bool[np.ndarray, " n_omega"],
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Apply a per-frequency channel action to the canonical state.

    The identity case is the caller's: a problem with no coupled flow, or the zero
    stage fraction, never reaches here.

    - ``state``: the state, or any array of its layout.
    - Returns: the state with its channel axis mixed on the active bins and untouched
      off them, where the block is zero and the action is the identity.
    """
    carried = state.copy()
    carried[:, :, active] = np.einsum("wij,jpw->ipw", action, state[:, :, active])
    return carried


def residual_action(
    flow: HeldFlow, section: int, weights: Float[np.ndarray, " n_flow_terms"]
) -> Complex[np.ndarray, "n_active_r n_modes_r_out n_modes_r_in"]:
    """Return the explicit residual of one stage, ``R_s(z_1)``, on the active bins.

    ``R_s(z) = D(z) - D(z_s) + H(z) - H(z_s) - C(z)``, evaluated as one linear
    combination of the held terms over the bundle's weights minus the frozen block of
    the **step's** section. The weights are zero outside the section containing the
    stage's own position, so no section's expansion is extrapolated; the frozen block
    is the step's, because the exponential the interaction picture is built on is, and
    freezing is a rearrangement of the same equation whichever block it names.

    - Returns: the residual in 1/m, shape ``(n_active, n_modes, n_modes)``.
    """
    assert flow.residual_terms is not None
    return combine(weights, flow.residual_terms) - flow.frozen[section]


class BlockResidual:
    """``R_s(z) state`` for a held flow's positions (SEP 16 phases 5 and 6).

    A history sample's residual is formed against the section of every trial it serves,
    about six; its action depends only on the section and the sample's flow weights, so
    each pair's action is formed once and kept, the least recently used dropped first:
    each trial's endpoint, at a new position, is used once.

    Parameters
    ----------
    flow : HeldFlow
        The held flow.
    """

    _KEPT = 64
    """Most actions kept: well above a history's samples times the sections it spans."""

    def __init__(self, flow: HeldFlow) -> None:
        """Hold the flow, with no action formed yet."""
        self._flow = flow
        self._actions: collections.OrderedDict[tuple[int, bytes], np.ndarray] = (
            collections.OrderedDict()
        )

    def __call__(
        self,
        section: int,
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
        weights: Float[np.ndarray, " n_flow_terms"],
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return ``R_s(z) state`` for the flow weights of a position.

        Parameters
        ----------
        section : int
            The step's section, whose frozen block the residual is relative to.
        state : numpy.ndarray
            The state the residual acts on.
        weights : numpy.ndarray
            The flow's term weights at the position.

        Returns
        -------
        numpy.ndarray
            The residual's term, zero off the scored active bins.
        """
        key = (section, np.ascontiguousarray(weights).tobytes())
        action = self._actions.get(key)
        if action is None:
            if len(self._actions) >= self._KEPT:
                self._actions.popitem(last=False)
            action = residual_action(self._flow, section, weights)
            self._actions[key] = action
        else:
            self._actions.move_to_end(key)
        flow = self._flow
        return residual_drive(state, action, flow.active, flow.residual_bins)


def residual_drive(
    lab: Complex[np.ndarray, "n_modes n_pol n_omega"],
    action: Complex[np.ndarray, "n_active_d n_modes_d_out n_modes_d_in"],
    active: Bool[np.ndarray, " n_omega"],
    scored: Bool[np.ndarray, " n_omega_scored"],
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return the explicit residual's contribution to one stage's lab-frame source.

    Unlike :func:`apply_flow`, which *transforms* a state and therefore leaves the
    bins it does not cover alone, this **adds a term** and is exactly zero off them:
    the residual is zero at every bin the wall formula was never evaluated at, and a
    copy of the state there would enter the source as a derivative of its own size.

    It is also zero on the active bins the error norm does not score, the spectral
    collar. There the residual can reach a spectral radius the
    embedded pair is unstable at, at steps the in-band error allows, and nothing
    would see the growth; the frozen flow, exact at any step, carries those bins
    alone. The block is diagonal in frequency, so no in-band bin reads them through
    it; only the nonlinear source couples them in, at the size of the collar's
    content.

    - ``action``: ``R_s(z_1)`` on the active bins, 1/m.
    - Returns: ``R_s(z_1) u`` in the state's layout, zero off the scored active bins.
    """
    drive = np.zeros_like(lab)
    drive[:, :, scored] = np.einsum(
        "wij,jpw->ipw", action[scored[active]], lab[:, :, scored]
    )
    return drive
