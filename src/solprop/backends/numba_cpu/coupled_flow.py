# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The coupled flow of a coherent wall or a split block, on the CPU backend.

The second execution path of the reference's ``linear.coupled_flow`` -- the
constant-Schur section (the scalar wall, the TE/TM wall and the split block on the
general route) and the varying ``"frozen_residual"`` flow with its explicit residual in
the Runge-Kutta source. A ``"cf4"`` section is refused by name: the fourth-order
commutator-free flow is not yet implemented on the compiled backends.

**What is host-side and what is compiled.** The Schur decomposition of every
section's frozen block is the reference's own code, run on the host once at
construction (``solprop.backends.numpy_ref.backend._coupled_flow_maps``), and so is
the eigenbasis of each triangular factor
(:func:`solprop.translate.coupled.schur_eigenbasis`), shared with the native backend.
The **actions** ``U = Z exp(l T) Z^H`` and ``U^-1 = Z exp(-l T) Z^H`` of every stage
length, and the certificate's measured ``norm(U) norm(U^-1)``, are formed by compiled
kernels into the held buffers whenever a step's section or length changes, exactly
when the reference's own memo re-forms them:
:func:`form_flow_actions` takes each bin by its eigenbasis, ``W exp(l Lambda) Y``, or
-- where that basis is ill-conditioned, a nearly degenerate nonnormal pair -- by the
reference's own scaling, sixteen-term series and squaring, and
:func:`flow_conditioning` measures both spectral norms by a Jacobi iteration on the
Gram matrix. The decision is the reference's (``_refuse_flow_conditioning``), in its
words. The matrices are no longer the reference's to the last bit; their departure
is bounded in ``tests/flow_parity.py`` (the formation bound) and carried through the
step's parity bound. An adaptive run, which changes its length every step, re-forms
the actions every trial; forming them on the host spent 44 to 68 % of such a step
there, this spends about one per cent.

What is **compiled** is every traversal of the state: the stage combination mixed
by the forward action, the pad onto the transform grid, the explicit residual's
combination over the stage weights and its application, the return to the
interaction variable by the inverse action with both accumulations, and the accept
pass with the error norm's partials. None of them allocates.

**Parity is a rounding statement** (SEP 25). The matrices are within the formation
bound of the reference's, and each pass's sums -- the stage combination over the
previous stages, each action's channel sum, the residual's weighted sum over the terms
-- are this backend's own order, held with the matrices to the parity bound of
``tests/flow_parity.py``, never to the reference's bits. The combination, the
accumulation and the accept pass thread over bins (columns for the accept) on a state
of at least :data:`FLOW_ELEMENTS` elements (:func:`flow_kernels`).
"""

import logging
import math
from typing import Any, no_type_check

import numpy as np
from jaxtyping import Bool, Complex, Float, Int
from numba import prange

from solprop._numba_cache import njit
from solprop.backends.numpy_ref.backend import (
    _TAYLOR_ORDER,
    _TAYLOR_SCALE,
    _coupled_flow_maps,
    _flow_squarings,
    _refuse_flow_conditioning,
)
from solprop.translate.coupled import schur_eigenbasis
from solprop.translate.problem import KernelProblem

__all__ = [
    "FLOW_ELEMENTS",
    "FLOW_LEDGER_ROWS",
    "FlowWorkspace",
    "flow_accept",
    "flow_accumulate",
    "flow_combine",
    "flow_conditioning",
    "flow_kernels",
    "flow_pad",
    "form_flow_actions",
    "make_flow_finish",
    "residual_action",
]

logger = logging.getLogger(__name__)

FLOW_LEDGER_ROWS = (
    "coupled_flow_factors",
    "coupled_flow_residual_terms",
    "coupled_flow_actions",
    "coupled_flow_eigenbasis",
    "coupled_flow_formation",
    "coupled_flow_residual_action",
    "coupled_flow_lab",
    "coupled_flow_drive",
    "coupled_flow_error",
    "coupled_flow_slots",
)
"""The coupled flow's ledger rows, named on every route and zero where it is absent.

The first three are the reference's rows with the reference's sizes: the Schur factors
and frozen blocks of every section, the residual's terms, and one forward and inverse
action per distinct stage fraction. The rest are this backend's own: every section's
eigenbasis with its route and row-sum norm, the formation kernels' scratch and the
measured conditioning, the stage's residual block, the lab-frame stage input the
residual and the pad both read, the stage's source before the inverse action, the
accepted step's mixed error and the two bin-to-block indices, the actions' and the
residual's, which leaves out the bins the error norm does not score.
"""

_JACOBI_SWEEPS = 30
"""Most cyclic Jacobi sweeps a Gram matrix is given before its bound is taken.

The iteration converges quadratically once the off-diagonal is small, and a
Hermitian matrix of a few channels settles in four to six sweeps. Whatever is left
is added to the largest diagonal entry (Weyl's inequality), so the norm returned is
an upper bound whether or not the sweeps ran out: a conditioning is never
under-measured, only, at worst, refused a rounding early.
"""

_UNIT = 2.0**-53
"""The unit roundoff of float64, the Jacobi iteration's stopping scale."""


# --- the kernels ---------------------------------------------------------------


@njit(cache=True, nogil=True)
@no_type_check
def flow_combine(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
    state: Complex[np.ndarray, "n_columns n_omega"],
    stages: Complex[np.ndarray, "n_stages n_columns n_omega"],
    weights: Float[np.ndarray, " n_previous"],
    n_previous: int,
    dz: float,
    propagator: Complex[np.ndarray, "n_columns n_omega"],
    action: Complex[np.ndarray, "n_active n_columns_out n_columns_in"],
    slots: Int[np.ndarray, " n_omega"],
    mix: bool,
    lab: Complex[np.ndarray, "n_columns n_omega"],
) -> None:
    """Form one stage's lab-frame input with the forward flow action.

    The reference's ``exp(dPhi_i) (U_i a + dz U_i sum_j a_ij k_j)``: the action is
    applied to the state and to the stage combination **separately**, as the
    reference's ``_channels`` does, and the combination is summed over the previous
    stages from zero in tableau order. A bin the block does not cover, and every bin
    at the zero fraction, is the diagonal route's own expression.

    Parameters
    ----------
    state : numpy.ndarray
        The state at the start of the step, complex, shape ``(n_columns, n_omega)``.
    stages : numpy.ndarray
        The stage derivatives so far.
    weights : numpy.ndarray
        The tableau row for this stage.
    n_previous : int
        How many stages the row combines.
    dz : float
        The step, m.
    propagator : numpy.ndarray
        ``exp(dPhi_i)`` for this stage.
    action : numpy.ndarray
        ``U_i`` on the active bins, shape ``(n_active, n_columns, n_columns)``.
    slots : numpy.ndarray
        Each bin's row of `action`, or ``-1`` off the block.
    mix : bool
        False at the zero fraction, where the action is the identity.
    lab : numpy.ndarray
        Written with the lab-frame stage input.
    """
    n_columns, n_omega = state.shape
    for j in prange(n_omega):
        slot = slots[j]
        for column in range(n_columns):
            if mix and slot >= 0:
                mixed = 0.0 + 0.0j
                combined = 0.0 + 0.0j
                for inner in range(n_columns):
                    accumulated = 0.0 + 0.0j
                    for previous in range(n_previous):
                        accumulated = (
                            accumulated + weights[previous] * stages[previous, inner, j]
                        )
                    mixed = mixed + action[slot, column, inner] * state[inner, j]
                    combined = combined + action[slot, column, inner] * accumulated
                lab[column, j] = propagator[column, j] * (mixed + dz * combined)
            else:
                accumulated = 0.0 + 0.0j
                for previous in range(n_previous):
                    accumulated = (
                        accumulated + weights[previous] * stages[previous, column, j]
                    )
                lab[column, j] = propagator[column, j] * (
                    state[column, j] + dz * accumulated
                )


@njit(cache=True, nogil=True)
@no_type_check
def flow_pad(
    lab: Complex[np.ndarray, "n_columns n_omega"],
    crop: Int[np.ndarray, " n_omega"],
    fine: Complex[np.ndarray, "n_columns n_omega_fine"],
    scale: float,
) -> None:
    """Scatter the lab-frame input onto a transform grid, zeroing the rest.

    The pad of :func:`solprop.backends.numba_cpu.kernels.make_combine_and_pad` on an
    input the flow has already formed: the whole of `fine` is written, because a
    complex-to-real plan destroys its input while it executes.

    Parameters
    ----------
    lab : numpy.ndarray
        The lab-frame stage input.
    crop : numpy.ndarray
        Where each coarse bin sits on the grid, ``numpy.intp``.
    fine : numpy.ndarray
        Written with the padded spectrum.
    scale : float
        The inverse transform's ``1 / n_to``, or one for the general route's coarse
        buffer.
    """
    n_columns, n_omega = lab.shape
    for column in range(n_columns):
        for j in range(fine.shape[1]):
            fine[column, j] = 0.0 + 0.0j
        for j in range(n_omega):
            fine[column, crop[j]] = lab[column, j] * scale


@njit(cache=True, nogil=True)
@no_type_check
def residual_action(
    weights: Float[np.ndarray, " n_terms"],
    terms: Complex[np.ndarray, "n_terms n_active n_modes_out n_modes_in"],
    frozen: Complex[np.ndarray, "n_active n_modes_out n_modes_in"],
    out: Complex[np.ndarray, "n_active n_modes_out n_modes_in"],
) -> None:
    """Form the explicit residual ``R_s(z_1) = sum_t w_t T_t - G_s`` of one stage.

    The reference's ``combine(weights, terms) - frozen[section]``, summed over the
    terms from zero in their order. A term of zero weight adds an exact zero, since
    every term is finite (``CoupledFlowTables`` refuses anything else), so it is
    skipped: the weights vanish outside the stage's own section.

    Parameters
    ----------
    weights : numpy.ndarray
        The stage bundle's ``flow_weights``, float64, shape ``(n_terms,)``.
    terms : numpy.ndarray
        The held coefficients of ``G(z) - C(z)`` on the active bins, 1/m.
    frozen : numpy.ndarray
        The step's section's frozen block, 1/m.
    out : numpy.ndarray
        Written with the residual, 1/m.
    """
    n_active, n_out, n_in = frozen.shape
    # Term by term over the whole block, so that the terms of zero weight -- every
    # section but the stage's own -- are skipped once rather than per element. Each
    # element still sums its terms from zero in their order.
    for slot in range(n_active):
        for row in range(n_out):
            for column in range(n_in):
                out[slot, row, column] = 0.0 + 0.0j
    for term in range(weights.shape[0]):
        weight = weights[term]
        if weight == 0.0:
            continue
        for slot in range(n_active):
            for row in range(n_out):
                for column in range(n_in):
                    out[slot, row, column] = (
                        out[slot, row, column] + weight * terms[term, slot, row, column]
                    )
    for slot in range(n_active):
        for row in range(n_out):
            for column in range(n_in):
                out[slot, row, column] = (
                    out[slot, row, column] - frozen[slot, row, column]
                )


def make_flow_finish(*, contiguous: bool, absorber: bool, residual: bool) -> Any:
    """Compile the stage's source before the inverse action: crop, frame, residual.

    Parameters
    ----------
    contiguous : bool
        Whether the crop index is the leading block of the transform grid.
    absorber : bool
        Whether the in-stage temporal absorber contributes its term.
    residual : bool
        Whether the flow carries an explicit residual (a ``"frozen_residual"``
        flow); a constant-Schur section's is identically zero and is never formed,
        as in the reference.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        The compiled kernel.
    """

    @no_type_check
    def flow_finish(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        spectrum: Complex[np.ndarray, "n_columns n_omega_fine"],
        crop: Int[np.ndarray, " n_omega"],
        prefactor: Complex[np.ndarray, "n_columns n_omega"],
        window: Float[np.ndarray, " n_omega"],
        inverse: Complex[np.ndarray, "n_columns n_omega"],
        damping: Complex[np.ndarray, "n_columns n_omega"],
        lab: Complex[np.ndarray, "n_columns n_omega"],
        block: Complex[np.ndarray, "n_active n_columns_out n_columns_in"],
        slots: Int[np.ndarray, " n_omega"],
        drive: Complex[np.ndarray, "n_columns n_omega"],
    ) -> None:
        """Write ``exp(-dPhi_i) (N(lab) + R_s lab)``, the source the flow carries back.

        The reference's ``stage_output`` followed by its ``drive + back * residual``,
        in that order: the residual's product is formed over the input channel from
        zero, multiplied by the diagonal inverse propagator and added. Off the block
        the residual is exactly zero and nothing is added.

        Parameters
        ----------
        spectrum : numpy.ndarray
            The forward transform's output.
        crop : numpy.ndarray
            Where each coarse bin sits in it, ``numpy.intp``.
        prefactor : numpy.ndarray
            The UPPE prefactor at this stage's position, times the crop scale.
        window : numpy.ndarray
            The spectral window ``grids.omega_window``.
        inverse : numpy.ndarray
            ``exp(-dPhi_i)``.
        damping : numpy.ndarray
            The in-stage absorber's term when `absorber` was compiled in, a buffer
            never read otherwise.
        lab : numpy.ndarray
            The lab-frame stage input the residual multiplies.
        block : numpy.ndarray
            ``R_s(z_1)`` on the active bins when `residual` was compiled in, a buffer
            never read otherwise.
        slots : numpy.ndarray
            Each bin's row of `block`, or ``-1`` off the bins the residual is applied
            at: the active bins the error norm scores (the reference's
            ``residual_bins``).
        drive : numpy.ndarray
            Written with the source in the interaction variable, before ``U^-1``.
        """
        n_columns, n_omega = drive.shape
        for column in range(n_columns):
            for j in range(n_omega):
                index = j if contiguous else crop[j]
                normalised = prefactor[column, j] * window[j] * spectrum[column, index]
                if absorber:
                    normalised = normalised + damping[column, j]
                value = inverse[column, j] * normalised
                if residual:
                    slot = slots[j]
                    if slot >= 0:
                        product = 0.0 + 0.0j
                        for inner in range(n_columns):
                            product = (
                                product + block[slot, column, inner] * lab[inner, j]
                            )
                        value = value + inverse[column, j] * product
                drive[column, j] = value

    return njit(cache=True, nogil=True)(flow_finish)


@njit(cache=True, nogil=True)
@no_type_check
def flow_accumulate(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
    drive: Complex[np.ndarray, "n_columns n_omega"],
    action: Complex[np.ndarray, "n_active n_columns_out n_columns_in"],
    slots: Int[np.ndarray, " n_omega"],
    mix: bool,
    weight: float,
    error_weight: float,
    stage: Complex[np.ndarray, "n_columns n_omega"],
    solution: Complex[np.ndarray, "n_columns n_omega"],
    error: Complex[np.ndarray, "n_columns n_omega"],
) -> None:
    """Carry a stage's source back by ``U_i^-1`` and accumulate it into both sums.

    Parameters
    ----------
    drive : numpy.ndarray
        The source from the finish kernel.
    action : numpy.ndarray
        ``U_i^-1`` on the active bins.
    slots : numpy.ndarray
        Each bin's row of `action`, or ``-1``.
    mix : bool
        False at the zero fraction, where the action is the identity.
    weight : float
        The tableau's solution weight for this stage.
    error_weight : float
        The tableau's embedded-difference weight.
    stage : numpy.ndarray
        Written with the stage derivative.
    solution : numpy.ndarray
        Accumulated into.
    error : numpy.ndarray
        Accumulated into.
    """
    n_columns, n_omega = drive.shape
    for j in prange(n_omega):
        slot = slots[j]
        for column in range(n_columns):
            if mix and slot >= 0:
                value = 0.0 + 0.0j
                for inner in range(n_columns):
                    value = value + action[slot, column, inner] * drive[inner, j]
            else:
                value = drive[column, j]
            stage[column, j] = value
            solution[column, j] += weight * value
            error[column, j] += error_weight * value


@njit(cache=True, nogil=True)
@no_type_check
def flow_accept(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
    state: Complex[np.ndarray, "n_columns n_omega"],
    propagator: Complex[np.ndarray, "n_columns n_omega"],
    dz: float,
    solution: Complex[np.ndarray, "n_columns n_omega"],
    error: Complex[np.ndarray, "n_columns n_omega"],
    action: Complex[np.ndarray, "n_active n_columns_out n_columns_in"],
    slots: Int[np.ndarray, " n_omega"],
    mask: Bool[np.ndarray, " n_omega"],
    advanced: Complex[np.ndarray, "n_columns n_omega"],
    mixed_error: Complex[np.ndarray, "n_columns n_omega"],
    partials: Float[np.ndarray, "three n_columns"],
) -> None:
    """Take the accept pass through the whole step's flow, with the norm's partials.

    The reference's ``exp(dPhi) (U a + dz U s)`` and the norm of
    ``exp(dPhi) (dz U e)``: the action is applied to the state, the solution sum and
    the error sum separately, and the error norm's ``|y|^2`` is of the state **before**
    the action, as the reference's ``error_norm`` takes it. The mixed error is kept,
    because a weighted controller's second traversal reads it.

    Parameters
    ----------
    state : numpy.ndarray
        The state at the start of the step.
    propagator : numpy.ndarray
        ``exp(dPhi)`` over the whole step.
    dz : float
        The step, m.
    solution : numpy.ndarray
        The weighted stage sum of the solution.
    error : numpy.ndarray
        The weighted stage sum of the embedded difference.
    action : numpy.ndarray
        ``U`` over the whole step, on the active bins.
    slots : numpy.ndarray
        Each bin's row of `action`, or ``-1``.
    mask : numpy.ndarray
        The in-band mask the norm is taken over.
    advanced : numpy.ndarray
        Written with the state at the end of the step.
    mixed_error : numpy.ndarray
        Written with ``U e``.
    partials : numpy.ndarray
        Written with the three sums per column, shape ``(3, n_columns)``.
    """
    n_columns, n_omega = state.shape
    for column in prange(n_columns):
        old = 0.0
        new = 0.0
        residual = 0.0
        for j in range(n_omega):
            slot = slots[j]
            if slot >= 0:
                mixed = 0.0 + 0.0j
                summed = 0.0 + 0.0j
                spread = 0.0 + 0.0j
                for inner in range(n_columns):
                    entry = action[slot, column, inner]
                    mixed = mixed + entry * state[inner, j]
                    summed = summed + entry * solution[inner, j]
                    spread = spread + entry * error[inner, j]
            else:
                mixed = state[column, j]
                summed = solution[column, j]
                spread = error[column, j]
            end = propagator[column, j] * (mixed + dz * summed)
            advanced[column, j] = end
            mixed_error[column, j] = spread
            if mask[j]:
                before = state[column, j]
                scaled = propagator[column, j] * (dz * spread)
                old += before.real * before.real + before.imag * before.imag
                new += end.real * end.real + end.imag * end.imag
                residual += scaled.real * scaled.real + scaled.imag * scaled.imag
        partials[0, column] = old
        partials[1, column] = new
        partials[2, column] = residual


FLOW_ELEMENTS = 1 << 15
"""State size from which the flow's stage combination, accumulation and accept thread.

Every bin of the combination and the accumulation is independent, and the accept pass
sums each column on its own, so the parallel variants (a ``prange`` over bins, and over
columns for the accept) agree with the serial ones bin by bin. Measured on six threads
of an Apple-silicon laptop under load, microseconds serial -> threaded: 4 channels x
8192 bins, combination 1121 -> 192, accumulation 330 -> 99, accept 352 -> 123; 8 x
16384, 10025 -> 2671, 1611 -> 351, 2862 -> 831; 2 x 4096 the accumulation and the
accept were slower threaded (18 -> 52, 43 -> 62).
"""


def _parallel(kernel: Any) -> Any:
    """Compile `kernel`'s body parallel, never disk-cached (``backends.md`` 3.2).

    Parameters
    ----------
    kernel : numba dispatcher or function
        The serial kernel, or its plain function under ``NUMBA_DISABLE_JIT``.

    Returns
    -------
    numba dispatcher
        The parallel compilation of the same body.
    """
    # The plain function under ``NUMBA_DISABLE_JIT``, which has no ``py_func``.
    body = getattr(kernel, "py_func", kernel)
    return njit(cache=False, nogil=True, parallel=True)(body)


_PARALLEL_FLOW = (
    _parallel(flow_combine),
    _parallel(flow_accumulate),
    _parallel(flow_accept),
)


def flow_kernels(threads: int, n_columns: int, n_omega: int) -> tuple[Any, Any, Any]:
    """Choose the serial or the threaded combination, accumulation and accept pass.

    Parameters
    ----------
    threads : int
        Threads the kernel pool has.
    n_columns : int
        Columns of the state.
    n_omega : int
        Bins per column.

    Returns
    -------
    tuple
        :func:`flow_combine`, :func:`flow_accumulate` and :func:`flow_accept`, or
        their parallel variants for a state of at least :data:`FLOW_ELEMENTS`.

    Examples
    --------
    >>> from solprop.backends.numba_cpu import coupled_flow
    >>> coupled_flow.flow_kernels(1, 4, 8192)[0] is coupled_flow.flow_combine
    True
    """
    if threads > 1 and n_columns * n_omega >= FLOW_ELEMENTS:
        return _PARALLEL_FLOW
    return flow_combine, flow_accumulate, flow_accept


# --- the matrix functions --------------------------------------------------------


@njit(cache=True, nogil=True)
@no_type_check
def _product(
    first: Complex[np.ndarray, "n_modes_a n_modes_a2"],
    second: Complex[np.ndarray, "n_modes_b n_modes_b2"],
    out: Complex[np.ndarray, "n_modes_o n_modes_o2"],
    adjoint: bool,
) -> None:
    """Write ``first @ second``, or ``first @ second^H``, each sum from zero.

    Parameters
    ----------
    first, second : numpy.ndarray
        Square matrices.
    out : numpy.ndarray
        Written with the product; distinct from both.
    adjoint : bool
        Whether `second` enters conjugate-transposed.
    """
    n_modes = first.shape[0]
    for row in range(n_modes):
        for column in range(n_modes):
            total = 0.0 + 0.0j
            for inner in range(n_modes):
                if adjoint:
                    total = (
                        total + first[row, inner] * second[column, inner].conjugate()
                    )
                else:
                    total = total + first[row, inner] * second[inner, column]
            out[row, column] = total


@njit(cache=True, nogil=True)
@no_type_check
def _squared_series_action(
    length: float,
    vectors: Complex[np.ndarray, "n_modes_z n_modes_z2"],
    triangular: Complex[np.ndarray, "n_modes_t n_modes_t2"],
    out: Complex[np.ndarray, "n_modes_o n_modes_o2"],
    work: Complex[np.ndarray, "three n_modes_w n_modes_w2"],
) -> None:
    """Write ``Z exp(l T) Z^H`` of one bin by the reference's scaling and squaring.

    ``_triangular_exponential`` and ``_flow_action`` on one block: scaled below
    :data:`solprop.backends.numpy_ref.backend._TAYLOR_SCALE` in row-sum norm, the
    sixteen-term series by Horner's rule from the identity, the squarings, then the
    back-transform ``(Z F) Z^H``, each product summed from zero over the inner index.
    The number of squarings is the bin's own; the reference takes the stack's largest,
    which is at least as many, and a length needing more than the admitted limit was
    refused before the kernel ran.

    Parameters
    ----------
    length : float
        The signed length ``l``, m.
    vectors : numpy.ndarray
        The bin's unitary Schur factor ``Z``.
    triangular : numpy.ndarray
        The bin's triangular factor ``T``, 1/m.
    out : numpy.ndarray
        Written with the action.
    work : numpy.ndarray
        Three ``n x n`` scratch blocks.
    """
    n_modes = triangular.shape[0]
    scaled = work[0]
    series = work[1]
    product = work[2]
    largest = 0.0
    for row in range(n_modes):
        total = 0.0
        for column in range(n_modes):
            total += abs(length * triangular[row, column])
        largest = max(largest, total)
    squarings = 0
    if largest > _TAYLOR_SCALE:
        squarings = math.ceil(math.log2(largest / _TAYLOR_SCALE))
    divisor = float(2**squarings)
    for row in range(n_modes):
        for column in range(n_modes):
            scaled[row, column] = (length * triangular[row, column]) / divisor
            series[row, column] = 1.0 + 0.0j if row == column else 0.0 + 0.0j
    for term in range(_TAYLOR_ORDER, 0, -1):
        _product(scaled, series, product, adjoint=False)
        for row in range(n_modes):
            for column in range(n_modes):
                diagonal = 1.0 + 0.0j if row == column else 0.0 + 0.0j
                series[row, column] = diagonal + product[row, column] / term
    for _ in range(squarings):
        _product(series, series, product, adjoint=False)
        series[:, :] = product
    _product(vectors, series, product, adjoint=False)
    _product(product, vectors, out, adjoint=True)


@njit(cache=True, nogil=True)
@no_type_check
def form_flow_actions(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
    fractions: Float[np.ndarray, " n_fractions"],
    dz: float,
    left: Complex[np.ndarray, "n_active n_modes_out n_modes_in"],
    right: Complex[np.ndarray, "n_active_r n_modes_r_out n_modes_r_in"],
    eigenvalues: Complex[np.ndarray, "n_active_e n_modes_e"],
    route: Bool[np.ndarray, " n_active_m"],
    vectors: Complex[np.ndarray, "n_active_z n_modes_z_out n_modes_z_in"],
    triangular: Complex[np.ndarray, "n_active_t n_modes_t_out n_modes_t_in"],
    forward: Complex[np.ndarray, "n_fractions_f n_active_f n_modes_f_out n_modes_f_in"],
    inverse: Complex[np.ndarray, "n_fractions_i n_active_i n_modes_i_out n_modes_i_in"],
    exponents: Complex[np.ndarray, "two n_modes_x"],
    work: Complex[np.ndarray, "three n_modes_w n_modes_w2"],
) -> None:
    """Form the forward and inverse actions of one section at every stage fraction.

    On a bin whose eigenbasis is well conditioned (``route``),
    ``U(l) = W exp(l Lambda) Y`` and ``U(-l) = W exp(-l Lambda) Y``: each entry
    ``sum_k (W_ik e_k) Y_kj`` from zero in increasing ``k``, with ``e_k`` from one
    real exponential and one cosine and sine of ``l lambda_k``, shared by the pair.
    Elsewhere the reference's scaling and squaring (``_squared_series_action``).
    A zero fraction is the identity and is left untouched, as the backend never
    applies it.

    Parameters
    ----------
    fractions : numpy.ndarray
        The distinct stage fractions, dimensionless.
    dz : float
        The step, m; the lengths are ``fraction * dz``, as the reference forms them.
    left, right : numpy.ndarray
        The section's ``W`` and ``Y`` per active bin.
    eigenvalues : numpy.ndarray
        The section's ``lambda`` per active bin, 1/m.
    route : numpy.ndarray
        True where a bin takes its eigenbasis.
    vectors, triangular : numpy.ndarray
        The section's Schur factors per active bin, for the other bins.
    forward, inverse : numpy.ndarray
        Written with the actions, one slot per fraction.
    exponents : numpy.ndarray
        Scratch for one bin's ``exp(+-l lambda)``.
    work : numpy.ndarray
        Scratch for one bin's scaling and squaring.
    """
    n_active, n_modes, _ = left.shape
    for bin_index in range(n_active):
        for slot in range(fractions.shape[0]):
            fraction = fractions[slot]
            if fraction == 0.0:
                continue
            length = fraction * dz
            if not route[bin_index]:
                _squared_series_action(
                    length,
                    vectors[bin_index],
                    triangular[bin_index],
                    forward[slot, bin_index],
                    work,
                )
                _squared_series_action(
                    -length,
                    vectors[bin_index],
                    triangular[bin_index],
                    inverse[slot, bin_index],
                    work,
                )
                continue
            for mode in range(n_modes):
                value = eigenvalues[bin_index, mode]
                real = length * value.real
                imaginary = length * value.imag
                cosine = math.cos(imaginary)
                sine = math.sin(imaginary)
                grow = math.exp(real)
                decay = math.exp(-real)
                exponents[0, mode] = complex(grow * cosine, grow * sine)
                exponents[1, mode] = complex(decay * cosine, -(decay * sine))
            for row in range(n_modes):
                for column in range(n_modes):
                    ahead = 0.0 + 0.0j
                    back = 0.0 + 0.0j
                    for inner in range(n_modes):
                        entry = left[bin_index, row, inner]
                        weight = right[bin_index, inner, column]
                        ahead = ahead + (entry * exponents[0, inner]) * weight
                        back = back + (entry * exponents[1, inner]) * weight
                    forward[slot, bin_index, row, column] = ahead
                    inverse[slot, bin_index, row, column] = back


@njit(cache=True, nogil=True)
@no_type_check
def _gram_extent(
    gram: Complex[np.ndarray, "n_modes_g n_modes_g2"],
) -> tuple[float, float]:
    """Return a Gram matrix's largest diagonal entry and its squared upper part.

    Parameters
    ----------
    gram : numpy.ndarray
        The Hermitian matrix.

    Returns
    -------
    tuple of float
        The largest real diagonal entry, and the sum of the squared moduli above the
        diagonal.
    """
    largest = 0.0
    off = 0.0
    for row in range(gram.shape[0]):
        largest = max(largest, gram[row, row].real)
        for column in range(row + 1, gram.shape[0]):
            value = gram[row, column]
            off += value.real * value.real + value.imag * value.imag
    return largest, off


@njit(cache=True, nogil=True)
@no_type_check
def _spectral_norm(
    matrix: Complex[np.ndarray, "n_modes_a n_modes_b"],
    gram: Complex[np.ndarray, "n_modes_g n_modes_g2"],
) -> float:
    """Return an upper bound on ``norm(matrix)_2`` to the rounding of its Gram matrix.

    The largest eigenvalue of ``M^H M`` by cyclic Jacobi rotations, each a phase
    that makes the pivot real followed by the symmetric rotation that zeroes it
    (Golub and Van Loan, section 8.5), stopped when the off-diagonal part is below the
    unit roundoff of the largest diagonal entry. Weyl's inequality puts the largest
    eigenvalue within the off-diagonal's Frobenius norm of the largest diagonal
    entry, and that norm is added, so what returns bounds the measured norm from
    above.

    Parameters
    ----------
    matrix : numpy.ndarray
        The square matrix, complex.
    gram : numpy.ndarray
        Scratch of the same shape.

    Returns
    -------
    float
        The bound on the spectral norm.
    """
    n_modes = matrix.shape[0]
    for row in range(n_modes):
        for column in range(n_modes):
            total = 0.0 + 0.0j
            for inner in range(n_modes):
                total = total + matrix[inner, row].conjugate() * matrix[inner, column]
            gram[row, column] = total
    sweeps = 0
    largest, off = _gram_extent(gram)
    while sweeps < _JACOBI_SWEEPS and off > (_UNIT * largest) ** 2:
        for first in range(n_modes):
            for second in range(first + 1, n_modes):
                pivot = gram[first, second]
                size = abs(pivot)
                if size == 0.0:
                    continue
                low = gram[first, first].real
                high = gram[second, second].real
                phase = pivot / size
                spread = (high - low) / (2.0 * size)
                tangent = 1.0 / (abs(spread) + math.sqrt(1.0 + spread * spread))
                if spread < 0.0:
                    tangent = -tangent
                cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
                sine = tangent * cosine
                back = phase.conjugate()
                for index in range(n_modes):
                    one = gram[index, first]
                    two = gram[index, second]
                    gram[index, first] = cosine * one - sine * back * two
                    gram[index, second] = sine * one + cosine * back * two
                for index in range(n_modes):
                    one = gram[first, index]
                    two = gram[second, index]
                    gram[first, index] = cosine * one - sine * phase * two
                    gram[second, index] = sine * one + cosine * phase * two
                gram[first, second] = 0.0 + 0.0j
                gram[second, first] = 0.0 + 0.0j
                gram[first, first] = low - tangent * size
                gram[second, second] = high + tangent * size
        sweeps += 1
        largest, off = _gram_extent(gram)
    return math.sqrt(largest + math.sqrt(2.0 * off))


@njit(cache=True, nogil=True)
@no_type_check
def flow_conditioning(
    forward: Complex[np.ndarray, "n_active n_modes_out n_modes_in"],
    inverse: Complex[np.ndarray, "n_active_i n_modes_i_out n_modes_i_in"],
    gram: Complex[np.ndarray, "n_modes_g n_modes_g2"],
    measured: Float[np.ndarray, " n_active_m"],
) -> None:
    """Measure ``norm(U)_2 norm(U^-1)_2`` of the whole step's actions per bin.

    The certificate's measurement (``_check_flow_conditioning``), each spectral norm
    by ``_spectral_norm``, within a few ``n u`` of the singular value the
    reference takes and never below it by more than the rotations' rounding; the
    comparison with the declared row is the reference's own.

    Parameters
    ----------
    forward, inverse : numpy.ndarray
        The whole step's actions per active bin.
    gram : numpy.ndarray
        Scratch, one ``n x n`` block.
    measured : numpy.ndarray
        Written with the product per bin.
    """
    for bin_index in range(forward.shape[0]):
        measured[bin_index] = _spectral_norm(forward[bin_index], gram) * _spectral_norm(
            inverse[bin_index], gram
        )


# --- the workspace -------------------------------------------------------------


class FlowWorkspace:
    """The coupled flow's held factors, actions and buffers, allocated once.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A problem the backend has admitted, carrying ``linear.coupled_flow``.
    fractions : numpy.ndarray
        The step's distinct stage fractions in increasing order, the backend's own
        propagator slots; an action pair is held per fraction.
    contiguous : bool
        Whether the crop index is the leading block of the transform grid.
    absorber : bool
        Whether the in-stage absorber contributes a term.
    """

    def __init__(
        self,
        problem: KernelProblem,
        fractions: Float[np.ndarray, " n_fractions"],
        *,
        contiguous: bool,
        absorber: bool,
    ) -> None:
        held = _coupled_flow_maps(problem)
        assert held is not None
        self.held = held
        self.fractions = fractions
        _, n_modes, n_pol, n_omega = problem.shape
        n_columns = n_modes * n_pol
        n_active = int(np.count_nonzero(held.active))
        slots = np.full(n_omega, -1, dtype=np.intp)
        slots[held.active] = np.arange(n_active, dtype=np.intp)
        self.slots = slots
        # The residual is applied only on the bins the error norm scores; in the
        # collar the frozen flow alone carries the state, as in the reference
        # (`_residual_drive`).
        self.residual_slots = np.where(held.residual_bins, slots, -1)
        block = (n_active, held.n_modes, held.n_modes)
        self.forward = np.zeros((fractions.size, *block), dtype=np.complex128)
        self.inverse = np.zeros_like(self.forward)
        bases = [
            schur_eigenbasis(vectors, triangular)
            for vectors, triangular in zip(held.vectors, held.triangular, strict=True)
        ]
        self.left = np.stack([basis[0] for basis in bases])
        self.right = np.stack([basis[1] for basis in bases])
        self.eigenvalues = np.stack([basis[2] for basis in bases])
        self.route = np.stack([basis[3] for basis in bases])
        # The reference's squaring count reads the stack's largest row-sum norm of
        # `l T`; it is `l` times this, to the rounding of one product.
        self.row_sum = np.max(np.sum(abs(held.triangular), axis=-1), axis=(1, 2))
        self._exponents = np.zeros((2, held.n_modes), dtype=np.complex128)
        self._work = np.zeros((3, held.n_modes, held.n_modes), dtype=np.complex128)
        self._gram = np.zeros((held.n_modes, held.n_modes), dtype=np.complex128)
        self.measured = np.zeros(n_active)
        self._whole = int(np.flatnonzero(fractions == 1.0)[0])
        self.residual = held.residual_terms is not None
        self.block = np.zeros(block if self.residual else (0, 0, 0), dtype=complex)
        self._terms = (
            held.residual_terms
            if held.residual_terms is not None
            else np.zeros((0, *block), dtype=np.complex128)
        )
        self.lab = np.zeros((n_columns, n_omega), dtype=np.complex128)
        self.drive = np.zeros_like(self.lab)
        self.error = np.zeros_like(self.lab)
        self.section = -1
        self._key: tuple[int, float] | None = None
        self.finish = make_flow_finish(
            contiguous=contiguous, absorber=absorber, residual=self.residual
        )

    def prepare(self, section: int, dz: float) -> None:
        """Hold the forward and inverse actions of one step, re-formed on a change.

        The reference's memo (``NumpyRefBackend._flow_actions``): the generator is
        frozen across the step, so an action depends on the stage fraction only
        through the length ``c_i dz``, and the pairs survive between steps for the one
        section and length last seen. A step that crosses into the next section
        re-forms them, which is the cache invalidation that makes a section edge a
        kink of the flow. The whole step's pair is checked against the certificate's
        declared conditioning, which bounds every shorter one. The pairs are formed
        by :func:`form_flow_actions` and measured by :func:`flow_conditioning` into
        the held buffers, so a new length allocates nothing either.

        Parameters
        ----------
        section : int
            The section the step's anchor lies in, from the step bundle's
            ``flow_section``.
        dz : float
            The trial step, m.

        Raises
        ------
        BackendError
            If a pair needs more squarings than admitted, or the whole step's
            measured conditioning exceeds the declared bound, in the reference's
            words under this backend's name.
        """
        key = (section, dz)
        if key == self._key:
            return
        # Forgotten first: a refusal below leaves no half-formed pair behind a key.
        self._key = None
        for fraction in self.fractions:
            if fraction != 0.0:
                _flow_squarings(
                    float(fraction) * dz * float(self.row_sum[section]),
                    backend="numba_cpu",
                )
        form_flow_actions(
            self.fractions,
            dz,
            self.left[section],
            self.right[section],
            self.eigenvalues[section],
            self.route[section],
            self.held.vectors[section],
            self.held.triangular[section],
            self.forward,
            self.inverse,
            self._exponents,
            self._work,
        )
        flow_conditioning(
            self.forward[self._whole],
            self.inverse[self._whole],
            self._gram,
            self.measured,
        )
        _refuse_flow_conditioning(
            self.held, section, self.measured, dz, backend="numba_cpu"
        )
        self.section = section
        self._key = key

    def form_residual(self, weights: Float[np.ndarray, " n_terms"] | None) -> None:
        """Form the stage's explicit residual into ``block``, when there is one.

        A constant-Schur section has none and forms nothing, as in the reference,
        whose ``_residual`` returns ``None`` there rather than multiplying by zero.

        Parameters
        ----------
        weights : numpy.ndarray or None
            The stage bundle's ``flow_weights``, which every stage of a problem
            carrying a flow has.
        """
        if not self.residual:
            return
        assert weights is not None
        residual_action(
            weights, self._terms, self.held.frozen[self.section], self.block
        )

    def memory_ledger(self) -> dict[str, int]:
        """Return the size of every buffer the flow holds, in bytes.

        Returns
        -------
        dict
            Every row of :data:`FLOW_LEDGER_ROWS`.
        """
        held = self.held
        return {
            "coupled_flow_factors": held.vectors.nbytes
            + held.triangular.nbytes
            + held.frozen.nbytes,
            "coupled_flow_residual_terms": 0
            if held.residual_terms is None
            else held.residual_terms.nbytes,
            "coupled_flow_actions": self.forward.nbytes + self.inverse.nbytes,
            "coupled_flow_eigenbasis": self.left.nbytes
            + self.right.nbytes
            + self.eigenvalues.nbytes
            + self.route.nbytes
            + self.row_sum.nbytes,
            "coupled_flow_formation": self._exponents.nbytes
            + self._work.nbytes
            + self._gram.nbytes
            + self.measured.nbytes,
            "coupled_flow_residual_action": self.block.nbytes,
            "coupled_flow_lab": self.lab.nbytes,
            "coupled_flow_drive": self.drive.nbytes,
            "coupled_flow_error": self.error.nbytes,
            "coupled_flow_slots": self.slots.nbytes + self.residual_slots.nbytes,
        }
