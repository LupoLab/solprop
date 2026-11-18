# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The field arithmetic of the reference's Adams steps (SEP 13, SEP 14, SEP 16).

The state-sized passes an Adams trial and its acceptance take, given the history's
rows and the trial's coefficients (:mod:`~solprop.backends.numpy_ref.adams_tables`).
:class:`AdamsKernels` is the interaction formulation's: the predictor, the corrector
with every scored defect, and the history's transport to the new anchor.
:class:`ExponentialKernels` is the exponential formulation's: the moments
``k! phi_{k+1}(x)`` and the flow ``exp(x)``, the passes in the monomial basis of the
interpolant, and a varying operator's carried compensation. Beside a coupled flow the
flow and moments are matrices on the block's bins: :class:`CoupledBlock` is a step's
frozen block and :func:`coupled_moments` its augmented-exponential moments
(:class:`MatrixMoments`). :func:`weighted_sum` is the accumulation every pass shares.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Complex
from scipy.linalg import expm

from solprop.backends.exponential import phi_functions
from solprop.backends.numpy_ref.passes.stepping import accumulate

__all__ = [
    "AdamsKernels",
    "CoupledBlock",
    "ExponentialKernels",
    "MatrixMoments",
    "coupled_moments",
    "weighted_sum",
]

_State = Complex[np.ndarray, "n_modes n_pol n_omega"]
_StoreRows = Complex[np.ndarray, "n_slots n_modes n_pol n_omega"]
_Shift = Complex[np.ndarray, "1 1 #n_omega"]
"""A carrier frame's shift (SEP 35), shape ``(1, 1, 1)``."""
_Factors = Complex[np.ndarray, "n_used 1 1 #n_omega"]
"""A carrier frame's rotation of each history row a trial reads, in its slots' order."""
_MomentRows = Complex[np.ndarray, "n_k n_modes n_pol n_omega"]
"""A trial's moments ``M_k`` per element, along a leading axis."""


def weighted_sum(weights: np.ndarray, values: list[_State]) -> _State:
    """Return ``sum_j w_j X_j``, accumulated in the order of the rows.

    Parameters
    ----------
    weights : numpy.ndarray
        One real weight per row.
    values : list of numpy.ndarray
        The rows, each in the state's layout.

    Returns
    -------
    numpy.ndarray
        The weighted sum, a new array.
    """
    total = np.zeros_like(values[0])
    for weight, value in zip(weights, values, strict=True):
        total = accumulate(total, value, float(weight))
    return total


class AdamsKernels:
    """The field arithmetic of a fused Adams step, in NumPy (SEP 14 phase 2).

    Three passes serve a step of the one-order-higher PEC family: the predictor's
    lab-frame field, the corrected field with the defects of the current and the
    neighbouring orders, and the history's transport to the new anchor. The history is
    a store of rows, newest first by ``slots``; the weights are
    :mod:`~solprop.backends.numpy_ref.adams_tables`'.
    """

    def predict(  # noqa: PLR0917 - called positionally
        self,
        state: _State,
        propagator: _State,
        store: Complex[np.ndarray, "n_slots n_modes n_pol n_omega"],
        slots: np.ndarray,
        weights: np.ndarray,
        dz: float,
    ) -> _State:
        """Return the predictor's lab-frame field ``U (a_n + dz sum_t w_t g_t)``.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state at the anchor.
        propagator : numpy.ndarray
            ``U(z_n + dz, z_n)``.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows the predictor combines, newest first, ``int64``.
        weights : numpy.ndarray
            One integrated weight per row.
        dz : float
            The step, m.

        Returns
        -------
        numpy.ndarray
            The predicted field at the end of the step.
        """
        total = weighted_sum(weights, [store[slot] for slot in slots])
        return propagator * (state + dz * total)

    def correct(  # noqa: PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        propagator: _State,
        source: _State,
        store: Complex[np.ndarray, "n_slots n_modes n_pol n_omega"],
        slots: np.ndarray,
        corrector: np.ndarray,
        rows: np.ndarray,
        dz: float,
        mask: np.ndarray,
        divisor: np.ndarray,
    ) -> tuple[_State, np.ndarray]:
        """Return the corrected lab-frame field and the scaled norms of the defects.

        With ``X = (source, store[slots[0]], ...)``, the field is
        ``U (a_n + dz sum_t c_t X_t)`` and each norm is the weak norm of
        ``U dz sum_t rows[r, t] X_t`` over the in-band bins.

        Parameters
        ----------
        state, propagator : numpy.ndarray
            As for :meth:`predict`.
        source : numpy.ndarray
            The source at the predicted field, in the anchor's frame.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows after the source, newest first, ``int64``.
        corrector : numpy.ndarray
            The corrector's weights over ``X``, zero past its own nodes.
        rows : numpy.ndarray
            One defect's weights over ``X`` per row, shape ``(n_rows, 1 + n_slots)``.
        dz : float
            The step, m.
        mask : numpy.ndarray
            The in-band bins, Boolean.
        divisor : numpy.ndarray
            The norm's scale per bin.

        Returns
        -------
        tuple
            The corrected field and one norm per row.
        """
        values = [source, *(store[slot] for slot in slots)]
        total = weighted_sum(corrector, values)
        norms = np.empty(rows.shape[0])
        for index, row in enumerate(rows):
            defect = weighted_sum(row, values)
            scaled = (propagator * (dz * defect))[..., mask] / divisor[mask]
            norms[index] = float(np.linalg.norm(scaled))
        return propagator * (state + dz * total), norms

    def transport(
        self,
        store: Complex[np.ndarray, "n_slots n_modes n_pol n_omega"],
        slots: np.ndarray,
        propagator: _State,
        new_slot: int,
        source: _State,
    ) -> None:
        """Re-anchor the history at the step's end and store the new derivative.

        Parameters
        ----------
        store : numpy.ndarray
            The history rows, updated in place.
        slots : numpy.ndarray
            The rows of the kept history, ``int64``.
        propagator : numpy.ndarray
            ``U(z_n + dz, z_n)``.
        new_slot : int
            The free row the new derivative goes to.
        source : numpy.ndarray
            The new derivative in the step's anchor frame.
        """
        for slot in slots:
            store[slot] *= propagator
        np.multiply(propagator, source, out=store[new_slot])


@dataclass(frozen=True)
class CoupledBlock:
    """A step's frozen coupled block on the bins that carry one (SEP 16 phase 5).

    Beside a coupled flow the cumulative diagonal ``Phi`` is a multiple of the identity
    in the channel axis (``docs/dev/kernel-contract.md``, ``coupled_flow``), so the
    step's reference is ``B = (Delta Phi / h) I + G_s`` with ``G_s = Z T Z^H`` the
    frozen block of the section the step is anchored in, and the two parts commute.

    Attributes
    ----------
    active : numpy.ndarray
        Which bins carry a block, bool, shape ``(n_omega,)``.
    vectors, triangular : numpy.ndarray
        ``Z`` and ``T`` of the section's block, 1/m for ``T``, shape
        ``(n_active, n_modes, n_modes)``.
    section : int
        The section the step is anchored in.
    end_weights : numpy.ndarray or None
        The flow's term weights at the step's end, read on the step's side of a
        section edge (``StageScalars.flow_weights``); ``None`` without a residual.
    residual : callable or None
        ``(state, weights) -> R_s(z) state`` for a frozen-residual flow: the
        generator at the position the weights belong to less this section's frozen
        block, on the active bins the error norm scores and zero elsewhere, as the
        Runge-Kutta stage applies it; ``None`` for a constant-Schur flow.
    """

    active: np.ndarray
    vectors: np.ndarray
    triangular: np.ndarray
    section: int
    end_weights: np.ndarray | None = None
    residual: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None


@dataclass(frozen=True)
class MatrixMoments:
    """A coupled trial's flow and moments: scalar per element, matrices on a block.

    Off the block's bins the step's flow and moments are the diagonal ones, per
    element; on them each is a matrix in the channel axis, applied to the channel
    vector of every polarisation. ``matrices`` and ``flow_matrix`` are
    laid out by bin, ``(..., n_bins, n_modes, n_modes)``; the products read the same
    matrices with the bin axis last, as the state holds it.

    Attributes
    ----------
    scalar, flow_scalar : numpy.ndarray
        The diagonal moments and flow, per element.
    matrices, flow_matrix : numpy.ndarray
        The block's moments and flow on its bins.
    active : numpy.ndarray
        Which bins carry the block, bool.
    """

    scalar: np.ndarray
    flow_scalar: np.ndarray
    matrices: np.ndarray
    flow_matrix: np.ndarray
    active: np.ndarray
    _columns: np.ndarray = field(init=False, repr=False, compare=False)
    _flow_columns: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Hold the matrices with the bin axis last, for the products."""
        object.__setattr__(
            self, "_columns", np.ascontiguousarray(np.moveaxis(self.matrices, -3, -1))
        )
        object.__setattr__(
            self,
            "_flow_columns",
            np.ascontiguousarray(np.moveaxis(self.flow_matrix, -3, -1)),
        )

    def _product(
        self, columns: np.ndarray, scalar: np.ndarray, values: _State
    ) -> _State:
        """Return the scalar product off the block and the matrix product on it.

        - ``scalar``: the diagonal factor, in the state's layout.
        - ``values``: the array, in the state's layout.
        """
        out = scalar * values
        selected = values[..., self.active]
        block = columns[:, 0][:, np.newaxis] * selected[np.newaxis, 0]
        for column in range(1, columns.shape[1]):
            block += columns[:, column][:, np.newaxis] * selected[np.newaxis, column]
        out[..., self.active] = block
        return out

    def apply(self, k: int, values: _State) -> _State:
        """Return ``M_k`` applied to a state-shaped array.

        Parameters
        ----------
        k : int
            The moment.
        values : numpy.ndarray
            The array, in the state's layout.

        Returns
        -------
        numpy.ndarray
            ``M_k values``.
        """
        columns = self._columns
        return self._product(columns[k], self.scalar[k], values)

    def flow(self, values: _State) -> _State:
        """Return the step's flow ``exp(h B)`` applied to a state-shaped array.

        Parameters
        ----------
        values : numpy.ndarray
            The array, in the state's layout.

        Returns
        -------
        numpy.ndarray
            ``exp(h B) values``.
        """
        columns = self._flow_columns
        return self._product(columns, self.flow_scalar, values)


def coupled_moments(
    exponent: _State,
    count: int,
    block: CoupledBlock,
    dz: float,
    *,
    carrier: _Shift | None = None,
) -> MatrixMoments:
    """Return a coupled trial's flow and moments ``k! phi_{k+1}(h B)``, ``k < count``.

    The reference's evaluation (SEP 16 section 4.5): on each bin of the block the
    functions of ``X = Delta Phi I + h T``, upper triangular, are read off the first
    block row of the exponential of the augmented matrix
    ``[[X, I, 0, ...], [0, 0, I, ...], ..., [0, ..., 0]]`` -- ``exp(X)``,
    ``phi_1(X)``, ..., ``phi_count(X)`` -- and carried back by ``Z``; off the block the
    diagonal functions of :meth:`ExponentialKernels.moments`.

    Parameters
    ----------
    exponent : numpy.ndarray
        ``Delta Phi`` over the step, the same in every channel.
    count : int
        The moments formed.
    block : CoupledBlock
        The step's frozen block.
    dz : float
        The step, m.
    carrier : numpy.ndarray or None, optional
        A carrier frame's ``sigma`` (SEP 35 section 4.5), broadcasting against
        `exponent`. The moments are then those of the shifted block
        ``h B - sigma I``, and the flow is ``exp(sigma)`` times its ``phi_0``:
        ``sigma I`` commutes with ``B``, so ``exp(h B) = exp(sigma) exp(h B - sigma
        I)`` and one evaluation gives both. ``None`` for the unshifted step.

    Returns
    -------
    MatrixMoments
        The flow ``exp(h B)`` and the moments.
    """
    unshifted = exponent
    if carrier is not None:
        exponent = np.ascontiguousarray(exponent - carrier)
    active = block.active
    # The block's bins take matrix moments below: the scalar ones are formed off it.
    scalar = np.zeros((count, *exponent.shape), dtype=np.complex128)
    if count and not np.all(active):
        scalar[..., ~active] = ExponentialKernels().moments(
            np.ascontiguousarray(exponent[..., ~active]), count
        )
    flow_scalar = np.exp(unshifted)
    n_modes = block.triangular.shape[-1]
    shift = exponent[0, 0, active]
    matrices = np.empty((count, *block.triangular.shape), dtype=np.complex128)
    flow_matrix = np.empty(block.triangular.shape, dtype=np.complex128)
    size = n_modes * (count + 1)
    identity = np.eye(n_modes)
    for index in range(block.triangular.shape[0]):
        augmented = np.zeros((size, size), dtype=np.complex128)
        augmented[:n_modes, :n_modes] = (
            shift[index] * identity + dz * block.triangular[index]
        )
        for row in range(count):
            start = (row + 1) * n_modes
            augmented[row * n_modes : start, start : start + n_modes] = identity
        functions = expm(augmented)[:n_modes]
        vectors = block.vectors[index]
        adjoint = np.conj(vectors.T)
        flow_matrix[index] = vectors @ functions[:, :n_modes] @ adjoint
        for k in range(count):
            column = (k + 1) * n_modes
            matrices[k, index] = (
                math.factorial(k)
                * vectors
                @ functions[:, column : column + n_modes]
                @ adjoint
            )
    if carrier is not None:
        factor = np.exp(np.broadcast_to(carrier, unshifted.shape)[0, 0, active])
        flow_matrix = factor[:, np.newaxis, np.newaxis] * flow_matrix
    return MatrixMoments(scalar, flow_scalar, matrices, flow_matrix, active)


class ExponentialKernels:
    """The field arithmetic of an exponential Adams step, in NumPy (SEP 16 phase 2).

    The history is a store of **untransported** source rows, newest first by
    ``slots``. A trial forms ``M_k = k! phi_{k+1}(x)``, ``x = h D`` per element, once,
    and applies every formula it needs in the monomial basis of its interpolant,
    ``h sum_k M_k (sum_j c_jk F_j)`` (``solprop.backends.exponential``), so no
    per-node weight volume is formed. Accepting a step re-anchors nothing. The
    native backend's kernels are checked against these passes.

    Beside a coupled flow (SEP 16 phase 5) a trial's flow and moments are
    :class:`MatrixMoments` (:func:`coupled_moments`): matrices in the channel axis on
    the bins that carry a block, elementwise elsewhere; the same passes apply them.
    Under a carrier frame (SEP 35) the passes take the moments at ``x - sigma`` and
    rows rotated by ``exp(sigma (1 - theta_j))``; the caller forms both
    (:class:`~solprop.backends.numpy_ref.multistep.ExponentialHistory`), so the passes
    themselves do not change.
    """

    def moments(
        self, exponent: _State, count: int, *, shift: _Shift | None = None
    ) -> Complex[np.ndarray, "n_k n_modes n_pol n_omega"]:
        """Return ``k! phi_{k+1}(x)`` for ``k < count`` along a new leading axis.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = Phi(z + h) - Phi(z)``, in the canonical layout.
        count : int
            The most interpolation nodes any formula of the trial has.
        shift : numpy.ndarray or None, optional
            A carrier frame's ``sigma`` (SEP 35): the moments are then those of
            ``x - sigma``, and the step's flow stays ``exp(x)`` (:meth:`flow`).

        Returns
        -------
        numpy.ndarray
            Shape ``(count, *x.shape)``.
        """
        if shift is not None:
            exponent = exponent - shift
        phi = phi_functions(exponent, count)
        factors = np.array([math.factorial(k) for k in range(count)], dtype=np.float64)
        return phi[1:] * factors.reshape((count, *([1] * exponent.ndim)))

    def flow(self, exponent: _State) -> _State:
        """Return the step's exact linear flow ``exp(x)``.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = Phi(z + h) - Phi(z)``.

        Returns
        -------
        numpy.ndarray
            ``exp(x)``, the propagator the interaction formulation would apply.
        """
        return np.exp(exponent)

    def carry(  # noqa: PLR0917 - called positionally
        self,
        exponent: _State,
        dz: float,
        rates: Complex[np.ndarray, "n_slots n_modes n_pol n_omega"],
        states: Complex[np.ndarray, "n_slots n_modes n_pol n_omega"],
        slots: np.ndarray,
        spans: np.ndarray,
    ) -> Complex[np.ndarray, "n_used n_modes n_pol n_omega"]:
        """Return each history sample's compensation carried to the step's end.

        ``exp(s_j x) (D(z_j) - x / dz) u_j`` with ``s_j = 1 - theta_j``, the
        difference of rates formed before the state multiplies it (SEP 16 4.2).

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = Phi(z + h) - Phi(z)``.
        dz : float
            The step, m.
        rates, states : numpy.ndarray
            The history's rates and states, by slot.
        slots : numpy.ndarray
            The samples carried, newest first, ``int64``.
        spans : numpy.ndarray
            ``1 - theta_j`` per sample: the flow's length in steps.

        Returns
        -------
        numpy.ndarray
            One carried row per sample.
        """
        reference = exponent / dz
        return np.stack(
            [
                np.exp(span * exponent) * ((rates[slot] - reference) * states[slot])
                for slot, span in zip(slots, spans, strict=True)
            ]
        )

    def carry_coupled(  # noqa: PLR0917 - called positionally
        self,
        exponent: _State,
        dz: float,
        rows: tuple[_StoreRows, _StoreRows, list[np.ndarray | None]],
        slots: np.ndarray,
        spans: np.ndarray,
        block: CoupledBlock,
    ) -> Complex[np.ndarray, "n_used n_modes n_pol n_omega"]:
        """Return each sample's compensation carried to the step's end (SEP 16 4.5).

        ``C_j = (D(z_j) - x / h) u_j + R_s(z_j) u_j``: the diagonal chord's
        difference of rates and the frozen-residual's residual relative to the
        **step's** section, formed from the sample's own flow weights -- so that a
        step in a new section rebuilds the history's residual against its block --
        and carried by the reference's forward flow ``exp((1 - theta_j) h B)``.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = Delta Phi`` over the step.
        dz : float
            The step, m.
        rows : tuple
            The history's rates and states, by slot, and each slot's flow weights.
        slots : numpy.ndarray
            The samples carried, newest first.
        spans : numpy.ndarray
            ``1 - theta_j`` per sample.
        block : CoupledBlock
            The step's frozen block.

        Returns
        -------
        numpy.ndarray
            One carried row per sample.
        """
        rates, states, weights = rows
        reference = exponent / dz
        carried = []
        for slot, span in zip(slots, spans, strict=True):
            term = (rates[slot] - reference) * states[slot]
            sample_weights = weights[slot]
            if block.residual is not None and sample_weights is not None:
                term = term + block.residual(states[slot], sample_weights)
            if span != 0.0:
                flow = coupled_moments(span * exponent, 0, block, span * dz)
                term = flow.flow(term)
            carried.append(term)
        return np.stack(carried)

    def apply(
        self,
        moments: _MomentRows | MatrixMoments,
        coefficients: np.ndarray,
        values: list[_State],
        dz: float,
    ) -> _State:
        """Return ``dz sum_k M_k (sum_j c_jk X_j)`` over the given rows.

        Parameters
        ----------
        moments : numpy.ndarray or MatrixMoments
            ``M_k`` from :meth:`moments` or :func:`coupled_moments`.
        coefficients : numpy.ndarray
            ``c_jk``, shape ``(len(values), n_k)`` with ``n_k <= len(moments)``.
        values : list of numpy.ndarray
            The samples ``X_j``.
        dz : float
            The step, m.

        Returns
        -------
        numpy.ndarray
            The increment.
        """
        total = np.zeros_like(values[0])
        for k in range(coefficients.shape[1]):
            combined = np.zeros_like(values[0])
            for weight, value in zip(coefficients[:, k], values, strict=True):
                if weight != 0.0:
                    combined = accumulate(combined, value, float(weight))
            total += self._times_moment(moments, k, combined)
        return dz * total

    def predict(  # noqa: PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        flow: _State | MatrixMoments,
        moments: _MomentRows | MatrixMoments,
        store: Complex[np.ndarray, "n_slots n_modes n_pol n_omega"],
        slots: np.ndarray,
        coefficients: np.ndarray,
        dz: float,
        *,
        factors: _Factors | None = None,
    ) -> _State:
        """Return the predicted field ``exp(x) u_n + dz sum_k M_k sum_j c_jk F_j``.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state at the anchor.
        flow : numpy.ndarray or MatrixMoments
            ``exp(x)``, the step's exact linear flow, or a coupled trial's.
        moments : numpy.ndarray or MatrixMoments
            ``M_k`` from :meth:`moments` or :func:`coupled_moments`.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows the predictor combines, newest first, ``int64``.
        coefficients : numpy.ndarray
            The predictor's Lagrange coefficients, shape ``(len(slots), len(slots))``.
        dz : float
            The step, m.
        factors : numpy.ndarray or None, optional
            A carrier frame's rotation of each row read, ``exp(sigma (1 - theta_j))``
            in the slots' order (SEP 35 section 6.2); ``None`` for none.

        Returns
        -------
        numpy.ndarray
            The predicted lab-frame field at the end of the step.
        """
        values = _rows_read(store, slots, factors)
        return self._times_flow(flow, state) + self.apply(
            moments, coefficients, values, dz
        )

    def correct(  # noqa: PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        flow: _State | MatrixMoments,
        moments: _MomentRows | MatrixMoments,
        source: _State,
        store: Complex[np.ndarray, "n_slots n_modes n_pol n_omega"],
        slots: np.ndarray,
        corrector: np.ndarray,
        rows: np.ndarray,
        dz: float,
        mask: np.ndarray,
        divisor: np.ndarray,
        extra: tuple[_State, list[_State]] | None = None,
        *,
        factors: _Factors | None = None,
    ) -> tuple[_State, np.ndarray]:
        """Return the corrected field and the scaled norms of the defects.

        With ``X = (source, store[slots[0]], ...)``, the field is
        ``exp(x) u_n + dz sum_k M_k sum_j corrector[j, k] X_j`` and each norm is the
        weak norm of ``dz sum_k M_k sum_j rows[r, j, k] X_j`` over the in-band bins;
        both are already at the end of the step. A varying operator's compensation
        adds its own corrector term to the field and one term to each defect.

        Parameters
        ----------
        state, flow, moments : numpy.ndarray
            As for :meth:`predict`.
        source : numpy.ndarray
            The untransported source at the predicted field.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows after the source, newest first, ``int64``.
        corrector : numpy.ndarray
            The corrector's coefficients over ``X``, shape ``(width, n_k)``, zero past
            its own nodes.
        rows : numpy.ndarray
            One defect's coefficients over ``X`` per row, shape ``(n_rows, width,
            n_k)``.
        dz : float
            The step, m.
        mask : numpy.ndarray
            The in-band bins, Boolean.
        divisor : numpy.ndarray
            The norm's scale per bin.
        extra : tuple or None, optional
            The compensation's corrector term and its term in each defect, or
            ``None`` for a constant operator.
        factors : numpy.ndarray or None, optional
            A carrier frame's rotation of each history row read, in the slots'
            order; the new source's is one.

        Returns
        -------
        tuple
            The corrected field and one norm per row.
        """
        values = [source, *_rows_read(store, slots, factors)]
        advanced = self._times_flow(flow, state) + self.apply(
            moments, corrector, values, dz
        )
        if extra is not None:
            advanced = advanced + extra[0]
        norms = np.empty(rows.shape[0])
        for index, row in enumerate(rows):
            defect = self.apply(moments, row, values, dz)
            if extra is not None:
                defect = defect + extra[1][index]
            norms[index] = float(np.linalg.norm(defect[..., mask] / divisor[mask]))
        return advanced, norms

    @staticmethod
    def _times_moment(
        moments: _MomentRows | MatrixMoments, k: int, values: _State
    ) -> _State:
        """Return ``M_k values``: elementwise, or through a coupled trial's block.

        - ``values``: the array, in the state's layout.
        """
        if isinstance(moments, MatrixMoments):
            return moments.apply(k, values)
        return moments[k] * values

    @staticmethod
    def _times_flow(flow: _State | MatrixMoments, values: _State) -> _State:
        """Return the step's flow applied to an array: elementwise, or through a block.

        - ``values``: the array, in the state's layout.
        """
        if isinstance(flow, MatrixMoments):
            return flow.flow(values)
        return flow * values


def _rows_read(
    store: Complex[np.ndarray, "n_slots n_modes n_pol n_omega"],
    slots: np.ndarray,
    factors: _Factors | None,
) -> list[_State]:
    """Return the history rows a pass combines, rotated when a carrier frame asks.

    - ``factors``: their rotations, in the slots' order; ``None`` for the rows
      themselves.
    """
    if factors is None:
        return [store[slot] for slot in slots]
    return [factor * store[slot] for factor, slot in zip(factors, slots, strict=True)]
