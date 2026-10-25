# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The fused Adams step's passes, compiled (SEP 14 phase 2, SEP 16 phase 3).

:class:`solprop.backends.multistep.AdamsKernels` states the arithmetic in NumPy; here
each pass is one traversal of the field, which on a modal problem of a few thousand
complex elements turns some thirty array operations a step into three loops. The
integrated weights are compiled too, with the reference's operations in the
reference's order, so both backends take the same weights.

The exponential formulation's passes (:class:`NumbaExponentialKernels`) form every
element's moments ``k! phi_{k+1}(x)`` and its flow ``exp(x)`` once per trial, in one
traversal, with the reference's algorithm (:mod:`solprop.backends.exponential`) and
reciprocal multiplications in place of its divisions; the predictor and the corrector
read them. The moments are a per-trial scratch of ``(q + 2)`` state-sized rows, reused
from trial to trial, never a cached coefficient volume (SEP 16 section 6): forming
them in each pass instead cost twice the moment work, which on the Cartesian pilot was
ten times the source evaluation.

Beside a coupled flow (:class:`NumbaCoupledExponentialKernels`, SEP 16 phase 5) the
trial's flow and moments are matrices in the channel axis on the bins that carry a
block. On the bins of the flow actions' eigenbasis route each is ``W f(x + h lambda)
Y``, formed per bin from the eigenvalues' moments in one compiled pass; the predictor
and the corrector apply them per bin and the diagonal ones elsewhere, one traversal
each, with the defects factored as on the diagonal path.

**Rows in parallel** (SEP 28). The diagonal passes are built by :func:`make_passes`
in two variants, the row loop a ``prange`` in the parallel one, which the backend
chooses for a large problem and which is never disk-cached (``docs/dev/backends.md``
section 3.2). A row's arithmetic is the same in both, and each defect's squared norm is
summed in two stages, one partial per row, in both.

**Strict passes** (SEP 25 section 4.2). The integrated weights, the phi-functions, the
correctors with their defects and the varying operator's compensation (``_carry``,
``_carry_flows``) form small differences of large numbers, so they compile with
``strict=True``: no contraction, called and never inlined. The predictors, the
transport and the moments' traversal take the default contraction.
"""

from functools import cache
from typing import Any, NamedTuple, no_type_check

import numpy as np
from jaxtyping import Bool, Complex, Float, Int
from numba import prange

from solprop._helpers import NUMBA
from solprop._numba_cache import njit
from solprop.backends.exponential import PHI_JOIN
from solprop.backends.multistep import (
    AdamsKernels,
    CoupledBlock,
    CoupledExponentialKernels,
    ExponentialKernels,
    coupled_moments,
    factored_defect_rows,
)
from solprop.stepper.multistep import MAX_ADAMS_ORDER

__all__ = [
    "AdamsPasses",
    "NumbaAdamsKernels",
    "NumbaCoupledExponentialKernels",
    "NumbaExponentialKernels",
    "make_passes",
]

_SERIES_TERMS = 24
"""Most Taylor terms of the highest phi function, the reference's fixed count."""

_EPSILON = float(np.finfo(np.float64).eps)

_FACTORIALS = np.array([float(np.prod(np.arange(1, k + 1))) for k in range(12)])
"""``k!`` for ``k < 12``, exact in float64."""

_State = Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
_Store = Complex[np.ndarray, "n_slots n_batch n_modes n_pol n_omega"]


@njit(cache=True, nogil=True, strict=True)
@no_type_check
def _weights(points: Any, upper: float) -> Any:  # pragma: no cover - compiled
    """Return the integrated Lagrange weights, as ``integrated_weights`` forms them.

    Parameters
    ----------
    points : numpy.ndarray
        Distinct normalised nodes.
    upper : float
        The integral's upper limit.

    Returns
    -------
    numpy.ndarray
        One weight per node.
    """
    count = points.size
    weights = np.empty(count)
    coefficients = np.empty(count)
    product = np.empty(count)
    for index in range(count):
        node = points[index]
        coefficients[0] = 1.0
        length = 1
        denominator = 1.0
        for other in range(count):
            if other == index:
                continue
            root = points[other]
            product[0] = 0.0
            for power in range(length):
                product[power + 1] = coefficients[power]
            for power in range(length):
                product[power] -= root * coefficients[power]
            length += 1
            for power in range(length):
                coefficients[power] = product[power]
            denominator *= node - root
        integral = 0.0
        for power in range(length - 1, -1, -1):
            integral = (integral + coefficients[power] / (power + 1)) * upper
        weights[index] = integral / denominator
    return weights


@njit(cache=True, nogil=True, strict=True)
@no_type_check
def _phi(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
    x: Any,
    exp_x: Any,
    order: int,
    reciprocal: Any,
    inverse_terms: Any,
    out: Any,
    old: Any,
) -> None:  # pragma: no cover - compiled
    """Write ``phi_0(x)`` to ``phi_order(x)`` into ``out`` for one element.

    Parameters
    ----------
    x, exp_x : complex
        The argument and its exponential.
    order : int
        The highest function.
    reciprocal, inverse_terms : numpy.ndarray
        ``1/k!`` and ``1/(m + order)``.
    out, old : numpy.ndarray
        Length ``order + 1``; ``old`` is scratch.
    """
    magnitude = abs(x)
    if magnitude == 0.0:
        # An inactive bin: phi_k(0) = 1/k! exactly, as the series gives it.
        for k in range(order + 1):
            out[k] = reciprocal[k]
        return
    if magnitude > PHI_JOIN:
        inverse = 1.0 / x
        out[0] = exp_x
        for k in range(1, order + 1):
            out[k] = (out[k - 1] - reciprocal[k - 1]) * inverse
        return
    squarings = 0
    if magnitude > 0.5:
        squarings = int(np.ceil(np.log2(magnitude / 0.5)))
    y = x * (0.5**squarings)
    term = complex(reciprocal[order])
    total = term
    for m in range(1, _SERIES_TERMS):
        term = term * y * inverse_terms[m]
        total = total + term
        # Each later term is below |y| / (m + order) of this one, |y| <= 1/2: once
        # this one is under a quarter ulp of the sum, the rest cannot change it.
        if abs(term) <= 0.25 * _EPSILON * abs(total):
            break
    out[order] = total
    for k in range(order - 1, 0, -1):
        out[k] = reciprocal[k] + y * out[k + 1]
    out[0] = np.exp(y)
    for _ in range(squarings):
        for k in range(order + 1):
            old[k] = out[k]
        out[0] = old[0] * old[0]
        for k in range(1, order + 1):
            total = old[0] * old[k]
            for i in range(1, k + 1):
                total = total + old[i] * reciprocal[k - i]
            out[k] = total * (0.5**k)


class AdamsPasses(NamedTuple):
    """The element-wise passes of both Adams formulations, one compiled variant.

    Attributes
    ----------
    predict, correct, transport : callable
        The interaction formulation's predictor, corrector and history transport.
    moments, exponential_predict, exponential_correct, carry : callable
        The exponential formulation's moments and flow, predictor, corrector and a
        varying operator's compensation.
    """

    predict: Any
    correct: Any
    transport: Any
    moments: Any
    exponential_predict: Any
    exponential_correct: Any
    carry: Any


@cache
def make_passes(*, parallel: bool) -> AdamsPasses:  # noqa: PLR0915 - seven kernels' bodies
    """Compile the Adams passes, their rows serial or in a ``prange``.

    Every pass traverses rows of ``n_omega`` bins, each row independent, so the row
    loop is the parallel one. A row's arithmetic is the same in either variant, and a
    defect's squared norm is two-stage in both: one partial per row, summed by the
    caller.

    Parameters
    ----------
    parallel : bool
        Whether independent rows execute in parallel.

    Returns
    -------
    AdamsPasses
        The compiled passes.
    """
    rows_of = prange if parallel else range

    @no_type_check
    def predict(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        state: Any,
        propagator: Any,
        store: Any,
        slots: Any,
        weights: Any,
        dz: float,
        out: Any,
    ) -> None:  # pragma: no cover - compiled
        """Write ``U (a_n + dz sum_t w_t store[slots[t]])`` over flattened rows.

        Parameters
        ----------
        state, propagator : numpy.ndarray
            Shape ``(n_rows, n_omega)``.
        store : numpy.ndarray
            Shape ``(n_slots, n_rows, n_omega)``.
        slots : numpy.ndarray
            The rows combined, ``int64``.
        weights : numpy.ndarray
            One weight per slot.
        dz : float
            The step, m.
        out : numpy.ndarray
            Written with the field, shape ``(n_rows, n_omega)``.
        """
        n_rows, n_omega = state.shape
        for row in rows_of(n_rows):
            for j in range(n_omega):
                total = 0.0 + 0.0j
                for t in range(slots.size):
                    total += weights[t] * store[slots[t], row, j]
                out[row, j] = propagator[row, j] * (state[row, j] + dz * total)

    @no_type_check
    def correct(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        state: Any,
        propagator: Any,
        source: Any,
        store: Any,
        slots: Any,
        corrector: Any,
        rows: Any,
        dz: float,
        mask: Any,
        divisor: Any,
        out: Any,
        partials: Any,
    ) -> None:  # pragma: no cover - compiled
        """Write the corrected field and each row's squared scaled defect norms.

        Parameters
        ----------
        state, propagator, source : numpy.ndarray
            Shape ``(n_rows, n_omega)``.
        store : numpy.ndarray
            Shape ``(n_slots, n_rows, n_omega)``.
        slots : numpy.ndarray
            The history rows after the source, ``int64``.
        corrector : numpy.ndarray
            The corrector's weights over ``(source, slots...)``.
        rows : numpy.ndarray
            The defects' weights, shape ``(n_defects, 1 + n_slots)``.
        dz : float
            The step, m.
        mask : numpy.ndarray
            The in-band bins.
        divisor : numpy.ndarray
            The scale per bin.
        out : numpy.ndarray
            Written with the corrected field, shape ``(n_rows, n_omega)``.
        partials : numpy.ndarray
            Written with each row's squared norm per defect, shape
            ``(n_rows, n_defects)``.
        """
        n_rows, n_omega = state.shape
        n_defects = rows.shape[0]
        for row in rows_of(n_rows):
            defects = np.empty(n_defects, dtype=np.complex128)
            for index in range(n_defects):
                partials[row, index] = 0.0
            for j in range(n_omega):
                value = source[row, j]
                total = corrector[0] * value
                for index in range(n_defects):
                    defects[index] = rows[index, 0] * value
                for t in range(slots.size):
                    value = store[slots[t], row, j]
                    total += corrector[t + 1] * value
                    for index in range(n_defects):
                        defects[index] += rows[index, t + 1] * value
                factor = propagator[row, j]
                out[row, j] = factor * (state[row, j] + dz * total)
                if mask[j]:
                    for index in range(n_defects):
                        scaled = factor * (dz * defects[index]) / divisor[j]
                        partials[row, index] += (
                            scaled.real * scaled.real + scaled.imag * scaled.imag
                        )

    @no_type_check
    def transport(
        store: Any, slots: Any, propagator: Any, new_slot: int, source: Any
    ) -> None:  # pragma: no cover - compiled
        """Multiply the kept rows by the propagator and store the new derivative.

        Parameters
        ----------
        store : numpy.ndarray
            Shape ``(n_slots, n_rows, n_omega)``, updated in place.
        slots : numpy.ndarray
            The kept rows, ``int64``.
        propagator : numpy.ndarray
            Shape ``(n_rows, n_omega)``.
        new_slot : int
            The row the new derivative goes to.
        source : numpy.ndarray
            The new derivative, shape ``(n_rows, n_omega)``.
        """
        n_rows, n_omega = propagator.shape
        for row in rows_of(n_rows):
            for j in range(n_omega):
                factor = propagator[row, j]
                for t in range(slots.size):
                    store[slots[t], row, j] *= factor
                store[new_slot, row, j] = factor * source[row, j]

    @no_type_check
    def moments(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        exponent: Any,
        count: int,
        reciprocal: Any,
        inverse_terms: Any,
        factorials: Any,
        moments: Any,
        flow: Any,
    ) -> None:  # pragma: no cover - compiled
        """Write ``k! phi_{k+1}(x)`` for ``k < count`` and ``exp(x)`` per element.

        The reference's algorithm (:func:`solprop.backends.exponential.phi_functions`)
        through :func:`_phi`: above the join an upward recurrence from ``exp(x)``;
        below it the highest series at ``|y| <= 1/2``, a downward recurrence and the
        doubling identity. Divisions by ``x`` and by ``m + count`` are multiplications
        by reciprocals here.

        Parameters
        ----------
        exponent : numpy.ndarray
            Shape ``(n_rows, n_omega)``.
        count : int
            The number of moments.
        reciprocal : numpy.ndarray
            ``1/k!`` for ``k <= count + 1``.
        inverse_terms : numpy.ndarray
            ``1/(m + count)`` for the series' terms.
        factorials : numpy.ndarray
            ``k!`` for ``k < count``.
        moments : numpy.ndarray
            Written, shape ``(count, n_rows, n_omega)``.
        flow : numpy.ndarray
            Written with ``exp(x)``, shape ``(n_rows, n_omega)``.
        """
        n_rows, n_omega = exponent.shape
        for row in rows_of(n_rows):
            out = np.empty(count + 1, dtype=np.complex128)
            old = np.empty(count + 1, dtype=np.complex128)
            for j in range(n_omega):
                x = exponent[row, j]
                flow[row, j] = np.exp(x)
                _phi(x, flow[row, j], count, reciprocal, inverse_terms, out, old)
                for k in range(count):
                    moments[k, row, j] = factorials[k] * out[k + 1]

    @no_type_check
    def exponential_predict(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        state: Any,
        flow: Any,
        moments: Any,
        store: Any,
        slots: Any,
        coefficients: Any,
        dz: float,
        out: Any,
    ) -> None:  # pragma: no cover - compiled
        """Write ``exp(x) u + dz sum_k M_k sum_j c_jk F_j`` over flattened rows.

        Parameters
        ----------
        state, flow : numpy.ndarray
            Shape ``(n_rows, n_omega)``.
        moments : numpy.ndarray
            ``M_k``, shape ``(n_k, n_rows, n_omega)``.
        store : numpy.ndarray
            Shape ``(n_slots, n_rows, n_omega)``.
        slots : numpy.ndarray
            The rows combined, ``int64``.
        coefficients : numpy.ndarray
            ``c_jk``, shape ``(len(slots), n_k)``.
        dz : float
            The step, m.
        out : numpy.ndarray
            Written with the field.
        """
        n_rows, n_omega = state.shape
        count = coefficients.shape[1]
        for row in rows_of(n_rows):
            for j in range(n_omega):
                total = 0.0 + 0.0j
                for k in range(count):
                    combined = 0.0 + 0.0j
                    for t in range(slots.size):
                        combined += coefficients[t, k] * store[slots[t], row, j]
                    total += moments[k, row, j] * combined
                out[row, j] = flow[row, j] * state[row, j] + dz * total

    @no_type_check
    def exponential_correct(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        state: Any,
        flow: Any,
        moments: Any,
        source: Any,
        store: Any,
        slots: Any,
        corrector: Any,
        rows: Any,
        dz: float,
        mask: Any,
        divisor: Any,
        extra: bool,
        extra_field: Any,
        extra_defects: Any,
        out: Any,
        partials: Any,
    ) -> None:  # pragma: no cover - compiled
        """Write the corrected field and each row's squared scaled defect norms.

        Over ``X = (source, store[slots]...)``: the field is
        ``exp(x) u + dz sum_k M_k sum_j corrector[j, k] X_j`` and each defect
        ``dz sum_k M_k sum_j rows[r, j, k] X_j``, already at the step's end.

        Parameters
        ----------
        state, flow : numpy.ndarray
            Shape ``(n_rows, n_omega)``.
        moments : numpy.ndarray
            ``M_k``, shape ``(n_k, n_rows, n_omega)``.
        source : numpy.ndarray
            The new sample, shape ``(n_rows, n_omega)``.
        store : numpy.ndarray
            Shape ``(n_slots, n_rows, n_omega)``.
        slots : numpy.ndarray
            The history rows after the source, ``int64``.
        corrector : numpy.ndarray
            Shape ``(width, n_k)``.
        rows : numpy.ndarray
            Each defect's divided-difference coefficients over the samples and its
            node polynomial's monomial coefficients, shape ``(n_defects, width + n_k)``.
        dz : float
            The step, m.
        mask, divisor : numpy.ndarray
            The in-band bins and the scale per bin.
        extra : bool
            Whether a varying operator's compensation terms are added.
        extra_field, extra_defects : numpy.ndarray
            Its corrector term, shape ``(n_rows, n_omega)``, and its term in each
            defect, shape ``(n_defects, n_rows, n_omega)``; read only when `extra`
            holds.
        out : numpy.ndarray
            Written with the corrected field.
        partials : numpy.ndarray
            Written with each row's squared norm per defect, shape
            ``(n_rows, n_defects)``.
        """
        n_rows, n_omega = state.shape
        width, count = corrector.shape
        n_defects = rows.shape[0]
        for row in rows_of(n_rows):
            values = np.empty(width, dtype=np.complex128)
            for index in range(n_defects):
                partials[row, index] = 0.0
            for j in range(n_omega):
                values[0] = source[row, j]
                for t in range(width - 1):
                    values[t + 1] = store[slots[t], row, j]
                total = 0.0 + 0.0j
                for k in range(count):
                    combined = 0.0 + 0.0j
                    for t in range(width):
                        combined += corrector[t, k] * values[t]
                    total += moments[k, row, j] * combined
                out[row, j] = flow[row, j] * state[row, j] + dz * total
                if extra:
                    out[row, j] += extra_field[row, j]
                if mask[j]:
                    for index in range(n_defects):
                        # A defect is a divided difference of the samples times one
                        # moment combination (NumbaExponentialKernels.defect_rows).
                        difference = 0.0 + 0.0j
                        for t in range(width):
                            difference += rows[index, t] * values[t]
                        weight = 0.0 + 0.0j
                        for k in range(count):
                            weight += rows[index, width + k] * moments[k, row, j]
                        defect = dz * (difference * weight)
                        if extra:
                            defect += extra_defects[index, row, j]
                        scaled = defect / divisor[j]
                        partials[row, index] += (
                            scaled.real * scaled.real + scaled.imag * scaled.imag
                        )

    @no_type_check
    def carry(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        exponent: Any,
        inverse_dz: float,
        rates: Any,
        states: Any,
        slots: Any,
        spans: Any,
        out: Any,
    ) -> None:  # pragma: no cover - compiled
        """Write ``exp(s_j x) (D_j - x / dz) u_j`` for each sample, in one traversal.

        Parameters
        ----------
        exponent : numpy.ndarray
            Shape ``(n_rows, n_omega)``.
        inverse_dz : float
            ``1 / dz``.
        rates, states : numpy.ndarray
            Shape ``(n_slots, n_rows, n_omega)``.
        slots : numpy.ndarray
            The samples, ``int64``.
        spans : numpy.ndarray
            ``1 - theta_j`` per sample; zero leaves a sample in its own frame.
        out : numpy.ndarray
            Written, shape ``(len(slots), n_rows, n_omega)``.
        """
        n_rows, n_omega = exponent.shape
        for row in rows_of(n_rows):
            for j in range(n_omega):
                x = exponent[row, j]
                reference = x * inverse_dz
                for t in range(slots.size):
                    slot = slots[t]
                    term = (rates[slot, row, j] - reference) * states[slot, row, j]
                    if spans[t] != 0.0:
                        term = np.exp(spans[t] * x) * term
                    out[t, row, j] = term

    compile_ = njit(cache=not parallel, nogil=True, parallel=parallel)
    strict = njit(cache=not parallel, nogil=True, parallel=parallel, strict=True)
    return AdamsPasses(
        compile_(predict),
        strict(correct),
        compile_(transport),
        compile_(moments),
        compile_(exponential_predict),
        strict(exponential_correct),
        strict(carry),
    )


@njit(cache=True, nogil=True)
@no_type_check
def _block_functions(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
    shift: Any,
    dz: float,
    eigenvalues: Any,
    left: Any,
    right: Any,
    route: Any,
    count: int,
    reciprocal: Any,
    inverse_terms: Any,
    factorials: Any,
    matrices: Any,
    flow_matrix: Any,
) -> None:  # pragma: no cover - compiled
    """Write ``W exp(X) Y`` and ``W k! phi_{k+1}(X) Y`` on the route's bins.

    ``X = shift + dz lambda`` per eigenvalue, the reference's moments of each
    (:func:`_phi`), carried by the bin's eigenbasis.

    Parameters
    ----------
    shift : numpy.ndarray
        ``Delta Phi`` per bin of the block, shape ``(n_bins,)``.
    dz : float
        The step, m.
    eigenvalues : numpy.ndarray
        Shape ``(n_bins, n_modes)``.
    left, right : numpy.ndarray
        ``W`` and ``Y``, shape ``(n_bins, n_modes, n_modes)``.
    route : numpy.ndarray
        The bins formed here, Boolean.
    count : int
        The moments formed.
    reciprocal, inverse_terms, factorials : numpy.ndarray
        As for :func:`_moments`.
    matrices : numpy.ndarray
        Written on the route, shape ``(count, n_bins, n_modes, n_modes)``.
    flow_matrix : numpy.ndarray
        Written on the route, shape ``(n_bins, n_modes, n_modes)``.
    """
    n_bins, n_modes = eigenvalues.shape
    out = np.empty(count + 1, dtype=np.complex128)
    old = np.empty(count + 1, dtype=np.complex128)
    values = np.empty((n_modes, count + 1), dtype=np.complex128)
    for bin_ in range(n_bins):
        if not route[bin_]:
            continue
        for m in range(n_modes):
            x = shift[bin_] + dz * eigenvalues[bin_, m]
            exp_x = np.exp(x)
            values[m, 0] = exp_x
            if count > 0:
                _phi(x, exp_x, count, reciprocal, inverse_terms, out, old)
                for k in range(count):
                    values[m, k + 1] = factorials[k] * out[k + 1]
        for i in range(n_modes):
            for j in range(n_modes):
                for k in range(count + 1):
                    total = 0.0 + 0.0j
                    for m in range(n_modes):
                        total += left[bin_, i, m] * values[m, k] * right[bin_, m, j]
                    if k == 0:
                        flow_matrix[bin_, i, j] = total
                    else:
                        matrices[k - 1, bin_, i, j] = total


@njit(cache=True, nogil=True)
@no_type_check
def _coupled_predict(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
    state: Any,
    flow: Any,
    moments: Any,
    store: Any,
    slots: Any,
    coefficients: Any,
    dz: float,
    blocks: Any,
    flow_matrix: Any,
    matrices: Any,
    out: Any,
) -> None:  # pragma: no cover - compiled
    """Write the predicted field beside a coupled flow, in one traversal.

    Parameters
    ----------
    state, flow : numpy.ndarray
        Shape ``(n_batch, n_modes, n_pol, n_omega)``; `flow` is read off the block.
    moments : numpy.ndarray
        The diagonal moments off the block, shape ``(n_k, *state.shape)``.
    store : numpy.ndarray
        Shape ``(n_slots, *state.shape)``.
    slots : numpy.ndarray
        The rows combined, ``int64``.
    coefficients : numpy.ndarray
        ``c_jk``, shape ``(len(slots), n_k)``.
    dz : float
        The step, m.
    blocks : numpy.ndarray
        Each bin's index in the block, ``-1`` off it, ``int64``.
    flow_matrix, matrices : numpy.ndarray
        The block's flow and moments, shapes ``(n_bins, n_modes, n_modes)`` and
        ``(n_k, n_bins, n_modes, n_modes)``.
    out : numpy.ndarray
        Written with the field.
    """
    n_batch, n_modes, n_pol, n_omega = state.shape
    count = coefficients.shape[1]
    combined = np.empty((count, n_modes), dtype=np.complex128)
    for b in range(n_batch):
        for p in range(n_pol):
            for w in range(n_omega):
                for k in range(count):
                    for i in range(n_modes):
                        total = 0.0 + 0.0j
                        for t in range(slots.size):
                            total += coefficients[t, k] * store[slots[t], b, i, p, w]
                        combined[k, i] = total
                bin_ = blocks[w]
                for i in range(n_modes):
                    total = 0.0 + 0.0j
                    if bin_ < 0:
                        for k in range(count):
                            total += moments[k, b, i, p, w] * combined[k, i]
                        out[b, i, p, w] = flow[b, i, p, w] * state[b, i, p, w] + (
                            dz * total
                        )
                    else:
                        linear = 0.0 + 0.0j
                        for j in range(n_modes):
                            linear += flow_matrix[bin_, i, j] * state[b, j, p, w]
                            for k in range(count):
                                total += matrices[k, bin_, i, j] * combined[k, j]
                        out[b, i, p, w] = linear + dz * total


@njit(cache=True, nogil=True, strict=True)
@no_type_check
def _coupled_correct(  # noqa: PLR0912, PLR0913, PLR0917 - one traversal's buffers
    state: Any,
    flow: Any,
    moments: Any,
    source: Any,
    store: Any,
    slots: Any,
    corrector: Any,
    rows: Any,
    dz: float,
    mask: Any,
    divisor: Any,
    extra: bool,
    extra_field: Any,
    extra_defects: Any,
    blocks: Any,
    flow_matrix: Any,
    matrices: Any,
    out: Any,
    sums: Any,
) -> None:  # pragma: no cover - compiled
    """Write the corrected field and each defect's squared norm beside a coupled flow.

    The diagonal pass of :func:`_exponential_correct` off the block; on it each moment
    is a matrix in the channel axis, and each factored defect's moment combination a
    matrix applied to its divided difference.

    Parameters
    ----------
    state, flow : numpy.ndarray
        Shape ``(n_batch, n_modes, n_pol, n_omega)``.
    moments : numpy.ndarray
        The diagonal moments off the block, shape ``(n_k, *state.shape)``.
    source : numpy.ndarray
        The new sample, in the state's shape.
    store : numpy.ndarray
        Shape ``(n_slots, *state.shape)``.
    slots : numpy.ndarray
        The history rows after the source, ``int64``.
    corrector : numpy.ndarray
        Shape ``(width, n_k)``.
    rows : numpy.ndarray
        The factored defects, shape ``(n_defects, width + n_k)``.
    dz : float
        The step, m.
    mask, divisor : numpy.ndarray
        The in-band bins and the scale per bin.
    extra : bool
        Whether the compensation's terms are added.
    extra_field, extra_defects : numpy.ndarray
        Its corrector term, in the state's shape, and its term in each defect, shape
        ``(n_defects, *state.shape)``; read only when `extra` holds.
    blocks : numpy.ndarray
        Each bin's index in the block, ``-1`` off it, ``int64``.
    flow_matrix, matrices : numpy.ndarray
        The block's flow and moments.
    out : numpy.ndarray
        Written with the corrected field.
    sums : numpy.ndarray
        One squared norm per defect, zeroed here.
    """
    n_batch, n_modes, n_pol, n_omega = state.shape
    width, count = corrector.shape
    n_defects = rows.shape[0]
    values = np.empty((width, n_modes), dtype=np.complex128)
    combined = np.empty((count, n_modes), dtype=np.complex128)
    difference = np.empty(n_modes, dtype=np.complex128)
    for index in range(n_defects):
        sums[index] = 0.0
    for b in range(n_batch):
        for p in range(n_pol):
            for w in range(n_omega):
                for i in range(n_modes):
                    values[0, i] = source[b, i, p, w]
                    for t in range(width - 1):
                        values[t + 1, i] = store[slots[t], b, i, p, w]
                    for k in range(count):
                        total = 0.0 + 0.0j
                        for t in range(width):
                            total += corrector[t, k] * values[t, i]
                        combined[k, i] = total
                bin_ = blocks[w]
                for i in range(n_modes):
                    total = 0.0 + 0.0j
                    if bin_ < 0:
                        for k in range(count):
                            total += moments[k, b, i, p, w] * combined[k, i]
                        field = flow[b, i, p, w] * state[b, i, p, w] + dz * total
                    else:
                        linear = 0.0 + 0.0j
                        for j in range(n_modes):
                            linear += flow_matrix[bin_, i, j] * state[b, j, p, w]
                            for k in range(count):
                                total += matrices[k, bin_, i, j] * combined[k, j]
                        field = linear + dz * total
                    if extra:
                        field += extra_field[b, i, p, w]
                    out[b, i, p, w] = field
                if not mask[w]:
                    continue
                for index in range(n_defects):
                    for i in range(n_modes):
                        total = 0.0 + 0.0j
                        for t in range(width):
                            total += rows[index, t] * values[t, i]
                        difference[i] = total
                    for i in range(n_modes):
                        defect = 0.0 + 0.0j
                        if bin_ < 0:
                            weight = 0.0 + 0.0j
                            for k in range(count):
                                weight += (
                                    rows[index, width + k] * moments[k, b, i, p, w]
                                )
                            defect = weight * difference[i]
                        else:
                            for j in range(n_modes):
                                weight = 0.0 + 0.0j
                                for k in range(count):
                                    weight += (
                                        rows[index, width + k] * matrices[k, bin_, i, j]
                                    )
                                defect += weight * difference[j]
                        defect = dz * defect
                        if extra:
                            defect += extra_defects[index, b, i, p, w]
                        scaled = defect / divisor[w]
                        sums[index] += (
                            scaled.real * scaled.real + scaled.imag * scaled.imag
                        )


@njit(cache=True, nogil=True, strict=True)
@no_type_check
def _carry_flows(  # noqa: PLR0912, PLR0913, PLR0917 - one traversal's buffers
    terms: Any,
    spans: Any,
    exponent: Any,
    blocks: Any,
    shift: Any,
    dz: float,
    eigenvalues: Any,
    left: Any,
    right: Any,
) -> None:  # pragma: no cover - compiled
    """Carry each sample's compensation by ``exp(s_j h B)``, in place.

    Off the block ``exp(s_j x)`` per element; on it ``W exp(s_j (x + h lambda)) Y``
    applied to the bin's channel vector.

    Parameters
    ----------
    terms : numpy.ndarray
        The samples' compensations, shape ``(n_used, n_batch, n_modes, n_pol,
        n_omega)``; overwritten with the carried ones.
    spans : numpy.ndarray
        ``s_j = 1 - theta_j`` per sample; zero leaves a sample in its own frame.
    exponent : numpy.ndarray
        ``x = Delta Phi`` over the step, in the state's shape.
    blocks : numpy.ndarray
        Each bin's index in the block, ``-1`` off it, ``int64``.
    shift : numpy.ndarray
        ``Delta Phi`` per bin of the block.
    dz : float
        The step, m.
    eigenvalues, left, right : numpy.ndarray
        The block's eigenbasis, every bin on its route.
    """
    n_used, n_batch, n_modes, n_pol, n_omega = terms.shape
    n_bins = eigenvalues.shape[0]
    factors = np.empty((n_bins, n_modes), dtype=np.complex128)
    rotated = np.empty(n_modes, dtype=np.complex128)
    for t in range(n_used):
        span = spans[t]
        if span == 0.0:
            continue
        for bin_ in range(n_bins):
            for m in range(n_modes):
                factors[bin_, m] = np.exp(
                    span * (shift[bin_] + dz * eigenvalues[bin_, m])
                )
        for b in range(n_batch):
            for p in range(n_pol):
                for w in range(n_omega):
                    bin_ = blocks[w]
                    if bin_ < 0:
                        for i in range(n_modes):
                            terms[t, b, i, p, w] *= np.exp(span * exponent[b, i, p, w])
                        continue
                    for m in range(n_modes):
                        total = 0.0 + 0.0j
                        for j in range(n_modes):
                            total += right[bin_, m, j] * terms[t, b, j, p, w]
                        rotated[m] = factors[bin_, m] * total
                    for i in range(n_modes):
                        total = 0.0 + 0.0j
                        for m in range(n_modes):
                            total += left[bin_, i, m] * rotated[m]
                        terms[t, b, i, p, w] = total


class NumbaAdamsKernels(AdamsKernels):
    """The fused passes of :class:`~solprop.backends.multistep.AdamsKernels`, compiled.

    The canonical ``(n_batch, n_modes, n_pol, n_omega)`` arrays are viewed as rows of
    ``n_omega`` bins; every pass reads each element once, its rows in a ``prange`` when
    the backend steps a large problem (:func:`make_passes`).

    Parameters
    ----------
    parallel : bool, optional
        Whether the passes run their rows in parallel.
    """

    def __init__(self, *, parallel: bool = False) -> None:
        """Choose the passes' variant."""
        super().__init__(helpers=NUMBA)
        self._passes = make_passes(parallel=parallel)

    def weights(
        self, nodes: list[float] | Float[np.ndarray, " n_nodes"], upper: float = 1.0
    ) -> Float[np.ndarray, " n_nodes"]:
        """Return the integrated weights of an interpolant on distinct nodes.

        Parameters
        ----------
        nodes : list of float or numpy.ndarray
            The normalised nodes.
        upper : float, optional
            The integral's upper limit.

        Returns
        -------
        numpy.ndarray
            One weight per node.
        """
        return _weights(np.asarray(nodes, dtype=np.float64), float(upper))

    def predict(
        self,
        state: _State,
        propagator: _State,
        store: _Store,
        slots: Int[np.ndarray, " n_used"],
        weights: Float[np.ndarray, " n_used"],
        dz: float,
    ) -> _State:
        """Return the predictor's lab-frame field in one pass.

        Parameters
        ----------
        state, propagator : numpy.ndarray
            The accepted state and ``U(z_n + dz, z_n)``.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows combined, newest first.
        weights : numpy.ndarray
            One weight per row.
        dz : float
            The step, m.

        Returns
        -------
        numpy.ndarray
            The predicted field at the end of the step.
        """
        out = np.empty_like(state)
        n_omega = state.shape[-1]
        self._passes.predict(
            state.reshape(-1, n_omega),
            propagator.reshape(-1, n_omega),
            store.reshape(store.shape[0], -1, n_omega),
            slots,
            weights,
            float(dz),
            out.reshape(-1, n_omega),
        )
        return out

    def correct(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        propagator: _State,
        source: _State,
        store: _Store,
        slots: Int[np.ndarray, " n_used"],
        corrector: Float[np.ndarray, " n_terms"],
        rows: Float[np.ndarray, "n_defects n_terms"],
        dz: float,
        mask: Bool[np.ndarray, " n_omega"],
        divisor: Float[np.ndarray, " n_omega"],
    ) -> tuple[_State, Float[np.ndarray, " n_defects"]]:
        """Return the corrected field and the defects' norms in one pass.

        Parameters
        ----------
        state, propagator, source : numpy.ndarray
            The accepted state, ``U(z_n + dz, z_n)`` and the new source.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows after the source, newest first.
        corrector : numpy.ndarray
            The corrector's weights.
        rows : numpy.ndarray
            The defects' weights.
        dz : float
            The step, m.
        mask : numpy.ndarray
            The in-band bins.
        divisor : numpy.ndarray
            The scale per bin.

        Returns
        -------
        tuple
            The corrected field and one norm per defect.
        """
        out = np.empty_like(state)
        n_omega = state.shape[-1]
        partials = np.empty((out.size // n_omega, rows.shape[0]))
        self._passes.correct(
            state.reshape(-1, n_omega),
            propagator.reshape(-1, n_omega),
            source.reshape(-1, n_omega),
            store.reshape(store.shape[0], -1, n_omega),
            slots,
            corrector,
            rows,
            float(dz),
            mask,
            divisor,
            out.reshape(-1, n_omega),
            partials,
        )
        return out, np.sqrt(partials.sum(axis=0))

    def transport(
        self,
        store: _Store,
        slots: Int[np.ndarray, " n_kept"],
        propagator: _State,
        new_slot: int,
        source: _State,
    ) -> None:
        """Re-anchor the kept rows and store the new derivative in one pass.

        Parameters
        ----------
        store : numpy.ndarray
            The history rows, updated in place.
        slots : numpy.ndarray
            The kept rows.
        propagator : numpy.ndarray
            ``U(z_n + dz, z_n)``.
        new_slot : int
            The row the new derivative goes to.
        source : numpy.ndarray
            The new derivative in the step's anchor frame.
        """
        n_omega = propagator.shape[-1]
        self._passes.transport(
            store.reshape(store.shape[0], -1, n_omega),
            slots,
            propagator.reshape(-1, n_omega),
            int(new_slot),
            source.reshape(-1, n_omega),
        )


class NumbaExponentialKernels(ExponentialKernels):
    """The exponential formulation's passes, compiled (SEP 16 phase 3).

    :meth:`moments` forms every element's moments and flow in one traversal into
    buffers reused from trial to trial; :meth:`predict` and :meth:`correct` are one
    traversal each, their rows in a ``prange`` when the backend steps a large problem
    (:func:`make_passes`). Dense observation keeps the reference's NumPy passes: it is
    one evaluation per plane.

    Parameters
    ----------
    parallel : bool, optional
        Whether the passes run their rows in parallel.
    """

    def __init__(self, *, parallel: bool = False) -> None:
        """Choose the passes' variant; the scratch is sized on the first trial."""
        super().__init__(helpers=NUMBA)
        self._parallel = parallel
        self._passes = make_passes(parallel=parallel)
        self._moments: np.ndarray | None = None
        self._flow: np.ndarray | None = None
        self._coupled: NumbaCoupledExponentialKernels | None = None

    def moments(
        self, exponent: _State, count: int
    ) -> Complex[np.ndarray, "n_k n_batch n_modes n_pol n_omega"]:
        """Return ``k! phi_{k+1}(x)`` for ``k < count``, forming ``exp(x)`` beside them.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = Phi(z + h) - Phi(z)``.
        count : int
            The most nodes a formula of the trial has.

        Returns
        -------
        numpy.ndarray
            Shape ``(count, *x.shape)``, a view of this object's scratch.
        """
        shape = (MAX_ADAMS_ORDER + 2, *exponent.shape)
        if self._moments is None or self._moments.shape != shape:
            self._moments = np.empty(shape, dtype=np.complex128)
            self._flow = np.empty(exponent.shape, dtype=np.complex128)
        assert self._flow is not None
        n_omega = exponent.shape[-1]
        moments = self._moments[:count]
        self._passes.moments(
            np.ascontiguousarray(exponent).reshape(-1, n_omega),
            int(count),
            1.0 / _FACTORIALS[: count + 2],
            1.0 / (np.arange(_SERIES_TERMS, dtype=np.float64) + count),
            _FACTORIALS[:count],
            moments.reshape(count, -1, n_omega),
            self._flow.reshape(-1, n_omega),
        )
        return moments

    def carry(
        self,
        exponent: _State,
        dz: float,
        rates: _Store,
        states: _Store,
        slots: Int[np.ndarray, " n_used"],
        spans: Float[np.ndarray, " n_used"],
    ) -> Complex[np.ndarray, "n_used n_batch n_modes n_pol n_omega"]:
        """Return each sample's carried compensation in one compiled traversal.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = Phi(z + h) - Phi(z)``.
        dz : float
            The step, m.
        rates, states : numpy.ndarray
            The history's rates and states, by slot.
        slots : numpy.ndarray
            The samples carried, newest first.
        spans : numpy.ndarray
            ``1 - theta_j`` per sample.

        Returns
        -------
        numpy.ndarray
            One carried row per sample.
        """
        n_omega = exponent.shape[-1]
        out = np.empty((slots.size, *exponent.shape), dtype=np.complex128)
        self._passes.carry(
            np.ascontiguousarray(exponent).reshape(-1, n_omega),
            1.0 / float(dz),
            rates.reshape(rates.shape[0], -1, n_omega),
            states.reshape(states.shape[0], -1, n_omega),
            np.ascontiguousarray(slots, dtype=np.int64),
            np.ascontiguousarray(spans, dtype=np.float64),
            out.reshape(slots.size, -1, n_omega),
        )
        return out

    def flow(self, exponent: _State) -> _State:
        """Return ``exp(x)`` as :meth:`moments` formed it for this trial.

        Parameters
        ----------
        exponent : numpy.ndarray
            The trial's exponent, whose moments were formed last.

        Returns
        -------
        numpy.ndarray
            The flow, a view of this object's scratch.
        """
        del exponent
        assert self._flow is not None
        return self._flow

    def defect_rows(
        self, nodes: list[float], orders: list[int], width: int
    ) -> Float[np.ndarray, "n_defects n_factored"]:
        """Return each defect factored: a divided difference and a node polynomial.

        Order ``k``'s defect, its ``k + 1``-node corrector through the new sample less
        its ``k``-node predictor, vanishes at the predictor's nodes, so it is
        ``f[1, theta_0, ..., theta_{k-1}] prod_{j<k} (theta - theta_j)``: one divided
        difference of the samples times one polynomial, whose integral against the
        flow is one combination of the moments. That is ``2 (k + 1)`` products an
        element instead of ``(k + 1) n_k``, the reference's monomial rows.

        Parameters
        ----------
        nodes : list of float
            The history's normalised positions, newest first.
        orders : list of int
            The orders scored, the step's own first.
        width : int
            ``1 + max(orders)``: the samples and the moments of the trial.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(orders), 2 width)``: the divided-difference coefficients over
            ``(source, newest, ...)``, then the node polynomial's ascending monomial
            coefficients.
        """
        return factored_defect_rows(nodes, orders, width)

    def predict(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        flow: _State,
        moments: Complex[np.ndarray, "n_k n_batch n_modes n_pol n_omega"],
        store: _Store,
        slots: Int[np.ndarray, " n_used"],
        coefficients: Float[np.ndarray, "n_used n_c"],
        dz: float,
    ) -> _State:
        """Return the predicted field in one pass.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state at the anchor.
        flow : numpy.ndarray
            ``exp(x)``.
        moments : numpy.ndarray
            ``M_k`` from :meth:`moments`.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows the predictor combines, newest first.
        coefficients : numpy.ndarray
            The predictor's Lagrange coefficients.
        dz : float
            The step, m.

        Returns
        -------
        numpy.ndarray
            The predicted lab-frame field at the end of the step.
        """
        out = np.empty_like(state)
        n_omega = state.shape[-1]
        self._passes.exponential_predict(
            state.reshape(-1, n_omega),
            flow.reshape(-1, n_omega),
            moments.reshape(moments.shape[0], -1, n_omega),
            store.reshape(store.shape[0], -1, n_omega),
            slots,
            np.ascontiguousarray(coefficients),
            float(dz),
            out.reshape(-1, n_omega),
        )
        return out

    def correct(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        flow: _State,
        moments: Complex[np.ndarray, "n_k n_batch n_modes n_pol n_omega"],
        source: _State,
        store: _Store,
        slots: Int[np.ndarray, " n_used"],
        corrector: Float[np.ndarray, "n_width n_c"],
        rows: Float[np.ndarray, "n_defects n_factored"],
        dz: float,
        mask: Bool[np.ndarray, " n_omega"],
        divisor: Float[np.ndarray, " n_omega"],
        extra: tuple[_State, list[_State]] | None = None,
    ) -> tuple[_State, Float[np.ndarray, " n_defects"]]:
        """Return the corrected field and the defects' norms in one pass.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state at the anchor.
        flow : numpy.ndarray
            ``exp(x)``.
        moments : numpy.ndarray
            ``M_k`` from :meth:`moments`.
        source : numpy.ndarray
            The untransported source at the predicted field.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows after the source, newest first.
        corrector : numpy.ndarray
            The corrector's coefficients over the samples.
        rows : numpy.ndarray
            The defects, factored by :meth:`defect_rows`.
        dz : float
            The step, m.
        mask : numpy.ndarray
            The in-band bins.
        divisor : numpy.ndarray
            The scale per bin.
        extra : tuple or None, optional
            A varying operator's compensation: its corrector term and its term in
            each defect.

        Returns
        -------
        tuple
            The corrected field and one norm per defect.
        """
        out = np.empty_like(state)
        n_omega = state.shape[-1]
        partials = np.empty((out.size // n_omega, rows.shape[0]))
        if extra is None:
            extra_field = np.zeros((1, 1), dtype=np.complex128)
            extra_defects = np.zeros((1, 1, 1), dtype=np.complex128)
        else:
            extra_field = np.ascontiguousarray(extra[0]).reshape(-1, n_omega)
            extra_defects = np.ascontiguousarray(np.stack(extra[1])).reshape(
                len(extra[1]), -1, n_omega
            )
        self._passes.exponential_correct(
            state.reshape(-1, n_omega),
            flow.reshape(-1, n_omega),
            moments.reshape(moments.shape[0], -1, n_omega),
            source.reshape(-1, n_omega),
            store.reshape(store.shape[0], -1, n_omega),
            slots,
            np.ascontiguousarray(corrector),
            np.ascontiguousarray(rows),
            float(dz),
            mask,
            divisor,
            extra is not None,
            extra_field,
            extra_defects,
            out.reshape(-1, n_omega),
            partials,
        )
        return out, np.sqrt(partials.sum(axis=0))

    def coupled(self) -> CoupledExponentialKernels:
        """Return the compiled passes beside a coupled flow, one per kernel object.

        Returns
        -------
        CoupledExponentialKernels
            A :class:`NumbaCoupledExponentialKernels`.
        """
        if self._coupled is None:
            self._coupled = NumbaCoupledExponentialKernels(parallel=self._parallel)
        return self._coupled


class NumbaCoupledExponentialKernels(
    NumbaExponentialKernels, CoupledExponentialKernels
):
    """The exponential formulation's passes beside a coupled flow, compiled.

    The reference's arithmetic
    (:class:`~solprop.backends.multistep.CoupledExponentialKernels`): the block's flow
    and moments on its eigenbasis route from one compiled pass, the bins off the route
    from the reference's augmented exponential, the predictor and the corrector one
    traversal each, and every sample's compensation carried in one more. These passes
    are serial in either variant: a coupled flow belongs to a guided problem, whose
    state is far below the parallel regime; ``parallel`` chooses the inherited diagonal
    passes' variant.
    """

    def coupled(self) -> CoupledExponentialKernels:
        """Return these passes.

        Returns
        -------
        CoupledExponentialKernels
            This object.
        """
        return self

    def coupled_moments(
        self,
        exponent: _State,
        count: int,
        block: CoupledBlock,
        dz: float,
        *,
        shift: _State | None = None,
    ) -> Any:
        """Return a coupled trial's flow and moments, the route's bins compiled.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``Delta Phi`` over the step.
        count : int
            The moments formed.
        block : CoupledBlock
            The step's frozen block.
        dz : float
            The step, m.
        shift : numpy.ndarray or None, optional
            A carrier frame's ``sigma``; this backend refuses one before a march
            starts, so only the unshifted step reaches it.

        Returns
        -------
        _MatrixMoments
            The flow and the moments.
        """
        return coupled_moments(
            exponent,
            count,
            block,
            dz,
            eigen=_fill_route,
            diagonal=_diagonal_moments,
            carrier=shift,
        )

    def carry_coupled(
        self,
        exponent: _State,
        dz: float,
        rows: tuple[Any, Any, list[np.ndarray | None]],
        slots: np.ndarray,
        spans: np.ndarray,
        block: CoupledBlock,
    ) -> Complex[np.ndarray, "n_used n_batch n_modes n_pol n_omega"]:
        """Return each sample's compensation, carried in one compiled pass.

        The terms are the reference's
        (:meth:`~solprop.backends.multistep.CoupledExponentialKernels.carry_coupled`);
        their flows, every sample's at once, are formed in the block's eigenbasis
        (``_carry_flows``). A block with bins off the route takes the reference's.

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
        basis = block.basis
        if basis is None or not np.all(basis[3]):
            return super().carry_coupled(exponent, dz, rows, slots, spans, block)
        rates, states, weights = rows
        reference = exponent / dz
        terms = np.empty((len(slots), *exponent.shape), dtype=np.complex128)
        for index, slot in enumerate(slots):
            terms[index] = (rates[slot] - reference) * states[slot]
            sample_weights = weights[slot]
            if block.residual is not None and sample_weights is not None:
                terms[index] += block.residual(states[slot], sample_weights)
        left, right, eigenvalues, _ = basis
        _carry_flows(
            terms,
            np.ascontiguousarray(spans, dtype=np.float64),
            np.ascontiguousarray(exponent),
            _block_index(block.active),
            np.ascontiguousarray(exponent[0, 0, 0, block.active]),
            float(dz),
            np.ascontiguousarray(eigenvalues),
            np.ascontiguousarray(left),
            np.ascontiguousarray(right),
        )
        return terms

    def predict(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        flow: Any,
        moments: Any,
        store: _Store,
        slots: Int[np.ndarray, " n_used"],
        coefficients: Float[np.ndarray, "n_used n_c"],
        dz: float,
    ) -> _State:
        """Return the predicted field through the trial's block, in one pass.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state at the anchor.
        flow, moments : _MatrixMoments
            The coupled trial's flow and moments, one object.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows the predictor combines, newest first.
        coefficients : numpy.ndarray
            The predictor's Lagrange coefficients.
        dz : float
            The step, m.

        Returns
        -------
        numpy.ndarray
            The predicted lab-frame field at the end of the step.
        """
        del flow
        out = np.empty_like(state)
        _coupled_predict(
            np.ascontiguousarray(state),
            moments.flow_scalar,
            moments.scalar,
            store,
            np.ascontiguousarray(slots, dtype=np.int64),
            np.ascontiguousarray(coefficients),
            float(dz),
            _block_index(moments.active),
            moments.flow_matrix,
            moments.matrices,
            out,
        )
        return out

    def correct(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        self,
        state: _State,
        flow: Any,
        moments: Any,
        source: _State,
        store: _Store,
        slots: Int[np.ndarray, " n_used"],
        corrector: Float[np.ndarray, "n_width n_c"],
        rows: Float[np.ndarray, "n_defects n_factored"],
        dz: float,
        mask: Bool[np.ndarray, " n_omega"],
        divisor: Float[np.ndarray, " n_omega"],
        extra: tuple[_State, list[_State]] | None = None,
    ) -> tuple[_State, Float[np.ndarray, " n_defects"]]:
        """Return the corrected field and the defects' norms through the block.

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state at the anchor.
        flow, moments : _MatrixMoments
            The coupled trial's flow and moments, one object.
        source : numpy.ndarray
            The untransported source at the predicted field.
        store : numpy.ndarray
            The history rows.
        slots : numpy.ndarray
            The rows after the source, newest first.
        corrector : numpy.ndarray
            The corrector's coefficients over the samples.
        rows : numpy.ndarray
            The defects, factored by
            :meth:`~solprop.backends.numba_cpu.multistep.NumbaExponentialKernels.defect_rows`.
        dz : float
            The step, m.
        mask : numpy.ndarray
            The in-band bins.
        divisor : numpy.ndarray
            The scale per bin.
        extra : tuple or None, optional
            The compensation's corrector term and its term in each defect.

        Returns
        -------
        tuple
            The corrected field and one norm per defect.
        """
        del flow
        out = np.empty_like(state)
        sums = np.empty(rows.shape[0])
        if extra is None:
            extra_field = np.zeros((1, 1, 1, 1), dtype=np.complex128)
            extra_defects = np.zeros((1, 1, 1, 1, 1), dtype=np.complex128)
        else:
            extra_field = np.ascontiguousarray(extra[0])
            extra_defects = np.ascontiguousarray(np.stack(extra[1]))
        _coupled_correct(
            np.ascontiguousarray(state),
            moments.flow_scalar,
            moments.scalar,
            np.ascontiguousarray(source),
            store,
            np.ascontiguousarray(slots, dtype=np.int64),
            np.ascontiguousarray(corrector),
            np.ascontiguousarray(rows),
            float(dz),
            mask,
            divisor,
            extra is not None,
            extra_field,
            extra_defects,
            _block_index(moments.active),
            moments.flow_matrix,
            moments.matrices,
            out,
            sums,
        )
        return out, np.sqrt(sums)


def _block_index(active: Bool[np.ndarray, " n_omega"]) -> Int[np.ndarray, " n_omega"]:
    """Return each bin's index in the block, ``-1`` off it.

    Parameters
    ----------
    active : numpy.ndarray
        The bins that carry a block.

    Returns
    -------
    numpy.ndarray
        ``int64``, shape ``(n_omega,)``.
    """
    index = np.full(active.shape, -1, dtype=np.int64)
    index[active] = np.arange(int(np.count_nonzero(active)), dtype=np.int64)
    return index


def _fill_route(
    shift: np.ndarray,
    dz: float,
    count: int,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    matrices: np.ndarray,
    flow_matrix: np.ndarray,
) -> None:
    """Write the eigenbasis route's flow and moments with :func:`_block_functions`.

    Parameters
    ----------
    shift : numpy.ndarray
        ``Delta Phi`` on each of the block's bins.
    dz : float
        The step, m.
    count : int
        The moments formed.
    basis : tuple of numpy.ndarray
        ``(W, Y, lambda, route)``.
    matrices, flow_matrix : numpy.ndarray
        Written on the route's bins.
    """
    left, right, eigenvalues, route = basis
    _block_functions(
        np.ascontiguousarray(shift),
        float(dz),
        np.ascontiguousarray(eigenvalues),
        np.ascontiguousarray(left),
        np.ascontiguousarray(right),
        np.ascontiguousarray(route),
        int(count),
        1.0 / _FACTORIALS[: count + 2],
        # A flow alone (count 0) forms no series: its terms are never read.
        1.0 / (np.arange(_SERIES_TERMS, dtype=np.float64) + max(count, 1)),
        _FACTORIALS[: max(count, 1)],
        matrices,
        flow_matrix,
    )


def _diagonal_moments(exponent: np.ndarray, count: int) -> np.ndarray:
    """Return the diagonal moments of an exponent off the block, compiled.

    Parameters
    ----------
    exponent : numpy.ndarray
        ``Delta Phi`` on the bins off the block, contiguous.
    count : int
        The moments formed.

    Returns
    -------
    numpy.ndarray
        Shape ``(count, *exponent.shape)``.
    """
    n_last = exponent.shape[-1]
    moments = np.empty((count, *exponent.shape), dtype=np.complex128)
    flow = np.empty(exponent.shape, dtype=np.complex128)
    make_passes(parallel=False).moments(
        exponent.reshape(-1, n_last),
        int(count),
        1.0 / _FACTORIALS[: count + 2],
        1.0 / (np.arange(_SERIES_TERMS, dtype=np.float64) + count),
        _FACTORIALS[:count],
        moments.reshape(count, -1, n_last),
        flow.reshape(-1, n_last),
    )
    return moments
