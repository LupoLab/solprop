# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage RK4IP march's register kernels and resident engine on ``numba_cpu``.

SEP 24 phase 2
(``docs/dev/seps/archive/sep-0024-low-storage-interaction-picture-runge-kutta.md``,
sections 4.3 and 6.6). :class:`NumbaLowStorage` is the backend's own
:class:`~solprop.backends.low_storage.LowStorageEngine`: it holds the accumulator ``B``
and the stage register ``C`` beside the state ``A`` and, tabulating, the flow table,
and makes a step's four passes through the route's own source pipeline with **no
Runge-Kutta buffer** -- no stage rows, accumulators, accept buffer, recycled stage,
propagator pair or exponent.

Each pass is three parts:

1. **The input.** One traversal reads one register, adds the ``modified`` noise draw at
   the pass's position and writes the pipeline's input: the padded modal spectrum, the
   general route's coarse lab field, the Cartesian transverse scratch or the planar
   column volume. On the separable kinds the draw's phase ``z Im D`` is formed per
   element from the operator's one-dimensional arrays, so no rotated copy is held.
2. **The core.** The route's transforms, response columns and in-stage absorber, shared
   with the Runge-Kutta march (``NumbaCpuBackend._free_pass`` and its siblings), leave
   the source in frequency.
3. **The finish.** One traversal forms the source at each element (the prefactor, the
   window, the absorber term) and folds it into the registers with the half-step flows
   (``define_bodies``' ``low_storage_fold``). The flows come from the table or, on the
   separable kinds when recomputing, from the same per-element body that gives the
   prefactor. The last pass also writes one partial of the accepted state's squared
   norm per column, summed by the engine.

**The in-stage exchange** (section 4.3). A Cartesian or line pass under an in-stage
absorber evaluates twice from the same input, the material response and then the
absorber's term. The first finish folds the material source and, in the same
traversal, re-forms the input in the transverse scratch from the register it came from,
so the material output never needs a row of its own; the second finish folds the
absorber's term. The modal, general and planar routes form both terms from one input
(the planar volume carries the absorber's half), so their passes make one finish.

**The elided stage register** (Q4). A Cartesian or line pass that reads its input
once -- a material response and no in-stage absorber -- needs no ``C``: the finish of
each of the first three passes writes the next pass's input, noise and all, straight
into the transverse scratch (``S <- A + h/2 S + eta(z_m)`` and its siblings), which
saves one state volume and the next pass's input traversal
(:func:`solprop.backends.memory.low_storage_stage_register`).

**Kernels are specialised by closure constant** where a flag decides what is read --
the noise, the crop's contiguity, the in-stage absorber term, the column threading --
and take the pass, the fold mode, the term and the flow treatment as arguments: those
branches are invariant over the traversal. The column loop is a ``prange`` in the
parallel variants, which are never disk-cached (``docs/dev/backends.md`` section 3.2);
each element's arithmetic is the same in either, and the norm is two-stage, one partial
per column.
"""

import math
from functools import cache
from typing import Any, NamedTuple, no_type_check

import numpy as np
from jaxtyping import Complex, Float
from numba import prange

from solprop._numba_cache import njit
from solprop.backends.bodies import FOLD_COMPLETE, FOLD_EXCHANGE, FOLD_SECOND
from solprop.backends.low_storage import schedule_key, table_steps
from solprop.backends.memory import (
    low_storage_flow_scratch,
    low_storage_flows,
    low_storage_stage_register,
)
from solprop.backends.numba_cpu.kernels import HOST
from solprop.stepper.low_storage import LowStorageRungeKutta

__all__ = [
    "FreeKernels",
    "ModalKernels",
    "NumbaLowStorage",
    "SlabKernels",
    "exponentiate_forward",
    "exponentiate_terms",
    "make_free_kernels",
    "make_modal_kernels",
    "make_slab_kernels",
]

_FOLD = HOST.low_storage_fold
_ELEMENT = HOST.spatial_element
_SLAB = HOST.slab_element
_PAIR = HOST.propagator_pair

_TERM_NONE = 0
_TERM_MATERIAL = 1
_TERM_ABSORBER = 2


class ModalKernels(NamedTuple):
    """The modal and general routes' register input and finish.

    Attributes
    ----------
    input : callable
        One register plus the drawn noise, padded onto the fine grid.
    fold : callable
        The source at each element folded into the registers.
    """

    input: Any
    fold: Any


class FreeKernels(NamedTuple):
    """The Cartesian and line routes' register input, finish and flow table.

    Attributes
    ----------
    input : callable
        One register plus the rotated noise, into the transverse scratch.
    fold : callable
        The material or absorber source folded into the registers, with the
        exchange's re-formed input.
    table : callable
        ``exp(distance D)`` at every element, one volume.
    """

    input: Any
    fold: Any
    table: Any


class SlabKernels(NamedTuple):
    """The hybrid planar route's register input, finish and flow table.

    Attributes
    ----------
    input : callable
        One register plus the rotated noise, into the column volume.
    fold : callable
        Both halves' source folded into the registers.
    table : callable
        ``exp(distance D)`` at every element, one volume.
    """

    input: Any
    fold: Any
    table: Any


@njit(cache=True, nogil=True)
@no_type_check
def exponentiate_forward(
    exponent: Complex[np.ndarray, "n_columns n_omega"],
    out: Complex[np.ndarray, "n_columns n_omega"],
) -> None:
    """Form ``exp(dPhi)`` alone, from one real ``exp`` and one ``sincos``.

    Parameters
    ----------
    exponent : numpy.ndarray
        ``dPhi``, complex, shape ``(n_columns, n_omega)``.
    out : numpy.ndarray
        Written with ``exp(dPhi)``; may be `exponent` itself.
    """
    for column in range(exponent.shape[0]):
        for j in range(exponent.shape[1]):
            value = exponent[column, j]
            real, imag, _, _ = _PAIR(value.real, value.imag)
            out[column, j] = complex(real, imag)


@njit(cache=True, nogil=True)
@no_type_check
def exponentiate_terms(
    weights: Float[np.ndarray, " n_terms"],
    terms: Complex[np.ndarray, "n_terms n_columns n_omega"],
    out: Complex[np.ndarray, "n_columns n_omega"],
) -> None:
    """Form ``exp(sum_t w_t Phi_t)`` per element, with no exponent array.

    The weighted sum is ``solprop.translate.stage.combine``'s, accumulated element by
    element: a single term weighted by one is the term itself, and a single term's
    product is one rounding, so a constant operator's ``D (z1 - z0)`` is formed as
    there.

    Parameters
    ----------
    weights : numpy.ndarray
        The bundle's ``phi_delta``, float64, ``(n_terms,)``.
    terms : numpy.ndarray
        The terms of ``Phi``, complex, ``(n_terms, n_columns, n_omega)``.
    out : numpy.ndarray
        Written with the flow, ``(n_columns, n_omega)``.
    """
    n_terms = terms.shape[0]
    for column in range(out.shape[0]):
        for j in range(out.shape[1]):
            real = 0.0
            imag = 0.0
            for term in range(n_terms):
                weight = weights[term]
                value = terms[term, column, j]
                real += weight * value.real
                imag += weight * value.imag
            forward_real, forward_imag, _, _ = _PAIR(real, imag)
            out[column, j] = complex(forward_real, forward_imag)


@cache
def make_modal_kernels(
    *, contiguous: bool, noise: bool, absorber: bool, parallel: bool
) -> ModalKernels:
    """Compile the modal routes' register input and finish.

    Parameters
    ----------
    contiguous : bool
        Whether the crop index is the leading block of the fine spectrum.
    noise : bool
        Whether the ``modified`` noise draw is added to every input.
    absorber : bool
        Whether the in-stage absorber's term is added to the source.
    parallel : bool
        Whether independent columns execute in parallel.

    Returns
    -------
    ModalKernels
        The compiled pair.
    """
    columns = prange if parallel else range

    @no_type_check
    def register_input(
        register: np.ndarray,
        drawn: np.ndarray,
        crop: Any,
        fine: np.ndarray,
        scale: float,
    ) -> None:
        """Pad one register and the drawn noise onto the fine grid, every bin written.

        Parameters
        ----------
        register : numpy.ndarray
            The pass's input register, ``(n_columns, n_omega)``.
        drawn : numpy.ndarray
            The noise at the pass's position, or any array when there is none.
        crop : numpy.ndarray
            Where each coarse bin sits in `fine`, ``numpy.intp``.
        fine : numpy.ndarray
            Written with the padded spectrum, the pad zeroed.
        scale : float
            The inverse transform's normalisation.
        """
        n_columns, n_omega = register.shape
        n_fine = fine.shape[1]
        for column in columns(n_columns):
            for j in range(n_omega if contiguous else 0, n_fine):
                fine[column, j] = 0.0 + 0.0j
            for j in range(n_omega):
                value = register[column, j]
                if noise:
                    value = value + drawn[column, j]
                fine[column, crop[j]] = value * scale

    @no_type_check
    def fold(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        stage: int,
        step: float,
        active: bool,
        spectrum: np.ndarray,
        crop: Any,
        prefactor: np.ndarray,
        window: np.ndarray,
        damping: np.ndarray,
        registers: tuple[np.ndarray, np.ndarray, np.ndarray],
        flows: tuple[np.ndarray, np.ndarray],
        mask: np.ndarray,
        partials: np.ndarray,
    ) -> None:
        """Form the source at each element and fold it into the registers.

        Parameters
        ----------
        stage : int
            The pass, 0 to 3.
        step : float
            The step, m.
        active : bool
            Whether a source was evaluated; a pass with neither a response nor an
            absorber folds zero.
        spectrum : numpy.ndarray
            The source on the fine grid (or the coarse grid with `crop` the identity).
        crop : numpy.ndarray
            Where each coarse bin sits in `spectrum`.
        prefactor : numpy.ndarray
            The UPPE prefactor at the pass's position, ``(n_columns, n_omega)``.
        window : numpy.ndarray
            The spectral window.
        damping : numpy.ndarray
            The in-stage absorber's term on the coarse grid when `absorber` was
            compiled in; never read otherwise.
        registers : tuple of numpy.ndarray
            ``A``, ``B`` and ``C``, updated in place.
        flows : tuple of numpy.ndarray
            ``P`` and ``Q``.
        mask : numpy.ndarray
            The norm mask, boolean, ``(n_omega,)``.
        partials : numpy.ndarray
            Written with each column's masked squared norm of the accepted state on
            the last pass.
        """
        state, accumulator, register = registers
        first, second = flows
        n_columns, n_omega = state.shape
        for column in columns(n_columns):
            total = 0.0
            for j in range(n_omega):
                source = 0.0 + 0.0j
                if active:
                    index = j if contiguous else crop[j]
                    source = prefactor[column, j] * window[j] * spectrum[column, index]
                    if absorber:
                        source = source + damping[column, j]
                a, b, c = _FOLD(
                    stage,
                    FOLD_COMPLETE,
                    step,
                    source,
                    state[column, j],
                    accumulator[column, j],
                    register[column, j],
                    first[column, j],
                    second[column, j],
                )
                state[column, j] = a
                accumulator[column, j] = b
                register[column, j] = c
                if stage == 3 and mask[j]:
                    total += a.real * a.real + a.imag * a.imag
            if stage == 3:
                partials[column] = total

    compile_ = njit(cache=not parallel, nogil=True, parallel=parallel)
    return ModalKernels(compile_(register_input), compile_(fold))


@cache
def make_free_kernels(*, noise: bool, parallel: bool) -> FreeKernels:  # noqa: PLR0915 - three kernels' bodies
    """Compile the Cartesian and line routes' register input, finish and table.

    Parameters
    ----------
    noise : bool
        Whether the ``modified`` noise draw is added to every input.
    parallel : bool
        Whether independent columns execute in parallel.

    Returns
    -------
    FreeKernels
        The compiled kernels.
    """
    columns = prange if parallel else range

    @no_type_check
    def register_input(
        data: Any,
        register: np.ndarray,
        drawn: np.ndarray,
        position: float,
        scale: float,
        scratch: np.ndarray,
    ) -> None:
        """Write one register and the rotated noise into the transverse scratch.

        Parameters
        ----------
        data : tuple
            The separable record (``NumbaCpuBackend._separable``).
        register : numpy.ndarray
            The pass's input register, ``(n_columns, n_omega)``.
        drawn : numpy.ndarray
            The unadvanced noise realisation, or any array when there is none.
        position : float
            The pass's position, m, where the noise's phase is taken.
        scale : float
            The transverse inverse transform's normalisation, ``1 / n_nodes``.
        scratch : numpy.ndarray
            The transverse scratch, overwritten.
        """
        n_pol, kperp = data[10], data[11]
        for column in columns(register.shape[0]):
            for j in range(register.shape[1]):
                value = register[column, j]
                if noise:
                    rate, _ = _ELEMENT(
                        data[0][j],
                        data[1][column // n_pol],
                        data[2][j],
                        data[3][column // n_pol],
                        data[4][j],
                        data[5][j],
                        data[6][j],
                        data[7][j],
                        data[8][j],
                        data[9],
                        kperp[column // n_pol],
                    )
                    angle = position * rate.imag
                    value = value + drawn[column, j] * complex(
                        math.cos(angle), math.sin(angle)
                    )
                scratch[column, j] = value * scale

    @no_type_check
    def fold(  # noqa: PLR0912, PLR0913, PLR0915, PLR0917 - one fused traversal
        data: Any,
        pass_: tuple[int, int, int],
        step: float,
        scratch: np.ndarray,
        window: np.ndarray,
        active: np.ndarray,
        registers: tuple[np.ndarray, np.ndarray, np.ndarray],
        table: np.ndarray,
        tabulated: bool,
        reform: tuple[np.ndarray, float, float, bool],
        mask: np.ndarray,
        partials: np.ndarray,
    ) -> None:
        """Fold the transformed source into the registers, re-forming at the exchange.

        Parameters
        ----------
        data : tuple
            The separable record.
        pass_ : tuple of int
            The pass (0 to 3), the fold mode (``bodies.FOLD_*``) and the term: 0
            none, 1 the material source ``prefactor * window * S``, 2 the absorber's
            ``mask * S``.
        step : float
            The step, m.
        scratch : numpy.ndarray
            The transverse scratch holding the source in frequency; overwritten at the
            exchange with the pass's input again.
        window : numpy.ndarray
            The spectral window.
        active : numpy.ndarray
            The active band, boolean, ``(n_omega,)``.
        registers : tuple of numpy.ndarray
            ``A``, ``B`` and ``C``, updated in place.
        table : numpy.ndarray
            ``exp(h D / 2)`` when `tabulated`; never read otherwise.
        tabulated : bool
            Whether the flow is read from `table` or formed here.
        reform : tuple
            The unadvanced noise, a position, m, the transverse inverse transform's
            normalisation, and whether the stage register is elided: at the exchange
            the pass's input is re-formed at the pass's own position; elided, each
            of the first three passes writes the next pass's input, at the next
            position, into the scratch in place of ``C``.
        mask : numpy.ndarray
            The norm mask, boolean.
        partials : numpy.ndarray
            Written with each column's masked squared norm on the last pass's final
            fold.
        """
        stage, mode, term = pass_
        state, accumulator, register = registers
        drawn, position, scale, forward = reform
        n_pol, kperp = data[10], data[11]
        half = 0.5 * step
        exchange = mode == FOLD_EXCHANGE
        second = mode == FOLD_SECOND
        final = stage == 3 and not exchange
        # What this pass reads and writes, invariant over the traversal: each fold
        # moves only the registers its row of the flow table names.
        flows = (
            stage == 0 or (stage == 2 and not exchange) or (stage == 3 and not second)
        )
        reads_register = (second and stage < 3) or (exchange and stage > 0)
        reads_accumulator = stage > 0 or second
        writes_state = (stage == 0 and not second) or stage == 3
        writes_register = stage < 3 and not forward
        reforms = exchange or (forward and stage < 3)
        element = (
            term == _TERM_MATERIAL or (flows and not tabulated) or (reforms and noise)
        )
        for column in columns(state.shape[0]):
            total = 0.0
            for j in range(state.shape[1]):
                rate = 0.0 + 0.0j
                prefactor = 0.0 + 0.0j
                if element:
                    rate, prefactor = _ELEMENT(
                        data[0][j],
                        data[1][column // n_pol],
                        data[2][j],
                        data[3][column // n_pol],
                        data[4][j],
                        data[5][j],
                        data[6][j],
                        data[7][j],
                        data[8][j],
                        data[9],
                        kperp[column // n_pol],
                    )
                source = 0.0 + 0.0j
                if term == _TERM_MATERIAL:
                    source = prefactor * window[j] * scratch[column, j]
                elif term == _TERM_ABSORBER and active[j]:
                    source = scratch[column, j]
                flow = 1.0 + 0.0j
                if flows and tabulated:
                    flow = table[column, j]
                elif flows:
                    exponent = half * rate
                    real, imag, _, _ = _PAIR(exponent.real, exponent.imag)
                    flow = complex(real, imag)
                a = state[column, j]
                b = accumulator[column, j] if reads_accumulator else 0.0 + 0.0j
                c = register[column, j] if reads_register else 0.0 + 0.0j
                if exchange:
                    value = a if stage == 0 else c
                    if noise:
                        angle = position * rate.imag
                        value = value + drawn[column, j] * complex(
                            math.cos(angle), math.sin(angle)
                        )
                    scratch[column, j] = value * scale
                a, b, c = _FOLD(stage, mode, step, source, a, b, c, flow, flow)
                if forward and stage < 3:
                    # The next pass's input straight into the scratch, the stage
                    # register elided (SEP 24 Q4).
                    value = c
                    if noise:
                        angle = position * rate.imag
                        value = value + drawn[column, j] * complex(
                            math.cos(angle), math.sin(angle)
                        )
                    scratch[column, j] = value * scale
                if writes_state:
                    state[column, j] = a
                if stage < 3:
                    accumulator[column, j] = b
                if writes_register:
                    register[column, j] = c
                if final and mask[j]:
                    total += a.real * a.real + a.imag * a.imag
            if final:
                partials[column] = total

    @no_type_check
    def table(data: Any, distance: float, out: np.ndarray) -> None:
        """Write ``exp(distance D)`` at every element.

        Parameters
        ----------
        data : tuple
            The separable record.
        distance : float
            Propagation increment, m.
        out : numpy.ndarray
            The table, ``(n_columns, n_omega)``, overwritten.
        """
        n_pol, kperp = data[10], data[11]
        for column in columns(out.shape[0]):
            for j in range(out.shape[1]):
                rate, _ = _ELEMENT(
                    data[0][j],
                    data[1][column // n_pol],
                    data[2][j],
                    data[3][column // n_pol],
                    data[4][j],
                    data[5][j],
                    data[6][j],
                    data[7][j],
                    data[8][j],
                    data[9],
                    kperp[column // n_pol],
                )
                exponent = distance * rate
                real, imag, _, _ = _PAIR(exponent.real, exponent.imag)
                out[column, j] = complex(real, imag)

    compile_ = njit(cache=not parallel, nogil=True, parallel=parallel)
    return FreeKernels(compile_(register_input), compile_(fold), compile_(table))


@cache
def make_slab_kernels(*, noise: bool, parallel: bool) -> SlabKernels:  # noqa: PLR0915 - three kernels' bodies
    """Compile the hybrid planar route's register input, finish and table.

    Parameters
    ----------
    noise : bool
        Whether the ``modified`` noise draw is added to every input.
    parallel : bool
        Whether independent columns execute in parallel.

    Returns
    -------
    SlabKernels
        The compiled kernels.
    """
    columns = prange if parallel else range

    @no_type_check
    def register_input(
        data: Any,
        register: np.ndarray,
        drawn: np.ndarray,
        position: float,
        scale: float,
        rows: np.ndarray,
    ) -> None:
        """Write one register and the rotated noise into the column volume.

        Parameters
        ----------
        data : tuple
            :func:`solprop.backends.numba_cpu.planar.slab_data`'s record.
        register : numpy.ndarray
            The pass's input register, ``(n_columns, n_omega)``.
        drawn : numpy.ndarray
            The unadvanced noise realisation, or any array when there is none.
        position : float
            The pass's position, m.
        scale : float
            The y inverse transform's normalisation, ``1 / n_y``.
        rows : numpy.ndarray
            The volume's material half, overwritten.
        """
        n_pol, n_channels = data[11], data[12]
        for column in columns(register.shape[0]):
            mode = column // n_pol
            channel = mode % n_channels
            for j in range(register.shape[1]):
                value = register[column, j]
                if noise:
                    rate, _ = _SLAB(
                        data[0][j],
                        data[1][mode],
                        data[2][j],
                        data[3][mode],
                        data[4][channel, j],
                        data[5][j],
                        data[6][j],
                        data[7][j],
                        data[8][j],
                        data[9][j],
                        data[10],
                        data[13][mode],
                    )
                    angle = position * rate.imag
                    value = value + drawn[column, j] * complex(
                        math.cos(angle), math.sin(angle)
                    )
                rows[column, j] = value * scale

    @no_type_check
    def fold(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        data: Any,
        stage: int,
        step: float,
        terms: tuple[bool, bool],
        sources: tuple[np.ndarray, np.ndarray],
        window: np.ndarray,
        active: np.ndarray,
        registers: tuple[np.ndarray, np.ndarray, np.ndarray],
        table: np.ndarray,
        tabulated: bool,
        mask: np.ndarray,
        partials: np.ndarray,
    ) -> None:
        """Fold ``prefactor * window * material + mask * absorbed`` into the registers.

        Parameters
        ----------
        data : tuple
            The planar record.
        stage : int
            The pass, 0 to 3.
        step : float
            The step, m.
        terms : tuple of bool
            Whether the material and the absorber terms are present.
        sources : tuple of numpy.ndarray
            The material and the absorber halves in frequency.
        window : numpy.ndarray
            The spectral window.
        active : numpy.ndarray
            The active band, boolean.
        registers : tuple of numpy.ndarray
            ``A``, ``B`` and ``C``, updated in place.
        table : numpy.ndarray
            ``exp(h D / 2)`` when `tabulated`.
        tabulated : bool
            Whether the flow is read from `table` or formed here.
        mask : numpy.ndarray
            The norm mask, boolean.
        partials : numpy.ndarray
            Written with each column's masked squared norm on the last pass.
        """
        material, absorbed = sources
        state, accumulator, register = registers
        n_pol, n_channels = data[11], data[12]
        half = 0.5 * step
        element = terms[0] or not tabulated
        for column in columns(state.shape[0]):
            mode = column // n_pol
            channel = mode % n_channels
            total = 0.0
            for j in range(state.shape[1]):
                rate = 0.0 + 0.0j
                prefactor = 0.0 + 0.0j
                if element:
                    rate, prefactor = _SLAB(
                        data[0][j],
                        data[1][mode],
                        data[2][j],
                        data[3][mode],
                        data[4][channel, j],
                        data[5][j],
                        data[6][j],
                        data[7][j],
                        data[8][j],
                        data[9][j],
                        data[10],
                        data[13][mode],
                    )
                source = 0.0 + 0.0j
                if terms[0]:
                    source = prefactor * window[j] * material[column, j]
                if terms[1] and active[j]:
                    source = source + absorbed[column, j]
                if tabulated:
                    flow = table[column, j]
                else:
                    exponent = half * rate
                    real, imag, _, _ = _PAIR(exponent.real, exponent.imag)
                    flow = complex(real, imag)
                a, b, c = _FOLD(
                    stage,
                    FOLD_COMPLETE,
                    step,
                    source,
                    state[column, j],
                    accumulator[column, j],
                    register[column, j],
                    flow,
                    flow,
                )
                state[column, j] = a
                accumulator[column, j] = b
                register[column, j] = c
                if stage == 3 and mask[j]:
                    total += a.real * a.real + a.imag * a.imag
            if stage == 3:
                partials[column] = total

    @no_type_check
    def table(data: Any, distance: float, out: np.ndarray) -> None:
        """Write ``exp(distance D)`` at every element.

        Parameters
        ----------
        data : tuple
            The planar record.
        distance : float
            Propagation increment, m.
        out : numpy.ndarray
            The table, ``(n_columns, n_omega)``, overwritten.
        """
        n_pol, n_channels = data[11], data[12]
        for column in columns(out.shape[0]):
            mode = column // n_pol
            channel = mode % n_channels
            for j in range(out.shape[1]):
                rate, _ = _SLAB(
                    data[0][j],
                    data[1][mode],
                    data[2][j],
                    data[3][mode],
                    data[4][channel, j],
                    data[5][j],
                    data[6][j],
                    data[7][j],
                    data[8][j],
                    data[9][j],
                    data[10],
                    data[13][mode],
                )
                exponent = distance * rate
                real, imag, _, _ = _PAIR(exponent.real, exponent.imag)
                out[column, j] = complex(real, imag)

    compile_ = njit(cache=not parallel, nogil=True, parallel=parallel)
    return SlabKernels(compile_(register_input), compile_(fold), compile_(table))


class NumbaLowStorage:
    """The resident low-storage engine: the four register passes on the route's kernels.

    ``numba_cpu``'s :class:`~solprop.backends.low_storage.LowStorageEngine` (SEP 24
    section 6.6): the backend's state is ``A``, updated in place, and this engine
    holds ``B``, ``C`` (unless it is elided, Q4) and the flows, allocated at
    construction so that the backend's ledger names them before the first step.

    **The flows.** Tabulating a constant operator holds one volume, ``exp(h_t D / 2)``
    for the key ``h_t`` of :func:`~solprop.backends.low_storage.table_steps` (the
    step's run's mean, SEP 24 Q10), built from the operator's origin as the host engine
    builds it and rebuilt only when the key changes. A varying modal operator's ``P``
    and ``Q`` are formed once per step into two held volumes. Recomputing, the
    Cartesian, line and planar finishes form ``exp(h D / 2)`` per element from the
    same body that gives the prefactor and hold nothing; a modal route forms the step's
    ``P``, then ``Q`` over it before pass 3, in one held scratch volume
    (:func:`~solprop.backends.memory.low_storage_flow_scratch`). A modal flow is
    exponentiated per element from the terms of ``Phi`` (:func:`exponentiate_terms`),
    so no exponent array exists outside the ledger either.

    **The stiffness indicator** (SEP 24 Q11, decided in phase 2) is not formed:
    ``stiffness`` is NaN, which silences the march's warning. A resident engine
    holds ``F2`` and ``F3`` exactly but ``g1`` and ``F2`` only as ``2 (C - A) / h``,
    which on ``numpy_ref``'s slopes agreed with the exact indicator to 1e-7 relative on
    a soliton and departed only on a nearly linear step, far below 2.785; but on the
    routes the march is for, the Cartesian and line kinds, ``C`` is elided (Q4) or the
    slope is split between the exchange's two evaluations, and recovering it would take
    the volume the march exists to save. The phase 2 report records the measurement.

    Parameters
    ----------
    backend : solprop.backends.numba_cpu.backend.NumbaCpuBackend
        The backend whose state, plans and route this engine steps.
    method : solprop.stepper.low_storage.LowStorageRungeKutta
        The march.

    Attributes
    ----------
    stiffness : float
        Always NaN: not formed on this backend.
    atomic : bool
        Always ``False``: each pass is its own compiled call, between which a signal
        may arrive.
    """

    def __init__(self, backend: Any, method: LowStorageRungeKutta) -> None:
        """Hold the registers and the flows, and choose the route's kernels."""
        self._backend = backend
        self._method = method
        self.stiffness = math.nan
        self.atomic = False
        problem = backend._problem
        shape = (backend._n_columns, backend._n_omega)
        self._accumulator = np.zeros(shape, dtype=np.complex128)
        # The stage register is elided where the pass reads its input once (SEP 24
        # Q4): each finish writes the next pass's input into the scratch instead.
        self._elided = not low_storage_stage_register(problem, "numba_cpu")
        self._register = np.zeros(
            (0, 0) if self._elided else shape, dtype=np.complex128
        )
        self._partials = np.zeros(shape[0])
        self._separable = backend._separable is not None or backend._slab
        # The estimate's own count of the flows (one shared predicate), which is
        # `StepDriver._propagator_mode`'s rule for this march: the backend cannot be
        # asked yet, since it reads the march this engine is being built for.
        held = low_storage_flows(problem, "numba_cpu")
        self._flows = np.zeros((held, *shape), dtype=np.complex128)
        self._tabulating = held > 0
        # Recomputing on a modal route, `P` and `Q` are formed in turn in one held
        # volume, pass 1 reading `P` and passes 3 and 4 `Q` (`low_storage_fold`).
        self._scratch = np.zeros(
            shape if low_storage_flow_scratch(problem, "numba_cpu") else (0, 0),
            dtype=np.complex128,
        )
        self._table_step = math.nan
        schedule = problem.stepper.schedule
        assert schedule is not None  # the translation refused the march without one
        self._keys, self._edges = table_steps(schedule), schedule.edges
        noise = problem.initial.noise_model == "modified"
        parallel = backend._parallel_spatial
        self._modal = make_modal_kernels(
            contiguous=backend._contiguous,
            noise=noise,
            absorber=backend._in_stage_absorber,
            parallel=parallel,
        )
        self._free = make_free_kernels(noise=noise, parallel=parallel)
        self._slab = make_slab_kernels(noise=noise, parallel=parallel)

    def held(self) -> dict[str, int]:
        """Return the bytes of every array the engine holds, by ledger row.

        Returns
        -------
        dict
            ``register_accumulator``, ``register_stage`` and ``propagators`` (the
            table, a varying operator's pair, or a recomputing modal route's flow
            scratch), zero once :meth:`close` ran.
        """
        return {
            "register_accumulator": self._accumulator.nbytes,
            "register_stage": self._register.nbytes,
            "propagators": self._flows.nbytes + self._scratch.nbytes,
        }

    def reset(self) -> None:
        """Drop the flow table's key: a resume builds it again from its own step."""
        self._table_step = math.nan
        self.stiffness = math.nan

    def close(self) -> None:
        """Release the registers and the flows."""
        self.reset()
        empty = np.zeros((0, 0), dtype=np.complex128)
        self._accumulator = self._register = self._scratch = empty
        self._flows = np.zeros((0, 0, 0), dtype=np.complex128)
        self._partials = np.zeros(0)

    def step(self, z: float, dz: float) -> float:
        """Take one RK4IP step in four register passes, accepting it in place.

        Parameters
        ----------
        z : float
            Where the step starts, m.
        dz : float
            The step, m, positive.

        Returns
        -------
        float
            The accepted state's squared norm over the norm mask, before the boundary
            treatments.
        """
        backend = self._backend
        midpoint, end = z + 0.5 * dz, z + dz
        flows = self._prepare(z, dz)
        state = backend._state
        if not state.flags.writeable:
            # An observation holds the accepted state read-only; it keeps its own.
            state = state.copy()
            backend._state = state
        registers = (
            state.reshape(backend._n_columns, backend._n_omega),
            self._accumulator,
            self._register,
        )
        positions = (z, midpoint, midpoint, end)
        for stage, position in enumerate(positions):
            bundle = backend._stage_scalars(position, position)
            source = registers[0] if stage == 0 else registers[2]
            if backend._slab:
                self._slab_pass(stage, dz, source, bundle, registers, flows)
            elif backend._free:
                following = positions[min(stage + 1, 3)]
                self._free_pass(
                    (stage, dz, following), source, bundle, registers, flows
                )
            else:
                if stage == 2 and self._scratch.size:
                    # Pass 2 read no flow; passes 3 and 4 read `Q` from the scratch.
                    self._exponentiate(
                        backend._stage_scalars(midpoint, end), self._scratch
                    )
                self._modal_pass(stage, dz, source, bundle, registers, flows)
        return float(np.sum(self._partials))

    def _prepare(self, z: float, dz: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Read the step's bundles and return the flows held for it.

        Parameters
        ----------
        z : float
            Where the step starts, m.
        dz : float
            The step, m.

        Returns
        -------
        tuple of numpy.ndarray or None
            ``P`` and ``Q`` as arrays, or ``None`` where a separable finish forms them
            per element.
        """
        backend = self._backend
        midpoint, end = z + 0.5 * dz, z + dz
        # One held volume is a constant operator's table (`low_storage_flows`).
        table = self._flows.shape[0] == 1
        backend._hoist(z)
        # Each evaluation reads its position as its own anchor; a modal flow formed
        # for the step reads its interval.
        modal = not self._separable and not table
        backend._stage_bundles(z, [z, midpoint] if modal else [z])
        backend._stage_bundles(midpoint, [midpoint, end] if modal else [midpoint])
        backend._stage_bundles(end, [end])
        if table:
            key = schedule_key(self._keys, self._edges, z, dz)
            if key != self._table_step:
                timers = backend._pass_timers
                mark = timers.now() if timers is not None else 0.0
                self._build_table(key)
                if timers is not None:
                    timers.lap("ls.table", mark)
                self._table_step = key
            return self._flows[0], self._flows[0]
        if self._separable:
            return None
        if not self._tabulating:
            # `Q` overwrites `P` in the scratch before pass 3 (`step`).
            self._exponentiate(backend._stage_scalars(z, midpoint), self._scratch)
            return self._scratch, self._scratch
        first, second = self._flows[0], self._flows[1]
        self._exponentiate(backend._stage_scalars(z, midpoint), first)
        self._exponentiate(backend._stage_scalars(midpoint, end), second)
        return first, second

    def _build_table(self, key: float) -> None:
        """Build ``exp(h_t D / 2)`` into the held table, from the operator's origin.

        Parameters
        ----------
        key : float
            The table's step ``h_t``, m.
        """
        backend = self._backend
        if backend._slab:
            self._slab.table(backend._slab_data, 0.5 * key, self._flows[0])
            return
        if backend._separable is not None:
            self._free.table(backend._separable, 0.5 * key, self._flows[0])
            return
        # The increment of a constant operator is the distance itself, so the table
        # is formed from the origin, not from fl(z + h/2) - z (the host engine's rule).
        origin = backend._problem.linear.z_span[0]
        half = origin + 0.5 * key
        backend._stage_bundles(origin, [half])
        self._exponentiate(backend._stage_scalars(origin, half), self._flows[0])

    def _exponentiate(self, bundle: Any, out: np.ndarray) -> None:
        """Form a modal flow ``exp(Phi(z1) - Phi(z0))`` into `out`.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The interval's scalars.
        out : numpy.ndarray
            Written with the flow, ``(n_columns, n_omega)``.
        """
        backend = self._backend
        host = backend._host_phi
        if host is not None:
            # The z table's difference lands in its own held result array.
            exponentiate_forward(
                host.delta(bundle.z0, bundle.z1).reshape(out.shape), out
            )
            return
        # Formed per element from the terms: no exponent array exists.
        phi, weights = backend._terms.phi, bundle.phi_delta
        assert phi is not None and weights is not None  # no z table: the terms hold Phi
        exponentiate_terms(weights, phi.reshape(phi.shape[0], *out.shape), out)

    def _modal_pass(
        self,
        stage: int,
        dz: float,
        source: np.ndarray,
        bundle: Any,
        registers: tuple[np.ndarray, np.ndarray, np.ndarray],
        flows: Any,
    ) -> None:
        """Evaluate one pass on the modal or general route and fold it.

        Parameters
        ----------
        stage : int
            The pass, 0 to 3.
        dz : float
            The step, m.
        source : numpy.ndarray
            The register the input is formed from.
        bundle : solprop.translate.stage.StageScalars
            The pass's scalars, anchored at its position.
        registers : tuple of numpy.ndarray
            ``A``, ``B`` and ``C``.
        flows : tuple of numpy.ndarray
            ``P`` and ``Q``.
        """
        backend = self._backend
        plans, grids = backend._plans, backend._problem.grids
        general = backend._general
        active = backend._nonlinear or backend._in_stage_absorber or general is not None
        spectrum, crop = plans.projected_spectrum, backend._crop
        prefactor = damping = source
        if active:
            noise = backend._noise_at(bundle)
            drawn = (
                source
                if noise is None
                else np.ascontiguousarray(noise.reshape(source.shape))
            )
            backend._last_bundle = bundle
            if general is None:
                self._modal.input(
                    source, drawn, crop, plans.modal_spectrum, backend._pad_scale
                )
                if backend._spec.family == "field_analytic":
                    # The analytic signal's one-sided spectrum, as the Runge-Kutta
                    # march's ``_double_positive_bins`` forms it.
                    fine = plans.modal_spectrum
                    fine[:, 1 : min(backend._n_omega, fine.shape[1] // 2)] *= 2.0
                backend._modal_to_time(bundle)
                if backend._nonlinear:
                    backend._polarisation_spectrum(bundle)
                else:
                    spectrum[:] = 0.0
            else:
                if backend._in_stage_absorber:
                    self._modal.input(
                        source, drawn, crop, plans.modal_spectrum, backend._pad_scale
                    )
                    plans.to_time.execute()
                crop = backend._coarse_index
                if general.responds:
                    self._modal.input(source, drawn, crop, general.lab, 1.0)
                    spectrum = general.source(
                        bundle.transform_weights, bundle.dispersive_kerr_densities
                    )
                else:
                    spectrum = general.output
                    spectrum[:] = 0.0
            damping = backend._absorber_term()
            prefactor = backend._prefactor_for(bundle)
        self._modal.fold(
            stage,
            dz,
            active,
            spectrum,
            crop,
            prefactor,
            grids.omega_window,
            damping,
            registers,
            flows,
            grids.norm_mask,
            self._partials,
        )

    def _free_pass(
        self,
        step: tuple[int, float, float],
        source: np.ndarray,
        bundle: Any,
        registers: tuple[np.ndarray, np.ndarray, np.ndarray],
        flows: Any,
    ) -> None:
        """Evaluate one Cartesian or line pass, with the exchange, and fold it.

        Parameters
        ----------
        step : tuple
            The pass (0 to 3), the step, m, and the next pass's position, m, where
            an elided stage register's input is formed.
        source : numpy.ndarray
            The register the input is formed from.
        bundle : solprop.translate.stage.StageScalars
            The pass's scalars, anchored at its position.
        registers : tuple of numpy.ndarray
            ``A``, ``B`` and ``C``.
        flows : tuple of numpy.ndarray or None
            The table, or ``None`` to form the flow per element.
        """
        stage, dz, following = step
        backend = self._backend
        transverse, data = backend._transverse, backend._separable
        assert transverse is not None and data is not None
        grids = backend._problem.grids
        shape = registers[0].shape
        scratch = transverse.spectrum.reshape(shape)
        material, absorber = backend._nonlinear, backend._in_stage_absorber
        drawn = (
            registers[0] if backend._noise is None else backend._noise.reshape(shape)
        )
        scale = 1.0 / backend._problem.transform.n_nodes
        position = bundle.z1
        table = registers[0] if flows is None else flows[0]
        timers = backend._pass_timers

        def fold(mode: int, term: int) -> None:
            """Fold the scratch's source into the registers in one mode.

            Parameters
            ----------
            mode : int
                ``bodies.FOLD_COMPLETE``, ``FOLD_EXCHANGE`` or ``FOLD_SECOND``.
            term : int
                What the scratch holds: none, the material's or the absorber's.
            """
            mark = timers.now() if timers is not None else 0.0
            self._free.fold(
                data,
                (stage, mode, term),
                dz,
                scratch,
                grids.omega_window,
                grids.active_mask,
                registers,
                table,
                flows is not None,
                (
                    drawn,
                    following if self._elided else position,
                    scale,
                    self._elided,
                ),
                grids.norm_mask,
                self._partials,
            )
            if timers is not None:
                timers.lap("ls.fold", mark)

        if not material and not absorber:
            fold(FOLD_COMPLETE, _TERM_NONE)
            return
        backend._last_bundle = bundle
        raman = (
            backend._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
            if material
            else None
        )
        if material and backend._spatial_dispersive is not None:
            backend._spatial_dispersive.begin(bundle)
        if stage == 0 or not self._elided:
            # Elided, passes 2 to 4 find their input in the scratch already.
            mark = timers.now() if timers is not None else 0.0
            self._free.input(data, source, drawn, position, scale, scratch)
            if timers is not None:
                timers.lap("ls.input", mark)
        backend._free_pass(bundle, material=material, raman=raman)
        if material and absorber:
            # The exchange: the material source is folded and the input re-formed in
            # the scratch in one traversal, then the absorber's term is evaluated.
            fold(FOLD_EXCHANGE, _TERM_MATERIAL)
            backend._free_pass(bundle, material=False, raman=None)
            fold(FOLD_SECOND, _TERM_ABSORBER)
            return
        fold(FOLD_COMPLETE, _TERM_MATERIAL if material else _TERM_ABSORBER)

    def _slab_pass(
        self,
        stage: int,
        dz: float,
        source: np.ndarray,
        bundle: Any,
        registers: tuple[np.ndarray, np.ndarray, np.ndarray],
        flows: Any,
    ) -> None:
        """Evaluate one hybrid planar pass and fold both halves.

        Parameters
        ----------
        stage : int
            The pass, 0 to 3.
        dz : float
            The step, m.
        source : numpy.ndarray
            The register the input is formed from.
        bundle : solprop.translate.stage.StageScalars
            The pass's scalars, anchored at its position.
        registers : tuple of numpy.ndarray
            ``A``, ``B`` and ``C``.
        flows : tuple of numpy.ndarray or None
            The table, or ``None`` to form the flow per element.
        """
        backend = self._backend
        slab, grids = backend._slab_plans, backend._problem.grids
        assert slab is not None
        material, absorber = backend._nonlinear, backend._in_stage_absorber
        if material or absorber:
            drawn = (
                source
                if backend._noise is None
                else backend._noise.reshape(source.shape)
            )
            backend._last_bundle = bundle
            self._slab.input(
                backend._slab_data,
                source,
                drawn,
                bundle.z1,
                1.0 / slab.spectrum.shape[1],
                slab.rows,
            )
            backend._slab_pass(bundle)
        self._slab.fold(
            backend._slab_data,
            stage,
            dz,
            (material, absorber),
            (slab.rows, slab.absorbed),
            grids.omega_window,
            grids.active_mask,
            registers,
            source if flows is None else flows[0],
            flows is not None,
            grids.norm_mask,
            self._partials,
        )
