# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The general modal guide's nonlinear source, compiled, on the CPU backend.

The second execution path of
:mod:`solprop.backends.numpy_ref.general_modal`, with the same four steps of one
stage -- scale, synthesise, respond, project -- and the same maps, derived by the
reference's own ``solprop.backends.numpy_ref.backend._general_maps`` so that the
two backends cannot read a record differently. What is this backend's own is the
execution: a **fourth FFT plan family**
(:class:`solprop.backends.numba_cpu.fft.GeneralPlans`) bound to buffers allocated
once, compiled kernels in place of ``einsum``, and nothing allocated inside a stage.

**Parity is a rounding statement** (SEP 25): this backend's source is within the
bound of ``tests/general_parity.py``, fixed from the contractions' term counts, the
transforms' lengths and the maps' conditioning, of the reference's -- not equal to it.
The contractions -- the synthesis over channels, the real cubic's symmetric
contraction (each node's coefficients, then the ten monomials of a sample), the
analytic form's contraction over ``k`` and ``l`` and then over ``j``, the delayed
contraction over ``j, k, l`` and the projection over nodes and components -- are
written for locality through the per-element bodies of
:func:`solprop.backends.bodies.define_bodies`, and their order is this backend's own.
Each is diagonal in the bin (or the sample), a batch of tiny matrix-vector products
with a different matrix per bin, not a GEMM, so BLAS has nothing to take: the loops
run the bin (or the sample) innermost over contiguous rows. FFTW's transforms differ
from pocketfft's by their own rounding, which the same bound covers.

**Normalisation.** FFTW applies none. The inverse transform's ``1 / n_to`` is folded
into the per-bin scale of step 1 with the pad scale and the analytic doubling, and the
doubled axis's ``1 / (2 n_to)`` into the delayed contraction; both are powers of two
on every grid this package builds, so the folding is exact.

**Chunks.** The node loop runs in the reference's node chunks, which size the plans
and the per-chunk buffers; the projection accumulates each chunk's partial into the
source. A partial last chunk executes its own plans on prefix views of the same
buffers.

**A moving guide.** Its maps are ``n_intervals * n_deg``
Chebyshev coefficients of each map, the reference's ``GeneralMaps.synthesis_terms``
and ``projection_terms`` (views of the record's own arrays, never copied), and a
stage evaluates them at its position from the bundle's ``transform_weights`` (kernel
schema 14). The combination is one compiled kernel, ``combine``, run once per map
into two stage buffers allocated once: every coefficient is flattened to one axis,
the axis is cut into fixed blocks that the threads share (``prange``), and each block
is zeroed and then accumulates ``w_t c_t`` term by term in the terms' order, skipping
a term of zero weight -- every section but the stage's own -- so that a stage reads
``n_deg`` of the coefficients and not all of them (a BLAS product over every term, as
the reference's ``numpy.tensordot`` is, would read all of them). The reference's sum
and this one differ by a rounding of ``n_deg`` terms, bounded in the backend parity
tests. A constant guide's maps are the reference's arrays read directly.
"""

import logging
from functools import cache
from typing import Any, NamedTuple, no_type_check

import numpy as np
from jaxtyping import Complex, Float
from numba import prange

from solprop import fourier
from solprop._numba_cache import njit
from solprop.backends.bodies import GENERAL_PAIRS
from solprop.backends.numba_cpu import fft
from solprop.backends.numba_cpu.dispersive_general import GeneralDispersive
from solprop.backends.numba_cpu.kernels import HOST
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.backends.numpy_ref.general_modal import PAIR_HISTORIES, GeneralMaps
from solprop.constants import EPS0
from solprop.exceptions import BackendError
from solprop.translate.problem import MAX_N_COMPONENTS, KernelProblem

__all__ = [
    "GENERAL_LEDGER_ROWS",
    "MAP_BLOCK",
    "GeneralKernels",
    "GeneralWorkspace",
    "make_general_kernels",
]

logger = logging.getLogger(__name__)

_KERR = HOST.general_kerr
_ROW = HOST.general_symmetric_row
_REAL_KERR = HOST.general_real_kerr
_DRIVE = HOST.general_pair_drive
_DELAYED = HOST.general_delayed


@njit(cache=True, nogil=True)
@no_type_check
def _symmetric_node(tensor: Any, samples: Any, base: int, polarisation: Any) -> None:
    """Write one node's real cubic, ``eps0`` times its symmetric contraction.

    The coefficients are formed once for the node and every sample summed over its ten
    monomials. A function of its own and not inlined into ``respond``:
    numba's array analysis of a ``prange`` body fails on the inlined tuples of
    coefficients and monomials (a "Dimension mismatch" assertion), and a call per
    node costs nothing beside a node's samples.

    Parameters
    ----------
    tensor : numpy.ndarray
        The node's ``chi3 T``, shape ``(3, 3, 3, 3)``, m²/V².
    samples : numpy.ndarray
        The chunk's real field, rows ``3 q + c``, V/m.
    base : int
        The node's first row, ``3 q``.
    polarisation : numpy.ndarray
        Written at the node's three rows, C/m².
    """
    rows = (_ROW(tensor, 0), _ROW(tensor, 1), _ROW(tensor, 2))
    for time in range(polarisation.shape[1]):
        cubic = _REAL_KERR(rows, samples, base, time)
        for component in range(3):
            polarisation[base + component, time] = EPS0 * cubic[component]


GENERAL_LEDGER_ROWS = (
    "general_synthesis",
    "general_projection",
    "general_stage_maps",
    "general_tensor",
    "general_electric",
    "general_magnetic",
    "general_backward_electric",
    "general_backward_magnetic",
    "general_reciprocal_gram",
    "general_coordinate_dual",
    "general_scale",
    "general_lab",
    "general_scaled_state",
    "general_source",
    "general_source_partial",
    "general_output",
    "general_node_spectrum",
    "general_node_field",
    "general_node_polarisation",
    "general_node_source",
    "general_node_drive",
    "general_node_drive_spectrum",
    "general_node_product",
    "general_node_histories",
    "general_node_delayed",
    "general_delayed_tensor",
    "general_delayed_spectrum",
    "general_fft_plan_heap_growth",
)
"""The ledger rows of ``numba_cpu``'s general modal route, named on every route.

The reference's names where the two backends hold the same buffer -- on a moving
guide ``general_synthesis`` and ``general_projection`` are every coefficient of the
maps and ``general_stage_maps`` the two maps a stage evaluates into, zero otherwise --
and the rest are this backend's own persistent buffers -- the per-bin scale, the
combined stage input, the source, its per-chunk partial and the scaled output, the
doubled axis's product buffer -- and the allocator's growth over planning the fourth
family. Zero on every other route.
"""

MAP_BLOCK = 4096
"""Map coefficients one thread of the per-stage combination takes at a time.

64 KiB of complex128 in the output block and as much per term read, so that a block's
running sum stays in a core's level-2 cache while its ``n_deg`` terms stream past; a
map of the graded-index taper has about 300 of them. Each element's sum runs over its
terms whatever the blocking, so the value does not depend on it.
"""


class GeneralKernels(NamedTuple):
    """The compiled kernels of one general modal problem.

    Attributes
    ----------
    synthesise : numba.core.dispatcher.Dispatcher
        Step 2: ``E(q, c, w) = sum_m S a`` into the chunk's fine spectrum.
    respond : numba.core.dispatcher.Dispatcher
        Step 3: the instantaneous polarisation, and the pair drive when delayed.
    product : numba.core.dispatcher.Dispatcher
        One response and channel's kernel spectrum times the drive spectrum.
    contract : numba.core.dispatcher.Dispatcher
        The delayed contraction of one response and channel, accumulated.
    add_delayed : numba.core.dispatcher.Dispatcher
        ``P += eps0 * total`` over the chunk's rows.
    project : numba.core.dispatcher.Dispatcher
        Step 4: the chunk's projection, accumulated into the source.
    combine : numba.core.dispatcher.Dispatcher
        A moving guide's map at a stage: ``sum_t w_t c_t`` over flattened
        coefficients.
    """

    synthesise: Any
    respond: Any
    product: Any
    contract: Any
    add_delayed: Any
    project: Any
    combine: Any


def _combine_body(loop: Any) -> Any:
    """Return the body of the per-stage map combination for one loop kind.

    Parameters
    ----------
    loop : callable
        ``range``, or ``numba.prange`` for the parallel variant.

    Returns
    -------
    callable
        The uncompiled ``combine`` of :class:`GeneralKernels`.
    """

    @no_type_check
    def combine(weights: Any, terms: Any, out: Any, block: int) -> None:
        """Write a moving map at a stage: ``out = sum_t w_t terms[t]``.

        Parameters
        ----------
        weights : numpy.ndarray
            The stage bundle's ``transform_weights``, float64, ``(n_terms,)``.
        terms : numpy.ndarray
            The map's coefficients, complex128, ``(n_terms, n_elements)``.
        out : numpy.ndarray
            Written with the map at the stage, ``(n_elements,)``.
        block : int
            Elements a thread takes at a time, positive.
        """
        n_terms, size = terms.shape
        n_blocks = (size + block - 1) // block
        for index in loop(n_blocks):
            start = index * block
            stop = min(start + block, size)
            for element in range(start, stop):
                out[element] = 0.0
            # Term by term over the block, so that the terms of zero weight -- every
            # section but the stage's own -- are skipped once rather than per
            # element; every coefficient is finite (the record refuses anything
            # else), so a skipped term is an exact zero.
            for term in range(n_terms):
                weight = weights[term]
                if weight == 0.0:
                    continue
                for element in range(start, stop):
                    out[element] = out[element] + weight * terms[term, element]

    return combine


def _synthesise_body(loop: Any, contiguous: bool) -> Any:
    """Return the body of the chunk's synthesis for one loop kind and crop.

    Parameters
    ----------
    loop : callable
        ``range``, or ``numba.prange`` for the parallel variant.
    contiguous : bool
        Whether ``crop`` is ``arange(n_omega)`` (:func:`make_general_kernels`).

    Returns
    -------
    callable
        The uncompiled ``synthesise`` of :class:`GeneralKernels`.
    """

    @no_type_check
    def synthesise(
        synthesis: Any, scaled: Any, crop: Any, first: int, count: int, fine: Any
    ) -> None:
        """Synthesise the chunk's three components on the fine spectrum.

        Parameters
        ----------
        synthesis : numpy.ndarray
            ``S``, shape ``(n_nodes, 3, n_modes, n_omega)``.
        scaled : numpy.ndarray
            The scaled channel spectra, shape ``(n_modes, n_omega)``.
        crop : numpy.ndarray
            Where each coarse bin sits in the fine spectrum.
        first : int
            The chunk's first node.
        count : int
            The chunk's node count.
        fine : numpy.ndarray
            Written with the fine spectrum, rows ``3 q + c``, every bin.
        """
        n_modes, n_omega = scaled.shape
        n_fine = fine.shape[1]
        for node in loop(count):
            for component in range(3):
                row = 3 * node + component
                if contiguous:
                    # The first channel writes the coarse bins and the rest add to
                    # them; only the bins above the crop are zeroed.
                    for bin_ in range(n_omega, n_fine):
                        fine[row, bin_] = 0.0
                    for bin_ in range(n_omega):
                        fine[row, bin_] = (
                            synthesis[first + node, component, 0, bin_]
                            * scaled[0, bin_]
                        )
                    for mode in range(1, n_modes):
                        for bin_ in range(n_omega):
                            fine[row, bin_] += (
                                synthesis[first + node, component, mode, bin_]
                                * scaled[mode, bin_]
                            )
                    continue
                for bin_ in range(n_fine):
                    fine[row, bin_] = 0.0
                for mode in range(n_modes):
                    for bin_ in range(n_omega):
                        fine[row, crop[bin_]] += (
                            synthesis[first + node, component, mode, bin_]
                            * scaled[mode, bin_]
                        )

    return synthesise


def _project_body(loop: Any, contiguous: bool) -> Any:
    """Return the body of the chunk's projection for one loop kind and crop.

    Parameters
    ----------
    loop : callable
        ``range``, or ``numba.prange`` for the parallel variant.
    contiguous : bool
        Whether ``crop`` is ``arange(n_omega)`` (:func:`make_general_kernels`).

    Returns
    -------
    callable
        The uncompiled ``project`` of :class:`GeneralKernels`.
    """

    @no_type_check
    def project(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        projection: Any,
        spectrum: Any,
        crop: Any,
        first: int,
        count: int,
        partial: Any,
        source: Any,
    ) -> None:
        """Project the chunk's polarisation spectrum and add it to the source.

        Parameters
        ----------
        projection : numpy.ndarray
            ``Q``, shape ``(n_modes, n_nodes, 3, n_omega)``.
        spectrum : numpy.ndarray
            The chunk's one-sided polarisation spectra, rows ``3 q + c``.
        crop : numpy.ndarray
            Where each coarse bin sits in the fine spectrum.
        first : int
            The chunk's first node.
        count : int
            The chunk's node count.
        partial : numpy.ndarray
            Scratch for the chunk's own sum, shape ``(n_modes, n_omega)``.
        source : numpy.ndarray
            Accumulated into, shape ``(n_modes, n_omega)``.
        """
        n_modes, n_omega = source.shape
        for mode in loop(n_modes):
            for bin_ in range(n_omega):
                partial[mode, bin_] = 0.0
            for node in range(count):
                for component in range(3):
                    row = 3 * node + component
                    if contiguous:
                        for bin_ in range(n_omega):
                            partial[mode, bin_] += (
                                projection[mode, first + node, component, bin_]
                                * spectrum[row, bin_]
                            )
                        continue
                    for bin_ in range(n_omega):
                        partial[mode, bin_] += (
                            projection[mode, first + node, component, bin_]
                            * spectrum[row, crop[bin_]]
                        )
            for bin_ in range(n_omega):
                source[mode, bin_] = source[mode, bin_] + partial[mode, bin_]

    return project


@cache
def make_general_kernels(
    *, analytic: bool, delayed: bool, parallel: bool, contiguous: bool = False
) -> GeneralKernels:
    """Compile the general route's kernels for one form, delay and threading regime.

    Every flag is a closure constant, so a problem without a delayed response pays
    nothing for it and the serial and parallel variants come from one body
    (``docs/dev/backends.md`` section 3.2). The node loops are independent per node
    and the projection's reduction is per output bin, so the result does not depend
    on how many threads ran either.

    With `contiguous` the coarse bins are the leading bins of the fine spectrum
    (``crop`` is ``arange(n_omega)``, as on every field grid), and the synthesis and the
    projection index the fine spectrum directly instead of through ``crop``: the loads
    are then contiguous and the bin loops vectorise. Measured on one thread of an
    Apple-silicon laptop on the benchmark case's 576 nodes, 2 channels and 513 bins:
    the synthesis 1.8 -> 1.35 ms a stage and the projection 1.4 to 1.6 -> 1.0 ms.

    Parameters
    ----------
    analytic : bool
        Whether the cubic is the analytic-signal form, which drives the delayed
        response with ``Re(E_k conj(E_l)) / 2``; otherwise the real field's
        ``E_k E_l``.
    delayed : bool
        Whether the respond kernel writes the delayed response's pair drive.
    parallel : bool
        Whether the node loops, and the projection's channel loop, are a ``prange``.
    contiguous : bool, optional
        Whether ``crop`` is ``arange(n_omega)``, which the caller has checked.

    Returns
    -------
    GeneralKernels
        The compiled kernels.
    """
    loop = prange if parallel else range
    pairs = GENERAL_PAIRS
    stored = PAIR_HISTORIES

    @no_type_check
    def respond(
        samples: Any,
        tensor: Any,
        first: int,
        count: int,
        polarisation: Any,
        drive: Any,
    ) -> None:
        """Write the chunk's instantaneous polarisation and the delayed drive.

        The real cubic by its symmetric contraction, its coefficients formed once
        per node; the analytic form densely. Which one is the closure's `analytic`.

        Parameters
        ----------
        samples : numpy.ndarray
            The field at the chunk's nodes: the analytic signal, or its real part.
        tensor : numpy.ndarray
            The per-node ``chi3 T``, shape ``(n_nodes, 3, 3, 3, 3)``.
        first : int
            The chunk's first node.
        count : int
            The chunk's node count.
        polarisation : numpy.ndarray
            Written with ``Re P`` in C/m², rows ``3 q + c``.
        drive : numpy.ndarray
            Written with the six pair entries, rows ``6 q + e``, when `delayed` was
            compiled in; never read otherwise.
        """
        n_to = polarisation.shape[1]
        for node in loop(count):
            base = 3 * node
            node_tensor = tensor[first + node]
            if not analytic:
                _symmetric_node(node_tensor, samples, base, polarisation)
            for time in range(n_to):
                if analytic:
                    values = _KERR(node_tensor, samples, base, time)
                    for component in range(3):
                        polarisation[base + component, time] = (
                            EPS0 * values[component].real
                        )
                if delayed:
                    for entry in range(stored):
                        row, column = pairs[entry]
                        drive[stored * node + entry, time] = _DRIVE(
                            samples[base + row, time],
                            samples[base + column, time],
                            analytic,
                        )

    @no_type_check
    def product(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        kernel: Any,
        drive_spectrum: Any,
        owner: Any,
        response: int,
        first: int,
        count: int,
        output: Any,
    ) -> None:
        """Multiply the owned nodes' drive spectra by one kernel's spectrum.

        Parameters
        ----------
        kernel : numpy.ndarray
            One response and channel's weighted kernel spectrum, ``(n_to + 1,)``.
        drive_spectrum : numpy.ndarray
            The chunk's drive spectra, rows ``6 q + e``.
        owner : numpy.ndarray
            Which response owns each node, ``-1`` for none.
        response : int
            The response being convolved.
        first : int
            The chunk's first node.
        count : int
            The chunk's node count.
        output : numpy.ndarray
            Written at the owned nodes' rows; the other rows are not read later.
        """
        n_bins = output.shape[1]
        for node in loop(count):
            if owner[first + node] != response:
                continue
            for entry in range(stored):
                row = stored * node + entry
                for bin_ in range(n_bins):
                    output[row, bin_] = kernel[bin_] * drive_spectrum[row, bin_]

    @no_type_check
    def contract(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        history: Any,
        tensor: Any,
        owner: Any,
        response: int,
        first: int,
        count: int,
        samples: Any,
        scale: float,
        total: Any,
    ) -> None:
        """Accumulate one response and channel's delayed contraction.

        Parameters
        ----------
        history : numpy.ndarray
            The convolved histories on the doubled axis, rows ``6 q + e``.
        tensor : numpy.ndarray
            The response and channel's ``chi3 D``, shape ``(3, 3, 3, 3)``.
        owner : numpy.ndarray
            Which response owns each node, ``-1`` for none.
        response : int
            The response being convolved.
        first : int
            The chunk's first node.
        count : int
            The chunk's node count.
        samples : numpy.ndarray
            The field at the chunk's nodes; its real part is read.
        scale : float
            The inverse transform's ``1 / (2 n_to)``.
        total : numpy.ndarray
            Accumulated into, rows ``3 q + c``, before ``eps0``.
        """
        n_to = total.shape[1]
        for node in loop(count):
            if owner[first + node] != response:
                continue
            for time in range(n_to):
                for component in range(3):
                    total[3 * node + component, time] += _DELAYED(
                        tensor,
                        component,
                        history,
                        stored * node,
                        samples,
                        3 * node,
                        time,
                        scale,
                    )

    @no_type_check
    def add_delayed(polarisation: Any, total: Any, rows: int) -> None:
        """Add ``eps0`` times the delayed total to the polarisation's leading rows.

        Parameters
        ----------
        polarisation : numpy.ndarray
            The instantaneous polarisation, C/m², modified in place.
        total : numpy.ndarray
            The delayed contraction before ``eps0``.
        rows : int
            Rows the chunk wrote.
        """
        n_to = total.shape[1]
        for row in loop(rows):
            for time in range(n_to):
                polarisation[row, time] = polarisation[row, time] + (
                    EPS0 * total[row, time]
                )

    # A parallel closure is never disk-cached: numba can load one variant's body for
    # another of the same name.
    compile_ = njit(cache=not parallel, nogil=True, parallel=parallel)
    return GeneralKernels(
        synthesise=compile_(_synthesise_body(loop, contiguous)),
        respond=compile_(respond),
        product=compile_(product),
        contract=compile_(contract),
        add_delayed=compile_(add_delayed),
        project=compile_(_project_body(loop, contiguous)),
        combine=compile_(_combine_body(loop)),
    )


class GeneralWorkspace:
    """The general modal route's buffers, plans and kernels, allocated once.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A problem the backend has admitted, carrying ``transform.modal_tables``.
    flag : str
        The FFTW planning flag's key.
    fftw_threads : int or None
        Override for FFTW's pool for the fourth family; ``None`` sizes it from the
        chunk's transform by the backend's own rule
        (:func:`solprop.backends.numba_cpu.fft.choose_pools`).
    parallel : bool
        Whether the node loops are a ``prange``.
    """

    def __init__(
        self,
        problem: KernelProblem,
        *,
        flag: str,
        fftw_threads: int | None,
        parallel: bool,
    ) -> None:
        maps = _general_maps(problem)
        tables = problem.transform.modal_tables
        assert maps is not None and tables is not None
        self.maps = maps
        self._tables = tables
        grids = problem.grids
        self._n_to = grids.n_to
        self._crop = np.ascontiguousarray(grids.crop_index)
        n_modes, n_omega = maps.n_modes, int(self._crop.size)
        # The reference's `pad_scale * doubling[crop_index]` and its inverse
        # transform's `1 / n_to`, both exact scalings on a power-of-two grid.
        doubling = np.full(fourier.n_omega_for(self._n_to, "field"), 2.0)
        doubling[0] = doubling[-1] = 1.0
        self._factor = (grids.pad_scale * doubling[self._crop]) / self._n_to
        # A gas region's dispersive channels respond where the regional tensors may
        # all be zero, as the reference's ``_general_responds`` reads them. With no
        # regional tensor and no delayed response the contraction is identically zero:
        # it and its transform are skipped, and the dispersive spectrum is written
        # rather than added (a guide whose only nonlinear region is a gas, measured at
        # seven times the contraction's cost of its regional neighbour's).
        self.responds = bool(np.any(maps.tensor)) or maps.dispersive is not None
        self._regional = bool(np.any(maps.tensor)) or maps.n_terms > 0
        self._allocate_stage_maps(maps)
        self.lab = np.zeros((n_modes, n_omega), dtype=np.complex128)
        self._scaled = np.zeros_like(self.lab)
        self._source = np.zeros_like(self.lab)
        self._partial = np.zeros_like(self.lab)
        self.output = np.zeros_like(self.lab)
        chunk = maps.node_chunk
        self._chunks = tuple(
            (first, min(chunk, maps.n_nodes - first))
            for first in range(0, maps.n_nodes, chunk)
        )
        self._delayed = maps.n_terms > 0
        threads = fft.choose_pools(
            0, 0, MAX_N_COMPONENTS * chunk * self._n_to, fftw_threads=fftw_threads
        ).fftw
        self.plans = fft.build_general_plans(
            chunk,
            maps.n_nodes % chunk,
            self._n_to,
            delayed=self._delayed,
            flag=flag,
            threads=threads,
        )
        self._samples = (
            self.plans.signal if maps.form == "analytic" else self.plans.signal.real
        )
        self._total = np.zeros(
            (MAX_N_COMPONENTS * chunk if self._delayed else 0, self._n_to)
        )
        owner = maps.node_response
        self._owner = (
            np.full(maps.n_nodes, -1, dtype=np.int32) if owner is None else owner
        )
        # Which responses a chunk convolves, fixed at construction: the reference's
        # `rows.any()` and `np.any(spectra[response])`, decided once rather than
        # allocated per stage.
        spectra = maps.delayed_spectrum
        live = (
            ()
            if spectra is None
            else tuple(
                response
                for response in range(spectra.shape[0])
                if np.any(spectra[response])
            )
        )
        self._chunk_responses = tuple(
            tuple(
                response
                for response in live
                if np.any(self._owner[first : first + count] == response)
            )
            for first, count in self._chunks
        )
        self._scale = 1.0 / (2 * self._n_to)
        self.kernels = make_general_kernels(
            analytic=maps.form == "analytic",
            delayed=self._delayed,
            parallel=parallel,
            contiguous=bool(
                np.array_equal(self._crop, np.arange(n_omega, dtype=self._crop.dtype))
            ),
        )
        self.parallel = parallel
        self.threads = threads
        self.dispersive = (
            None
            if maps.dispersive is None or maps.node_region is None
            else GeneralDispersive(
                maps.dispersive,
                maps.node_region,
                self._chunks,
                self.plans,
                parallel=parallel,
                flag=flag,
                threads=threads,
            )
        )
        logger.debug(
            "numba_cpu general modal route: %d channels on %d nodes, chunk %d, "
            "%s cubic, %d delayed term(s), %s",
            n_modes,
            maps.n_nodes,
            chunk,
            maps.form,
            maps.n_terms,
            "parallel" if parallel else "serial",
        )

    def _allocate_stage_maps(self, maps: GeneralMaps) -> None:
        """Hold the maps a stage reads: the reference's, or a moving guide's buffers.

        A constant guide's stage reads the reference's arrays directly. A moving
        guide's coefficients are the reference's too, viewed flat over their element
        axes, and its stage writes the two maps it evaluates into buffers allocated
        here, once.

        Parameters
        ----------
        maps : GeneralMaps
            The reference's derived maps of the problem.
        """
        self.moving = maps.moving
        if not maps.moving:
            self.synthesis, self.projection = maps.synthesis, maps.projection
            self._terms: tuple[np.ndarray, ...] = ()
            return
        assert maps.synthesis_terms is not None and maps.projection_terms is not None
        count = int(maps.synthesis_terms.shape[0])
        self._terms = (
            maps.synthesis_terms.reshape(count, -1),
            maps.projection_terms.reshape(count, -1),
        )
        self.synthesis = np.zeros_like(maps.synthesis)
        self.projection = np.zeros_like(maps.projection)

    def move(self, weights: Float[np.ndarray, " n_map_weights"] | None) -> None:
        """Evaluate a moving guide's maps at a stage, from its ``transform_weights``.

        The reference's ``GeneralMaps.at`` on this backend's buffers: nothing for a
        constant guide, and for a moving one the two weighted sums of the
        coefficients by :func:`make_general_kernels`' ``combine``, allocating nothing.

        Parameters
        ----------
        weights : numpy.ndarray or None
            The stage bundle's ``transform_weights``, float64, one per coefficient;
            ``None`` for maps that do not move.

        Raises
        ------
        BackendError
            If moving maps are asked for with no weights, or constant ones with some,
            in the reference's words.
        """
        if not self.moving:
            if weights is not None:
                msg = (
                    "a stage carries transform_weights for maps that do not move; the "
                    "bundle and the tables come from different problems"
                )
                raise BackendError(msg)
            return
        if weights is None:
            msg = (
                "a moving guide's maps are evaluated from the stage bundle's "
                "transform_weights, and this stage carries none"
            )
            raise BackendError(msg)
        synthesis, projection = self._terms
        self.kernels.combine(weights, synthesis, self.synthesis.reshape(-1), MAP_BLOCK)
        self.kernels.combine(
            weights, projection, self.projection.reshape(-1), MAP_BLOCK
        )

    def source(
        self,
        transform_weights: Float[np.ndarray, " n_map_weights"] | None = None,
        dispersive_densities: tuple[float, ...] = (),
    ) -> Complex[np.ndarray, "n_modes n_omega"]:
        """Evaluate one stage's modal source from ``lab`` into ``output``.

        The reference's :func:`solprop.backends.numpy_ref.general_modal.general_source`
        on this backend's buffers, allocating nothing. SEP 6's dispersive channels
        filter each chunk's synthesised spectrum and land their products' spectra in
        its polarisation spectrum, before the projection
        (:class:`~solprop.backends.numba_cpu.dispersive_general.GeneralDispersive`).

        Parameters
        ----------
        transform_weights : numpy.ndarray or None, optional
            The stage bundle's weights of a moving guide's maps (:meth:`move`);
            ``None`` for maps that do not move.
        dispersive_densities : tuple of float, optional
            Each dispersive channel's number density at the stage, m^-3: the stage
            bundle's ``dispersive_kerr_densities``.

        Returns
        -------
        numpy.ndarray
            ``output``: the projected polarisation times ``-i omega``, shape
            ``(n_modes, n_omega)``, in sqrt(W)/m before the crop scale.
        """
        maps, plans, kernels = self.maps, self.plans, self.kernels
        dispersive = self.dispersive
        self.move(transform_weights)
        if dispersive is not None:
            dispersive.begin(dispersive_densities)
        np.multiply(self.lab, self._factor, out=self._scaled)
        self._source[:] = 0.0
        full = maps.node_chunk
        for index, (first, count) in enumerate(self._chunks):
            tail = count != full
            kernels.synthesise(
                self.synthesis, self._scaled, self._crop, first, count, plans.spectrum
            )
            (plans.tail_to_time if tail else plans.to_time).execute()
            if self._regional:
                kernels.respond(
                    self._samples,
                    maps.tensor,
                    first,
                    count,
                    plans.polarisation,
                    plans.drive if self._delayed else plans.polarisation,
                )
            if self._delayed:
                self._delayed_pass(index, first, count, tail)
            # The synthesised spectrum survives the out-of-place inverse transform.
            live = dispersive is not None and dispersive.respond(index, first, count)
            if self._regional:
                (plans.tail_to_frequency if tail else plans.to_frequency).execute()
            elif not live:
                plans.polarisation_spectrum[:] = 0.0
            if live and dispersive is not None:
                dispersive.combine(overwrite=not self._regional)
            kernels.project(
                self.projection,
                plans.polarisation_spectrum,
                self._crop,
                first,
                count,
                self._partial,
                self._source,
            )
        np.multiply(self._source, maps.frequency_factor, out=self.output)
        return self.output

    def _delayed_pass(self, index: int, first: int, count: int, tail: bool) -> None:
        """Add one chunk's delayed polarisation, response by response.

        Parameters
        ----------
        index : int
            The chunk's position in the chunk list.
        first : int
            The chunk's first node.
        count : int
            The chunk's node count.
        tail : bool
            Whether this is the partial last chunk, which has its own plans.
        """
        maps, plans, kernels = self.maps, self.plans, self.kernels
        tensors, spectra = maps.delayed_tensor, maps.delayed_spectrum
        assert tensors is not None and spectra is not None
        rows = MAX_N_COMPONENTS * count
        self._total[:rows] = 0.0
        (plans.tail_drive_to_frequency if tail else plans.drive_to_frequency).execute()
        inverse = plans.tail_history_to_time if tail else plans.history_to_time
        for response in self._chunk_responses[index]:
            for term in range(maps.n_terms):
                kernels.product(
                    spectra[response, term],
                    plans.drive_spectrum,
                    self._owner,
                    response,
                    first,
                    count,
                    plans.product,
                )
                inverse.execute()
                kernels.contract(
                    plans.history,
                    tensors[response, term],
                    self._owner,
                    response,
                    first,
                    count,
                    self._samples,
                    self._scale,
                    self._total,
                )
        kernels.add_delayed(plans.polarisation, self._total, rows)

    def memory_ledger(self) -> dict[str, int]:
        """Return the bytes the route holds, one row per named buffer.

        Returns
        -------
        dict
            Every row of :data:`GENERAL_LEDGER_ROWS`: the maps and the record's
            resident reference, counted as the reference backend counts them, and
            this backend's persistent buffers and plan heap growth.
        """
        maps, tables = self.maps, self._tables
        resident = {
            f"general_{name}": 0
            if getattr(tables, name) is None
            else int(getattr(tables, name).nbytes)
            for name in (
                "electric",
                "magnetic",
                "backward_electric",
                "backward_magnetic",
                "reciprocal_gram",
                "coordinate_dual",
            )
        }
        # A moving guide holds every coefficient of both maps -- the reference's
        # arrays, viewed -- and the two maps a stage evaluates into.
        stored = (maps.synthesis, maps.projection) if not self.moving else self._terms
        return {
            "general_synthesis": int(stored[0].nbytes),
            "general_projection": int(stored[1].nbytes),
            "general_stage_maps": 0
            if not self.moving
            else self.synthesis.nbytes + self.projection.nbytes,
            "general_tensor": int(maps.tensor.nbytes),
            **resident,
            "general_scale": self._factor.nbytes,
            "general_lab": self.lab.nbytes,
            "general_scaled_state": self._scaled.nbytes,
            "general_source": self._source.nbytes,
            "general_source_partial": self._partial.nbytes,
            "general_output": self.output.nbytes,
            **self.plans.buffer_bytes,
            "general_node_delayed": self._total.nbytes,
            "general_delayed_tensor": 0
            if maps.delayed_tensor is None
            else int(maps.delayed_tensor.nbytes),
            "general_delayed_spectrum": 0
            if maps.delayed_spectrum is None
            else int(maps.delayed_spectrum.nbytes),
        }
