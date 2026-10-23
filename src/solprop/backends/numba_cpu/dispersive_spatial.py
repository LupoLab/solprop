# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's frequency-dependent electronic response on the Cartesian and planar routes.

The fused node path's workspace (:mod:`solprop.backends.numba_cpu.dispersive`) moves the
response's spectral multipliers onto the modal columns of a whole stage. On the
Cartesian free-space and line routes the columns **are** the nodes, and on the hybrid
planar route a block of ``(p, y)`` rows goes to time together and its ``(x, y)`` nodes
are synthesised from them by a real matrix in ``x`` that does not depend on frequency.
Either way each chunk of columns -- a node chunk, or a block of rows -- is transformed
to time and back on its own, so the response runs **per chunk**, with the same five
steps:

1. the input multipliers act on the chunk's padded spectrum, which the stage packed for
   its own inverse transform, and one batched inverse transform gives every filtered
   field of the chunk (none of the node field: the stage's own transform gives it);
2. on the planar route the filtered fields are synthesised at the block's nodes by the
   slab rule, as the bare field is; the products are formed at the nodes in the leaf's
   sector, and on the planar route projected back to the block's rows;
3. the collar multiplies the products in time, ``G[w (G E)^3]``;
4. one batched forward transform takes them to frequency;
5. after the stage's forward transform of the chunk's polarisation -- which a lone
   response with buffers of its own replaces -- their output multipliers and the
   channels' ``eps0 gamma0 rho`` land them in the chunk's polarisation spectrum, which
   the stage crops back to its coarse columns.

A lone symmetric response -- of one species, or of a mixture under its mixture filter
-- whose product has the polarisation's dtype holds no buffer of its own: it filters
the chunk's spectrum in place (on the planar route only when no in-stage absorber reads
the bare rows afterwards), its product is the stage's polarisation, and ``g`` finishes
it in the stage's spectrum. The polynomial groups species of equal leg bound as on the
fused path (:class:`~solprop.backends.numba_cpu.dispersive.DispersiveTerms`).

Neither route has an embedded quadrature rule, so there is nothing to finish when a
step is accepted.
"""

import time
from typing import Any

import numpy as np
from jaxtyping import Complex

from solprop.backends.numba_cpu import fft
from solprop.backends.numba_cpu.dispersive import (
    DISPERSIVE_LEDGER_ROWS,
    StageWeights,
    add_spectra,
    apply_window,
    combine_spectra,
    dispersive_form,
    dispersive_terms_layout,
    filter_spectra,
    make_dispersive_pairs,
    make_dispersive_self,
    weigh_spectrum,
)
from solprop.backends.numba_cpu.openmp import pyfftw
from solprop.translate.problem import KernelProblem
from solprop.translate.stage import StageScalars

__all__ = ["SpatialDispersive", "SpatialLayout"]


class SpatialLayout:
    """How the backend runs a Cartesian or planar stage, as the response needs it.

    Parameters
    ----------
    rows : int
        Rows one temporal transform of the stage takes: a node chunk's columns in free
        space, a block's ``(p, y)`` rows on the planar route.
    node_rows : int
        Rows of the backend's node buffer for one chunk: the columns again in free
        space, the block's ``(x, y)`` nodes times their components on the planar route.
    n_pol_node : int
        Components at one node.
    block : int
        The planar route's y rows per full block, of which ``rows`` and ``node_rows``
        are multiples; one in free space.
    slab : bool
        Whether the route is the hybrid planar one, whose nodes are synthesised from
        the rows.
    alone : bool
        Whether the dispersive response is the only one.
    bare_time_free : bool
        Whether nothing reads the chunk's bare field in time after the response, so
        that a lone symmetric response may filter the chunk's spectrum in place.
    parallel : bool
        Whether the node loops are ``prange``.
    """

    def __init__(  # noqa: PLR0913 - the route's layout, named
        self,
        *,
        rows: int,
        node_rows: int,
        n_pol_node: int,
        block: int,
        slab: bool,
        alone: bool,
        bare_time_free: bool,
        parallel: bool,
    ) -> None:
        self.rows = rows
        self.node_rows = node_rows
        self.n_pol_node = n_pol_node
        self.block = block
        self.slab = slab
        self.alone = alone
        self.bare_time_free = bare_time_free
        self.parallel = parallel


class SpatialDispersive:
    """Preallocate the response's buffers and plans, and run it chunk by chunk.

    Per chunk the backend calls :meth:`prepare` on the packed spectrum before its own
    inverse transform, :meth:`respond` with the chunk's bare field in time (and, on
    the planar route, the slab synthesis and projection), :meth:`finish` before its
    forward transform -- which it skips when :attr:`replaces_transform` -- and
    :meth:`combine` after. :meth:`begin` weighs the channels once per stage.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem, carrying ``responses.dispersive_kerr`` on a Cartesian or planar
        transform.
    plans : solprop.backends.numba_cpu.fft.TransformPlans
        The backend's per-chunk plans, whose packed spectrum, field, polarisation and
        polarisation spectrum the response reads, borrows or adds into.
    layout : SpatialLayout
        How the backend runs the route.
    flag : str
        The FFTW planning flag, a key of
        :data:`solprop.backends.numba_cpu.fft.PLAN_FLAGS`.
    fftw_threads : int or None
        The FFTW pool override; ``None`` sizes each batched plan by the backend's rule.
    """

    def __init__(
        self,
        problem: KernelProblem,
        plans: fft.TransformPlans,
        layout: SpatialLayout,
        *,
        flag: str,
        fftw_threads: int | None,
    ) -> None:
        leaf = problem.responses.dispersive_kerr
        assert leaf is not None
        self.leaf, self.layout, self.plans = leaf, layout, plans
        kind = problem.grid_kind
        self.form = dispersive_form(kind, leaf.sector)
        self.symmetric = leaf.evaluator == "symmetric"
        analytic = self.form == "conserving" and kind == "field"
        complex_time = self.form != "real"
        time_dtype = np.complex128 if complex_time else np.float64
        n_to = problem.grids.n_to
        self.window = problem.grids.t_window_fine
        self.terms = dispersive_terms_layout(leaf, kind, analytic=analytic)
        self.stage = StageWeights(leaf, self.terms)
        self.carrier = (
            np.ascontiguousarray(problem.grids.carrier_2w0)
            if leaf.carrier and problem.grids.carrier_2w0 is not None
            else np.ones(1, dtype=np.complex128)
        )
        # One product: a symmetric response of one species, or a mixture's under its
        # mixture filter.
        lone = layout.alone and self.symmetric and self.terms.n_products == 1
        self.borrowed = lone and not analytic
        # Alone with nothing reading the bare field afterwards, a lone symmetric
        # response filters the chunk's own spectrum, and the stage's inverse transform
        # gives the filtered field.
        self.in_place = lone and layout.bare_time_free
        self.shifted = analytic and not self.symmetric
        self._owned: dict[str, int] = dict.fromkeys(DISPERSIVE_LEDGER_ROWS, 0)
        rows, n_fields = layout.rows, self.terms.n_fields
        n_products = self.terms.n_products
        self.dc = np.zeros(rows if self.shifted else 0, dtype=np.complex128)
        # Free space: the bare field's constant per column is subtracted where the
        # product reads it; the planar route synthesises it to the nodes.
        self.offset = np.zeros(
            (layout.node_rows if self.shifted and layout.slab else 1,),
            dtype=np.complex128,
        )
        # The slab rule in x, real and independent of frequency: ``(n_x, n_p)`` and
        # ``(n_p, n_x)``, the synthesis carrying the temporal pad scale.
        transform = problem.transform
        self.synthesis = np.ascontiguousarray(transform.synthesis, dtype=np.float64)
        self.projection = np.ascontiguousarray(transform.projection, dtype=np.float64)
        self._count = layout.block
        threads = fft.choose_pools(
            0, 0, max(n_fields, n_products) * rows * n_to, fftw_threads=fftw_threads
        ).fftw
        heap_before = fft._heap_bytes()
        started = time.perf_counter()
        self._plan(
            time_dtype,
            n_to if complex_time else leaf.positive.shape[0],
            flag=fft.PLAN_FLAGS[flag],
            threads=threads,
        )
        self.planning_seconds = time.perf_counter() - started
        self._owned["dispersive_fft_plan_heap_growth"] = max(
            0, fft._heap_bytes() - heap_before
        )
        # The planar route's node slots: the filtered field and its product, or the
        # polynomial's two derivative fields, at the block's nodes.
        n_slots = (
            0
            if not layout.slab or self.borrowed or self.in_place
            else 1
            if self.symmetric
            else 2
        )
        self.slots = np.zeros((n_slots * layout.node_rows, n_to), dtype=time_dtype)
        self._owned["dispersive_node_fields"] = self.slots.nbytes
        self.self_product = make_dispersive_self(
            self.form, layout.n_pol_node, parallel=layout.parallel, shifted=self.shifted
        )
        self.pairs = (
            None
            if self.symmetric
            else make_dispersive_pairs(
                self.form,
                layout.n_pol_node,
                parallel=layout.parallel,
                shifted=self.shifted,
            )
        )
        self._owned["dispersive_multipliers"] = (
            self.stage.nbytes
            + self.carrier.nbytes
            + self.dc.nbytes
            + self.offset.nbytes
        )

    def _plan(
        self, time_dtype: type, n_product_bins: int, *, flag: str, threads: int
    ) -> None:
        """Allocate the products, the filtered fields and their spectra, and plan.

        Parameters
        ----------
        time_dtype : type
            The dtype of the filtered fields and of the products.
        n_product_bins : int
            Bins of a product's spectrum: one-sided for real products only.
        flag : str
            The FFTW flag name.
        threads : int
            The FFTW pool.
        """
        layout, plans = self.layout, self.plans
        rows, n_to = layout.rows, self.window.shape[0]
        n_fields, n_products = self.terms.n_fields, self.terms.n_products
        self.products: np.ndarray = np.empty((0, n_to), dtype=time_dtype)
        self.spectra: np.ndarray = np.empty((0, n_product_bins), dtype=np.complex128)
        self.forward: Any = None
        self.to_time: Any = None
        self.filtered: np.ndarray = plans.modal_time
        self.filtered_spectrum: np.ndarray = plans.modal_spectrum
        if not self.borrowed:
            self.products = self._aligned(
                "products", (n_products * rows, n_to), time_dtype
            )
            self.spectra = self._aligned(
                "product_spectra", (n_products * rows, n_product_bins), np.complex128
            )
            self.forward = fft._plan(
                self.products, self.spectra, forward=True, flag=flag, threads=threads
            )
        if self.in_place:
            return
        # Free space forms each product over its own filtered field, so the inverse
        # transform writes the filtered fields straight into the product blocks they
        # become: every block but the polynomial's cube. The planar route keeps them
        # on the rows, to be synthesised.
        if layout.slab or self.borrowed:
            self.filtered = self._aligned(
                "filtered_time", (n_fields * rows, n_to), time_dtype
            )
        else:
            first = 0 if self.symmetric else rows
            self.filtered = self.products[first : first + n_fields * rows]
        self.filtered_spectrum = (
            self._aligned(
                "product_spectra",
                (n_fields * rows, plans.modal_spectrum.shape[1]),
                np.complex128,
            )
            if self.borrowed
            else self.spectra[: n_fields * rows]
        )
        self.to_time = fft._plan(
            self.filtered_spectrum,
            self.filtered,
            forward=False,
            flag=flag,
            threads=threads,
        )

    @property
    def replaces_transform(self) -> bool:
        """Return whether the response writes the chunk's polarisation spectrum itself.

        Returns
        -------
        bool
            True when the response is alone with products of its own, so that the
            stage skips its forward transform of an unused polarisation buffer.
        """
        return self.layout.alone and not self.borrowed

    def _aligned(self, name: str, shape: tuple[int, int], dtype: type) -> np.ndarray:
        """Allocate an aligned plan buffer and count it under its ledger row.

        Parameters
        ----------
        name : str
            The row, without the ``dispersive_`` prefix.
        shape : tuple of int
            The buffer's shape.
        dtype : type
            Its dtype.

        Returns
        -------
        numpy.ndarray
            The buffer, zeroed.
        """
        buffer = pyfftw.empty_aligned(shape, dtype=np.dtype(dtype).name)
        buffer[:] = 0
        self._owned[f"dispersive_{name}"] += buffer.nbytes
        return buffer

    # --- one stage, one chunk at a time -------------------------------------------

    def begin(self, bundle: StageScalars) -> None:
        """Weigh the channels at the stage, once for all its chunks.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's scalars, whose channel densities weight the terms and build a
            symmetric mixture's filter.
        """
        self.stage.update(bundle.dispersive_kerr_densities)

    def prepare(self, spectrum: Complex[np.ndarray, "n_rows n_bins"]) -> None:
        """Filter the chunk's packed spectrum and transform every filtered field.

        Called after the stage has packed the chunk and before its inverse transform,
        which destroys the spectrum. In place, the stage's transform gives the
        filtered field.

        Parameters
        ----------
        spectrum : numpy.ndarray
            The packed spectrum (doubled positive bins on the analytic family).
        """
        if self.shifted:
            np.copyto(self.dc, spectrum[:, 0])
        filter_spectra(spectrum, self.stage.inputs, self.filtered_spectrum)
        if self.to_time is not None:
            self.to_time.execute()

    def respond(self, bare: np.ndarray, count: int = 0) -> None:
        """Form the chunk's products, at its nodes, onto its rows.

        Parameters
        ----------
        bare : numpy.ndarray
            The bare field at the chunk's nodes: the stage's own field of the chunk in
            free space, ``(rows, n_to)``; the backend's synthesised node buffer on the
            planar route. Read, and overwritten only by the lone symmetric response,
            whose filtered field it holds.
        count : int, optional
            The planar block's y rows, which a partial last block has fewer of; zero
            in free space, where the transforms take whole chunks.
        """
        if not self.layout.slab:
            self._respond_columns(bare)
        else:
            self._respond_nodes(bare, count)

    def _slab(self, matrix: np.ndarray, source: np.ndarray, target: np.ndarray) -> None:
        """Apply the slab rule in x to every y row of a block, in time.

        The backend's own batched real GEMM (``_slab_product``): a complex block is
        taken as its interleaved real view.

        Parameters
        ----------
        matrix : numpy.ndarray
            The synthesis ``(n_x, n_p)`` or the projection ``(n_p, n_x)``.
        source : numpy.ndarray
            The block's rows of the input, contiguous.
        target : numpy.ndarray
            The block's rows of the output, overwritten.
        """
        count = self._count
        if np.iscomplexobj(source):
            source, target = source.view(np.float64), target.view(np.float64)
        np.matmul(
            matrix,
            source.reshape(count, matrix.shape[1], -1),
            out=target.reshape(count, matrix.shape[0], -1),
        )

    def _respond_columns(self, bare: np.ndarray) -> None:
        """Form the products where the columns are the nodes.

        Parameters
        ----------
        bare : numpy.ndarray
            The stage's field of the chunk in time: the filtered field when in place.
        """
        rows, carrier = self.layout.rows, self.carrier
        # The bare analytic signal's zero-frequency constant, per column.
        offset = self.dc if self.shifted else self.offset
        if self.borrowed:
            self.self_product(bare, self.plans.projected, carrier, offset)
            return
        products = self.products
        if self.in_place:
            self.self_product(bare, products[:rows], carrier, offset)
            return
        if self.symmetric:
            for group in range(self.terms.n_fields):
                block = products[group * rows : (group + 1) * rows]
                self.self_product(block, block, carrier, offset)
            return
        assert self.pairs is not None
        pairs = self.stage.pairs
        # The derivative fields sit in the blocks their products take, and the bare
        # field is read until the cube, which writes a block of its own, is formed.
        for group in range(len(self.terms.groups)):
            first = products[(1 + 2 * group) * rows : (2 + 2 * group) * rows]
            second = products[(2 + 2 * group) * rows : (3 + 2 * group) * rows]
            self.pairs(
                bare, first, second, pairs[group, 0], pairs[group, 1], carrier, offset
            )
        self.self_product(bare, products[:rows], carrier, offset)

    def _respond_nodes(self, bare: np.ndarray, count: int) -> None:
        """Form the products at the planar block's nodes and project them to rows.

        Parameters
        ----------
        bare : numpy.ndarray
            The backend's node buffer of the block: the bare field, or, for the lone
            symmetric response filtering in place, the filtered one.
        count : int
            The block's y rows.
        """
        layout = self.layout
        self._count = count
        width = count * (layout.rows // layout.block)
        size = count * (layout.node_rows // layout.block)
        stride, slot_stride = layout.rows, layout.node_rows
        synthesis, projection = self.synthesis, self.projection
        carrier = self.carrier
        offset = self.offset[:size] if self.shifted else self.offset
        if self.shifted:
            self._slab(
                synthesis, self.dc[:width].reshape(width, 1), offset.reshape(size, 1)
            )
        bare = bare[:size]
        if self.borrowed or self.in_place:
            if not self.in_place:
                self._slab(synthesis, self.filtered[:width], bare)
            self.self_product(bare, bare, carrier, offset)
            target = self.plans.projected if self.borrowed else self.products
            self._slab(projection, bare, target[:width])
            return
        products, filtered = self.products, self.filtered
        first = self.slots[:size]
        if self.symmetric:
            for group in range(self.terms.n_fields):
                rows = slice(group * stride, group * stride + width)
                self._slab(synthesis, filtered[rows], first)
                self.self_product(first, first, carrier, offset)
                self._slab(projection, first, products[rows])
            return
        assert self.pairs is not None
        second = self.slots[slot_stride : slot_stride + size]
        self.self_product(bare, first, carrier, offset)
        self._slab(projection, first, products[:width])
        pairs = self.stage.pairs
        for group in range(len(self.terms.groups)):
            low = slice(2 * group * stride, 2 * group * stride + width)
            high = slice((2 * group + 1) * stride, (2 * group + 1) * stride + width)
            self._slab(synthesis, filtered[low], first)
            self._slab(synthesis, filtered[high], second)
            self.pairs(
                bare, first, second, pairs[group, 0], pairs[group, 1], carrier, offset
            )
            cross = (1 + 2 * group) * stride
            self._slab(projection, first, products[cross : cross + width])
            self._slab(
                projection, second, products[cross + stride : cross + stride + width]
            )

    def finish(self) -> None:
        """Apply the collar to the chunk's products and transform them.

        Before the stage's forward transform: a borrowed product is the stage's own
        polarisation buffer, which that transform takes.
        """
        if self.borrowed:
            apply_window(self.plans.projected, self.window)
            return
        apply_window(self.products, self.window)
        self.forward.execute()

    def combine(self) -> None:
        """Land the products' weighted spectra in the chunk's polarisation spectrum."""
        weights, target = self.stage.multipliers, self.plans.projected_spectrum
        if self.borrowed:
            weigh_spectrum(weights, target)
        elif self.layout.alone:
            combine_spectra(self.spectra, weights, target)
        else:
            add_spectra(self.spectra, weights, target)

    def memory_ledger(self) -> dict[str, int]:
        """Return the bytes of every buffer the response owns, by name.

        Returns
        -------
        dict
            Every row of ``DISPERSIVE_LEDGER_ROWS`` of
            :mod:`solprop.backends.numba_cpu.dispersive`.
        """
        return dict(self._owned)
