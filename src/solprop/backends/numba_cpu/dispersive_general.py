# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's frequency-dependent electronic response on the general modal route.

The general modal guide synthesises each node's three Cartesian components in
frequency, through maps that depend on frequency, and transforms them to the analytic
signal at the node (:mod:`solprop.backends.numba_cpu.general_modal`). So the input
multipliers act **in synthesis**: on the synthesised spectrum of a node chunk, before
the inverse transform, one batched inverse transform giving every filtered field of the
chunk (Phase 1b's decision, measured in Phase 2b). The products are formed at the nodes
of each channel's own region -- a gas region's fill is one channel per species -- and
zero elsewhere; one batched forward transform takes them to frequency, and their output
multipliers and the channels' ``eps0 gamma0 rho`` land them in the chunk's polarisation
spectrum after the route's own forward transform, ahead of the projection. The route
has no temporal collar. The symmetric evaluator forms one product per region, a
mixture's under its fill's mixture filter.

The sector is the leaf's: the full cubic beside ``kerr_form="real"``, formed on the
real field, and the number-conserving one beside ``"analytic"``, formed on the analytic
signals with ``Pi_+``; both contract the three components through the isotropic
Kleinman tensor (SEP 6, decision Q3). The polynomial groups the species of one region
with equal leg bounds, and forms one cube per region
(:class:`~solprop.backends.numba_cpu.dispersive.DispersiveTerms`).
"""

import time
from functools import cache
from typing import Any, NamedTuple, no_type_check

import numpy as np
from jaxtyping import Inexact, Int32
from numba import prange

from solprop._numba_cache import njit
from solprop.backends.numba_cpu import fft
from solprop.backends.numba_cpu.dispersive import (
    DISPERSIVE_LEDGER_ROWS,
    StageWeights,
    add_spectra,
    combine_spectra,
    dispersive_terms_layout,
    filter_spectra,
)
from solprop.backends.numba_cpu.kernels import HOST
from solprop.backends.numba_cpu.openmp import pyfftw
from solprop.translate.responses import DispersiveKerrTerms

__all__ = ["GeneralDispersive", "GeneralDispersiveKernels", "make_general_dispersive"]


class GeneralDispersiveKernels(NamedTuple):
    """The general route's product kernels for one form.

    Attributes
    ----------
    cube : numba.core.dispatcher.Dispatcher
        ``(source, target, node_region, region, first, count, offset)``: ``T(E, E, E)``
        at the chunk's nodes of one region, zero at the others.
    pairs : numba.core.dispatcher.Dispatcher
        ``(field, first_field, second_field, out_first, out_second, c0, c1,
        node_region, region, first, count, offset)``: ``T(E, E, F)`` and
        ``c0 T(E, E, H) + c1 T(E, F, F)`` likewise.
    """

    cube: Any
    pairs: Any


@cache
def make_general_dispersive(  # noqa: PLR0915 - two fused traversals, see the kernels
    form: str, *, parallel: bool, shifted: bool = False
) -> GeneralDispersiveKernels:
    """Compile the general route's three-component product kernels.

    Parameters
    ----------
    form : {'real', 'conserving'}
        The full cubic of the real field, or the number-conserving group of the
        analytic signals.
    parallel : bool
        Whether the node loop is a ``prange``.
    shifted : bool, optional
        Whether the bare field's zero-frequency constant is subtracted per row as it is
        read: the synthesised analytic signal keeps that bin once, which ``Pi_+``
        excludes.

    Returns
    -------
    GeneralDispersiveKernels
        The compiled kernels.
    """
    real = form == "real"
    contract = HOST.dispersive_isotropic3 if real else HOST.dispersive_conserving3
    loop = prange if parallel else range

    @no_type_check
    def cube(  # noqa: PLR0913, PLR0917 - a kernel's fields, region and chunk
        source: Inexact[np.ndarray, "n_rows n_to"],
        target: Inexact[np.ndarray, "n_rows n_to"],
        node_region: Int32[np.ndarray, " n_nodes"],
        region: int,
        first: int,
        count: int,
        offset: Inexact[np.ndarray, " n_rows_offset"],
    ) -> None:
        """Write ``T(E, E, E)`` at the region's nodes and zero at the others.

        Parameters
        ----------
        source : numpy.ndarray
            The field at the chunk's nodes, rows ``3 q + c``.
        target : numpy.ndarray
            Written, same row layout.
        node_region : numpy.ndarray
            Every node's region.
        region : int
            The region whose nodes respond.
        first : int
            The chunk's first node.
        count : int
            The chunk's node count.
        offset : numpy.ndarray
            The constant subtracted from each row when `shifted`.
        """
        n_to = target.shape[1]
        for node in loop(count):
            base = 3 * node
            inside = node_region[first + node] == region
            for k in range(n_to):
                if not inside:
                    for component in range(3):
                        target[base + component, k] = 0.0
                    continue
                x0, x1, x2 = source[base, k], source[base + 1, k], source[base + 2, k]
                if shifted:
                    x0 -= offset[base]
                    x1 -= offset[base + 1]
                    x2 -= offset[base + 2]
                p0, p1, p2 = contract(x0, x1, x2, x0, x1, x2, x0, x1, x2)
                target[base, k] = p0
                target[base + 1, k] = p1
                target[base + 2, k] = p2

    @no_type_check
    def pairs(  # noqa: PLR0913, PLR0917 - a kernel's fields, weights and region
        field: Inexact[np.ndarray, "n_rows n_to"],
        first_field: Inexact[np.ndarray, "n_rows n_to"],
        second_field: Inexact[np.ndarray, "n_rows n_to"],
        out_first: Inexact[np.ndarray, "n_rows n_to"],
        out_second: Inexact[np.ndarray, "n_rows n_to"],
        c0: float,
        c1: float,
        node_region: Int32[np.ndarray, " n_nodes"],
        region: int,
        first: int,
        count: int,
        offset: Inexact[np.ndarray, " n_rows_offset"],
    ) -> None:
        """Write the polynomial's two derivative products at the region's nodes.

        Parameters
        ----------
        field : numpy.ndarray
            The bare field ``E``, rows ``3 q + c``.
        first_field, second_field : numpy.ndarray
            ``D E`` and ``D^2 E``.
        out_first, out_second : numpy.ndarray
            Written with ``T(E, E, D E)`` and ``c0 T(E, E, D^2 E) + c1 T(E, D E,
            D E)``; zero at the other regions' nodes.
        c0, c1 : float
            The group's weights at the stage.
        node_region : numpy.ndarray
            Every node's region.
        region : int
            The region whose nodes respond.
        first : int
            The chunk's first node.
        count : int
            The chunk's node count.
        offset : numpy.ndarray
            The constant subtracted from each row of `field` when `shifted`.
        """
        n_to = out_first.shape[1]
        for node in loop(count):
            base = 3 * node
            inside = node_region[first + node] == region
            for k in range(n_to):
                if not inside:
                    for component in range(3):
                        out_first[base + component, k] = 0.0
                        out_second[base + component, k] = 0.0
                    continue
                x0, x1, x2 = field[base, k], field[base + 1, k], field[base + 2, k]
                if shifted:
                    x0 -= offset[base]
                    x1 -= offset[base + 1]
                    x2 -= offset[base + 2]
                f0 = first_field[base, k]
                f1 = first_field[base + 1, k]
                f2 = first_field[base + 2, k]
                h0 = second_field[base, k]
                h1 = second_field[base + 1, k]
                h2 = second_field[base + 2, k]
                p0, p1, p2 = contract(x0, x1, x2, x0, x1, x2, f0, f1, f2)
                a0, a1, a2 = contract(x0, x1, x2, x0, x1, x2, h0, h1, h2)
                b0, b1, b2 = contract(x0, x1, x2, f0, f1, f2, f0, f1, f2)
                out_first[base, k] = p0
                out_first[base + 1, k] = p1
                out_first[base + 2, k] = p2
                out_second[base, k] = c0 * a0 + c1 * b0
                out_second[base + 1, k] = c0 * a1 + c1 * b1
                out_second[base + 2, k] = c0 * a2 + c1 * b2

    # Never disk-cached: the closure captures dispatchers, whose pickled form differs
    # per process, and a parallel closure may not be cached.
    compile_ = njit(cache=False, nogil=True, parallel=parallel)
    return GeneralDispersiveKernels(cube=compile_(cube), pairs=compile_(pairs))


class _Layout(NamedTuple):
    """What the backend's metadata reads of a dispersive workspace's layout."""

    alone: bool


class GeneralDispersive:
    """The general route's dispersive buffers, plans and kernels, allocated once.

    Parameters
    ----------
    leaf : solprop.translate.responses.DispersiveKerrTerms
        The compiled leaf, carrying each channel's node region.
    node_region : numpy.ndarray
        Every node's region, int32.
    chunks : tuple of tuple of int
        The route's node chunks, ``(first, count)``.
    plans : solprop.backends.numba_cpu.fft.GeneralPlans
        The route's plans, whose synthesised spectrum, analytic signal and
        polarisation spectrum the response reads and adds into.
    parallel : bool
        Whether the node loops are ``prange``.
    flag : str
        The FFTW planning flag, a key of
        :data:`solprop.backends.numba_cpu.fft.PLAN_FLAGS`.
    threads : int
        The FFTW pool of the route's own plans.
    """

    def __init__(  # noqa: PLR0913 - the route's leaf, nodes, plans and configuration
        self,
        leaf: DispersiveKerrTerms,
        node_region: Int32[np.ndarray, " n_nodes"],
        chunks: tuple[tuple[int, int], ...],
        plans: fft.GeneralPlans,
        *,
        parallel: bool,
        flag: str,
        threads: int,
    ) -> None:
        assert leaf.regions is not None
        self.leaf, self.plans = leaf, plans
        self.node_region = np.ascontiguousarray(node_region, dtype=np.int32)
        self.layout = _Layout(alone=False)
        self.real = leaf.sector == "full"
        analytic = not self.real
        # Grouped by the leaf's own node regions.
        self.terms = dispersive_terms_layout(leaf, "field", analytic=analytic)
        self.stage = StageWeights(leaf, self.terms)
        self.symmetric = self.terms.symmetric
        self.shifted = analytic and not self.symmetric
        rows, n_to = plans.spectrum.shape
        n_bins = n_to // 2 + 1
        terms = self.terms
        # Which groups have a node in each chunk, decided once.
        self.live = tuple(
            any(
                np.any(self.node_region[start : start + count] == key)
                for key in terms.keys
            )
            for start, count in chunks
        )
        self._owned: dict[str, int] = dict.fromkeys(DISPERSIVE_LEDGER_ROWS, 0)
        heap_before = fft._heap_bytes()
        started = time.perf_counter()
        planning = fft.PLAN_FLAGS[flag]
        self.filtered_spectrum = self._aligned(
            "product_spectra", (terms.n_fields * rows, n_to), np.complex128
        )
        self.filtered = self._aligned(
            "filtered_time", (terms.n_fields * rows, n_to), np.complex128
        )
        self.to_time = fft._plan(
            self.filtered_spectrum,
            self.filtered,
            forward=False,
            flag=planning,
            threads=threads,
        )
        self.products = self._aligned(
            "products",
            (terms.n_products * rows, n_to),
            np.float64 if self.real else np.complex128,
        )
        self.spectra = self._aligned(
            "product_spectra",
            (terms.n_products * rows, n_bins if self.real else n_to),
            np.complex128,
        )
        self.forward = fft._plan(
            self.products, self.spectra, forward=True, flag=planning, threads=threads
        )
        self.planning_seconds = time.perf_counter() - started
        self._owned["dispersive_fft_plan_heap_growth"] = max(
            0, fft._heap_bytes() - heap_before
        )
        self.dc = np.zeros(rows if self.shifted else 1, dtype=np.complex128)
        self.kernels = make_general_dispersive(
            "real" if self.real else "conserving",
            parallel=parallel,
            shifted=self.shifted,
        )
        self._owned["dispersive_multipliers"] = self.stage.nbytes + self.dc.nbytes
        self._rows = rows

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

    def begin(self, densities: tuple[float, ...]) -> None:
        """Weigh the channels at a stage, once for all its chunks.

        Parameters
        ----------
        densities : tuple of float
            Each channel's number density, m^-3.
        """
        self.stage.update(densities)

    def respond(self, index: int, first: int, count: int) -> bool:
        """Filter a chunk's synthesised spectrum, form and transform its products.

        Called after the route's inverse transform of the synthesised spectrum, which
        leaves the spectrum intact (a complex-to-complex plan out of place).

        Parameters
        ----------
        index : int
            The chunk's position in the route's chunk list.
        first, count : int
            The chunk's first node and node count.

        Returns
        -------
        bool
            Whether the chunk has a node in any channel's region, so that
            :meth:`combine` has something to add.
        """
        if not self.live[index]:
            return False
        plans, terms, kernels = self.plans, self.terms, self.kernels
        rows = self._rows
        if self.shifted:
            np.copyto(self.dc, plans.spectrum[:, 0])
        filter_spectra(plans.spectrum, self.stage.inputs, self.filtered_spectrum)
        self.to_time.execute()
        filtered: np.ndarray = np.real(self.filtered) if self.real else self.filtered
        signal: np.ndarray = plans.signal
        bare = np.real(signal) if self.real else signal
        products, regions = self.products, self.node_region
        if self.symmetric:
            for group, key in enumerate(terms.keys):
                block = slice(group * rows, (group + 1) * rows)
                kernels.cube(
                    filtered[block],
                    products[block],
                    regions,
                    key,
                    first,
                    count,
                    self.dc,
                )
        else:
            for cube in range(terms.n_cubes):
                key = terms.keys[terms.cubes.index(cube)]
                block = slice(cube * rows, (cube + 1) * rows)
                kernels.cube(bare, products[block], regions, key, first, count, self.dc)
            pairs = self.stage.pairs
            for group, key in enumerate(terms.keys):
                low, high = terms.pair_products(group)
                kernels.pairs(
                    bare,
                    filtered[2 * group * rows : (2 * group + 1) * rows],
                    filtered[(2 * group + 1) * rows : (2 * group + 2) * rows],
                    products[low * rows : (low + 1) * rows],
                    products[high * rows : (high + 1) * rows],
                    pairs[group, 0],
                    pairs[group, 1],
                    regions,
                    key,
                    first,
                    count,
                    self.dc,
                )
        self.forward.execute()
        return True

    def combine(self, *, overwrite: bool = False) -> None:
        """Land the products' weighted spectra in the chunk's polarisation spectrum.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to write the spectrum rather than add to it: when no regional
            tensor or delayed response made one.
        """
        target = self.plans.polarisation_spectrum
        if overwrite:
            combine_spectra(self.spectra, self.stage.multipliers, target)
        else:
            add_spectra(self.spectra, self.stage.multipliers, target)

    def memory_ledger(self) -> dict[str, int]:
        """Return the bytes of every buffer the response owns, by name.

        Returns
        -------
        dict
            Every row of ``DISPERSIVE_LEDGER_ROWS`` of
            :mod:`solprop.backends.numba_cpu.dispersive`.
        """
        return dict(self._owned)
