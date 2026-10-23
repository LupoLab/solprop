# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's frequency-dependent electronic response on the fused node path.

The leaf ``responses.dispersive_kerr`` carries each species' spectral multipliers on the
fine grid; :func:`solprop.nonlinear.responses.dispersive_cubic` is the arithmetic, and
:func:`solprop.backends.numpy_ref.passes.dispersive_kerr_polarisation` evaluates it at
every node. This module evaluates the same operator with the multipliers moved off the
nodes and onto the modal columns, where they cost the fewest transforms.

**Why the columns.** The legacy transverse transform -- one mode-averaged node, a fixed
multimode quadrature, the radial Hankel pair -- synthesises and projects with real
matrices that do not depend on frequency, and the temporal collar is the same at every
node. A spectral multiplier and the collar therefore commute with both:
``P G[w (G E)^3] = G[w P (G E)^3]`` for a projection ``P``, a filter ``G`` and the
collar ``w``. So

1. the input multipliers act on the modal spectrum the stage combination has already
   formed, before its inverse transform (no forward transform of the node field), and
   one batched inverse transform gives every filtered modal field;
2. the filtered fields are synthesised at the nodes of a chunk by the same real GEMM as
   the bare field, and the products are formed there, in the leaf's sector;
3. each product is projected onto the modal columns (and onto the embedded rule's, for
   the quadrature alarm) by the same GEMM as the polarisation;
4. the collar multiplies the projected products in time, one batched forward transform
   takes them to frequency, and their output multipliers and the channels'
   ``eps0 gamma0 rho``, summed per product, land them in the stage's own polarisation
   spectrum, after the stage's forward transform has made it of every other response.

The collar acts before the output multipliers, ``G[w (G E)^3]`` (John Travers,
2026-10-08), so the response has no inverse transform of its own: its output joins the
stage in frequency.

**When the response is alone** -- no constant Kerr term, Raman, plasma or envelope
plasma beside it -- the response column and the polarisation's projection are skipped:
the node field buffer holds the filtered field and its product in place, and the stage's
forward transform is skipped, the response writing the polarisation spectrum itself. A
symmetric response of one species whose product has the polarisation's dtype goes
further: its product is projected into the stage's own projected buffer, the stage's
forward transform is the product's, and ``g`` finishes it in the stage's spectrum -- the
constant cubic's transforms exactly -- and with no in-stage absorber reading the bare
modal field it filters the stage's spectrum in place, holding no field-sized buffer at
all. Otherwise the bare field stays where Raman, plasma and Kerr read it, and the
response holds its own buffers (:meth:`DispersiveWorkspace.memory_ledger`).

**Mixtures.** The symmetric evaluator forms one product for a whole mixture, under the
filter built from the mixture's own Kerr diagonal,
``g_mix = [sum_s f_s g_s^4 / sum_s f_s]^(1/4)`` with ``f_s = eps0 gamma3_0 rho_s``
(:func:`solprop.nonlinear.responses.mixture_filter`), rebuilt from the channels'
densities whenever they change (:class:`StageWeights`); a mixture costs what one gas
costs, and takes the lone path when it is alone. The polynomial groups species with
equal leg bounds, which share their derivative fields and add their coefficient tables
density-weighted at every stage (:class:`DispersiveTerms`): an exact rewriting of the
per-species sum, whose transform count then grows with the number of distinct leg
bounds rather than of species.

The embedded rule's products are projected at every stage, as the polarisation's are,
and windowed, transformed and combined only when a step is accepted
(:meth:`DispersiveWorkspace.finish_coarse`), from the last stage's buffers, as the
envelope currents' are.
"""

import time
from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple, no_type_check

import numpy as np
from jaxtyping import Complex, Float, Inexact, Int
from numba import prange

from solprop._numba_cache import njit
from solprop.backends.numba_cpu import fft
from solprop.backends.numba_cpu.kernels import HOST
from solprop.backends.numba_cpu.openmp import pyfftw
from solprop.translate.problem import KernelProblem
from solprop.translate.responses import DispersiveKerrTerms
from solprop.translate.stage import StageScalars

__all__ = [
    "DISPERSIVE_LEDGER_ROWS",
    "DispersiveLayout",
    "DispersiveMatrices",
    "DispersiveTerms",
    "DispersiveWorkspace",
    "StageWeights",
    "add_spectra",
    "apply_window",
    "combine_spectra",
    "dispersive_form",
    "dispersive_terms_layout",
    "filter_spectra",
    "make_dispersive_pairs",
    "make_dispersive_self",
    "mixture_rows",
    "weigh_spectrum",
    "weigh_terms",
]

DISPERSIVE_LEDGER_ROWS = (
    "dispersive_filtered_time",
    "dispersive_node_fields",
    "dispersive_products",
    "dispersive_coarse_products",
    "dispersive_product_spectra",
    "dispersive_chunk_scratch",
    "dispersive_synthesis",
    "dispersive_multipliers",
    "dispersive_fft_plan_heap_growth",
)
"""The response's rows of the memory ledger, named (at zero) on every problem."""


def dispersive_form(kind: str, sector: str) -> str:
    """Return the product form of a grid kind and a sector.

    Parameters
    ----------
    kind : {'field', 'envelope'}
        The spectral layout.
    sector : {'full', 'conserving'}
        Which quartets the leaf keeps.

    Returns
    -------
    str
        ``"real"`` (the full cubic of a real field), ``"conserving"`` (the
        number-conserving group of analytic signals or envelopes) or ``"harmonic"``
        (an envelope's full cubic: the conserving group and both harmonic groups).

    Examples
    --------
    >>> from solprop.backends.numba_cpu.dispersive import dispersive_form
    >>> dispersive_form("field", "full"), dispersive_form("envelope", "full")
    ('real', 'harmonic')
    """
    if kind == "field":
        return "real" if sector == "full" else "conserving"
    return "harmonic" if sector == "full" else "conserving"


# --- the kernels ------------------------------------------------------------------


@cache
def make_dispersive_self(
    form: str, n_pol: int, *, parallel: bool, shifted: bool = False
) -> Any:
    """Compile the product ``(u, u, u)`` of one field at every node of a chunk.

    The symmetric evaluator's only product, of the filtered field, and the polynomial's
    ``E^3``. It may run in place: each sample reads every component before it writes.

    Parameters
    ----------
    form : {'real', 'conserving', 'harmonic'}
        The product form (:func:`dispersive_form`).
    n_pol : int
        Components at one node, 1 or 2.
    parallel : bool
        Whether the node loop is a ``prange``.
    shifted : bool, optional
        Whether a constant per row is subtracted from the source as it is read: the
        zero-frequency bin of a field grid's bare analytic signal, which ``Pi_+``
        excludes and the backend's analytic family keeps.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        The compiled kernel ``(source, target, carrier, offset)``.
    """
    two = n_pol == 2
    real = form == "real"
    harmonic = form == "harmonic"
    isotropic = HOST.dispersive_isotropic
    conserving = HOST.dispersive_conserving
    harmonic_groups = HOST.dispersive_harmonic
    real_self = HOST.dispersive_real_self
    conserving_self = HOST.dispersive_conserving_self
    harmonic_self = HOST.dispersive_harmonic_self

    @no_type_check
    def self_product(
        source: Inexact[np.ndarray, "n_rows n_to"],
        target: Inexact[np.ndarray, "n_rows n_to"],
        carrier: Complex[np.ndarray, " n_to"],
        offset: Inexact[np.ndarray, " n_rows_offset"],
    ) -> None:
        """Write the product of each node's field with itself twice.

        Parameters
        ----------
        source : numpy.ndarray
            The field at the chunk's nodes, rows ``node * n_pol + component``: real on
            a field grid's full cubic, the analytic signal or envelope otherwise.
        target : numpy.ndarray
            Written with the product, same shape and dtype; may be `source`.
        carrier : numpy.ndarray
            ``exp(2 i omega0 t)`` for the harmonic form; any array otherwise.
        offset : numpy.ndarray
            The constant subtracted from each row when `shifted`; any array otherwise.
        """
        n_rows, n_to = source.shape
        for node in prange(n_rows // n_pol):
            base = node * n_pol
            for k in range(n_to):
                if two:
                    x0 = source[base, k]
                    x1 = source[base + 1, k]
                    if shifted:
                        x0 = x0 - offset[base]
                        x1 = x1 - offset[base + 1]
                    if real:
                        p0, p1 = isotropic(x0, x1, x0, x1, x0, x1)
                    else:
                        p0, p1 = conserving(x0, x1, x0, x1, x0, x1)
                        if harmonic:
                            h0, h1 = harmonic_groups(x0, x1, x0, x1, x0, x1, carrier[k])
                            p0 = p0 + h0
                            p1 = p1 + h1
                    target[base, k] = p0
                    target[base + 1, k] = p1
                elif real:
                    target[base, k] = real_self(source[base, k])
                else:
                    u = source[base, k]
                    if shifted:
                        u = u - offset[base]
                    value = conserving_self(u)
                    if harmonic:
                        value = value + harmonic_self(u, carrier[k])
                    target[base, k] = value

    # Never disk-cached: the closure captures dispatchers, whose pickled form differs
    # per process, and a parallel closure may not be cached.
    return njit(cache=False, nogil=True, parallel=parallel)(self_product)


@cache
def make_dispersive_pairs(  # noqa: PLR0915 - one fused traversal, see the kernel
    form: str, n_pol: int, *, parallel: bool, shifted: bool = False
) -> Any:
    """Compile the polynomial's two derivative products at every node of a chunk.

    With ``E`` the bare field, ``F = D E`` and ``H = D^2 E``, it writes
    ``T(E, E, F)`` over ``F`` and ``c0 T(E, E, H) + c1 T(E, F, F)`` over ``H``, in
    place: the grouped polynomial's second and third terms, whose output multipliers
    are ``3 a + 6 b d`` and one (SEP 6, section 4.5 and the Phase 0 report).

    Parameters
    ----------
    form : {'real', 'conserving', 'harmonic'}
        The product form (:func:`dispersive_form`).
    n_pol : int
        Components at one node, 1 or 2.
    parallel : bool
        Whether the node loop is a ``prange``.
    shifted : bool, optional
        Whether the bare field's zero-frequency constant is subtracted per row as it is
        read (:func:`make_dispersive_self`).

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        The compiled kernel ``(field, first, second, c0, c1, carrier, offset)``.
    """
    two = n_pol == 2
    real = form == "real"
    harmonic = form == "harmonic"
    isotropic = HOST.dispersive_isotropic
    conserving = HOST.dispersive_conserving
    harmonic_groups = HOST.dispersive_harmonic
    real_pair, real_cross = HOST.dispersive_real_pair, HOST.dispersive_real_cross
    conserving_pair = HOST.dispersive_conserving_pair
    conserving_cross = HOST.dispersive_conserving_cross
    harmonic_pair = HOST.dispersive_harmonic_pair
    harmonic_cross = HOST.dispersive_harmonic_cross

    @no_type_check
    def pairs(  # noqa: PLR0913, PLR0917 - a kernel's fields, weights and carrier
        field: Inexact[np.ndarray, "n_rows n_to"],
        first: Inexact[np.ndarray, "n_rows n_to"],
        second: Inexact[np.ndarray, "n_rows n_to"],
        c0: float,
        c1: float,
        carrier: Complex[np.ndarray, " n_to"],
        offset: Inexact[np.ndarray, " n_rows_offset"],
    ) -> None:
        """Overwrite ``F`` and ``H`` with the two derivative products.

        Parameters
        ----------
        field : numpy.ndarray
            The bare field ``E`` at the chunk's nodes.
        first : numpy.ndarray
            ``D E`` at the same nodes; overwritten with ``T(E, E, D E)``.
        second : numpy.ndarray
            ``D^2 E``; overwritten with ``c0 T(E, E, D^2 E) + c1 T(E, D E, D E)``.
        c0, c1 : float
            ``3 (b + b')`` and ``6 b``, m^4.
        carrier : numpy.ndarray
            ``exp(2 i omega0 t)`` for the harmonic form; any array otherwise.
        offset : numpy.ndarray
            The constant subtracted from each row of `field` when `shifted`.
        """
        n_rows, n_to = field.shape
        for node in prange(n_rows // n_pol):
            base = node * n_pol
            for k in range(n_to):
                if two:
                    x0, x1 = field[base, k], field[base + 1, k]
                    if shifted:
                        x0, x1 = x0 - offset[base], x1 - offset[base + 1]
                    f0, f1 = first[base, k], first[base + 1, k]
                    h0, h1 = second[base, k], second[base + 1, k]
                    if real:
                        p0, p1 = isotropic(x0, x1, x0, x1, f0, f1)
                        a0, a1 = isotropic(x0, x1, x0, x1, h0, h1)
                        b0, b1 = isotropic(x0, x1, f0, f1, f0, f1)
                    else:
                        p0, p1 = conserving(x0, x1, x0, x1, f0, f1)
                        a0, a1 = conserving(x0, x1, x0, x1, h0, h1)
                        b0, b1 = conserving(x0, x1, f0, f1, f0, f1)
                        if harmonic:
                            c = carrier[k]
                            g0, g1 = harmonic_groups(x0, x1, x0, x1, f0, f1, c)
                            p0, p1 = p0 + g0, p1 + g1
                            g0, g1 = harmonic_groups(x0, x1, x0, x1, h0, h1, c)
                            a0, a1 = a0 + g0, a1 + g1
                            g0, g1 = harmonic_groups(x0, x1, f0, f1, f0, f1, c)
                            b0, b1 = b0 + g0, b1 + g1
                    first[base, k] = p0
                    first[base + 1, k] = p1
                    second[base, k] = c0 * a0 + c1 * b0
                    second[base + 1, k] = c0 * a1 + c1 * b1
                elif real:
                    x, f, h = field[base, k], first[base, k], second[base, k]
                    first[base, k] = real_pair(x, f)
                    second[base, k] = c0 * real_pair(x, h) + c1 * real_cross(x, f)
                else:
                    u, f, h = field[base, k], first[base, k], second[base, k]
                    if shifted:
                        u = u - offset[base]
                    pair = conserving_pair(u, f)
                    weighted = c0 * conserving_pair(u, h) + c1 * conserving_cross(u, f)
                    if harmonic:
                        c = carrier[k]
                        pair = pair + harmonic_pair(u, f, c)
                        weighted = weighted + (
                            c0 * harmonic_pair(u, h, c) + c1 * harmonic_cross(u, f, c)
                        )
                    first[base, k] = pair
                    second[base, k] = weighted

    # Never disk-cached, for the self product's two reasons.
    return njit(cache=False, nogil=True, parallel=parallel)(pairs)


@njit(cache=True, nogil=True)
@no_type_check
def filter_spectra(
    spectrum: Complex[np.ndarray, "n_columns n_bins"],
    multipliers: Float[np.ndarray, "n_fields n_weighted"],
    out: Complex[np.ndarray, "n_rows n_bins"],
) -> None:
    """Write each input multiplier times the modal spectrum, one block per field.

    Row ``field * n_columns + column`` of `out` is ``multipliers[field] *
    spectrum[column]`` on the first ``n_weighted`` bins and zero beyond, which is where
    an analytic family's full-length spectrum holds nothing.

    Parameters
    ----------
    spectrum : numpy.ndarray
        The padded modal spectrum the stage's inverse transform reads, complex.
    multipliers : numpy.ndarray
        The input multipliers on the leaf's spectral layout, float64.
    out : numpy.ndarray
        Written whole, shape ``(n_fields * n_columns, n_bins)``: a complex-to-real plan
        destroys its input, so nothing written once can be relied on.
    """
    n_columns, n_bins = spectrum.shape
    n_fields, n_weighted = multipliers.shape
    for field in range(n_fields):
        for column in range(n_columns):
            row = field * n_columns + column
            for k in range(n_weighted):
                out[row, k] = multipliers[field, k] * spectrum[column, k]
            for k in range(n_weighted, n_bins):
                out[row, k] = 0.0 + 0.0j


@njit(cache=True, nogil=True)
@no_type_check
def mixture_rows(  # noqa: PLR0913, PLR0917 - a kernel's tables, members and outputs
    quartics: Float[np.ndarray, "n_channels n_bins"],
    members: Int[np.ndarray, " n_members"],
    factors: Float[np.ndarray, " n_factors"],
    input_scale: Float[np.ndarray, " n_bins"],
    output_scale: Float[np.ndarray, " n_bins"],
    inputs: Float[np.ndarray, " n_bins"],
    outputs: Float[np.ndarray, " n_bins"],
) -> float:
    """Write a symmetric mixture's filter rows at a stage; return its coefficient.

    ``g_mix = [sum_s f_s K_s / sum_s f_s]^(1/4)`` over the group's channels, with
    ``K_s = g_s^4`` and ``f_s`` the channel's ``eps0 gamma3_0 rho`` at the stage: the
    arithmetic of :func:`solprop.nonlinear.responses.mixture_filter` -- the weighted
    species sum, the fourth root as two square roots -- within its rounding bound (a
    sum over a few channels, SEP 25), not to its bits: the order of the sum is free and
    the bin loop may vectorise. At a vacuum (no positive total) the filter is one and
    the coefficient zero.

    Parameters
    ----------
    quartics : numpy.ndarray
        ``K_s = (g_s^2)^2`` per channel on the leaf's layout.
    members : numpy.ndarray
        The group's channels, int64.
    factors : numpy.ndarray
        ``f_s`` per channel at the stage (at least the members').
    input_scale : numpy.ndarray
        What the input multiplier carries besides ``g_mix``: ``Pi_+`` on a field
        grid's analytic family, one otherwise.
    output_scale : numpy.ndarray
        The sector's ``Pi_+`` and factor, which the output multiplier carries.
    inputs, outputs : numpy.ndarray
        Written with ``g_mix input_scale`` and ``g_mix output_scale``.

    Returns
    -------
    float
        ``sum_s f_s``, the mixture's static coefficient at the stage.
    """
    n_bins = inputs.shape[0]
    total = 0.0
    for member in range(members.shape[0]):
        total += factors[members[member]]
    if not total > 0.0:
        for k in range(n_bins):
            inputs[k] = input_scale[k]
            outputs[k] = output_scale[k]
        return 0.0
    for k in range(n_bins):
        quartic = 0.0
        for member in range(members.shape[0]):
            channel = members[member]
            quartic += factors[channel] * quartics[channel, k]
        curve = np.sqrt(np.sqrt(quartic / total))
        inputs[k] = curve * input_scale[k]
        outputs[k] = curve * output_scale[k]
    return total


@njit(cache=True, nogil=True)
@no_type_check
def weigh_terms(
    rows: Int[np.ndarray, " n_terms"],
    channels: Int[np.ndarray, " n_terms"],
    multipliers: Float[np.ndarray, "n_terms n_bins"],
    factors: Float[np.ndarray, " n_channels"],
    out: Float[np.ndarray, "n_products n_bins"],
) -> None:
    """Sum the output terms' multipliers that share a product, at this stage.

    ``out[r] = sum_{t: row_t = r} factors[channel_t] multipliers[t]``: each term one
    product of one channel, its multiplier carrying the sector's ``Pi_+`` and factor,
    weighted by the channel's ``eps0 gamma3_0 rho``. Products shared by channels -- the
    polynomial's ``E^3`` and a group's ``T(E, E, D E)`` -- get one multiplier.

    Parameters
    ----------
    rows : numpy.ndarray
        Each term's product, int64.
    channels : numpy.ndarray
        Each term's channel, int64.
    multipliers : numpy.ndarray
        Each term's output multiplier, float64.
    factors : numpy.ndarray
        ``eps0 gamma3_0 rho`` per channel at this stage.
    out : numpy.ndarray
        Written with one multiplier per product.
    """
    n_products, n_bins = out.shape
    for product in range(n_products):
        for k in range(n_bins):
            out[product, k] = 0.0
    for term in range(rows.shape[0]):
        factor = factors[channels[term]]
        product = rows[term]
        for k in range(n_bins):
            out[product, k] += factor * multipliers[term, k]


@njit(cache=True, nogil=True)
@no_type_check
def combine_spectra(
    spectra: Complex[np.ndarray, "n_rows n_bins_product"],
    weights: Float[np.ndarray, "n_products n_bins"],
    out: Complex[np.ndarray, "n_columns n_bins"],
) -> None:
    """Sum the projected products' spectra under their stage multipliers.

    ``out[c, k] = sum_r weights[r, k] spectra[r C + c, k]``. `out` may be the first
    block of `spectra`: the first product writes each bin from itself, and every later
    product reads a later block.

    Parameters
    ----------
    spectra : numpy.ndarray
        The forward transforms of the projected products, one block of ``n_columns``
        rows per product, at least ``n_bins`` long.
    weights : numpy.ndarray
        One multiplier per product (:func:`weigh_terms`).
    out : numpy.ndarray
        Written with the summed spectrum, ``(n_columns, n_bins)``.
    """
    n_columns, n_bins = out.shape
    for column in range(n_columns):
        for k in range(n_bins):
            out[column, k] = weights[0, k] * spectra[column, k]
    for product in range(1, weights.shape[0]):
        for column in range(n_columns):
            row = product * n_columns + column
            for k in range(n_bins):
                out[column, k] += weights[product, k] * spectra[row, k]


@njit(cache=True, nogil=True)
@no_type_check
def add_spectra(
    spectra: Complex[np.ndarray, "n_rows n_bins_product"],
    weights: Float[np.ndarray, "n_products n_bins"],
    out: Complex[np.ndarray, "n_columns n_bins"],
) -> None:
    """Add the projected products' spectra under their stage multipliers.

    ``out[c, k] += sum_r weights[r, k] spectra[r C + c, k]``: the response's share of
    the polarisation spectrum, added to what the stage's forward transform made of
    every other response's. The products' sum is formed first, in product order, and
    added once, so the result does not depend on what `out` held.

    Parameters
    ----------
    spectra : numpy.ndarray
        The forward transforms of the windowed products, one block of ``n_columns``
        rows per product, at least ``n_bins`` long.
    weights : numpy.ndarray
        One multiplier per product (:func:`weigh_terms`).
    out : numpy.ndarray
        The stage's polarisation spectrum, ``(n_columns, n_bins)``, added to.
    """
    n_columns, n_bins = out.shape
    for column in range(n_columns):
        for k in range(n_bins):
            total = weights[0, k] * spectra[column, k]
            for product in range(1, weights.shape[0]):
                total += weights[product, k] * spectra[product * n_columns + column, k]
            out[column, k] += total


@njit(cache=True, nogil=True)
@no_type_check
def weigh_spectrum(
    weights: Float[np.ndarray, "n_products n_bins"],
    spectrum: Complex[np.ndarray, "n_columns n_bins"],
) -> None:
    """Multiply the stage's spectrum, in place, by the one product's multiplier.

    The lone symmetric response of one species: its windowed product is the stage's
    projected polarisation, the stage's forward transform has made its spectrum, and
    the output multiplier ``g`` with ``eps0 gamma3_0 rho`` finishes it there.

    Parameters
    ----------
    weights : numpy.ndarray
        The stage multipliers (:func:`weigh_terms`); row zero is read.
    spectrum : numpy.ndarray
        The stage's polarisation spectrum, modified in place.
    """
    n_columns, n_bins = spectrum.shape
    for column in range(n_columns):
        for k in range(n_bins):
            spectrum[column, k] = weights[0, k] * spectrum[column, k]


@njit(cache=True, nogil=True)
@no_type_check
def apply_window(
    target: Inexact[np.ndarray, "n_columns n_to"],
    window: Float[np.ndarray, " n_to"],
) -> None:
    """Multiply the response's products by the temporal collar, in place.

    The collar acts on each product before ``Pi_+`` and the output multipliers:
    ``G[w (G E)^3]`` (SEP 6, Phase 2a, open question 1, decided 2026-10-08).

    Parameters
    ----------
    target : numpy.ndarray
        The products in time, one row per column of each product.
    window : numpy.ndarray
        The temporal collar ``grids.t_window_fine``.
    """
    n_columns, n_to = target.shape
    for column in range(n_columns):
        for k in range(n_to):
            target[column, k] = window[k] * target[column, k]


def _gemm(array: np.ndarray) -> np.ndarray:
    """Return an array as a real GEMM operand: its interleaved view when complex.

    Parameters
    ----------
    array : numpy.ndarray
        A contiguous 2-D array.

    Returns
    -------
    numpy.ndarray
        The array, or its float64 view.
    """
    return array.view(np.float64) if np.iscomplexobj(array) else array


# --- the layout of fields, products and terms --------------------------------------


class DispersiveTerms(NamedTuple):
    """The response's filtered fields, products and output terms, laid out once.

    The channels are grouped as the leaf says
    (:meth:`~solprop.translate.responses.DispersiveKerrTerms.groups`). The symmetric
    evaluator forms one filtered field and one product per group: a group of one
    species takes its own ``g``, fixed, and a **mixture** -- a group of several, the
    channels that act at the same nodes -- the filter of its own Kerr diagonal, which
    :class:`StageWeights` rebuilds from the channels' densities
    (:func:`mixture_rows`; SEP 6, section 6, adopted for mixtures on 2026-10-08). The
    polynomial's channels are **grouped by their leg weight**: species whose leg bounds
    are equal have the same ``d``, so they share ``D E`` and ``D^2 E`` and their four
    coefficient tables add, density-weighted, at every stage -- ``sum_s N_s gamma0_s
    (1, A_s, B_s, B'_s)`` -- an exact rewriting of the per-species sum (SEP 6, section
    6). The first ``n_cubes`` products are ``E^3``, one per key -- a single one shared
    by every channel, but on the general modal route one per node region, whose nodes
    alone it is formed at; group ``g`` then owns products ``n_cubes + 2 g``,
    ``T(E, E, D E)``, and ``n_cubes + 2 g + 1``,
    ``c0_g T(E, E, D^2 E) + c1_g T(E, D E, D E)`` with
    ``c0_g = sum_s f_s 3 (b_s + b'_s)`` and ``c1_g = sum_s f_s 6 b_s`` over its
    channels, ``f_s = eps0 gamma0_s rho_s`` at the stage.

    Attributes
    ----------
    symmetric : bool
        Whether the evaluator is the symmetric one.
    groups : tuple of tuple of int
        The channels of each group, in channel order.
    keys : tuple of int
        Each group's key: its node region on the general modal route, zero otherwise.
    cubes : tuple of int
        Each group's cube product, for the polynomial; empty for the symmetric
        evaluator.
    mixtures : tuple of int
        The symmetric groups of more than one channel, whose filter is rebuilt per
        stage; empty for the polynomial. Mixture ``m``'s field and product are its
        group's, and its coefficient is factor ``n_channels + 1 + m``.
    input_multipliers : numpy.ndarray
        One row per filtered field: ``g`` per group, or ``d`` and ``d^2`` per group,
        times ``Pi_+`` on a field grid's analytic family; a mixture's row is its
        stage's.
    term_rows : numpy.ndarray
        Each output term's product, int64.
    term_channels : numpy.ndarray
        Each term's factor, int64: its channel; ``n_channels``, the unit factor, for a
        product that already carries its channels' factors; ``n_channels + 1 + m`` for
        mixture ``m``.
    term_multipliers : numpy.ndarray
        Each term's output multiplier times the sector's ``Pi_+`` and factor, float64;
        a mixture's row is its stage's.
    pair_constants : numpy.ndarray
        ``3 (b + b')`` and ``6 b`` per channel, m^4, shape ``(n_channels, 2)``; zero for
        the symmetric evaluator.
    quartics : numpy.ndarray
        ``g_s^4`` per channel of a mixture, ``(n_channels, n_bins)``; ``(0, 0)`` without
        a mixture.
    input_scale : numpy.ndarray
        What a symmetric input multiplier carries besides ``g``: ``Pi_+`` on a field
        grid's analytic family, one otherwise.
    output_scale : numpy.ndarray
        The sector's ``Pi_+`` and factor, which every output multiplier carries.
    """

    symmetric: bool
    groups: tuple[tuple[int, ...], ...]
    keys: tuple[int, ...]
    cubes: tuple[int, ...]
    mixtures: tuple[int, ...]
    input_multipliers: np.ndarray
    term_rows: np.ndarray
    term_channels: np.ndarray
    term_multipliers: np.ndarray
    pair_constants: np.ndarray
    quartics: np.ndarray
    input_scale: np.ndarray
    output_scale: np.ndarray

    @property
    def n_fields(self) -> int:
        """Return the number of filtered fields.

        Returns
        -------
        int
            One per group (symmetric) or two per group (polynomial).
        """
        return int(self.input_multipliers.shape[0])

    @property
    def n_products(self) -> int:
        """Return the number of products.

        Returns
        -------
        int
            One per group (symmetric), or ``n_cubes + 2 n_groups`` (polynomial).
        """
        if self.symmetric:
            return len(self.groups)
        return self.n_cubes + 2 * len(self.groups)

    @property
    def n_cubes(self) -> int:
        """Return the number of cube products.

        Returns
        -------
        int
            One per distinct key for the polynomial, none for the symmetric evaluator.
        """
        return len(set(self.cubes))

    def pair_products(self, group: int) -> tuple[int, int]:
        """Return a polynomial group's two derivative products.

        Parameters
        ----------
        group : int
            The group.

        Returns
        -------
        tuple of int
            ``T(E, E, D E)``'s product and the weighted pair's.
        """
        first = self.n_cubes + 2 * group
        return first, first + 1


def dispersive_terms_layout(
    leaf: DispersiveKerrTerms,
    kind: str,
    *,
    analytic: bool,
    keys: tuple[int, ...] | None = None,
) -> DispersiveTerms:
    """Lay out a leaf's filtered fields, products and output terms.

    Every multiplier carries the sector's own factor: one on a field grid's full
    cubic, ``Pi_+ / 8`` for its conserving sector (the one-sided spectrum of the real
    part of a quarter of the analytic polarisation), ``Pi_+ / 4`` on an envelope grid.
    No transform normalisation: the products' spectra land in the stage's own
    polarisation spectrum, which its forward transform made unnormalised too.

    Parameters
    ----------
    leaf : solprop.translate.responses.DispersiveKerrTerms
        The compiled leaf.
    kind : {'field', 'envelope'}
        The spectral layout.
    analytic : bool
        Whether the products are a field grid's analytic signals.
    keys : tuple of int or None, optional
        A key per channel that must also agree for two channels to share a group: the
        general modal route's node region. ``None`` takes the leaf's own regions, a
        single key where it has none.

    Returns
    -------
    DispersiveTerms
        The layout.
    """
    symmetric = leaf.evaluator == "symmetric"
    n_channels = leaf.n_channels
    positive = leaf.positive.astype(np.float64)
    if kind == "envelope":
        scale = positive / 4.0
    elif analytic:
        scale = positive / 8.0
    else:
        scale = np.ones(positive.shape)
    input_scale = positive if analytic else np.ones(positive.shape)
    groups = leaf.groups(keys)
    if keys is None and leaf.regions is not None:
        keys = tuple(int(region) for region in leaf.regions)
    group_keys = tuple(0 if keys is None else int(keys[group[0]]) for group in groups)
    distinct = list(dict.fromkeys(group_keys))
    cubes = () if symmetric else tuple(distinct.index(key) for key in group_keys)
    n_cubes = len(distinct)
    mixtures = tuple(
        index for index, group in enumerate(groups) if symmetric and len(group) > 1
    )
    fields = []
    rows, channels, multipliers = [], [], []
    for index, group in enumerate(groups):
        weight = leaf.filters[group[0]]
        if symmetric:
            # A mixture's rows are its stage's; the first species' stand until then.
            fields.append(weight)
            rows.append(index)
            channels.append(
                n_channels + 1 + mixtures.index(index)
                if index in mixtures
                else group[0]
            )
            multipliers.append(weight * scale)
            continue
        fields.extend([weight, weight * weight])
        assert leaf.outputs is not None
        first = n_cubes + 2 * index
        for channel in group:
            output = leaf.outputs[channel]
            rows.extend([cubes[index], first])
            channels.extend([channel, channel])
            multipliers.extend([output[0] * scale, output[1] * scale])
        rows.append(first + 1)
        channels.append(n_channels)
        multipliers.append(scale.copy())
    stacked = np.array(fields, dtype=np.float64)
    if analytic:
        stacked = stacked * leaf.positive
    constants = (
        np.zeros((n_channels, 2))
        if leaf.products is None
        else np.array(leaf.products, dtype=np.float64)
    )
    if mixtures:
        squares = leaf.filters * leaf.filters
        quartics = np.ascontiguousarray(squares * squares)
    else:
        quartics = np.zeros((0, 0))
    return DispersiveTerms(
        symmetric=symmetric,
        groups=groups,
        keys=group_keys,
        cubes=cubes,
        mixtures=mixtures,
        input_multipliers=np.ascontiguousarray(stacked),
        term_rows=np.array(rows, dtype=np.int64),
        term_channels=np.array(channels, dtype=np.int64),
        term_multipliers=np.ascontiguousarray(np.array(multipliers)),
        pair_constants=np.ascontiguousarray(constants),
        quartics=quartics,
        input_scale=np.ascontiguousarray(input_scale),
        output_scale=np.ascontiguousarray(scale),
    )


class StageWeights:
    """The terms' weights at one stage: ``eps0 gamma0 rho`` per channel and per group.

    It holds the stage's own copies of the input and output multipliers, whose rows a
    symmetric mixture rebuilds from the channels' densities (:func:`mixture_rows`), and
    updates only when the densities change: a fill of constant composition builds its
    filter once, and every stage of a run of varying composition its own.

    Parameters
    ----------
    leaf : solprop.translate.responses.DispersiveKerrTerms
        The compiled leaf, for its static coefficients.
    terms : DispersiveTerms
        The layout.
    """

    def __init__(self, leaf: DispersiveKerrTerms, terms: DispersiveTerms) -> None:
        self.coefficients = np.array(leaf.coefficients, dtype=np.float64)
        self.terms = terms
        n_channels = leaf.n_channels
        # After the channels, the unit factor for the products that carry their own,
        # then each mixture's coefficient.
        self.factors = np.ones(n_channels + 1 + len(terms.mixtures))
        self.pairs = np.zeros((len(terms.groups), 2))
        self.multipliers = np.zeros((terms.n_products, terms.term_multipliers.shape[1]))
        self.inputs = terms.input_multipliers.copy()
        self.outputs = terms.term_multipliers.copy()
        self._members = tuple(
            np.array(terms.groups[group], dtype=np.int64) for group in terms.mixtures
        )
        self._densities: tuple[float, ...] | None = None

    @property
    def nbytes(self) -> int:
        """Return the bytes the weights hold.

        Returns
        -------
        int
            The factors, group weights, stage multipliers and the layout's arrays.
        """
        terms = self.terms
        return int(
            self.coefficients.nbytes
            + self.factors.nbytes
            + self.pairs.nbytes
            + self.multipliers.nbytes
            + self.inputs.nbytes
            + self.outputs.nbytes
            + terms.input_multipliers.nbytes
            + terms.term_rows.nbytes
            + terms.term_channels.nbytes
            + terms.term_multipliers.nbytes
            + terms.pair_constants.nbytes
            + terms.quartics.nbytes
            + terms.input_scale.nbytes
            + terms.output_scale.nbytes
        )

    def update(self, densities: tuple[float, ...]) -> None:
        """Weigh the channels by their densities at a stage, allocating nothing.

        A no-op when the densities are those of the last update: everything here is a
        function of them alone.

        Parameters
        ----------
        densities : tuple of float
            Each channel's number density, m^-3, the stage bundle's
            ``dispersive_kerr_densities``.
        """
        if densities == self._densities:
            return
        self._densities = densities
        factors, coefficients = self.factors, self.coefficients
        for channel, density in enumerate(densities):
            factors[channel] = coefficients[channel] * density
        terms = self.terms
        n_channels = len(densities)
        for index, group in enumerate(terms.mixtures):
            factors[n_channels + 1 + index] = mixture_rows(
                terms.quartics,
                self._members[index],
                factors,
                terms.input_scale,
                terms.output_scale,
                self.inputs[group],
                self.outputs[group],
            )
        if not terms.symmetric:
            constants = terms.pair_constants
            for index, group in enumerate(terms.groups):
                c0 = c1 = 0.0
                for channel in group:
                    c0 += factors[channel] * constants[channel, 0]
                    c1 += factors[channel] * constants[channel, 1]
                self.pairs[index, 0] = c0
                self.pairs[index, 1] = c1
        weigh_terms(
            terms.term_rows,
            terms.term_channels,
            self.outputs,
            factors,
            self.multipliers,
        )


# --- the workspace ----------------------------------------------------------------


class DispersiveLayout(NamedTuple):
    """How the backend runs the fused node path, as the response needs it.

    Attributes
    ----------
    n_modes : int
        Modal rows of the transform.
    n_pol_state : int
        Polarisation components the state carries.
    chunk_rows : int
        Rows of the node buffers: chunk nodes times components per node.
    chunks : int
        Node chunks per stage; more than one needs a scratch to accumulate into.
    alone : bool
        Whether the dispersive response is the only one, so that the node field buffer,
        the projected buffer and the polarisation spectrum are the response's.
    embedded : bool
        Whether the quadrature has an embedded rule, projected for the alarm.
    real_synthesis : bool
        Whether the backend's scaled synthesis is real; otherwise the response holds a
        real copy for its real GEMMs.
    bare_time_free : bool
        Whether nothing reads the bare modal field once the nodes are synthesised --
        there is no in-stage absorber -- so that a lone symmetric response of one
        species may filter the stage's own spectrum in place.
    parallel : bool
        Whether the node loops are ``prange``.
    """

    n_modes: int
    n_pol_state: int
    chunk_rows: int
    chunks: int
    alone: bool
    embedded: bool
    real_synthesis: bool
    bare_time_free: bool
    parallel: bool


class DispersiveMatrices(NamedTuple):
    """The backend's taper-scaled transform matrices, rescaled in place every stage.

    Attributes
    ----------
    synthesis : numpy.ndarray
        The scaled synthesis, ``(n_transform_rows, n_modes)``; real, or a complex cast
        of a real matrix that the response replaces with a real copy of its own.
    projection : numpy.ndarray
        The scaled projection, real, ``(n_modes, n_transform_rows)``.
    embedded : numpy.ndarray or None
        The embedded rule's scaled projection, or ``None``.
    """

    synthesis: np.ndarray
    projection: np.ndarray
    embedded: np.ndarray | None


class _Chunk(NamedTuple):
    """One node chunk's views, made once: the stage only fills them."""

    first: bool
    synthesis: np.ndarray
    projections: tuple[np.ndarray, ...]
    bare: np.ndarray
    slots: tuple[np.ndarray, ...]
    flat: tuple[np.ndarray, ...]
    offset: np.ndarray
    offset_flat: np.ndarray


@dataclass
class _Rule:
    """One projection rule's products, their transform and the spectrum they land in.

    ``forward`` is ``None`` when the products are the stage's own projected buffer,
    which the stage's forward transform takes into ``spectra``, its own spectrum.
    """

    products: np.ndarray
    forward: Any
    spectra: np.ndarray
    target: np.ndarray


class DispersiveWorkspace:
    """Preallocate the response's buffers and plans, and run it on the fused path.

    The stage drives it in five calls: :meth:`prepare` weighs the channels at the stage
    and filters the padded modal spectrum, :meth:`begin` scales the synthesis,
    :meth:`respond_chunk` forms and
    projects the products of each node chunk, :meth:`finish` applies the collar to them
    and transforms them, and after the stage's own forward transform -- skipped when
    :attr:`replaces_transform` -- :meth:`combine` adds their weighted spectra to the
    polarisation spectrum. The embedded rule's products are finished and combined when
    a step is accepted (:meth:`finish_coarse`, :meth:`combine_coarse`).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem, carrying ``responses.dispersive_kerr`` on a legacy transform.
    plans : solprop.backends.numba_cpu.fft.TransformPlans
        The backend's plans, whose projected buffers and spectra the response adds into
        and, when it is alone, uses as its own.
    nodes : numpy.ndarray
        The backend's node field buffer, ``(chunk_rows, n_to)``: read as the bare
        field, and, when the response is alone and symmetric, used as its own.
    scratch : numpy.ndarray
        The backend's chunk projection scratch, ``(n_columns, n_to)``, shared between
        sequential uses when its dtype is the products'.
    matrices : DispersiveMatrices
        The backend's scaled transform matrices, whose views the response holds.
    layout : DispersiveLayout
        How the backend runs the path.
    flag : str
        The FFTW planning flag, a key of
        :data:`solprop.backends.numba_cpu.fft.PLAN_FLAGS`.
    fftw_threads : int or None
        The FFTW pool override; ``None`` sizes each batched plan by the backend's rule.
    """

    def __init__(  # noqa: PLR0913 - the backend's buffers and its configuration
        self,
        problem: KernelProblem,
        plans: fft.TransformPlans,
        nodes: np.ndarray,
        scratch: np.ndarray,
        matrices: DispersiveMatrices,
        layout: DispersiveLayout,
        *,
        flag: str,
        fftw_threads: int | None,
    ) -> None:
        leaf = problem.responses.dispersive_kerr
        assert leaf is not None
        self.leaf = leaf
        self.layout = layout
        kind = problem.grid_kind
        self.form = dispersive_form(kind, leaf.sector)
        self.symmetric = leaf.evaluator == "symmetric"
        n_to = problem.grids.n_to
        n_columns = layout.n_modes * layout.n_pol_state
        self.n_columns = n_columns
        analytic = self.form == "conserving" and kind == "field"
        complex_time = self.form != "real"
        time_dtype = np.complex128 if complex_time else np.float64
        # The products' spectrum is one-sided only for real products.
        n_product_bins = n_to if complex_time else leaf.positive.shape[0]
        self.terms = dispersive_terms_layout(leaf, kind, analytic=analytic)
        self.n_fields = self.terms.n_fields
        self.n_products = self.terms.n_products
        self.stage = StageWeights(leaf, self.terms)
        self.window = problem.grids.t_window_fine
        self.carrier = (
            np.ascontiguousarray(problem.grids.carrier_2w0)
            if leaf.carrier and problem.grids.carrier_2w0 is not None
            # Read only by the harmonic form; any array otherwise.
            else np.ones(1, dtype=np.complex128)
        )
        # One product: a symmetric response of one species, or of a mixture under its
        # mixture filter.
        lone = layout.alone and self.symmetric and self.n_products == 1
        self.in_place = lone and layout.bare_time_free
        # A lone symmetric response whose products have the polarisation's dtype
        # takes the stage's projected buffer for its product: the stage's forward
        # transform is the product's, and the output multiplier finishes it in the
        # stage's spectrum.
        self.borrowed = lone and not analytic
        self.synthesis = (
            None
            if layout.real_synthesis
            else np.empty(problem.transform.synthesis.shape, dtype=np.float64)
        )
        self.transform_synthesis = problem.transform.synthesis
        self._coarse_pending = False
        self._owned: dict[str, int] = dict.fromkeys(DISPERSIVE_LEDGER_ROWS, 0)

        threads = fft.choose_pools(
            0,
            0,
            max(self.n_fields, self.n_products) * n_columns * n_to,
            fftw_threads=fftw_threads,
        ).fftw
        heap_before = fft._heap_bytes()
        started = time.perf_counter()
        self._plan(
            plans,
            time_dtype,
            n_product_bins,
            flag=fft.PLAN_FLAGS[flag],
            threads=threads,
        )
        self.planning_seconds = time.perf_counter() - started
        self._owned["dispersive_fft_plan_heap_growth"] = max(
            0, fft._heap_bytes() - heap_before
        )
        rows = layout.chunk_rows
        if layout.alone and self.symmetric:
            self.node_fields = nodes
        else:
            self.node_fields = self._held(
                "node_fields", (rows if self.symmetric else 2 * rows, n_to), time_dtype
            )
        self.scratch = (
            np.empty((0, 0), dtype=time_dtype)
            if layout.chunks == 1
            else scratch
            if scratch.dtype == time_dtype
            else self._held("chunk_scratch", (n_columns, n_to), time_dtype)
        )
        self._kernels(problem, analytic)
        self._owned["dispersive_synthesis"] = (
            0 if self.synthesis is None else self.synthesis.nbytes
        )
        self._views(problem, nodes, matrices)
        self._owned["dispersive_multipliers"] = (
            self.stage.nbytes
            + self.carrier.nbytes
            + self.dc.nbytes
            + self.offset.nbytes
        )

    @property
    def replaces_transform(self) -> bool:
        """Return whether the response writes the stage's polarisation spectrum itself.

        Alone with products of its own, nothing else is projected, so the stage's
        forward transform of its projected buffer would transform nothing; the
        response's :meth:`combine` writes the spectrum instead.

        Returns
        -------
        bool
            True when the stage skips its forward transform.
        """
        return self.layout.alone and not self.borrowed

    # --- construction ---------------------------------------------------------

    def _kernels(self, problem: KernelProblem, analytic: bool) -> None:
        """Compile the product kernels and hold the bare field's zero-frequency offset.

        Parameters
        ----------
        problem : solprop.translate.problem.KernelProblem
            The problem, for the components per node.
        analytic : bool
            Whether the products are a field grid's analytic signals.
        """
        layout, n_columns, rows = self.layout, self.n_columns, self.layout.chunk_rows
        n_pol = problem.transform.n_pol * layout.n_pol_state
        # The polynomial reads the bare analytic signal the backend synthesised, which
        # keeps the zero-frequency bin once; Pi_+ excludes it, so its constant is
        # subtracted at the nodes (the state carries some, from the split-step
        # absorber's product in time).
        self.shifted = analytic and not self.symmetric
        self.dc = np.zeros((n_columns if self.shifted else 0,), dtype=np.complex128)
        self.offset = np.zeros((rows if self.shifted else 1,), dtype=np.complex128)
        self.self_product = make_dispersive_self(
            self.form, n_pol, parallel=layout.parallel, shifted=self.shifted
        )
        self.pairs = (
            None
            if self.symmetric
            else make_dispersive_pairs(
                self.form, n_pol, parallel=layout.parallel, shifted=self.shifted
            )
        )

    def _plan(
        self,
        plans: fft.TransformPlans,
        time_dtype: type,
        n_product_bins: int,
        *,
        flag: str,
        threads: int,
    ) -> None:
        """Allocate the transformed buffers and plan every transform on them.

        Parameters
        ----------
        plans : solprop.backends.numba_cpu.fft.TransformPlans
            The backend's plans.
        time_dtype : type
            The dtype of the filtered fields and of the products.
        n_product_bins : int
            Bins of a product's spectrum: one-sided for real products only.
        flag : str
            The FFTW flag name.
        threads : int
            The FFTW pool.
        """
        n_columns, n_to = self.n_columns, self.window.shape[0]
        if self.borrowed:
            products, spectra = plans.projected, plans.projected_spectrum
            forward = None
        else:
            products = self._aligned(
                "products", (self.n_products * n_columns, n_to), time_dtype
            )
            spectra = self._aligned(
                "product_spectra",
                (self.n_products * n_columns, n_product_bins),
                np.complex128,
            )
            forward = fft._plan(
                products, spectra, forward=True, flag=flag, threads=threads
            )
        if self.in_place:
            # The stage's own inverse transform gives the filtered field.
            self.filtered_spectrum = plans.modal_spectrum
            self.filtered_time = plans.modal_time
            self.to_time = None
        else:
            # The filtered spectra live in the products' spectrum, which is free until
            # the products are transformed; the rows have the same length on every
            # layout.
            self.filtered_spectrum = spectra[: self.n_fields * n_columns]
            self.filtered_time = self._aligned(
                "filtered_time", (self.n_fields * n_columns, n_to), time_dtype
            )
            self.to_time = fft._plan(
                self.filtered_spectrum,
                self.filtered_time,
                forward=False,
                flag=flag,
                threads=threads,
            )
        self.fine = _Rule(products, forward, spectra, plans.projected_spectrum)
        self.coarse: _Rule | None = None
        if not self.layout.embedded:
            return
        assert plans.coarse_projected_spectrum is not None
        if self.borrowed:
            self.coarse = _Rule(
                plans.coarse_projected, None, plans.coarse_projected_spectrum,
                plans.coarse_projected_spectrum,
            )  # fmt: skip
            return
        coarse_products = self._aligned("coarse_products", products.shape, time_dtype)
        # The fine rule's spectra are consumed by its combination before the embedded
        # rule is finished, on the accepted step, so the two rules share them.
        self.coarse = _Rule(
            coarse_products,
            fft._plan(
                coarse_products, spectra, forward=True, flag=flag, threads=threads
            ),
            spectra,
            plans.coarse_projected_spectrum,
        )

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

    def _held(self, name: str, shape: tuple[int, int], dtype: type) -> np.ndarray:
        """Allocate a plain buffer and count it under its ledger row.

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
        buffer = np.zeros(shape, dtype=dtype)
        self._owned[f"dispersive_{name}"] += buffer.nbytes
        return buffer

    # --- one stage -------------------------------------------------------------

    def prepare(
        self, spectrum: Complex[np.ndarray, "n_columns n_bins"], bundle: StageScalars
    ) -> None:
        """Weigh the channels, filter the stage's spectrum and transform the fields.

        Called after the stage combination has padded the modal spectrum and before
        the stage's own inverse transform, which destroys it. In place, the filtered
        spectrum is the stage's own, and that transform gives the filtered field.

        Parameters
        ----------
        spectrum : numpy.ndarray
            The padded modal spectrum (doubled positive bins on the analytic family).
        bundle : solprop.translate.stage.StageScalars
            The stage's scalars, whose channel densities weigh the terms and build a
            symmetric mixture's filter, which the input multipliers read.
        """
        self.stage.update(bundle.dispersive_kerr_densities)
        if self.shifted:
            np.copyto(self.dc, spectrum[:, 0])
        filter_spectra(spectrum, self.stage.inputs, self.filtered_spectrum)
        if self.to_time is not None:
            self.to_time.execute()

    def _views(
        self, problem: KernelProblem, nodes: np.ndarray, matrices: DispersiveMatrices
    ) -> None:
        """Make every view a stage reads or writes, once.

        Parameters
        ----------
        problem : solprop.translate.problem.KernelProblem
            The problem, for the node layout.
        nodes : numpy.ndarray
            The backend's node field buffer.
        matrices : DispersiveMatrices
            The backend's scaled matrices.
        """
        layout, n_columns = self.layout, self.n_columns
        n_modes = layout.n_modes
        transform = problem.transform
        per_node = transform.n_pol * layout.n_pol_state
        chunk_nodes = layout.chunk_rows // per_node
        synthesis = matrices.synthesis if self.synthesis is None else self.synthesis
        self._modal = tuple(
            _gemm(
                self.filtered_time[index * n_columns : (index + 1) * n_columns].reshape(
                    n_modes, -1
                )
            )
            for index in range(self.n_fields)
        )
        self._dc_flat = (
            _gemm(self.dc.reshape(n_modes, -1)) if self.shifted else np.empty((0, 0))
        )
        rules = (self.fine,) if self.coarse is None else (self.fine, self.coarse)
        self._targets = tuple(
            tuple(
                (
                    rule.products[block * n_columns : (block + 1) * n_columns],
                    _gemm(
                        rule.products[
                            block * n_columns : (block + 1) * n_columns
                        ].reshape(n_modes, -1)
                    ),
                )
                for block in range(self.n_products)
            )
            for rule in rules
        )
        self._scratch_flat = (
            _gemm(self.scratch.reshape(n_modes, -1))
            if self.scratch.size
            else np.empty((0, 0))
        )
        # The polynomial's cube is projected before its derivative fields are formed,
        # so two slots hold the three products in turn.
        n_slots = 1 if self.symmetric else 2
        chunks = []
        for first in range(0, transform.n_nodes, chunk_nodes):
            last = min(first + chunk_nodes, transform.n_nodes)
            rows = slice(first * transform.n_pol, last * transform.n_pol)
            width, count = rows.stop - rows.start, (last - first) * per_node
            slots = tuple(
                self.node_fields[
                    slot * layout.chunk_rows : slot * layout.chunk_rows + count
                ]
                for slot in range(n_slots)
            )
            offset = self.offset[:count] if self.shifted else self.offset
            projections = (matrices.projection[:, rows],)
            if self.coarse is not None and matrices.embedded is not None:
                projections = (*projections, matrices.embedded[:, rows])
            chunks.append(
                _Chunk(
                    first=first == 0,
                    synthesis=synthesis[rows],
                    projections=projections,
                    bare=nodes[:count],
                    slots=slots,
                    flat=tuple(_gemm(slot.reshape(width, -1)) for slot in slots),
                    offset=offset,
                    offset_flat=_gemm(offset.reshape(width, -1))
                    if self.shifted
                    else offset,
                )
            )
        self._chunks = tuple(chunks)

    def begin(self, bundle: StageScalars, taper: float) -> None:
        """Weigh the channels at the stage and scale the real copy of the synthesis.

        :meth:`prepare` has weighed them already on the stage path, and the weights
        update only when the densities change.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's scalars, whose channel densities weight the terms and, for the
            polynomial, its groups' derivative products.
        taper : float
            The stage's taper scale.
        """
        self.stage.update(bundle.dispersive_kerr_densities)
        if self.synthesis is not None:
            np.multiply(self.transform_synthesis, taper, out=self.synthesis)

    def respond_chunk(self, index: int) -> None:
        """Synthesise the filtered fields, form the products and project them.

        Parameters
        ----------
        index : int
            The node chunk, in the backend's order; the bare field of a polynomial
            response is the backend's node buffer, already synthesised for it.
        """
        chunk = self._chunks[index]
        carrier = self.carrier
        if self.symmetric:
            slot, flat = chunk.slots[0], chunk.flat[0]
            for channel in range(self.n_fields):
                np.dot(chunk.synthesis, self._modal[channel], out=flat)
                self.self_product(slot, slot, carrier, chunk.offset)
                self._project(chunk, flat, channel)
            return
        assert self.pairs is not None
        derivative, second = chunk.slots
        if self.shifted:
            np.dot(chunk.synthesis, self._dc_flat, out=chunk.offset_flat)
        self.self_product(chunk.bare, derivative, carrier, chunk.offset)
        self._project(chunk, chunk.flat[0], 0)
        pairs = self.stage.pairs
        for group in range(len(self.terms.groups)):
            np.dot(chunk.synthesis, self._modal[2 * group], out=chunk.flat[0])
            np.dot(chunk.synthesis, self._modal[2 * group + 1], out=chunk.flat[1])
            self.pairs(
                chunk.bare,
                derivative,
                second,
                pairs[group, 0],
                pairs[group, 1],
                carrier,
                chunk.offset,
            )
            self._project(chunk, chunk.flat[0], 1 + 2 * group)
            self._project(chunk, chunk.flat[1], 2 + 2 * group)

    def _project(self, chunk: _Chunk, flat: np.ndarray, product: int) -> None:
        """Project one chunk's product onto the modal columns of both rules.

        Parameters
        ----------
        chunk : _Chunk
            The chunk's views.
        flat : numpy.ndarray
            The product at the chunk's nodes, as a GEMM operand.
        product : int
            The product's block.
        """
        for targets, matrix in zip(self._targets, chunk.projections, strict=True):
            block, target = targets[product]
            if chunk.first:
                np.dot(matrix, flat, out=target)
                continue
            np.dot(matrix, flat, out=self._scratch_flat)
            np.add(block, self.scratch, out=block)

    def finish(self) -> None:
        """Apply the collar to the projected products and transform them.

        Before the stage's forward transform: a borrowed product is the stage's
        projected buffer, which that transform takes.
        """
        self._finish(self.fine)
        self._coarse_pending = self.coarse is not None

    def combine(self) -> None:
        """Land the products' weighted spectra in the stage's polarisation spectrum.

        After the stage's forward transform (or instead of it, when
        :attr:`replaces_transform`).
        """
        self._combine(self.fine)

    def finish_coarse(self, bundle: StageScalars) -> None:
        """Apply the collar to the embedded rule's products and transform them.

        Called when a step is accepted, before the stage's transform of the embedded
        projection; the products are the last stage's, as the projection is, and so
        are the scalars that weigh them. Once per evaluation: the products hold the
        collar until another stage projects them again.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The last stage's scalars.
        """
        if self.coarse is not None and self._coarse_pending:
            self.stage.update(bundle.dispersive_kerr_densities)
            self._finish(self.coarse)
            self._coarse_pending = False

    def combine_coarse(self) -> None:
        """Land the embedded rule's weighted spectra in its polarisation spectrum."""
        if self.coarse is not None:
            self._combine(self.coarse)

    def _finish(self, rule: _Rule) -> None:
        """Window one rule's products and transform them unless the stage will.

        Parameters
        ----------
        rule : _Rule
            The rule.
        """
        apply_window(rule.products, self.window)
        if rule.forward is not None:
            rule.forward.execute()

    def _combine(self, rule: _Rule) -> None:
        """Weigh and land one rule's spectra.

        Parameters
        ----------
        rule : _Rule
            The rule.
        """
        weights = self.stage.multipliers
        if rule.forward is None:
            weigh_spectrum(weights, rule.target)
        elif self.layout.alone:
            combine_spectra(rule.spectra, weights, rule.target)
        else:
            add_spectra(rule.spectra, weights, rule.target)

    def memory_ledger(self) -> dict[str, int]:
        """Return the bytes of every buffer the response owns, by name.

        Buffers it borrows from the backend -- the node field buffer, the chunk
        scratch, the projected buffers and their spectra -- are the backend's rows.

        Returns
        -------
        dict
            Every row of :data:`DISPERSIVE_LEDGER_ROWS`.
        """
        return dict(self._owned)
