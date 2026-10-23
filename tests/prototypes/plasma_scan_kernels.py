# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prototype deterministic temporal prefixes with a fixed segment partition (rejected).

An experiment kept beside its measurements; no backend of the package uses it.

The prototype predeclares the actual addition-depth and serial-reference error terms.
These kernels are experimental; the production response column does not use them.
Every work array belongs to the caller, and no compiled pass allocates storage.
"""

import math
from typing import Any, NamedTuple, no_type_check

import numpy as np
from jaxtyping import Float
from numba import prange

from solprop._numba_cache import njit
from solprop.exceptions import BackendError

__all__ = ["ScanKernels", "make_scan_kernels"]


class ScanKernels(NamedTuple):
    """Keep the three prefix passes and their fixed geometry together.

    Attributes
    ----------
    local, carries, apply : callable
        Compiled segment, ordered carry and carry-application passes. The local pass
        takes increments, prefix work and segment totals; the remaining passes take
        totals alone and prefix work with totals, respectively.
    segments : int
        Segment count per row, independent of the number of workers.
    depth : int
        Conservative maximum addition depth of a completed prefix.
    """

    # Numba dispatchers have no callable stubs; confine their dynamic types here.
    local: Any
    carries: Any
    apply: Any
    segments: int
    depth: int


def make_scan_kernels(
    n_to: int, *, segments: int = 16, parallel: bool = False
) -> ScanKernels:
    """Compile a fixed balanced partition and its ordered inter-segment carry.

    Parameters
    ----------
    n_to : int
        Positive power-of-two time extent of each input and prefix-work row.
    segments : int, optional
        Positive power-of-two partition count, capped at ``n_to``. The caller
        preallocates float64 totals shaped ``(n_rows, min(segments, n_to))`` and
        prefix work shaped ``(n_rows, n_to)``. Inputs must have the same row shape.
    parallel : bool, optional
        Whether independent segments and rows use the current Numba worker pool.
        The summation tree and segment count remain fixed under worker changes.

    Returns
    -------
    ScanKernels
        Three allocation-free passes, actual segment count and addition depth.

    Raises
    ------
    BackendError
        If either extent is not a positive power of two.
    """
    if n_to < 1 or n_to & (n_to - 1):
        msg = "n_to must be a positive power of two for the blocked scan"
        raise BackendError(msg)
    if segments < 1 or segments & (segments - 1):
        msg = "segments must be a positive power of two for the blocked scan"
        raise BackendError(msg)
    count = min(segments, n_to)
    width = n_to // count
    # The callable distinguishes serial/parallel closure cache identities too.
    loop = prange if parallel else range
    # A parallel closure is never written to numba's disk cache.
    # Numba names a parfor's gufunc from the address of its IR node and a per-process
    # counter, so two variants of one closure compiled in different processes can carry
    # the same symbol; loaded from the cache into one process, the second variant's
    # call binds to the first's gufunc and runs the other variant's constants (a
    # 300-variant probe found 14 bound wrong). Serial closures have no gufunc.
    cached = not parallel

    @njit(cache=cached, nogil=True, parallel=parallel)
    @no_type_check
    def local(
        increments: Float[np.ndarray, "n_rows n_to"],
        prefix: Float[np.ndarray, "n_rows n_to"],
        totals: Float[np.ndarray, "n_rows n_segments"],
    ) -> None:
        """Write each segment's inclusive prefix and balanced total.

        Parameters
        ----------
        increments, prefix : numpy.ndarray
            Input increments and caller-owned output, shaped (n_rows, n_to).
        totals : numpy.ndarray
            Caller-owned totals, shaped (n_rows, n_segments).
        """
        for task in loop(increments.shape[0] * count):
            row, segment = task // count, task % count
            start = segment * width
            for k in range(width):
                prefix[row, start + k] = increments[row, start + k]
            stride = 2
            while stride <= width:
                for end in range(stride - 1, width, stride):
                    prefix[row, start + end] += prefix[row, start + end - stride // 2]
                stride *= 2
            totals[row, segment] = prefix[row, start + width - 1]
            prefix[row, start + width - 1] = 0.0
            stride = width
            while stride > 1:
                for end in range(stride - 1, width, stride):
                    left = start + end - stride // 2
                    right = start + end
                    subtotal = prefix[row, left]
                    prefix[row, left] = prefix[row, right]
                    prefix[row, right] += subtotal
                stride //= 2
            for k in range(width):
                prefix[row, start + k] += increments[row, start + k]

    @njit(cache=cached, nogil=True, parallel=parallel)
    @no_type_check
    def carries(totals: Float[np.ndarray, "n_rows n_segments"]) -> None:
        """Replace segment totals with their exclusive ordered carries.

        Parameters
        ----------
        totals : numpy.ndarray
            Segment totals on entry and preceding-segment carries on return.
        """
        for row in loop(totals.shape[0]):
            running = 0.0
            for segment in range(count):
                subtotal = totals[row, segment]
                totals[row, segment] = running
                running += subtotal

    @njit(cache=cached, nogil=True, parallel=parallel)
    @no_type_check
    def apply(
        prefix: Float[np.ndarray, "n_rows n_to"],
        totals: Float[np.ndarray, "n_rows n_segments"],
    ) -> None:
        """Apply the fixed carries to each segment's local prefixes.

        Parameters
        ----------
        prefix : numpy.ndarray
            Local inclusive prefixes, overwritten with complete row prefixes.
        totals : numpy.ndarray
            Preceding-segment carries, shaped (n_rows, n_segments).
        """
        for task in loop(prefix.shape[0] * count):
            row, segment = task // count, task % count
            carry = totals[row, segment]
            start = segment * width
            for k in range(width):
                prefix[row, start + k] += carry

    depth = 2 * math.ceil(math.log2(width)) + count + 3
    return ScanKernels(local, carries, apply, count, depth)
