# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rounding budgets and comparisons for this backend's tests (solprop SEP 25).

Independent floating-point computations are compared by numerical error, not bits:
two thread counts, a batch member and its solo run, a repeated run or a staged and a
direct path differ by rounding alone, and each comparison holds them to a bound fixed
from operation and reduction counts, transform lengths and a stated margin *before*
it runs, never fitted to the discrepancy it then accepts. Exact comparison remains
right for structural contracts: lossless copies, integer accounting, identity
metadata, hard masks and explicitly exact no-ops.

The names and constants mirror solprop's ``tests/bounds.py``, which this repository's
tests cannot import. ``u`` is the unit roundoff ``2**-53`` and ``EPS = 2 u`` NumPy's
machine epsilon; :func:`gamma` and :func:`fft_bound` keep the equivalence ladder's
constants, written in ``EPS``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

EPS = float(np.finfo(np.float64).eps)
"""NumPy's machine epsilon, ``2 u``."""

UNIT_ROUNDOFF = 2.0**-53
"""The unit roundoff ``u`` of binary64 round-to-nearest arithmetic."""


def gamma(operations: int) -> float:
    """Return the ladder's ``k eps / (1 - k eps)``: twice Higham's gamma, in ``EPS``."""
    scaled = operations * EPS
    return scaled / (1.0 - scaled)


def higham_gamma(depth: int) -> float:
    """Return Higham's ``gamma_k = k u / (1 - k u)`` for an addition depth `depth`."""
    scaled = depth * UNIT_ROUNDOFF
    if scaled >= 1.0:
        raise ValueError(f"depth {depth} is beyond gamma's range")
    return scaled / (1.0 - scaled)


def fft_bound(samples: int) -> float:
    """Bound one FFT round trip and its associated pointwise operations."""
    return 512 * EPS * math.ceil(math.log2(samples))


def step_bound(samples: int, steps: int, *, stages: int = 7) -> float:
    """Return a relative rounding budget of `steps` fixed steps of an explicit pair.

    Every stage evaluates the nonlinear term through one transform round trip
    (:func:`fft_bound`, which carries its pointwise work and the stage combination),
    the seven stages of DOPRI5 by default; the linear flow is unitary and the cases
    using this budget have nonlinear phases far below one radian over their windows,
    so nothing amplifies the perturbations and the steps' budgets add.
    """
    return steps * stages * fft_bound(samples)


def departure(
    actual: npt.ArrayLike, expected: npt.ArrayLike, *, scale: float | None = None
) -> float:
    """Return ``||actual - expected|| / scale``, scale ``||expected||`` by default."""
    left = np.asarray(actual)
    right = np.asarray(expected)
    denominator = float(np.linalg.norm(right)) if scale is None else scale
    return float(np.linalg.norm(left - right) / denominator)


def assert_within(
    actual: npt.ArrayLike,
    expected: npt.ArrayLike,
    bound: float,
    *,
    scale: float | None = None,
    what: str = "",
) -> float:
    """Assert two results agree inside a bound fixed beforehand; return the margin.

    Both arrays must be finite and the reference nontrivial (a zero reference makes a
    relative comparison vacuous: test zero results structurally instead). The
    comparison is normwise relative to ``||expected||``, or to `scale` when given (an
    absolute bound passes ``scale=1``).

    Returns
    -------
    float
        The observed departure as a fraction of the bound.
    """
    left = np.asarray(actual)
    right = np.asarray(expected)
    label = f"{what}: " if what else ""
    if left.shape != right.shape:
        raise AssertionError(f"{label}shapes differ, {left.shape} != {right.shape}")
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        raise AssertionError(f"{label}non-finite values")
    if scale is None and not np.linalg.norm(right):
        raise AssertionError(f"{label}the reference is identically zero")
    if bound <= 0:
        raise AssertionError(f"{label}a bound must be positive, got {bound}")
    found = departure(left, right, scale=scale)
    if found > bound:
        raise AssertionError(
            f"{label}departure {found:.3e} exceeds the bound {bound:.3e}"
        )
    return found / bound


def assert_work_within(
    actual: int, expected: int, *, factor: float, what: str = ""
) -> None:
    """Assert an adaptive work count is within a band of a baseline (SEP 25 4.4).

    Independent adaptive runs may take different steps and rejections; their counts
    are performance diagnostics held to a declared band, never to equality.
    """
    label = f"{what}: " if what else ""
    if not expected / factor <= actual <= expected * factor:
        raise AssertionError(
            f"{label}{actual} outside [{expected / factor:g}, {expected * factor:g}]"
        )
