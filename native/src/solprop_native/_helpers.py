# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native backend's per-stage and per-step evaluations, for solprop's shared code.

solprop's driver, Adams march and engines take the stage tables' Chebyshev rows and
the Adams basis from the backend that runs them (``StepDriver.helpers``, solprop
SEP 29 section 6.6), so that a native run never reaches solprop's Numba dispatchers
and the NumPy reference never reaches this package. :data:`NATIVE` is this backend's
record: the C++ transcriptions in ``include/solprop_native/chebyshev.hpp`` and
``lagrange_basis.hpp``, which perform solprop's operations in its order under strict
arithmetic.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from solprop._helpers import BasisFunction, Helpers, PointsFunction

from . import _core

__all__ = ["NATIVE"]


def _basis(
    points: npt.NDArray[np.float64], upper: float, integrate: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the Lagrange basis on `points`: its coefficients, or its integrals.

    Parameters
    ----------
    points
        The distinct nodes, float64, C-contiguous, at most 32 of them.
    upper
        The integral's upper limit.
    integrate
        Whether to return the integrals rather than the coefficients.

    Returns
    -------
    tuple
        The coefficients, shape ``(n, n)`` (zero when integrating), and the integrals,
        shape ``(n,)`` (unset when not).
    """
    count = points.shape[0]
    rows = np.empty((count, count))
    weights = np.empty(count)
    _core.basis_rows(points, upper, integrate, rows, weights)
    return rows, weights


def _load_points() -> PointsFunction:
    """Return the native Chebyshev rows binding."""
    return _core.clenshaw_points


def _load_basis() -> BasisFunction:
    """Return the native Adams basis."""
    return _basis


NATIVE = Helpers("native", _load_points, _load_basis)
"""The native backend's helper record (``NativeBackend.helpers``)."""
