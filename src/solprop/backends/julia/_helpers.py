# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Julia backend's per-stage and per-step evaluations (SEP 29 section 6.6).

:data:`JULIA` is the backend's record for solprop's shared code: the Chebyshev rows of
the stage tables and the Adams basis, by the engine's transcriptions in
``julia/src/helpers.jl``, which perform the Python forms' operations in their order,
so the values are numpy_ref's bit for bit. A call costs about 17 us against the
Python pass's 100 us.
"""

import numpy as np

from solprop._helpers import BasisFunction, Helpers, PointsFunction
from solprop.backends.julia._runtime import runtime

__all__ = ["JULIA"]


def _load_points() -> PointsFunction:
    """Return the engine's Chebyshev rows.

    Returns
    -------
    collections.abc.Callable
        The :data:`solprop._helpers.PointsFunction` of ``clenshaw_points!``.
    """
    function = runtime().clenshaw_points_b

    def points(
        z: np.ndarray,
        edges: np.ndarray,
        lower: float,
        upper: float,
        c: np.ndarray,
        out: np.ndarray,
    ) -> int:
        """Write a table's rows at the positions into `out`; see ``PointsFunction``.

        Parameters
        ----------
        z, edges : numpy.ndarray
            The positions and the breakpoints, float64.
        lower, upper : float
            The span with its slack.
        c, out : numpy.ndarray
            The coefficients and the values, C-contiguous.

        Returns
        -------
        int
            The first position outside the span, or -1.
        """
        return int(function(z, edges, lower, upper, c, out))

    return points


def _load_basis() -> BasisFunction:
    """Return the engine's Adams basis.

    Returns
    -------
    collections.abc.Callable
        The :data:`solprop._helpers.BasisFunction` of ``basis_rows!``.
    """
    function = runtime().basis_rows_b

    def basis(
        points: np.ndarray, upper: float, integrate: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the basis' coefficients or integrals; see ``BasisFunction``.

        Parameters
        ----------
        points : numpy.ndarray
            The distinct nodes, float64.
        upper : float
            The integral's upper limit.
        integrate : bool
            Whether to return the integrals.

        Returns
        -------
        tuple
            The coefficients ``(n, n)`` and the integrals ``(n,)``.
        """
        n = points.shape[0]
        rows, weights = np.zeros((n, n)), np.empty(n)
        function(points, float(upper), bool(integrate), rows, weights)
        return rows, weights

    return basis


JULIA = Helpers("julia", _load_points, _load_basis)
"""The Julia engine's record; a trial's tables are formed in the engine itself."""
