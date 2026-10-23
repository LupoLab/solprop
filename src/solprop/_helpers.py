# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The evaluations a backend supplies to solprop's shared per-stage and per-step code.

Two small scalar computations run on every backend's path: a scalar table's Chebyshev
rows at a step's positions (behind every stage bundle and anchor, and an expansion's
rate rows; :meth:`solprop.translate.chebyshev.PiecewiseChebyshev.at_positions`) and
the Lagrange basis of an Adams formula on its actual nodes
(:func:`solprop.stepper.multistep.integrated_weights` and
:func:`~solprop.stepper.multistep.lagrange_coefficients`). The code around them is
shared, but **which implementation evaluates them belongs to the backend that runs the
stage** (SEP 29 section 6.6, R16), never to a process-wide switch or to what happens
to be installed: the NumPy reference must not run compiled code it exists to check,
and one simulation must not change another's arithmetic.

A backend names its record in ``StepDriver.helpers`` and the shared driver, march and
engines pass it down: ``numpy_ref`` takes :data:`PLAIN`, the uncompiled Python forms;
``numba_cpu`` takes :data:`NUMBA`, the same functions compiled; the native backend
takes its own record of C++ bindings, defined in its package. All three perform the
same IEEE operations in the same order, strict (no contraction, no reassociation;
SEP 25 section 4.2), so they agree to the last bit wherever their arithmetic is IEEE.

Translation is not a backend's: it runs before any backend exists and builds the
problem every backend receives, so it keeps the shared reference compiled by Numba
(:func:`solprop.translate.chebyshev.clenshaw`, and a table called without a record),
and the shared functions' defaults are :data:`NUMBA` for callers outside a backend. A
z table's state-sized evaluation is not a stage scalar and stays on that shared pass.

Like ``_display`` and ``_numba_cache``, this module imports nothing from the rest of
the package at import time, so any layer may use it; each record imports its
functions from where they live at the first evaluation.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = ["NUMBA", "PLAIN", "BasisFunction", "Helpers", "PointsFunction"]

PointsFunction = Callable[
    [np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray], int
]
"""``points(z, edges, lower, upper, c, out) -> int``: a table's rows at positions.

The contract of :func:`solprop.translate.chebyshev._clenshaw_points`: ``z`` float64
``(n_points,)``, ``edges`` float64 ``(n_intervals + 1,)``, ``c`` float64 or complex128
``(n_intervals, n_deg, n_values)`` and ``out`` of ``c``'s dtype ``(n_points,
n_values)``, all C-contiguous; the result is the first position outside ``[lower,
upper]`` or ``-1``.
"""

BasisFunction = Callable[[np.ndarray, float, bool], tuple[np.ndarray, np.ndarray]]
"""``basis(points, upper, integrate) -> (rows, weights)``: an Adams formula's basis.

The contract of :func:`solprop.stepper.multistep._basis_rows`: ``points`` float64
``(n,)`` and C-contiguous; ``rows`` ``(n, n)`` (zero when integrating) and ``weights``
``(n,)`` (unset when not).
"""


@dataclass(frozen=True)
class Helpers:
    """The per-stage and per-step evaluations of one backend.

    Each operation is loaded at its first use and kept: a record names its functions
    without importing them, so constructing or importing one loads nothing (the Numba
    record compiles at its first evaluation, exactly as the cached dispatchers it
    holds always have).

    Attributes
    ----------
    name : str
        What the record is, for messages and tests.
    load_points : collections.abc.Callable
        Returns the :data:`PointsFunction`.
    load_basis : collections.abc.Callable
        Returns the :data:`BasisFunction`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop._helpers import PLAIN
    >>> rows, weights = PLAIN.basis(np.array([1.0, 0.0]), 1.0, True)
    >>> weights.tolist()
    [0.5, 0.5]
    """

    name: str
    load_points: Callable[[], PointsFunction] = field(repr=False)
    load_basis: Callable[[], BasisFunction] = field(repr=False)

    @functools.cached_property
    def points(self) -> PointsFunction:
        """Return the Chebyshev rows at several positions.

        Returns
        -------
        collections.abc.Callable
            The :data:`PointsFunction`.
        """
        return self.load_points()

    @functools.cached_property
    def basis(self) -> BasisFunction:
        """Return the Lagrange basis on an Adams formula's nodes.

        Returns
        -------
        collections.abc.Callable
            The :data:`BasisFunction`.
        """
        return self.load_basis()


def _plain_points() -> PointsFunction:
    """Return the Chebyshev rows' Python form.

    Returns
    -------
    collections.abc.Callable
        :func:`solprop.translate.chebyshev._clenshaw_points`.
    """
    from solprop.translate import chebyshev  # noqa: PLC0415 - where it lives

    return chebyshev._clenshaw_points


def _plain_basis() -> BasisFunction:
    """Return the Adams basis' Python form.

    Returns
    -------
    collections.abc.Callable
        :func:`solprop.stepper.multistep._basis_rows`.
    """
    from solprop.stepper import multistep  # noqa: PLC0415 - where it lives

    return multistep._basis_rows


def _numba_points() -> PointsFunction:
    """Return the Chebyshev rows compiled by Numba.

    Returns
    -------
    collections.abc.Callable
        The cached dispatcher of :func:`solprop.translate.chebyshev._clenshaw_batch`.
    """
    from solprop.translate import chebyshev  # noqa: PLC0415 - numba on first use

    return chebyshev._clenshaw_batch()


def _numba_basis() -> BasisFunction:
    """Return the Adams basis compiled by Numba.

    Returns
    -------
    collections.abc.Callable
        The cached dispatcher of :func:`solprop.stepper.multistep._basis`.
    """
    from solprop.stepper import multistep  # noqa: PLC0415 - numba on first use

    return multistep._basis()


PLAIN = Helpers("plain", _plain_points, _plain_basis)
"""The uncompiled Python forms: the NumPy reference backend's."""

NUMBA = Helpers("numba", _numba_points, _numba_basis)
"""The same forms compiled by Numba: ``numba_cpu``'s, and the shared default."""
