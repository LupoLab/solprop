# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Evaluate the reference z-table representation into persistent host stage arrays."""

import numpy as np
from jaxtyping import Complex

from solprop.translate.chebyshev import PiecewiseChebyshev

__all__ = ["HostTable"]


class HostTable:
    """Hold a Chebyshev table's Clenshaw work and stage output on the host.

    The operation order follows NumPy's ``chebval``. Only this adapter handles the
    general z table; the existing constant and expansion stage routes are unchanged.

    Parameters
    ----------
    table : PiecewiseChebyshev
        Array-valued complex table sampled in translation.

    Attributes
    ----------
    table : PiecewiseChebyshev
        The immutable translated coefficients.
    work : numpy.ndarray
        Three Clenshaw arrays, shape ``(3, *value_shape)``, complex128.
    value : numpy.ndarray
        Evaluation result, shape value_shape, complex128.
    left : numpy.ndarray
        Saved left endpoint for a difference, shape value_shape, complex128.
    """

    table: PiecewiseChebyshev
    work: np.ndarray
    value: np.ndarray
    left: np.ndarray

    def __init__(self, table: PiecewiseChebyshev) -> None:
        self.table = table
        self.work = np.empty((3, *table.value_shape), dtype=complex)
        self.value = np.empty(table.value_shape, dtype=complex)
        self.left = np.empty_like(self.value)

    def at(self, z: float) -> Complex[np.ndarray, " *value"]:
        """Evaluate the table at one position into its reusable result array.

        Parameters
        ----------
        z : float
            Position in m, within the sampled span.

        Returns
        -------
        numpy.ndarray
            The result, overwritten by the next call.
        """
        index = self.table.interval_containing(z)
        lo, hi = self.table.breakpoints[index : index + 2]
        x = float((2.0 * z - lo - hi) / (hi - lo))
        coefficients = self.table.coefficients[index]
        if len(coefficients) == 1:
            np.copyto(self.value, coefficients[0])
            return self.value
        first, second, temporary = self.work
        np.copyto(first, coefficients[-2])
        np.copyto(second, coefficients[-1])
        for offset in range(3, len(coefficients) + 1):
            np.subtract(coefficients[-offset], second, out=temporary)
            np.multiply(second, 2.0 * x, out=second)
            np.add(first, second, out=second)
            first, temporary = temporary, first
        np.multiply(second, x, out=self.value)
        np.add(first, self.value, out=self.value)
        return self.value

    def delta(self, z0: float, z1: float) -> Complex[np.ndarray, " *value"]:
        """Subtract two endpoint values into the reusable result array.

        Parameters
        ----------
        z0, z1 : float
            Endpoint positions in m, within the sampled span.

        Returns
        -------
        numpy.ndarray
            Table(z1) - table(z0), overwritten by the next call.
        """
        np.copyto(self.left, self.at(z0))
        self.at(z1)
        np.subtract(self.value, self.left, out=self.value)
        return self.value

    @property
    def nbytes(self) -> int:
        """Return the bytes of owned work arrays.

        Returns
        -------
        int
            Resident scratch and results in bytes, excluding immutable coefficients.
        """
        return self.work.nbytes + self.value.nbytes + self.left.nbytes
