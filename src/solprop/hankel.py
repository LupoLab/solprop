# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transform an axisymmetric field with the raw order-zero QDHT.

The forward transform is ``H(k) = integral r*f(r)*J0(k*r) dr``; its inverse
integrates ``k*H(k)*J0(k*r)``. Angular frequencies are in rad/m and no ``2*pi``
is hidden in either transform. The integration weights include the azimuthal
integral. See ``docs/guide/free-space/hankel.md`` for the scalings and finite-transform
error.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float, Inexact
from numpy.typing import ArrayLike
from scipy.special import jn_zeros, jv

from solprop.exceptions import GridError

__all__ = ["QDHT"]


@dataclass(frozen=True, eq=False)
class QDHT:
    """Construct the raw order-zero quasi-discrete Hankel transform.

    The matrix is symmetric and approximately self-inverse. Its finite truncation
    error is distinct from roundoff; it is never replaced by an orthogonal matrix.

    Attributes
    ----------
    radius : float
        Outer Dirichlet boundary in m, finite and positive.
    n_r : int
        Number of interior radial samples, at least 2.
    r : numpy.ndarray
        Interior radial samples in m, shape ``(n_r,)``; zero is not a sample.
    k_r : numpy.ndarray
        Transverse angular frequencies in rad/m, shape ``(n_r,)``.
    zeros : numpy.ndarray
        First ``n_r + 1`` positive zeros of J0, dimensionless.
    j1 : numpy.ndarray
        Absolute J1 values at the first ``n_r`` zeros, dimensionless.
    matrix : numpy.ndarray
        Symmetric, dimensionless QDHT matrix, shape ``(n_r, n_r)``.
    radial_weights : numpy.ndarray
        Weights of the area integral ``2*pi*r*dr``, in m^2, shape ``(n_r,)``.
    spectral_weights : numpy.ndarray
        Weights of ``2*pi*k_r*dk_r``, in m^-2, shape ``(n_r,)``.
    forward_scale : float
        Common forward scaling ``radius**2 / zeros[-1]``, in m^2.
    inverse_scale : float
        Common inverse scaling ``zeros[-1] / radius**2``, in m^-2.

    Raises
    ------
    GridError
        If the size, radius or resulting floating-point scales are invalid.

    References
    ----------
    M. Guizar-Sicairos and J. C. Gutiérrez-Vega, JOSA A 21, 53-58 (2004),
    doi:10.1364/JOSAA.21.000053.

    Examples
    --------
    >>> from solprop.hankel import QDHT
    >>> transform = QDHT(1e-3, 32)
    >>> transform.r.shape, transform.matrix.shape
    ((32,), (32, 32))
    """

    radius: float
    n_r: int
    r: Float[np.ndarray, " n_r"] = field(init=False, repr=False)
    k_r: Float[np.ndarray, " n_r"] = field(init=False, repr=False)
    zeros: Float[np.ndarray, " n_zeros"] = field(init=False, repr=False)
    j1: Float[np.ndarray, " n_r"] = field(init=False, repr=False)
    matrix: Float[np.ndarray, "n_r n_r"] = field(init=False, repr=False)
    radial_weights: Float[np.ndarray, " n_r"] = field(init=False, repr=False)
    spectral_weights: Float[np.ndarray, " n_r"] = field(init=False, repr=False)
    forward_scale: float = field(init=False)
    inverse_scale: float = field(init=False)

    def __post_init__(self) -> None:
        """Build independent, read-only arrays after validating the dimensions."""
        if isinstance(self.n_r, bool) or not isinstance(self.n_r, int) or self.n_r < 2:
            msg = f"n_r must be an integer of at least 2, got {self.n_r!r}"
            raise GridError(msg)
        if not math.isfinite(self.radius) or self.radius <= 0:
            msg = f"radius must be finite and positive, got {self.radius!r}"
            raise GridError(msg)
        zeros = jn_zeros(0, self.n_r + 1)
        edge = float(zeros[-1])
        forward_scale = (self.radius * self.radius) / edge
        if not math.isfinite(forward_scale) or forward_scale < 1 / np.finfo(float).max:
            msg = "radius produces an unrepresentable Hankel scaling"
            raise GridError(msg)
        inverse_scale = 1 / forward_scale
        j1 = np.abs(jv(1, zeros[:-1]))
        integration = 4 * np.pi / (edge * j1**2)
        if max(forward_scale, inverse_scale) > np.finfo(float).max / float(
            integration.max()
        ):
            msg = "radius produces unrepresentable radial integration weights"
            raise GridError(msg)
        matrix = 2 * jv(0, np.outer(zeros[:-1], zeros[:-1]) / edge)
        matrix /= edge * np.outer(j1, j1)
        arrays = {
            "zeros": zeros,
            "j1": j1,
            "r": self.radius * (zeros[:-1] / edge),
            "k_r": zeros[:-1] / self.radius,
            "matrix": matrix,
            "radial_weights": forward_scale * integration,
            "spectral_weights": inverse_scale * integration,
        }
        if any(not np.all(np.isfinite(values)) for values in arrays.values()):
            msg = "radius produces unrepresentable radial axes or integration weights"
            raise GridError(msg)
        for name, values in arrays.items():
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "forward_scale", forward_scale)
        object.__setattr__(self, "inverse_scale", inverse_scale)

    def forward(self, values: ArrayLike) -> Inexact[np.ndarray, "... n_r"]:
        """Transform radial samples to a Hankel spectrum along the last axis.

        Parameters
        ----------
        values : array_like
            Finite real or complex field samples, shape ``(..., n_r)``, in V/m.

        Returns
        -------
        numpy.ndarray
            Spectrum of the same shape, in V m; float64 or complex128.

        Raises
        ------
        GridError
            If the last axis has the wrong length or the values are nonfinite.
        """
        return self._apply(values, self.forward_scale)

    def inverse(self, values: ArrayLike) -> Inexact[np.ndarray, "... n_r"]:
        """Reconstruct radial samples from a Hankel spectrum along the last axis.

        Parameters
        ----------
        values : array_like
            Finite real or complex spectra, shape ``(..., n_r)``, in V m.

        Returns
        -------
        numpy.ndarray
            Field of the same shape, in V/m; float64 or complex128.

        Raises
        ------
        GridError
            If the last axis has the wrong length or the values are nonfinite.
        """
        return self._apply(values, self.inverse_scale)

    def _apply(self, values: ArrayLike, scale: float) -> Inexact[np.ndarray, "... n_r"]:
        """Apply the dimensionless transform between its two diagonal scalings.

        Parameters
        ----------
        values : array_like
            Finite real or complex samples with n_r on the final axis.
        scale : float
            Forward or inverse common scaling, in m^2 or m^-2.

        Returns
        -------
        numpy.ndarray
            Transformed samples, retaining the leading dimensions.
        """
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[-1] != self.n_r:
            msg = (
                f"Hankel samples must have last axis n_r={self.n_r}, got {array.shape}"
            )
            raise GridError(msg)
        if array.dtype.kind not in "biufc" or not np.all(np.isfinite(array)):
            msg = "Hankel samples must be finite real or complex numbers"
            raise GridError(msg)
        array = np.asarray(
            array, dtype=np.complex128 if np.iscomplexobj(array) else np.float64
        )
        return ((array / self.j1) @ self.matrix) * (scale * self.j1)
