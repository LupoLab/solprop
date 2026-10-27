# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The reference backend's discrete Fourier transforms, chosen by the operand's format.

A binary64 operand goes to ``numpy.fft``, which keeps the scalar oracle's transform
arithmetic bit for bit. A binary32 operand goes to ``scipy.fft``, which sums in
binary32. NumPy's float32 transforms return complex64 but are computed in binary64 and
rounded once, so every bin is accurate to binary64. That under-represents the binary32
transform error a reduced-precision backend carries, of order :math:`u_{32}\log n` of
the input's norm in every bin, which a resonant response such as stimulated Raman
scattering amplifies from below the true spectrum. The single-precision oracle
(SEP 33) must round as its backends do.

Each function takes the arguments of its ``numpy.fft`` namesake that the backend uses.
``out``, where given, receives the result, which is also returned.

Examples
--------
>>> import numpy as np
>>> from solprop.backends.numpy_ref import transforms
>>> transforms.rfft(np.ones(4, dtype=np.float32)).dtype
dtype('complex64')
>>> transforms.rfft(np.ones(4)).dtype
dtype('complex128')
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import scipy.fft
from jaxtyping import Inexact, Shaped

__all__ = ["fft", "fft2", "ifft", "ifft2", "irfft", "rfft"]

_BINARY32 = (np.dtype(np.float32), np.dtype(np.complex64))


def _library(values: np.ndarray) -> ModuleType:
    """Return the transform library for an operand's format.

    Parameters
    ----------
    values : numpy.ndarray
        The operand.

    Returns
    -------
    types.ModuleType
        ``scipy.fft`` for a binary32 operand, ``numpy.fft`` otherwise.
    """
    return scipy.fft if values.dtype in _BINARY32 else np.fft


def _into(result: np.ndarray, out: np.ndarray | None) -> np.ndarray:
    """Return a transform's result, written into a given array first.

    Parameters
    ----------
    result : numpy.ndarray
        The transform's result.
    out : numpy.ndarray or None
        Where to write it, or ``None`` to return `result` itself.

    Returns
    -------
    numpy.ndarray
        `out` holding `result`, or `result`.
    """
    if out is None:
        return result
    out[...] = result
    return out


def rfft(
    values: Inexact[np.ndarray, "*input"], n: int | None = None, axis: int = -1
) -> Shaped[np.ndarray, "*output"]:
    """Return the forward real transform along an axis.

    Parameters
    ----------
    values : numpy.ndarray
        Real input.
    n : int or None, optional
        Length to zero-pad or crop the axis to; ``None`` keeps it.
    axis : int, optional
        The transformed axis, the last by default.

    Returns
    -------
    numpy.ndarray
        The half spectrum, complex in the operand's precision.
    """
    return _library(values).rfft(values, n=n, axis=axis)


def irfft(
    values: Inexact[np.ndarray, "*input"], n: int | None = None, axis: int = -1
) -> Shaped[np.ndarray, "*output"]:
    """Return the inverse real transform along an axis.

    Parameters
    ----------
    values : numpy.ndarray
        Half spectrum.
    n : int or None, optional
        Length of the real output; ``None`` takes ``2 * (m - 1)``.
    axis : int, optional
        The transformed axis, the last by default.

    Returns
    -------
    numpy.ndarray
        The real signal in the operand's precision.
    """
    return _library(values).irfft(values, n=n, axis=axis)


def fft(
    values: Inexact[np.ndarray, "*input"], axis: int = -1
) -> Shaped[np.ndarray, "*output"]:
    """Return the forward complex transform along an axis.

    Parameters
    ----------
    values : numpy.ndarray
        Input.
    axis : int, optional
        The transformed axis, the last by default.

    Returns
    -------
    numpy.ndarray
        The spectrum, complex in the operand's precision.
    """
    return _library(values).fft(values, axis=axis)


def ifft(
    values: Inexact[np.ndarray, "*input"], axis: int = -1
) -> Shaped[np.ndarray, "*output"]:
    """Return the inverse complex transform along an axis.

    Parameters
    ----------
    values : numpy.ndarray
        Spectrum.
    axis : int, optional
        The transformed axis, the last by default.

    Returns
    -------
    numpy.ndarray
        The signal, complex in the operand's precision.
    """
    return _library(values).ifft(values, axis=axis)


def fft2(
    values: Inexact[np.ndarray, "*input"],
    axes: tuple[int, int] = (0, 1),
    out: np.ndarray | None = None,
) -> Shaped[np.ndarray, "*output"]:
    """Return the forward two-dimensional transform over two axes.

    Parameters
    ----------
    values : numpy.ndarray
        Input.
    axes : tuple of int, optional
        The transformed axes, the first two by default.
    out : numpy.ndarray or None, optional
        Where to write the result.

    Returns
    -------
    numpy.ndarray
        The spectrum, in `out` when given.
    """
    return _into(_library(values).fft2(values, axes=axes), out)


def ifft2(
    values: Inexact[np.ndarray, "*input"],
    axes: tuple[int, int] = (0, 1),
    out: np.ndarray | None = None,
) -> Shaped[np.ndarray, "*output"]:
    """Return the inverse two-dimensional transform over two axes.

    Parameters
    ----------
    values : numpy.ndarray
        Spectrum.
    axes : tuple of int, optional
        The transformed axes, the first two by default.
    out : numpy.ndarray or None, optional
        Where to write the result.

    Returns
    -------
    numpy.ndarray
        The signal, in `out` when given.
    """
    return _into(_library(values).ifft2(values, axes=axes), out)
