# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Construct Jones optics and describe transverse polarisation in the optics convention.

Jones vectors are ``[E_x, E_y]``. Stokes vectors are ``[I, Q, U, V]`` with
``V = -2 Im(E_x conj(E_y))``, so ``(x + i y)/sqrt(2)`` has positive V. A waveplate's
fast axis is x: its retardance delays the vertical component relative to the
horizontal one, and in the optics convention a delay is a positive phase
(``conventions.md`` §2), so a quarter-wave plate turns light linear at +45° into
``(x + i y)/sqrt(2)``. Convert internal FFT-convention fields at the existing Fourier
boundary before using these helpers.

The ellipse orientation is ``atan2(U, Q)/2``, which keeps the sign of ``U``. Circular
and zero fields use zero orientation by convention; normalised Stokes coordinates
require nonzero intensity.
"""

import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float
from numpy.typing import ArrayLike

from solprop.exceptions import PolarisationError

__all__ = [
    "PolarisationEllipse",
    "cartesian",
    "ellipse",
    "ellipticity",
    "horizontal",
    "linear_polariser",
    "rotate",
    "rotation",
    "stokes",
    "waveplate",
]


@dataclass(frozen=True)
class PolarisationEllipse:
    """Describe the ellipse associated with a Stokes vector.

    Attributes
    ----------
    major : float
        Major semiaxis, in the field's amplitude units, non-negative.
    minor : float
        Minor semiaxis in the same units, between zero and `major`.
    angle : float
        Major-axis orientation from x towards y, rad, in [-pi/2, pi/2]. Zero for
        circular or zero fields, whose orientation is undefined.
    handedness : int
        Sign of Stokes V: -1, 0 or +1. This names the algebraic sign without assuming
        an observer's viewing direction.
    """

    major: float
    minor: float
    angle: float
    handedness: int


def _angle(value: float) -> None:
    """Reject a non-finite angle before evaluating trigonometric functions.

    Parameters
    ----------
    value : float
        Angle in rad.
    """
    if not math.isfinite(value):
        raise PolarisationError(f"angle must be finite, got {value!r}")


def _complex_array(
    value: ArrayLike, shape: tuple[int, ...]
) -> Complex[np.ndarray, " *shape"]:
    """Validate a finite Jones vector or matrix without broadcasting.

    Parameters
    ----------
    value : array_like
        Jones data in field units or dimensionless for a matrix.
    shape : tuple of int
        Required dimensions.

    Returns
    -------
    numpy.ndarray
        Finite complex128 data with the required shape.
    """
    array = np.asarray(value, dtype=np.complex128)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise PolarisationError(
            f"expected a finite Jones array of shape {shape}, got {array!r}"
        )
    return array


def _stokes_vector(value: ArrayLike) -> Float[np.ndarray, " 4"]:
    """Validate the Stokes cone with arithmetic roundoff at the surface.

    Parameters
    ----------
    value : array_like
        Stokes vector of shape ``(4,)`` in squared field units.

    Returns
    -------
    numpy.ndarray
        Finite real Stokes vector inside the physical cone.
    """
    array = np.asarray(value)
    if np.iscomplexobj(array) or array.shape != (4,) or not np.all(np.isfinite(array)):
        raise PolarisationError("a Stokes vector must contain four finite real values")
    result = np.asarray(array, dtype=np.float64)
    intensity = float(result[0])
    polarised = math.hypot(*(float(part) for part in result[1:]))
    # Four squares/products and a reduction put a pure Jones state on the cone to
    # O(epsilon). Sixteen epsilon covers these operations; it is not a fit to data.
    if intensity < 0.0 or polarised > intensity * (1.0 + 16 * np.finfo(float).eps):
        raise PolarisationError(
            "Stokes I must be non-negative and at least hypot(Q, U, V)"
        )
    return result


def horizontal() -> Complex[np.ndarray, " 2"]:
    """Return the unit Jones vector along x.

    Returns
    -------
    numpy.ndarray
        Dimensionless ``[1, 0]``, shape ``(2,)``, complex128.
    """
    return np.array([1.0, 0.0], dtype=np.complex128)


def linear_polariser() -> Complex[np.ndarray, "two two"]:
    """Return an ideal linear polariser transmitting x.

    Returns
    -------
    numpy.ndarray
        Dimensionless Jones matrix ``diag(1, 0)``, shape ``(2, 2)``.
    """
    return np.diag(np.array([1.0, 0.0], dtype=np.complex128))


def waveplate(retardance: float) -> Complex[np.ndarray, "two two"]:
    """Return a waveplate with its fast axis along x in the optics convention.

    Parameters
    ----------
    retardance : float
        Finite retardance of the slow y component behind the fast x one, rad. A
        delay is a positive optics-convention phase, so y gains ``+retardance``
        relative to x.

    Returns
    -------
    numpy.ndarray
        Unitary Jones matrix ``diag(exp(-i phase/2), exp(+i phase/2))``.

    Raises
    ------
    PolarisationError
        If the retardance is not finite.
    """
    _angle(retardance)
    return np.diag(np.exp(0.5j * retardance * np.array([-1.0, 1.0])))


def rotation(angle: float) -> Float[np.ndarray, "two two"]:
    """Return a passive rotation of the transverse coordinate axes.

    Parameters
    ----------
    angle : float
        Finite angle from x towards y, rad.

    Returns
    -------
    numpy.ndarray
        Dimensionless matrix ``[[cos, sin], [-sin, cos]]``.

    Raises
    ------
    PolarisationError
        If the angle is not finite.
    """
    _angle(angle)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, sine], [-sine, cosine]])


def rotate(matrix: ArrayLike, angle: float) -> Complex[np.ndarray, "two two"]:
    """Rotate a Jones optical element from x towards y.

    Parameters
    ----------
    matrix : array_like
        Finite dimensionless Jones matrix, shape ``(2, 2)``.
    angle : float
        Finite physical rotation angle, rad.

    Returns
    -------
    numpy.ndarray
        ``rotation(-angle) @ matrix @ rotation(angle)``.

    Raises
    ------
    PolarisationError
        If the matrix or angle is invalid.
    """
    element = _complex_array(matrix, (2, 2))
    return rotation(-angle) @ element @ rotation(angle)


def stokes(field: ArrayLike, *, normalise: bool = False) -> Float[np.ndarray, " 4"]:
    """Return Stokes parameters for one optics-convention Jones vector.

    Parameters
    ----------
    field : array_like
        Finite Jones vector ``[E_x, E_y]``, shape ``(2,)``.
    normalise : bool, optional
        Divide by the total intensity; requires a nonzero field.

    Returns
    -------
    numpy.ndarray
        ``[I, Q, U, V]``, shape ``(4,)``, in squared field units or dimensionless
        when normalised. Positive V corresponds to ``(x + i y)/sqrt(2)``.

    Raises
    ------
    PolarisationError
        If the vector is invalid, squared amplitudes overflow, or normalisation is
        requested for zero intensity.

    Examples
    --------
    >>> from solprop.polarisation import stokes
    >>> stokes([1, 1j], normalise=True).tolist()
    [1.0, 0.0, 0.0, 1.0]
    """
    values = _complex_array(field, (2,))
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        if normalise:
            raise PolarisationError("cannot normalise zero intensity")
        return np.zeros(4)
    x, y = values / scale
    cross = x * np.conj(y)
    result = np.array(
        [
            abs(x) ** 2 + abs(y) ** 2,
            abs(x) ** 2 - abs(y) ** 2,
            2 * cross.real,
            -2 * cross.imag,
        ],
        dtype=np.float64,
    )
    if normalise:
        return result / result[0]
    if scale > math.sqrt(np.finfo(float).max / float(result[0])):
        raise PolarisationError("squared Jones amplitudes overflow float64")
    return result * scale * scale


def cartesian(parameters: ArrayLike) -> Float[np.ndarray, " 3"]:
    """Return normalised Stokes coordinates on or inside the Poincare sphere.

    Parameters
    ----------
    parameters : array_like
        Physical Stokes vector, shape ``(4,)``, with positive I.

    Returns
    -------
    numpy.ndarray
        Dimensionless ``[Q, U, V]/I``, shape ``(3,)``.

    Raises
    ------
    PolarisationError
        If the vector is invalid or I is zero.
    """
    values = _stokes_vector(parameters)
    if values[0] == 0.0:
        raise PolarisationError("cannot normalise zero intensity")
    return values[1:] / values[0]


def ellipse(parameters: ArrayLike) -> PolarisationEllipse:
    """Return ellipse semiaxes, orientation and handedness from Stokes parameters.

    Parameters
    ----------
    parameters : array_like
        Physical Stokes vector ``[I, Q, U, V]``, shape ``(4,)``.

    Returns
    -------
    PolarisationEllipse
        Semiaxes in field units, orientation in rad, and the sign of V. For a
        partially polarised Stokes vector this describes its equivalent ellipse.

    Raises
    ------
    PolarisationError
        If the Stokes vector is nonphysical or not finite.
    """
    intensity, q, u, v = (float(value) for value in _stokes_vector(parameters))
    linear = min(math.hypot(q, u), intensity)
    return PolarisationEllipse(
        major=math.sqrt(intensity / 2 + linear / 2),
        minor=math.sqrt(max(0.0, intensity / 2 - linear / 2)),
        angle=0.0 if linear == 0.0 else math.atan2(u, q) / 2,
        handedness=int(np.sign(v)),
    )


def ellipticity(parameters: ArrayLike) -> float:
    """Return the unsigned minor-to-major semiaxis ratio.

    Parameters
    ----------
    parameters : array_like
        Physical Stokes vector, shape ``(4,)``, with positive intensity.

    Returns
    -------
    float
        Ratio in [0, 1]: zero for linear polarisation, one for circular.

    Raises
    ------
    PolarisationError
        If the vector is invalid or the field has zero intensity.
    """
    result = ellipse(parameters)
    if result.major == 0.0:
        raise PolarisationError("zero intensity has no ellipticity")
    return result.minor / result.major
