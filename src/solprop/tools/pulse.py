# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later


"""Pulse shape numbers: the natural width, peak power and energy of a pulse.

The ``sech`` and Gaussian shapes the design numbers are defined for, and the carrier
frequency of a vacuum wavelength the other tools modules share.
"""

import math
from typing import Literal, get_args

from solprop._validation import choice, positive
from solprop.constants import C
from solprop.exceptions import (
    ToolsError,
)

__all__ = ["PulseShape", "pulse_energy", "pulse_peak_power", "pulse_width"]

PulseShape = Literal["sech", "gauss"]
"""The two pulse shapes the design numbers are defined for."""


_SECH_WIDTH = 2.0 * math.log(1.0 + math.sqrt(2.0))
"""``fwhm / tau0`` of ``sech^2(t / tau0)``: twice the inverse cosh of the root of 2."""


_GAUSS_WIDTH = 2.0 * math.sqrt(math.log(2.0))
"""``fwhm / tau0`` of ``exp(-t^2 / tau0^2)``."""


def _omega(wavelength: float) -> float:
    """Return the angular frequency of a vacuum wavelength.

    - ``wavelength``: vacuum wavelength in m, > 0.
    - Returns: angular frequency in rad/s.
    """
    return 2.0 * math.pi * C / positive(wavelength, "wavelength", ToolsError)


def pulse_width(fwhm: float, shape: PulseShape = "sech") -> float:
    r"""Return the natural width of a pulse from its full width at half maximum.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum of the **power** in s, > 0.
    shape : {'sech', 'gauss'}, optional
        ``"sech"`` for :math:`P\propto{\rm sech}^2(t/\tau_0)`, ``"gauss"`` for
        :math:`P\propto\exp(-t^2/\tau_0^2)`.

    Returns
    -------
    float
        :math:`\tau_0` in s: ``fwhm / (2 ln(1 + sqrt 2))`` or ``fwhm / (2 sqrt(ln 2))``.

    Raises
    ------
    ToolsError
        If the shape is unknown or the width is not positive.

    Examples
    --------
    >>> from solprop.tools import pulse_width
    >>> f"{pulse_width(10e-15) * 1e15:.4f}"
    '5.6730'
    >>> f"{pulse_width(10e-15, 'gauss') * 1e15:.4f}"
    '6.0056'
    """
    choice(shape, "shape", get_args(PulseShape), ToolsError)
    return positive(fwhm, "fwhm", ToolsError) / (
        _SECH_WIDTH if shape == "sech" else _GAUSS_WIDTH
    )


def pulse_peak_power(energy: float, fwhm: float, shape: PulseShape = "sech") -> float:
    r"""Return the peak power of a pulse of given energy, duration and shape.

    Parameters
    ----------
    energy : float
        Pulse energy in J, > 0.
    fwhm : float
        Full width at half maximum of the power in s, > 0.
    shape : {'sech', 'gauss'}, optional
        The pulse shape; see :func:`pulse_width`.

    Returns
    -------
    float
        Peak power in W: :math:`E/(2\tau_0)` for ``sech`` and
        :math:`E/(\sqrt\pi\,\tau_0)` for a Gaussian, the integrals of the two shapes.

    Raises
    ------
    ToolsError
        If the shape is unknown or a quantity is not positive.

    Examples
    --------
    >>> from solprop.tools import pulse_peak_power
    >>> f"{pulse_peak_power(1e-3, 10e-15) / 1e9:.3f}"
    '88.137'
    >>> f"{pulse_peak_power(1e-3, 10e-15, 'gauss') / 1e9:.3f}"
    '93.944'
    """
    width = pulse_width(fwhm, shape)
    integral = 2.0 if shape == "sech" else math.sqrt(math.pi)
    return positive(energy, "energy", ToolsError) / (integral * width)


def pulse_energy(peak_power: float, fwhm: float, shape: PulseShape = "sech") -> float:
    """Return the energy of a pulse of given peak power, duration and shape.

    The inverse of :func:`pulse_peak_power`, for the same shape.

    Parameters
    ----------
    peak_power : float
        Peak power in W, > 0.
    fwhm : float
        Full width at half maximum of the power in s, > 0.
    shape : {'sech', 'gauss'}, optional
        The pulse shape; see :func:`pulse_width`.

    Returns
    -------
    float
        Pulse energy in J.

    Raises
    ------
    ToolsError
        If the shape is unknown or a quantity is not positive.

    Examples
    --------
    >>> from solprop.tools import pulse_energy
    >>> f"{pulse_energy(88.137e9, 10e-15) * 1e3:.4f}"
    '1.0000'
    """
    # The peak power is linear in the energy at a fixed width and shape.
    return positive(peak_power, "peak_power", ToolsError) / pulse_peak_power(
        1.0, fwhm, shape
    )
