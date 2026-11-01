# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Derived quantities of any mode: beta, alpha, group delay, dispersion, ZDW.

Everything here is a function of a :class:`~solprop.modes.protocol.Mode`, not a method
on one, so a new mode family gets them for nothing. The conversions are the package's
(``docs/dev/conventions.md`` section 2):

.. math::
   \beta = \frac{\omega}{c}\,\mathrm{Re}\,n_{\rm eff},\qquad
   \alpha = \frac{2\omega}{c}\,\mathrm{Im}\,n_{\rm eff},

so a positive imaginary part of the effective index is loss and ``alpha`` is a power
loss coefficient per metre.

:func:`beta1` is the group delay per unit length that the co-moving frame rides on. It
is **analytic**, straight from the mode's ``dneff_domega``: differencing ``beta`` with a
seven-point stencil every stage would cost about seven index evaluations and leave 1e-10
of roughness that floors the whole cumulative operator at 1e-8 (``docs/dev/stepper.md``
section 5).

:func:`beta2` is one step further. The second derivative of a gas susceptibility is not
part of the material interface, so it is a Richardson-extrapolated central difference of
the *analytic* ``beta1`` -- one derivative differenced instead of two, which is a very
different thing from differencing ``beta`` itself, and reaches about 1e-10 relative
rather than 1e-6. :func:`zdw` brackets its root.
"""

import logging
import math
from typing import Final

import numpy as np
from jaxtyping import Float
from scipy.integrate import quad_vec
from scipy.optimize import brentq

from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.modes.protocol import Mode

__all__ = [
    "BETA2_RELATIVE_STEP",
    "aeff",
    "alpha",
    "beta",
    "beta1",
    "beta2",
    "db_per_m",
    "loss_length",
    "transmission",
    "zdw",
]

logger = logging.getLogger(__name__)

BETA2_RELATIVE_STEP: Final = 3e-3
"""Relative frequency step of the Richardson difference behind :func:`beta2`.

Larger than a finite difference usually wants, and deliberately so. ``beta_1`` is of
order ``1/c`` while its variation over the step is ``beta_2 h``, which for a capillary
is nine orders smaller, so the rounding error of the difference is ``eps beta_1 / h``
-- it *falls* as the step grows. Truncation, ``(h^2/6) beta_4``, rises. Measured
against the closed-form second derivative of an evacuated capillary
(``tests/modes/test_base.py``), the two cross near a relative step of 3e-3, where the
result is good to 5e-9; at 1e-4 it is 1.6e-7 and at 1e-5 it is 2.3e-6.

That is three orders better than differencing ``beta`` itself, and it is
enough: nothing in the solver consumes ``beta_2``. The co-moving frame uses ``beta_1``,
which is analytic. Making this analytic too needs the second derivative of the gas
susceptibility, which is not part of the material interface.
"""

_DECIBEL_PER_NEPER: Final = 10.0 / math.log(10.0)
"""Conversion from a power loss coefficient in 1/m to dB/m."""


def _as_omega(omega: Float[np.ndarray, " n_omega"]) -> Float[np.ndarray, " n_omega"]:
    """Validate an angular-frequency argument for a derived quantity.

    Parameters
    ----------
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        The same values as a float array.

    Raises
    ------
    ModeError
        If any value is not finite.
    """
    values = np.asarray(omega, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ModeError("angular frequency must be finite")
    return values


def beta(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the propagation constant of a mode.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``beta = omega Re(n_eff) / c`` in 1/m, shape ``(n_omega,)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import beta
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{beta(mode, np.array([2.356e15]))[0]:.6f}"
    '7858997.479689'
    """
    values = _as_omega(omega)
    return np.asarray(values * np.real(mode.neff(values, z)) / C)


def alpha(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the power loss coefficient of a mode.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``alpha = 2 omega Im(n_eff) / c`` in 1/m, shape ``(n_omega,)``. Positive is
        loss, because ``Im n_eff > 0`` is.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import alpha
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{alpha(mode, np.array([2.356e15]))[0]:.6f}"
    '0.141459'
    """
    values = _as_omega(omega)
    return np.asarray(2.0 * values * np.imag(mode.neff(values, z)) / C)


def beta1(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the group delay per unit length, analytically.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``beta_1 = d(beta)/d(omega) = (Re n_eff + omega d(Re n_eff)/d(omega)) / c`` in
        s/m, shape ``(n_omega,)``, from the mode's analytic derivative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import beta1
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{beta1(mode, np.array([2.356e15]))[0]:.10e}"
    '3.3357582351e-09'
    """
    values = _as_omega(omega)
    index = np.real(mode.neff(values, z))
    derivative = np.real(mode.dneff_domega(values, z))
    return np.asarray((index + values * derivative) / C)


def beta2(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the group-velocity dispersion, by differencing the analytic `beta1`.

    Departs from the analytic derivatives elsewhere in this module, and says so: the
    second derivative of a gas susceptibility is not part of the material interface,
    so this is a Richardson-extrapolated central difference of :func:`beta1` at a
    relative step of :data:`BETA2_RELATIVE_STEP`. It reaches about 1e-10 relative,
    against the 1e-6 of differencing ``beta`` twice, and only the co-moving frame
    needs better -- which uses ``beta1``, not this.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``, > 0.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``beta_2 = d^2(beta)/d(omega)^2`` in s^2/m, shape ``(n_omega,)``.

    Raises
    ------
    ModeError
        If any frequency is not finite or not positive, since the step is relative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import beta2
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{beta2(mode, np.array([2.356e15]))[0]:.6e}"
    '-7.491419e-30'
    """
    values = _as_omega(omega)
    if np.any(values <= 0.0):
        raise ModeError(
            "beta2 differences beta1 with a relative step, so the frequency must be "
            "positive"
        )
    step = BETA2_RELATIVE_STEP * values
    wide = (beta1(mode, values + step, z) - beta1(mode, values - step, z)) / (
        2.0 * step
    )
    half = step / 2.0
    narrow = (beta1(mode, values + half, z) - beta1(mode, values - half, z)) / (
        2.0 * half
    )
    return np.asarray((4.0 * narrow - wide) / 3.0)


def zdw(
    mode: Mode,
    z: float = 0.0,
    *,
    wavelength_limits: tuple[float, float] = (100e-9, 3e-6),
) -> float:
    """Return the zero-dispersion wavelength of a mode.

    Parameters
    ----------
    mode : Mode
        The mode.
    z : float, optional
        Position along the waveguide, m.
    wavelength_limits : tuple of float, optional
        Shortest and longest wavelength in m to bracket the root in. Defaults to
        100 nm to 3 um. A bracket wider than the validity range of a
        material model behind the mode makes that model warn, once per evaluation:
        the ends are only probes, but they are still extrapolation.

    Returns
    -------
    float
        The wavelength in m where ``beta_2`` vanishes.

    Raises
    ------
    ModeError
        If the range is not ordered and positive, or if ``beta_2`` does not change
        sign across it: a mode with no zero-dispersion point in the window is a fact
        the caller has to see, not a missing value to propagate.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import zdw
    >>> mode = MarcatiliMode(125e-6, "He", 0.4, model="reduced", cladding=1.45)
    >>> f"{zdw(mode, wavelength_limits=(150e-9, 900e-9)) * 1e9:.1f}"
    '376.7'
    """
    shortest, longest = wavelength_limits
    if not 0.0 < shortest < longest:
        raise ModeError(
            f"wavelength_limits must be positive and ordered, got {wavelength_limits}"
        )
    lower = 2.0 * math.pi * C / longest
    upper = 2.0 * math.pi * C / shortest

    def dispersion(value: float) -> float:
        """Return beta2 at one angular frequency, for the root finder.

        Parameters
        ----------
        value : float
            Angular frequency in rad/s.

        Returns
        -------
        float
            ``beta_2`` in s^2/m.
        """
        return float(beta2(mode, np.array([value]), z)[0])

    at_lower, at_upper = dispersion(lower), dispersion(upper)
    if at_lower * at_upper > 0.0:
        raise ModeError(
            f"beta2 does not change sign over {shortest * 1e9:.4g}-"
            f"{longest * 1e9:.4g} nm (it is {at_lower:.4g} and {at_upper:.4g} s^2/m at "
            "the ends), so there is no zero-dispersion wavelength in that window"
        )
    root = brentq(dispersion, lower, upper)
    return 2.0 * math.pi * C / float(root)


def loss_length(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the 1/e power loss length of a mode.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``1 / alpha`` in m, shape ``(n_omega,)``; infinite where the mode is lossless.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import loss_length
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{loss_length(mode, np.array([2.356e15]))[0]:.4f}"
    '7.0692'
    """
    with np.errstate(divide="ignore"):
        return np.asarray(1.0 / alpha(mode, omega, z))


def db_per_m(
    mode: Mode, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
) -> Float[np.ndarray, " n_omega"]:
    """Return the loss of a mode in decibels per metre.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    numpy.ndarray
        ``10 alpha / ln(10)`` in dB/m, shape ``(n_omega,)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import db_per_m
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{db_per_m(mode, np.array([2.356e15]))[0]:.6f}"
    '0.614348'
    """
    return np.asarray(_DECIBEL_PER_NEPER * alpha(mode, omega, z))


def transmission(
    mode: Mode,
    omega: Float[np.ndarray, " n_omega"],
    length: float,
    start: float = 0.0,
) -> Float[np.ndarray, " n_omega"]:
    """Return the power transmission of a length of waveguide.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    length : float
        Length propagated in m, >= 0.
    start : float, optional
        Position the length starts at, m.

    Returns
    -------
    numpy.ndarray
        ``exp(-int alpha dz)`` over ``[start, start + length]``, shape ``(n_omega,)``.
        The integral is taken numerically because a taper or a gradient makes
        ``alpha`` a function of position; for a uniform waveguide it is exactly
        ``exp(-alpha L)``.

    Raises
    ------
    ModeError
        If `length` is negative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import transmission
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{transmission(mode, np.array([2.356e15]), 1.0)[0]:.6f}"
    '0.868091'
    """
    values = _as_omega(omega)
    if length < 0.0:
        raise ModeError(f"a propagation length must not be negative, got {length}")
    if length == 0.0:
        return np.ones_like(values)
    integral = quad_vec(
        lambda position: alpha(mode, values, position), start, start + length
    )[0]
    return np.asarray(np.exp(-integral))


def aeff(mode: Mode, z: float = 0.0) -> float:
    """Return the effective area of a mode.

    A uniform accessor beside :func:`beta` and :func:`alpha`, so that code working over
    a mode set never has to mix method calls with function calls.

    Parameters
    ----------
    mode : Mode
        The mode.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    float
        ``A_eff`` in m^2.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.base import aeff
    >>> f"{aeff(MarcatiliMode(75e-6, 'He', 1.0)):.6e}"
    '8.421575e-09'
    """
    return mode.aeff(z)
