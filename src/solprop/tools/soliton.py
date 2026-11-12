# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later


"""Soliton numbers: dispersion and nonlinear lengths, orders and constituents.

The closed forms of a pulse in a mode of given dispersion and nonlinear coefficient,
the soliton content of an unchirped input, and the fill's nonlinear coefficient.
"""

import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import erfcinv

from solprop._validation import positive
from solprop.constants import C
from solprop.exceptions import (
    ToolsError,
)
from solprop.modes import base
from solprop.modes.protocol import Mode
from solprop.tools.pulse import PulseShape, _omega, pulse_width

__all__ = [
    "SolitonContent",
    "dispersion_length",
    "fission_length",
    "nonlinear_coefficient",
    "nonlinear_length",
    "soliton_content",
    "soliton_order",
    "soliton_peak_power",
]


def dispersion_length(fwhm: float, beta2: float, shape: PulseShape = "sech") -> float:
    r"""Return the dispersion length :math:`L_D = \tau_0^2/|\beta_2|`.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum of the power in s, > 0.
    beta2 : float
        Group-velocity dispersion in s^2/m, non-zero.
    shape : {'sech', 'gauss'}, optional
        The pulse shape; see :func:`pulse_width`.

    Returns
    -------
    float
        Dispersion length in m.

    Raises
    ------
    ToolsError
        If the dispersion is zero or not finite, or the width is not positive.

    Examples
    --------
    >>> from solprop.tools import dispersion_length
    >>> f"{dispersion_length(10e-15, -8e-30):.3f}"
    '4.023'
    """
    if not math.isfinite(beta2) or beta2 == 0.0:
        raise ToolsError(f"beta2 must be finite and non-zero, got {beta2!r}")
    return pulse_width(fwhm, shape) ** 2 / abs(beta2)


def nonlinear_length(peak_power: float, gamma: float) -> float:
    r"""Return the nonlinear length :math:`L_{NL} = 1/(\gamma P_0)`.

    Parameters
    ----------
    peak_power : float
        Peak power in W, > 0.
    gamma : float
        Nonlinear coefficient in rad/(W m), > 0.

    Returns
    -------
    float
        Nonlinear length in m.

    Raises
    ------
    ToolsError
        If either quantity is not positive.

    Examples
    --------
    >>> from solprop.tools import nonlinear_length
    >>> nonlinear_length(1e9, 1e-9)
    1.0
    """
    return 1.0 / (
        positive(gamma, "gamma", ToolsError)
        * positive(peak_power, "peak_power", ToolsError)
    )


def soliton_order(
    peak_power: float,
    fwhm: float,
    gamma: float,
    beta2: float,
    shape: PulseShape = "sech",
) -> float:
    r"""Return the soliton order :math:`N = \sqrt{L_D/L_{NL}}`.

    The number is defined for either sign of the dispersion; it is a
    soliton order in the physical sense only where ``beta2 < 0``.

    Parameters
    ----------
    peak_power : float
        Peak power in W, > 0.
    fwhm : float
        Full width at half maximum of the power in s, > 0.
    gamma : float
        Nonlinear coefficient in rad/(W m), > 0.
    beta2 : float
        Group-velocity dispersion in s^2/m, non-zero.
    shape : {'sech', 'gauss'}, optional
        The pulse shape, used for the width **throughout**.

    Returns
    -------
    float
        The soliton order, dimensionless.

    Examples
    --------
    >>> from solprop.tools import soliton_order
    >>> f"{soliton_order(1e9, 10e-15, 1e-9, -8e-30):.4f}"
    '2.0057'
    """
    return math.sqrt(
        dispersion_length(fwhm, beta2, shape) / nonlinear_length(peak_power, gamma)
    )


def soliton_peak_power(
    order: float,
    fwhm: float,
    gamma: float,
    beta2: float,
    shape: PulseShape = "sech",
) -> float:
    r"""Return the peak power that gives a soliton order.

    The inverse of :func:`soliton_order` at fixed dispersion, nonlinear coefficient
    and width: :math:`N^2 = \gamma P_0\tau_0^2/|\beta_2|` is linear in the peak
    power, so :math:`P_0 = N^2|\beta_2|/(\gamma\tau_0^2)`. :func:`pulse_energy`
    turns it into an energy. On a mode, take `beta2` from
    :func:`solprop.modes.base.beta2` at the pump and `gamma` from
    :func:`nonlinear_coefficient`, both at the launch position.

    Parameters
    ----------
    order : float
        Conventional soliton order, > 0.
    fwhm : float
        Full width at half maximum of the power in s, > 0.
    gamma : float
        Nonlinear coefficient in rad/(W m), > 0.
    beta2 : float
        Group-velocity dispersion in s^2/m, non-zero.
    shape : {'sech', 'gauss'}, optional
        The pulse shape, used for the width.

    Returns
    -------
    float
        Peak power in W.

    Raises
    ------
    ToolsError
        If a quantity is not positive, the dispersion is zero, or the shape is
        unknown.

    Examples
    --------
    >>> from solprop.tools import soliton_order, soliton_peak_power
    >>> power = soliton_peak_power(2.0, 10e-15, 1e-9, -8e-30)
    >>> f"{power / 1e9:.4f} {soliton_order(power, 10e-15, 1e-9, -8e-30):.4f}"
    '0.9943 2.0000'
    """
    return positive(order, "order", ToolsError) ** 2 / (
        positive(gamma, "gamma", ToolsError) * dispersion_length(fwhm, beta2, shape)
    )


@dataclass(frozen=True)
class SolitonContent:
    """Describe the discrete soliton content of an unchirped single-lobe pulse.

    Attributes
    ----------
    area_order : float
        Dimensionless field area divided by pi, not the conventional soliton order.
    count : int
        Number of strictly positive discrete eigenvalues; zero at first threshold.
    largest_eigenvalue : float or None
        Imaginary part of the largest Zakharov-Shabat eigenvalue, dimensionless
        with time scaled by the input shape's natural width. None if count is zero.
    largest_peak_power : float or None
        Peak power of the largest constituent in W, or None if count is zero.
    largest_width : float or None
        Natural sech width of that constituent in s, or None if count is zero.
    largest_nonlinear_wavenumber : float or None
        Positive optics-convention phase rate in 1/m, or None if count is zero.
    """

    area_order: float
    count: int
    largest_eigenvalue: float | None
    largest_peak_power: float | None
    largest_width: float | None
    largest_nonlinear_wavenumber: float | None


def _scattering_angle(
    x: float, theta: Float[np.ndarray, " n_components"], order: float, eta: float
) -> list[float]:
    """Evaluate the real Zakharov-Shabat phase equation for a Gaussian."""
    return [-order * math.exp(-x * x / 2.0) - eta * math.sin(2 * theta[0])]


def _gaussian_eigenvalue(order: float) -> float:
    """Solve the half-line scattering problem for the largest Gaussian eigenvalue.

    - ``order``: conventional soliton order, above the first Gaussian threshold.
    """
    # The omitted field area is <= 1e-13. Moving the boundary with the amplitude
    # avoids assuming that the Gaussian tail is negligible for every input N.
    extent = math.sqrt(2.0) * float(erfcinv(1e-13 / (order * math.sqrt(math.pi / 2.0))))

    def residual(eta: float) -> float:
        """Return the ground-state symmetry residual at the pulse centre."""
        if eta == 0.0:
            # The exact area protects the first threshold from tail truncation.
            return math.pi / 4.0 - order * math.sqrt(math.pi / 2.0)
        solution = solve_ivp(
            _scattering_angle,
            (-extent, 0.0),
            [0.0],
            method="DOP853",
            args=(order, eta),
            rtol=1e-11,
            atol=1e-13,
        )
        if not solution.success:
            raise ToolsError(
                f"Gaussian scattering integration failed: {solution.message}"
            )
        return float(solution.y[0, -1]) + math.pi / 4.0

    # The ground-state angle crosses -pi/4 once for 0 < eta < max(q).
    # Unlike shooting the amplitudes, this does not overflow in the tails.
    try:
        return float(brentq(residual, 0.0, order, xtol=1e-13, rtol=1e-11))
    except (ValueError, RuntimeError) as error:
        raise ToolsError(f"Gaussian eigenvalue refinement failed: {error}") from error


def soliton_content(
    peak_power: float,
    fwhm: float,
    gamma: float,
    beta2: float,
    shape: PulseShape = "sech",
) -> SolitonContent:
    r"""Calculate the soliton count and largest constituent of an unchirped pulse.

    This is the scattering content of the lossless cubic focusing NLSE, not a
    prediction of fission dynamics with higher-order dispersion, Raman or plasma.
    The conventional :func:`soliton_order` remains a separate design parameter.

    Parameters
    ----------
    peak_power : float
        Input peak power in W, finite and positive.
    fwhm : float
        Input power FWHM in s, finite and positive.
    gamma : float
        Nonlinear coefficient in rad/(W m), finite and positive.
    beta2 : float
        Anomalous group-velocity dispersion in s^2/m, finite and negative.
    shape : {'sech', 'gauss'}, optional
        Unchirped input field shape; the Gaussian power is exp(-t^2/tau0^2).

    Returns
    -------
    SolitonContent
        Analytic count and largest constituent. The sech eigenvalue is analytic;
        the Gaussian eigenvalue is numerical, with dimensionless absolute accuracy
        about 1e-10 away from thresholds. Relative accuracy degrades at threshold.

    Raises
    ------
    ToolsError
        If inputs are invalid, dispersion is not anomalous, or scattering fails.

    References
    ----------
    .. [1] Klaus and Shaw, SIAM J. Math. Anal. 34, 759 (2003),
       doi:10.1137/S0036141002403067.
    .. [2] Kim, Lee and Lyng, arXiv:1310.4145, sections 1-2.
    """
    if not math.isfinite(beta2) or beta2 >= 0.0:
        raise ToolsError(
            "soliton content requires finite anomalous dispersion (beta2 < 0)"
        )
    order = soliton_order(peak_power, fwhm, gamma, beta2, shape)
    area_order = order if shape == "sech" else math.sqrt(2.0 / math.pi) * order
    if not math.isfinite(area_order):
        raise ToolsError("the normalised pulse area must be finite")
    # Strictly positive eigenvalues only: at a half-integer area the new state is
    # at zero and is not a square-integrable soliton (Klaus-Shaw threshold).
    count = max(0, math.ceil(area_order - 0.5))
    if count == 0:
        return SolitonContent(area_order, 0, None, None, None, None)
    eta = order - 0.5 if shape == "sech" else _gaussian_eigenvalue(order)
    if eta <= 0.0:
        raise ToolsError(
            "the positive Gaussian eigenvalue is too close to threshold to resolve"
        )
    power = peak_power * (2.0 * eta / order) ** 2
    width = pulse_width(fwhm, shape) / (2.0 * eta)
    return SolitonContent(area_order, count, eta, power, width, 0.5 * gamma * power)


def fission_length(
    peak_power: float,
    fwhm: float,
    gamma: float,
    beta2: float,
    shape: PulseShape = "sech",
) -> float:
    r"""Return the soliton fission length :math:`L_{\rm fiss} = L_D/N`.

    Parameters
    ----------
    peak_power : float
        Peak power in W, > 0.
    fwhm : float
        Full width at half maximum of the power in s, > 0.
    gamma : float
        Nonlinear coefficient in rad/(W m), > 0.
    beta2 : float
        Group-velocity dispersion in s^2/m, non-zero.
    shape : {'sech', 'gauss'}, optional
        The pulse shape; see :func:`pulse_width`.

    Returns
    -------
    float
        Fission length in m, the geometric mean of the two lengths.

    Examples
    --------
    >>> from solprop.tools import fission_length
    >>> f"{fission_length(1e9, 10e-15, 1e-9, -8e-30):.4f}"
    '2.0057'
    """
    order = soliton_order(peak_power, fwhm, gamma, beta2, shape)
    return dispersion_length(fwhm, beta2, shape) / order


def nonlinear_coefficient(
    mode: Mode, n2: float, wavelength: float, z: float = 0.0
) -> float:
    r"""Return the nonlinear coefficient :math:`\gamma = n_2\omega/(cA_{\rm eff})`.

    Parameters
    ----------
    mode : Mode
        The mode, for its effective area.
    n2 : float
        Nonlinear index of the fill in m^2/W, >= 0.
    wavelength : float
        Vacuum wavelength in m, > 0.
    z : float, optional
        Position along the waveguide, m.

    Returns
    -------
    float
        Nonlinear coefficient in rad/(W m).

    Raises
    ------
    ToolsError
        If the nonlinear index is negative or not finite.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.tools import nonlinear_coefficient
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> f"{nonlinear_coefficient(mode, 3.55e-25, 800e-9):.4e}"
    '1.1919e-10'
    """
    if not math.isfinite(n2) or n2 < 0.0:
        raise ToolsError(f"n2 must be finite and not negative, got {n2!r}")
    return n2 * _omega(wavelength) / (C * base.aeff(mode, z))


def _area_per_order(shape: PulseShape) -> float:
    """Return the pulse area over pi per unit conventional soliton order.

    - ``shape``: the pulse shape.
    """
    return 1.0 if shape == "sech" else math.sqrt(2.0 / math.pi)


def _gaussian_order(eigenvalue: float) -> float:
    """Return the conventional order whose Gaussian has a given largest eigenvalue.

    - ``eigenvalue``: target imaginary part of the largest Zakharov-Shabat eigenvalue, >
      0.
    """
    threshold = 0.5 / _area_per_order("gauss")

    def residual(order: float) -> float:
        """Return the largest eigenvalue at an order less the target."""
        # At and below the first threshold there is no bound state: the eigenvalue
        # that appears there starts from zero.
        if order <= threshold:
            return -eigenvalue
        return _gaussian_eigenvalue(order) - eigenvalue

    # N - eta of the Gaussian falls from sqrt(pi/8) = 0.627 at the first threshold
    # towards 1/2 as N grows (0.579 at N = 1, 0.506 at N = 10, measured with
    # _gaussian_eigenvalue), so the root lies between eta + 1/2 and eta + 0.627. The
    # upper end carries a margin; brentq refuses a bracket that does not hold the root.
    low = max(threshold, eigenvalue + 0.5)
    high = eigenvalue + 0.7
    try:
        return float(brentq(residual, low, high, xtol=1e-12, rtol=1e-11))
    except ValueError as error:
        raise ToolsError(
            f"no Gaussian soliton order between {low:.6g} and {high:.6g} has a largest "
            f"eigenvalue of {eigenvalue:.6g}: {error}"
        ) from error
