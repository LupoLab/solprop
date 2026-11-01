# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Design calculators: soliton numbers and dispersive-wave phase matching.

The numbers a person works out before a run. For a pulse of energy :math:`E` and
full width at half maximum :math:`\tau_{\rm fwhm}` in a mode with group-velocity
dispersion :math:`\beta_2`, effective area :math:`A_{\rm eff}` and a fill of nonlinear
index :math:`n_2`,

.. math::
   \gamma = \frac{n_2\omega_0}{cA_{\rm eff}},\quad
   L_D = \frac{\tau_0^2}{|\beta_2|},\quad
   L_{NL} = \frac{1}{\gamma P_0},\quad
   N = \sqrt{L_D/L_{NL}},\quad
   L_{\rm fiss} = L_D/N,

with the natural width :math:`\tau_0` and the peak power :math:`P_0` of the chosen
pulse shape (:func:`pulse_width`, :func:`pulse_peak_power`). One shape runs through the
whole chain, so the Gaussian numbers are never computed with a ``sech`` width.

A dispersive wave is phase matched to a soliton at the frequencies where

.. math::
   \Delta\beta(\omega) = \beta_{\rm dw}(\omega) - \beta_{\rm s}(\omega_0)
   - \beta_{1,\rm s}(\omega_0)(\omega-\omega_0) - k_{\rm NL}

vanishes (:func:`phase_mismatch`). The last term is the soliton's nonlinear wavenumber
and uses the largest unchirped constituent when pulse data are given
(:func:`soliton_content`). Without pulse data the condition is linear.
``docs/guide/workflows/design-tools.md`` derives both and says what the nonlinear term
assumes.

Everything is SI except pressure, which is in bar as at the rest of the user interface
(``docs/dev/conventions.md`` section 1).
"""

import logging
import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, cast

import numpy as np
from jaxtyping import Float
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import erfcinv

from solprop import spatial
from solprop._display import Displayed, line
from solprop.constants import EPS0, ROOM_TEMPERATURE, C
from solprop.exceptions import (
    DiluteScalingWarning,
    MaterialError,
    MaterialRangeWarning,
    ModeError,
    ToolsError,
)
from solprop.modes import base
from solprop.modes.antiresonant import TubeLattice, VincettiMode
from solprop.modes.capillary import LossSpec, MarcatiliMode, MarcatiliModel, ModeKind
from solprop.modes.protocol import Mode
from solprop.modes.quadrature import BASIS_NORMALISATION, mode_matrix, on_axis_rule
from solprop.physdata import Gas, Mixture, chi3_from_n2, n2_from_chi3, species
from solprop.physdata.glasses import GlassIndex, fused_silica
from solprop.physdata.nonlinear_data import carriers_agree
from solprop.physdata.selection import MaterialSelection
from solprop.profiles import ZProfile
from solprop.pulses import GaussianPulse, SechPulse

__all__ = [
    "RDW",
    "ZDW",
    "DesignBuild",
    "DesignRoots",
    "DesignTarget",
    "Fill",
    "GasRatio",
    "NonlinearPhase",
    "PulseShape",
    "RootChoice",
    "SolitonContent",
    "SolitonOrder",
    "SolitonParams",
    "capillary_design",
    "capillary_params",
    "dispersion_length",
    "field_to_intensity",
    "fission_length",
    "gas_ratio",
    "intensity_to_field",
    "lambda_rdw",
    "lambda_rdw_all",
    "nonlinear_coefficient",
    "nonlinear_length",
    "phase_mismatch",
    "pressure_rdw",
    "pressure_zdw",
    "pulse_energy",
    "pulse_peak_power",
    "pulse_width",
    "radius_zdw",
    "self_focusing_power",
    "soliton_content",
    "soliton_order",
    "soliton_params",
    "soliton_peak_power",
    "solve_parameter",
]

logger = logging.getLogger(__name__)

PulseShape = Literal["sech", "gauss"]
"""The two pulse shapes the design numbers are defined for."""

NonlinearPhase = Literal["largest", "supplied_power", "linear"]
"""Choose the constituent, supplied-power or linear RDW phase model."""

Fill = Gas | str | Mapping[str, float] | Mixture
"""A fill as the run interfaces accept it: a species, mole fractions or a mixture."""

_SECH_WIDTH = 2.0 * math.log(1.0 + math.sqrt(2.0))
"""``fwhm / tau0`` of ``sech^2(t / tau0)``: twice the inverse cosh of the root of 2."""

_GAUSS_WIDTH = 2.0 * math.sqrt(math.log(2.0))
"""``fwhm / tau0`` of ``exp(-t^2 / tau0^2)``."""

_MISMATCH_SAMPLES = 2048
"""Frequency samples the phase-matching search brackets its roots on.

The mismatch of a hollow waveguide is smooth on the scale of the search window (its
structure is set by the zero-dispersion points, of which there are a handful), so two
roots closer than a two-thousandth of the window are not separated. The cost is one
vectorised index evaluation.
"""

_PRESSURE_FLOOR = 1e-6
"""Lower end of the pressure brackets in bar; a true vacuum has no gas term."""

_ZDW_RANGE = (100e-9, 3e-6)
"""Window of the zero-dispersion search in m: :func:`modes.base.zdw`'s."""

_ZDW_SAMPLES = 512
"""Samples, evenly spaced in frequency, on which the dispersion's sign is read."""

_POLE_FACTOR = 2.0
"""How far the group delay at a refined sign change of ``beta2`` may sit from its value
at the bracket's ends, in units of the change a bounded ``beta2`` could produce across
the bracket, before the sign change is read as a pole rather than a zero. At a zero
``beta1`` is extremal and moves by at most half that unit; at a gas resonance it
diverges. The differenced ``beta2`` itself cannot tell them apart: ``beta1`` is even
about a pole, so its central difference vanishes there too."""

_SEARCH_SAMPLES = 129
"""Linear samples of a parameter search, and logarithmic ones beside them where the
range is positive: :func:`pressure_rdw`'s grid, which :func:`solve_parameter` shares."""

_ATTAINABLE_SAMPLES = 17
"""Linear samples, and logarithmic ones beside them, on which a search without a root
reports the attainable range of its target's quantity. Coarser than the search: the
report rebuilds each mode, and the zero-dispersion search alone costs some tens of
index evaluations per mode. Both ends of the range are among them, which is where a
quantity monotonic in the parameter takes its extremes."""

_ROOT_TOLERANCE = 2e-12
"""Brent's absolute tolerance on a bracketed root, in units of the bracket's larger
end: the default ``xtol`` of :func:`scipy.optimize.brentq` (2e-12) made relative, so a
radius in m and a pressure in bar converge to the same number of digits."""

_ROOT_RESIDUAL = 1e-3
"""Largest residual at a refined sign change, as a fraction of its larger value at the
bracket's ends, for the change to be a root. Brent's method converges onto a pole as
readily as onto a zero; at a zero the residual it ends on is of the order of the slope
times the tolerance, many orders below the ends, and at a pole it diverges past both.
A parameter has no integral like ``beta1`` to test the change against (see
:data:`_POLE_FACTOR`), so the residual's own size decides."""


@contextmanager
def _probing() -> Iterator[None]:
    """Silence material-range and dilute-scaling warnings while a search samples.

    A root search evaluates the index across a window far wider than its answer --
    in frequency, or in pressure up to a hundred bar, where an optmat fill's linear
    density scaling departs from its finite-density law -- and the ends are probes,
    not results. The answer itself is re-evaluated outside this context, so a root
    that lies outside a model's range, or at a density beyond the dilute law,
    still warns.

    Yields
    ------
    None
        Control, with the warning categories ignored.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        warnings.simplefilter("ignore", DiluteScalingWarning)
        yield


def _positive(name: str, value: float) -> float:
    """Return a finite positive number or refuse it by name.

    Parameters
    ----------
    name : str
        The argument's name, for the message.
    value : float
        The value.

    Returns
    -------
    float
        The value as a float.

    Raises
    ------
    ToolsError
        If it is not finite and positive.
    """
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ToolsError(f"{name} must be finite and positive, got {value!r}")
    return number


def _omega(wavelength: float) -> float:
    """Return the angular frequency of a vacuum wavelength.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength in m, > 0.

    Returns
    -------
    float
        Angular frequency in rad/s.
    """
    return 2.0 * math.pi * C / _positive("wavelength", wavelength)


def _check_shape(shape: str) -> None:
    """Refuse a pulse shape the design numbers are not defined for.

    Parameters
    ----------
    shape : str
        The shape keyword.

    Raises
    ------
    ToolsError
        If it is neither ``"sech"`` nor ``"gauss"``.
    """
    if shape not in ("sech", "gauss"):
        raise ToolsError(f"shape must be 'sech' or 'gauss', got {shape!r}")


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
    _check_shape(shape)
    return _positive("fwhm", fwhm) / (_SECH_WIDTH if shape == "sech" else _GAUSS_WIDTH)


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
    return _positive("energy", energy) / (integral * width)


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
    return _positive("peak_power", peak_power) / pulse_peak_power(1.0, fwhm, shape)


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
    return 1.0 / (_positive("gamma", gamma) * _positive("peak_power", peak_power))


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
    return _positive("order", order) ** 2 / (
        _positive("gamma", gamma) * dispersion_length(fwhm, beta2, shape)
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
    """Evaluate the real Zakharov-Shabat phase equation for a Gaussian.

    Parameters
    ----------
    x : float
        Time divided by the input width.
    theta : numpy.ndarray
        Unwrapped scattering angle, one element.
    order : float
        Conventional soliton order, the field's peak.
    eta : float
        Trial imaginary eigenvalue.

    Returns
    -------
    list of float
        Angle derivative.
    """
    return [-order * math.exp(-x * x / 2.0) - eta * math.sin(2 * theta[0])]


def _gaussian_eigenvalue(order: float) -> float:
    """Solve the half-line scattering problem for the largest Gaussian eigenvalue.

    Parameters
    ----------
    order : float
        Conventional soliton order, above the first Gaussian threshold.

    Returns
    -------
    float
        Positive imaginary part of the dimensionless eigenvalue.

    Raises
    ------
    ToolsError
        If integration or eigenvalue refinement fails.
    """
    # The omitted field area is <= 1e-13. Moving the boundary with the amplitude
    # avoids assuming that the Gaussian tail is negligible for every input N.
    extent = math.sqrt(2.0) * float(erfcinv(1e-13 / (order * math.sqrt(math.pi / 2.0))))

    def residual(eta: float) -> float:
        """Return the ground-state symmetry residual at the pulse centre.

        Parameters
        ----------
        eta : float
            Trial imaginary eigenvalue.

        Returns
        -------
        float
            Phase angle plus pi/4.
        """
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


def self_focusing_power(wavelength: float, index: float, n2: float) -> float:
    r"""Return the critical power for self-focusing, as physical power.

    :math:`P_{\rm cr} = 1.86226\,\lambda^2/(4\pi n_0 n_2)`, the Townes power of Fibich
    and Gaeta (Opt. Lett. 25, 335 (2000)). It is
    :func:`solprop.spatial.critical_power` -- the package's one Townes constant --
    at the linear index, both measuring the physical axial power in the medium
    (SEP 34). The paper prints the constant as 1.86225; the
    computed value is 1.8622555, of which 1.86226 is the rounding.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength in m, > 0.
    index : float
        Linear refractive index :math:`n_0`, > 0.
    n2 : float
        Nonlinear index in m^2/W, > 0.

    Returns
    -------
    float
        Critical power in W.

    Examples
    --------
    >>> from solprop.tools import self_focusing_power
    >>> f"{self_focusing_power(800e-9, 1.0, 3.55e-25) / 1e9:.2f}"
    '267.17'
    """
    linear = _positive("index", index)
    chi3 = chi3_from_n2(_positive("n2", n2), linear)
    return spatial.critical_power(
        _positive("wavelength", wavelength), chi3, index=linear
    )


def field_to_intensity(field: float) -> float:
    r"""Return the intensity of a field amplitude, :math:`I = \epsilon_0 c E^2/2`.

    Parameters
    ----------
    field : float
        Field amplitude in V/m.

    Returns
    -------
    float
        Intensity in W/m^2.

    Examples
    --------
    >>> from solprop.tools import field_to_intensity
    >>> f"{field_to_intensity(1e10):.4e}"
    '1.3272e+17'
    """
    return 0.5 * EPS0 * C * float(field) ** 2


def intensity_to_field(intensity: float) -> float:
    r"""Return the field amplitude of an intensity, :math:`E = \sqrt{2I/\epsilon_0 c}`.

    Parameters
    ----------
    intensity : float
        Intensity in W/m^2, >= 0.

    Returns
    -------
    float
        Field amplitude in V/m.

    Raises
    ------
    ToolsError
        If the intensity is negative.

    Examples
    --------
    >>> from solprop.tools import field_to_intensity, intensity_to_field
    >>> f"{intensity_to_field(field_to_intensity(1e10)):.6e}"
    '1.000000e+10'
    """
    if not intensity >= 0.0:
        raise ToolsError(f"an intensity must not be negative, got {intensity!r}")
    return math.sqrt(2.0 * float(intensity) / (EPS0 * C))


# ---------------------------------------------------------------- soliton parameters


@dataclass(frozen=True)
class GasRatio:
    """Ratios of the bulk dispersion and the Kerr response of two gases.

    Attributes
    ----------
    beta2 : float
        Ratio of the bulk group-velocity dispersions, first gas over second.
    chi3 : float
        Ratio of the third-order susceptibilities, first gas over second.
    """

    beta2: float
    chi3: float


@dataclass(frozen=True, eq=False)
class SolitonParams(Displayed):
    """The design numbers of one pulse in one mode and one fill.

    Attributes
    ----------
    energy : float
        Pulse energy in J.
    fwhm : float
        Full width at half maximum of the power in s.
    width : float
        Natural width ``tau0`` of the shape in s (:func:`pulse_width`).
    omega : float
        Angular frequency of the pump in rad/s.
    wavelength : float
        Vacuum wavelength of the pump in m.
    gas : str
        The fill's species labels, joined by ``+``.
    pressure : float
        Total pressure at the position in bar; NaN for a mixture of independent
        populations, which has no single pressure.
    temperature : float
        Fill temperature in K.
    shape : {'sech', 'gauss'}
        ``"sech"`` or ``"gauss"``.
    peak_power : float
        Peak power in W.
    beta2 : float
        Group-velocity dispersion of the mode at the pump in s^2/m.
    density : float
        Total number density of the fill in m^-3.
    index : float
        Linear refractive index of the fill at the pump.
    n2 : float
        Nonlinear index of the fill in m^2/W.
    gamma : float
        Nonlinear coefficient in rad/(W m).
    soliton_order : float
        ``sqrt(L_D / L_NL)``.
    soliton_content : SolitonContent or None
        Unchirped NLSE constituent count and largest soliton; None for normal GVD.
    intensity : float
        Mode-averaged intensity ``peak_power / effective_area`` in W/m^2.
    peak_intensity : float
        On-axis intensity of the mode at the peak power in W/m^2: what a run's
        ``peak_intensity`` statistic reports at its first plane. For the fundamental
        of a capillary it is 1.77 times `intensity`. NaN for a mode without a
        transverse profile (a dispersion-specified one).
    critical_power : float
        Self-focusing power in W (:func:`self_focusing_power`).
    dispersion_length : float
        In m.
    nonlinear_length : float
        In m.
    fission_length : float
        In m.
    zdw : float
        Zero-dispersion wavelength in m within 100 nm to 3 um, or NaN when the
        dispersion does not change sign there. No second, guess-seeded search is
        tried.
    loss_length : float
        The 1/e **power** loss length of the mode at the pump in m.
    effective_area : float
        In m^2.
    mode : Mode
        The mode.
    """

    energy: float
    fwhm: float
    width: float
    omega: float
    wavelength: float
    gas: str
    pressure: float
    temperature: float
    shape: PulseShape
    peak_power: float
    beta2: float
    density: float
    index: float
    n2: float
    gamma: float
    soliton_order: float
    soliton_content: SolitonContent | None
    intensity: float
    peak_intensity: float
    critical_power: float
    dispersion_length: float
    nonlinear_length: float
    fission_length: float
    zdw: float
    loss_length: float
    effective_area: float
    mode: Mode

    def __repr__(self) -> str:
        """Return a one-line description: the pulse, the fill and the soliton order.

        Returns
        -------
        str
            For example ``SolitonParams(300 uJ, 10 fs at 800 nm, He 0.4 bar, N=2.23)``.
        """
        return line(
            "SolitonParams",
            [
                ("", f"{self.energy * 1e6:.4g} uJ"),
                ("", f"{self.fwhm * 1e15:.4g} fs at {self.wavelength * 1e9:.4g} nm"),
                ("", f"{self.gas} {self.pressure:.4g} bar"),
                ("N", f"{self.soliton_order:.3g}"),
            ],
        )

    def __str__(self) -> str:
        """Return the parameter summary, section by section.

        Returns
        -------
        str
            MODE, WAVEGUIDE, FILL, PULSE, DISPERSION, POWER, INTENSITY and SOLITON,
            in the customary units: ``n2`` in cm^2/W, intensities in W/cm^2.
        """
        zdw = (
            "none in 100-3000 nm"
            if math.isnan(self.zdw)
            else f"{self.zdw * 1e9:.1f} nm"
        )
        ratio = self.peak_power / self.critical_power
        sections = (
            ("MODE", repr(self.mode)),
            (
                "WAVEGUIDE",
                f"Aeff = {self.effective_area:.2e} m^2, "
                f"Lloss = {self.loss_length:.2e} m (1/e power)",
            ),
            (
                "FILL",
                f"{self.pressure:.3g} bar {self.gas} at {self.temperature:g} K, "
                f"Pcr = {self.critical_power:.2e} W,\n  "
                f"gamma = {self.gamma:.2e} /(W m), n2 = {self.n2 * 1e4:.2e} cm^2/W",
            ),
            (
                "PULSE",
                f"{self.energy:.3e} J, {self.fwhm:.3e} s @ "
                f"{self.wavelength * 1e9:.1f} nm (shape: {self.shape})",
            ),
            (
                "DISPERSION",
                f"{self.beta2:.3e} s^2/m @ {self.wavelength * 1e9:.1f} nm, ZDW = {zdw}",
            ),
            ("POWER", f"{self.peak_power:.2e} W ({ratio:.4f} of Pcr)"),
            (
                "INTENSITY",
                f"{self.intensity * 1e-4:.2e} W/cm^2 mode-averaged, "
                f"{self.peak_intensity * 1e-4:.2e} W/cm^2 on axis",
            ),
            (
                "SOLITON",
                f"Ld = {self.dispersion_length:.2e} m, Lnl = "
                f"{self.nonlinear_length:.2e} m, Lfiss = {self.fission_length:.2e} m, "
                f"N = {self.soliton_order:.2f}",
            ),
        )
        content = self.soliton_content
        if content is None:
            constituents = "unavailable (normal dispersion)"
        elif content.count == 0:
            constituents = "0 discrete solitons"
        else:
            constituents = (
                f"{content.count} discrete solitons, "
                f"area/pi = {content.area_order:.3g}\n  "
                f"largest P = {content.largest_peak_power:.3e} W, "
                f"T = {content.largest_width:.3e} s, "
                f"kNL = {content.largest_nonlinear_wavenumber:.3e} 1/m"
            )
        return "\n".join(f"{name}:\n  {text}" for name, text in sections) + (
            f"\nCONSTITUENTS (unchirped NLSE):\n  {constituents}"
        )


def _resolve_fill(
    gas: Fill,
    pressure: float | None,
    temperature: float | None,
    materials: MaterialSelection | None = None,
) -> Mixture:
    """Return the mixture a fill declaration means, as the run interfaces read it.

    Parameters
    ----------
    gas : Gas, str, Mapping or Mixture
        A built-in gas, ``"air"`` (nitrogen and oxygen), mole fractions, or a
        complete mixture.
    pressure : float or None
        Total pressure in bar; must be ``None`` beside a `Mixture`, which carries its
        own, and defaults to one bar otherwise.
    temperature : float or None
        Fill temperature in K; ``None`` is room temperature, or the mixture's own
        beside a `Mixture`, which refuses any other.
    materials : MaterialSelection or None, optional
        Resolve a name, a fraction mapping's names and ``"air"`` from optmat under
        this selection, as the drivers' ``capillary_fill`` does; ``None`` is
        ``MaterialSelection()``. A mixture is taken as it is.

    Returns
    -------
    Mixture
        The ordered populations.

    Raises
    ------
    ToolsError
        If a pressure, or a temperature other than its own, is given beside a
        mixture, or the mixture's temperature varies along z.
    """
    if isinstance(gas, Mixture):
        if pressure is not None:
            raise ToolsError("a Mixture carries its own pressure; omit pressure")
        if isinstance(gas.temperature, ZProfile):
            raise ToolsError(
                "the design calculators take one temperature in K, and this mixture's "
                "varies along z; give a Mixture at one temperature"
            )
        if temperature is not None and float(temperature) != gas.temperature:
            # As the run interfaces: the mixture's densities were set at its own
            # temperature, so another one cannot be honoured.
            raise ToolsError(
                f"the Mixture carries its own temperature ({gas.temperature:g} K), "
                f"and temperature={temperature!r} was given beside it: omit "
                "temperature=, or build the Mixture at the temperature you mean"
            )
        return gas
    temperature = ROOM_TEMPERATURE if temperature is None else temperature
    total = 1.0 if pressure is None else float(pressure)
    if isinstance(gas, Mapping):
        fractions = {
            species(name, selection=materials): fraction
            for name, fraction in gas.items()
        }
        return Mixture.fill(fractions, total, temperature=temperature)
    if gas is Gas.Air or (isinstance(gas, str) and gas.strip().lower() == "air"):
        return Mixture.air(
            total, temperature=temperature, argon=False, selection=materials
        )
    return Mixture.fill(
        {species(gas, selection=materials): 1.0}, total, temperature=temperature
    )


def _check_carrier(materials: MaterialSelection | None, wavelength: float) -> None:
    """Refuse a selection whose carrier is not the tool's pump wavelength.

    A run under a selection refuses a carrier other than its grid's reference
    wavelength, so a design tool refuses one other than the wavelength it evaluates
    the Kerr coefficient at: the two then evaluate it at the same carrier.

    Parameters
    ----------
    materials : MaterialSelection or None
        The design's material selection.
    wavelength : float
        The pump wavelength in m.

    Raises
    ------
    ToolsError
        If the selection names another carrier.
    """
    carrier = None if materials is None else materials.carrier_wavelength
    if carrier is not None and not carriers_agree(float(carrier), wavelength):
        msg = (
            f"the selection's carrier_wavelength is {carrier:g} m and the design's "
            f"wavelength {wavelength:g} m: one carrier sets the Kerr coefficient of "
            "every optmat species, as in a run. Leave the selection's "
            "carrier_wavelength unset, or design at it"
        )
        raise ToolsError(msg)


def _default_cladding(
    cladding: GlassIndex | complex | None,
) -> GlassIndex | complex:
    """Return the cladding a design's capillary takes, as the capillary driver does.

    Parameters
    ----------
    cladding : GlassIndex, complex or None
        The cladding given; ``None`` is the package default.

    Returns
    -------
    GlassIndex or complex
        A given cladding as it is; the default as optmat's recommended fused
        silica.
    """
    return fused_silica() if cladding is None else cladding


def _fill_chi3(mixture: Mixture, wavelength: float, z: float = 0.0) -> float:
    """Return a fill's density-weighted third-order susceptibility.

    Parameters
    ----------
    mixture : Mixture
        The fill.
    wavelength : float
        Vacuum wavelength in m, the carrier at which an optmat Kerr model is taken.
    z : float, optional
        Position in m at which the densities are evaluated.

    Returns
    -------
    float
        ``chi3`` in m^2/V^2.

    Raises
    ------
    ToolsError
        If a component of the fill has no Kerr data.
    """
    try:
        return float(mixture.gamma3_path(wavelength)(np.array([float(z)]))[0])
    except MaterialError as error:
        raise ToolsError(f"the fill has no nonlinear index: {error}") from error


def _fill_n2(
    mixture: Mixture, wavelength: float, z: float = 0.0
) -> tuple[float, float]:
    """Return a fill's nonlinear index and its linear index at a wavelength.

    Parameters
    ----------
    mixture : Mixture
        The fill.
    wavelength : float
        Vacuum wavelength in m, the carrier at which an optmat Kerr model is taken.
    z : float, optional
        Position in m at which the densities are evaluated.

    Returns
    -------
    tuple of float
        ``n2`` in m^2/W, from the density-weighted ``chi3`` and the exact ``n0^2`` of
        the fill, and that linear index ``n0``.

    Raises
    ------
    ToolsError
        If a component of the fill has no Kerr data.
    """
    chi3 = _fill_chi3(mixture, wavelength, z)
    here = np.array([float(z)])
    susceptibility = mixture.susceptibility(np.array([_omega(wavelength)]), here)[0, 0]
    index = float(np.real(np.sqrt(1.0 + susceptibility)))
    return n2_from_chi3(chi3, index), index


def _on_axis_intensity(mode: Mode, peak_power: float, z: float) -> float:
    """Return the on-axis intensity of a mode carrying a given power.

    Parameters
    ----------
    mode : Mode
        The mode.
    peak_power : float
        Power in W.
    z : float
        Position along the waveguide, m.

    Returns
    -------
    float
        Intensity on the axis in W/m^2, both transverse components included; NaN for
        a mode that declares no transverse profile.
    """
    # The same synthesis the run's peak_intensity statistic uses: the normalised mode
    # field at the origin in V/m per sqrt(W), so that eps0 c |e|^2 / 2 is the
    # intensity per watt.
    try:
        on_axis = mode_matrix(
            (mode,), on_axis_rule(mode.dimlimits(z)[0]), z, components=("x", "y")
        )
    except ModeError:
        # A dispersion-specified mode has an area and no transverse profile, so it has
        # a mode-averaged intensity and no on-axis one.
        return math.nan
    per_watt = BASIS_NORMALISATION * float(np.sum(np.abs(on_axis.synthesis) ** 2))
    return per_watt * peak_power


def _zdw_near(mode: Mode, omega: float, z: float) -> float:
    """Return the zero-dispersion wavelength nearest a pump frequency.

    :func:`solprop.modes.base.zdw` brackets one sign change across its whole window,
    which a gas resonance in the vacuum ultraviolet defeats: krypton flips the sign of
    the dispersion twice near 120 nm, the ends then agree, and the zero at 769 nm of a
    14 um capillary at 15 bar is refused. Here the sign is sampled, every change is
    refined, the changes through a pole are dropped, and the zero nearest the pump is
    the one reported. Two limits of a sampled search remain: a zero and a pole inside
    one sample interval (3.6e13 rad/s) restore the sign and are both missed, and a
    sample that is not finite cannot bracket anything.

    Parameters
    ----------
    mode : Mode
        The mode.
    omega : float
        Pump angular frequency in rad/s.
    z : float
        Position along the waveguide, m.

    Returns
    -------
    float
        Wavelength in m, or NaN when the dispersion has no zero in 100 nm to 3 um.
    """
    grid = np.linspace(
        2.0 * math.pi * C / _ZDW_RANGE[1],
        2.0 * math.pi * C / _ZDW_RANGE[0],
        _ZDW_SAMPLES,
    )
    zeros = []
    # Brent's method converges onto a pole as readily as onto a zero, and a model may
    # divide by zero exactly there; the group-delay test below tells the two apart.
    with _probing(), np.errstate(divide="ignore", invalid="ignore"):
        sampled = base.beta2(mode, grid, z)
        for i in np.flatnonzero(np.sign(sampled[:-1]) * np.sign(sampled[1:]) < 0.0):
            low, high = float(grid[i]), float(grid[i + 1])
            root = float(
                brentq(
                    lambda value: float(base.beta2(mode, np.array([value]), z)[0]),
                    low,
                    high,
                )
            )
            delay = base.beta1(mode, np.array([low, root, high]), z)
            bounded = max(abs(sampled[i]), abs(sampled[i + 1])) * (high - low)
            if abs(delay[1] - delay[0]) <= _POLE_FACTOR * bounded:
                zeros.append(root)
    if not zeros:
        return math.nan
    nearest = min(zeros, key=lambda value: abs(value - omega))
    # Outside the probing context, so a zero beyond the index model's range warns.
    base.beta(mode, np.array([nearest]), z)
    return 2.0 * math.pi * C / nearest


def soliton_params(  # noqa: PLR0913 - a pulse (three numbers and a shape) and a fill
    mode: Mode,
    gas: Fill,
    pressure: float | None = None,
    *,
    energy: float,
    fwhm: float,
    wavelength: float,
    shape: PulseShape = "sech",
    temperature: float | None = None,
    z: float = 0.0,
    materials: MaterialSelection | None = None,
) -> SolitonParams:
    """Return the design numbers of a pulse in any mode.

    The mode supplies the dispersion, the area and the loss; the fill supplies the
    nonlinear index. They are separate arguments because a mode need not know its
    fill's Kerr response (a tabulated mode does not), and the caller is trusted to
    give the fill the mode was built with -- :func:`capillary_params` does both.

    Parameters
    ----------
    mode : Mode
        Any mode: a capillary, an antiresonant or tabulated fibre, a rectangular
        guide.
    gas : Gas, str, Mapping or Mixture
        The fill: a registry species, ``"air"``, mole fractions, or a mixture. A
        mixture of species selected from optmat takes their Kerr coefficients at
        `wavelength`, the carrier.
    pressure : float or None, optional
        Total pressure in **bar**; ``None`` is one bar, and is required beside a
        `Mixture`.
    energy : float
        Pulse energy in J, > 0.
    fwhm : float
        Full width at half maximum of the power in s, > 0.
    wavelength : float
        Pump wavelength in m, > 0.
    shape : {'sech', 'gauss'}, optional
        The pulse shape, used consistently for the width and the peak power.
    temperature : float or None, optional
        Fill temperature in K; ``None`` is room temperature, or the mixture's own
        beside a `Mixture`, which refuses any other.
    z : float, optional
        Position along the waveguide at which everything is evaluated, m.
    materials : MaterialSelection or None, optional
        Resolve a named fill from optmat under this selection, as a run's
        ``materials=`` does: its index, density and Kerr coefficient at
        `wavelength`. ``None`` is ``MaterialSelection()``; a `Mixture` is taken as
        it is. The mode is the caller's either way.

    Returns
    -------
    SolitonParams
        The numbers, with the mode.

    Raises
    ------
    ToolsError
        If a quantity is not positive, the shape is unknown, the mode has no
        dispersion at the pump, a component of the fill has no Kerr data, the
        selection names a carrier other than `wavelength`, or a pressure or a
        temperature other than its own is given beside a `Mixture`.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.tools import soliton_params
    >>> mode = MarcatiliMode(125e-6, "He", 0.4, model="reduced", cladding=1.45)
    >>> p = soliton_params(mode, "He", 0.4, energy=300e-6, fwhm=10e-15,
    ...                    wavelength=800e-9)
    >>> f"{p.soliton_order:.2f} {p.fission_length:.2f}"
    '2.26 1.75'
    """
    materials = MaterialSelection() if materials is None else materials
    _check_carrier(materials, wavelength)
    mixture = _resolve_fill(gas, pressure, temperature, materials)
    omega = _omega(wavelength)
    here = np.array([float(z)])
    n2, index = _fill_n2(mixture, wavelength, float(z))
    peak_power = pulse_peak_power(energy, fwhm, shape)
    beta2 = float(base.beta2(mode, np.array([omega]), z)[0])
    gamma = nonlinear_coefficient(mode, n2, wavelength, z)
    area = base.aeff(mode, z)
    zdw = _zdw_near(mode, omega, float(z))
    shared = mixture.shared
    return SolitonParams(
        energy=float(energy),
        fwhm=float(fwhm),
        width=pulse_width(fwhm, shape),
        omega=omega,
        wavelength=float(wavelength),
        gas="+".join(mixture.labels),
        pressure=math.nan if shared is None else float(shared(here)[0]),
        # _resolve_fill refused a temperature that varies along z.
        temperature=cast("float", mixture.temperature),
        shape=shape,
        peak_power=peak_power,
        beta2=beta2,
        density=float(mixture.total_density(here)[0]),
        index=index,
        n2=n2,
        gamma=gamma,
        soliton_order=soliton_order(peak_power, fwhm, gamma, beta2, shape),
        soliton_content=(
            soliton_content(peak_power, fwhm, gamma, beta2, shape)
            if beta2 < 0
            else None
        ),
        intensity=peak_power / area,
        peak_intensity=_on_axis_intensity(mode, peak_power, float(z)),
        critical_power=self_focusing_power(wavelength, index, n2),
        dispersion_length=dispersion_length(fwhm, beta2, shape),
        nonlinear_length=nonlinear_length(peak_power, gamma),
        fission_length=fission_length(peak_power, fwhm, gamma, beta2, shape),
        zdw=zdw,
        loss_length=float(base.loss_length(mode, np.array([omega]), z)[0]),
        effective_area=area,
        mode=mode,
    )


def _capillary(  # noqa: PLR0913 - the capillary's keywords
    radius: float,
    mixture: Mixture,
    *,
    n: int,
    m: int,
    kind: ModeKind,
    model: MarcatiliModel,
    loss: LossSpec,
    cladding: GlassIndex | complex | None,
) -> MarcatiliMode:
    """Build the HE, TE or TM capillary the capillary design tools report on.

    Parameters
    ----------
    radius : float
        Core radius in m, > 0.
    mixture : Mixture
        The resolved fill.
    n : int
        Azimuthal mode index.
    m : int
        Radial mode index, >= 1.
    kind : {'HE', 'TE', 'TM'}
        Mode family.
    model : {'full', 'reduced'}
        Marcatili model.
    loss : bool, float or callable
        Loss policy of the mode.
    cladding : GlassIndex, complex or None
        Cladding index; ``None`` is fused silica.

    Returns
    -------
    MarcatiliMode
        The mode, as `prop_capillary` builds it for the same arguments.
    """
    return MarcatiliMode(
        _positive("radius", radius),
        mixture=mixture,
        n=n,
        m=m,
        kind=kind,
        model=model,
        loss=loss,
        cladding=_default_cladding(cladding),
    )


def capillary_params(  # noqa: PLR0913 - the capillary's keywords beside the pulse's
    radius: float,
    gas: Fill,
    pressure: float | None = None,
    *,
    energy: float,
    fwhm: float,
    wavelength: float,
    shape: PulseShape = "sech",
    temperature: float | None = None,
    n: int = 1,
    m: int = 1,
    kind: ModeKind = "HE",
    model: MarcatiliModel = "full",
    loss: LossSpec = True,
    cladding: GlassIndex | complex | None = None,
    materials: MaterialSelection | None = None,
) -> SolitonParams:
    """Return the design numbers of a pulse in a gas-filled capillary.

    Builds the mode as :func:`solprop.driver.interface.prop_capillary` does for the same
    arguments -- a :class:`~solprop.modes.capillary.MarcatiliMode` over the fill's
    mixture -- and hands it to :func:`soliton_params`.

    Parameters
    ----------
    radius : float
        Core radius in m, > 0.
    gas : Gas, str, Mapping or Mixture
        The fill, as `prop_capillary` takes it.
    pressure : float or None, optional
        Total pressure in **bar**; ``None`` is one bar. For a gradient, evaluate at
        the pressure of the position of interest.
    energy : float
        Pulse energy in J, > 0.
    fwhm : float
        Full width at half maximum of the power in s, > 0.
    wavelength : float
        Pump wavelength in m, > 0.
    shape : {'sech', 'gauss'}, optional
        The pulse shape.
    temperature : float or None, optional
        Fill temperature in K; ``None`` is room temperature, or the mixture's own
        beside a `Mixture`, which refuses any other.
    n : int, optional
        Azimuthal mode index.
    m : int, optional
        Radial mode index, >= 1.
    kind : {'HE', 'TE', 'TM'}, optional
        Mode family.
    model : {'full', 'reduced'}, optional
        Marcatili model.
    loss : bool, float or callable, optional
        Loss policy of the mode.
    cladding : GlassIndex, complex or None, optional
        Cladding index; ``None`` is fused silica.
    materials : MaterialSelection or None, optional
        Build the capillary as `prop_capillary` does under the same ``materials=``:
        the fill from optmat (index, density and Kerr coefficient at `wavelength`)
        and ``cladding=None`` as optmat's recommended fused silica. ``None`` is
        ``MaterialSelection()``.

    Returns
    -------
    SolitonParams
        The numbers, with the mode that was built.

    Examples
    --------
    >>> from solprop.tools import capillary_params
    >>> p = capillary_params(125e-6, "He", 0.4, energy=300e-6, fwhm=10e-15,
    ...                      wavelength=800e-9, model="reduced", cladding=1.45)
    >>> f"{p.gamma:.3e} {p.intensity:.3e}"
    '4.884e-11 1.130e+18'
    """
    materials = MaterialSelection() if materials is None else materials
    _check_carrier(materials, wavelength)
    mixture = _resolve_fill(gas, pressure, temperature, materials)
    mode = _capillary(
        radius,
        mixture,
        n=n,
        m=m,
        kind=kind,
        model=model,
        loss=loss,
        cladding=cladding,
    )
    return soliton_params(
        mode, mixture, energy=energy, fwhm=fwhm, wavelength=wavelength, shape=shape
    )


def _area_per_order(shape: PulseShape) -> float:
    """Return the pulse area over pi per unit conventional soliton order.

    Parameters
    ----------
    shape : {'sech', 'gauss'}
        The pulse shape.

    Returns
    -------
    float
        One for sech, ``sqrt(2 / pi)`` for a Gaussian (:func:`soliton_content`).
    """
    return 1.0 if shape == "sech" else math.sqrt(2.0 / math.pi)


def _gaussian_order(eigenvalue: float) -> float:
    """Return the conventional order whose Gaussian has a given largest eigenvalue.

    Parameters
    ----------
    eigenvalue : float
        Target imaginary part of the largest Zakharov-Shabat eigenvalue, > 0.

    Returns
    -------
    float
        The conventional soliton order.

    Raises
    ------
    ToolsError
        If the eigenvalue solve fails or the bracket does not hold the root.
    """
    threshold = 0.5 / _area_per_order("gauss")

    def residual(order: float) -> float:
        """Return the largest eigenvalue at an order less the target.

        Parameters
        ----------
        order : float
            Conventional soliton order.

        Returns
        -------
        float
            The difference, dimensionless.
        """
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


def _target_order(
    order: float | None,
    count: int | None,
    eigenvalue: float | None,
    shape: PulseShape,
) -> float:
    """Return the conventional soliton order that meets exactly one design target.

    Parameters
    ----------
    order : float or None
        A conventional soliton order, > 0.
    count : int or None
        A number of solitons, >= 1.
    eigenvalue : float or None
        The largest Zakharov-Shabat eigenvalue, > 0.
    shape : {'sech', 'gauss'}
        The pulse shape.

    Returns
    -------
    float
        The conventional soliton order.

    Raises
    ------
    ToolsError
        If not exactly one target is given, or it is out of range.
    """
    given = [
        name
        for name, value in (
            ("soliton_order", order),
            ("soliton_count", count),
            ("largest_eigenvalue", eigenvalue),
        )
        if value is not None
    ]
    if len(given) != 1:
        raise ToolsError(
            "give exactly one of soliton_order, soliton_count and largest_eigenvalue, "
            f"got {', '.join(given) or 'none'}"
        )
    if order is not None:
        return _positive("soliton_order", order)
    if count is not None:
        if count < 1:
            raise ToolsError(f"soliton_count must be at least 1, got {count}")
        # The count is M = ceil(S - 1/2) for an area S (soliton_content), so the
        # pulses with M solitons have S in (M - 1/2, M + 1/2]. The design takes the
        # middle, S = M: the pure M-soliton for sech, the farthest from either
        # threshold for a Gaussian.
        return count / _area_per_order(shape)
    eta = _positive("largest_eigenvalue", cast("float", eigenvalue))
    return eta + 0.5 if shape == "sech" else _gaussian_order(eta)


def capillary_design(  # noqa: PLR0913 - the capillary's keywords beside the pulse's and targets
    radius: float,
    gas: Gas | str,
    *,
    zdw_wavelength: float,
    fwhm: float,
    wavelength: float,
    soliton_order: float | None = None,
    soliton_count: int | None = None,
    largest_eigenvalue: float | None = None,
    shape: PulseShape = "sech",
    temperature: float = ROOM_TEMPERATURE,
    m: int = 1,
    model: MarcatiliModel = "full",
    loss: LossSpec = True,
    cladding: GlassIndex | complex | None = None,
    materials: MaterialSelection | None = None,
    pressure_max: float = 100.0,
) -> SolitonParams:
    r"""Return the pressure and energy that give a capillary a soliton design.

    The inverse of :func:`capillary_params`: the pressure puts the zero-dispersion
    point at `zdw_wavelength` (:func:`pressure_zdw`), and the energy then gives the
    pulse one soliton target. At that pressure :math:`\beta_2` and :math:`\gamma` do
    not depend on the energy and :math:`N^2 = \gamma P_0\tau_0^2/|\beta_2|`, so the
    peak power is :math:`P_0 = N^2|\beta_2|/(\gamma\tau_0^2)`. Give exactly one
    target:

    - `soliton_order`, the conventional :math:`N`;
    - `soliton_count`, a number :math:`M` of solitons in the scattering content
      (:func:`soliton_content`): the pulse area is set to :math:`M\pi`, the middle
      of the band of areas with :math:`M` solitons, which is :math:`N = M` for
      sech and :math:`N = M\sqrt{\pi/2}` for a Gaussian;
    - `largest_eigenvalue`, the :math:`\eta` of the largest constituent, which
      fixes its width :math:`\tau_0/(2\eta)`: :math:`N = \eta + 1/2` for sech, and
      a root of the Gaussian scattering problem otherwise.

    Parameters
    ----------
    radius : float
        Core radius in m, > 0.
    gas : Gas or str
        A single built-in gas, as :func:`pressure_zdw` takes it.
    zdw_wavelength : float
        Wavelength at which the dispersion should vanish, m.
    fwhm : float
        Full width at half maximum of the power in s, > 0.
    wavelength : float
        Pump wavelength in m, > 0.
    soliton_order : float or None, optional
        Target conventional soliton order, > 0.
    soliton_count : int or None, optional
        Target number of solitons, >= 1.
    largest_eigenvalue : float or None, optional
        Target imaginary part of the largest Zakharov-Shabat eigenvalue,
        dimensionless with time in units of the shape's natural width, > 0.
    shape : {'sech', 'gauss'}, optional
        The pulse shape, used for the width, the peak power and the content.
    temperature : float, optional
        Fill temperature in K.
    m : int, optional
        Radial index of the HE_1m mode.
    model : {'full', 'reduced'}, optional
        Marcatili model.
    loss : bool, float or callable, optional
        Loss policy of the mode in the returned numbers.
    cladding : GlassIndex, complex or None, optional
        Cladding index; ``None`` is fused silica.
    materials : MaterialSelection or None, optional
        Design under this selection, as :func:`capillary_params` and
        :func:`pressure_zdw` do. ``None`` is ``MaterialSelection()``.
    pressure_max : float, optional
        Upper end of the pressure bracket in **bar**.

    Returns
    -------
    SolitonParams
        The numbers at the designed pressure and energy, with the mode; the
        pressure is ``.pressure`` (bar) and the energy ``.energy`` (J).

    Raises
    ------
    ToolsError
        If not exactly one target is given or it is out of range, no pressure in the
        bracket gives the zero-dispersion point, or a content target is asked of a
        pump in normal dispersion.

    Examples
    --------
    >>> from solprop.tools import capillary_design
    >>> p = capillary_design(125e-6, "Ar", zdw_wavelength=500e-9, fwhm=10e-15,
    ...                      wavelength=800e-9, soliton_order=3.0, cladding=1.45)
    >>> f"{p.pressure:.4f} bar {p.energy * 1e6:.1f} uJ N = {p.soliton_order:.3f}"
    '0.0664 bar 100.5 uJ N = 3.000'
    >>> g = capillary_design(125e-6, "Ar", zdw_wavelength=500e-9, fwhm=10e-15,
    ...                      wavelength=800e-9, soliton_count=2, shape="gauss",
    ...                      cladding=1.45)
    >>> f"{g.soliton_order:.4f} {g.soliton_content.count}"
    '2.5066 2'
    """
    materials = MaterialSelection() if materials is None else materials
    _check_carrier(materials, wavelength)
    _check_shape(shape)
    order = _target_order(soliton_order, soliton_count, largest_eigenvalue, shape)
    pressure = pressure_zdw(
        radius,
        gas,
        zdw_wavelength,
        pressure_max=pressure_max,
        temperature=temperature,
        m=m,
        model=model,
        cladding=cladding,
        materials=materials,
    )
    mixture = _resolve_fill(gas, pressure, temperature, materials)
    mode = _capillary(
        radius,
        mixture,
        n=1,
        m=m,
        kind="HE",
        model=model,
        loss=loss,
        cladding=cladding,
    )
    beta2 = float(base.beta2(mode, np.array([_omega(wavelength)]))[0])
    if soliton_order is None and beta2 >= 0.0:
        raise ToolsError(
            f"a soliton count or eigenvalue needs anomalous dispersion at the pump, "
            f"but beta2 = {beta2:.4g} s^2/m at {wavelength * 1e9:.4g} nm with the "
            f"zero-dispersion point at {zdw_wavelength * 1e9:.4g} nm"
        )
    n2, _ = _fill_n2(mixture, wavelength, 0.0)
    gamma = nonlinear_coefficient(mode, n2, wavelength)
    peak_power = soliton_peak_power(order, fwhm, gamma, beta2, shape)
    energy = pulse_energy(peak_power, fwhm, shape)
    return soliton_params(
        mode, mixture, energy=energy, fwhm=fwhm, wavelength=wavelength, shape=shape
    )


def gas_ratio(
    gas_a: Gas | str,
    gas_b: Gas | str,
    wavelength: float,
    *,
    pressure: float = 1.0,
    temperature: float = ROOM_TEMPERATURE,
    materials: MaterialSelection | None = None,
) -> GasRatio:
    """Return the ratios of bulk dispersion and Kerr response of two gases.

    The bulk (unguided) dispersion of each gas, not a mode's: it answers "how much
    more pressure of B matches A", which is a property of the gases alone.

    Parameters
    ----------
    gas_a : Gas or str
        The numerator's species.
    gas_b : Gas or str
        The denominator's species.
    wavelength : float
        Wavelength in m at which the dispersions are compared, > 0.
    pressure : float, optional
        Pressure of both gases in **bar**, > 0.
    temperature : float, optional
        Temperature of both in K.
    materials : MaterialSelection or None, optional
        Take both gases from optmat under this selection, as a run's ``materials=``
        does: their index, their density and their Kerr coefficient at
        `wavelength`; ``"air"`` is nitrogen and oxygen. ``None`` is
        ``MaterialSelection()``.

    Returns
    -------
    GasRatio
        Named ``beta2`` and ``chi3`` ratios.

    Raises
    ------
    ToolsError
        If the selection names a carrier other than `wavelength`, or a gas has no
        Kerr data under it.

    Examples
    --------
    >>> from solprop.tools import gas_ratio
    >>> ratio = gas_ratio("Ar", "He", 800e-9)
    >>> f"{ratio.beta2:.1f} {ratio.chi3:.1f}"
    '18.6 28.4'
    """
    omega = _omega(wavelength)
    materials = MaterialSelection() if materials is None else materials
    _check_carrier(materials, wavelength)
    dispersions = []
    susceptibilities = []
    for gas in (gas_a, gas_b):
        fill = _resolve_fill(
            gas, _positive("pressure", pressure), temperature, materials
        )
        dispersions.append(_bulk_beta2(_BulkGas(fill), omega))
        susceptibilities.append(_fill_chi3(fill, wavelength))
    return GasRatio(
        beta2=dispersions[0] / dispersions[1],
        chi3=susceptibilities[0] / susceptibilities[1],
    )


@dataclass(frozen=True)
class _BulkGas:
    """A uniform gas as an index model, for bulk dispersion.

    Attributes
    ----------
    fill : Mixture
        The gas resolved under a material selection, whose index is taken.
    """

    fill: Mixture

    def excess_beta(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Return the gas's part of the bulk propagation constant.

        The vacuum's ``omega / c`` has no second derivative, and leaving it out of a
        difference quotient removes its rounding, which at a gas's ``n - 1`` of 1e-5 is
        the larger part of the quotient.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s.

        Returns
        -------
        numpy.ndarray
            ``omega (n - 1) / c`` in 1/m, with ``n - 1`` formed without cancellation.
        """
        chi = self.fill.susceptibility(omega, np.zeros(1))[0]
        excess = chi / (1.0 + np.sqrt(1.0 + chi))
        return np.asarray(omega * np.real(excess) / C)


def _bulk_beta2(bulk: _BulkGas, omega: float) -> float:
    """Return the bulk group-velocity dispersion by a Richardson central difference.

    Parameters
    ----------
    bulk : _BulkGas
        The gas.
    omega : float
        Angular frequency in rad/s.

    Returns
    -------
    float
        ``d^2 beta / d omega^2`` in s^2/m.
    """
    # A second difference of the gas's part of beta: only a ratio of two such numbers is
    # used, so the few parts in 1e5 this reaches at the step modes.base chose are ample.
    step = base.BETA2_RELATIVE_STEP * omega

    def second(h: float) -> float:
        """Return the three-point second difference at step ``h``.

        Parameters
        ----------
        h : float
            Step in rad/s.

        Returns
        -------
        float
            The difference quotient in s^2/m.
        """
        values = bulk.excess_beta(np.array([omega - h, omega, omega + h]))
        return float(values[0] - 2.0 * values[1] + values[2]) / h**2

    return (4.0 * second(step / 2.0) - second(step)) / 3.0


# -------------------------------------------------------------------- phase matching


def _check_phase(nonlinear_phase: str) -> None:
    """Refuse a nonlinear phase model the phase matching does not define.

    Parameters
    ----------
    nonlinear_phase : str
        The phase keyword.

    Raises
    ------
    ToolsError
        If it is not ``"largest"``, ``"supplied_power"`` or ``"linear"``.
    """
    if nonlinear_phase not in ("largest", "supplied_power", "linear"):
        raise ToolsError(
            "nonlinear_phase must be 'largest', 'supplied_power' or 'linear', "
            f"got {nonlinear_phase!r}"
        )


@dataclass(frozen=True)
class _PhasePulse:
    """Validate the pulse declaration before a phase-matching search."""

    peak_power: float | None
    gamma: float | None
    fwhm: float | None
    shape: PulseShape
    nonlinear_phase: NonlinearPhase

    def __post_init__(self) -> None:
        """Refuse incomplete or contradictory phase declarations.

        Raises
        ------
        ToolsError
            If the phase choice or pulse declaration is invalid.
        """
        _check_shape(self.shape)
        _check_phase(self.nonlinear_phase)
        if (self.peak_power is None) != (self.gamma is None):
            raise ToolsError(
                "the nonlinear term needs both peak_power and gamma; "
                "give both or neither"
            )
        if self.nonlinear_phase == "linear" and (
            self.peak_power is not None or self.fwhm is not None
        ):
            raise ToolsError(
                "nonlinear_phase='linear' requires omitting pulse power, gamma and fwhm"
            )
        if self.peak_power is None:
            if self.fwhm is not None:
                raise ToolsError("fwhm requires peak_power and gamma")
            return
        _positive("peak_power", self.peak_power)
        assert self.gamma is not None  # The paired declaration was validated above.
        _positive("gamma", self.gamma)
        if self.nonlinear_phase == "largest":
            if self.fwhm is None:
                raise ToolsError(
                    "the largest constituent needs fwhm; supply it or select "
                    "nonlinear_phase='supplied_power'"
                )
            _positive("fwhm", self.fwhm)
        elif self.fwhm is not None:
            raise ToolsError(
                "nonlinear_phase='supplied_power' does not use fwhm; omit it"
            )

    def wavenumber(self, mode: Mode, omega: float, z: float) -> float:
        """Calculate the nonlinear phase in the soliton mode.

        Parameters
        ----------
        mode : Mode
            Mode carrying the soliton.
        omega : float
            Pump angular frequency in rad/s.
        z : float
            Position in m.

        Returns
        -------
        float
            Nonlinear wavenumber in 1/m.

        Raises
        ------
        ToolsError
            If the largest-constituent model has no discrete soliton.
        """
        if self.peak_power is None or self.gamma is None:
            return 0.0
        if self.fwhm is None:
            return 0.5 * self.gamma * self.peak_power
        dispersion = float(base.beta2(mode, np.array([omega]), z)[0])
        content = soliton_content(
            self.peak_power, self.fwhm, self.gamma, dispersion, self.shape
        )
        if content.largest_nonlinear_wavenumber is None:
            raise ToolsError(
                "the pulse has no discrete soliton for "
                "largest-constituent phase matching"
            )
        return content.largest_nonlinear_wavenumber


@dataclass(frozen=True)
class _PhaseMismatch:
    """Evaluate a mismatch with the pump and nonlinear phase prepared once."""

    mode: Mode
    omega0: float
    beta0: float
    slope: float
    nonlinear: float
    z: float

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Evaluate the mismatch at trial frequencies.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s.

        Returns
        -------
        numpy.ndarray
            Mismatch in 1/m.
        """
        return np.asarray(
            base.beta(self.mode, omega, self.z)
            - self.beta0
            - self.slope * (omega - self.omega0)
            - self.nonlinear
        )


def _prepare_mismatch(
    mode: Mode,
    wavelength: float,
    soliton_mode: Mode | None,
    z: float,
    pulse: _PhasePulse,
) -> _PhaseMismatch:
    """Prepare a phase-matching calculation at one position.

    Parameters
    ----------
    mode : Mode
        Dispersive-wave mode.
    wavelength : float
        Pump wavelength in m.
    soliton_mode : Mode or None
        Pump mode, or None to use mode.
    z : float
        Position in m.
    pulse : _PhasePulse
        Validated pulse and phase choice.

    Returns
    -------
    _PhaseMismatch
        Callable with the nonlinear phase evaluated once.
    """
    soliton = mode if soliton_mode is None else soliton_mode
    omega0 = _omega(wavelength)
    pump = np.array([omega0])
    return _PhaseMismatch(
        mode,
        omega0,
        float(base.beta(soliton, pump, z)[0]),
        float(base.beta1(soliton, pump, z)[0]),
        pulse.wavenumber(soliton, omega0, z),
        z,
    )


def phase_mismatch(  # noqa: PLR0913 - two modes, a position and the two Kerr numbers
    mode: Mode,
    omega: Float[np.ndarray, " n_omega"],
    wavelength: float,
    *,
    soliton_mode: Mode | None = None,
    z: float = 0.0,
    peak_power: float | None = None,
    gamma: float | None = None,
    fwhm: float | None = None,
    shape: PulseShape = "sech",
    nonlinear_phase: NonlinearPhase = "largest",
) -> Float[np.ndarray, " n_omega"]:
    r"""Return the phase mismatch between a soliton and a dispersive wave.

    .. math::
       \Delta\beta(\omega) = \beta_{\rm dw}(\omega) - \beta_{\rm s}(\omega_0)
       - \beta_{1,\rm s}(\omega_0)(\omega-\omega_0) - k_{\rm NL} .

    Parameters
    ----------
    mode : Mode
        The mode the dispersive wave is emitted into.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.
    wavelength : float
        Wavelength of the soliton in m, > 0.
    soliton_mode : Mode or None, optional
        The mode the soliton travels in; ``None`` is `mode`. A different one gives
        intermodal phase matching.
    z : float, optional
        Position along the waveguide, m.
    peak_power : float or None, optional
        Input peak power in W. With `gamma` and `fwhm`, adds the soliton's nonlinear
        wavenumber; both ``None`` (the default) gives the linear condition.
    gamma : float or None, optional
        Nonlinear coefficient of the soliton's mode in rad/(W m)
        (:func:`nonlinear_coefficient`).
    fwhm : float or None, optional
        Input power FWHM in s, positive; required with power for the default phase.
    shape : {'sech', 'gauss'}, optional
        Unchirped input shape for the largest constituent.
    nonlinear_phase : {'largest', 'supplied_power', 'linear'}, optional
        Default uses the largest constituent. Supplied power uses gamma P0/2 and
        requires omitting fwhm. Linear requires omitting all pulse numbers.
        Without pulse numbers the default is also linear.

    Returns
    -------
    numpy.ndarray
        Mismatch in 1/m, shape ``(n_omega,)``. It vanishes at the pump in the linear
        case, with zero slope there.

    Raises
    ------
    ToolsError
        If pulse arguments are incomplete or contradictory, or the largest
        constituent does not exist in anomalous dispersion.

    Notes
    -----
    The default nonlinear term is gamma times the largest constituent's power / 2
    (:func:`soliton_content`). The supplied-power option treats the quoted power as
    a fundamental soliton, advancing as exp(i gamma P0 z / 2) (Akhmediev and
    Karlsson, Phys. Rev. A 51, 2602 (1995)). Both are local estimates; loss,
    higher-order dispersion, Raman, chirp and plasma evolution are not included.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.tools import phase_mismatch
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> pump = 2 * np.pi * 299792458.0 / 800e-9
    >>> abs(float(phase_mismatch(mode, np.array([pump]), 800e-9)[0])) < 1e-6
    True
    """
    pulse = _PhasePulse(peak_power, gamma, fwhm, shape, nonlinear_phase)
    prepared = _prepare_mismatch(mode, wavelength, soliton_mode, z, pulse)
    return prepared(np.asarray(omega, dtype=float))


def lambda_rdw_all(  # noqa: PLR0913 - phase_mismatch's keywords and the search window
    mode: Mode,
    wavelength: float,
    *,
    soliton_mode: Mode | None = None,
    z: float = 0.0,
    peak_power: float | None = None,
    gamma: float | None = None,
    fwhm: float | None = None,
    shape: PulseShape = "sech",
    nonlinear_phase: NonlinearPhase = "largest",
    wavelength_limits: tuple[float, float] | None = None,
) -> tuple[float, ...]:
    """Return every phase-matched wavelength in a window, shortest first.

    Parameters
    ----------
    mode : Mode
        The mode the dispersive wave is emitted into.
    wavelength : float
        Wavelength of the soliton in m, > 0.
    soliton_mode : Mode or None, optional
        The soliton's mode; ``None`` is `mode`.
    z : float, optional
        Position along the waveguide, m.
    peak_power : float or None, optional
        Peak power in W for the nonlinear term; see :func:`phase_mismatch`.
    gamma : float or None, optional
        Nonlinear coefficient in rad/(W m) for the nonlinear term.
    fwhm : float or None, optional
        Input power FWHM in s, positive; required with power for the default phase.
    shape : {'sech', 'gauss'}, optional
        Unchirped input shape for the largest constituent.
    nonlinear_phase : {'largest', 'supplied_power', 'linear'}, optional
        Default uses the largest constituent. Supplied power uses gamma P0/2 and
        requires omitting fwhm. Linear requires omitting all pulse numbers.
        Without pulse numbers the default is also linear.
    wavelength_limits : tuple of float or None, optional
        Shortest and longest wavelength searched, m. ``None`` is 100 nm to 0.9 of the
        pump wavelength, which keeps the double root at the pump out.

    Returns
    -------
    tuple of float
        The roots in m, ascending; empty when there is none. The mismatch is sampled
        on 2048 frequencies across the window, each sign change is refined with
        Brent's method, and roots closer than one sample are not separated.

    Raises
    ------
    ToolsError
        If the window is not positive and ordered.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.tools import lambda_rdw_all
    >>> mode = MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)
    >>> [f"{root * 1e9:.0f}" for root in lambda_rdw_all(mode, 800e-9)]
    ['200']
    """
    pulse = _PhasePulse(peak_power, gamma, fwhm, shape, nonlinear_phase)
    prepared = _prepare_mismatch(mode, wavelength, soliton_mode, z, pulse)
    limits = (
        (100e-9, 0.9 * wavelength) if wavelength_limits is None else wavelength_limits
    )
    shortest, longest = limits
    if not 0.0 < shortest < longest:
        raise ToolsError(
            f"the wavelength limits must be positive and ordered: {limits}"
        )
    grid = np.linspace(
        2.0 * math.pi * C / longest, 2.0 * math.pi * C / shortest, _MISMATCH_SAMPLES
    )

    def mismatch(value: float) -> float:
        """Return the mismatch at one frequency, for the root finder.

        Parameters
        ----------
        value : float
            Angular frequency in rad/s.

        Returns
        -------
        float
            Mismatch in 1/m.
        """
        return float(prepared(np.array([value]))[0])

    with _probing():
        sampled = prepared(grid)
        if not np.all(np.isfinite(sampled)):
            # A sample that is not a number compares false with everything, so it
            # would delete any root beside it without a word.
            missing = int(np.sum(~np.isfinite(sampled)))
            raise ToolsError(
                f"the phase mismatch is not finite at {missing} of {sampled.size} "
                f"samples in {shortest * 1e9:.4g}-{longest * 1e9:.4g} nm: the index "
                "model does not cover the window"
            )
        signs = np.sign(sampled)
        found = [float(grid[i]) for i in np.flatnonzero(signs == 0.0)]
        found.extend(
            float(brentq(mismatch, float(grid[i]), float(grid[i + 1])))
            for i in np.flatnonzero(signs[:-1] * signs[1:] < 0.0)
        )
    if found:
        # Outside the probing context, so a root beyond the index model's range warns.
        base.beta(mode, np.array(sorted(found)), z)
    return tuple(sorted(2.0 * math.pi * C / root for root in found))


def lambda_rdw(  # noqa: PLR0913 - phase_mismatch's keywords and the search window
    mode: Mode,
    wavelength: float,
    *,
    soliton_mode: Mode | None = None,
    z: float = 0.0,
    peak_power: float | None = None,
    gamma: float | None = None,
    fwhm: float | None = None,
    shape: PulseShape = "sech",
    nonlinear_phase: NonlinearPhase = "largest",
    wavelength_limits: tuple[float, float] | None = None,
) -> float:
    """Return the phase-matched dispersive-wave wavelength nearest the pump.

    Parameters
    ----------
    mode : Mode
        The mode the dispersive wave is emitted into.
    wavelength : float
        Wavelength of the soliton in m, > 0.
    soliton_mode : Mode or None, optional
        The soliton's mode; ``None`` is `mode`.
    z : float, optional
        Position along the waveguide, m.
    peak_power : float or None, optional
        Peak power in W for the nonlinear term; see :func:`phase_mismatch`.
    gamma : float or None, optional
        Nonlinear coefficient in rad/(W m) for the nonlinear term.
    fwhm : float or None, optional
        Input power FWHM in s, positive; required with power for the default phase.
    shape : {'sech', 'gauss'}, optional
        Unchirped input shape for the largest constituent.
    nonlinear_phase : {'largest', 'supplied_power', 'linear'}, optional
        Default uses the largest constituent. Supplied power uses gamma P0/2 and
        requires omitting fwhm. Linear requires omitting all pulse numbers.
        Without pulse numbers the default is also linear.
    wavelength_limits : tuple of float or None, optional
        Window searched, m; ``None`` is 100 nm to 0.9 of the pump wavelength.

    Returns
    -------
    float
        The longest root inside the window in m, which with the default window is the
        short-wavelength dispersive wave nearest the pump. :func:`lambda_rdw_all`
        returns all of them.

    Raises
    ------
    ToolsError
        If the window holds no root, rather than returning a missing value that would
        propagate silently.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.tools import lambda_rdw
    >>> mode = MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)
    >>> f"{lambda_rdw(mode, 800e-9) * 1e9:.1f}"
    '200.1'
    """
    roots = lambda_rdw_all(
        mode,
        wavelength,
        soliton_mode=soliton_mode,
        z=z,
        peak_power=peak_power,
        gamma=gamma,
        fwhm=fwhm,
        shape=shape,
        nonlinear_phase=nonlinear_phase,
        wavelength_limits=wavelength_limits,
    )
    if not roots:
        limits = (
            (100e-9, 0.9 * wavelength)
            if wavelength_limits is None
            else wavelength_limits
        )
        ends = phase_mismatch(
            mode,
            2.0 * math.pi * C / np.array(limits[::-1]),
            wavelength,
            soliton_mode=soliton_mode,
            z=z,
            peak_power=peak_power,
            gamma=gamma,
            fwhm=fwhm,
            shape=shape,
            nonlinear_phase=nonlinear_phase,
        )
        raise ToolsError(
            f"no phase-matched wavelength in {limits[0] * 1e9:.4g}-"
            f"{limits[1] * 1e9:.4g} nm: the mismatch is {ends[1]:.4g} 1/m at the short "
            f"end and {ends[0]:.4g} 1/m at the long end and never changes sign"
        )
    return roots[-1]


def _pressure_root(
    residual: Callable[[float], float], pressure_max: float, what: str
) -> float:
    """Return the pressure at which a function of pressure vanishes.

    Parameters
    ----------
    residual : callable
        A function of the pressure in bar.
    pressure_max : float
        Upper end of the bracket in bar.
    what : str
        What is being matched, for the refusal.

    Returns
    -------
    float
        The root in bar.

    Raises
    ------
    ToolsError
        If the function has the same sign at both ends of the bracket.
    """
    top = _positive("pressure_max", pressure_max)
    if top <= _PRESSURE_FLOOR:
        raise ToolsError(f"pressure_max must exceed {_PRESSURE_FLOOR} bar, got {top}")
    with _probing():
        low, high = residual(_PRESSURE_FLOOR), residual(top)
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ToolsError(
            f"the residual for {what} is not finite at the ends of the pressure "
            f"bracket ({low!r}, {high!r}): the index model does not reach that far"
        )
    if low * high > 0.0:
        raise ToolsError(
            f"no pressure between {_PRESSURE_FLOOR} and {top:.4g} bar gives {what}: "
            f"the residual is {low:.4g} at the low end and {high:.4g} at the high end"
        )
    with _probing():
        root = float(brentq(residual, _PRESSURE_FLOOR, top))
    # Outside the probing context, so the answer's own warnings are raised.
    residual(root)
    return root


def _design_capillary(  # noqa: PLR0913 - the capillary's keywords
    radius: float,
    gas: Gas | str,
    *,
    temperature: float,
    m: int,
    model: MarcatiliModel,
    cladding: GlassIndex | complex | None,
    lattice: TubeLattice | None,
    materials: MaterialSelection | None,
) -> MarcatiliMode:
    """Build the single-fill capillary a pressure search re-evaluates.

    The mode `prop_capillary` builds with the same ``materials=``: the gas (or
    ``"air"``) as an optmat fill at one bar and ``cladding=None`` as optmat's
    recommended silica.

    Parameters
    ----------
    radius : float
        Core radius in m, > 0.
    gas : Gas or str
        The species.
    temperature : float
        Fill temperature in K.
    m : int
        Radial index of the HE_1m mode.
    model : {'full', 'reduced'}
        Marcatili model.
    cladding : GlassIndex, complex or None
        Cladding index; ``None`` is fused silica.
    lattice : TubeLattice or None
        Tube geometry for a VincettiMode; None builds a Marcatili capillary.
    materials : MaterialSelection or None
        The design's material selection.

    Returns
    -------
    MarcatiliMode
        The capillary at one bar.
    """
    bore = _positive("radius", radius)
    wall = _default_cladding(cladding)
    # The fill is an optmat mixture, which carries its own pressure and temperature.
    mixture = _resolve_fill(gas, 1.0, temperature, materials)
    if lattice is None:
        return MarcatiliMode(
            bore, None, 0.0, m=m, model=model, cladding=wall, mixture=mixture
        )
    return VincettiMode(
        bore,
        None,
        0.0,
        m=m,
        model=model,
        cladding=wall,
        mixture=mixture,
        lattice=lattice,
    )


def _capillary_at(mode: MarcatiliMode, pressure: float) -> MarcatiliMode:
    """Return a single-fill capillary re-evaluated at another pressure.

    The mode is built once and frozen at each density through its parameterised form,
    so a root finder costs an equation-of-state call and an index evaluation per step
    rather than a mode construction. A selected fill takes its density from its
    species' own equation of state (optmat's); a selected air, two populations at
    one shared pressure, is frozen at that pressure.

    Parameters
    ----------
    mode : MarcatiliMode
        The capillary, filled at any pressure.
    pressure : float
        Pressure in bar.

    Returns
    -------
    MarcatiliMode
        The same capillary at that pressure.
    """
    # pressure_rdw and pressure_zdw build this mode at their one scalar temperature.
    temperature = cast("float", mode.temperature)
    # `_design_capillary` always builds the mode on an optmat fill.
    fill = cast("Mixture", mode.mixture)
    if len(fill.components) > 1:
        return mode.at_parameters({"pressure": float(pressure)})
    state = next(iter(fill.components)).species.equation_of_state
    rho = float(state.density(float(pressure), temperature))
    return mode.at_parameters({"density": rho})


def _search_grid(
    low: float, high: float, samples: int = _SEARCH_SAMPLES
) -> Float[np.ndarray, " n_samples"]:
    """Return the values a sampled root search evaluates its residual at.

    Parameters
    ----------
    low, high : float
        Finite ordered ends of the search range.
    samples : int, optional
        Number of linear samples, and of logarithmic ones.

    Returns
    -------
    numpy.ndarray
        `samples` linear values, and as many logarithmic ones where ``low > 0``,
        merged and ascending; both ends are among them.
    """
    # The logarithmic grid resolves the low end of a range spanning decades (a
    # pressure from 1e-6 bar); the linear grid resolves structure at its high end.
    # Pairs of roots, or admissible intervals, narrower than a sample are not
    # guaranteed: narrowing the range resolves them.
    linear = np.linspace(low, high, samples)
    if low <= 0.0:
        return linear
    return np.unique(np.concatenate((np.geomspace(low, high, samples), linear)))


@dataclass(frozen=True)
class _Sampled:
    """The outcome of a sampled root search.

    Attributes
    ----------
    roots : tuple of float
        The roots, ascending.
    values : tuple of float
        The sampled values of the parameter, ascending.
    residuals : tuple of float or None
        The residual at each sample; None where the target is undefined.
    """

    roots: tuple[float, ...]
    values: tuple[float, ...]
    residuals: tuple[float | None, ...]


def _sampled_roots(
    residual: Callable[[float], float | None],
    limits: tuple[float, float],
    *,
    what: str,
    narrow: str,
) -> _Sampled:
    """Find every root of a residual over a range, without crossing a pole.

    The residual is sampled on :func:`_search_grid`; each sign change between two
    samples where it is defined is refined with Brent's method, and the refined
    point is a root only if the residual there is below :data:`_ROOT_RESIDUAL` of
    its larger value at the bracket's ends. A sample that is exactly zero is a root.

    Parameters
    ----------
    residual : callable
        A function of the parameter returning a finite float, or None where the
        target is undefined (an inadmissible state, such as no soliton).
    limits : tuple of float
        Finite ordered ends of the search range.
    what : str
        What the residual is, for the refusals.
    narrow : str
        What the caller narrows to resolve a bracket, for the refusals.

    Returns
    -------
    _Sampled
        The roots, and the samples.

    Raises
    ------
    ToolsError
        If the residual is not finite at a value, or a refinement reaches a value
        where it is undefined.
    """
    low, high = limits
    grid = _search_grid(low, high)
    # Brent's method re-evaluates the ends of its bracket and the root it returns,
    # which are samples already: one evaluation per distinct value.
    seen: dict[float, float | None] = {}

    def checked(value: float) -> float | None:
        """Evaluate the residual once per value, refusing a value that is not finite.

        Parameters
        ----------
        value : float
            Trial value of the parameter.

        Returns
        -------
        float or None
            Finite residual, or None where the target is undefined.

        Raises
        ------
        ToolsError
            If the residual is not finite.
        """
        if value not in seen:
            result = residual(value)
            if result is not None and not math.isfinite(result):
                raise ToolsError(f"{what} is not finite at {value:g}")
            seen[value] = result
        return seen[value]

    def refine(value: float) -> float:
        """Evaluate the residual inside one bracket, where it must be defined.

        Parameters
        ----------
        value : float
            Trial value of the parameter.

        Returns
        -------
        float
            The residual.

        Raises
        ------
        ToolsError
            If the bracket crossed a value where the target is undefined.
        """
        result = checked(value)
        if result is None:
            # The only undefined state a target has is the largest constituent's
            # absence, below the first soliton threshold or in normal dispersion.
            raise ToolsError(
                f"the refinement of {what} crossed a no-soliton region at "
                f"{value:g}, between two samples outside it; narrow {narrow}"
            )
        return result

    with _probing():
        sampled = [checked(float(value)) for value in grid]
        roots = [
            float(value)
            for value, result in zip(grid, sampled, strict=True)
            if result == 0.0
        ]
        for i, (left, right) in enumerate(pairwise(sampled)):
            if left is None or right is None or left * right >= 0.0:
                continue
            start, stop = float(grid[i]), float(grid[i + 1])
            root = float(
                brentq(
                    refine,
                    start,
                    stop,
                    xtol=_ROOT_TOLERANCE * max(abs(start), abs(stop)),
                )
            )
            if abs(refine(root)) <= _ROOT_RESIDUAL * max(abs(left), abs(right)):
                roots.append(root)
            else:
                logger.debug(
                    "discarded the sign change of %s at %g as a pole", what, root
                )
    return _Sampled(
        tuple(sorted(roots)),
        tuple(float(value) for value in grid),
        tuple(sampled),
    )


def _rdw_pressure_root(
    residual: Callable[[float], float | None], limits: tuple[float, float]
) -> float:
    """Find a unique pressure root without crossing a sampled inadmissible region.

    Parameters
    ----------
    residual : callable
        Mismatch in 1/m, or None only where no focusing discrete soliton exists.
    limits : tuple of float
        Positive ordered pressure bracket in bar.

    Returns
    -------
    float
        Unique pressure root in bar.

    Raises
    ------
    ToolsError
        If the bracket is invalid, the mismatch is nonfinite, a refinement enters
        an inadmissible region, or there is no unique sampled root.
    """
    low, high = limits
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
        raise ToolsError("pressure_limits must be finite, positive and ordered")
    roots = _sampled_roots(
        residual, limits, what="the RDW pressure residual", narrow="pressure_limits"
    ).roots
    if not roots:
        raise ToolsError(
            f"no pressure between {low:g} and {high:g} bar gives an admissible "
            "RDW root; narrow pressure_limits if needed"
        )
    root, *others = roots
    if others:
        found = ", ".join(f"{value:.4g}" for value in roots)
        raise ToolsError(
            f"ambiguous RDW pressure: {len(roots)} roots, at {found} bar; "
            "choose one with pressure_limits"
        )
    # Re-evaluate the accepted answer outside the probing warning filter.
    residual(root)
    return root


def pressure_rdw(  # noqa: PLR0913 - the capillary's keywords beside the two wavelengths
    radius: float,
    gas: Gas | str,
    target_wavelength: float,
    wavelength: float,
    *,
    pressure_max: float = 100.0,
    peak_power: float | None = None,
    fwhm: float | None = None,
    shape: PulseShape = "sech",
    nonlinear_phase: NonlinearPhase = "largest",
    lattice: TubeLattice | None = None,
    pressure_limits: tuple[float, float] | None = None,
    temperature: float = ROOM_TEMPERATURE,
    m: int = 1,
    model: MarcatiliModel = "full",
    cladding: GlassIndex | complex | None = None,
    materials: MaterialSelection | None = None,
) -> float:
    """Return the pressure that phase matches a dispersive wave at a target wavelength.

    Parameters
    ----------
    radius : float
        Core radius in m, > 0.
    gas : Gas or str
        A single built-in gas.
    target_wavelength : float
        Wavelength the dispersive wave should have, m.
    wavelength : float
        Pump wavelength in m.
    pressure_max : float, optional
        Upper end of the pressure bracket in **bar**; the lower end is 1e-6 bar.
    peak_power : float or None, optional
        Peak power in W. When given, the soliton's nonlinear wavenumber is included
        and its nonlinear coefficient follows the pressure.
    fwhm : float or None, optional
        Positive input power FWHM in s, required with power for the default phase.
    shape : {'sech', 'gauss'}, optional
        Unchirped input shape.
    nonlinear_phase : {'largest', 'supplied_power', 'linear'}, optional
        Largest constituent by default; supplied power preserves gamma P0/2
        without fwhm. Linear requires omitting pulse numbers.
    lattice : TubeLattice or None, optional
        Tube geometry for VincettiMode; None builds a Marcatili capillary.
    pressure_limits : tuple of float or None, optional
        Positive ordered search range in bar, within (1e-6, pressure_max).
        None uses that whole range. Searches 129 logarithmic and 129 linear
        samples, refining admissible sign changes; sub-grid root pairs or valid
        intervals can be missed. Narrow the range to resolve them.
    temperature : float, optional
        Fill temperature in K.
    m : int, optional
        Radial index of the HE_1m mode.
    model : {'full', 'reduced'}, optional
        Marcatili model.
    cladding : GlassIndex, complex or None, optional
        Cladding index; ``None`` is fused silica.
    materials : MaterialSelection or None, optional
        Design the capillary `prop_capillary` builds under the same
        ``materials=``: the gas from optmat (index, density, and the Kerr
        coefficient at `wavelength` for the nonlinear phase) and ``cladding=None``
        as optmat's recommended fused silica. ``None`` is
        ``MaterialSelection()``.

    Returns
    -------
    float
        Pressure in bar.

    Raises
    ------
    ToolsError
        If no pressure in the bracket matches, or the selection names a carrier
        other than `wavelength`.

    Examples
    --------
    >>> from solprop.tools import pressure_rdw
    >>> f"{pressure_rdw(125e-6, 'He', 199.75e-9, 800e-9, cladding=1.45):.2f}"
    '1.00'
    """
    # Validate before scanning; the nonlinear coefficient follows the pressure.
    design = RDW(
        wavelength,
        target_wavelength,
        peak_power=peak_power,
        fwhm=fwhm,
        shape=shape,
        nonlinear_phase=nonlinear_phase,
    )
    top = _positive("pressure_max", pressure_max)
    if top <= _PRESSURE_FLOOR:
        raise ToolsError(f"pressure_max must exceed {_PRESSURE_FLOOR} bar, got {top}")
    limits = (_PRESSURE_FLOOR, top) if pressure_limits is None else pressure_limits
    if limits[0] < _PRESSURE_FLOOR or limits[1] > top:
        raise ToolsError("pressure_limits must lie within (1e-6, pressure_max) bar")
    materials = MaterialSelection() if materials is None else materials
    _check_carrier(materials, wavelength)
    base_mode = _design_capillary(
        radius,
        gas,
        temperature=temperature,
        m=m,
        model=model,
        cladding=cladding,
        lattice=lattice,
        materials=materials,
    )

    def residual(pressure: float) -> float | None:
        """Return the mismatch at the target for one pressure.

        Parameters
        ----------
        pressure : float
            Pressure in bar.

        Returns
        -------
        float or None
            Mismatch in 1/m, or None when the constituent does not exist.
        """
        mode = _capillary_at(base_mode, pressure)
        gamma = None
        if peak_power is not None:
            assert mode.mixture is not None
            index = _fill_n2(mode.mixture, wavelength)[0]
            gamma = nonlinear_coefficient(mode, index, wavelength)
        return _rdw_residual(design, mode, gamma, 0.0)

    return _rdw_pressure_root(residual, limits)


def pressure_zdw(  # noqa: PLR0913 - the capillary's keywords
    radius: float,
    gas: Gas | str,
    zdw_wavelength: float,
    *,
    pressure_max: float = 100.0,
    temperature: float = ROOM_TEMPERATURE,
    m: int = 1,
    model: MarcatiliModel = "full",
    cladding: GlassIndex | complex | None = None,
    materials: MaterialSelection | None = None,
) -> float:
    """Return the pressure that puts the zero-dispersion point at a wavelength.

    Parameters
    ----------
    radius : float
        Core radius in m, > 0.
    gas : Gas or str
        A single built-in gas.
    zdw_wavelength : float
        Wavelength at which the dispersion should vanish, m.
    pressure_max : float, optional
        Upper end of the pressure bracket in **bar**; the lower end is 1e-6 bar.
    temperature : float, optional
        Fill temperature in K.
    m : int, optional
        Radial index of the HE_1m mode.
    model : {'full', 'reduced'}, optional
        Marcatili model.
    cladding : GlassIndex, complex or None, optional
        Cladding index; ``None`` is fused silica.
    materials : MaterialSelection or None, optional
        Design the capillary `prop_capillary` builds under the same
        ``materials=``: the gas's index and density from optmat and
        ``cladding=None`` as optmat's recommended fused silica. ``None`` is
        ``MaterialSelection()``.

    Returns
    -------
    float
        Pressure in bar.

    Raises
    ------
    ToolsError
        If no pressure in the bracket does.

    Examples
    --------
    >>> from solprop.tools import pressure_zdw
    >>> f"{pressure_zdw(125e-6, 'Ar', 500e-9, cladding=1.45):.3f}"
    '0.066'
    """
    materials = MaterialSelection() if materials is None else materials
    base_mode = _design_capillary(
        radius,
        gas,
        temperature=temperature,
        m=m,
        model=model,
        cladding=cladding,
        lattice=None,
        materials=materials,
    )
    omega = np.array([_omega(zdw_wavelength)])

    def residual(pressure: float) -> float:
        """Return the dispersion at the wavelength for one pressure.

        Parameters
        ----------
        pressure : float
            Pressure in bar.

        Returns
        -------
        float
            ``beta_2`` in s^2/m, scaled to order one.
        """
        mode = _capillary_at(base_mode, pressure)
        # Scaled so that brentq's absolute tolerance on the residual is meaningless
        # and its tolerance on the pressure decides, as intended.
        return float(base.beta2(mode, omega)[0]) * 1e30

    return _pressure_root(
        residual,
        pressure_max,
        f"a zero-dispersion wavelength of {zdw_wavelength * 1e9:.4g} nm",
    )


# ------------------------------------------------------------------- design solvers


@dataclass(frozen=True)
class ZDW:
    r"""Target a zero-dispersion wavelength: :math:`\beta_2 = 0` at a wavelength.

    The residual :func:`solve_parameter` brackets is :math:`\beta_2` of the trial mode
    at `wavelength`, in s^2/m.

    Attributes
    ----------
    wavelength : float
        Wavelength at which the dispersion should vanish, m, > 0.

    Raises
    ------
    ToolsError
        If the wavelength is not finite and positive.

    Examples
    --------
    >>> from solprop.tools import ZDW
    >>> ZDW(500e-9)
    ZDW(wavelength=5e-07)
    """

    wavelength: float

    def __post_init__(self) -> None:
        """Refuse a wavelength that is not finite and positive.

        Raises
        ------
        ToolsError
            If the wavelength is not finite and positive.
        """
        _positive("wavelength", self.wavelength)


@dataclass(frozen=True)
class RDW:
    """Target a resonant dispersive wave phase matched at a wavelength.

    The residual :func:`solve_parameter` brackets is :func:`phase_mismatch` of the
    trial mode at `rdw`, in 1/m, exactly as :func:`pressure_rdw` evaluates it: with
    `peak_power` the soliton's nonlinear wavenumber is included, with the trial
    mode's own nonlinear coefficient, and with the largest constituent (`fwhm`
    given) a trial value in normal dispersion or below the first soliton threshold
    has no residual and cannot bracket a root.

    Attributes
    ----------
    pump : float
        Wavelength of the soliton, m, > 0.
    rdw : float
        Wavelength the dispersive wave should have, m, > 0 and not the pump's.
    peak_power : float or None
        Input peak power in W for the nonlinear term; ``None`` is the linear
        condition. With it, the builder returns ``(mode, gamma)``.
    fwhm : float or None
        Input power FWHM in s; required with `peak_power` for the default phase.
    shape : {'sech', 'gauss'}
        Unchirped input shape for the largest constituent.
    nonlinear_phase : {'largest', 'supplied_power', 'linear'}
        The nonlinear phase model, as :func:`phase_mismatch` takes it.

    Raises
    ------
    ToolsError
        If a wavelength is not positive, the two are equal, or the pulse numbers
        are incomplete or contradictory for the phase model.

    Examples
    --------
    >>> from solprop.tools import RDW
    >>> RDW(800e-9, 200e-9, peak_power=2e9, fwhm=10e-15).nonlinear_phase
    'largest'
    """

    pump: float
    rdw: float
    peak_power: float | None = None
    fwhm: float | None = None
    shape: PulseShape = "sech"
    nonlinear_phase: NonlinearPhase = "largest"

    def __post_init__(self) -> None:
        """Refuse wavelengths or pulse numbers the phase matching does not define.

        Raises
        ------
        ToolsError
            If a wavelength is not positive, the two are equal, or the pulse
            numbers are incomplete or contradictory.
        """
        # A placeholder gamma: the real one comes with each trial mode.
        _PhasePulse(
            self.peak_power,
            None if self.peak_power is None else 1.0,
            self.fwhm,
            self.shape,
            self.nonlinear_phase,
        )
        _positive("pump", self.pump)
        _positive("rdw", self.rdw)
        if self.rdw == self.pump:
            # The linear mismatch and its slope vanish at the pump for every mode.
            raise ToolsError(
                f"the dispersive wave cannot be at the pump wavelength, {self.pump:g} m"
            )


@dataclass(frozen=True)
class SolitonOrder:
    """Target a conventional soliton order for a fixed pulse.

    The residual :func:`solve_parameter` brackets is :func:`soliton_order` of the
    pulse in the trial mode, with the trial mode's own nonlinear coefficient, minus
    `order`: a guide or fill parameter is solved for, the pulse is held. The order
    is the conventional measure on either side of a zero-dispersion point, so a
    search across one may find a root on each side. The opposite question, the
    pulse that has an order on a given mode, is closed form:
    :func:`soliton_peak_power` and :func:`pulse_energy`.

    Attributes
    ----------
    order : float
        Conventional soliton order, > 0.
    pulse : GaussianPulse or SechPulse
        The pulse: its wavelength is the pump, its FWHM and its energy or peak power
        are held, and its class sets the shape. An ordinary Gaussian (``m = 1``)
        with no spectral phase beyond a carrier-envelope phase and a delay.

    Raises
    ------
    ToolsError
        If the order is not positive, the Gaussian is super-Gaussian, or the pulse
        is chirped.

    Examples
    --------
    >>> from solprop.pulses import SechPulse
    >>> from solprop.tools import SolitonOrder
    >>> target = SolitonOrder(3.0, SechPulse(wavelength=800e-9, fwhm=10e-15,
    ...                                      energy=100e-6))
    >>> f"{target.peak_power / 1e9:.3f} GW"
    '8.814 GW'
    """

    order: float
    pulse: GaussianPulse | SechPulse

    def __post_init__(self) -> None:
        """Refuse an order or a pulse the design numbers are not defined for.

        Raises
        ------
        ToolsError
            If the order is not positive, the Gaussian is super-Gaussian, or the
            pulse is chirped.
        """
        _positive("order", self.order)
        if isinstance(self.pulse, GaussianPulse) and self.pulse.m != 1:
            raise ToolsError(
                "the soliton order is defined for sech and Gaussian pulses, and this "
                f"is a super-Gaussian of order m={self.pulse.m}"
            )
        if np.any(np.asarray(self.pulse.phase, dtype=float)[2:] != 0.0):
            # The design numbers take the width and peak power of an unchirped pulse.
            raise ToolsError(
                "the soliton order is defined for an unchirped pulse; this one has "
                "spectral phase beyond the carrier-envelope phase and the delay"
            )

    @property
    def shape(self) -> PulseShape:
        """Return the pulse shape the design numbers use.

        Returns
        -------
        {'sech', 'gauss'}
            ``"gauss"`` for a `GaussianPulse`, ``"sech"`` for a `SechPulse`.
        """
        return "gauss" if isinstance(self.pulse, GaussianPulse) else "sech"

    @property
    def fwhm(self) -> float:
        """Return the pulse's full width at half maximum of the power.

        Returns
        -------
        float
            FWHM in s; a `SechPulse` given its natural width converts it.
        """
        pulse = self.pulse
        if isinstance(pulse, GaussianPulse):
            return pulse.fwhm
        # SechPulse holds exactly one of the two widths (its own validation).
        return (
            pulse.fwhm
            if pulse.fwhm is not None
            else cast("float", pulse.width) * _SECH_WIDTH
        )

    @property
    def peak_power(self) -> float:
        """Return the pulse's peak power.

        Returns
        -------
        float
            Peak power in W: the pulse's own, or that of its energy
            (:func:`pulse_peak_power`).
        """
        if self.pulse.peak_power is not None:
            return self.pulse.peak_power
        # ShapedPulse holds exactly one of energy and peak power.
        return pulse_peak_power(cast("float", self.pulse.energy), self.fwhm, self.shape)


DesignTarget = ZDW | RDW | SolitonOrder
"""A target of :func:`solve_parameter`."""

RootChoice = Literal["lowest", "highest"] | int
"""Which of several roots: the lowest, the highest, or an index into them."""

DesignBuild = Callable[[float], Mode | tuple[Mode, float]]
"""A builder: a trial value of the parameter to a mode, or to a mode and the fill's
nonlinear coefficient in rad/(W m) (:func:`nonlinear_coefficient`)."""


@dataclass(frozen=True)
class DesignRoots:
    """Every root of a design target over a parameter's search range.

    Iterating gives the roots, and its length is their number.

    Attributes
    ----------
    roots : tuple of float
        The values of the parameter that meet the target, ascending, in the
        parameter's unit; empty when none does.
    search : tuple of float
        The search range, ``(low, high)``.
    target : ZDW, RDW or SolitonOrder
        The target.
    builds : int
        How many times the builder was called, the search's cost.
    quantity : str
        What `attainable` measures, with its unit: the zero-dispersion wavelength
        nearest the target's (m), the phase-matched wavelength nearest the target's
        (m), or the soliton order.
    attainable : tuple of float or None
        When there is no root, the smallest and largest `quantity` on a coarser grid
        of the search range (17 linear and, where ``low > 0``, 17 logarithmic
        values, the ends included), where it is defined: a mode with no zero
        dispersion in 100 nm to 3 um, or with no phase-matched wavelength in the
        RDW window or no soliton, has none. ``None`` when there are roots, or no
        value defines it.

    Examples
    --------
    >>> from solprop.tools import DesignRoots, ZDW
    >>> roots = DesignRoots((1.0, 2.0), (0.5, 3.0), ZDW(5e-7), 260,
    ...                     "zero-dispersion wavelength in m")
    >>> len(roots), roots.pick("highest"), roots.pick(0)
    (2, 2.0, 1.0)
    """

    roots: tuple[float, ...]
    search: tuple[float, float]
    target: DesignTarget
    builds: int
    quantity: str
    attainable: tuple[float, float] | None = None

    def __len__(self) -> int:
        """Return the number of roots.

        Returns
        -------
        int
            The number of roots.
        """
        return len(self.roots)

    def __iter__(self) -> Iterator[float]:
        """Iterate over the roots, ascending.

        Returns
        -------
        Iterator
            The roots.
        """
        return iter(self.roots)

    def pick(self, root: RootChoice = "lowest") -> float:
        """Return one root.

        Parameters
        ----------
        root : {'lowest', 'highest'} or int, optional
            The lowest or highest root, or an index into the ascending roots
            (negative from the end, as a sequence's).

        Returns
        -------
        float
            The root, in the parameter's unit.

        Raises
        ------
        ToolsError
            If there is no root, naming the attainable range, or the index is out of
            range, naming the roots.

        Examples
        --------
        >>> from solprop.tools import DesignRoots, ZDW
        >>> DesignRoots((1.0, 2.0), (0.5, 3.0), ZDW(5e-7), 260, "").pick(-1)
        2.0
        """
        if not self.roots:
            raise ToolsError(self._unreachable())
        if root == "lowest":
            return self.roots[0]
        if root == "highest":
            return self.roots[-1]
        if isinstance(root, bool) or not isinstance(root, int):
            raise ToolsError(
                f"root must be 'lowest', 'highest' or an index, got {root!r}"
            )
        count = len(self.roots)
        if not -count <= root < count:
            found = ", ".join(f"{value:.6g}" for value in self.roots)
            raise ToolsError(
                f"root index {root} is out of range: {count} root"
                f"{'' if count == 1 else 's'}, at {found}"
            )
        return self.roots[root]

    def _unreachable(self) -> str:
        """Return why no root was found.

        Returns
        -------
        str
            The search range and the attainable range of the quantity.
        """
        low, high = self.search
        if self.attainable is None:
            seen = f"the {self.quantity} is undefined across it"
        else:
            seen = (
                f"across it the {self.quantity} spans "
                f"{self.attainable[0]:.6g} to {self.attainable[1]:.6g}"
            )
        return (
            f"no value between {low:.6g} and {high:.6g} gives "
            f"{_describe(self.target)}: {seen}"
        )


def _describe(target: DesignTarget) -> str:
    """Return a target in words, for messages.

    Parameters
    ----------
    target : ZDW, RDW or SolitonOrder
        The target.

    Returns
    -------
    str
        For example ``"a zero-dispersion wavelength of 500 nm"``.
    """
    if isinstance(target, ZDW):
        return f"a zero-dispersion wavelength of {target.wavelength * 1e9:.6g} nm"
    if isinstance(target, RDW):
        return (
            f"a dispersive wave at {target.rdw * 1e9:.6g} nm from the pump at "
            f"{target.pump * 1e9:.6g} nm"
        )
    return f"a soliton order of {target.order:.6g}"


def _quantity_name(target: DesignTarget) -> str:
    """Return what a target's attainable range measures, with its unit.

    Parameters
    ----------
    target : ZDW, RDW or SolitonOrder
        The target.

    Returns
    -------
    str
        The quantity and its unit.
    """
    if isinstance(target, ZDW):
        return "zero-dispersion wavelength nearest the target (m)"
    if isinstance(target, RDW):
        return "phase-matched wavelength nearest the target (m)"
    return "soliton order"


def _needs_gamma(target: DesignTarget) -> bool:
    """Return whether a target needs the fill's nonlinear coefficient.

    Parameters
    ----------
    target : ZDW, RDW or SolitonOrder
        The target.

    Returns
    -------
    bool
        True for a soliton order and for an RDW with a peak power.
    """
    if isinstance(target, RDW):
        return target.peak_power is not None
    return isinstance(target, SolitonOrder)


def _pump_dispersion(mode: Mode, wavelength: float, z: float) -> float:
    """Return the group-velocity dispersion at a pump, refusing one not finite.

    Parameters
    ----------
    mode : Mode
        The mode.
    wavelength : float
        Pump wavelength in m.
    z : float
        Position along the waveguide, m.

    Returns
    -------
    float
        ``beta_2`` in s^2/m.

    Raises
    ------
    ToolsError
        If it is not finite.
    """
    dispersion = float(base.beta2(mode, np.array([_omega(wavelength)]), z)[0])
    if not math.isfinite(dispersion):
        raise ToolsError(
            f"the pump dispersion is not finite at {wavelength * 1e9:.6g} nm in a "
            "trial mode"
        )
    return dispersion


def _rdw_residual(
    target: RDW, mode: Mode, gamma: float | None, z: float
) -> float | None:
    """Return the phase mismatch at a dispersive-wave target, as the RDW search uses it.

    Parameters
    ----------
    target : RDW
        The target.
    mode : Mode
        The trial mode, carrying both the soliton and the wave.
    gamma : float or None
        The mode's nonlinear coefficient in rad/(W m); given with a peak power.
    z : float
        Position along the waveguide, m.

    Returns
    -------
    float or None
        Mismatch in 1/m, or None when the largest constituent does not exist: in
        normal dispersion, or at a pulse area at or below the first threshold.

    Raises
    ------
    ToolsError
        If the pump dispersion is not finite.
    """
    if target.peak_power is not None and target.fwhm is not None:
        dispersion = _pump_dispersion(mode, target.pump, z)
        if dispersion >= 0:
            return None
        order = soliton_order(
            target.peak_power,
            target.fwhm,
            cast("float", gamma),
            dispersion,
            target.shape,
        )
        if order * _area_per_order(target.shape) <= 0.5:
            return None
    return float(
        phase_mismatch(
            mode,
            np.array([_omega(target.rdw)]),
            target.pump,
            z=z,
            peak_power=target.peak_power,
            gamma=gamma,
            fwhm=target.fwhm,
            shape=target.shape,
            nonlinear_phase=target.nonlinear_phase,
        )[0]
    )


def _target_residual(
    target: DesignTarget, mode: Mode, gamma: float | None, z: float
) -> float | None:
    """Return a target's residual on one trial mode.

    Parameters
    ----------
    target : ZDW, RDW or SolitonOrder
        The target.
    mode : Mode
        The trial mode.
    gamma : float or None
        Its nonlinear coefficient in rad/(W m), where the target needs one.
    z : float
        Position along the waveguide, m.

    Returns
    -------
    float or None
        ``beta_2`` in s^2/m, the mismatch in 1/m (None where no soliton exists), or
        the soliton order less the target's.
    """
    if isinstance(target, ZDW):
        return float(base.beta2(mode, np.array([_omega(target.wavelength)]), z)[0])
    if isinstance(target, RDW):
        return _rdw_residual(target, mode, gamma, z)
    dispersion = _pump_dispersion(mode, target.pulse.wavelength, z)
    order = soliton_order(
        target.peak_power, target.fwhm, cast("float", gamma), dispersion, target.shape
    )
    return order - target.order


def _rdw_window(target: RDW) -> tuple[float, float]:
    """Return the window an RDW target's attainable wavelengths are searched in.

    Parameters
    ----------
    target : RDW
        The target.

    Returns
    -------
    tuple of float
        :func:`lambda_rdw`'s default window, 100 nm to 0.9 of the pump, when the
        target is on the blue side of the pump, widened to include it; and the
        mirror of it, 1/0.9 of the pump to 3 um, on the red side.
    """
    if target.rdw < target.pump:
        return (min(_ZDW_RANGE[0], 0.9 * target.rdw), 0.9 * target.pump)
    return (target.pump / 0.9, max(_ZDW_RANGE[1], target.rdw / 0.9))


def _target_quantity(
    target: DesignTarget, mode: Mode, gamma: float | None, z: float
) -> float:
    """Return what a target measures on one trial mode, for the attainable range.

    Parameters
    ----------
    target : ZDW, RDW or SolitonOrder
        The target.
    mode : Mode
        The trial mode.
    gamma : float or None
        Its nonlinear coefficient in rad/(W m), where the target needs one.
    z : float
        Position along the waveguide, m.

    Returns
    -------
    float
        The zero-dispersion or phase-matched wavelength nearest the target's in m,
        or the soliton order; NaN where the mode has none.
    """
    if isinstance(target, ZDW):
        return _zdw_near(mode, _omega(target.wavelength), z)
    if isinstance(target, SolitonOrder):
        residual = _target_residual(target, mode, gamma, z)
        return target.order + cast("float", residual)
    if _rdw_residual(target, mode, gamma, z) is None:
        return math.nan
    try:
        roots = lambda_rdw_all(
            mode,
            target.pump,
            z=z,
            peak_power=target.peak_power,
            gamma=gamma,
            fwhm=target.fwhm,
            shape=target.shape,
            nonlinear_phase=target.nonlinear_phase,
            wavelength_limits=_rdw_window(target),
        )
    except ToolsError:
        # The window reaches beyond the mode's index model; this is a diagnostic
        # for a target already found unreachable, so the sample has no wavelength.
        return math.nan
    if not roots:
        return math.nan
    return min(roots, key=lambda root: abs(root - target.rdw))


def _built(
    build: DesignBuild, value: float, target: DesignTarget
) -> tuple[Mode, float | None]:
    """Call a builder and check what it returned.

    Parameters
    ----------
    build : callable
        The builder.
    value : float
        Trial value of the parameter.
    target : ZDW, RDW or SolitonOrder
        The target, which says whether a nonlinear coefficient is needed.

    Returns
    -------
    tuple
        The mode, and the nonlinear coefficient in rad/(W m), or None where the
        target does not use one.

    Raises
    ------
    ToolsError
        If the builder returned neither a mode nor a mode and a positive
        coefficient, or no coefficient for a target that needs one.
    """
    result = build(value)
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ToolsError(
                f"build({value:g}) returned a tuple of {len(result)}; give a mode "
                "or (mode, gamma)"
            )
        mode, gamma = result
        gamma = _positive("gamma", gamma)
    else:
        mode, gamma = result, None
    if not isinstance(mode, Mode):
        raise ToolsError(
            f"build({value:g}) returned {type(mode).__name__}, not a mode or "
            "(mode, gamma)"
        )
    if not _needs_gamma(target):
        # One builder serves every target: a linear target ignores the coefficient.
        return mode, None
    if gamma is None:
        raise ToolsError(
            f"{_describe(target)} needs the fill's nonlinear coefficient: build "
            "returns (mode, gamma), with gamma from nonlinear_coefficient"
        )
    return mode, gamma


def solve_parameter(
    build: DesignBuild,
    search: tuple[float, float],
    target: DesignTarget,
    *,
    z: float = 0.0,
) -> DesignRoots:
    """Return every value of a guide or fill parameter that meets a design target.

    The general form of :func:`pressure_zdw`, :func:`pressure_rdw` and
    :func:`radius_zdw`, for any mode and any parameter: a radius, a pressure, a core
    size or a wall thickness. The target's residual is sampled over `search` on 129
    linear values and, where ``low > 0``, 129 logarithmic ones (as
    :func:`pressure_rdw` samples pressure); each sign change between two samples is
    refined with Brent's method to a relative tolerance of 2e-12; and a refined
    change is kept as a root only where the residual has fallen to a thousandth of
    its value at the bracket's ends, which discards a change through a pole (a wall
    resonance, where it diverges instead). Roots closer together than one sample
    are not separated: narrow `search` to resolve them.

    Parameters
    ----------
    build : callable
        ``build(value)`` returns the mode at a trial value of the parameter, or
        ``(mode, gamma)`` with the fill's nonlinear coefficient in rad/(W m)
        (:func:`nonlinear_coefficient`), which an RDW target with a peak power and a
        soliton-order target need: a mode need not know its fill's Kerr response.
    search : tuple of float
        Finite ordered range of the parameter, ``(low, high)``, in its unit.
    target : ZDW, RDW or SolitonOrder
        What the parameter should achieve.
    z : float, optional
        Position along the waveguide at which each mode is evaluated, m: the launch
        position.

    Returns
    -------
    DesignRoots
        Every root, ascending, with the attainable range of the target's quantity
        when there is none; ``.pick(root)`` chooses one.

    Raises
    ------
    ToolsError
        If the range or position is not finite or not ordered, the builder returns
        something other than a mode (and a coefficient where one is needed), a
        residual is not finite, or a refinement crosses a no-soliton region.

    Examples
    --------
    The radius of a helium capillary at 1 bar with its zero dispersion at 400 nm,
    re-evaluating one mode at each trial radius:

    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.tools import ZDW, solve_parameter
    >>> mode = MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)
    >>> roots = solve_parameter(
    ...     lambda radius: mode.at_parameters({"core_radius": radius}),
    ...     (20e-6, 500e-6),
    ...     ZDW(400e-9),
    ... )
    >>> [f"{radius * 1e6:.2f} um" for radius in roots]
    ['89.49 um']
    """
    low, high = (float(value) for value in search)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ToolsError(f"search must be finite and ordered, got {search!r}")
    if not math.isfinite(z):
        raise ToolsError(f"z must be finite, got {z!r}")
    builds = 0

    def residual(value: float) -> float | None:
        """Build the mode at one value and return the target's residual.

        Parameters
        ----------
        value : float
            Trial value of the parameter.

        Returns
        -------
        float or None
            The residual, or None where the target is undefined.
        """
        nonlocal builds
        builds += 1
        mode, gamma = _built(build, value, target)
        return _target_residual(target, mode, gamma, z)

    sampled = _sampled_roots(
        residual, (low, high), what="the target's residual", narrow="search"
    )
    # Outside the probing context, so a root beyond an index model's range warns.
    for root in sampled.roots:
        residual(root)
    attainable = None
    if not sampled.roots:
        # The modes are rebuilt rather than kept from the search, on a coarser grid:
        # a tabulated mode can be large, and only a search without a root pays.
        with _probing():
            values = []
            for value in _search_grid(low, high, _ATTAINABLE_SAMPLES):
                builds += 1
                mode, gamma = _built(build, float(value), target)
                values.append(_target_quantity(target, mode, gamma, z))
        finite = [value for value in values if math.isfinite(value)]
        if finite:
            attainable = (min(finite), max(finite))
    return DesignRoots(
        roots=sampled.roots,
        search=(low, high),
        target=target,
        builds=builds,
        quantity=_quantity_name(target),
        attainable=attainable,
    )


def radius_zdw(  # noqa: PLR0913 - the capillary's keywords
    gas: Gas | str,
    pressure: float,
    zdw_wavelength: float,
    *,
    radius_limits: tuple[float, float] = (5e-6, 1e-3),
    temperature: float = ROOM_TEMPERATURE,
    m: int = 1,
    model: MarcatiliModel = "full",
    cladding: GlassIndex | complex | None = None,
    materials: MaterialSelection | None = None,
    root: RootChoice = "lowest",
) -> float:
    """Return the core radius that puts a capillary's zero dispersion at a wavelength.

    The radius counterpart of :func:`pressure_zdw`, through
    :func:`solve_parameter`: the capillary is built once at `pressure` and
    re-evaluated at each trial radius.

    Parameters
    ----------
    gas : Gas or str
        A single built-in gas.
    pressure : float
        Fill pressure in **bar**, > 0.
    zdw_wavelength : float
        Wavelength at which the dispersion should vanish, m.
    radius_limits : tuple of float, optional
        Positive ordered search range of the core radius, m.
    temperature : float, optional
        Fill temperature in K.
    m : int, optional
        Radial index of the HE_1m mode.
    model : {'full', 'reduced'}, optional
        Marcatili model.
    cladding : GlassIndex, complex or None, optional
        Cladding index; ``None`` is fused silica.
    materials : MaterialSelection or None, optional
        Design the capillary `prop_capillary` builds under the same
        ``materials=``, as :func:`pressure_zdw` does. ``None`` is
        ``MaterialSelection()``.
    root : {'lowest', 'highest'} or int, optional
        Which root, should there be several (``DesignRoots.pick``).

    Returns
    -------
    float
        Core radius in m.

    Raises
    ------
    ToolsError
        If the radius limits are not positive and ordered, the pressure is not
        positive, or no radius in the range gives the zero (naming the range of
        zero-dispersion wavelengths it does give).

    Examples
    --------
    >>> from solprop.tools import radius_zdw
    >>> f"{radius_zdw('He', 1.0, 400e-9, cladding=1.45) * 1e6:.2f} um"
    '89.49 um'
    """
    low, high = radius_limits
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ToolsError(
            f"radius_limits must be finite, positive and ordered, got {radius_limits!r}"
        )
    materials = MaterialSelection() if materials is None else materials
    capillary = _capillary_at(
        _design_capillary(
            high,
            gas,
            temperature=temperature,
            m=m,
            model=model,
            cladding=cladding,
            lattice=None,
            materials=materials,
        ),
        _positive("pressure", pressure),
    )
    roots = solve_parameter(
        lambda radius: capillary.at_parameters({"core_radius": radius}),
        (low, high),
        ZDW(zdw_wavelength),
    )
    return roots.pick(root)
