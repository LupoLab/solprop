# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later


"""Dispersive-wave phase matching: the mismatch and its roots in wavelength."""

import math
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from jaxtyping import Float
from scipy.optimize import brentq

from solprop._validation import choice, positive
from solprop.constants import C
from solprop.exceptions import (
    ToolsError,
)
from solprop.modes import base
from solprop.modes.protocol import Mode
from solprop.tools.pulse import PulseShape, _omega
from solprop.tools.soliton import soliton_content
from solprop.tools.waveguide import _probing

__all__ = ["NonlinearPhase", "lambda_rdw", "lambda_rdw_all", "phase_mismatch"]

NonlinearPhase = Literal["largest", "supplied_power", "linear"]
"""Choose the constituent, supplied-power or linear RDW phase model."""


_MISMATCH_SAMPLES = 2048
"""Frequency samples the phase-matching search brackets its roots on.

The mismatch of a hollow waveguide is smooth on the scale of the search window (its
structure is set by the zero-dispersion points, of which there are a handful), so two
roots closer than a two-thousandth of the window are not separated. The cost is one
vectorised index evaluation.
"""


@dataclass(frozen=True)
class _PhasePulse:
    """Validate the pulse declaration before a phase-matching search."""

    peak_power: float | None
    gamma: float | None
    fwhm: float | None
    shape: PulseShape
    nonlinear_phase: NonlinearPhase

    def __post_init__(self) -> None:
        """Refuse incomplete or contradictory phase declarations."""
        choice(self.shape, "shape", get_args(PulseShape), ToolsError)
        choice(
            self.nonlinear_phase,
            "nonlinear_phase",
            get_args(NonlinearPhase),
            ToolsError,
        )
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
        positive(self.peak_power, "peak_power", ToolsError)
        assert self.gamma is not None  # The paired declaration was validated above.
        positive(self.gamma, "gamma", ToolsError)
        if self.nonlinear_phase == "largest":
            if self.fwhm is None:
                raise ToolsError(
                    "the largest constituent needs fwhm; supply it or select "
                    "nonlinear_phase='supplied_power'"
                )
            positive(self.fwhm, "fwhm", ToolsError)
        elif self.fwhm is not None:
            raise ToolsError(
                "nonlinear_phase='supplied_power' does not use fwhm; omit it"
            )

    def wavenumber(self, mode: Mode, omega: float, z: float) -> float:
        """Calculate the nonlinear phase in the soliton mode.

        - ``omega``: pump angular frequency in rad/s.
        - ``z``: position in m.
        - Returns: nonlinear wavenumber in 1/m.
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

        - ``omega``: angular frequencies in rad/s.
        - Returns: mismatch in 1/m.
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

    - ``wavelength``: pump wavelength in m.
    - ``soliton_mode``: pump mode, or None to use mode.
    - ``z``: position in m.
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


def phase_mismatch(
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


def lambda_rdw_all(
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
        Brent's method, and roots closer than one sample are not separated. A sign
        change through a resonance of the guide, where the mismatch has a pole or is
        not a number, is not a root and is skipped.

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

    def _mismatch(value: float) -> float:
        """Return the mismatch at one frequency, for the root finder.

        - ``value``: angular frequency in rad/s.
        - Returns: mismatch in 1/m.
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
        for i in np.flatnonzero(signs[:-1] * signs[1:] < 0.0):
            try:
                # The refinement may land on a resonance exactly, where a model may
                # divide by zero: such a point is no root, and is told apart below.
                with np.errstate(divide="ignore", invalid="ignore"):
                    root = float(brentq(_mismatch, float(grid[i]), float(grid[i + 1])))
            except ValueError:
                # The index model is not a number inside the bracket: a resonance of
                # the guide (an antiresonant fibre's), not a phase-matched wavelength.
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                residual = _mismatch(root)
            if not abs(residual) <= min(abs(sampled[i]), abs(sampled[i + 1])):
                # Refined onto a pole, through which the sign changed: no root.
                continue
            found.append(root)
    if found:
        # Outside the probing context, so a root beyond the index model's range warns.
        base.beta(mode, np.array(sorted(found)), z)
    return tuple(sorted(2.0 * math.pi * C / root for root in found))


def lambda_rdw(
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
