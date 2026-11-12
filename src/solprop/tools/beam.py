# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later


"""Free-space beam numbers: the critical power, beam parameters and intensities."""

import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from solprop import spatial
from solprop._validation import positive
from solprop.constants import EPS0, C
from solprop.exceptions import (
    ToolsError,
)
from solprop.physdata import (
    Gas,
    chi3,
    chi3_from_n2,
    refractive_index,
)
from solprop.tools.pulse import _omega

__all__ = [
    "BeamParams",
    "beam_params",
    "beam_through_lens",
    "field_to_intensity",
    "intensity_to_field",
    "self_focusing_power",
]


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
    linear = positive(index, "index", ToolsError)
    chi3 = chi3_from_n2(positive(n2, "n2", ToolsError), linear)
    return spatial.critical_power(
        positive(wavelength, "wavelength", ToolsError), chi3, index=linear
    )


@dataclass(frozen=True)
class BeamParams:
    r"""The characteristic numbers of a Gaussian beam in a gas.

    Attributes
    ----------
    waist : float
        The 1/e² intensity radius at the waist, m.
    focus : float
        Where the waist lies from the launch plane, m.
    wavelength : float
        The vacuum wavelength, m.
    index : float
        The linear refractive index.
    rayleigh_range : float
        :math:`\pi n w_0^2/\lambda`, m.
    critical_power : float
        The Townes critical power, W (:func:`solprop.spatial.critical_power`).
    peak_power : float
        The beam's peak power, W.
    power_ratio : float
        Its peak power over the critical power.
    collapse_distance : float or None
        Where a monochromatic beam in a pure Kerr medium would collapse by Marburger's
        empirical formula, m from the launch plane, with a lens's focus combined by
        Talanov's law. A reference, not a prediction of a broadband, ionising run;
        ``None`` at or below the critical power, or where a beam already diverging
        leaves no collapse ahead.
    """

    waist: float
    focus: float
    wavelength: float
    index: float
    rayleigh_range: float
    critical_power: float
    peak_power: float
    power_ratio: float
    collapse_distance: float | None

    def radius(self, z: Float[np.ndarray, "*shape"]) -> Float[np.ndarray, "*shape"]:
        """Return the linear beam radius along z, in the absence of any nonlinearity.

        Parameters
        ----------
        z : numpy.ndarray
            Positions from the launch plane, m.

        Returns
        -------
        numpy.ndarray
            The 1/e² intensity radius, m.
        """
        offset = (np.asarray(z, dtype=float) - self.focus) / self.rayleigh_range
        return self.waist * np.sqrt(1.0 + offset**2)


def beam_params(
    waist: float,
    wavelength: float,
    gas: Gas | str,
    pressure: float,
    peak_power: float,
    *,
    focus: float = 0.0,
) -> BeamParams:
    r"""Return the characteristic numbers of a Gaussian beam in a gas.

    The Rayleigh range, the critical power, and Marburger's collapse distance
    :math:`0.367 z_R/\sqrt{(\sqrt{P/P_{\rm cr}}-0.852)^2-0.0219}`, for a beam collimated
    at the launch plane. A beam launched a distance :math:`d` before its waist has, at
    launch, the radius :math:`w_0\sqrt{1 + (d/z_R)^2}` and a wavefront converging with
    radius :math:`R = d + z_R^2/d`; Marburger's distance for that radius is combined
    with :math:`R` by Talanov's lens law, :math:`1/z_c' = 1/z_c + 1/R`. A waist just
    ahead is nearly collimated, so the distance moves smoothly with :math:`d`.

    Parameters
    ----------
    waist : float
        The 1/e² intensity radius at the waist, m.
    wavelength : float
        The vacuum wavelength, m.
    gas : Gas or str
        The gas.
    pressure : float
        Its pressure, bar.
    peak_power : float
        The beam's peak power, W.
    focus : float, optional
        Where the waist lies from the launch plane, m.

    Returns
    -------
    BeamParams
        The numbers.

    Examples
    --------
    >>> from solprop.tools import beam_params
    >>> beam = beam_params(300e-6, 800e-9, "Ar", 1.0, 27.6e9)
    >>> f"{beam.rayleigh_range:.3f} m, P/Pcr {beam.power_ratio:.2f}"
    '0.354 m, P/Pcr 3.00'
    >>> f"{beam.collapse_distance:.3f} m"
    '0.150 m'
    """
    omega = _omega(wavelength)
    index = float(refractive_index(gas, omega, pressure=pressure))
    power = spatial.critical_power(
        wavelength, chi3(gas, pressure=pressure, wavelength=wavelength), index=index
    )
    rayleigh = np.pi * index * positive(waist, "waist", ToolsError) ** 2 / wavelength
    ratio = positive(peak_power, "peak_power", ToolsError) / power
    collapse = None
    if ratio > 1.0:
        # (sqrt(r) - 0.852)^2 - 0.0219 vanishes at r = 1, so it is positive above it.
        root = (np.sqrt(ratio) - 0.852) ** 2 - 0.0219
        # At launch: the radius there, and the wavefront's radius of curvature.
        launch = rayleigh * (1.0 + (focus / rayleigh) ** 2)
        collapse = float(0.367 * launch / np.sqrt(root))
        if focus != 0:
            curvature = focus + rayleigh**2 / focus
            inverse = 1.0 / collapse + 1.0 / curvature
            collapse = float(1.0 / inverse) if inverse > 0 else None
    return BeamParams(
        waist,
        focus,
        wavelength,
        index,
        float(rayleigh),
        float(power),
        peak_power,
        float(ratio),
        collapse,
    )


def beam_through_lens(
    focal_length: float, radius: float, wavelength: float
) -> tuple[float, float]:
    r"""Return the waist a thin lens makes of a collimated Gaussian beam.

    :math:`w_0 = \lambda f / (\pi w) / \sqrt{1 + (\lambda f / \pi w^2)^2}`, and the
    waist's distance from the lens, :math:`f / (1 + (\lambda f / \pi w^2)^2)`, which
    falls short of the focal length for a weakly focused beam. Paraxial, in vacuum.

    Parameters
    ----------
    focal_length : float
        The lens's focal length, m.
    radius : float
        The beam's 1/e² intensity radius on the lens, m.
    wavelength : float
        The vacuum wavelength, m.

    Returns
    -------
    tuple of float
        The waist, m, and its distance from the lens, m.

    Examples
    --------
    >>> from solprop.tools import beam_through_lens
    >>> waist, distance = beam_through_lens(1.0, 5e-3, 800e-9)
    >>> f"{waist * 1e6:.1f} um at {distance:.4f} m"
    '50.9 um at 0.9999 m'
    """
    focal = positive(focal_length, "focal_length", ToolsError)
    beam = positive(radius, "radius", ToolsError)
    light = positive(wavelength, "wavelength", ToolsError)
    ratio = light * focal / (np.pi * beam**2)
    factor = 1.0 / (1.0 + ratio**2)
    return float(light * focal / (np.pi * beam) * np.sqrt(factor)), float(
        focal * factor
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
