# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Single-ring tube-lattice modes with the Vincetti empirical fits.

The transverse field and protocol area remain Marcatili's. The spectral model follows
the published Vincetti fits (``docs/guide/waveguides/antiresonant.md``): where the
printed papers and earlier published implementations of them, pyJCMwave among them,
differ,
``CONVENTIONS`` names the choice and ``LEGACY_CONVENTIONS`` the form those
implementations share, so that a test can assemble the latter from the same building
blocks. The constants are transcribed from the three papers; the 2016 paper's
equations were read from a degraded text extraction and have not been checked against
the typeset page.
"""

import logging
import math
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
from jaxtyping import Float, Inexact

from solprop._display import Displayed, engineering, line
from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.modes.capillary import MarcatiliMode, MarcatiliTerms
from solprop.physdata.glasses import GlassIndex
from solprop.profiles import as_profile

__all__ = [
    "AREA_FACTOR",
    "DISPERSION_PREFACTOR",
    "DISPERSION_WIDTH",
    "LOSS_WIDTH",
    "MAX_TERMS",
    "MINIMUM_LOSS_PREFACTOR",
    "RADIUS_COEFFICIENT",
    "VALIDATED_TUBE_COUNT",
    "VALIDATED_TUBE_INDEX",
    "VALIDATED_WALL_RATIO",
    "TubeLattice",
    "VincettiMode",
    "core_radius_from_gap",
    "tube_gap",
    "tube_radius_from_gap",
]

logger = logging.getLogger(__name__)

LOSS_WIDTH: Final = 0.003
"""Half-width of the loss Lorentzians in the normalised frequency.

``gamma_l`` of Vincetti and Rosa (2019) eq. 2.
"""

DISPERSION_WIDTH: Final = 0.03
"""Damping of the dispersive Lorentzians.

``gamma_d`` of Rosa, Melli and Vincetti (2021) eq. 2. The paper prints 0.003; earlier
implementations carry 0.03, which fits the FEM index near the resonances 3.2 (vacuum)
and 1.9 (argon) times better.
"""

RADIUS_COEFFICIENT: Final = 10.0
"""Coefficient of ``wavelength / core_radius`` in the effective radius.

The gap-penetration term of Rosa 2021 eq. 10 as printed; earlier implementations carry
20. The factor is empirical: no derivation decides between 10 and 20, and no erratum or
author confirmation has been found.
"""

MINIMUM_LOSS_PREFACTOR: Final = 3e-4
"""Prefactor of the minimum confinement loss in dB/m.

Vincetti (2016) eq. 3 as revised by the 2019 paper.
"""

DISPERSION_PREFACTOR: Final = 4.5e-7
"""Prefactor of the dispersive resonance correction, Rosa 2021 eq. 6."""

AREA_FACTOR: Final = 0.48
"""Prefactor of the fitted effective area, Rosa 2021 eq. 22.

``0.48 pi`` is Marcatili's HE11 area factor, which the fit reduces to away from
resonance.
"""

MAX_TERMS: Final = 8
"""Largest number of terms in the resonance sums.

The EH cutoff fit of the 2019 paper, eq. 5, carries ``1 / (7.75 - nu)``, so more terms
extrapolate through its pole; the papers stop at eight.
"""

VALIDATED_WALL_RATIO: Final = (0.05, 0.2)
"""Wall thickness over outer tube radius of the 2019 validation fibres."""

VALIDATED_TUBE_COUNT: Final = (6, 10)
"""Tube counts of the 2019 validation fibres."""

VALIDATED_TUBE_INDEX: Final = (1.44, 2.42)
"""Real tube indices of the 2019 validation fibres."""


@dataclass(frozen=True)
class _Conventions:
    """Name the four choices in which the papers and earlier implementations differ.

    Attributes
    ----------
    radius_coefficient : float
        Coefficient of ``wavelength / core_radius`` in the effective radius: 10 in
        Rosa 2021 eq. 10, 20 in earlier implementations.
    single_power_decrement : bool
        True divides the waveguide decrement ``(u c / (omega R_eff))^2 / 2`` by the
        gas index once, the first-order expansion of ``beta = sqrt(n_g^2 k^2 -
        (u / R)^2)`` and Rosa 2021 eqs. 21 and 23; False divides by its square, as
        earlier implementations do.
    area_over_gas_index : bool
        True keeps the ``1 / n_g`` prefactor of Rosa 2021 eq. 22 in the fitted area,
        which with the single-power decrement makes it fill independent away from
        resonance; False omits it, as earlier implementations do.
    filled_frequency : bool
        True forms the normalised frequency with the core index, ``(2 t / lambda)
        sqrt(n_t^2 - n_g^2)``, the slab resonance of a wall in the filled core and Rosa
        2021 eq. 23; False uses unity in place of ``n_g``, as earlier implementations
        do. The minimum loss keeps its printed ``n_t^2 - 1`` factors either way.
    """

    radius_coefficient: float
    single_power_decrement: bool
    area_over_gas_index: bool
    filled_frequency: bool


# The form solprop evaluates: the printed radius coefficient, the single-power
# decrement, the area's gas-index prefactor and the filled-core frequency.
CONVENTIONS: Final = _Conventions(RADIUS_COEFFICIENT, True, True, True)
# The form earlier published implementations share, for the witness tests only;
# production code never passes it.
LEGACY_CONVENTIONS: Final = _Conventions(20.0, False, False, False)


def _positive(value: float, name: str) -> None:
    """Check a positive finite dimension.

    Parameters
    ----------
    value : float
        Dimension in m.
    name : str
        Name used in the diagnostic.

    Raises
    ------
    ModeError
        If the dimension is nonpositive or nonfinite.
    """
    if not math.isfinite(value) or value <= 0:
        raise ModeError(f"{name} must be positive and finite, got {value}")


def _tube_angle(n_tubes: int) -> float:
    """Return the sine of half the angular tube spacing.

    Parameters
    ----------
    n_tubes : int
        Number of tubes, at least three.

    Returns
    -------
    float
        Sine of pi divided by the count.

    Raises
    ------
    ModeError
        If the count is invalid.
    """
    if isinstance(n_tubes, bool) or not isinstance(n_tubes, int) or n_tubes < 3:
        raise ModeError("n_tubes must be an integer >= 3")
    return math.sin(math.pi / n_tubes)


def tube_gap(core_radius: float, tube_radius: float, n_tubes: int) -> float:
    """Return the gap between neighbouring tubes.

    Parameters
    ----------
    core_radius : float
        Positive inscribed core radius in m.
    tube_radius : float
        Positive outer tube radius in m.
    n_tubes : int
        Number of tubes, at least three.

    Returns
    -------
    float
        Gap in m; negative means overlapping tubes.

    Raises
    ------
    ModeError
        If a radius is not positive and finite or the count is invalid.
    """
    _positive(core_radius, "core_radius")
    _positive(tube_radius, "tube_radius")
    return 2 * ((core_radius + tube_radius) * _tube_angle(n_tubes) - tube_radius)


def core_radius_from_gap(tube_radius: float, n_tubes: int, gap: float) -> float:
    """Return the inscribed core radius for a tube gap.

    Parameters
    ----------
    tube_radius : float
        Positive outer tube radius in m.
    n_tubes : int
        Number of tubes, at least three.
    gap : float
        Nonnegative finite gap in m.

    Returns
    -------
    float
        Core radius in m.

    Raises
    ------
    ModeError
        If the geometry is invalid.
    """
    _positive(tube_radius, "tube_radius")
    _check_gap(gap)
    return (tube_radius + gap / 2) / _tube_angle(n_tubes) - tube_radius


def tube_radius_from_gap(core_radius: float, n_tubes: int, gap: float) -> float:
    """Return the outer tube radius for an inscribed core and gap.

    Parameters
    ----------
    core_radius : float
        Positive inscribed core radius in m.
    n_tubes : int
        Number of tubes, at least three.
    gap : float
        Nonnegative finite gap in m, small enough to leave a positive tube radius.

    Returns
    -------
    float
        Outer tube radius in m.

    Raises
    ------
    ModeError
        If the geometry is invalid.
    """
    _positive(core_radius, "core_radius")
    _check_gap(gap)
    sine = _tube_angle(n_tubes)
    radius = (core_radius * sine - gap / 2) / (1 - sine)
    _positive(radius, "tube_radius implied by gap")
    return radius


def _check_gap(gap: float) -> None:
    """Check that tubes do not overlap.

    Parameters
    ----------
    gap : float
        Tube separation in m.

    Raises
    ------
    ModeError
        If the gap is negative or nonfinite.
    """
    if not math.isfinite(gap) or gap < 0:
        raise ModeError(f"tube gap must be nonnegative and finite, got {gap} m")


@dataclass(frozen=True)
class TubeLattice(Displayed):
    """Describe a single ring of identical hollow tubes.

    Attributes
    ----------
    wall_thickness : float
        Positive finite wall thickness in m, smaller than the outer radius.
    tube_radius : float
        Positive finite outer tube radius in m.
    n_tubes : int
        Number of tubes, at least three.
    n_terms : int
        Number of radial and azimuthal fit terms, between one and
        :data:`~solprop.modes.antiresonant.MAX_TERMS`; default eight, the papers'
        truncation.
    """

    wall_thickness: float
    tube_radius: float
    n_tubes: int
    n_terms: int = MAX_TERMS

    def __repr__(self) -> str:
        """Return a one-line description of the tube ring.

        Returns
        -------
        str
            For example ``TubeLattice(t=180 nm, r_ext=9 um, N=6)``.
        """
        fields: list[tuple[str, object]] = [
            ("t", engineering(self.wall_thickness, "m")),
            ("r_ext", engineering(self.tube_radius, "m")),
            ("N", self.n_tubes),
        ]
        return line("TubeLattice", fields)

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "TubeLattice: a single ring of cladding tubes", [
            ("wall thickness", engineering(self.wall_thickness, "m")),
            ("tube outer radius", engineering(self.tube_radius, "m")),
            ("tubes", self.n_tubes),
            ("resonance terms", self.n_terms),
        ]

    def __post_init__(self) -> None:
        """Validate the tube dimensions and integer counts.

        Raises
        ------
        ModeError
            If dimensions or counts cannot describe hollow tubes, or more fit terms
            are asked for than the cutoff fits support.
        """
        _positive(self.wall_thickness, "wall_thickness")
        _positive(self.tube_radius, "tube_radius")
        _tube_angle(self.n_tubes)
        if self.wall_thickness >= self.tube_radius:
            raise ModeError("wall_thickness must be smaller than tube_radius")
        if (
            isinstance(self.n_terms, bool)
            or not isinstance(self.n_terms, int)
            or not 1 <= self.n_terms <= MAX_TERMS
        ):
            raise ModeError(
                f"n_terms must be an integer between 1 and {MAX_TERMS}: the EH cutoff "
                "fit of Vincetti and Rosa (2019), eq. 5, has a pole at nu = 7.75 and "
                f"the papers stop at {MAX_TERMS}; got {self.n_terms}"
            )

    def gap(self, core_radius: float) -> float:
        """Return the tube gap at a core radius.

        Parameters
        ----------
        core_radius : float
            Inscribed core radius in m, positive and finite.

        Returns
        -------
        float
            Gap in m, negative for overlapping tubes.
        """
        return tube_gap(core_radius, self.tube_radius, self.n_tubes)

    @classmethod
    def from_inner_diameter_ratio(
        cls,
        core_radius: float,
        wall_thickness: float,
        inner_diameter_ratio: float,
        n_tubes: int,
    ) -> "TubeLattice":  # noqa: UP037 - beartype evaluates the class body
        """Construct a lattice from the inner tube to core diameter ratio.

        Parameters
        ----------
        core_radius : float
            Positive inscribed core radius in m; the core diameter is twice this.
        wall_thickness : float
            Positive tube wall thickness in m.
        inner_diameter_ratio : float
            Positive finite d/D, where d is the tube's **inner** diameter and D the
            inscribed core diameter (some papers quote d/D with the outer diameter).
        n_tubes : int
            Number of resonators, at least three.

        Returns
        -------
        TubeLattice
            Lattice with outer tube radius
            ``core_radius * inner_diameter_ratio + wall_thickness``.

        Raises
        ------
        ModeError
            If dimensions are invalid or neighbouring tubes overlap.
        """
        _positive(core_radius, "core_radius")
        _positive(inner_diameter_ratio, "inner_diameter_ratio")
        lattice = cls(
            wall_thickness, core_radius * inner_diameter_ratio + wall_thickness, n_tubes
        )
        _check_gap(lattice.gap(core_radius))
        return lattice

    @classmethod
    def from_gap(
        cls, core_radius: float, gap: float, n_tubes: int, wall_thickness: float
    ) -> "TubeLattice":  # noqa: UP037 - beartype evaluates the class body
        """Construct a lattice from a core radius and gap.

        Parameters
        ----------
        core_radius : float
            Positive inscribed core radius in m.
        gap : float
            Nonnegative gap in m.
        n_tubes : int
            Number of tubes, at least three.
        wall_thickness : float
            Positive tube wall thickness in m.

        Returns
        -------
        TubeLattice
            Lattice with the implied outer radius and eight fit terms.
        """
        return cls(
            wall_thickness, tube_radius_from_gap(core_radius, n_tubes, gap), n_tubes
        )

    def core_radius_for(self, gap: float) -> float:
        """Return the core radius for a gap.

        Parameters
        ----------
        gap : float
            Nonnegative finite tube gap in m.

        Returns
        -------
        float
            Inscribed core radius in m.
        """
        return core_radius_from_gap(self.tube_radius, self.n_tubes, gap)


@dataclass(frozen=True)
class _Spectral:
    """Carry a value and its analytic frequency derivative through the fits.

    Product, quotient and power rules are evaluated together with the expression,
    including the logarithmic derivative of tube-index-dependent exponents. This
    keeps each fit coefficient in one place without numerical differentiation. The
    leading axes carry the fit's azimuthal index where the sums broadcast over it;
    a constant operand may be a scalar or an array that broadcasts the same way.
    """

    value: Inexact[np.ndarray, "... n_omega"]
    slope: Inexact[np.ndarray, "... n_omega"]
    # NumPy defers to the reflected operators, so an array on the left of a product
    # or a sum broadcasts into the value and slope instead of becoming an object
    # array of _Spectral elements.
    __array_ufunc__ = None

    def __add__(
        self,
        other: "_Spectral | complex | Inexact[np.ndarray, '...']",  # noqa: UP037 - beartype evaluates before class binding
    ) -> "_Spectral":  # noqa: UP037 - beartype evaluates before class binding
        """Apply the sum rule.

        Parameters
        ----------
        other : _Spectral, complex or numpy.ndarray
            Other operand with its slope, or a frequency-independent constant,
            scalar or broadcastable.

        Returns
        -------
        _Spectral
            Resulting value and analytic frequency slope.
        """
        if isinstance(other, _Spectral):
            return _Spectral(self.value + other.value, self.slope + other.slope)
        return _Spectral(self.value + other, self.slope)

    __radd__ = __add__

    def __neg__(self) -> "_Spectral":  # noqa: UP037 - beartype evaluates before class binding
        """Negate value and slope.

        Returns
        -------
        _Spectral
            Resulting value and analytic frequency slope.
        """
        return _Spectral(-self.value, -self.slope)

    def __sub__(
        self,
        other: "_Spectral | complex | Inexact[np.ndarray, '...']",  # noqa: UP037 - beartype evaluates before class binding
    ) -> "_Spectral":  # noqa: UP037 - beartype evaluates before class binding
        """Subtract a spectral quantity.

        Parameters
        ----------
        other : _Spectral, complex or numpy.ndarray
            Other operand with its slope, or a frequency-independent constant,
            scalar or broadcastable.

        Returns
        -------
        _Spectral
            Resulting value and analytic frequency slope.
        """
        return self + (-other)

    def __rsub__(self, other: "complex | Inexact[np.ndarray, '...']") -> "_Spectral":  # noqa: UP037 - beartype evaluates before class binding
        """Subtract from a constant.

        Parameters
        ----------
        other : _Spectral, complex or numpy.ndarray
            Other operand with its slope, or a frequency-independent constant,
            scalar or broadcastable.

        Returns
        -------
        _Spectral
            Resulting value and analytic frequency slope.
        """
        return -self + other

    def __mul__(
        self,
        other: "_Spectral | complex | Inexact[np.ndarray, '...']",  # noqa: UP037 - beartype evaluates before class binding
    ) -> "_Spectral":  # noqa: UP037 - beartype evaluates before class binding
        """Apply the product rule.

        Parameters
        ----------
        other : _Spectral, complex or numpy.ndarray
            Other operand with its slope, or a frequency-independent constant,
            scalar or broadcastable.

        Returns
        -------
        _Spectral
            Resulting value and analytic frequency slope.
        """
        if isinstance(other, _Spectral):
            return _Spectral(
                self.value * other.value,
                self.slope * other.value + self.value * other.slope,
            )
        return _Spectral(self.value * other, self.slope * other)

    __rmul__ = __mul__

    def __truediv__(
        self,
        other: "_Spectral | complex | Inexact[np.ndarray, '...']",  # noqa: UP037 - beartype evaluates before class binding
    ) -> "_Spectral":  # noqa: UP037 - beartype evaluates before class binding
        """Apply the quotient rule.

        Parameters
        ----------
        other : _Spectral, complex or numpy.ndarray
            Other operand with its slope, or a frequency-independent constant,
            scalar or broadcastable.

        Returns
        -------
        _Spectral
            Resulting value and analytic frequency slope.
        """
        if isinstance(other, _Spectral):
            value = self.value / other.value
            return _Spectral(value, (self.slope - value * other.slope) / other.value)
        return _Spectral(self.value / other, self.slope / other)

    def __rtruediv__(
        self,
        other: "complex | Inexact[np.ndarray, '...']",  # noqa: UP037 - beartype evaluates before class binding
    ) -> "_Spectral":  # noqa: UP037 - beartype evaluates before class binding
        """Divide a constant by a spectral quantity.

        Parameters
        ----------
        other : _Spectral, complex or numpy.ndarray
            Other operand with its slope, or a frequency-independent constant,
            scalar or broadcastable.

        Returns
        -------
        _Spectral
            Resulting value and analytic frequency slope.
        """
        return (self**-1) * other

    def __pow__(self, power: float) -> "_Spectral":  # noqa: UP037 - beartype evaluates before class binding
        """Apply the constant-exponent power rule.

        Parameters
        ----------
        power : float
            Constant exponent.

        Returns
        -------
        _Spectral
            Resulting value and analytic frequency slope.
        """
        return _Spectral(
            self.value**power, power * self.value ** (power - 1) * self.slope
        )

    def __rpow__(self, base: float) -> "_Spectral":  # noqa: UP037 - beartype evaluates before class binding
        """Differentiate a frequency-dependent exponent.

        Parameters
        ----------
        base : float
            Positive constant base.

        Returns
        -------
        _Spectral
            Resulting value and analytic frequency slope.
        """
        value = base**self.value
        return _Spectral(value, value * math.log(base) * self.slope)

    def exp(self) -> "_Spectral":  # noqa: UP037 - beartype evaluates before class binding
        """Apply the exponential chain rule.

        Returns
        -------
        _Spectral
            Resulting value and analytic frequency slope.
        """
        value = np.exp(self.value)
        return _Spectral(value, value * self.slope)


@dataclass(frozen=True)
class _SpectralFit:
    """Hold one evaluation of the fits, each piece with its analytic slope.

    Attributes
    ----------
    refractivity : _Spectral
        ``n_eff - 1`` with the model's own loss, before the loss policy and the floor.
    core_index : _Spectral
        The gas index ``n_g`` from the parent's refractivity route; complex for an
        absorbing species.
    effective_radius : _Spectral
        ``R_eff`` in m.
    correction : _Spectral
        The dispersive resonance term ``delta n``.
    decrement : _Spectral
        The waveguide decrement, so that ``Re n_eff = n_g - decrement + correction``
        and the fitted area's denominator is ``decrement - correction``.
    """

    refractivity: _Spectral
    core_index: _Spectral
    effective_radius: _Spectral
    correction: _Spectral
    decrement: _Spectral


@dataclass(frozen=True, eq=False)
class _VincettiTerms(MarcatiliTerms):
    """Extend the parent samples with tube material dispersion.

    Attributes
    ----------
    tube_index : numpy.ndarray
        Real tube index, shape ``(n_omega,)``.
    dtube_index_domega : numpy.ndarray
        Tube index derivative in s/rad, shape ``(n_omega,)``.
    wavelength : numpy.ndarray
        Vacuum wavelength in m, shape ``(n_omega,)``.
    """

    tube_index: Float[np.ndarray, " n_omega"]
    dtube_index_domega: Float[np.ndarray, " n_omega"]
    wavelength: Inexact[np.ndarray, " n_omega"]


def _cutoffs(
    mu: Float[np.ndarray, "n_terms 1"], nu: int, ratio: float, tube: _Spectral
) -> tuple[_Spectral, _Spectral]:
    """Evaluate the HE and EH resonance fits of the 2019 paper, eqs. 4-5.

    Parameters
    ----------
    mu : numpy.ndarray
        Azimuthal tube-mode indices, positive, as a column so the cutoffs come out
        with one row per index.
    nu : int
        Radial tube-mode index, positive.
    ratio : float
        Dimensionless wall thickness divided by outer tube radius.
    tube : _Spectral
        Real tube index and its slope in s/rad.

    Returns
    -------
    tuple of _Spectral
        HE and EH dimensionless cutoff frequencies and their slopes, shape
        ``(n_terms, n_omega)``.
    """
    if nu == 1:
        he = (
            np.abs(0.21 + 0.175 * mu - 0.1 / (mu - 0.35) ** 2)
            * ratio ** (0.55 + 0.005 * (tube**4 - 1) ** 0.5)
            + 0.04 * np.sqrt(mu) * ratio
        )
        eh = (0.73 + 0.57 * (mu**0.8 + 1.5) / 4 - 0.04 / (mu - 0.35)) * ratio ** (
            0.5 - (tube - 1) / (10 * (mu + 0.5) ** 0.1)
        )
    else:
        he = 0.3 / tube**0.3 * (2 / nu) ** 1.2 * np.abs(mu - 0.8) * ratio + nu - 1
        eh = (
            (
                11.5
                / (nu**1.2 * (7.75 - nu))
                * (0.34 + mu / 4 * (tube / 1.2) ** 1.15)
                / (mu + 0.2 / tube) ** 0.15
                * ratio
                ** (0.75 + 0.06 / tube**1.15 + 0.1 * (1.44 / tube) ** 0.5 * (nu - 2))
            )
            + nu
            - 1
        )
    return he, eh


def _resonance_sums(
    frequency: _Spectral, tube: _Spectral, lattice: TubeLattice
) -> tuple[_Spectral, _Spectral]:
    """Sum loss resonances and the fundamental-mode dispersion correction.

    Parameters
    ----------
    frequency : _Spectral
        Dimensionless normalised frequency and its slope.
    tube : _Spectral
        Real tube index and its slope.
    lattice : TubeLattice
        Wall, tube dimensions and truncation count.

    Returns
    -------
    tuple of _Spectral
        Weighted loss and dispersion sums with analytic slopes.
    """
    ratio = lattice.wall_thickness / lattice.tube_radius
    # One row per azimuthal index: the cutoffs and Lorentzians of every mu are
    # evaluated at once, element by element as the papers write them, and summed
    # in a fixed order (nu outer, mu inner, HE before EH) on the plain arrays.
    mu = np.arange(1, lattice.n_terms + 1, dtype=float)[:, None]
    # The weight of the 2019 paper's eq. 3; the 2021 correction keeps mu = 1.
    amplitude = 2e3 * np.exp(-0.05 * np.abs(mu - 1) ** 2.6)
    zero = np.zeros_like(frequency.value)
    loss_value, loss_slope = zero, zero
    dispersion_value, dispersion_slope = zero, zero
    for nu in range(1, lattice.n_terms + 1):
        radial_value, radial_slope = zero, zero
        lorentzians = []
        for cutoff in _cutoffs(mu, nu, ratio, tube):
            lorentzians.append(
                amplitude * LOSS_WIDTH**2 / (LOSS_WIDTH**2 + (frequency - cutoff) ** 2)
            )
            detuning = cutoff**2 - frequency**2
            dispersive = (
                amplitude
                * detuning
                / (detuning**2 + (DISPERSION_WIDTH * frequency) ** 2)
            )
            dispersion_value = dispersion_value + dispersive.value[0]
            dispersion_slope = dispersion_slope + dispersive.slope[0]
        for row in range(lattice.n_terms):
            for lorentzian in lorentzians:
                radial_value = radial_value + lorentzian.value[row]
                radial_slope = radial_slope + lorentzian.slope[row]
        loss_value, loss_slope = loss_value + radial_value, loss_slope + radial_slope
    return _Spectral(loss_value, loss_slope), _Spectral(
        dispersion_value, dispersion_slope
    )


@dataclass(frozen=True, kw_only=True, repr=False)
class VincettiMode(MarcatiliMode):
    """Model a single-ring tube-lattice fibre with the Vincetti fits.

    All inherited capillary parameters retain their meaning. ``cladding`` is the
    tube index; its real part enters the fits. Only ``model="full"`` is accepted.
    Loss follows the parent policy and positive extinction attenuates power. The
    real index is floored at ``EVANESCENT_INDEX_FLOOR`` with zero derivative where
    the floor binds. The protocol area and transverse fields remain Marcatili's.
    The gas index enters the normalised frequency and the waveguide decrement
    through the parent's refractivity route, so an absorbing species carries its
    small imaginary index into both, as the capillary carries it into its square
    root.

    Attributes
    ----------
    lattice : TubeLattice
        Tube wall, outer radius, count and number of fit terms.

    References
    ----------
    .. [1] Vincetti, Opt. Express 24, 10313 (2016), doi:10.1364/OE.24.010313.
    .. [2] Vincetti and Rosa, Opt. Express 27, 5230 (2019), doi:10.1364/OE.27.005230.
    .. [3] Rosa et al., Fibers 9, 58 (2021), doi:10.3390/fib9100058.
    """

    lattice: TubeLattice

    def _check_arguments(self) -> None:
        """Validate the parent inputs and the tube geometry.

        Raises
        ------
        ModeError
            If the reduced model is requested or a declared radius gives
            overlapping tubes.
        """
        super()._check_arguments()
        if self.model == "reduced":
            raise ModeError('VincettiMode does not support model="reduced"')
        path = as_profile(self.radius)
        positions = np.unique(np.concatenate(([0.0], path.breakpoints)))
        for radius in path(positions):
            _check_gap(self.lattice.gap(float(radius)))
        if (self.kind, self.n, self.m) != ("HE", 1, 1):
            logger.warning(
                "Vincetti loss and dispersion corrections are fundamental-mode fits"
            )
        # The ranges are the geometries of [2-3], Table 1, not a guarantee outside
        # their sampled frequencies: they are sampled geometries, not a proven fit
        # domain. Logged at every construction, so a ``dataclasses.replace``
        # repeats the message.
        outside = []
        ratio = self.lattice.wall_thickness / self.lattice.tube_radius
        low, high = VALIDATED_WALL_RATIO
        if not low <= ratio <= high:
            outside.append(
                f"wall_thickness/tube_radius={ratio:g} outside [{low}, {high}]"
            )
        low, high = VALIDATED_TUBE_COUNT
        if not low <= self.lattice.n_tubes <= high:
            outside.append(f"n_tubes={self.lattice.n_tubes} outside [{low}, {high}]")
        low, high = VALIDATED_TUBE_INDEX
        if (
            self.cladding is not None
            and not isinstance(self.cladding, GlassIndex)
            and not low <= self.cladding.real <= high
        ):
            outside.append(f"tube index={self.cladding.real:g} outside [{low}, {high}]")
        if outside:
            logger.warning(
                "Vincetti geometry outside published validation ranges: %s",
                "; ".join(outside),
            )

    def sampled_terms(self, omega: Float[np.ndarray, " n_omega"]) -> _VincettiTerms:
        """Sample gas and tube dispersion at positive angular frequencies.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive finite angular frequencies in rad/s, shape ``(n_omega,)``.

        Returns
        -------
        _VincettiTerms
            The parent's samples extended with the tube index, its analytic
            derivative and the vacuum wavelength.

        Raises
        ------
        ModeError
            If frequencies are invalid or the real tube index is not greater than one.
        """
        parent = super().sampled_terms(omega)
        if isinstance(self.cladding_index, GlassIndex):
            tube = np.asarray(np.real(self.cladding_index.index(omega)))
            slope = np.asarray(np.real(self.cladding_index.dindex_domega(omega)))
        else:
            tube = np.full(omega.shape, self.cladding_index.real)
            slope = np.zeros_like(omega)
        if not np.all(np.isfinite(tube) & (tube > 1)):
            raise ModeError(
                "Vincetti fits require a finite real tube index greater than one"
            )
        return _VincettiTerms(
            parent.omega,
            parent.u_nm,
            parent.core_susceptibility,
            parent.dcore_susceptibility_domega,
            parent.cladding_term,
            parent.dcladding_term_domega,
            tube,
            slope,
            np.asarray(2 * math.pi * C / parent.omega),
        )

    def _spectral_model(
        self,
        terms: _VincettiTerms,
        radius: float,
        density: float | Float[np.ndarray, " n_species"],
        conventions: _Conventions = CONVENTIONS,
    ) -> _SpectralFit:
        """Evaluate the empirical index, radius and dispersion fits together.

        Parameters
        ----------
        terms : _VincettiTerms
            Frequency samples, including both material slopes.
        radius : float
            Core radius in m.
        density : float or numpy.ndarray
            Number density in m^-3, scalar or per species.
        conventions : _Conventions, optional
            Which form of the disputed terms to evaluate; solprop's by default.

        Returns
        -------
        _SpectralFit
            Refractivity, gas index, effective radius, dispersion correction and
            waveguide decrement, each paired with its analytic frequency derivative.
        """
        # Zeroing only the Bessel root uses precisely the parent's susceptibility
        # reduction and stable refractivity arithmetic, including complex species.
        gas_value, gas_slope = super()._model_values(
            replace(terms, u_nm=0.0), radius, density
        )
        gas = _Spectral(gas_value, gas_slope)
        core = gas + 1
        omega = _Spectral(terms.omega, np.ones_like(terms.omega))
        wavelength = _Spectral(terms.wavelength, -terms.wavelength / terms.omega)
        tube = _Spectral(terms.tube_index, terms.dtube_index_domega)
        wall, outer = self.lattice.wall_thickness, self.lattice.tube_radius
        ratio = wall / outer
        # The wall's contrast against vacuum stays in the 2016 minimum loss, whose
        # fit is the evacuated one; the resonance positions move with the fill.
        contrast = tube**2 - 1
        if conventions.filled_frequency:
            frequency = 2 * wall / wavelength * (tube**2 - core**2) ** 0.5
        else:
            frequency = 2 * wall / wavelength * contrast**0.5
        loss_sum, dispersion_sum = _resonance_sums(frequency, tube, self.lattice)
        minimum = (
            MINIMUM_LOSS_PREFACTOR
            * wavelength**4.5
            / radius**4
            * (1 - ratio) ** -12
            * contrast**0.5
            / (wall * math.sqrt(outer))
            * (2 * wavelength / (outer * contrast)).exp()
        )
        confinement = minimum * loss_sum
        effective_radius = (1.027 + 0.001 * (frequency + 2 / frequency**4)) * (
            radius**2
            + self.lattice.n_tubes
            / math.pi
            * 3
            / 64
            * outer**2
            * (
                1
                + (3 + conventions.radius_coefficient * wavelength / radius)
                * self.lattice.gap(radius)
                / outer
            )
        ) ** 0.5
        correction = (
            DISPERSION_PREFACTOR
            * (1 - ratio) ** -4
            * (wavelength / radius) ** 2
            * dispersion_sum
        )
        # The first-order expansion of beta = sqrt(n_g^2 k^2 - (u / R_eff)^2) gives
        # n_eff = n_g - (u c / (omega R_eff))^2 / (2 n_g); LEGACY_CONVENTIONS divides by
        # n_g once more.
        waveguide = 0.5 * (self.u_nm * C / (omega * effective_radius)) ** 2
        decrement = (
            waveguide / core
            if conventions.single_power_decrement
            else waveguide / core**2
        )
        refractivity = (
            gas
            - decrement
            + correction
            + 1j * C * math.log(10) / 20 * confinement / omega
        )
        return _SpectralFit(refractivity, core, effective_radius, correction, decrement)

    def _model_values(
        self,
        terms: MarcatiliTerms,
        radius: float,
        density: float | Float[np.ndarray, " n_species"],
    ) -> tuple[Inexact[np.ndarray, " n_omega"], Inexact[np.ndarray, " n_omega"]]:
        """Return refractivity and slope before the inherited loss policy.

        Parameters
        ----------
        terms : MarcatiliTerms
            Samples from this mode, carrying its tube material.
        radius : float
            Core radius in m.
        density : float or numpy.ndarray
            Number density in m^-3, scalar or per species.

        Returns
        -------
        tuple of numpy.ndarray
            Refractivity and derivative in s/rad, shape ``(n_omega,)``.

        Raises
        ------
        ModeError
            If samples were taken from another model.
        """
        if not isinstance(terms, _VincettiTerms):
            raise ModeError("VincettiMode requires its own sampled_terms")
        fit = self._spectral_model(terms, radius, density)
        return fit.refractivity.value, fit.refractivity.slope

    def effective_area(
        self,
        omega: Float[np.ndarray, " n_omega"],
        z: float = 0.0,
        *,
        conventions: _Conventions = CONVENTIONS,
    ) -> Float[np.ndarray, " n_omega"]:
        """Return the fitted spectral area, distinct from the protocol area.

        Rosa 2021 eq. 22: ``0.48 / (8 pi n_g) (u lambda)^2 / (n_g - Re n_eff)``,
        which away from resonance reduces to ``0.48 pi R_eff^2``, independent of the
        fill. The fit can diverge or become negative near resonances where its index
        denominator crosses zero, and the denominator is taken real for an absorbing
        species. It is not the area used for propagation.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive angular frequencies in rad/s, shape ``(n_omega,)``.
        z : float, optional
            Position in m.
        conventions : _Conventions, optional
            Which form of the disputed terms to evaluate; solprop's by default.
            ``LEGACY_CONVENTIONS`` gives the area of earlier implementations, for the
            witness tests.

        Returns
        -------
        numpy.ndarray
            Fitted effective area in m^2, shape ``(n_omega,)``.
        """
        terms = self.sampled_terms(omega)
        fit = self._spectral_model(terms, *self._scalars(z), conventions=conventions)
        area = (
            AREA_FACTOR
            / (8 * math.pi)
            * (self.u_nm * 2 * math.pi * C / omega) ** 2
            / np.real(fit.decrement.value - fit.correction.value)
        )
        if conventions.area_over_gas_index:
            area = area / np.real(fit.core_index.value)
        return np.asarray(area)

    def _area_rows(self) -> list[tuple[str, object]]:
        """Return no area row: this mode's area is not a constant of the geometry.

        Returns
        -------
        list of tuple
            Empty.
        """
        return []

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        title, rows = super()._describe()
        lattice = self.lattice
        wall = engineering(lattice.wall_thickness, "m")
        outer = engineering(lattice.tube_radius, "m")
        tubes = f"{lattice.n_tubes} tubes, wall {wall}, outer radius {outer}"
        kept = [row for row in rows if row[0] != "model"]
        return title, [*kept, ("lattice", tubes)]

    def __repr__(self) -> str:
        """Return the mode family, tube geometry and fill.

        Returns
        -------
        str
            For example ``VincettiMode(HE11, a=15 um, t=180 nm, r_ext=9 um, N=6,
            Ar 1 bar, loss=True)``; a gradient shows its pressure at ``z = 0``.
        """
        radius = float(self.core_radius(np.array([0.0]))[0])
        if self.gas is None:
            fill = "evacuated" if self.mixture is None else "mixture"
        else:
            pressure = float(as_profile(self.pressure)(np.array([0.0]))[0])
            fill = f"{self.gas} {pressure:g} bar"
        return (
            f"VincettiMode({self.kind}{self.n}{self.m}, a={radius * 1e6:g} um, "
            f"t={self.lattice.wall_thickness * 1e9:g} nm, "
            f"r_ext={self.lattice.tube_radius * 1e6:g} um, "
            f"N={self.lattice.n_tubes}, {fill}, loss={self.loss!r})"
        )
