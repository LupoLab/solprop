# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build waveguide modes from loaded effective-index arrays.

An evacuated table gains its fill through ``n_eff**2 = n_v**2 + chi_g``;
a filled table already contains the whole linear response. Both hollow modes
retain Marcatili's transverse field. See ``docs/guide/waveguides/tabulated-modes.md``.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from jaxtyping import Complex, Float, Inexact
from numpy.typing import ArrayLike
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from solprop._display import Displayed
from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.modes.capillary import MarcatiliMode, MarcatiliTerms, _checked_omega
from solprop.modes.protocol import Mode
from solprop.modes.stepindex import StepIndexMode
from solprop.profiles import as_profile

__all__ = [
    "MIN_TABLE_POINTS",
    "IndexTable",
    "TabulatedHollowMode",
    "TabulatedMode",
    "sample_index_table",
]

MIN_TABLE_POINTS = 4
"""Fewest samples required by the house cubic interpolant."""

# Ulp margin admitted at a table end before a query is clamped onto the end
# node. A query that came from a wavelength is not the frequency it was
# converted from: 2 pi C / (2 pi C / omega) differs from omega for about five
# per cent of doubles, in either direction. Four ulp covers both roundings of
# such a round trip with a margin, and is far below any spline resolution.
_END_ULP = 4

logger = logging.getLogger(__name__)


def _relaxed_end(value: float, direction: float) -> float:
    """Shift a table end by the admitted ulp margin.

    Parameters
    ----------
    value : float
        End node in rad/s.
    direction : float
        Sign of the shift, ``-numpy.inf`` below the node or ``numpy.inf`` above.

    Returns
    -------
    float
        The node moved ``_END_ULP`` representable steps in that direction.
    """
    result = float(value)
    for _ in range(_END_ULP):
        result = float(np.nextafter(result, direction))
    return result


def _positive_vector(values: ArrayLike, name: str) -> Float[np.ndarray, " n_nodes"]:
    """Copy a positive finite vector.

    Parameters
    ----------
    values : array_like
        One-dimensional coordinates or areas in the caller's SI units.
    name : str
        Quantity named in an error.

    Returns
    -------
    numpy.ndarray
        Positive finite float64 values, shape ``(n_nodes,)``.

    Raises
    ------
    ModeError
        If the input is not a positive finite vector.
    """
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 1 or not np.all(np.isfinite(result) & (result > 0)):
        raise ModeError(f"{name} must be a one-dimensional positive finite array")
    return result


@dataclass(frozen=True, eq=False, repr=False, init=False)
class IndexTable(Displayed):
    """Interpolate a passive mode's complex index and optional effective area.

    Real refractivity and area use cubic splines in angular frequency. Loss uses
    a cubic spline of its logarithm by default; ``linear`` means a cubic spline
    of the imaginary index itself, with negative interpolated values clamped.
    No evaluation extrapolates. Constructor arrays are sorted and copied.

    Parameters
    ----------
    omega : array_like
        Positive finite angular frequencies in rad/s, shape ``(n_nodes,)``,
        at least four distinct samples, in either order.
    neff_values : array_like
        Finite complex effective indices, shape ``(n_neff_nodes,)`` matching
        ``omega``. Imaginary parts must be non-negative (strictly positive for
        logarithmic interpolation). Stored as read-only complex128.
    area : array_like or None, optional
        Positive finite effective areas in m^2, shape ``(n_area_nodes,)`` matching
        ``omega``, or no area. Stored as read-only float64.
    loss_interpolation : {'log', 'linear'}, optional
        Imaginary-index interpolation policy, default ``'log'``.

    Examples
    --------
    >>> table = IndexTable.from_wavelength(
    ...     [500e-9, 600e-9, 700e-9, 800e-9], [1.45 + 1e-9j] * 4)
    >>> table.neff(table.omega).real.tolist()
    [1.45, 1.45, 1.45, 1.45]
    """

    omega: Float[np.ndarray, " n_nodes"]
    neff_values: Complex[np.ndarray, " n_neff_nodes"]
    area: Float[np.ndarray, " n_area_nodes"] | None = None
    loss_interpolation: Literal["log", "linear"] = "log"
    _real: CubicSpline | CubicHermiteSpline = field(init=False, repr=False)
    _imaginary: CubicSpline | CubicHermiteSpline = field(init=False, repr=False)
    _area: CubicSpline | None = field(init=False, repr=False)
    _clamp_logged: bool = field(init=False, default=False, repr=False)

    def __init__(
        self,
        omega: ArrayLike,
        neff_values: ArrayLike,
        area: ArrayLike | None = None,
        loss_interpolation: Literal["log", "linear"] = "log",
    ) -> None:
        """Accept array-like inputs while storing precisely typed immutable arrays.

        Parameters
        ----------
        omega : array_like
            Positive frequencies in rad/s, shape ``(n_nodes,)``.
        neff_values : array_like
            Passive indices, shape ``(n_neff_nodes,)`` matching the frequencies.
        area : array_like or None, optional
            Positive areas in m^2, shape ``(n_area_nodes,)`` matching the frequencies.
        loss_interpolation : {'log', 'linear'}, optional
            Imaginary-index interpolation policy.
        """
        # Constructor fields remain init=True for dataclass serialization, while
        # independent stored-axis names leave production shape validation reachable.
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "neff_values", neff_values)
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "loss_interpolation", loss_interpolation)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate, sort and freeze the samples, then build the interpolants.

        Raises
        ------
        ModeError
            If samples disagree in shape, are non-finite, duplicate, too few,
            or violate the index, area or loss-interpolation requirements.
        """
        omega = _positive_vector(self.omega, "omega")
        values = np.array(self.neff_values, dtype=np.complex128, copy=True)
        if omega.shape != values.shape:
            raise ModeError("omega and neff_values must have the same shape")
        if omega.size < MIN_TABLE_POINTS:
            raise ModeError(f"a cubic spline needs at least {MIN_TABLE_POINTS} points")
        if not np.all(np.isfinite(values)):
            raise ModeError("neff_values must be finite")
        if np.any(values.imag < 0):
            raise ModeError(
                "negative imaginary index declares gain; a table must be passive"
            )
        if self.loss_interpolation not in ("log", "linear"):
            raise ModeError("loss_interpolation must be 'log' or 'linear'")
        if self.loss_interpolation == "log" and np.any(values.imag <= 0):
            rows = np.flatnonzero(values.imag <= 0).tolist()
            raise ModeError(
                f"log loss interpolation needs positive imaginary index; rows {rows} "
                'are non-positive; use loss_interpolation="linear"'
            )
        order = np.argsort(omega)
        omega, values = omega[order], values[order]
        if np.any(np.diff(omega) == 0):
            raise ModeError("duplicate frequencies are refused")
        area = None if self.area is None else _positive_vector(self.area, "area")
        if area is not None:
            if area.shape != omega.shape:
                raise ModeError("omega and area must have the same shape")
            area = area[order]
            area.setflags(write=False)
        omega.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "neff_values", values)
        object.__setattr__(self, "area", area)
        object.__setattr__(
            self, "_real", CubicSpline(omega, values.real - 1, extrapolate=False)
        )
        imaginary = (
            np.log(values.imag) if self.loss_interpolation == "log" else values.imag
        )
        object.__setattr__(
            self, "_imaginary", CubicSpline(omega, imaginary, extrapolate=False)
        )
        object.__setattr__(
            self,
            "_area",
            None if area is None else CubicSpline(omega, area, extrapolate=False),
        )

    @classmethod
    def from_wavelength(
        cls,
        wavelength: ArrayLike,
        neff: ArrayLike,
        *,
        area: ArrayLike | None = None,
        loss_interpolation: Literal["log", "linear"] = "log",
    ) -> "IndexTable":  # noqa: UP037 - runtime type hook resolves the class during construction
        """Build a table from wavelengths and complex indices in either order.

        Parameters
        ----------
        wavelength : array_like
            Positive finite wavelengths in m, shape ``(n_nodes,)``.
        neff : array_like
            Passive complex indices, shape ``(n_neff_nodes,)`` matching wavelengths.
        area : array_like or None, optional
            Positive effective areas in m^2 at the same samples.
        loss_interpolation : {'log', 'linear'}, optional
            Imaginary-index interpolation policy.

        Returns
        -------
        IndexTable
            Validated table sorted by increasing angular frequency.

        Raises
        ------
        ModeError
            If coordinates or samples violate the table requirements.
        """
        wavelength = _positive_vector(wavelength, "wavelength")
        return cls(2 * np.pi * C / wavelength, neff, area, loss_interpolation)

    @classmethod
    def from_loss(
        cls,
        wavelength: ArrayLike,
        index: ArrayLike,
        loss_db_per_m: ArrayLike,
        *,
        area: ArrayLike | None = None,
        loss_interpolation: Literal["log", "linear"] = "log",
    ) -> "IndexTable":  # noqa: UP037 - runtime type hook resolves the class during construction
        """Build a table from wavelengths, real indices and power loss in dB/m.

        Parameters
        ----------
        wavelength : array_like
            Positive wavelengths in m, shape ``(n_nodes,)``.
        index : array_like
            Finite real indices at the wavelengths, same shape.
        loss_db_per_m : array_like
            Finite non-negative power loss in dB/m, same shape.
        area : array_like or None, optional
            Positive effective areas in m^2, same shape.
        loss_interpolation : {'log', 'linear'}, optional
            Imaginary-index interpolation policy.

        Returns
        -------
        IndexTable
            Table with ``Im n = wavelength * loss * ln(10) / (40 pi)``.

        Raises
        ------
        ModeError
            If samples differ in shape, are complex, non-finite or declare gain.
        """
        wavelength = _positive_vector(wavelength, "wavelength")
        real, loss = np.asarray(index), np.asarray(loss_db_per_m)
        if real.shape != wavelength.shape or loss.shape != wavelength.shape:
            raise ModeError("wavelength, index and loss must have the same shape")
        if np.iscomplexobj(real) or np.iscomplexobj(loss):
            raise ModeError("index and loss must be real arrays")
        if not np.all(np.isfinite(loss) & (loss >= 0)):
            raise ModeError("loss must be finite and non-negative")
        # Intensity attenuation alpha = ln(10) L/10 = 2 omega Im(n)/c.
        return cls.from_wavelength(
            wavelength,
            real + 1j * wavelength * loss * np.log(10) / (40 * np.pi),
            area=area,
            loss_interpolation=loss_interpolation,
        )

    @property
    def wavelength_range(self) -> tuple[float, float]:
        """Return the inclusive wavelength range in metres.

        Returns
        -------
        tuple of float
            Shortest and longest wavelength, m.
        """
        omega = np.asarray(self.omega)
        return float(2 * np.pi * C / omega[-1]), float(2 * np.pi * C / omega[0])

    def _checked(self, omega: ArrayLike) -> Float[np.ndarray, " n_omega"]:
        """Require a positive finite vector inside the table's support.

        Queries within ``_END_ULP`` (four) of either end node are admitted and
        clamped onto it, so that a frequency converted from a table-end
        wavelength evaluates instead of landing outside the splines.

        Parameters
        ----------
        omega : array_like
            Angular frequencies in rad/s, shape ``(n_omega,)``.

        Returns
        -------
        numpy.ndarray
            Validated frequencies in rad/s, inside the table's support.

        Raises
        ------
        ModeError
            If frequencies are invalid or outside the table.
        """
        values = _positive_vector(omega, "omega")
        nodes = np.asarray(self.omega)
        if values.size and (
            values.min() < _relaxed_end(nodes[0], -np.inf)
            or values.max() > _relaxed_end(nodes[-1], np.inf)
        ):
            low, high = self.wavelength_range
            raise ModeError(
                f"requested {values.min():.9e} to {values.max():.9e} rad/s "
                f"({2 * np.pi * C / values.max() * 1e9:.9g} to "
                f"{2 * np.pi * C / values.min() * 1e9:.9g} nm), outside table "
                f"{nodes[0]:.9e} to {nodes[-1]:.9e} rad/s "
                f"({low * 1e9:.9g} to {high * 1e9:.9g} nm); wavelength_limits "
                "must lie inside the table, including the active-band collars"
            )
        if values.size:
            np.clip(values, nodes[0], nodes[-1], out=values)
        return values

    def refractivity(
        self, omega: ArrayLike
    ) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
        """Return refractivity and its exact interpolant derivative.

        Parameters
        ----------
        omega : array_like
            Positive finite angular frequencies in rad/s, shape ``(n_omega,)``.

        Returns
        -------
        tuple of numpy.ndarray
            Complex ``n - 1`` and ``dn/domega`` in s/rad, each shape ``(n_omega,)``.

        Raises
        ------
        ModeError
            If frequencies are invalid or outside the table's support.
        """
        values = self._checked(omega)
        imaginary = np.asarray(self._imaginary(values))
        slope = np.asarray(self._imaginary(values, 1))
        if self.loss_interpolation == "log":
            imaginary = np.exp(imaginary)
            slope = imaginary * slope
        else:
            negative = imaginary < 0
            if np.any(negative):
                if not self._clamp_logged:
                    logger.warning(
                        "%s: negative interpolated loss clamped to zero "
                        "with its derivative",
                        self,
                    )
                    object.__setattr__(self, "_clamp_logged", True)
                imaginary = np.where(negative, 0.0, imaginary)
                slope = np.where(negative, 0.0, slope)
        return np.asarray(self._real(values) + 1j * imaginary), np.asarray(
            self._real(values, 1) + 1j * slope
        )

    def neff(self, omega: ArrayLike) -> Complex[np.ndarray, " n_omega"]:
        """Return the interpolated complex effective index.

        Parameters
        ----------
        omega : array_like
            Angular frequencies in rad/s, shape ``(n_omega,)``, inside the table.

        Returns
        -------
        numpy.ndarray
            Complex index, shape ``(n_omega,)``.
        """
        return 1 + self.refractivity(omega)[0]

    def dneff_domega(self, omega: ArrayLike) -> Complex[np.ndarray, " n_omega"]:
        """Return the frequency derivative of the interpolated index.

        Parameters
        ----------
        omega : array_like
            Angular frequencies in rad/s, shape ``(n_omega,)``, inside the table.

        Returns
        -------
        numpy.ndarray
            Complex derivative in s/rad, shape ``(n_omega,)``.
        """
        return self.refractivity(omega)[1]

    def beta(self, omega: ArrayLike) -> Float[np.ndarray, " n_omega"]:
        """Return the propagation constant of the interpolated index.

        Parameters
        ----------
        omega : array_like
            Angular frequencies in rad/s, shape ``(n_omega,)``, inside the table.

        Returns
        -------
        numpy.ndarray
            Propagation constant in rad/m, shape ``(n_omega,)``.
        """
        return np.asarray(omega) * np.real(self.neff(omega)) / C

    def dbeta_domega(self, omega: ArrayLike) -> Float[np.ndarray, " n_omega"]:
        """Return the inverse group velocity of the interpolated index.

        Parameters
        ----------
        omega : array_like
            Angular frequencies in rad/s, shape ``(n_omega,)``, inside the table.

        Returns
        -------
        numpy.ndarray
            Inverse group velocity in s/m, shape ``(n_omega,)``.
        """
        refractivity, derivative = self.refractivity(omega)
        return (1 + np.real(refractivity) + np.asarray(omega) * np.real(derivative)) / C

    def alpha(self, omega: ArrayLike) -> Float[np.ndarray, " n_omega"]:
        """Return the power attenuation of the interpolated index.

        Parameters
        ----------
        omega : array_like
            Angular frequencies in rad/s, shape ``(n_omega,)``, inside the table.

        Returns
        -------
        numpy.ndarray
            Non-negative intensity attenuation in 1/m, shape ``(n_omega,)``.
        """
        return 2 * np.asarray(omega) * np.imag(self.refractivity(omega)[0]) / C

    def effective_area(self, omega: ArrayLike) -> Float[np.ndarray, " n_omega"]:
        """Return the interpolated effective-area column.

        Parameters
        ----------
        omega : array_like
            Angular frequencies in rad/s, shape ``(n_omega,)``, inside the table.

        Returns
        -------
        numpy.ndarray
            Effective areas in m^2, shape ``(n_omega,)``.

        Raises
        ------
        ModeError
            If no area exists, the query is outside the table, or interpolation
            produces a non-positive or non-finite area.
        """
        if self._area is None:
            raise ModeError("the table has no effective-area column")
        result = np.asarray(self._area(self._checked(omega)))
        if not np.all(np.isfinite(result) & (result > 0)):
            raise ModeError("the interpolated area must be positive and finite")
        return result

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        low, high = self.wavelength_range
        return "IndexTable: a tabulated effective index", [
            ("rows", np.asarray(self.omega).size),
            ("support", f"{low * 1e9:.4g}-{high * 1e9:.4g} nm"),
            ("loss interpolation", self.loss_interpolation),
        ]

    def __repr__(self) -> str:
        """Return the table's support, row count and interpolation policy.

        Returns
        -------
        str
            Compact table description.
        """
        low, high = self.wavelength_range
        return (
            f"IndexTable({np.asarray(self.omega).size} rows, "
            f"{low * 1e9:g}-{high * 1e9:g} nm, loss={self.loss_interpolation})"
        )


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class _TableMode(MarcatiliMode):
    """Share fixed Marcatili fields and a loaded table between the two fill rules."""

    table: IndexTable
    reference_wavelength: float | None = None

    def _check_arguments(self) -> None:
        """Refuse geometry or area declarations incompatible with a fixed table.

        Raises
        ------
        ModeError
            If the inherited inputs, geometry or reference wavelength are invalid.
        """
        super()._check_arguments()
        if self.model != "full":
            raise ModeError('a tabulated mode does not support model="reduced"')
        if not as_profile(self.radius).is_constant:
            raise ModeError("a table is one geometry; a tapered radius is refused")
        if self.cladding is not None:
            raise ModeError("the table carries the wall; omit cladding")
        if (self.table.area is None) != (self.reference_wavelength is None):
            raise ModeError(
                "the table's area is the propagation area at one wavelength; "
                "say which with reference_wavelength only when an area column exists"
            )
        if self.reference_wavelength is not None:
            low, high = self.table.wavelength_range
            if not low <= self.reference_wavelength <= high:
                raise ModeError(
                    "reference_wavelength must lie inside the table's wavelength range"
                )
            # aeff queries the table in frequency, so refuse here what that query
            # would refuse later: the wavelength round trip is not exact, and a
            # reference wavelength equal to a table end lands a few ulp outside it.
            self.table._checked([2 * np.pi * C / self.reference_wavelength])

    def sampled_terms(self, omega: Float[np.ndarray, " n_omega"]) -> MarcatiliTerms:
        """Sample the gas response without evaluating any cladding model.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive angular frequencies in rad/s, shape ``(n_omega,)``.

        Returns
        -------
        MarcatiliTerms
            Gas susceptibility and derivative, with zero cladding terms.

        Raises
        ------
        ModeError
            If a frequency is not positive and finite or lies outside the table.
        """
        # The parent's check first, so that the whole grid including its DC bin
        # still earns the capillary's zero-frequency diagnostic rather than the
        # table's generic array message.
        values = self.table._checked(_checked_omega(omega))
        susceptibility, derivative = self._sampled_susceptibility(values)
        return MarcatiliTerms(
            values,
            self.u_nm,
            susceptibility,
            derivative,
            np.zeros_like(values, dtype=complex),
            np.zeros_like(values, dtype=complex),
        )

    def aeff(self, z: float = 0.0) -> float:
        """Return the propagation area at the declared reference wavelength.

        Parameters
        ----------
        z : float, optional
            Position in m; the table fixes the geometry.

        Returns
        -------
        float
            Area in m^2, or Marcatili's closed form when there is no area column.
        """
        if self.reference_wavelength is None:
            return super().aeff(z)
        return float(
            self.table.effective_area([2 * np.pi * C / self.reference_wavelength])[0]
        )

    def effective_area(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Float[np.ndarray, " n_omega"]:
        """Return the table's frequency-dependent effective area.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s, shape ``(n_omega,)``, inside the table.
        z : float, optional
            Position in m; ignored because the geometry is fixed.

        Returns
        -------
        numpy.ndarray
            Areas in m^2, shape ``(n_omega,)``.

        Raises
        ------
        ModeError
            If there is no area column or the table cannot evaluate this query.
        """
        del z
        return self.table.effective_area(omega)

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
        kept = [row for row in rows if row[0] not in ("model", "cladding")]
        return title, [*kept, ("table", repr(self.table))]

    def __repr__(self) -> str:
        """Return the mode, geometry, fill and table support.

        Returns
        -------
        str
            Compact mode description.
        """
        radius = float(self.core_radius(np.array([0.0]))[0])
        fill = (
            repr(self.mixture)
            if self.mixture is not None
            else f"{self.gas}, pressure={self.pressure!r}"
        )
        return (
            f"{type(self).__name__}({self.kind}{self.n}{self.m}, "
            f"a={radius * 1e6:g} um, {fill}, {self.table!r})"
        )


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class TabulatedHollowMode(_TableMode):
    """Add a gas fill to an evacuated table using fixed transverse wavenumber.

    ``n_eff**2 = n_v**2 + chi_g`` is exact for the full Marcatili model. It
    neglects the fill's change of wall contrast in a general fibre (about -0.25 %
    in the resonance wavelengths at 10 bar of argon).
    Inherited gas, pressure and mixture paths supply both linear and nonlinear
    responses. Radius is constant; cladding and the reduced model are refused.

    Attributes
    ----------
    table : IndexTable
        Evacuated complex effective index and optional area for this geometry.
    reference_wavelength : float or None, optional
        Wavelength in m at which the table's area is used for propagation.
        Required exactly when the table contains an area column.
    """

    def _waveguide(
        self, terms: MarcatiliTerms, radius: float
    ) -> tuple[Inexact[np.ndarray, " n_omega"], Inexact[np.ndarray, " n_omega"]]:
        """Return the evacuated transverse-wavenumber term and its derivative.

        Parameters
        ----------
        terms : MarcatiliTerms
            Frequency samples in rad/s.
        radius : float
            Field radius in m; the table fixes the waveguide geometry.

        Returns
        -------
        tuple of numpy.ndarray
            Dimensionless ``1 - n_v**2`` and its derivative in s/rad.
        """
        # Geometry is fixed by the table; radius serves the field, normalisation
        # and the closed-form area only. Form 1 - n_v^2 without cancellation.
        del radius
        refractivity, derivative = self.table.refractivity(np.real(terms.omega))
        return -refractivity * (2 + refractivity), -derivative * (2 + 2 * refractivity)


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class TabulatedMode(_TableMode):
    """Use a filled table's whole linear response with Marcatili transverse fields.

    Gas or mixture declarations supply nonlinear responses only, and must have
    constant paths. A table computed at 10 bar paired with 20 bar nonlinear
    responses is a modelling error the code cannot detect. Loss policies and the
    real-index floor are inherited from MarcatiliMode.

    Attributes
    ----------
    table : IndexTable
        Whole complex effective index and optional area at one fill and geometry.
    reference_wavelength : float or None, optional
        Wavelength in m selecting the propagation area, required exactly when
        the table has an area column.
    """

    def __post_init__(self) -> None:
        """Validate the inherited fields and refuse a varying fill.

        Raises
        ------
        ModeError
            If any material path varies or the table-mode requirements fail.
        """
        super().__post_init__()
        if not self.z_constant:
            raise ModeError(
                "the table holds one fill; a gradient needs TabulatedHollowMode "
                "with an evacuated table"
            )

    def sampled_terms(self, omega: Float[np.ndarray, " n_omega"]) -> MarcatiliTerms:
        """Sample nothing: the filled table is the whole linear response.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive angular frequencies in rad/s, shape ``(n_omega,)``.

        Returns
        -------
        MarcatiliTerms
            Zero gas and cladding terms, one susceptibility row per population.

        Raises
        ------
        ModeError
            If a frequency is not positive and finite or lies outside the table.
        """
        # The declared fill supplies nonlinear responses only: _model_values
        # returns the table and absorption_by_species returns zeros, so sampling
        # the gas fit here could only warn about a range this mode never asks
        # for. The species axis is kept so that a mixture's shape is
        # still checked against the frequencies.
        values = self.table._checked(_checked_omega(omega))
        rows = () if self.mixture is None else (len(self.mixture.components),)
        susceptibility = np.zeros((*rows, values.size), dtype=complex)
        return MarcatiliTerms(
            values,
            self.u_nm,
            susceptibility,
            np.zeros_like(susceptibility),
            np.zeros_like(values, dtype=complex),
            np.zeros_like(values, dtype=complex),
        )

    def _model_values(
        self,
        terms: MarcatiliTerms,
        radius: float,
        density: float | Float[np.ndarray, " n_species"],
    ) -> tuple[Inexact[np.ndarray, " n_omega"], Inexact[np.ndarray, " n_omega"]]:
        """Return the filled table's refractivity and derivative.

        Parameters
        ----------
        terms : MarcatiliTerms
            Frequency samples in rad/s.
        radius : float
            Field radius in m; ignored by the stored linear response.
        density : float or numpy.ndarray
            Number densities in m^-3; ignored by the stored linear response.

        Returns
        -------
        tuple of numpy.ndarray
            Complex refractivity and its derivative in s/rad.
        """
        # Both geometry and fill are already included in the stored linear index.
        del radius, density
        return self.table.refractivity(np.real(terms.omega))

    def absorption_by_species(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Float[np.ndarray, "n_species n_omega"]:
        """Return zero species attribution for a table's indivisible total loss.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive finite angular frequencies in rad/s, shape ``(n_omega,)``.
        z : float, optional
            Position in m; ignored, the fill is fixed.

        Returns
        -------
        numpy.ndarray
            Zeros in 1/m, shape ``(n_species, n_omega)``; one row for a legacy gas.
        """
        del z
        values = self.table._checked(omega)
        count = 1 if self.mixture is None else len(self.mixture.components)
        return np.zeros((count, values.size))


@dataclass(frozen=True, eq=False, repr=False, init=False)
class _SampledIndexTable(IndexTable):
    """Retain sampled dispersion derivatives and snapshot provenance.

    ``repr=False``: a regenerated dataclass ``__repr__`` dumped 809 characters of
    arrays and undid :class:`IndexTable`'s one-line description.

    Parameters
    ----------
    omega : array_like
        Positive finite angular frequencies in rad/s, shape ``(n_nodes,)``.
    neff_values : array_like
        Finite complex effective indices matching `omega`.
    area : array_like or None, optional
        Positive finite effective areas in m^2 at the same samples.
    loss_interpolation : {'log', 'linear'}, optional
        Imaginary-index interpolation policy; ``linear`` by default here, because
        a lossless source has no logarithm.
    derivative_values : array_like
        Source ``dneff_domega`` in s/rad at the same frequencies, stored sorted
        and read-only, shape ``(n_sampled_derivatives,)``.
    source_identity : collections.abc.Mapping
        The sampled mode's type, repr, snapshot position, frequency domain and
        area model.
    """

    derivative_values: Complex[np.ndarray, " n_sampled_derivatives"]
    source_identity: Mapping[str, object]

    def __init__(
        self,
        omega: ArrayLike,
        neff_values: ArrayLike,
        area: ArrayLike | None = None,
        loss_interpolation: Literal["log", "linear"] = "linear",
        *,
        derivative_values: ArrayLike,
        source_identity: Mapping[str, object],
    ) -> None:
        """Store the sampled derivatives, then validate as any index table.

        Parameters
        ----------
        omega : array_like
            Positive frequencies in rad/s, shape ``(n_nodes,)``, in either order.
        neff_values : array_like
            Passive indices at those frequencies, same shape.
        area : array_like or None, optional
            Positive effective areas in m^2 at the same samples.
        loss_interpolation : {'log', 'linear'}, optional
            Imaginary-index interpolation policy; ``linear`` here, because a
            lossless source has no logarithm.
        derivative_values : array_like
            Source ``dneff_domega`` in s/rad at the same frequencies.
        source_identity : collections.abc.Mapping
            Provenance of the sampled mode.
        """
        # Set before delegating: IndexTable.__init__ calls __post_init__, which
        # fits the Hermite interpolants from these two fields. Declaring them as
        # ordinary init fields is what makes `dataclasses.replace` work on this
        # subclass -- it reconstructs through this signature.
        object.__setattr__(self, "derivative_values", derivative_values)
        object.__setattr__(self, "source_identity", dict(source_identity))
        super().__init__(omega, neff_values, area, loss_interpolation)

    def __post_init__(self) -> None:
        """Sort the derivatives with the samples and fit Hermite interpolants.

        Raises
        ------
        ModeError
            If the derivatives are non-finite or disagree with the frequencies,
            or if the inherited table requirements fail.
        """
        # The base sorts by frequency and does not report the permutation, so take
        # it from the same input here and apply it to the derivatives.
        order = np.argsort(_positive_vector(self.omega, "omega"))
        derivatives = np.array(self.derivative_values, dtype=complex, copy=True)
        if derivatives.shape != order.shape or not np.all(np.isfinite(derivatives)):
            raise ModeError("mode frequency derivative must be finite and match omega")
        derivatives = derivatives[order]
        derivatives.setflags(write=False)
        object.__setattr__(self, "derivative_values", derivatives)
        super().__post_init__()
        object.__setattr__(
            self,
            "_real",
            CubicHermiteSpline(
                self.omega,
                np.real(self.neff_values) - 1,
                np.real(derivatives),
                extrapolate=False,
            ),
        )
        object.__setattr__(
            self,
            "_imaginary",
            CubicHermiteSpline(
                self.omega,
                np.imag(self.neff_values),
                np.imag(derivatives),
                extrapolate=False,
            ),
        )


def sample_index_table(
    mode: Mode, omega: ArrayLike, *, z: float = 0.0, include_area: bool = True
) -> IndexTable:
    """Sample a mode into a dense index snapshot with analytic Hermite slopes.

    Parameters
    ----------
    mode : Mode
        Source mode with an index and its declared frequency derivative.
    omega : array_like
        At least four distinct positive finite frequencies in rad/s, in either order.
    z : float, optional
        Finite snapshot position in m within the source's declared paths.
    include_area : bool, optional
        Include area in m². Step-index profiles are evaluated at each frequency;
        other modes retain their declared frozen-profile area at z.

    Returns
    -------
    IndexTable
        Sorted, immutable index samples and a Hermite interpolant using the
        supplied derivative. No extrapolation or E/H reconstruction is implied.
        The object is a private :class:`IndexTable` subclass, so it also carries
        ``derivative_values`` (the source slopes in s/rad) and ``source_identity``
        (the mode's type, repr, snapshot position, frequency domain and area
        model); every other attribute and method is the public table's.

        The table is built with ``loss_interpolation="linear"``, not the table
        default ``"log"``: a lossless source has an imaginary index of exactly
        zero and no logarithm, and the logarithmic policy refuses a non-positive
        one outright.

    Raises
    ------
    ModeError
        If frequencies, derivatives, position or required source data are invalid.
    """
    frequencies = np.sort(_positive_vector(omega, "omega"))
    if not np.isfinite(z):
        raise ModeError("snapshot z must be finite")
    if frequencies.size < MIN_TABLE_POINTS or np.any(np.diff(frequencies) <= 0):
        raise ModeError("sampler needs at least four distinct frequencies")
    values = mode.neff(frequencies, z)
    derivatives = mode.dneff_domega(frequencies, z)
    area = None
    if include_area:
        area = np.array(
            [
                replace(mode, profile_wavelength=2 * np.pi * C / w).aeff(z)
                if isinstance(mode, StepIndexMode)
                else mode.aeff(z)
                for w in frequencies
            ]
        )
    # A lossless source has an imaginary index of exactly zero, which has no
    # logarithm and which the table's default policy refuses outright.
    return _SampledIndexTable(
        frequencies,
        values,
        area,
        "linear",
        derivative_values=derivatives,
        source_identity={
            "mode_type": f"{type(mode).__module__}.{type(mode).__qualname__}",
            "mode": repr(mode),
            "z": z,
            "omega_domain": (float(frequencies[0]), float(frequencies[-1])),
            "area_model": "frequency-resolved"
            if isinstance(mode, StepIndexMode)
            else "frozen-profile",
        },
    )
