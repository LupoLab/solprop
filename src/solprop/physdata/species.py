# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Describe a gas species: a built-in gas from optmat, or a caller's own data.

Optical susceptibilities are per particle in m^3. Partial pressures are in bar,
the same interface convention as :mod:`solprop.physdata.equation_of_state`.
"""

import dataclasses
import functools
import math
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Complex, Float
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline, PchipInterpolator

from solprop._display import Displayed, line
from solprop.constants import BAR, K_B, ROOM_TEMPERATURE, C
from solprop.exceptions import MaterialError
from solprop.physdata import _optmat
from solprop.physdata.equation_of_state import ideal_gas_density
from solprop.physdata.gases import Gas, as_gas
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.physdata.raman_types import MolecularRaman
from solprop.physdata.selection import MaterialRecord, MaterialSelection
from solprop.physdata.sellmeier import GasSusceptibility, OptmatSusceptibility

__all__ = [
    "EquationOfState",
    "IdealGas",
    "IonisationData",
    "OptmatGas",
    "Species",
    "TabulatedSusceptibility",
    "ionisation_potential",
    "species",
]


@runtime_checkable
class EquationOfState(Protocol):
    """Convert a species' partial pressure and temperature to number density."""

    def density(self, pressure_bar: float, temperature: float) -> float:
        """Return number density in m^-3.

        Parameters
        ----------
        pressure_bar : float
            Non-negative partial pressure in bar.
        temperature : float
            Positive temperature in K.

        Returns
        -------
        float
            Number density in m^-3.
        """
        ...

    def pressure(self, number_density: float, temperature: float) -> float:
        """Return partial pressure in bar.

        Parameters
        ----------
        number_density : float
            Non-negative number density in m^-3.
        temperature : float
            Positive temperature in K.

        Returns
        -------
        float
            Partial pressure in bar.
        """
        ...


@dataclass(frozen=True)
class IdealGas:
    """Use the ideal-gas equation of state for a partial population."""

    def density(
        self, pressure_bar: float, temperature: float = ROOM_TEMPERATURE
    ) -> float:
        """Return ideal-gas number density.

        Parameters
        ----------
        pressure_bar : float
            Non-negative partial pressure in bar.
        temperature : float, optional
            Positive temperature in K.

        Returns
        -------
        float
            Number density in m^-3.

        Raises
        ------
        MaterialError
            If pressure or temperature is non-finite or outside its domain.
        """
        if not math.isfinite(pressure_bar) or pressure_bar < 0:
            raise MaterialError("partial pressure must be finite and non-negative")
        if not math.isfinite(temperature) or temperature <= 0:
            raise MaterialError("temperature must be finite and positive")
        return ideal_gas_density(pressure_bar, temperature)

    def pressure(
        self, number_density: float, temperature: float = ROOM_TEMPERATURE
    ) -> float:
        """Return ideal-gas partial pressure.

        Parameters
        ----------
        number_density : float
            Non-negative number density in m^-3.
        temperature : float, optional
            Positive temperature in K.

        Returns
        -------
        float
            Partial pressure in bar.

        Raises
        ------
        MaterialError
            If density or temperature is outside its domain.
        """
        self.density(0.0, temperature)
        if not math.isfinite(number_density) or number_density < 0:
            raise MaterialError("number density must be finite and non-negative")
        return number_density * K_B * temperature / BAR


_AIR_COMPOSITION: Final = "dry air (N2, O2, Ar)"
"""The composition whose constituents give ``Gas.Air``'s transport properties."""


@dataclass(frozen=True)
class OptmatGas:
    """Take a gas's equation of state and transport properties from optmat.

    The density is optmat's real-gas law ``N = p / (Z k_B T)`` and the pressure its
    exact inverse, on the gas's phase domain: below saturation, or above the
    critical temperature. A condensed state is refused, naming the saturation
    pressure. Vacuum is exact: zero pressure is zero density and back, without
    calling optmat. ``Gas.Air`` takes CoolProp's pseudo-pure dry air (optmat's
    ``air_equation_of_state``) for its density, and the constituents of optmat's
    ``"dry air (N2, O2, Ar)"`` for its transport properties.

    The transport properties are optmat's dilute viscosity, ideal-gas heat capacity
    and molar mass, each over its own temperature range; :mod:`solprop.gasflow`
    takes them for a fill of species that carry this equation of state.

    Pressures are in bar at this interface; optmat receives pascals.

    Attributes
    ----------
    gas : Gas
        The species.

    Examples
    --------
    >>> from solprop.physdata.species import OptmatGas
    >>> argon = OptmatGas("Ar")
    >>> f"{argon.density(1.0, 293.15):.6e}"
    '2.472418e+25'
    >>> argon.density(0.0, 293.15)
    0.0
    >>> f"{argon.pressure(argon.density(3.0, 293.15), 293.15):.12f}"
    '3.000000000000'
    """

    gas: Gas | str

    def __post_init__(self) -> None:
        """Coerce the species name."""
        object.__setattr__(self, "gas", as_gas(self.gas))

    @property
    def key(self) -> Gas:
        """Return the species as a registry member.

        Returns
        -------
        Gas
            The gas.
        """
        return as_gas(self.gas)

    @functools.cached_property
    def _equation(self) -> _optmat.Untyped:
        """Return optmat's equation of state of the gas, resolved once.

        Returns
        -------
        optmat.gas.eos.EquationOfState
            The gas's, or the pseudo-pure dry air's for ``Gas.Air``.
        """
        from optmat.gas.eos import (  # noqa: PLC0415 - optmat is imported lazily
            air_equation_of_state,
            equation_of_state,
        )

        if self.key is Gas.Air:
            return air_equation_of_state()
        return equation_of_state(self.key.value)

    @functools.cached_property
    def _transport(self) -> tuple[tuple[_optmat.Untyped, float], ...]:
        """Return optmat's transport model of each constituent, with its fraction.

        Returns
        -------
        tuple
            ``(TransportModel, mole fraction)`` pairs; one, with fraction 1, for a
            pure gas.
        """
        from optmat.gas.transport import transport_model  # noqa: PLC0415 - lazily

        return tuple(
            (transport_model(name), fraction) for name, fraction in self.constituents
        )

    @property
    def constituents(self) -> tuple[tuple[str, float], ...]:
        """Return the gases and mole fractions the transport properties combine.

        Returns
        -------
        tuple
            ``(gas, mole fraction)`` pairs: the gas itself, or air's constituents.
        """
        if self.key is Gas.Air:
            return tuple(_optmat.composition_fractions(_AIR_COMPOSITION).items())
        return ((self.key.value, 1.0),)

    @property
    def material_records(self) -> tuple[MaterialRecord, ...]:
        """Return the records of the equation of state and transport models.

        Returns
        -------
        tuple of MaterialRecord
            The equation of state first, then each constituent's transport model.
        """
        return (
            _optmat.record(self._equation),
            *(_optmat.record(model) for model, _ in self._transport),
        )

    def _state_error(
        self, error: Exception, pressure_bar: float, temperature: float
    ) -> MaterialError:
        """Return the error for a state optmat's equation of state refuses.

        Parameters
        ----------
        error : Exception
            The ``StateError`` optmat raised.
        pressure_bar : float
            The pressure asked for, in bar, or NaN when a density was given.
        temperature : float
            The temperature asked for, in K.

        Returns
        -------
        MaterialError
            The refusal, naming the saturation pressure below the critical point.
        """
        equation = self._equation
        low, high = equation.temperature_range
        inside = low <= temperature <= high
        name = (
            f"{self.key.value}: optmat's equation of state ({equation.identity.name})"
        )
        note = ""
        if inside and temperature < equation.critical_temperature:
            saturation = _optmat.pa_to_bar(equation.saturation_pressure(temperature))
            note = (
                f"; its saturation pressure at {temperature:g} K is {saturation:.6g} "
                "bar, above which it is a liquid or two-phase"
            )
        state = (
            f"{temperature:g} K"
            if math.isnan(pressure_bar)
            else f"{pressure_bar:g} bar and {temperature:g} K"
        )
        return MaterialError(
            f"{name} has no gas-phase solution at {state}{note}: {error}"
        )

    def density(
        self, pressure_bar: float, temperature: float = ROOM_TEMPERATURE
    ) -> float:
        """Return the real-gas number density at a partial pressure.

        Parameters
        ----------
        pressure_bar : float
            Non-negative partial pressure in bar; zero is vacuum.
        temperature : float, optional
            Positive temperature in K.

        Returns
        -------
        float
            Number density in m^-3; exactly 0 at zero pressure.

        Raises
        ------
        MaterialError
            If the pressure or temperature is outside its domain, or the state is
            condensed or outside the equation's range.
        """
        from optmat.conditions import State  # noqa: PLC0415 - optmat is lazy
        from optmat.exceptions import StateError  # noqa: PLC0415 - as above

        IdealGas().density(pressure_bar, temperature)
        if pressure_bar == 0.0:
            return 0.0
        _optmat.check_status(self._equation, self.key.value)
        try:
            return float(
                self._equation.number_density(
                    State(_optmat.bar_to_pa(pressure_bar), float(temperature))
                )
            )
        except StateError as error:
            raise self._state_error(error, pressure_bar, temperature) from error

    def pressure(
        self, number_density: float, temperature: float = ROOM_TEMPERATURE
    ) -> float:
        """Return the partial pressure at a number density, the density's inverse.

        Parameters
        ----------
        number_density : float
            Non-negative number density in m^-3; zero is vacuum.
        temperature : float, optional
            Positive temperature in K.

        Returns
        -------
        float
            Partial pressure in bar; exactly 0 at zero density. It reproduces the
            density to optmat's round-trip tolerance, 1e-12 relative.

        Raises
        ------
        MaterialError
            If the density or temperature is outside its domain, or the gas reaches
            the density only condensed.
        """
        from optmat.exceptions import StateError  # noqa: PLC0415 - optmat is lazy

        IdealGas().pressure(number_density, temperature)
        if number_density == 0.0:
            return 0.0
        _optmat.check_status(self._equation, self.key.value)
        try:
            pascals = self._equation.pressure(float(number_density), float(temperature))
        except StateError as error:
            raise self._state_error(error, math.nan, temperature) from error
        return _optmat.pa_to_bar(float(pascals))

    @property
    def temperature_range(self) -> tuple[float, float]:
        """Return the temperatures over which every transport property is defined.

        Returns
        -------
        tuple of float
            The lowest and highest temperature in K, over every constituent's
            viscosity and heat capacity.
        """
        lows, highs = [], []
        for model, _ in self._transport:
            for quantity in ("viscosity", "heat_capacity"):
                low, high = model.temperature_range(quantity)
                lows.append(float(low))
                highs.append(float(high))
        return max(lows), min(highs)

    def _checked(self, temperature: float) -> float:
        """Refuse a temperature outside the transport range; apply the status policy.

        Parameters
        ----------
        temperature : float
            Temperature in K.

        Returns
        -------
        float
            The temperature as a float.

        Raises
        ------
        MaterialError
            If the temperature is not finite or lies outside the range.
        """
        low, high = self.temperature_range
        value = float(temperature)
        if not (math.isfinite(value) and low <= value <= high):
            msg = (
                f"the transport properties of {self.key.value} are declared for "
                f"{low:g} K to {high:g} K by optmat, and {temperature} K was asked for"
            )
            raise MaterialError(msg)
        for model, _ in self._transport:
            _optmat.check_status(model, self.key.value)
        return value

    @property
    def molar_mass(self) -> float:
        """Return the molar mass of the fluid whose equation of state gives the density.

        optmat's equation of state's own (``EquationOfState.molar_mass``), so that a
        mass density formed from :meth:`density` is the equation's: for a pure gas
        it equals the transport model's, and for ``Gas.Air`` it is CoolProp's
        pseudo-pure dry air's, 0.02896546 kg/mol, not the mean of the N2/O2/Ar
        constituents (0.0289597), which enters only Wilke's viscosity rule.

        Returns
        -------
        float
            Molar mass in kg mol^-1.
        """
        return float(self._equation.molar_mass)

    def viscosity(self, temperature: float) -> float:
        """Return the dilute viscosity; Wilke's rule over air's constituents.

        Parameters
        ----------
        temperature : float
            Temperature in K, within :attr:`temperature_range`.

        Returns
        -------
        float
            Dynamic viscosity in Pa s.

        Raises
        ------
        MaterialError
            If the temperature is outside the range.

        Warns
        -----
        MaterialStatusWarning
            If a transport model is provisional (optmat's CH4 viscosity).
        """
        from optmat.gas.transport import mixture_viscosity  # noqa: PLC0415 - lazily

        kelvin = self._checked(temperature)
        if len(self._transport) == 1:
            return float(self._transport[0][0].viscosity(kelvin))
        return float(mixture_viscosity(dict(self.constituents), kelvin))

    def molar_heat_capacity(self, temperature: float) -> float:
        """Return the ideal-gas isobaric molar heat capacity, mole-weighted for air.

        Parameters
        ----------
        temperature : float
            Temperature in K, within :attr:`temperature_range`.

        Returns
        -------
        float
            ``c_p`` in J mol^-1 K^-1.

        Raises
        ------
        MaterialError
            If the temperature is outside the range.
        """
        kelvin = self._checked(temperature)
        return math.fsum(
            fraction * float(model.heat_capacity(kelvin))
            for model, fraction in self._transport
        )

    def sound_speed(self, pressure_bar: float, temperature: float) -> float:
        """Return the real gas's speed of sound from its equation of state.

        Parameters
        ----------
        pressure_bar : float
            Pressure in bar, > 0.
        temperature : float
            Temperature in K, within :attr:`temperature_range`.

        Returns
        -------
        float
            Speed of sound in m s^-1.

        Raises
        ------
        MaterialError
            If the pressure is not positive, or the state is outside the domain.
        """
        from optmat.conditions import State  # noqa: PLC0415 - optmat is lazy
        from optmat.exceptions import StateError  # noqa: PLC0415 - as above

        kelvin = self._checked(temperature)
        if not (math.isfinite(pressure_bar) and pressure_bar > 0.0):
            raise MaterialError(f"pressure must be positive, got {pressure_bar} bar")
        try:
            return float(
                self._equation.speed_of_sound(
                    State(_optmat.bar_to_pa(pressure_bar), kelvin)
                )
            )
        except StateError as error:
            raise self._state_error(error, pressure_bar, kelvin) from error


@dataclass(frozen=True)
class IonisationData:
    """Carry the energy a plasma response pays per ionisation event of a species.

    A built-in gas carries optmat's curated first ionisation energy
    (:func:`ionisation_potential`); a caller's species carries its own. The rate
    itself is never derived from it: it is optmat's
    (:class:`~solprop.nonlinear.optmat_rates.OptmatRate`) or the caller's (SEP 3, D1).

    Attributes
    ----------
    potential : float
        Positive first ionisation potential in J.
    citation : str
        Provenance of the value.
    """

    potential: float
    citation: str = ""

    def __post_init__(self) -> None:
        """Reject an invalid energy."""
        if not math.isfinite(self.potential) or self.potential <= 0:
            raise MaterialError("ionisation potential must be finite and positive")


@dataclass(frozen=True, eq=False)
class TabulatedSusceptibility:
    """Interpolate measured dilute refractivity and absorption with explicit SI units.

    The real susceptibility per particle is twice the supplied refractivity per
    density. The imaginary part is ``c * absorption_cross_section / omega``. This
    conversion describes dilute optical data; the medium subsequently forms the
    exact square root of its summed susceptibility. Extrapolation is refused.

    Attributes
    ----------
    omega : numpy.ndarray
        Strictly increasing positive angular frequencies in rad/s.
    refractivity : numpy.ndarray
        Real ``(n - 1) / rho`` in m^3, at each supplied frequency.
    absorption_cross_section : numpy.ndarray
        Non-negative intensity absorption cross-section in m^2.
    wavelength_range : tuple of float
        Declared shortest and longest supported wavelength in m, within the table.
    citation : str
        Source and measurement conditions of the supplied optical data.
    """

    omega: Float[np.ndarray, " n_samples"]
    refractivity: Float[np.ndarray, " n_refractivity"]
    absorption_cross_section: Float[np.ndarray, " n_absorption"]
    wavelength_range: tuple[float, float]
    citation: str = ""
    _real: CubicSpline = field(init=False, repr=False)
    _absorption: PchipInterpolator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the support and build immutable spline data."""
        names = ("omega", "refractivity", "absorption_cross_section")
        for name in names:
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise MaterialError(f"{name} must be a finite one-dimensional table")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if (
            self.omega.size < 2
            or np.any(self.omega <= 0)
            or np.any(np.diff(self.omega) <= 0)
        ):
            raise MaterialError(
                "omega must contain at least two increasing positive frequencies"
            )
        if (
            self.refractivity.shape != self.omega.shape
            or self.absorption_cross_section.shape != self.omega.shape
        ):
            raise MaterialError("optical tables must have the same shape as omega")
        if np.any(self.absorption_cross_section < 0):
            raise MaterialError("absorption_cross_section must be non-negative")
        low, high = self.wavelength_range
        if not (0 < low < high < math.inf):
            raise MaterialError(
                "wavelength_range must be finite, positive and increasing"
            )
        # A few ulps admit the inverse round trip of supplied endpoint wavelengths.
        eps = 4 * np.finfo(float).eps
        if 2 * np.pi * C / high < self.omega[0] * (
            1 - eps
        ) or 2 * np.pi * C / low > self.omega[-1] * (1 + eps):
            raise MaterialError("wavelength_range extends beyond the optical table")
        object.__setattr__(
            self, "_real", CubicSpline(self.omega, self.refractivity, extrapolate=False)
        )
        # Shape preservation prevents spurious gain between absorption knots.
        object.__setattr__(
            self,
            "_absorption",
            PchipInterpolator(
                self.omega, self.absorption_cross_section, extrapolate=False
            ),
        )

    def _frequencies(self, omega: ArrayLike | float) -> Float[np.ndarray, "*shape"]:
        """Check requested frequencies against the declared optical support.

        Parameters
        ----------
        omega : array_like or float
            Angular frequencies in rad/s.

        Returns
        -------
        numpy.ndarray
            Finite frequencies in rad/s within the table support.
        """
        values = np.asarray(omega, dtype=float)
        low, high = self.wavelength_range
        lower, upper = 2 * np.pi * C / high, 2 * np.pi * C / low
        eps = 4 * np.finfo(float).eps
        if (
            not np.all(np.isfinite(values))
            or np.any(values < lower * (1 - eps))
            or np.any(values > upper * (1 + eps))
        ):
            raise MaterialError(
                "frequency outside tabulated wavelength support "
                f"{self.wavelength_range} m"
            )
        return np.clip(values, self.omega[0], self.omega[-1])

    @property
    def source(self) -> str:
        """Return the source name used in material diagnostics.

        Returns
        -------
        str
            The explicit tabulated-data source name.
        """
        return "tabulated"

    def susceptibility(self, omega: ArrayLike | float) -> Complex[np.ndarray, "*shape"]:
        """Return complex per-particle susceptibility within the declared support.

        Parameters
        ----------
        omega : array_like or float
            Positive angular frequencies in rad/s, within the declared support.

        Returns
        -------
        numpy.ndarray
            Complex susceptibility per particle in m^3, with the input shape.

        Raises
        ------
        MaterialError
            If a requested frequency is outside the declared support.
        """
        values = self._frequencies(omega)
        return np.asarray(
            2 * self._real(values) + 1j * C * self._absorption(values) / values
        )

    def dsusceptibility_domega(
        self, omega: ArrayLike | float
    ) -> Complex[np.ndarray, "*shape"]:
        """Differentiate the optical interpolant analytically.

        Parameters
        ----------
        omega : array_like or float
            Positive angular frequencies in rad/s, within the declared support.

        Returns
        -------
        numpy.ndarray
            Frequency derivative in m^3 s, with the input shape.

        Raises
        ------
        MaterialError
            If a requested frequency is outside the declared support.
        """
        values = self._frequencies(omega)
        return np.asarray(
            2 * self._real(values, 1)
            + 1j
            * C
            * (
                self._absorption(values, 1) / values
                - self._absorption(values) / values**2
            )
        )


@dataclass(frozen=True, eq=False)
class Species(Displayed):
    """Describe one chemical species and the response data it supplies.

    Attributes
    ----------
    name : str
        Non-empty species name, used as the default component label.
    susceptibility : GasSusceptibility or TabulatedSusceptibility
        Linear susceptibility per particle in m^3.
    equation_of_state : EquationOfState
        Partial-pressure equation of state; ideal gas unless supplied.
    gamma3 : float or None
        A caller's third-order hyperpolarisability in m^5/V^2, one number at every
        wavelength; absent data is explicit. ``None`` beside a `kerr` model, and for
        every built-in gas.
    ionisation : IonisationData or None
        The energy per ionisation event, when available: optmat's for a built-in
        gas, the caller's otherwise.
    raman : MolecularRaman or None
        A caller's molecular Raman constants; ``None`` for a built-in gas, whose
        response is optmat's (:mod:`solprop.nonlinear.optmat_raman`).
    citation : str
        Provenance of custom data.
    kerr : OptmatKerr or None
        optmat's dispersive Kerr model, evaluated at a run's carrier
        (:meth:`kerr_coefficient`) instead of a fixed `gamma3`; ``None`` for supplied
        data. A run's record carries it through the species'
        :attr:`material_records`, not through the configuration.
    """

    name: str
    susceptibility: GasSusceptibility | TabulatedSusceptibility
    equation_of_state: EquationOfState = field(default_factory=IdealGas)
    gamma3: float | None = None
    ionisation: IonisationData | None = None
    raman: MolecularRaman | None = None
    citation: str = ""
    # Recorded through the material manifest, so that a species without one keeps
    # its configuration record bit for bit.
    kerr: OptmatKerr | None = field(default=None, metadata={"record": False})

    def __repr__(self) -> str:
        """Return a one-line description: the name and which data the species has.

        Returns
        -------
        str
            For example ``Species(N2, kerr, ionisation, raman)``.
        """
        carried = [
            name
            for name, present in (
                ("kerr", self.gamma3 is not None or self.kerr is not None),
                ("ionisation", self.ionisation is not None),
                ("raman", self.raman is not None),
            )
            if present
        ]
        return line("Species", [("", self.name), *(("", name) for name in carried)])

    @property
    def material_records(self) -> tuple[MaterialRecord, ...]:
        """Return the records of the optmat models the species carries.

        Returns
        -------
        tuple of MaterialRecord
            The linear model's and the equation of state's, then the transport
            models', then the Kerr model's; empty for a species of supplied data.
        """
        records: list[MaterialRecord] = []
        for part in (self.susceptibility, self.equation_of_state, self.kerr):
            if isinstance(part, (OptmatSusceptibility, OptmatGas, OptmatKerr)):
                records.extend(part.material_records)
        return tuple(records)

    def kerr_coefficient(self, wavelength: float | None = None) -> float | None:
        """Return the per-particle Kerr coefficient at a carrier wavelength.

        Parameters
        ----------
        wavelength : float or None, optional
            The run's vacuum carrier wavelength in m. Required for an optmat `kerr`
            model, which is dispersive; ignored by a fixed `gamma3`.

        Returns
        -------
        float or None
            ``gamma3`` in m^5/V^2, or ``None`` for a species with no Kerr data.

        Raises
        ------
        MaterialError
            If the species has an optmat model and no carrier is given.

        Examples
        --------
        >>> from solprop.physdata import species
        >>> f"{species('Ar').kerr_coefficient(800e-9):.6e}"
        '1.476195e-51'
        """
        if self.kerr is None:
            return self.gamma3
        if wavelength is None:
            msg = (
                f"{self.name}: optmat's Kerr coefficient is dispersive and is "
                "evaluated at the run's carrier; give the carrier wavelength"
            )
            raise MaterialError(msg)
        return self.kerr.gamma3(wavelength)

    def __post_init__(self) -> None:
        """Reject unnamed species and non-finite or doubled nonlinear coefficients."""
        if not self.name.strip():
            raise MaterialError("species name must not be empty")
        if self.gamma3 is not None and not math.isfinite(self.gamma3):
            raise MaterialError(f"gamma3 for {self.name} must be finite")
        if self.gamma3 is not None and self.kerr is not None:
            msg = (
                f"{self.name} has a fixed gamma3 and an optmat Kerr model; a species "
                "carries one or the other"
            )
            raise MaterialError(msg)


def default_species(gas: Gas | str) -> Species:
    """Return the built-in gas a name means where no selection is given, once.

    ``species(gas)`` under ``MaterialSelection()`` (SEP 3, Phase 4a), cached so that
    every mode and medium that names the gas shares one set of optmat adapters, and
    with them one evaluation cache. A species is immutable, so sharing it is safe.

    Parameters
    ----------
    gas : Gas or str
        A built-in gas's name.

    Returns
    -------
    Species
        The species with optmat's linear law, equation of state and Kerr model.
    """
    return _default_species(as_gas(gas))


def fill_species(gas: Gas | str, susceptibility: GasSusceptibility | None) -> Species:
    """Return the single species a fill of a named gas carries.

    The built-in gas (:func:`default_species`), whose equation of state gives the
    fill's density; with an optmat law of the same gas given beside the name (a
    selection's, SEP 3, Q27), the species carries that law instead, so that the
    species a fill records is the one whose index acts. A caller's own law of another
    kind is the caller's and leaves the built-in species as it is.

    Parameters
    ----------
    gas : Gas or str
        A built-in gas's name.
    susceptibility : GasSusceptibility or None
        The law given beside it, if any.

    Returns
    -------
    Species
        The species.

    Examples
    --------
    >>> from solprop.physdata.species import fill_species
    >>> fill_species("Ar", None).susceptibility.source
    'fit/Ar'
    """
    material = default_species(gas)
    if (
        isinstance(susceptibility, OptmatSusceptibility)
        and susceptibility.gas.value == material.name
        and susceptibility != material.susceptibility
    ):
        return dataclasses.replace(material, susceptibility=susceptibility)
    return material


@functools.cache
def _default_species(gas: Gas) -> Species:
    """Build :func:`default_species` once per gas, keyed by the :class:`Gas` member.

    Parameters
    ----------
    gas : Gas
        The gas.

    Returns
    -------
    Species
        ``species(gas)``.
    """
    return species(gas)


def built_in_population(material: Species) -> bool:
    """Return whether a species is a built-in gas rather than a supplied one.

    A built-in gas is one :func:`species` resolved: a species named after a
    :class:`Gas` that carries optmat models. A supplied species is never matched to a
    built-in gas by its name.

    Parameters
    ----------
    material : Species
        The species.

    Returns
    -------
    bool
        Whether optmat can stand for it by name: as a Raman composition's
        constituent, or a rate's species.
    """
    gas = next((key for key in Gas if key.value == material.name), None)
    return gas is not None and bool(material.material_records)


@functools.cache
def _optmat_ionisation(gas: Gas) -> IonisationData:
    """Return optmat's curated first ionisation energy of a gas, once per gas.

    Parameters
    ----------
    gas : Gas
        A built-in gas other than air.

    Returns
    -------
    IonisationData
        The energy in J, cited to optmat's dataset key. optmat curates one for every
        built-in gas but air, which its callers refuse first.
    """
    from optmat.gas.ionisation import ionisation_parameters  # noqa: PLC0415 - lazy

    parameters = ionisation_parameters(gas.value)
    return IonisationData(
        float(parameters.ionization_energy),
        citation=(
            f"optmat ionisation parameters of {gas.value}: "
            f"{parameters.ionization_energy_kind} first ionisation energy, "
            f"{parameters.ionization_energy_source}"
        ),
    )


def ionisation_potential(gas: Gas | str) -> float:
    """Return a built-in gas's first ionisation energy, optmat's curated value.

    The energy a plasma response pays per event when its rate declares none of its
    own; an optmat rate pays the energy its own exponent uses instead (SEP 3, D8).

    Parameters
    ----------
    gas : Gas or str
        A built-in gas other than air.

    Returns
    -------
    float
        The first ionisation energy in J (adiabatic, as optmat curates it).

    Raises
    ------
    MaterialError
        If the gas is unknown or is air.

    Examples
    --------
    >>> from solprop.physdata import ionisation_potential
    >>> f"{ionisation_potential('Ar'):.6e}"
    '2.524968e-18'
    """
    key = as_gas(gas)
    if key is Gas.Air:
        msg = (
            "air has no single ionisation potential: its populations (N2, O2, Ar) "
            "each carry their own"
        )
        raise MaterialError(msg)
    return _optmat_ionisation(key).potential


def species(
    gas: Gas | str | Species,
    *,
    selection: MaterialSelection | None = None,
) -> Species:
    """Return a custom species unchanged, or build a built-in gas from optmat.

    Parameters
    ----------
    gas : Gas or str or Species
        A built-in gas's name, or an existing custom species.
    selection : MaterialSelection or None, optional
        ``None`` (the default) is ``MaterialSelection()``. The linear susceptibility
        is optmat's (``selection.linear``: the recommended model, a model name, or an
        optmat model object for this gas) with its equation of state and transport
        properties (:class:`~solprop.physdata.species.OptmatGas`), and the electronic
        Kerr model (:class:`~solprop.physdata.nonlinear_data.OptmatKerr`, evaluated at
        a run's carrier by
        :meth:`~solprop.physdata.species.Species.kerr_coefficient`; ``gamma3`` is then
        ``None``). The species carries optmat's first ionisation energy; its rate and
        Raman response are resolved by the drivers under the same selection.

    Returns
    -------
    Species
        The species, with its linear law, equation of state and data.

    Raises
    ------
    MaterialError
        If the name is unknown, a selection is given beside a custom species, or the
        selected linear model describes another gas.

    Examples
    --------
    >>> from solprop.physdata import species
    >>> argon = species("Ar")
    >>> argon.susceptibility.source, type(argon.equation_of_state).__name__
    ('fit/Ar', 'OptmatGas')
    >>> from solprop.physdata import MaterialSelection
    >>> chosen = MaterialSelection(linear="borzsonyi_2008/sellmeier_Ar")
    >>> species("Ar", selection=chosen).susceptibility.source
    'borzsonyi_2008/sellmeier_Ar'
    """
    if isinstance(gas, Species):
        if selection is not None:
            msg = f"a selection cannot override custom species {gas.name}"
            raise MaterialError(msg)
        return gas
    key = as_gas(gas)
    if selection is None:
        selection = MaterialSelection()
    optical = OptmatSusceptibility.resolve(key, selection.linear)
    # Air has no hyperpolarizability of its own in optmat: its populations carry theirs.
    kerr = None if key is Gas.Air else OptmatKerr(key)
    ionisation = None if key is Gas.Air else _optmat_ionisation(key)
    return Species(
        key.value,
        optical,
        OptmatGas(key),
        None,
        ionisation,
        None,
        optical.citation,
        kerr,
    )
