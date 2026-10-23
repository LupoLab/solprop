# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Steady gas flow in a hollow fibre: the fill, the choked end and the flow regime.

A hollow fibre filled from gas cells at its ends carries a steady laminar flow. For an
isothermal ideal gas in Poiseuille flow the mass flow is constant along each stage and
proportional to :math:`a^4\,p\,\mathrm{d}p/\mathrm{d}z`, which makes :math:`p^2` linear
in the hydraulic resistance :math:`R = \int a^{-4}\,\mathrm{d}z`
(:class:`solprop.profiles.PressureGradient`, Pennetta et al., APL Photonics 4, 056105
(2019), eq. 6). The PV throughput of a stage is

.. math:: Q = \frac{\pi\,(p_h^2 - p_l^2)}{16\,\eta\,R}.

**Choked ends.** An end that exhausts into a pumped chamber cannot pass more than the
sonic throughput :math:`p_e \pi a_e^2 c`. Equating the two, as HiSol.jl's
``choked_pressure`` does (which then drops :math:`p_e^2`), gives

.. math:: k (p_h^2 - p_e^2) = p_e, \quad k = \frac{1}{16\,\eta\,c\,a_e^2 R}, \quad
   p_e = \frac{2 k p_h^2}{1 + \sqrt{1 + 4 k^2 p_h^2}},

solved here in that non-cancelling form. :func:`steady_fill` applies
:math:`\max(\text{declared}, \text{choked}, \text{floor})` at each outer end that is the
low end of its stage: a fill "to vacuum" ends at its choked pressure, not at zero and
not at a numerical floor. ``c`` is the adiabatic sound speed at the fill temperature,
as HiSol takes it; an independent integration of the one-dimensional compressible
flow (``tests/test_gasflow.py``) brackets the result between an isothermal and an
adiabatic wall, which is why the adiabatic speed is kept (``docs/dev/decisions.md``).

**Flow regime.** Gebhardt et al., Opt. Express 29, 6957 (2021), measured output-beam
instabilities in differentially pumped capillaries from :math:`Re \approx 2040`, worst
near 2400, and recommend :math:`Re < 2000` (their eq. 4); the laminar fill itself is not
valid there. With :math:`Re = 4\dot m/(\pi d \eta)` and the mass flow constant along a
stage, the narrowest point of a taper has the largest Reynolds number. At the other
extreme the gas stops being a continuum: with :math:`Kn = \lambda / d`, first-order
wall slip raises the throughput by :math:`8\,Kn`, and above 0.1 the flow is
transitional. :func:`steady_fill` warns
(:class:`~solprop.exceptions.GasFlowWarning`) for a Reynolds number above 2000, for a
stage whose Knudsen number at its mean pressure and narrowest bore exceeds 0.01, or for
one with more than 1 % of its gas where the Knudsen number exceeds 0.1, and otherwise
logs the exit's Knudsen number. It does not replace the laminar law: a turbulent or
rarefied fill is out of scope.

Pressures are in **bar** at this interface (``docs/dev/conventions.md`` section 1);
throughputs are in Pa m^3 s^-1, with :func:`mbar_litres_per_second` and
:func:`standard_litres_per_minute` for the units vacuum practice quotes.
"""

import dataclasses
import functools
import itertools
import logging
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, get_args

import numpy as np
from jaxtyping import Float
from numpy.polynomial import Chebyshev
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.optimize import brentq

from solprop._display import Displayed, engineering, line
from solprop.constants import BAR, K_B, N_A, ROOM_TEMPERATURE, STD_ATMOSPHERE
from solprop.exceptions import GasFlowWarning, MaterialError, TranslationError
from solprop.physdata.mixtures import Mixture
from solprop.physdata.transport import (
    Composition,
    _composition,
    mean_free_path,
    molar_mass,
    sound_speed,
    viscosity,
)
from solprop.profiles import (
    Constant,
    PressureGradient,
    SeriesOfPath,
    ZProfile,
    _checked_edges,
    _checked_pressures,
    _checked_temperature,
    as_profile,
    hydraulic_integral,
)

__all__ = [
    "KNUDSEN_SLIP_LIMIT",
    "KNUDSEN_TRANSITION_LIMIT",
    "REYNOLDS_LAMINAR_LIMIT",
    "STANDARD_PRESSURE",
    "STANDARD_TEMPERATURE",
    "TRANSITIONAL_COLUMN_LIMIT",
    "FlowReport",
    "StageFlow",
    "VacuumEnd",
    "chamber_pressure",
    "choked_pressure",
    "flow_report",
    "knudsen_number",
    "mass_flow",
    "mbar_litres_per_second",
    "reynolds_number",
    "standard_litres_per_minute",
    "steady_fill",
    "temperature_weight",
    "throughput",
]

logger = logging.getLogger(__name__)

VacuumEnd = Literal["choked", "floor"]
"""How a fill treats an outer end below its choked pressure.

``"choked"`` raises it to the choked pressure of the flow, the default wherever the fill
knows the gas and the bore; ``"floor"`` keeps the declared value, raised only to the
numerical floor of :class:`solprop.profiles.PressureGradient`, which is what a
comparison with a fill whose end is a true vacuum needs. Either way a taper's fill
follows the bore: the choice concerns the ends only.
"""

REYNOLDS_LAMINAR_LIMIT: Final = 2000.0
"""Peak Reynolds number above which a fill warns that it is not laminar.

Gebhardt et al., Opt. Express 29, 6957 (2021), eq. (4) and Fig. 6: output-beam
instabilities appear from Re = 2040 and they recommend Re < 2000; pipe flow becomes
sustained-turbulent near Re = 2040 (Avila et al., Science 333, 192 (2011)).
"""

KNUDSEN_SLIP_LIMIT: Final = 0.01
"""Mean-pressure Knudsen number ``lambda(p_bar) / d_min`` above which a stage warns.

``p_bar`` is the mean of the stage's end pressures and ``d_min`` its narrowest
diameter. First-order wall slip raises a tube's throughput to
``Q_Poiseuille (1 + 8 Kn(p_bar))``, so above 0.01 the laminar throughput, the Reynolds
number and the choked pressure are off by about ten per cent or more.
"""

KNUDSEN_TRANSITION_LIMIT: Final = 0.1
"""Knudsen number ``lambda / d`` above which the gas is transitional, not a continuum.

The conventional transition-flow boundary for pipe flow.
"""

TRANSITIONAL_COLUMN_LIMIT: Final = 0.01
"""Share of a stage's column density ``int p dz`` that may be transitional unwarned.

A choked helium exit is transitional over micrometres (about 6 um of a 1 m helium
gradient fibre), which carries a negligible part of the gas and of the nonlinear phase;
a stage warns only when more than one per cent of its gas lies where the Knudsen number
exceeds :data:`KNUDSEN_TRANSITION_LIMIT`.
"""

STANDARD_TEMPERATURE: Final = 273.15
"""Temperature in K of a standard litre, for :func:`standard_litres_per_minute`."""

STANDARD_PRESSURE: Final = STD_ATMOSPHERE
"""Pressure in Pa of a standard litre, for :func:`standard_litres_per_minute`."""

_GAS_CONSTANT: Final = K_B * N_A

_SAMPLES_PER_STAGE: Final = 257
"""Evenly spaced positions per stage, beside its breakpoints, for the regime checks.

The Reynolds number follows ``1/a`` and the Knudsen number ``1/(p a)``: both are
extremal at a breakpoint or an end for a piecewise-linear bore, which are always
sampled; the interior samples only cover a smooth radius, whose extremum they locate
to a fraction of a percent of the stage.
"""

_CHOKE_MATCH: Final = 1e-9
"""Relative tolerance at which an end pressure counts as its choked value."""

_VISCOSITY_DEGREES: Final = (8, 16, 32, 64, 128, 256)
"""Chebyshev degrees tried in turn for the viscosity law of a temperature path.

Chosen on solprop's former transport data (before SEP 3), over 200 to 600 K: CoolProp's
N2, O2, Ar, H2 and CH4 viscosities met the tolerance below by degree 8, the tabulated
gases and SF6 by 32, and CoolProp's helium, whose model is not smooth at the 1e-6 level
(its series stalls near 2e-7 even at degree 256), by 64, as did a 95/5 He/Xe mixture.
optmat's laws take the same ladder; D2's over 30 to 600 K is not reproduced even at
degree 256, so such a path is refused (SEP 3, Q33).
"""

_VISCOSITY_TOLERANCE: Final = 1e-6
"""Largest relative error of the viscosity law against direct calls, on a check grid.

Three thousand times below the transport models' own uncertainty (0.3 % to 3 %), so the
law adds nothing to what the data already leave open; the fill depends on the weight
only through ratios of its integrals, which move by at most about twice this. A tighter
tolerance is not reachable for helium (the :data:`_VISCOSITY_DEGREES` note). The flow
weight built from the law is then integrated exactly.
"""

_VISCOSITY_CHECKS: Final = 4
"""Check points per degree at which a trial viscosity law is compared with direct calls:
``4 d + 1`` evenly spaced temperatures, never the interpolation nodes themselves."""

_PACKAGE_FILES: Final = (str(Path(__file__).resolve().parent),)
"""Frames a flow warning skips, so that it names the caller's line, not solprop's.

A fill is built several calls deep inside the package (a constructor, the mixture, the
builder), and no fixed ``stacklevel`` reaches the user's code from all of them.
"""


@dataclass(frozen=True)
class _FlowGas:
    """The gas properties a flow calculation needs, evaluated once.

    Attributes
    ----------
    composition : Mapping or Gas or str or Species
        The resolved gas or mole-fraction mapping.
    temperature : float
        Temperature in K.
    viscosity : float
        Dynamic viscosity in Pa s.
    sound_speed : float
        Dilute adiabatic sound speed in m s^-1.
    molar_mass : float
        Mean molar mass in kg mol^-1.
    """

    composition: Composition
    temperature: float
    viscosity: float
    sound_speed: float
    molar_mass: float

    @classmethod
    def of(cls, gas: Composition | Mixture, temperature: float) -> "_FlowGas":  # noqa: UP037 - evaluated while the class body still runs
        """Evaluate the properties of a gas or a shared mixture fill.

        Parameters
        ----------
        gas : Gas, str, Species, Mapping or Mixture
            One built-in gas, gases to mole fractions, or a mixture with a shared fill.
        temperature : float
            Temperature in K.

        Returns
        -------
        _FlowGas
            The properties.

        Raises
        ------
        MaterialError
            If the gas has no transport data at this temperature, or a mixture has
            independent populations, which are not one flow.
        """
        composition = _as_composition(gas)
        return cls(
            composition=composition,
            temperature=float(temperature),
            viscosity=viscosity(composition, temperature),
            sound_speed=sound_speed(composition, temperature),
            molar_mass=molar_mass(composition),
        )


def _as_composition(gas: Composition | Mixture) -> Composition:
    """Return the composition a flow calculation takes, from a gas or a mixture.

    Parameters
    ----------
    gas : Gas, str, Species, Mapping or Mixture
        The gas.

    Returns
    -------
    Gas, str, Species or Mapping
        A single gas, or species to mole fractions.

    Raises
    ------
    MaterialError
        If a mixture has independent populations.
    """
    if not isinstance(gas, Mixture):
        return gas
    if gas.shared is None or gas.fractions is None:
        raise MaterialError(
            "a mixture of independent partial-pressure populations is not one flow: "
            "the steady-flow law, the choke and the flow regime apply only to a "
            "shared fill (Mixture.fill)"
        )
    return {
        component.species: fraction
        for component, fraction in zip(gas.components, gas.fractions, strict=True)
    }


@dataclass(frozen=True)
class _TemperatureFlow:
    """The gas properties of a flow whose temperature varies along the fibre.

    The viscosity enters the flow law at every position, through the weight
    ``eta(T) T``, the Reynolds number and the mean free path, so it is represented
    once as a converged Chebyshev series in the temperature over the range the path
    takes, checked to :data:`_VISCOSITY_TOLERANCE`, which is cheap to evaluate at every
    sample.
    The properties of one end (the choke) are evaluated there directly.

    Attributes
    ----------
    composition : Mapping or Gas or str or Species
        The resolved gas or mole-fraction mapping.
    temperature : ZProfile
        The temperature path in K.
    viscosity_law : numpy.polynomial.Chebyshev
        ``eta(T)`` in Pa s, with its domain the temperature range in K.
    molar_mass : float
        Mean molar mass in kg mol^-1.
    """

    composition: Composition
    temperature: ZProfile
    viscosity_law: Chebyshev
    molar_mass: float

    @classmethod
    def of(
        cls,
        gas: Composition | Mixture,
        temperature: ZProfile,
        positions: Float[np.ndarray, " n_positions"],
    ) -> "_TemperatureFlow":  # noqa: UP037 - evaluated while the class body still runs
        """Fit the viscosity law over the temperatures a path takes on a fill.

        Parameters
        ----------
        gas : Gas, str, Species, Mapping or Mixture
            One built-in gas, gases to mole fractions, or a mixture with a shared fill.
        temperature : ZProfile
            The temperature path in K.
        positions : numpy.ndarray
            The fill's stage boundaries in m.

        Returns
        -------
        _TemperatureFlow
            The properties.

        Raises
        ------
        MaterialError
            If the gas has no transport data over the temperature range, or its
            viscosity does not converge to a series by the highest degree tried.
        """
        composition = _as_composition(gas)
        start, end = float(positions[0]), float(positions[-1])
        kinks = np.asarray(temperature.breakpoints, dtype=float)
        probes = np.unique(
            np.concatenate(
                [
                    np.linspace(start, end, _SAMPLES_PER_STAGE),
                    positions,
                    kinks[(kinks > start) & (kinks < end)],
                ]
            )
        )
        values = np.asarray(temperature(probes), dtype=float)
        low, high = float(np.min(values)), float(np.max(values))
        if high == low:
            high = low * (1.0 + 1e-9)
        return cls(
            composition=composition,
            temperature=temperature,
            viscosity_law=_viscosity_law(composition, low, high),
            molar_mass=molar_mass(composition),
        )

    @property
    def weight(self) -> SeriesOfPath:
        """Return the flow weight ``eta(T) T`` along the fibre, in Pa s K.

        Returns
        -------
        solprop.profiles.SeriesOfPath
            The weight as a series of the temperature path.
        """
        law = self.viscosity_law
        return SeriesOfPath(
            self.temperature, law * Chebyshev.identity(domain=law.domain)
        )

    def at(self, z: float) -> _FlowGas:
        """Return the properties at one position, evaluated directly.

        Parameters
        ----------
        z : float
            Position in m.

        Returns
        -------
        _FlowGas
            The properties at the temperature there.
        """
        local = float(self.temperature(np.array([z]))[0])
        return _FlowGas.of(self.composition, local)

    def viscosity(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the viscosity at each position, from the series.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            ``eta(T(z))`` in Pa s.
        """
        return np.asarray(self.viscosity_law(self.temperature(z)), dtype=float)

    def mean_free_path_at_one_bar(
        self, z: Float[np.ndarray, " n_z"]
    ) -> Float[np.ndarray, " n_z"]:
        """Return the mean free path at one bar and the local temperature.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            ``lambda`` in m, :func:`solprop.physdata.transport.mean_free_path` at 1 bar.
        """
        local = np.asarray(self.temperature(z), dtype=float)
        speed = np.sqrt(math.pi * _GAS_CONSTANT * local / (2.0 * self.molar_mass))
        return self.viscosity(z) / BAR * speed


def _viscosity_law(
    composition: Composition,
    low: float,
    high: float,
    degrees: tuple[int, ...] = _VISCOSITY_DEGREES,
) -> Chebyshev:
    """Return a Chebyshev series of a gas's viscosity in the temperature, checked.

    Each trial degree is accepted once the series reproduces direct calls to
    :data:`_VISCOSITY_TOLERANCE` on a grid of check temperatures between its nodes: a
    measured error, not a coefficient tail, since CoolProp's helium viscosity is not
    smooth enough for its tail to fall.

    Parameters
    ----------
    composition : Gas, str, Species or Mapping
        The gas.
    low, high : float
        The temperature range in K, ``low < high``.
    degrees : tuple of int, optional
        The degrees tried in turn.

    Returns
    -------
    numpy.polynomial.Chebyshev
        ``eta(T)`` in Pa s on ``[low, high]``.

    Raises
    ------
    MaterialError
        If the gas has no transport data there, or no degree tried meets the
        tolerance.
    """

    def values(temperatures: Float[np.ndarray, " n_t"]) -> Float[np.ndarray, " n_t"]:
        """Return the viscosity at each temperature.

        Parameters
        ----------
        temperatures : numpy.ndarray
            Temperatures in K.

        Returns
        -------
        numpy.ndarray
            ``eta`` in Pa s.
        """
        return np.array([viscosity(composition, float(t)) for t in temperatures])

    for degree in degrees:
        law = Chebyshev.interpolate(values, degree, domain=[low, high])
        checks = np.linspace(low, high, _VISCOSITY_CHECKS * degree + 1)
        error = np.max(np.abs(law(checks) / values(checks) - 1.0))
        if error <= _VISCOSITY_TOLERANCE:
            return law
    raise MaterialError(
        f"the viscosity of {composition!r} between {low:g} K and {high:g} K was not "
        f"reproduced to {_VISCOSITY_TOLERANCE:g} by a Chebyshev series of degree up to "
        f"{degrees[-1]}"
    )


def _positive(value: float, what: str) -> float:
    """Return a finite positive number, or refuse it by name.

    Parameters
    ----------
    value : float
        The number.
    what : str
        Its name and unit, for the error.

    Returns
    -------
    float
        The number as a float.

    Raises
    ------
    TranslationError
        If it is not finite and positive.
    """
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise TranslationError(f"{what} must be finite and positive, got {value}")
    return number


def _checked_choice(value: object, name: str, choices: tuple[str, ...]) -> None:
    """Refuse, by name, a keyword outside its literal choices.

    The annotations already say which strings are allowed; this is the check a caller
    without a type checker meets.

    Parameters
    ----------
    value : object
        What was passed.
    name : str
        The keyword, for the error.
    choices : tuple of str
        The allowed values.

    Raises
    ------
    TranslationError
        If `value` is not one of `choices`.
    """
    if value not in choices:
        raise TranslationError(f"{name} must be one of {choices}, got {value!r}")


def _choked_from(high_pa: float, factor: float) -> float:
    """Solve ``k (p_h^2 - p_e^2) = p_e`` for the exit pressure without cancellation.

    Parameters
    ----------
    high_pa : float
        The stage's other end pressure in Pa, >= 0.
    factor : float
        ``k = 1 / (16 eta c a_e^2 R)`` in Pa^-1, > 0.

    Returns
    -------
    float
        The choked exit pressure in Pa.
    """
    # The positive root 2 k p^2 / (1 + sqrt(1 + 4 k^2 p^2)) rather than
    # (-1 + sqrt(1 + 4 k^2 p^2)) / (2 k): the latter cancels to nothing when k p << 1,
    # which is the usual case (p_e << p_h), and its small-k p limit is HiSol's k p^2.
    product = factor * high_pa
    return 2.0 * product * high_pa / (1.0 + math.hypot(1.0, 2.0 * product))


def _choke(
    flow: _FlowGas, high_pa: float, exit_radius: float, resistance: float
) -> float:
    """Return the choked exit pressure in Pa of a stage.

    Parameters
    ----------
    flow : _FlowGas
        The gas properties.
    high_pa : float
        The other end of the stage in Pa.
    exit_radius : float
        The core radius at the exit in m.
    resistance : float
        The stage's ``int a^-4 dz`` in m^-3.

    Returns
    -------
    float
        The choked exit pressure in Pa.
    """
    factor = 1.0 / (
        16.0 * flow.viscosity * flow.sound_speed * exit_radius**2 * resistance
    )
    return _choked_from(high_pa, factor)


def _resistance(radius: float | ZProfile, length: float) -> tuple[ZProfile, float]:
    """Return a radius path and its hydraulic resistance over ``[0, length]``.

    Parameters
    ----------
    radius : float or ZProfile
        Core radius in m.
    length : float
        Length in m, > 0.

    Returns
    -------
    tuple
        The radius as a path, and ``int_0^length a^-4 dz`` in m^-3.
    """
    path = as_profile(radius)
    edges = np.array([0.0, _positive(length, "length in m")])
    return path, float(hydraulic_integral(path, edges)[0])


def _checked_pressure(value: float, what: str) -> float:
    """Return a finite non-negative pressure in bar, or refuse it by name.

    Parameters
    ----------
    value : float
        The pressure in bar.
    what : str
        Its name, for the error.

    Returns
    -------
    float
        The pressure as a float.

    Raises
    ------
    TranslationError
        If it is negative or not finite.
    """
    number = float(value)
    if not (math.isfinite(number) and number >= 0.0):
        raise TranslationError(f"{what} must be finite and not negative, got {value}")
    return number


def throughput(
    gas: Composition | Mixture,
    radius: float | ZProfile,
    length: float,
    start_bar: float,
    end_bar: float,
    *,
    temperature: float = ROOM_TEMPERATURE,
) -> float:
    """Return the laminar PV throughput of a bore between two pressures.

    ``Q = pi (p_h^2 - p_l^2) / (16 eta R)`` with ``R = int_0^length a^-4 dz``.

    Parameters
    ----------
    gas : Gas, str, Species, Mapping or Mixture
        One built-in gas, gases to mole fractions, or a mixture with a shared fill.
    radius : float or ZProfile
        Core radius in m, > 0, over ``[0, length]``.
    length : float
        Length in m, > 0.
    start_bar, end_bar : float
        Pressures in bar at ``z = 0`` and ``z = length``, >= 0.
    temperature : float, optional
        Gas temperature in K.

    Returns
    -------
    float
        The throughput in Pa m^3 s^-1, from the higher to the lower end, >= 0.

    Raises
    ------
    TranslationError
        If a pressure, the length or the radius is invalid.
    MaterialError
        If the gas has no transport data at this temperature.

    Examples
    --------
    Gebhardt et al.'s 5 m, 500 um nitrogen capillary at 4.2 bar into vacuum:

    >>> from solprop.gasflow import mbar_litres_per_second, throughput
    >>> q = throughput("N2", 250e-6, 5.0, 4.2, 0.0)
    >>> f"{mbar_litres_per_second(q):.2f}"
    '15.41'
    """
    flow = _FlowGas.of(gas, temperature)
    _, resistance = _resistance(radius, length)
    high = _checked_pressure(start_bar, "start_bar") * BAR
    low = _checked_pressure(end_bar, "end_bar") * BAR
    return _poiseuille(flow, high, low, resistance)


def _poiseuille(
    flow: _FlowGas, first_pa: float, second_pa: float, resistance: float
) -> float:
    """Return the laminar throughput between two pressures in Pa.

    Parameters
    ----------
    flow : _FlowGas
        The gas properties.
    first_pa, second_pa : float
        The two end pressures in Pa, in either order.
    resistance : float
        ``int a^-4 dz`` in m^-3.

    Returns
    -------
    float
        ``pi |p_1^2 - p_2^2| / (16 eta R)`` in Pa m^3 s^-1.
    """
    # (p1 - p2)(p1 + p2): the difference of the squares without their cancellation.
    difference = abs(first_pa - second_pa) * (first_pa + second_pa)
    return math.pi * difference / (16.0 * flow.viscosity * resistance)


def choked_pressure(
    gas: Composition | Mixture,
    radius: float | ZProfile,
    length: float,
    high_bar: float,
    *,
    exit_end: Literal["start", "end"] = "end",
    temperature: float = ROOM_TEMPERATURE,
) -> float:
    """Return the choked exit pressure of a bore fed at one end and pumped at the other.

    Parameters
    ----------
    gas : Gas, str, Species, Mapping or Mixture
        One built-in gas, gases to mole fractions, or a mixture with a shared fill.
    radius : float or ZProfile
        Core radius in m, > 0, over ``[0, length]``.
    length : float
        Length in m, > 0.
    high_bar : float
        Pressure in bar at the fed end, >= 0.
    exit_end : {'end', 'start'}, optional
        Which end exhausts: ``z = length`` (the default) or ``z = 0``. Its radius sets
        the sonic throughput.
    temperature : float, optional
        Gas temperature in K.

    Returns
    -------
    float
        The exit pressure in bar below which the exit throughput would exceed the
        sonic limit; the module docstring gives the relation.

    Raises
    ------
    TranslationError
        If a pressure, the length, the radius or `exit_end` is invalid.
    MaterialError
        If the gas has no transport data at this temperature.

    Examples
    --------
    >>> from solprop.gasflow import choked_pressure
    >>> f"{choked_pressure('N2', 250e-6, 5.0, 4.2) * 1e3:.1f}"
    '224.3'
    >>> f"{choked_pressure('Ar', 15e-6, 1.0, 20.0) * 1e3:.1f}"
    '79.1'
    """
    _checked_choice(exit_end, "exit_end", ("start", "end"))
    flow = _FlowGas.of(gas, temperature)
    path, resistance = _resistance(radius, length)
    position = 0.0 if exit_end == "start" else float(length)
    exit_radius = float(path(np.array([position]))[0])
    high = _checked_pressure(high_bar, "high_bar") * BAR
    return _choke(flow, high, exit_radius, resistance) / BAR


def mass_flow(
    gas: Composition | Mixture,
    throughput: float,
    pressure_bar: float,
    *,
    temperature: float = ROOM_TEMPERATURE,
) -> float:
    """Return the mass flow carried by a PV throughput measured at a given pressure.

    ``mdot = Q rho(p, T) / p`` with the real-gas mass density at the reference state
    ``(pressure_bar, temperature)``; a mixture sums its species' partial densities. A
    throughput is conventionally quoted at the low-pressure end, where the gas is
    close to ideal and this is ``Q M / (R T)``, which is also the limit taken at zero
    pressure.

    Parameters
    ----------
    gas : Gas, str, Species, Mapping or Mixture
        One built-in gas, gases to mole fractions, or a mixture with a shared fill.
    throughput : float
        PV throughput in Pa m^3 s^-1, >= 0.
    pressure_bar : float
        Pressure in bar at which the throughput is measured, >= 0.
    temperature : float, optional
        Gas temperature in K.

    Returns
    -------
    float
        Mass flow in kg s^-1.

    Raises
    ------
    TranslationError
        If the throughput or the pressure is invalid.
    MaterialError
        If the gas has no transport data at this temperature.

    Examples
    --------
    >>> from solprop.gasflow import mass_flow
    >>> f"{mass_flow('N2', 1.0, 0.0):.4e}"
    '1.1493e-05'
    """
    flow = _FlowGas.of(gas, temperature)
    rate = _checked_pressure(throughput, "throughput")
    return rate * _mass_per_pv(flow, _checked_pressure(pressure_bar, "pressure_bar"))


def _mass_per_pv(flow: _FlowGas, pressure_bar: float) -> float:
    """Return ``rho / p`` in kg per Pa m^3 at a pressure, real-gas where it matters.

    Parameters
    ----------
    flow : _FlowGas
        The gas properties.
    pressure_bar : float
        Pressure in bar, >= 0.

    Returns
    -------
    float
        Mass density over pressure, ``M / (Z R T)``.
    """
    if pressure_bar == 0.0:
        return flow.molar_mass / (_GAS_CONSTANT * flow.temperature)
    total = math.fsum(
        member.density(fraction * pressure_bar, flow.temperature)
        * member.molar_mass
        / N_A
        for member, fraction in _composition(flow.composition, "density")
    )
    return total / (pressure_bar * BAR)


def reynolds_number(
    gas: Composition | Mixture,
    mass_flow: float,
    radius: float | ZProfile,
    z: Float[np.ndarray, " n_z"],
    *,
    temperature: float = ROOM_TEMPERATURE,
) -> Float[np.ndarray, " n_z"]:
    """Return the Reynolds number of a steady flow at each position along a bore.

    ``Re = 4 mdot / (pi d eta) = 2 mdot / (pi a eta)``: Gebhardt et al.'s
    ``4 M Q / (R T pi d eta)`` written with the mass flow, which is constant along a
    stage, so the narrowest point has the largest value.

    Parameters
    ----------
    gas : Gas, str, Species, Mapping or Mixture
        One built-in gas, gases to mole fractions, or a mixture with a shared fill.
    mass_flow : float
        Mass flow in kg s^-1, >= 0.
    radius : float or ZProfile
        Core radius in m, > 0.
    z : numpy.ndarray
        Positions in m, shape ``(n_z,)``.
    temperature : float, optional
        Gas temperature in K.

    Returns
    -------
    numpy.ndarray
        The Reynolds number at each position, shape ``(n_z,)``.

    Raises
    ------
    TranslationError
        If the mass flow is negative or the radius is not positive.
    MaterialError
        If the gas has no transport data at this temperature.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.gasflow import reynolds_number
    >>> reynolds_number("N2", 1e-5, 250e-6, np.array([0.0])).round(1)
    array([1450.2])
    """
    flow = _FlowGas.of(gas, temperature)
    rate = _checked_pressure(mass_flow, "mass_flow")
    return _reynolds(flow, rate, as_profile(radius), np.asarray(z, dtype=float))


def _reynolds(
    flow: _FlowGas, rate: float, radius: ZProfile, z: Float[np.ndarray, " n_z"]
) -> Float[np.ndarray, " n_z"]:
    """Return ``2 mdot / (pi a eta)`` at each position.

    Parameters
    ----------
    flow : _FlowGas
        The gas properties.
    rate : float
        Mass flow in kg s^-1.
    radius : ZProfile
        Core radius in m.
    z : numpy.ndarray
        Positions in m.

    Returns
    -------
    numpy.ndarray
        The Reynolds number at each position.

    Raises
    ------
    TranslationError
        If the radius is not positive at a position.
    """
    values = np.asarray(radius(z), dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise TranslationError("the core radius must be finite and positive")
    return 2.0 * rate / (math.pi * values * flow.viscosity)


def knudsen_number(
    gas: Composition | Mixture,
    pressure_bar: ArrayLike | float,
    radius: ArrayLike | float,
    *,
    temperature: float = ROOM_TEMPERATURE,
) -> Float[np.ndarray, "*shape"]:
    """Return the Knudsen number ``lambda / d`` of a gas in a bore.

    Parameters
    ----------
    gas : Gas, str, Species, Mapping or Mixture
        One built-in gas, gases to mole fractions, or a mixture with a shared fill.
    pressure_bar : array_like or float
        Pressure in bar, >= 0; zero gives an infinite Knudsen number.
    radius : array_like or float
        Core radius in m, > 0, broadcast against `pressure_bar`.
    temperature : float, optional
        Gas temperature in K.

    Returns
    -------
    numpy.ndarray
        ``lambda(p, T) / (2 a)`` with the mean free path of
        :func:`solprop.physdata.transport.mean_free_path`, in the broadcast shape.

    Raises
    ------
    TranslationError
        If a pressure is negative or a radius is not positive.
    MaterialError
        If the gas has no transport data at this temperature.

    Examples
    --------
    >>> from solprop.gasflow import knudsen_number
    >>> f"{float(knudsen_number('N2', 0.01, 250e-6)):.4f}"
    '0.0130'
    """
    pressures = np.asarray(pressure_bar, dtype=float)
    radii = np.asarray(radius, dtype=float)
    if not np.all(np.isfinite(pressures)) or np.any(pressures < 0.0):
        raise TranslationError("pressures must be finite and not negative")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise TranslationError("the core radius must be finite and positive")
    at_one_bar = mean_free_path(_as_composition(gas), 1.0, temperature)
    with np.errstate(divide="ignore"):
        # The mean free path is inversely proportional to the pressure at fixed T.
        return np.asarray(at_one_bar / (pressures * 2.0 * radii))


def chamber_pressure(
    throughput: float, pump_speed: float | Callable[[float], float]
) -> float:
    """Return the pressure at which a pump removes a given throughput.

    Solves ``p S(p) = Q``, as HiSol.jl's ``gradient_end_pressure`` does: the pressure
    of the chamber an end exhausts into. If it is below the fill's choked pressure the
    fibre end is choked and stays at that value.

    Parameters
    ----------
    throughput : float
        PV throughput in Pa m^3 s^-1, >= 0.
    pump_speed : float or callable
        Pumping speed in m^3 s^-1, > 0, or a function of the chamber pressure in bar
        returning it (a pump's characteristic curve).

    Returns
    -------
    float
        The chamber pressure in bar.

    Raises
    ------
    TranslationError
        If the throughput or a speed is invalid, or a characteristic's throughput
        ``p S(p)`` does not reach `throughput` between 1e-12 bar and 10 bar.

    Examples
    --------
    A 35 m^3/h pump taking 15.3 mbar l/s:

    >>> from solprop.gasflow import chamber_pressure
    >>> f"{chamber_pressure(1.534, 35.0 / 3600.0) * 1e3:.3f}"
    '1.578'
    """
    rate = _checked_pressure(throughput, "throughput")
    if not callable(pump_speed):
        return rate / _positive(pump_speed, "pump speed in m^3/s") / BAR
    if rate == 0.0:
        return 0.0
    excess = functools.partial(_pump_excess, pump_speed=pump_speed, rate=rate)
    low, high = 1e-12, 10.0
    if excess(low) > 0.0 or excess(high) < 0.0:
        raise TranslationError(
            "the pump's throughput p S(p) does not reach the gas throughput "
            f"{rate:.4g} Pa m^3/s between {low:g} bar and {high:g} bar"
        )
    return float(brentq(excess, low, high, xtol=1e-300, rtol=1e-12))


def _pump_excess(
    pressure_bar: float, *, pump_speed: Callable[[float], float], rate: float
) -> float:
    """Return a pump's throughput less the gas's at a chamber pressure.

    Parameters
    ----------
    pressure_bar : float
        Chamber pressure in bar.
    pump_speed : callable
        Pumping speed in m^3 s^-1 as a function of the pressure in bar.
    rate : float
        The gas throughput in Pa m^3 s^-1.

    Returns
    -------
    float
        ``p S(p) - Q`` in Pa m^3 s^-1.
    """
    speed = _positive(pump_speed(pressure_bar), "pump speed in m^3/s")
    return pressure_bar * BAR * speed - rate


def mbar_litres_per_second(throughput: float) -> float:
    """Convert a PV throughput to mbar l/s.

    Parameters
    ----------
    throughput : float
        PV throughput in Pa m^3 s^-1.

    Returns
    -------
    float
        The throughput in mbar l s^-1 (1 Pa m^3/s = 10 mbar l/s).

    Examples
    --------
    >>> from solprop.gasflow import mbar_litres_per_second
    >>> mbar_litres_per_second(1.0)
    10.0
    """
    return float(throughput) / 100.0 * 1000.0


def standard_litres_per_minute(
    throughput: float, temperature: float = ROOM_TEMPERATURE
) -> float:
    """Convert a PV throughput at a gas temperature to standard litres per minute.

    A standard litre is the ideal-gas amount in one litre at
    :data:`STANDARD_TEMPERATURE` and :data:`STANDARD_PRESSURE` (0 degrees C, one
    atmosphere). The throughput ``Q = p dV/dt`` at temperature ``T`` carries
    ``Q/(k_B T)`` molecules per second, so the flow is ``Q (T_std/T) / p_std`` standard
    cubic metres per second, returned in litres per minute. HiSol.jl's ``slpm`` omits
    the ratio ``T_std/T`` (7 % at 293.15 K). A mass-flow controller calibrated to
    another standard temperature ``T_ref`` (20 degrees C, for example) reads
    ``T_ref/T_std`` times this number.

    Parameters
    ----------
    throughput : float
        PV throughput in Pa m^3 s^-1 at `temperature`.
    temperature : float, optional
        Temperature in K at which the throughput is measured, > 0.

    Returns
    -------
    float
        The flow in standard litres per minute.

    Examples
    --------
    >>> from solprop.gasflow import standard_litres_per_minute
    >>> f"{standard_litres_per_minute(1.0, 273.15):.4f}"
    '0.5922'
    """
    scale = STANDARD_TEMPERATURE / _positive(temperature, "temperature in K")
    return float(throughput) * scale / STANDARD_PRESSURE * 1000.0 * 60.0


@dataclass(frozen=True)
class StageFlow(Displayed):
    """The steady flow through one stage of a fill.

    Attributes
    ----------
    start, end : float
        The stage's ends in m.
    start_bar, end_bar : float
        The fill's pressures in bar at the two ends.
    temperature : float
        Gas temperature in K.
    throughput : float
        Laminar PV throughput in Pa m^3 s^-1, from the higher to the lower end.
    mass_flow : float
        Mass flow in kg s^-1, from the throughput at the stage's low end
        (:func:`mass_flow`).
    peak_reynolds : float
        The largest Reynolds number on the stage.
    peak_reynolds_position : float
        Where it is, in m: the narrowest point of the bore.
    knudsen_start, knudsen_end : float
        Knudsen numbers ``lambda / d`` at the two ends; infinite at zero pressure.
    peak_knudsen : float
        The largest Knudsen number on the stage.
    peak_knudsen_position : float
        Where it is, in m.
    mean_knudsen : float
        The Knudsen number at the stage's mean end pressure and narrowest diameter,
        which sets the first-order slip correction ``8 Kn`` of its throughput.
    transitional_length : float
        The length in m over which the Knudsen number exceeds
        :data:`KNUDSEN_TRANSITION_LIMIT`.
    transitional_fraction : float
        The share of the stage's column density ``int p dz`` that lies there.
    """

    start: float
    end: float
    start_bar: float
    end_bar: float
    temperature: float
    throughput: float
    mass_flow: float
    peak_reynolds: float
    peak_reynolds_position: float
    knudsen_start: float
    knudsen_end: float
    peak_knudsen: float
    peak_knudsen_position: float
    mean_knudsen: float
    transitional_length: float
    transitional_fraction: float

    @property
    def exit_knudsen(self) -> float:
        """Return the Knudsen number at the stage's lower-pressure end.

        Returns
        -------
        float
            ``knudsen_end`` when the flow runs towards +z, else ``knudsen_start``.
        """
        return (
            self.knudsen_end if self.end_bar <= self.start_bar else self.knudsen_start
        )

    @property
    def throughput_mbar_litres_per_second(self) -> float:
        """Return the throughput in mbar l/s.

        Returns
        -------
        float
            :func:`mbar_litres_per_second` of :attr:`throughput`.
        """
        return mbar_litres_per_second(self.throughput)

    @property
    def throughput_slpm(self) -> float:
        """Return the throughput in standard litres per minute.

        Returns
        -------
        float
            :func:`standard_litres_per_minute` of :attr:`throughput`.
        """
        return standard_litres_per_minute(self.throughput, self.temperature)

    def __repr__(self) -> str:
        """Return a one-line description: the span, the throughput and the peak Re.

        Returns
        -------
        str
            For example ``StageFlow(0 m to 5 m, 15.36 mbar l/s, Re 2539)``.
        """
        span = f"{engineering(self.start, 'm')} to {engineering(self.end, 'm')}"
        return line(
            "StageFlow",
            [
                ("", span),
                ("", f"{self.throughput_mbar_litres_per_second:.4g} mbar l/s"),
                ("", f"Re {self.peak_reynolds:.4g}"),
            ],
        )


@dataclass(frozen=True)
class FlowReport(Displayed):
    """The steady flow of a fill: per stage, and at the outer ends.

    Attributes
    ----------
    stages : tuple of StageFlow
        One entry per stage, in order along z.
    viscosity : float
        The gas viscosity in Pa s.
    sound_speed : float
        The dilute adiabatic sound speed in m s^-1.
    choked_start_bar, choked_end_bar : float or None
        The choked pressure in bar of the first and last outer end, where that end is
        the low end of its stage; ``None`` where it is not.
    start_is_choked, end_is_choked : bool
        Whether that end's pressure is at (or, with ``vacuum_end="floor"``, below)
        its choked value, so that the exit throughput is the sonic limit.
    """

    stages: tuple[StageFlow, ...]
    viscosity: float
    sound_speed: float
    choked_start_bar: float | None
    choked_end_bar: float | None
    start_is_choked: bool
    end_is_choked: bool

    @property
    def peak_reynolds(self) -> float:
        """Return the largest Reynolds number of the fill.

        Returns
        -------
        float
            The maximum over the stages.
        """
        return max(stage.peak_reynolds for stage in self.stages)

    @property
    def peak_reynolds_position(self) -> float:
        """Return where the largest Reynolds number is.

        Returns
        -------
        float
            The position in m.
        """
        return max(self.stages, key=lambda s: s.peak_reynolds).peak_reynolds_position

    @property
    def peak_knudsen(self) -> float:
        """Return the largest Knudsen number of the fill.

        Returns
        -------
        float
            The maximum over the stages.
        """
        return max(stage.peak_knudsen for stage in self.stages)

    def __repr__(self) -> str:
        """Return a one-line description: stages, peak Reynolds number and chokes.

        Returns
        -------
        str
            For example ``FlowReport(1 stage, peak Re 2539, end choked)``.
        """
        choked = [
            name
            for name, flag in (
                ("start", self.start_is_choked),
                ("end", self.end_is_choked),
            )
            if flag
        ]
        count = len(self.stages)
        return line(
            "FlowReport",
            [
                ("", f"{count} stage{'s' if count != 1 else ''}"),
                ("", f"peak Re {self.peak_reynolds:.4g}"),
                ("", f"{' and '.join(choked)} choked" if choked else "no choked end"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        rows: list[tuple[str, object]] = [
            ("viscosity", f"{self.viscosity * 1e6:.4g} uPa s"),
            ("sound speed", f"{self.sound_speed:.4g} m/s"),
        ]
        for stage in self.stages:
            span = f"{engineering(stage.start, 'm')} to {engineering(stage.end, 'm')}"
            rows.append(
                (
                    span,
                    f"{stage.throughput_mbar_litres_per_second:.4g} mbar l/s, "
                    f"{stage.throughput_slpm:.4g} slpm, peak Re "
                    f"{stage.peak_reynolds:.4g} at "
                    f"{engineering(stage.peak_reynolds_position, 'm')}, peak Kn "
                    f"{stage.peak_knudsen:.3g}",
                )
            )
        for name, value, flag in (
            ("start", self.choked_start_bar, self.start_is_choked),
            ("end", self.choked_end_bar, self.end_is_choked),
        ):
            if value is not None:
                state = "choked" if flag else "not choked"
                rows.append((name, f"{state} (choked pressure {value:.4g} bar)"))
        return "FlowReport: steady laminar flow", rows


def _stage_samples(
    fill: PressureGradient, stage: int
) -> Float[np.ndarray, " n_samples"]:
    """Return the positions at which a stage's flow regime is sampled.

    Parameters
    ----------
    fill : PressureGradient
        The fill.
    stage : int
        The stage index.

    Returns
    -------
    numpy.ndarray
        The stage's breakpoints and evenly spaced interior positions, in m.
    """
    start, end = float(fill.positions[stage]), float(fill.positions[stage + 1])
    kinks = np.asarray(fill.breakpoints, dtype=float)
    return np.unique(
        np.concatenate(
            [
                np.linspace(start, end, _SAMPLES_PER_STAGE),
                kinks[(kinks >= start) & (kinks <= end)],
            ]
        )
    )


def _stage_flow(
    flow: _FlowGas,
    fill: PressureGradient,
    radius: ZProfile,
    stage: int,
    resistance: float,
    along: _TemperatureFlow | None = None,
) -> StageFlow:
    """Return the flow through one stage.

    Parameters
    ----------
    flow : _FlowGas
        The gas properties; for a temperature path, those at the fill's start.
    fill : PressureGradient
        The fill.
    radius : ZProfile
        Core radius in m.
    stage : int
        The stage index.
    resistance : float
        The stage's ``int a^-4 dz`` in m^-3; for a temperature path its
        ``int eta T a^-4 dz`` in Pa s K m^-3.
    along : _TemperatureFlow or None, optional
        The properties along a temperature path; ``None`` for a constant temperature.

    Returns
    -------
    StageFlow
        The stage's flow.
    """
    first, second = (float(p) for p in fill.pressures[stage : stage + 2])
    samples = _stage_samples(fill, stage)
    radii = np.asarray(radius(samples), dtype=float)
    if along is None:
        rate = _poiseuille(flow, first * BAR, second * BAR, resistance)
        mass = rate * _mass_per_pv(flow, min(first, second))
        reynolds = _reynolds(flow, mass, radius, samples)
        at_one_bar = mean_free_path(flow.composition, 1.0, flow.temperature)
        temperature = flow.temperature
        free_path = np.full(samples.shape, at_one_bar)
        narrowest = float(np.min(radii))
        with np.errstate(divide="ignore"):
            mean_knudsen = at_one_bar / (0.5 * (first + second) * 2.0 * narrowest)

        def free_path_of(z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
            """Return the mean free path at one bar, the same everywhere.

            Parameters
            ----------
            z : numpy.ndarray
                Positions in m.

            Returns
            -------
            numpy.ndarray
                ``lambda(1 bar)`` in m at each position.
            """
            return np.full(np.shape(z), at_one_bar)

    else:
        # The throughput and the mass flow are quoted at the stage's low end, at the
        # temperature there: Q = pi T_e (p_h^2 - p_l^2) / (16 int eta T a^-4 dz),
        # which is _poiseuille with the resistance int (eta T / eta_e T_e) a^-4 dz.
        low = samples[-1] if second <= first else samples[0]
        local = along.at(float(low))
        equivalent = resistance / (local.viscosity * local.temperature)
        rate = _poiseuille(local, first * BAR, second * BAR, equivalent)
        mass = rate * _mass_per_pv(local, min(first, second))
        reynolds = 2.0 * mass / (math.pi * radii * along.viscosity(samples))
        temperature = local.temperature
        free_path = along.mean_free_path_at_one_bar(samples)
        with np.errstate(divide="ignore"):
            mean_knudsen = float(np.max(free_path / (2.0 * radii))) / (
                0.5 * (first + second)
            )
        free_path_of = along.mean_free_path_at_one_bar
    with np.errstate(divide="ignore"):
        knudsen = free_path / (fill(samples) * 2.0 * radii)
    peak_re, peak_kn = int(np.argmax(reynolds)), int(np.argmax(knudsen))
    length, fraction = _transitional(fill, radius, samples, free_path_of)
    return StageFlow(
        start=float(samples[0]),
        end=float(samples[-1]),
        start_bar=first,
        end_bar=second,
        temperature=temperature,
        throughput=rate,
        mass_flow=mass,
        peak_reynolds=float(reynolds[peak_re]),
        peak_reynolds_position=float(samples[peak_re]),
        knudsen_start=float(knudsen[0]),
        knudsen_end=float(knudsen[-1]),
        peak_knudsen=float(knudsen[peak_kn]),
        peak_knudsen_position=float(samples[peak_kn]),
        mean_knudsen=float(mean_knudsen),
        transitional_length=length,
        transitional_fraction=fraction,
    )


def _transitional(
    fill: PressureGradient,
    radius: ZProfile,
    samples: Float[np.ndarray, " n_samples"],
    free_path_of: Callable[[Float[np.ndarray, " n_z"]], Float[np.ndarray, " n_z"]],
) -> tuple[float, float]:
    """Return the length and column share of a stage where the gas is transitional.

    ``Kn > 0.1`` is ``p a < lambda(1 bar, T) / 0.2`` in bar and m. Its boundaries are
    bracketed on the samples and refined by a root solve, so a region at a choked end
    a micrometre long is measured exactly; one lying wholly between two interior
    samples (a waist narrower than the sample spacing) would be missed. The column
    density ``int p dz`` is integrated by adaptive quadrature across the fill's
    breakpoints, where ``p`` kinks.

    Parameters
    ----------
    fill : PressureGradient
        The fill.
    radius : ZProfile
        Core radius in m.
    samples : numpy.ndarray
        The stage's sampled positions in m, its breakpoints included.
    free_path_of : callable
        The mean free path at one bar in m at given positions: the same everywhere
        at a constant temperature, the local one along a temperature path.

    Returns
    -------
    tuple of float
        The transitional length in m and its share of the column density.
    """

    def threshold(z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the transitional threshold of ``p a`` at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            ``lambda(1 bar) / (2 Kn_t)`` in bar m.
        """
        return free_path_of(z) / (2.0 * KNUDSEN_TRANSITION_LIMIT)

    def excess(z: float) -> float:
        """Return ``p a`` less its transitional threshold at one position.

        Parameters
        ----------
        z : float
            Position in m.

        Returns
        -------
        float
            ``p(z) a(z) - lambda(1 bar) / 0.2`` in bar m.
        """
        position = np.array([z])
        return float(fill(position)[0] * radius(position)[0] - threshold(position)[0])

    values = fill(samples) * np.asarray(radius(samples), dtype=float) - threshold(
        samples
    )
    edges = [float(samples[0])]
    for k in np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:])):
        # A sample exactly on the threshold is its own root, which brentq returns.
        edges.append(
            float(brentq(excess, samples[k], samples[k + 1], xtol=1e-15, rtol=1e-12))
        )
    edges.append(float(samples[-1]))
    kinks = np.asarray(fill.breakpoints, dtype=float)

    def column(low: float, high: float) -> float:
        """Return the column density between two positions.

        Parameters
        ----------
        low, high : float
            The interval in m.

        Returns
        -------
        float
            ``int p dz`` in bar m.
        """
        inside = kinks[(kinks > low) & (kinks < high)]
        value, _ = quad(
            lambda z: float(fill(np.array([z]))[0]),
            low,
            high,
            points=inside if inside.size else None,
            epsabs=0.0,
            epsrel=1e-10,
            limit=200,
        )
        return float(value)

    total = column(edges[0], edges[-1])
    length = part = 0.0
    for low, high in itertools.pairwise(edges):
        middle = np.array([0.5 * (low + high)])
        if float(fill(middle)[0] * radius(middle)[0]) < float(threshold(middle)[0]):
            length += high - low
            part += column(low, high)
    return length, (part / total if total > 0.0 else 1.0)


def _outer_chokes(
    flows: tuple[_FlowGas, _FlowGas],
    pressures_bar: Float[np.ndarray, " n_positions"],
    exit_radii: tuple[float, float],
    resistances: tuple[float, float],
) -> tuple[float | None, float | None]:
    """Return the choked pressure in bar of each outer end that is a stage's low end.

    Parameters
    ----------
    flows : tuple of _FlowGas
        The gas properties at the first and the last end: the same object at a
        constant temperature, each end's own along a temperature path.
    pressures_bar : numpy.ndarray
        Stage pressures in bar.
    exit_radii : tuple of float
        The core radius in m at the first and the last position.
    resistances : tuple of float
        The ``int a^-4 dz`` in m^-3 of the first and the last stage; along a
        temperature path ``int (eta T) / (eta_e T_e) a^-4 dz`` with the end's own
        ``eta_e T_e``, which turns the choke into ``T_e / (16 c_e a_e^2 int eta T
        a^-4 dz)``.

    Returns
    -------
    tuple
        The first and last end's choked pressure in bar, or ``None`` where that end
        is not lower than the other end of its stage.
    """
    chokes: list[float | None] = []
    for end, other, flow, exit_radius, resistance in (
        (0, 1, flows[0], exit_radii[0], resistances[0]),
        (-1, -2, flows[1], exit_radii[1], resistances[1]),
    ):
        if pressures_bar[end] < pressures_bar[other]:
            high = float(pressures_bar[other]) * BAR
            chokes.append(_choke(flow, high, exit_radius, float(resistance)) / BAR)
        else:
            chokes.append(None)
    return chokes[0], chokes[1]


def _end_flows(
    gas: Composition | Mixture,
    temperature: float | ZProfile,
    radius: ZProfile,
    positions: Float[np.ndarray, " n_positions"],
) -> tuple[
    tuple[_FlowGas, _FlowGas],
    tuple[float, float],
    Float[np.ndarray, " n_stages"],
    _TemperatureFlow | None,
]:
    """Return what the chokes and the report of a fill need, for either temperature.

    Parameters
    ----------
    gas : Gas, str, Species, Mapping or Mixture
        The gas.
    temperature : float or ZProfile
        Temperature in K, validated: a number, or a varying path.
    radius : ZProfile
        Core radius in m.
    positions : numpy.ndarray
        Stage boundaries in m.

    Returns
    -------
    tuple
        The gas at the first and last end, the equivalent resistances of the first
        and last stage (:func:`_outer_chokes`), every stage's resistance (``int
        a^-4 dz`` in m^-3, or ``int eta T a^-4 dz`` in Pa s K m^-3 along a
        temperature path), and the properties along the path (``None`` at a
        constant temperature).

    Raises
    ------
    MaterialError
        If the gas has no transport data at the temperatures.
    """
    if not isinstance(temperature, ZProfile):
        flow = _FlowGas.of(gas, temperature)
        resistances = hydraulic_integral(radius, positions)
        return (
            (flow, flow),
            (float(resistances[0]), float(resistances[-1])),
            resistances,
            None,
        )
    along = _TemperatureFlow.of(gas, temperature, positions)
    resistances = hydraulic_integral(radius, positions, weight=along.weight)
    first, last = along.at(float(positions[0])), along.at(float(positions[-1]))
    return (
        (first, last),
        (
            float(resistances[0]) / (first.viscosity * first.temperature),
            float(resistances[-1]) / (last.viscosity * last.temperature),
        ),
        resistances,
        along,
    )


def _at_choke(pressure: float, choked: float | None) -> bool:
    """Return whether an end pressure is at or below its choked value.

    Parameters
    ----------
    pressure : float
        The end's pressure in bar.
    choked : float or None
        Its choked pressure in bar, or ``None`` where the end is not a low end.

    Returns
    -------
    bool
        ``True`` when the exit throughput is the sonic limit.
    """
    return choked is not None and pressure <= choked * (1.0 + _CHOKE_MATCH)


def flow_report(
    fill: PressureGradient,
    gas: Composition | Mixture,
    *,
    radius: float | ZProfile,
    temperature: float | ZProfile = ROOM_TEMPERATURE,
) -> FlowReport:
    """Return the steady laminar flow a fill implies, stage by stage.

    Parameters
    ----------
    fill : solprop.profiles.PressureGradient
        The fill, in bar.
    gas : Gas, str, Species, Mapping or Mixture
        One built-in gas, gases to mole fractions, or a mixture with a shared fill.
    radius : float or ZProfile
        The core radius in m the fill runs through, > 0.
    temperature : float or ZProfile, optional
        Gas temperature in K, or a path of it along z in m. Along a path the flow
        law is weighted by ``eta(T) T``, the Reynolds number takes the local
        viscosity, the Knudsen numbers the local mean free path, a choked end the
        viscosity and sound speed at its own temperature, and each stage's
        throughput is quoted at its low end's temperature; the report's
        `viscosity` and `sound_speed` are those at the fill's first position.

    Returns
    -------
    FlowReport
        Per stage the throughput, mass flow, peak Reynolds number and Knudsen numbers,
        and the choked state of the outer ends.

    Raises
    ------
    TranslationError
        If the radius is not positive or cannot be integrated.
    MaterialError
        If the gas has no transport data at this temperature, or a mixture has
        independent populations.

    Examples
    --------
    >>> from solprop.gasflow import flow_report, steady_fill
    >>> fill = steady_fill("N2", [0.0, 5.0], [2.9, 0.0], radius=250e-6)
    >>> report = flow_report(fill, "N2", radius=250e-6)
    >>> round(report.peak_reynolds), report.end_is_choked
    (1223, True)
    """
    kelvin = _checked_temperature(temperature, "the flow")
    path = as_profile(radius)
    flows, equivalent, resistances, along = _end_flows(
        gas, kelvin, path, fill.positions
    )
    ends = np.asarray(path(fill.positions[[0, -1]]), dtype=float)
    choked_start, choked_end = _outer_chokes(
        flows, fill.pressures, (float(ends[0]), float(ends[1])), equivalent
    )
    flow = flows[0]
    return FlowReport(
        stages=tuple(
            _stage_flow(flow, fill, path, stage, float(resistance), along)
            for stage, resistance in enumerate(resistances)
        ),
        viscosity=flow.viscosity,
        sound_speed=flow.sound_speed,
        choked_start_bar=choked_start,
        choked_end_bar=choked_end,
        start_is_choked=_at_choke(float(fill.pressures[0]), choked_start),
        end_is_choked=_at_choke(float(fill.pressures[-1]), choked_end),
    )


def temperature_weight(
    gas: Composition | Mixture,
    temperature: float | ZProfile,
    positions: Sequence[float] | np.ndarray,
) -> ZProfile:
    """Return the flow weight ``eta(T) T`` of a gas along a fill, in Pa s K.

    A steady laminar flow of an ideal gas carries the mass flow
    ``mdot = pi a^4 M d(p^2)/dz / (16 R eta(T) T)``, so ``p^2`` is linear in
    ``int eta(T) T a^-4 dz``; :class:`solprop.profiles.PressureGradient` takes this
    weight as its `weight`. Along a temperature path the viscosity is a converged
    Chebyshev series in ``T`` over the temperatures the path takes on the fill, and
    the weight a :class:`solprop.profiles.SeriesOfPath` of the path.

    Parameters
    ----------
    gas : Gas, str, Species, Mapping or Mixture
        One built-in gas, gases to mole fractions, or a mixture with a shared fill.
    temperature : float or ZProfile
        Gas temperature in K, or a path of it along z in m.
    positions : array_like
        The fill's stage boundaries in m, strictly increasing: the span over which
        the temperature range is taken.

    Returns
    -------
    ZProfile
        The weight along z: a :class:`~solprop.profiles.Constant` for a constant
        temperature, a :class:`~solprop.profiles.SeriesOfPath` otherwise.

    Raises
    ------
    TranslationError
        If the positions are invalid.
    MaterialError
        If the temperature is not positive, or the gas has no transport data over
        its range.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.gasflow import temperature_weight
    >>> from solprop.profiles import Linear
    >>> weight = temperature_weight("N2", Linear(293.15, 350.0, 1.0), [0.0, 1.0])
    >>> f"{float(weight(np.array([0.0]))[0]):.4e}"
    '5.1476e-03'
    """
    edges = _checked_edges(positions, "temperature_weight")
    kelvin = _checked_temperature(temperature, "the flow")
    if not isinstance(kelvin, ZProfile):
        return Constant(viscosity(_as_composition(gas), kelvin) * kelvin)
    return _TemperatureFlow.of(gas, kelvin, edges).weight


def steady_fill(  # noqa: PLR0913 - the fill's gas, table, bore, temperature and ends
    gas: Composition | Mixture,
    positions: Sequence[float] | np.ndarray,
    pressures_bar: Sequence[float] | np.ndarray,
    *,
    radius: float | ZProfile,
    temperature: float | ZProfile = ROOM_TEMPERATURE,
    vacuum_end: VacuumEnd = "choked",
    floor: float | None = None,
) -> PressureGradient:
    """Build the steady-flow fill of a bore: taper-aware, choked, regime-checked.

    The fill follows the bore (:class:`solprop.profiles.PressureGradient` with
    `radius`). With ``vacuum_end="choked"`` each outer end that is the low end of its
    stage takes ``max(declared, choked, floor)``: a declared zero replaced by the
    choked pressure is logged at INFO, a nonzero declared pressure raised to it at
    WARNING, naming both. Interior stage pressures are never choked (the chamber
    between two fibres is not modelled); an interior zero keeps the numerical floor,
    which :meth:`~solprop.profiles.PressureGradient.from_stages` logs. With
    ``vacuum_end="floor"`` the declared pressures are kept, exactly as before; the
    fill still follows a varying bore (a uniform one is bitwise the previous fill).

    The flow is then checked (:func:`flow_report`) and warns with
    :class:`~solprop.exceptions.GasFlowWarning` for a peak Reynolds number above
    :data:`REYNOLDS_LAMINAR_LIMIT`, or for a stage whose Knudsen number at its mean
    pressure and narrowest bore exceeds :data:`KNUDSEN_SLIP_LIMIT`, or with more than
    :data:`TRANSITIONAL_COLUMN_LIMIT` of its column density where the Knudsen number
    exceeds :data:`KNUDSEN_TRANSITION_LIMIT`; otherwise each stage's exit Knudsen
    number and transitional length are logged at INFO.

    Parameters
    ----------
    gas : Gas, str, Species, Mapping or Mixture
        One built-in gas, gases to mole fractions (a shared fill), or a mixture with
        a shared fill.
    positions : array_like
        Stage boundaries in m, strictly increasing, at least two.
    pressures_bar : array_like
        Declared pressure in bar at each boundary, >= 0.
    radius : float or ZProfile
        Core radius in m, > 0, over the fill.
    temperature : float or ZProfile, optional
        Gas temperature in K, which sets the viscosity and the sound speed, or a
        path of it along z in m: the fill then follows the flow weight
        ``eta(T) T`` (:class:`solprop.profiles.PressureGradient`), a choked end
        takes the viscosity and the sound speed at its own temperature, and the
        regime checks the local ones (:func:`flow_report`). A constant path is the
        number, bit for bit.
    vacuum_end : {'choked', 'floor'}, optional
        How an outer end below its choked pressure is treated (:data:`VacuumEnd`).
    floor : float or None, optional
        The numerical floor in bar; ``None`` is
        :data:`solprop.profiles.DEFAULT_PRESSURE_FLOOR_FRACTION` of the highest
        stage pressure, and ``0.0`` with ``vacuum_end="floor"`` a true vacuum end.

    Returns
    -------
    solprop.profiles.PressureGradient
        The fill in bar.

    Raises
    ------
    TranslationError
        If the stage table, the radius or `vacuum_end` is invalid.
    MaterialError
        If the gas has no transport data at this temperature (a custom species, or a
        temperature outside the data's range): pass ``vacuum_end="floor"`` to build
        the fill without the choke, whose flow regime is then not checked. A
        temperature path needs the data whatever `vacuum_end` is.

    Warns
    -----
    GasFlowWarning
        If the fill is not laminar, or is transitional between continuum and
        molecular flow.

    Examples
    --------
    >>> from solprop.gasflow import steady_fill
    >>> fill = steady_fill("He", [0.0, 3.0], [5.0, 0.0], radius=125e-6)
    >>> f"{fill.pressures[-1] * 1e3:.2f} mbar"
    '41.18 mbar'
    """
    _checked_choice(vacuum_end, "vacuum_end", get_args(VacuumEnd))
    edges = _checked_edges(positions, "steady_fill")
    declared = _checked_pressures(edges, pressures_bar)
    path = as_profile(radius)
    kelvin = _checked_temperature(temperature, "the steady-flow fill")
    try:
        flows, equivalent, _, along = _end_flows(gas, kelvin, path, edges)
    except MaterialError as error:
        if isinstance(kelvin, ZProfile):
            raise MaterialError(
                f"{error}; a temperature path weights the steady-flow law by the "
                "gas's viscosity, eta(T) T, which needs its transport data whatever "
                "vacuum_end is"
            ) from error
        if vacuum_end == "choked":
            raise MaterialError(
                f"{error}; pass vacuum_end='floor' to build the fill without the "
                "choke, whose flow regime is then not checked"
            ) from error
        logger.info("steady-flow fill built without its flow-regime check: %s", error)
        return PressureGradient.from_stages(edges, declared, floor=floor, radius=path)
    applied = declared.copy()
    if vacuum_end == "choked":
        applied = _apply_chokes(flows, equivalent, edges, declared, path)
    fill = PressureGradient.from_stages(
        edges,
        applied,
        floor=floor,
        radius=path,
        weight=None if along is None else along.weight,
    )
    chokes = (bool(applied[0] != declared[0]), bool(applied[-1] != declared[-1]))
    if any(chokes):
        # The choked ends depend on the gas: marking them lets a composition change
        # that keeps the fill refuse rather than keep a stale end (Mixture.with_fill).
        fill = dataclasses.replace(fill, choked_ends=chokes)
    report = flow_report(fill, flows[0].composition, radius=path, temperature=kelvin)
    _check_regime(report)
    return fill


def _apply_chokes(
    flows: tuple[_FlowGas, _FlowGas],
    resistances: tuple[float, float],
    edges: Float[np.ndarray, " n_positions"],
    declared: Float[np.ndarray, " n_positions"],
    radius: ZProfile,
) -> Float[np.ndarray, " n_positions"]:
    """Raise each outer low end to its choked pressure, logging what changed.

    Parameters
    ----------
    flows : tuple of _FlowGas
        The gas properties at the first and the last end.
    resistances : tuple of float
        The equivalent resistances of the first and last stage
        (:func:`_outer_chokes`).
    edges : numpy.ndarray
        Stage boundaries in m.
    declared : numpy.ndarray
        Declared pressures in bar.
    radius : ZProfile
        Core radius in m.

    Returns
    -------
    numpy.ndarray
        The pressures in bar with every choked end applied.
    """
    ends = np.asarray(radius(edges[[0, -1]]), dtype=float)
    applied = declared.copy()
    chokes = _outer_chokes(
        flows, declared, (float(ends[0]), float(ends[1])), resistances
    )
    for end, choked in zip((0, -1), chokes, strict=True):
        if choked is None or declared[end] >= choked:
            continue
        position = float(edges[end])
        if declared[end] == 0.0:
            logger.info(
                "vacuum end of the steady-flow fill at z = %g m set to its choked "
                "pressure %.4g bar: a fibre exhausting into a pumped chamber cannot "
                "pass more than the sonic throughput (vacuum_end='floor' keeps the "
                "numerical floor instead)",
                position,
                choked,
            )
        else:
            logger.warning(
                "declared end pressure %.4g bar at z = %g m is below the choked "
                "pressure %.4g bar of this flow and was raised to it: the chamber "
                "may be at the declared pressure, but the fibre end cannot be, "
                "because the exit throughput is limited to the sonic one. Pass "
                "vacuum_end='floor' to keep the declared value",
                float(declared[end]),
                position,
                choked,
            )
        applied[end] = choked
    return applied


def _check_regime(report: FlowReport) -> None:
    """Warn or log where a fill leaves the laminar continuum regime.

    Parameters
    ----------
    report : FlowReport
        The fill's flow.

    Warns
    -----
    GasFlowWarning
        For a peak Reynolds number above :data:`REYNOLDS_LAMINAR_LIMIT`, or a stage
        whose mean-pressure Knudsen number exceeds :data:`KNUDSEN_SLIP_LIMIT` or whose
        transitional share of the column density exceeds
        :data:`TRANSITIONAL_COLUMN_LIMIT`.
    """
    if report.peak_reynolds > REYNOLDS_LAMINAR_LIMIT:
        warnings.warn(
            f"the steady-flow fill has a peak Reynolds number of "
            f"{report.peak_reynolds:.0f} at z = {report.peak_reynolds_position:.4g} m, "
            f"above {REYNOLDS_LAMINAR_LIMIT:.0f}: the laminar fill is not valid there, "
            "and output-beam instabilities have been measured in differentially "
            "pumped capillaries from Re = 2040, worst near 2400 (Gebhardt et al., "
            "Opt. Express 29, 6957 (2021)). Lower the pressure difference or widen the "
            "bore",
            GasFlowWarning,
            stacklevel=2,
            skip_file_prefixes=_PACKAGE_FILES,
        )
    for stage in report.stages:
        span = f"{stage.start:.4g} m to {stage.end:.4g} m"
        if (
            stage.mean_knudsen > KNUDSEN_SLIP_LIMIT
            or stage.transitional_fraction > TRANSITIONAL_COLUMN_LIMIT
        ):
            warnings.warn(
                f"the steady-flow fill from {span} is rarefied: its Knudsen number at "
                f"the mean pressure and narrowest bore is {stage.mean_knudsen:.3g} "
                f"(first-order wall slip raises the laminar throughput by 8 Kn; the "
                f"limit is {KNUDSEN_SLIP_LIMIT:g}), and "
                f"{stage.transitional_fraction:.3g} of its gas lies where the "
                f"Knudsen number exceeds {KNUDSEN_TRANSITION_LIMIT:g} (limit "
                f"{TRANSITIONAL_COLUMN_LIMIT:g}). The laminar conductance, the "
                "Reynolds number and the choked estimate are off by ten per cent or "
                "more",
                GasFlowWarning,
                stacklevel=2,
                skip_file_prefixes=_PACKAGE_FILES,
            )
        else:
            logger.info(
                "steady-flow fill from %s: exit Knudsen number %.3g, transitional "
                "(Kn > %g) over %.3g m holding %.3g of the stage's gas",
                span,
                stage.exit_knudsen,
                KNUDSEN_TRANSITION_LIMIT,
                stage.transitional_length,
                stage.transitional_fraction,
            )
