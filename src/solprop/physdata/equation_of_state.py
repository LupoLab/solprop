# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Gas number density and pressure from the equation of state.

:func:`density` is optmat's real-gas equation of state (SEP 3), CoolProp's
Helmholtz-energy equations through ``optmat.gas.eos``, adapted by
:class:`~solprop.physdata.species.OptmatGas`; :func:`pressure` is its exact inverse, and
:func:`ideal_gas_density` the ideal-gas value for comparison.

Pressures are in **bar**, the one documented departure from SI at the user interface
(``docs/dev/conventions.md`` §1); number densities are in m^-3 and temperatures in K.
optmat imports CoolProp, which costs about a second, so the import happens inside the
functions and ``import solprop`` stays cheap; lookups are cached, and none is made at
module load. The manual page is ``docs/guide/media/materials.md``, "Density".
"""

import functools
import logging
import math
import numbers

from solprop.constants import BAR, K_B, ROOM_TEMPERATURE
from solprop.exceptions import MaterialError
from solprop.physdata.gases import Gas, as_gas

__all__ = ["density", "ideal_gas_density", "pressure"]

logger = logging.getLogger(__name__)


@functools.cache
def _optmat_density(gas: Gas, pressure: float, temperature: float) -> float:
    """Return optmat's real-gas number density (cached).

    - ``pressure``: absolute pressure in bar, > 0.
    - ``temperature``: temperature in K, > 0.
    - Returns: number density in m^-3.
    """
    # species imports this module for the ideal-gas law, so the adapter is imported
    # where it is needed.
    from solprop.physdata.species import OptmatGas  # noqa: PLC0415 - a module cycle

    return OptmatGas(gas).density(pressure, temperature)


@functools.cache
def _optmat_pressure(gas: Gas, number_density: float, temperature: float) -> float:
    """Return optmat's real-gas pressure at a number density (cached).

    - ``number_density``: number density in m^-3, > 0.
    - ``temperature``: temperature in K, > 0.
    - Returns: absolute pressure in bar.
    """
    from solprop.physdata.species import OptmatGas  # noqa: PLC0415 - as above

    return OptmatGas(gas).pressure(number_density, temperature)


def _check_temperature(temperature: object) -> None:
    """Reject a non-physical temperature at the boundary rather than inside CoolProp.

    - ``temperature``: temperature in K; typed ``object`` because refusing what is not a
      number, a temperature path above all, is this check's job: one state has one
      temperature.
    """
    if isinstance(temperature, bool) or not isinstance(temperature, numbers.Real):
        raise MaterialError(
            "the equation of state takes one temperature in K, got "
            f"{type(temperature).__name__}: a temperature path along z reaches the "
            "density through solprop.profiles.DensityPath, not a single state"
        )
    value = float(temperature)
    if not (math.isfinite(value) and value > 0.0):
        raise MaterialError(
            f"temperature must be finite and positive, got {temperature} K"
        )


def _check_state(quantity: str, value: float, unit: str, temperature: object) -> None:
    """Refuse a state the ideal law would turn into NaN or infinity, on every path.

    - ``quantity``: what ``value`` is, for the message: a pressure or a number density.
    - ``unit``: the unit of ``value`` as the message prints it (``" bar"``, or empty).
    """
    if not (math.isfinite(value) and value >= 0.0):
        raise MaterialError(
            f"{quantity} must be finite and non-negative, got {value}{unit}"
        )
    _check_temperature(temperature)


def ideal_gas_density(pressure: float, temperature: float = ROOM_TEMPERATURE) -> float:
    """Return the ideal-gas number density.

    ``rho = p / (k_B T)``. Independent of the species, so it takes no gas argument.

    Parameters
    ----------
    pressure : float
        Absolute pressure in **bar** (the documented non-SI user-interface unit), >= 0.
    temperature : float, optional
        Temperature in K, > 0. Defaults to 293.15 K.

    Returns
    -------
    float
        Number density in m^-3.

    Raises
    ------
    MaterialError
        If the pressure is negative or not finite, or the temperature is not positive
        and finite.

    Examples
    --------
    The Loschmidt constant is the ideal-gas density at 1 atm and 0 degrees Celsius.

    >>> from solprop.physdata import ideal_gas_density
    >>> f"{ideal_gas_density(1.01325, 273.15):.6e}"
    '2.686780e+25'
    """
    _check_state("pressure", pressure, " bar", temperature)
    return pressure * BAR / (K_B * temperature)


def density(
    gas: Gas | str,
    pressure: float,
    temperature: float = ROOM_TEMPERATURE,
    *,
    ideal: bool = False,
) -> float:
    """Return the number density of a gas.

    Parameters
    ----------
    gas : Gas or str
        The species.
    pressure : float
        Absolute pressure in **bar** (the documented non-SI user-interface unit), >= 0.
        Zero returns zero without consulting the equation of state, which is undefined
        there; a pressure gradient terminating in vacuum needs this.
    temperature : float, optional
        Temperature in K, > 0. Defaults to 293.15 K.
    ideal : bool, optional
        Use ``p / (k_B T)`` instead of the equation of state. Default ``False``.

    Returns
    -------
    float
        Number density in m^-3.

    Raises
    ------
    MaterialError
        If the gas is unknown, the pressure is negative or not finite, the
        temperature is not positive and finite, or the state lies outside the
        equation of state's domain; the ideal law refuses the same states.

    Notes
    -----
    The equation of state is optmat's real-gas one: at 293.15 K the deviation from
    ideal is about 0.14 % at 2 bar and 2.5 % at 40 bar for argon, so it matters for the
    high-pressure fills used in self-compression.

    Examples
    --------
    >>> from solprop.physdata import Gas, density
    >>> f"{density(Gas.Ar, 1.0):.6e}"
    '2.472418e+25'
    >>> density(Gas.He, 0.0)
    0.0
    """
    species = as_gas(gas)
    _check_state("pressure", pressure, " bar", temperature)
    if pressure == 0.0:
        return 0.0
    if ideal:
        return ideal_gas_density(pressure, temperature)
    return _optmat_density(species, float(pressure), float(temperature))


def pressure(
    gas: Gas | str,
    number_density: float,
    temperature: float = ROOM_TEMPERATURE,
    *,
    ideal: bool = False,
) -> float:
    """Return the pressure at which a gas reaches a given number density.

    The inverse of :func:`density` at fixed temperature.

    Parameters
    ----------
    gas : Gas or str
        The species.
    number_density : float
        Number density in m^-3, >= 0. Zero returns zero.
    temperature : float, optional
        Temperature in K, > 0. Defaults to 293.15 K.
    ideal : bool, optional
        Use ``p = rho k_B T`` instead of the equation of state. Default ``False``.

    Returns
    -------
    float
        Absolute pressure in **bar** (the documented non-SI user-interface unit).

    Raises
    ------
    MaterialError
        If the gas is unknown, the density is negative or not finite, the
        temperature is not positive and finite, or the state lies outside the
        equation of state's domain; the ideal law refuses the same states.

    Examples
    --------
    >>> from solprop.physdata import Gas, density, pressure
    >>> f"{pressure(Gas.Ar, density(Gas.Ar, 5.0)):.6f}"
    '5.000000'
    """
    species = as_gas(gas)
    _check_state("number density", number_density, " m^-3", temperature)
    if number_density == 0.0:
        return 0.0
    if ideal:
        return number_density * K_B * temperature / BAR
    return _optmat_pressure(species, float(number_density), float(temperature))
