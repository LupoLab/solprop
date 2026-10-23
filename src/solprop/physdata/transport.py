# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Transport properties of the fill gases: viscosity, sound speed, mean free path.

The steady-flow fill of a hollow fibre (:mod:`solprop.gasflow`) needs three gas
properties beyond the equation of state: the dynamic viscosity, which sets the
Poiseuille conductance; the speed of sound, which bounds the throughput of an end
that exhausts into vacuum; and the mean free path, which says whether the gas is a
continuum at all. All three are **dilute-gas** values here, which is what the laminar
flow law and the choke use.

The material properties are optmat's (SEP 3), through the
:class:`~solprop.physdata.species.OptmatGas` a built-in gas or species carries: the
dilute viscosity, the ideal-gas heat capacity and the molar mass of
``optmat.gas.transport``, each over optmat's own temperature range; the real-gas
sound speed of its equation of state; and, for a mixture, optmat's Wilke rule
(``mixture_viscosity``; C. R. Wilke, J. Chem. Phys. **18**, 517 (1950),
https://doi.org/10.1063/1.1747673, about 1 to 2 % accurate). solprop keeps:

* the **dilute sound speed** :math:`c = \sqrt{\gamma R T / M}` with the dilute
  heat-capacity ratio :math:`\gamma = c_p / c_v`, :math:`c_v = c_p - R`, and for a
  mixture the mole-weighted molar mass and
  :math:`\gamma = \sum x_i c_{p,i} / \sum x_i c_{v,i}` (ideal-gas mixing of heat
  capacities);
* the **mean free path**, the viscosity-based kinetic-theory form
  :math:`\lambda = (\eta / p)\sqrt{\pi k_B T / (2 m)}`, which needs no collision
  diameter. Other conventions exist; the Knudsen thresholds that use it assume this
  form and the core diameter.

A temperature outside a gas's range is refused, as is a custom
:class:`~solprop.physdata.species.Species` that does not carry optmat's equation of
state, which carries none of these data.

Pressures are in **bar** at this interface (``docs/dev/conventions.md`` section 1).
"""

import math
from collections.abc import Mapping
from typing import Final

from solprop.constants import BAR, K_B, N_A
from solprop.exceptions import MaterialError
from solprop.physdata.gases import Gas, as_gas
from solprop.physdata.species import OptmatGas, Species

__all__ = [
    "heat_capacity_ratio",
    "mean_free_path",
    "molar_mass",
    "sound_speed",
    "temperature_range",
    "viscosity",
]

# A single gas: a built-in gas's name, a Gas, or a Species carrying optmat's data.
GasLike = Gas | str | Species

# A single gas, or a mapping of gases to mole fractions summing to one. The mapping is
# spelled out per key type because a mapping is invariant in its key: a dict of names,
# of members or of species is each accepted as it is written. Neither alias is in
# __all__: they are annotations, documented where they are used.
Composition = (
    GasLike
    | Mapping[Gas, float]
    | Mapping[str, float]
    | Mapping[Species, float]
    | Mapping[GasLike, float]
)

_GAS_CONSTANT: Final = K_B * N_A


def temperature_range(gas: GasLike) -> tuple[float, float]:
    """Return the temperatures over which a gas's transport properties are valid.

    Parameters
    ----------
    gas : Gas, str or Species
        A built-in gas, or a species carrying optmat's equation of state.

    Returns
    -------
    tuple of float
        The lowest and highest temperature in K, optmat's.

    Raises
    ------
    MaterialError
        If the gas is unknown or a custom species.

    Examples
    --------
    >>> from solprop.physdata.transport import temperature_range
    >>> temperature_range("N2O")
    (182.33, 525.0)
    """
    return _member(gas, "transport").temperature_range


def _member(gas: GasLike, quantity: str) -> OptmatGas:
    """Return the data source behind a declaration, or refuse a custom species.

    Parameters
    ----------
    gas : Gas, str or Species
        The gas.
    quantity : str
        The property asked for, named in the error.

    Returns
    -------
    OptmatGas
        The optmat equation of state and transport of a name or a species.

    Raises
    ------
    MaterialError
        If the gas is unknown, or a custom species without optmat's data.
    """
    if isinstance(gas, Species):
        state = gas.equation_of_state
        if isinstance(state, OptmatGas):
            return state
        raise MaterialError(
            f"custom species {gas.name!r} carries no {quantity} data: the gas-flow "
            "calculation (viscosity, sound speed, mean free path) needs a built-in "
            "gas (solprop.physdata.Gas), whose properties are optmat's"
        )
    # A name is the built-in gas, as for species(name): a flow built from a name and a
    # fill built from it agree.
    return OptmatGas(as_gas(gas))


def _composition(
    gas: Composition, quantity: str
) -> tuple[tuple[OptmatGas, float], ...]:
    """Resolve a gas or a mole-fraction mapping to its data sources and fractions.

    Parameters
    ----------
    gas : Gas, str, Species or Mapping
        One gas, or gases to mole fractions summing to one.
    quantity : str
        The property asked for, named in errors.

    Returns
    -------
    tuple
        ``(member, fraction)`` pairs with positive fractions, one per distinct gas.

    Raises
    ------
    MaterialError
        If a fraction is negative or not finite, the fractions do not sum to one, or
        a gas carries no optmat data.
    """
    if not isinstance(gas, Mapping):
        return ((_member(gas, quantity), 1.0),)
    merged: dict[OptmatGas, float] = {}
    for key, fraction in gas.items():
        value = float(fraction)
        if not math.isfinite(value) or value < 0.0:
            raise MaterialError("mole fractions must be finite and not negative")
        member = _member(key, quantity)
        merged[member] = merged.get(member, 0.0) + value
    total = math.fsum(merged.values())
    # A distribution normalised in floating point is one to a few ulps per entry.
    if not merged or abs(total - 1.0) > 1e-12:
        raise MaterialError(f"mole fractions must sum to one, got {total:.17g}")
    return tuple((member, value) for member, value in merged.items() if value > 0.0)


def _viscosity(parts: tuple[tuple[OptmatGas, float], ...], temperature: float) -> float:
    """Return a composition's dilute viscosity.

    Parameters
    ----------
    parts : tuple
        ``(OptmatGas, fraction)`` pairs from :func:`_composition`.
    temperature : float
        Temperature in K.

    Returns
    -------
    float
        Dynamic viscosity in Pa s: the gas's own, or optmat's Wilke rule over the
        constituents (air's resolved into its own).
    """
    from optmat.gas.transport import mixture_viscosity  # noqa: PLC0415 - optmat is lazy

    members = [member for member, _ in parts]
    values = [member.viscosity(temperature) for member in members]
    if len(parts) == 1:
        return values[0]
    fractions: dict[str, float] = {}
    for member, (_, share) in zip(members, parts, strict=True):
        for name, fraction in member.constituents:
            fractions[name] = fractions.get(name, 0.0) + share * fraction
    return float(mixture_viscosity(fractions, temperature))


def molar_mass(gas: Composition) -> float:
    """Return the molar mass of a gas, or the mole-weighted mean of a mixture.

    optmat's molar mass of each gas; for ``Gas.Air`` that of the pseudo-pure dry air
    its equation of state describes.

    Parameters
    ----------
    gas : Gas, str, Species or Mapping
        One built-in gas, or gases to mole fractions summing to one.

    Returns
    -------
    float
        Molar mass in kg mol^-1.

    Raises
    ------
    MaterialError
        If a gas is unknown or a custom species, or the fractions are invalid.

    Examples
    --------
    >>> from solprop.physdata.transport import molar_mass
    >>> f"{molar_mass('N2'):.6f}"
    '0.028013'
    >>> f"{molar_mass({'He': 0.5, 'Ar': 0.5}):.6f}"
    '0.021975'
    """
    return math.fsum(
        fraction * member.molar_mass
        for member, fraction in _composition(gas, "molar-mass")
    )


def viscosity(gas: Composition, temperature: float) -> float:
    """Return the dilute dynamic viscosity of a gas or a mixture.

    Parameters
    ----------
    gas : Gas, str, Species or Mapping
        One built-in gas, or gases to mole fractions summing to one, which take
        Wilke's mixing rule.
    temperature : float
        Temperature in K, inside :func:`temperature_range` of every gas.

    Returns
    -------
    float
        Dynamic viscosity in Pa s.

    Raises
    ------
    MaterialError
        If a gas is unknown or a custom species, the fractions are invalid, or the
        temperature is outside a gas's range.

    Examples
    --------
    >>> from solprop.physdata.transport import viscosity
    >>> f"{viscosity('N2', 293.15):.4e}"
    '1.7560e-05'
    >>> f"{viscosity('Xe', 300.0):.4e}"
    '2.3146e-05'
    """
    return _viscosity(_composition(gas, "viscosity"), temperature)


def heat_capacity_ratio(gas: Composition, temperature: float) -> float:
    """Return the dilute heat-capacity ratio of a gas or a mixture.

    Parameters
    ----------
    gas : Gas, str, Species or Mapping
        One built-in gas, or gases to mole fractions summing to one.
    temperature : float
        Temperature in K, inside :func:`temperature_range` of every gas.

    Returns
    -------
    float
        ``gamma = sum(x c_p) / sum(x (c_p - R))``, dimensionless.

    Raises
    ------
    MaterialError
        If a gas is unknown or a custom species, the fractions are invalid, or the
        temperature is outside a gas's range.

    Examples
    --------
    >>> from solprop.physdata.transport import heat_capacity_ratio
    >>> f"{heat_capacity_ratio('Ar', 293.15):.4f}"
    '1.6667'
    >>> f"{heat_capacity_ratio('N2', 293.15):.3f}"
    '1.400'
    """
    parts = _composition(gas, "heat-capacity")
    isobaric = math.fsum(
        fraction * member.molar_heat_capacity(temperature) for member, fraction in parts
    )
    return isobaric / (isobaric - _GAS_CONSTANT)


def sound_speed(
    gas: Composition, temperature: float, pressure_bar: float | None = None
) -> float:
    """Return the adiabatic speed of sound in a gas or a mixture.

    Parameters
    ----------
    gas : Gas, str, Species or Mapping
        One built-in gas, or gases to mole fractions summing to one.
    temperature : float
        Temperature in K, inside :func:`temperature_range` of every gas.
    pressure_bar : float or None, optional
        ``None`` (the default) gives the dilute value
        ``sqrt(gamma R T / M)`` of :func:`heat_capacity_ratio` and
        :func:`molar_mass`. A pressure in **bar**, > 0, gives the real-gas value
        from the equation of state, for a pure gas only.

    Returns
    -------
    float
        Speed of sound in m s^-1.

    Raises
    ------
    MaterialError
        If a gas is unknown or a custom species, the fractions are invalid, the
        temperature is outside a gas's range, the pressure is not positive, or a
        pressure is given for a mixture.

    Examples
    --------
    >>> from solprop.physdata.transport import sound_speed
    >>> f"{sound_speed('N2', 293.15):.1f}"
    '349.0'
    >>> f"{sound_speed('N2', 293.15, 1.0):.1f}"
    '349.1'
    """
    if pressure_bar is None:
        ratio = heat_capacity_ratio(gas, temperature)
        return math.sqrt(ratio * _GAS_CONSTANT * temperature / molar_mass(gas))
    parts = _composition(gas, "sound-speed")
    if len(parts) != 1:
        raise MaterialError(
            "a real-gas sound speed at a stated pressure is available for a pure gas "
            "only; pass pressure_bar=None for a mixture's dilute value"
        )
    if not (math.isfinite(pressure_bar) and pressure_bar > 0.0):
        raise MaterialError(f"pressure must be positive, got {pressure_bar} bar")
    return parts[0][0].sound_speed(pressure_bar, temperature)


def mean_free_path(gas: Composition, pressure_bar: float, temperature: float) -> float:
    r"""Return the mean free path of a gas from its viscosity.

    Uses :math:`\lambda = (\eta / p)\sqrt{\pi k_B T / (2 m)}` with the molecular
    mass :math:`m = M / N_A` (for a mixture, the mole-weighted mean) and the viscosity
    of :func:`viscosity`: the hard-sphere kinetic-theory form written so that it needs
    no collision diameter.

    Parameters
    ----------
    gas : Gas, str, Species or Mapping
        One built-in gas, or gases to mole fractions summing to one.
    pressure_bar : float
        Pressure in **bar**, > 0.
    temperature : float
        Temperature in K, inside :func:`temperature_range` of every gas.

    Returns
    -------
    float
        Mean free path in m.

    Raises
    ------
    MaterialError
        If the pressure is not positive, or as :func:`viscosity`.

    Examples
    --------
    >>> from solprop.physdata.transport import mean_free_path
    >>> f"{mean_free_path('N2', 1.0, 293.15) * 1e9:.1f}"
    '64.9'
    """
    if not (math.isfinite(pressure_bar) and pressure_bar > 0.0):
        raise MaterialError(f"pressure must be positive, got {pressure_bar} bar")
    speed = math.sqrt(math.pi * _GAS_CONSTANT * temperature / (2.0 * molar_mass(gas)))
    return viscosity(gas, temperature) / (pressure_bar * BAR) * speed
