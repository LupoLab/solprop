# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Physical constants in SI units.

Thin, named wrappers over :mod:`scipy.constants` (CODATA) so that the rest of the
package never spells a constant by hand. All values are SI (m, s, kg, J, W, V/m, Pa, K)
as fixed in ``docs/dev/conventions.md``; the two non-SI pressure units are provided as
conversion factors because the user interface accepts pressures in bar.
"""

from scipy import constants as _constants

__all__ = [
    "AMAGAT",
    "ATOMIC_MASS",
    "AU_EFIELD",
    "AU_ENERGY",
    "AU_POLARISABILITY",
    "AU_TIME",
    "BAR",
    "BOHR_RADIUS",
    "CHARGE_MASS_RATIO",
    "EPS0",
    "E_CHARGE",
    "FINE_STRUCTURE",
    "HBAR",
    "K_B",
    "MU0",
    "M_E",
    "N_A",
    "PLANCK",
    "ROOM_TEMPERATURE",
    "STD_ATMOSPHERE",
    "C",
]

C: float = _constants.c
"""Speed of light in vacuum, m/s."""

EPS0: float = _constants.epsilon_0
"""Vacuum permittivity, F/m."""

MU0: float = _constants.mu_0
"""Vacuum permeability, H/m."""

HBAR: float = _constants.hbar
"""Reduced Planck constant, J·s."""

PLANCK: float = _constants.h
"""Planck constant, J·s."""

E_CHARGE: float = _constants.e
"""Elementary charge, C."""

M_E: float = _constants.m_e
"""Electron mass, kg."""

CHARGE_MASS_RATIO: float = E_CHARGE**2 / M_E
"""``e^2 / m_e`` in C^2/kg, the constant of the plasma's free-electron current.

The one derived constant here, rather than a CODATA value: the plasma response and the
compiled plasma coefficients both need it, and the physics layer and the translation
layer may not import each other.
"""

K_B: float = _constants.k
"""Boltzmann constant, J/K."""

N_A: float = _constants.N_A
"""Avogadro constant, 1/mol."""

FINE_STRUCTURE: float = _constants.fine_structure
"""Fine-structure constant, dimensionless."""

BOHR_RADIUS: float = _constants.physical_constants["Bohr radius"][0]
"""Bohr radius, m."""

AU_ENERGY: float = _constants.physical_constants["Hartree energy"][0]
"""Atomic unit of energy (Hartree), J."""

AU_EFIELD: float = _constants.physical_constants["atomic unit of electric field"][0]
"""Atomic unit of electric field, V/m."""

AU_POLARISABILITY: float = _constants.physical_constants[
    "atomic unit of electric polarizability"
][0]
"""Atomic unit of electric polarisability, C^2 m^2 / J.

The SI polarisability, not the volume polarisability: a static field ``E`` shifts a
level by ``-alpha E^2 / 2``, so the unit is an energy divided by a field squared. Equal
to ``e^2 a_0^2 / E_h``.
"""

AU_TIME: float = _constants.physical_constants["atomic unit of time"][0]
"""Atomic unit of time, s."""

BAR: float = _constants.bar
"""One bar in pascal."""

STD_ATMOSPHERE: float = _constants.atm
"""One standard atmosphere in pascal."""

ATOMIC_MASS: float = _constants.physical_constants["atomic mass constant"][0]
"""Atomic mass constant (unified atomic mass unit), kg."""

AMAGAT: float = _constants.atm / (_constants.k * 273.15)
"""Loschmidt constant, m^-3.

The ideal-gas number density at 1 atm and 0 degrees Celsius.

The amagat is this quantity used as a unit; the Raman linewidth coefficients
are tabulated in it. Exact in the SI since 2019, because the atmosphere, the
Boltzmann constant and the temperature are all exact here.
"""

ROOM_TEMPERATURE: float = 293.15
"""The package-wide default temperature, K (``docs/dev/conventions.md`` section 1).

20 °C, the usual laboratory reference and Luna's room temperature (``roomtemp``). Not
a physical constant but a modelling default, kept here so that every module which
needs one spells it the same way.
"""
