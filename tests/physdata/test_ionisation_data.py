# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The first ionisation potential of every built-in gas: optmat's (SEP 3).

solprop's own ionisation tables -- quantum numbers, polarisabilities, the MO-ADK
coefficients -- were removed with the rate formulae that used them (Phase 4b); the
potential a species carries, and that a plasma pays per ionisation event, is optmat's
first ionisation energy.
"""

import json
from pathlib import Path

import pytest

from solprop.constants import AU_ENERGY, E_CHARGE
from solprop.exceptions import MaterialError
from solprop.physdata import Gas, ionisation_potential, species

REFERENCE = json.loads(
    (Path(__file__).parents[1] / "data/luna_physdata.json").read_text()
)
WITH_POTENTIAL = [gas for gas in Gas if gas is not Gas.Air]

# Measured first ionisation energies in eV, for an order-of-magnitude sanity check that
# does not depend on Luna: every gas here ionises between 12 and 25 eV.
POTENTIAL_RANGE_EV = (12.0, 25.0)


@pytest.mark.parametrize("gas", WITH_POTENTIAL)
def test_potential_is_in_joules_and_physically_plausible(gas):
    low, high = POTENTIAL_RANGE_EV
    assert low < ionisation_potential(gas) / E_CHARGE < high


@pytest.mark.parametrize("gas", WITH_POTENTIAL)
def test_potential_converts_to_atomic_units(gas):
    # The rate formulae work in hartree; the conversion must be exact, not a second
    # tabulation.
    hartree = ionisation_potential(gas) / AU_ENERGY
    assert 0.4 < hartree < 1.0


@pytest.mark.parametrize("gas", WITH_POTENTIAL)
def test_a_built_in_species_carries_the_potential(gas):
    data = species(gas).ionisation
    assert data is not None
    assert data.potential == ionisation_potential(gas)
    assert data.citation


def test_air_has_no_single_potential():
    with pytest.raises(MaterialError, match="air has no single ionisation potential"):
        ionisation_potential(Gas.Air)
    assert species(Gas.Air).ionisation is None


def test_unknown_gas_is_rejected():
    with pytest.raises(MaterialError, match="unknown gas"):
        ionisation_potential("Argon")


@pytest.mark.parametrize(
    "name", ["He", "Ne", "Ar", "Kr", "Xe", "N2", "H2", "D2", "O2", "CH4", "N2O"]
)
def test_potential_agrees_with_luna(name):
    # Deliberately solprop's own data (optmat's, which takes no inputs to freeze): what
    # is measured is its difference from Luna's table.
    # optmat's first ionisation energies against Luna's own table: measured 9.5e-4 at
    # worst (N2O), 5.4e-4 for Kr and 4.1e-4 for CH4, the rest below 1e-4. Before SEP 3
    # solprop carried Luna's values, exact bar the CODATA hartree.
    assert ionisation_potential(Gas(name)) == pytest.approx(
        REFERENCE["ionisation_potential"][name], rel=1e-3
    )


def test_sulfur_hexafluoride_replaces_lunas_placeholder():
    # Luna's SF6 entry is 0.5 hartree = 13.61 eV, the atomic-hydrogen value copied from
    # its :H branch. NIST lists adiabatic determinations of 15.30 and 15.35 eV; optmat
    # takes 15.35.
    assert ionisation_potential(Gas.SF6) / E_CHARGE == pytest.approx(15.35, rel=1e-12)
    luna_ev = REFERENCE["ionisation_potential"]["SF6"] / E_CHARGE
    assert luna_ev == pytest.approx(13.606, rel=1e-3)


def test_helium_has_the_highest_potential():
    # A structural check: helium is the least ionisable gas there is.
    assert ionisation_potential(Gas.He) == max(
        ionisation_potential(gas) for gas in WITH_POTENTIAL
    )
