# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from solprop.exceptions import MaterialError
from solprop.physdata import Gas, as_gas, is_molecule


def test_gas_members_are_strings():
    assert Gas.He == "He"
    assert Gas.N2O.value == "N2O"
    assert isinstance(Gas.Ar, str)


def test_gas_registry_is_the_thirteen_species():
    assert {g.value for g in Gas} == {
        "Air",
        "He",
        "Ne",
        "Ar",
        "Kr",
        "Xe",
        "N2",
        "H2",
        "D2",
        "O2",
        "CH4",
        "SF6",
        "N2O",
    }


@pytest.mark.parametrize("gas", list(Gas))
def test_as_gas_round_trips_members_and_names(gas):
    assert as_gas(gas) is gas
    assert as_gas(gas.value) is gas


@pytest.mark.parametrize("name", ["he", "Helium", "Ar2", "", "N20"])
def test_as_gas_rejects_unknown_names(name):
    # Deliberately case-sensitive: "he" is a typo, not a synonym.
    with pytest.raises(MaterialError, match="unknown gas"):
        as_gas(name)


def test_as_gas_error_lists_the_known_species():
    with pytest.raises(MaterialError, match="He, Ne, Ar"):
        as_gas("Argon")


@pytest.mark.parametrize("gas", ["He", "Ne", "Ar", "Kr", "Xe"])
def test_the_noble_gases_are_not_molecules(gas):
    assert not is_molecule(gas)


@pytest.mark.parametrize("gas", ["N2", "H2", "D2", "O2", "CH4", "SF6", "N2O"])
def test_every_other_species_is_a_molecule(gas):
    assert is_molecule(gas)


def test_air_counts_as_molecular():
    # A mixture of diatomic molecules, so the molecular ionisation rate is the right
    # default for it; Luna's ADK list leaves it with the atomic one.
    assert is_molecule(Gas.Air)


def test_the_two_classes_partition_the_registry():
    # A new species must land in one class or the other, never be left unclassified.
    molecules = {gas for gas in Gas if is_molecule(gas)}
    atoms = {gas for gas in Gas if not is_molecule(gas)}
    assert molecules | atoms == set(Gas)
    assert not molecules & atoms


def test_is_molecule_rejects_an_unknown_species():
    with pytest.raises(MaterialError, match="unknown gas"):
        is_molecule("Unobtainium")
