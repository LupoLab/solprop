# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""optmat's equation of state behind a species, and mixtures over it."""

import math

import numpy as np
import pytest
from optmat.conditions import State
from optmat.gas.composition import composition
from optmat.gas.eos import air_equation_of_state, equation_of_state

from solprop.constants import ROOM_TEMPERATURE, C
from solprop.exceptions import MaterialError
from solprop.physdata import Gas, Mixture, species
from solprop.physdata.mixtures import Component
from solprop.physdata.selection import MaterialSelection
from solprop.physdata.species import EquationOfState, OptmatGas
from solprop.profiles import Constant, DensityPath, Linear, PressureGradient

SELECTION = MaterialSelection()
GASES = [gas for gas in Gas if gas is not Gas.Air]
# A pressure each gas is a gas at, at room temperature: SF6 condenses near 21 bar,
# N2O near 52.
PRESSURES = (1e-3, 0.5, 1.0, 10.0)


@pytest.mark.parametrize("gas", list(Gas))
def test_density_is_optmats_forward_law_at_the_converted_state(gas):
    eos = OptmatGas(gas)
    assert isinstance(eos, EquationOfState)
    reference = (
        air_equation_of_state() if gas is Gas.Air else equation_of_state(gas.value)
    )
    for pressure in PRESSURES:
        expected = reference.number_density(State(pressure * 1e5, ROOM_TEMPERATURE))
        assert eos.density(pressure, ROOM_TEMPERATURE) == expected


@pytest.mark.parametrize("gas", list(Gas))
def test_density_and_pressure_round_trip(gas):
    eos = OptmatGas(gas)
    for pressure in PRESSURES:
        # Not below 284 K: optmat 0.1.0's inverse brackets its root up to the
        # equation's top pressure, past the melting line of Ar, N2 and CH4 there,
        # and refuses (reported upstream).
        for temperature in (ROOM_TEMPERATURE, 400.0):
            back = eos.pressure(eos.density(pressure, temperature), temperature)
            # optmat's inverse holds N(p(N)) to 1e-12; dN/dp is near N/p.
            assert back == pytest.approx(pressure, rel=2e-12)


def test_vacuum_is_exact_and_does_not_call_optmat():
    eos = OptmatGas("Ar")
    assert eos.density(0.0, ROOM_TEMPERATURE) == 0.0
    assert eos.pressure(0.0, ROOM_TEMPERATURE) == 0.0
    # The equation of state is resolved lazily, so vacuum never builds it.
    assert "_equation" not in vars(eos)


def test_temperature_is_explicit_and_defaults_to_room_temperature():
    eos = OptmatGas("N2")
    assert eos.density(1.0) == eos.density(1.0, ROOM_TEMPERATURE)
    assert eos.density(1.0, 350.0) < eos.density(1.0, ROOM_TEMPERATURE)
    assert eos.pressure(2.5e25) == eos.pressure(2.5e25, ROOM_TEMPERATURE)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda e: e.density(-1.0, ROOM_TEMPERATURE), "non-negative"),
        (lambda e: e.density(math.nan, ROOM_TEMPERATURE), "non-negative"),
        (lambda e: e.density(1.0, 0.0), "temperature"),
        (lambda e: e.pressure(-1.0, ROOM_TEMPERATURE), "non-negative"),
        (lambda e: e.pressure(1e25, -3.0), "temperature"),
    ],
)
def test_invalid_states_are_refused(call, match):
    with pytest.raises(MaterialError, match=match):
        call(OptmatGas("Ar"))


def test_a_condensed_state_is_refused_naming_the_saturation_pressure():
    sf6 = OptmatGas("SF6")
    pascals = equation_of_state("SF6").saturation_pressure(ROOM_TEMPERATURE)
    assert pascals is not None
    saturation = pascals / 1e5
    with pytest.raises(
        MaterialError, match=f"saturation pressure.*{saturation:.6g} bar"
    ):
        sf6.density(30.0, ROOM_TEMPERATURE)
    with pytest.raises(MaterialError, match="saturation pressure"):
        sf6.pressure(5e27, ROOM_TEMPERATURE)


def test_a_temperature_outside_the_equation_is_refused_without_a_saturation_note():
    with pytest.raises(
        MaterialError, match="no gas-phase solution at 1 bar and 5000 K"
    ):
        OptmatGas("Ar").density(1.0, 5000.0)
    # Above the critical temperature there is no saturation pressure to name.
    with pytest.raises(MaterialError) as error:
        OptmatGas("He").density(1e6, ROOM_TEMPERATURE)
    assert "saturation" not in str(error.value)
    # A density beyond the top of a supercritical domain is a true limit.
    with pytest.raises(MaterialError, match="no gas-phase solution at 294 K"):
        OptmatGas("He").pressure(1e30, 294.0)


@pytest.mark.parametrize(
    ("gas", "temperature"), [("Ar", 250.0), ("N2", 280.0), ("CH4", 255.0)]
)
def test_a_cooled_supercritical_fill_inverts_its_density(gas, temperature):
    # optmat 0.1.0's inverse failed here (SEP 3, Q2); 0.1.1 ends the domain at the
    # melting line, and solprop requires it.
    eos = OptmatGas(gas)
    density = eos.density(1.0, temperature)
    assert eos.pressure(density, temperature) == pytest.approx(1.0, rel=1e-12)


def test_a_density_path_takes_the_species_equation_of_state():
    argon = species("Ar", selection=SELECTION)
    gradient = PressureGradient.two_point(1.0, 2.0, 0.0)
    path = DensityPath(argon, gradient, ROOM_TEMPERATURE)
    z = np.array([0.0, 0.5, 1.0])
    expected = [
        argon.equation_of_state.density(float(p), ROOM_TEMPERATURE) for p in gradient(z)
    ]
    # A cubic spline over 1024 samples of a smooth N(p).
    np.testing.assert_allclose(path(z), expected, rtol=1e-11, atol=0)
    warm = DensityPath(argon, Constant(1.0), Linear(ROOM_TEMPERATURE, 350.0, 1.0))
    assert warm(np.array([1.0]))[0] == pytest.approx(
        argon.equation_of_state.density(1.0, 350.0), rel=1e-12
    )


def test_one_component_equals_the_pure_gas():
    pure = OptmatGas("He").density(3.0, ROOM_TEMPERATURE)
    fill = Mixture.fill({"He": 1.0}, 3.0, selection=SELECTION)
    assert float(fill.density(np.array([0.0]))[0, 0]) == pytest.approx(pure, rel=1e-13)


def test_each_population_is_the_pure_gas_at_its_partial_pressure():
    # SEP 3, D17: the partial-pressure rule over each species' optmat equation.
    fill = Mixture.fill({"He": 0.25, "Ar": 0.75}, 4.0, selection=SELECTION)
    densities = fill.density(np.array([0.0]))[:, 0]
    np.testing.assert_allclose(
        densities,
        [
            OptmatGas("He").density(1.0, ROOM_TEMPERATURE),
            OptmatGas("Ar").density(3.0, ROOM_TEMPERATURE),
        ],
        rtol=1e-12,
    )


def test_splitting_or_reordering_labels_changes_no_material_result():
    omega = 2 * math.pi * C / np.array([400e-9, 800e-9])
    z = np.array([0.0])
    whole = Mixture.fill({"Ar": 1.0}, 2.0, selection=SELECTION)
    argon = species("Ar", selection=SELECTION)
    halves = Mixture.of(
        Component(argon, DensityPath(argon, Constant(1.0)), "a"),
        Component(argon, DensityPath(argon, Constant(1.0)), "b"),
    )
    # Two partial pressures of one bar are not one of two: the real-gas density is
    # not linear in pressure. Equal densities are what must agree.
    density = whole.total_density(z)[0]
    split = halves.with_densities({"a": density / 3, "b": 2 * density / 3})
    np.testing.assert_allclose(
        split.susceptibility(omega, z), whole.susceptibility(omega, z), rtol=4e-16
    )
    mixed = Mixture.fill({"He": 0.5, "Ne": 0.5}, 1.0, selection=SELECTION)
    swapped = Mixture.fill({"Ne": 0.5, "He": 0.5}, 1.0, selection=SELECTION)
    np.testing.assert_allclose(
        mixed.susceptibility(omega, z), swapped.susceptibility(omega, z), rtol=4e-16
    )


def test_a_vacuum_mixture_is_exactly_zero():
    fill = Mixture.fill({"He": 0.5, "Ar": 0.5}, 0.0, selection=SELECTION)
    omega = 2 * math.pi * C / np.array([800e-9])
    assert np.all(fill.density(np.array([0.0])) == 0.0)
    assert np.all(fill.susceptibility(omega, np.array([0.0])) == 0.0)


def test_named_air_compositions_are_optmats():
    for argon, name in ((True, "dry air (N2, O2, Ar)"), (False, "dry air (N2, O2)")):
        selected = Mixture.air(1.0, argon=argon, selection=SELECTION)
        expected = tuple(c.mole_fraction for c in composition(name))
        assert selected.fractions == expected
        assert selected.labels == tuple(c.material for c in composition(name))
        assert Mixture.air(1.0, argon=argon).fractions == expected  # the default
    # Explicit fractions are the caller's, whichever species they resolve to.
    custom = Mixture.air(1.0, fractions={"N2": 0.8, "O2": 0.2}, selection=SELECTION)
    assert custom.fractions == (0.8, 0.2)


def test_mixture_records_each_model_once():
    fill = Mixture.partial_pressures(
        {"N2": 1.0, "O2": 0.5}, selection=SELECTION, temperature=300.0
    )
    names = [r.name for r in fill.material_records]
    assert names == [
        "fit/N2",
        "eos/N2",
        "transport/N2",
        "hyperpolarizability/N2",
        "fit/O2",
        "eos/O2",
        "transport/O2",
        "hyperpolarizability/O2",
    ]
    twice = Mixture.of(
        *fill.components,
        *(Component(c.species, c.density, f"{c.label}-2") for c in fill.components),
    )
    assert len(twice.material_records) == len(names)


def test_gas_air_takes_the_pseudo_pure_density_and_the_populations_their_own():
    # SEP 3, section 4.2: the partial-pressure populations are 1.9e-4 lower at 1 bar.
    pseudo = OptmatGas(Gas.Air).density(1.0, ROOM_TEMPERATURE)
    populations = Mixture.air(1.0, selection=SELECTION).total_density(np.array([0.0]))
    assert float(populations[0]) / pseudo - 1 == pytest.approx(-1.9e-4, abs=0.2e-4)
