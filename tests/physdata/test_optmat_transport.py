# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transport properties of optmat-selected species, and the gas flow over them."""

import math

import pytest
from optmat.conditions import State
from optmat.gas.eos import air_equation_of_state, equation_of_state
from optmat.gas.transport import mixture_viscosity, transport_model

from solprop.constants import K_B, N_A
from solprop.exceptions import MaterialError, MaterialStatusWarning
from solprop.gasflow import mass_flow
from solprop.physdata import Gas, Mixture, species
from solprop.physdata.selection import MaterialSelection
from solprop.physdata.species import OptmatGas
from solprop.physdata.transport import (
    heat_capacity_ratio,
    molar_mass,
    sound_speed,
    temperature_range,
    viscosity,
)

SELECTION = MaterialSelection()
GAS_CONSTANT = K_B * N_A
# CH4's viscosity is provisional in optmat 0.1.0, and warns; it has its own test.
QUIET = ["He", "Ne", "Ar", "Kr", "Xe", "H2", "D2", "N2", "O2", "SF6", "N2O"]


def selected(gas):
    return species(gas, selection=SELECTION)


@pytest.mark.parametrize("gas", QUIET)
def test_pure_properties_are_optmats(gas):
    model = transport_model(gas)
    chosen = selected(gas)
    assert viscosity(chosen, 294.0) == float(model.viscosity(294.0))
    assert molar_mass(chosen) == model.molar_mass
    cp = float(model.heat_capacity(294.0))
    assert heat_capacity_ratio(chosen, 294.0) == pytest.approx(
        cp / (cp - GAS_CONSTANT), rel=1e-15
    )
    real = equation_of_state(gas).speed_of_sound(State(2e5, 294.0))
    assert sound_speed(chosen, 294.0, 2.0) == real


def test_a_mixture_takes_optmats_wilke_rule():
    he, ar = selected("He"), selected("Ar")
    composition = {he: 0.3, ar: 0.7}
    assert viscosity(composition, 300.0) == pytest.approx(
        float(mixture_viscosity({"He": 0.3, "Ar": 0.7}, 300.0)), rel=1e-15
    )
    assert molar_mass(composition) == pytest.approx(
        0.3 * transport_model("He").molar_mass + 0.7 * transport_model("Ar").molar_mass,
        rel=1e-15,
    )
    # One population split into two labels is the same composition.
    assert viscosity({he: 0.3, ar: 0.4, selected("Ar"): 0.3}, 300.0) == pytest.approx(
        viscosity(composition, 300.0), rel=1e-15
    )
    with pytest.raises(MaterialError, match="pure gas only"):
        sound_speed(composition, 300.0, 1.0)


def test_air_takes_its_constituents_transport():
    air = OptmatGas(Gas.Air)
    names = [name for name, _ in air.constituents]
    assert names == ["N2", "O2", "Ar"]
    fractions = dict(air.constituents)
    assert air.viscosity(294.0) == pytest.approx(
        float(mixture_viscosity(fractions, 294.0)), rel=1e-15
    )
    # Inside a mixture, air's constituents join the others'.
    with_helium = {species("Air", selection=SELECTION): 0.5, selected("He"): 0.5}
    joined = {name: 0.5 * x for name, x in fractions.items()} | {"He": 0.5}
    assert viscosity(with_helium, 294.0) == pytest.approx(
        float(mixture_viscosity(joined, 294.0)), rel=1e-15
    )
    assert heat_capacity_ratio(species("Air", selection=SELECTION), 294.0) == (
        pytest.approx(1.40, abs=2e-3)
    )
    kinds = [r.kind for r in air.material_records]
    assert kinds == ["equation_of_state", "transport", "transport", "transport"]
    assert air.temperature_range[1] <= 1700.0


def test_air_mass_is_the_pseudo_pure_equations_own():
    # The density of Gas.Air is the pseudo-pure dry air's (Lemmon 2000); its molar
    # mass must be the same fluid's, not the N2/O2/Ar mean, for a mass density.
    assert air_equation_of_state().fluid == "Air"
    air = OptmatGas(Gas.Air)
    # Lemmon et al. (2000), 28.9655 g/mol, read from the same CoolProp fluid.
    assert air.molar_mass == pytest.approx(0.0289655, rel=1e-5)
    mean = sum(x * transport_model(name).molar_mass for name, x in air.constituents)
    assert air.molar_mass / mean - 1 == pytest.approx(2.0e-4, abs=0.2e-4)
    chosen = species("Air", selection=SELECTION)
    assert molar_mass(chosen) == air.molar_mass
    # The mass flow uses density times that mass: rho / p = N M / (N_A p).
    throughput, pressure = 1e-6, 2.0
    density = air.density(pressure, 294.0)
    expected = throughput * density * air.molar_mass / N_A / (pressure * 1e5)
    assert mass_flow(chosen, throughput, pressure) == pytest.approx(expected, rel=1e-14)


def test_temperature_range_is_optmats_and_refused_outside():
    chosen = selected("N2O")
    low, high = temperature_range(chosen)
    assert (low, high) == pytest.approx((182.33, 525.0))
    with pytest.raises(MaterialError, match=r"declared for .* by optmat"):
        viscosity(chosen, 600.0)
    with pytest.raises(MaterialError, match="by optmat"):
        heat_capacity_ratio(chosen, math.nan)
    with pytest.raises(MaterialError, match="pressure must be positive"):
        sound_speed(chosen, 294.0, 0.0)
    with pytest.raises(MaterialError, match="pressure must be positive"):
        OptmatGas("N2O").sound_speed(-1.0, 294.0)


def test_sound_speed_refuses_a_condensed_state():
    with pytest.raises(MaterialError, match="saturation pressure"):
        sound_speed(selected("SF6"), 294.0, 30.0)


def test_a_provisional_viscosity_warns_once_per_flow():
    with pytest.warns(MaterialStatusWarning, match="transport/CH4") as caught:
        fill = Mixture.fill(
            {"CH4": 1.0},
            (1.0, 0.1),
            length=1.0,
            radius=100e-6,
            selection=SELECTION,
        )
    assert len(caught) == 1
    assert fill.shared is not None
