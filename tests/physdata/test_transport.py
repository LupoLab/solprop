# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transport data: viscosity, heat-capacity ratio, sound speed and mean free path.

Every gas's transport properties are optmat's (SEP 3; their agreement with optmat
itself is in test_optmat_transport.py). These tests hold them to independent
measurements and the kinetic-theory relations solprop builds on them, and the
refusals.
"""

import math

import numpy as np
import pytest

from solprop.constants import BAR, K_B, N_A
from solprop.exceptions import MaterialError
from solprop.physdata import Gas, IdealGas, Species, gamma, species
from solprop.physdata.transport import (
    heat_capacity_ratio,
    mean_free_path,
    molar_mass,
    sound_speed,
    temperature_range,
    viscosity,
)

GAS_CONSTANT = K_B * N_A

# NIST Chemistry WebBook (SRD 69) isobaric data at 0.1 MPa and 294 K, micro-Pa s,
# retrieved 2026-09-23.
WEBBOOK_AT_294_K = {Gas.N2: 17.6123, Gas.Ar: 22.3604}

# The same WebBook source at 225 K and every 50 K above, micro-Pa s.
WEBBOOK_MIDPOINTS = {
    Gas.Ne: (26.1921, 30.0409, 33.5823, 36.8988, 40.0433, 43.0503, 45.9430, 48.7381),
    Gas.Kr: (19.5063, 23.5560, 27.3405, 30.8860, 34.2293, 37.4027, 40.4309, 43.3325),
    Gas.Xe: (17.3607, 21.2726, 25.0819, 28.7402, 32.2335, 35.5670, 38.7541, 41.8109),
    Gas.D2: (10.3574, 11.9023, 13.3607, 14.7522, 16.0898, 17.3828, 18.6378, 19.8598),
    Gas.N2O: (11.2376, 13.7246, 16.1118, 18.3808, 20.5338, 22.5814),
}

# Ab initio krypton viscosity at low density, micro-Pa s: F. Sharipov and V. J.
# Benites, Phys. Fluids 32, 077104 (2020), arXiv:2006.08687, Table II (x1 = 0).
KRYPTON_AB_INITIO = {200.0: 17.3212, 300.0: 25.4260, 500.0: 39.4057}


@pytest.mark.parametrize(("gas", "expected"), WEBBOOK_AT_294_K.items())
def test_the_viscosity_matches_the_webbook_at_room_temperature(gas, expected):
    # optmat's viscosity is the zero-density limit and the WebBook's is at 1 bar: the
    # density correction there is 7.5e-4 for both gases (measured), and 1e-3 bounds it.
    assert viscosity(gas, 294.0) == pytest.approx(expected * 1e-6, rel=1e-3)


@pytest.mark.parametrize("gas", list(WEBBOOK_MIDPOINTS))
def test_the_viscosity_agrees_with_the_webbook_across_temperature(gas):
    values = WEBBOOK_MIDPOINTS[gas]
    temperatures = 225.0 + 50.0 * np.arange(len(values))
    got = np.array([viscosity(gas, float(t)) for t in temperatures])
    # Two independent evaluations of the same measurements: measured worst 6.5e-3
    # (D2 at 575 K, 4e-3 for Xe and N2O at 225 K); 1e-2 is the best stated uncertainty
    # of these models (xenon, 1 %).
    np.testing.assert_allclose(got, np.array(values) * 1e-6, rtol=1e-2)


@pytest.mark.parametrize(("temperature", "expected"), KRYPTON_AB_INITIO.items())
def test_krypton_agrees_with_an_independent_ab_initio_value(temperature, expected):
    # The WebBook states 3 % for krypton's gas viscosity; the ab initio values are
    # good to 0.1 %. Measured differences 0.04 %, 0.08 % and 1.2 %.
    assert viscosity("Kr", temperature) == pytest.approx(expected * 1e-6, rel=0.015)


def test_wilke_reduces_to_the_pure_gas():
    pure = viscosity("N2", 294.0)
    assert viscosity({"N2": 1.0}, 294.0) == pure
    # The same gas under two keys is merged before mixing.
    assert viscosity({species("N2"): 0.25, "N2": 0.75}, 294.0) == pure


def test_wilke_gives_the_viscosity_of_air_from_its_constituents():
    # Air is optmat's N2, O2 and Ar mixed by Wilke's rule; nitrogen and oxygen alone
    # give it to the argon's share, 0.24 % (measured), and Wilke's rule is documented
    # to about 1-2 % for such gases (Poling, Prausnitz and O'Connell, The Properties
    # of Gases and Liquids, 5th ed., section 9-5).
    mixed = viscosity({"N2": 0.7885, "O2": 0.2115}, 294.0)
    assert mixed == pytest.approx(viscosity("Air", 294.0), rel=0.01)


def test_the_heat_capacity_ratio_of_a_monatomic_gas_is_five_thirds():
    # optmat's ideal-gas c_p of argon is 5R/2 to rounding.
    assert heat_capacity_ratio("Ar", 294.0) == pytest.approx(5.0 / 3.0, rel=1e-12)
    assert heat_capacity_ratio({"He": 0.5, "Ar": 0.5}, 294.0) == pytest.approx(
        5.0 / 3.0, rel=1e-12
    )


@pytest.mark.parametrize("gas", ["N2", "He", "Xe"])
def test_the_dilute_sound_speed_is_the_ideal_gas_form(gas):
    expected = math.sqrt(
        heat_capacity_ratio(gas, 294.0) * GAS_CONSTANT * 294.0 / molar_mass(gas)
    )
    assert sound_speed(gas, 294.0) == pytest.approx(expected, rel=1e-14)


def test_the_mixture_sound_speed_uses_the_mean_mass_and_mixed_heat_capacities():
    fractions = {"He": 0.3, "Ar": 0.7}
    mean = 0.3 * molar_mass("He") + 0.7 * molar_mass("Ar")
    expected = math.sqrt(5.0 / 3.0 * GAS_CONSTANT * 294.0 / mean)
    assert molar_mass(fractions) == pytest.approx(mean, rel=1e-15)
    # Both noble gases' c_p are 5R/2 to rounding, so the mixture's ratio is 5/3.
    assert sound_speed(fractions, 294.0) == pytest.approx(expected, rel=1e-12)


def test_the_mean_free_path_is_the_viscosity_form():
    eta = viscosity("N2", 294.0)
    mass = molar_mass("N2") / N_A
    expected = eta / (0.5 * BAR) * math.sqrt(math.pi * K_B * 294.0 / (2.0 * mass))
    assert mean_free_path("N2", 0.5, 294.0) == pytest.approx(expected, rel=1e-14)
    # 65 nm at one bar and room temperature, the textbook figure for nitrogen.
    assert mean_free_path("N2", 1.0, 294.0) == pytest.approx(65e-9, rel=0.02)


def test_temperature_ranges_are_optmats_per_gas():
    assert temperature_range("N2") == pytest.approx((63.151, 2000.0))
    assert temperature_range("SF6") == pytest.approx((223.555, 625.0))
    assert temperature_range(species("N2O")) == pytest.approx((182.33, 525.0))


@pytest.mark.parametrize(
    ("gas", "temperature"),
    [("N2", 50.0), ("Xe", 2000.0), ("SF6", 210.0), ("N2O", 600.0), ("He", math.nan)],
)
def test_a_temperature_outside_the_range_is_refused(gas, temperature):
    with pytest.raises(MaterialError, match=r"declared for .* by optmat"):
        viscosity(gas, temperature)


def test_a_custom_species_is_refused_by_name():
    custom = Species("Mystery", gamma(Gas.He), IdealGas())
    with pytest.raises(MaterialError, match="custom species 'Mystery'"):
        viscosity(custom, 294.0)
    with pytest.raises(MaterialError, match="custom species 'Mystery'"):
        temperature_range(custom)


@pytest.mark.parametrize(
    ("fractions", "message"),
    [
        ({"N2": 0.5, "O2": 0.4}, "sum to one"),
        ({"N2": 1.2, "O2": -0.2}, "not negative"),
        ({}, "sum to one"),
    ],
)
def test_invalid_fractions_are_refused(fractions, message):
    with pytest.raises(MaterialError, match=message):
        viscosity(fractions, 294.0)


def test_an_unknown_gas_is_refused():
    with pytest.raises(MaterialError, match="unknown gas"):
        molar_mass("Unobtainium")


@pytest.mark.parametrize("pressure", [0.0, -1.0, math.inf])
def test_a_non_positive_pressure_is_refused(pressure):
    with pytest.raises(MaterialError, match="positive"):
        sound_speed("N2", 294.0, pressure)
    with pytest.raises(MaterialError, match="positive"):
        mean_free_path("N2", pressure, 294.0)


def test_a_real_gas_sound_speed_of_a_mixture_is_refused():
    with pytest.raises(MaterialError, match="pure gas only"):
        sound_speed({"N2": 0.5, "O2": 0.5}, 294.0, 1.0)
