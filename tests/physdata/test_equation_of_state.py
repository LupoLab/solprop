# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The equation of state behind every built-in gas: optmat's (SEP 3)."""

import numpy as np
import pytest

from solprop.constants import BAR, K_B, STD_ATMOSPHERE
from solprop.exceptions import MaterialError
from solprop.physdata import Gas, density, ideal_gas_density, pressure

# To first order in the virial expansion Z = 1 + B(T) rho, so the real gas departs from
# the ideal one by |B rho|. The worst case here is SF6 (B ~ -280 cm^3/mol at 294 K, i.e.
# -4.6e-28 m^3 per particle), giving 1.2e-3 at 0.1 bar and 1.2e-5 at 1 mbar. The
# measured SF6 deviations are 1.16e-3 and 1.16e-5, so the bounds hold with ~40 % margin.
IDEAL_GAS_TOLERANCE_100_MBAR = 2e-3
IDEAL_GAS_TOLERANCE_1_MBAR = 2e-5


@pytest.mark.parametrize("gas", list(Gas))
def test_density_approaches_the_ideal_gas_at_low_pressure(gas):
    assert density(gas, 0.1) == pytest.approx(
        ideal_gas_density(0.1), rel=IDEAL_GAS_TOLERANCE_100_MBAR
    )
    assert density(gas, 1e-3) == pytest.approx(
        ideal_gas_density(1e-3), rel=IDEAL_GAS_TOLERANCE_1_MBAR
    )


@pytest.mark.parametrize("gas", list(Gas))
def test_departure_from_the_ideal_gas_is_affine_in_pressure(gas):
    # The virial expansion Z = 1 + B(T) rho + C(T) rho^2 makes the relative departure
    # from p/(k_B T) affine in pressure up to the C term, so its second difference over
    # equally spaced pressures must vanish. It does not vanish exactly because
    # optmat's equations of state carry their own gas constant, which differs from
    # the current N_A k_B by up to 7e-8 relative (measured) and shows up as a pressure-
    # independent offset -- hence a second difference rather than a ratio test. The
    # largest measured residual is SF6 at 2.1e-4 of the span, so 1e-3 keeps 5x in hand.
    def departure(bar):
        return density(gas, bar) / ideal_gas_density(bar) - 1.0

    low, mid, high = departure(0.01), departure(0.02), departure(0.03)
    second_difference = low - 2.0 * mid + high
    assert abs(second_difference) < 1e-3 * abs(high - low)


@pytest.mark.parametrize("gas", list(Gas))
def test_zero_pressure_limit_of_the_departure_is_the_gas_constant_offset(gas):
    # Extrapolating the departure to p = 0 leaves only optmat's per-fluid gas constant
    # against N_A k_B,
    # 7e-8 at most (measured). Bounding it at 2e-5 pins the accuracy of the ideal-gas
    # fallback.
    def departure(bar):
        return density(gas, bar) / ideal_gas_density(bar) - 1.0

    low, high = departure(0.01), departure(0.03)
    assert abs(low - 0.5 * (high - low)) < 2e-5


@pytest.mark.parametrize("gas", list(Gas))
def test_density_matches_the_ideal_flag_at_the_same_state(gas):
    assert density(gas, 0.1, ideal=True) == ideal_gas_density(0.1)


def test_ideal_gas_density_is_the_loschmidt_constant_at_stp():
    # Loschmidt constant, CODATA 2022: 2.686 780 111e25 m^-3 at 1 atm, 273.15 K. Exact
    # here because k_B and the atmosphere are both exact in the SI since 2019.
    assert ideal_gas_density(STD_ATMOSPHERE / BAR, 273.15) == pytest.approx(
        2.686780111e25, rel=1e-9
    )


@pytest.mark.parametrize("gas", list(Gas))
def test_density_is_strictly_increasing_in_pressure(gas):
    pressures = [0.0, 0.01, 0.1, 1.0, 5.0, 20.0]
    values = [density(gas, p) for p in pressures]
    assert np.all(np.diff(values) > 0.0)


@pytest.mark.parametrize("gas", list(Gas))
def test_density_is_decreasing_in_temperature(gas):
    assert density(gas, 1.0, 500.0) < density(gas, 1.0, 294.0)


def test_density_is_linear_in_pressure_in_the_ideal_limit():
    # A real-gas check of Boyle's law: at 1e-3 bar the deviation is below 1e-5.
    assert density(Gas.Ar, 2e-3) == pytest.approx(2 * density(Gas.Ar, 1e-3), rel=1e-5)


@pytest.mark.parametrize("gas", list(Gas))
def test_zero_pressure_gives_zero_density(gas):
    # The equation of state is undefined at p = 0; a gradient fill ending in vacuum
    # needs this short circuit.
    assert density(gas, 0.0) == 0.0
    assert density(gas, 0.0, ideal=True) == 0.0


@pytest.mark.parametrize("ideal", [False, True])
@pytest.mark.parametrize("gas", list(Gas))
def test_pressure_inverts_density(gas, ideal):
    for want in (0.05, 1.0, 20.0):
        got = pressure(gas, density(gas, want, ideal=ideal), ideal=ideal)
        # The round trip is two equation-of-state evaluations, each converged by
        # optmat to ~1e-10 relative; the ideal path is exact to rounding.
        assert got == pytest.approx(want, rel=1e-9)


def test_pressure_of_zero_density_is_zero():
    assert pressure(Gas.Ar, 0.0) == 0.0
    assert pressure(Gas.Ar, 0.0, ideal=True) == 0.0


def test_ideal_pressure_is_the_ideal_gas_law():
    assert pressure(Gas.Ar, 1e25, 300.0, ideal=True) == pytest.approx(
        1e25 * K_B * 300.0 / BAR, rel=1e-15
    )


@pytest.mark.parametrize("func", [density, pressure])
def test_unknown_gas_is_rejected(func):
    with pytest.raises(MaterialError, match="unknown gas"):
        func("Argon", 1.0)


@pytest.mark.parametrize(
    ("func", "bad"), [(density, -1e-12), (pressure, -1.0), (ideal_gas_density, -1.0)]
)
def test_negative_argument_is_rejected(func, bad):
    with pytest.raises(MaterialError, match="must not be negative"):
        func(bad) if func is ideal_gas_density else func(Gas.Ar, bad)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_non_positive_temperature_is_rejected(temperature):
    with pytest.raises(MaterialError, match="temperature must be positive"):
        density(Gas.Ar, 1.0, temperature)
    with pytest.raises(MaterialError, match="temperature must be positive"):
        pressure(Gas.Ar, 1e25, temperature)
    with pytest.raises(MaterialError, match="temperature must be positive"):
        ideal_gas_density(1.0, temperature)


def test_state_outside_the_equation_of_state_is_rejected():
    # 1 K is below argon's triple point, outside optmat's 83.806-2000 K, so it refuses.
    with pytest.raises(MaterialError, match="no gas-phase solution at 1 bar and 1 K"):
        density(Gas.Ar, 1.0, 1.0)


def test_pressure_outside_the_equation_of_state_is_rejected():
    # The inverse refuses the same state rather than extrapolating into the solid.
    with pytest.raises(MaterialError, match="no gas-phase solution at 1 K"):
        pressure(Gas.Ar, 1e25, 1.0)


def test_a_condensed_state_is_rejected_naming_the_saturation_pressure():
    # SF6 at 294 K condenses near 21 bar: optmat refuses rather than return a liquid.
    with pytest.raises(MaterialError, match="saturation pressure"):
        density(Gas.SF6, 40.0, 294.0)
