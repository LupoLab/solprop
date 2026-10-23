# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Third-order data: optmat's Kerr coefficient at a carrier, and the n2 conversion.

Every gas's Kerr coefficient is optmat's, evaluated at the run's carrier (SEP 3;
the adapter's agreement with optmat itself is in test_optmat_kerr.py). These tests
hold the scaling and the conversion solprop builds on it.
"""

import pytest

from solprop.constants import EPS0, C
from solprop.exceptions import MaterialError
from solprop.physdata import (
    Gas,
    chi3,
    chi3_from_n2,
    density,
    gamma3,
    n2,
    n2_from_chi3,
)

CARRIER = 800e-9
WITH_DATA = [gas for gas in Gas if gas is not Gas.Air]


# ------------------------------------------------------------------ the gases


def test_air_has_no_coefficient_of_its_own():
    # Air is a composition: its populations each carry theirs (Mixture.air).
    with pytest.raises(MaterialError, match="no Kerr coefficient of its own"):
        gamma3(Gas.Air, wavelength=CARRIER)
    with pytest.raises(MaterialError, match="no Kerr coefficient for Air"):
        chi3(Gas.Air, pressure=1.0, wavelength=CARRIER)


def test_unknown_gas_is_rejected():
    with pytest.raises(MaterialError, match="unknown gas"):
        gamma3("Argon", wavelength=CARRIER)


def test_the_coefficient_needs_a_carrier():
    with pytest.raises(MaterialError, match="carrier wavelength"):
        chi3(Gas.Ar, pressure=1.0)
    with pytest.raises(MaterialError, match="carrier wavelength"):
        n2(Gas.Ar, pressure=1.0)


# ----------------------------------------------------------------- the conversion


@pytest.mark.parametrize("index", [1.0, 1.0003, 1.45])
@pytest.mark.parametrize("value", [1e-27, 3e-26, 5e-20])
def test_n2_and_chi3_round_trip(value, index):
    # Exact to rounding: the conversion is one multiplication and its inverse.
    assert chi3_from_n2(n2_from_chi3(value, index), index) == pytest.approx(
        value, rel=1e-15
    )
    assert n2_from_chi3(chi3_from_n2(value, index), index) == pytest.approx(
        value, rel=1e-15
    )


def test_conversion_is_the_stated_formula():
    assert n2_from_chi3(3e-26, 1.2) == pytest.approx(
        3.0 * 3e-26 / (4.0 * EPS0 * C * 1.2**2), rel=1e-15
    )


# ------------------------------------------------------------------ the scaling


@pytest.mark.parametrize("gas", WITH_DATA)
def test_chi3_is_linear_in_number_density(gas):
    low = chi3(gas, pressure=0.5, wavelength=CARRIER)
    high = chi3(gas, pressure=5.0, wavelength=CARRIER)
    assert high / low == pytest.approx(density(gas, 5.0) / density(gas, 0.5), rel=1e-15)


@pytest.mark.parametrize("gas", WITH_DATA)
def test_chi3_is_the_per_particle_value_times_the_density(gas):
    assert chi3(
        gas, pressure=3.0, temperature=350.0, wavelength=CARRIER
    ) == pytest.approx(
        gamma3(gas, wavelength=CARRIER) * density(gas, 3.0, 350.0), rel=1e-15
    )


@pytest.mark.parametrize("gas", WITH_DATA)
def test_chi3_vanishes_in_vacuum(gas):
    assert chi3(gas, pressure=0.0, wavelength=CARRIER) == 0.0
    assert n2(gas, pressure=0.0, wavelength=CARRIER) == 0.0


@pytest.mark.parametrize("gas", WITH_DATA)
def test_n2_is_almost_linear_in_pressure(gas):
    # n2 = 3 chi3 / (4 eps0 c n0^2) and chi3 is exactly linear in density, so the only
    # departure from proportionality is n0^2, which differs from one by 2 (n0 - 1) --
    # below 1.6e-3 at 1 bar for the most refractive gas here, xenon.
    ratio = n2(gas, pressure=1.0, wavelength=CARRIER) / n2(
        gas, pressure=0.1, wavelength=CARRIER
    )
    density_ratio = density(gas, 1.0) / density(gas, 0.1)
    assert ratio == pytest.approx(density_ratio, rel=2e-3)


# ------------------------------------------------------------ absolute magnitudes


def test_helium_n2_matches_the_literature():
    # The most quoted gas value: n2(He) is 3.6e-25 m^2/W near 1 bar and 800 nm.
    # optmat's gives 3.62e-25 (measured).
    assert n2(Gas.He, pressure=1.0, wavelength=CARRIER) == pytest.approx(
        3.6e-25, rel=0.02
    )


@pytest.mark.parametrize("gas", WITH_DATA)
def test_every_gas_is_self_focusing_and_of_a_plausible_size(gas):
    # All of these gases have a positive electronic Kerr coefficient, spanning about
    # two decades from helium to xenon.
    value = n2(gas, pressure=1.0, wavelength=CARRIER)
    assert 1e-25 < value < 1e-22
