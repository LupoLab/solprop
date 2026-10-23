# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math

from scipy import constants as sc

from solprop import constants


def test_constants_match_scipy():
    assert sc.c == constants.C
    assert sc.epsilon_0 == constants.EPS0
    assert sc.mu_0 == constants.MU0
    assert sc.hbar == constants.HBAR
    assert sc.e == constants.E_CHARGE
    assert constants.BAR == 1e5
    assert constants.STD_ATMOSPHERE == 101325.0


def test_vacuum_impedance_identity():
    # Since the 2019 SI redefinition mu0 is measured, not exact, so c^2 eps0 mu0 = 1
    # holds only to the relative uncertainty of mu0 (1.6e-10, CODATA 2022).
    assert math.isclose(
        constants.C**2 * constants.EPS0 * constants.MU0, 1.0, rel_tol=1e-9
    )


def test_atomic_units_are_consistent():
    # E_au = e / (4 pi eps0 a0^2), to the rounding of the tabulated values.
    expected = constants.E_CHARGE / (
        4 * math.pi * constants.EPS0 * constants.BOHR_RADIUS**2
    )
    assert math.isclose(constants.AU_EFIELD, expected, rel_tol=1e-9)
    assert math.isclose(
        constants.AU_TIME, constants.HBAR / constants.AU_ENERGY, rel_tol=1e-9
    )


def test_all_names_are_exported():
    for name in constants.__all__:
        assert isinstance(getattr(constants, name), float)
