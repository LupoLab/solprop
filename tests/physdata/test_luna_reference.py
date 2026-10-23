# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Comparison against values computed from Luna on this machine.

The stored file is a regression witness, not truth: where solprop deliberately differs
from Luna the difference is asserted exactly, not merely tolerated. solprop carries no
gas laws of its own since SEP 3; the laws compared are the published fits supplied as a
caller's own data (tests/physdata/former_gas_laws.py), which tests the supplied types
against an independent implementation of the same expressions. It was generated once, by
``tests/data/generate_luna_reference.py``, and is frozen: never regenerated, every input
the comparison depends on stated in the test (``docs/dev/testing.md``); the file records
the Luna commit it came from.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from former_gas_laws import (
    AIR_BORZSONYI,
    AR_BORZSONYI,
    AR_KINGSTON,
    D2_PECK,
    H2_PECK,
    HE_BORZSONYI,
    HE_ERMOLOV,
    KR_BORZSONYI,
    N2_BORZSONYI,
    NE_BORZSONYI,
    O2_ZHANG,
    QUANFU_HE_COEFFICIENTS,
    XE_BORZSONYI,
    quanfu_he,
)

from solprop.constants import C
from solprop.exceptions import MaterialError
from solprop.physdata import Gas, density

REFERENCE = json.loads(
    (Path(__file__).parents[1] / "data/luna_physdata.json").read_text()
)
STATE = (1.0, 294.0)
"""The witness's state, bar and K (its provenance block)."""

# The witness is frozen (``docs/dev/testing.md``), so every input is stated: the state
# above, and the real-gas equation of state Luna's CoolProp is (``ideal=False``, today's
# default) at every density below.
REAL_GAS = {"ideal": False}

# Laws where solprop reproduces Luna's expression exactly, supplied as a caller's own
# data (tests/physdata/former_gas_laws.py: the published fits solprop's former
# registry carried, the inputs of this witness). The residual is the two equations of
# state, which differ by a constant ratio per gas (test_density_matches_luna below);
# it cancels, because n - 1 depends only on the ratio of the density to the
# reference density. Measured 3.2e-12 at worst (helium), as before SEP 3.
FAITHFUL = {
    "Air-borzsonyi": AIR_BORZSONYI,
    "Ar-borzsonyi": AR_BORZSONYI,
    "Ar-kingston": AR_KINGSTON,
    "D2-peck": D2_PECK,
    "H2-peck": H2_PECK,
    "He-borzsonyi": HE_BORZSONYI,
    "Kr-borzsonyi": KR_BORZSONYI,
    "N2-borzsonyi": N2_BORZSONYI,
    "Ne-borzsonyi": NE_BORZSONYI,
    "O2-zhang": O2_ZHANG,
    "Xe-borzsonyi": XE_BORZSONYI,
}
FAITHFUL_TOLERANCE = 1e-11


def stored(section, key):
    return np.asarray(REFERENCE[section][key], dtype=float)


def omega_for(key):
    return 2.0 * math.pi * C / stored("wavelengths", key)


def susceptibility(law, omega):
    """Return chi and d chi / d omega of a supplied law at the witness's state."""
    number_density = density(law.gas, *STATE, **REAL_GAS)
    return (
        law.susceptibility(omega) * number_density,
        law.dsusceptibility_domega(omega) * number_density,
    )


def refractivity(law, omega):
    """Return n - 1 without the cancellation, ``chi / (sqrt(1 + chi) + 1)``."""
    chi, _ = susceptibility(law, omega)
    return chi / (np.sqrt(1.0 + chi) + 1.0)


def test_provenance_is_recorded():
    provenance = REFERENCE["provenance"]
    assert len(provenance["luna_commit"]) == 40
    assert provenance["luna_branch"] == "modal-fixed"
    assert provenance["julia_version"].startswith("julia version")
    assert provenance["pressure_bar"] == 1.0


def test_every_comparable_model_is_classified():
    # A new Luna entry must be triaged, not silently skipped.
    documented = (
        set(FAITHFUL)
        | {"He-ermolov"}
        | {f"{gas.value}-quanfu-he" for gas in QUANFU_HE_COEFFICIENTS}
    )
    assert set(REFERENCE["refractive_index"]) == documented


@pytest.mark.parametrize("key", FAITHFUL)
def test_refractive_index_matches_luna(key):
    ours = refractivity(FAITHFUL[key], omega_for(key))
    theirs = stored("refractive_index", key) - 1.0
    # Compared as n - 1: the two agree to 16 digits in n itself, which would hide a
    # 1e-8 error in the refractivity that is all a gas simulation depends on.
    assert np.max(np.abs(ours / theirs - 1.0)) < FAITHFUL_TOLERANCE


@pytest.mark.parametrize("key", FAITHFUL)
def test_analytic_beta1_matches_luna_finite_differences(key):
    # Luna computes beta_1 with high-order adaptive central differences of omega n / c.
    # Agreeing with it to 1e-12 validates the closed-form derivative against a wholly
    # independent implementation -- the check ADR-006 asks for.
    omega = omega_for(key)
    chi, dchi = susceptibility(FAITHFUL[key], omega)
    index = np.sqrt(1.0 + chi)
    derivative = dchi / (2.0 * index)
    ours = (index + omega * derivative) / C
    assert np.max(np.abs(ours / stored("beta1", key) - 1.0)) < 1e-12


def test_helium_differs_from_luna_by_exactly_the_reference_state():
    # solprop normalises the Ermolov expansion by the 1 atm density, Luna by the 1 bar
    # density. Nothing else differs, so the ratio of the two susceptibilities is exactly
    # the ratio of the two reference densities -- 1.0132, i.e. Luna's helium
    # refractivity is 1.3 % high. The relation is exact in chi rather than in n - 1,
    # which carries a second-order term chi^2 / 8; the tolerance is set by Luna's
    # stored n, whose n^2 - 1 keeps about twelve digits.
    key = "He-ermolov"
    ours, _ = susceptibility(HE_ERMOLOV, omega_for(key))
    theirs = stored("refractive_index", key) ** 2 - 1.0
    expected = HE_ERMOLOV.reference_density / density(Gas.He, 1.0, 273.15, **REAL_GAS)
    assert np.allclose(theirs / ours, expected, rtol=1e-9)
    assert expected == pytest.approx(1.01324, rel=1e-4)


@pytest.mark.parametrize("gas", list(QUANFU_HE_COEFFICIENTS))
def test_quanfu_he_models_differ_from_luna_by_the_missing_square(gas):
    # Luna's gamma_QuanfuHe returns n / rho where every other model returns
    # (n^2 - 1) / rho, so its susceptibility is larger by n / (n^2 - 1). With
    # n - 1 = d that is (1 + d) / (d (2 + d)): about 1200 for methane, whose d is
    # 4e-4, and about 650 for the more refractive sulfur hexafluoride.
    key = f"{gas.value}-quanfu-he"
    a, b, c, _ = QUANFU_HE_COEFFICIENTS[gas]
    wavenumber = 1e-2 / stored("wavelengths", key)
    published = 1e-8 * (a + b / (c - wavenumber**2))

    ours, _ = susceptibility(quanfu_he(gas), omega_for(key))
    theirs = stored("refractive_index", key)
    ratio = (theirs**2 - 1.0) / ours
    expected = (1.0 + published) / (published * (2.0 + published))
    assert np.allclose(ratio, expected, rtol=1e-9)
    assert np.all(expected > 500.0)
    # And the visible consequence: Luna reports a glass-like index for a gas.
    assert np.all(theirs > 1.4)


def test_density_matches_luna():
    """optmat's equations of state and Luna's CoolProp differ by a constant per gas.

    The ratio is the same at every state of a gas to 4e-11 (N2O, near its saturation
    line; 3e-14 elsewhere), so the two are one equation of state up to a constant --
    the molar mass or gas constant each code carries: 5.7e-6 for Air, Ar, CH4 and
    N2, 1.1e-6 for H2, Kr, N2O, SF6 and Xe, -3.6e-7 for He and Ne, -7.8e-8 for D2 and
    -1.5e-5 for O2 (measured). Before SEP 3 solprop's own CoolProp route differed by
    the CODATA Avogadro constant alone, 1.6e-8. A state optmat places in the liquid
    or two-phase region, where CoolProp returned a liquid density, is refused.
    """
    ratios = {}
    refused = set()
    for key, luna in REFERENCE["density"].items():
        if luna is None:
            continue
        gas, pressure, temperature = key.split("-")
        try:
            ours = density(Gas(gas), float(pressure), float(temperature), **REAL_GAS)
        except MaterialError:
            refused.add(key)
            continue
        ratios.setdefault(gas, []).append(ours / luna - 1.0)
    assert sum(len(values) for values in ratios.values()) > 150
    assert refused == {
        "N2O-40.0-200.0",
        "N2O-5.0-200.0",
        "SF6-40.0-200.0",
        "SF6-40.0-294.0",
        "SF6-5.0-200.0",
        "Xe-40.0-200.0",
    }
    for values in ratios.values():
        assert max(values) - min(values) < 1e-10
        assert max(abs(value) for value in values) < 2e-5
