# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The window glasses: named optmat models, analytic derivative, range, lookup.

Every built-in window glass is optmat's (tests/physdata/test_optmat_solids.py tests
the recommended models). These tests check named Sellmeier models of each window
glass through the factories and :func:`solprop.physdata.glass`.
"""

import math

import numpy as np
import pytest

from solprop.constants import C
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.physdata import (
    OptmatGlass,
    barium_fluoride,
    bk7,
    calcium_fluoride,
    fused_silica,
    glass,
    glass_names,
    magnesium_fluoride,
    sapphire,
)

# A named Sellmeier model of each window glass, by its factory.
MAPPED = {
    "caf2-daimon": (calcium_fluoride, "daimon_2002/sellmeier_20C"),
    "caf2-malitson": (calcium_fluoride, "malitson_1963/sellmeier"),
    "mgf2": (magnesium_fluoride, "dodge_1984/sellmeier_o"),
    "bk7": (bk7, "schott_2023/sellmeier"),
    "sapphire": (sapphire, "weber_2003/sellmeier_o"),
    "baf2": (barium_fluoride, "malitson_1964/sellmeier"),
}


def omega_of(wavelength):
    return 2.0 * math.pi * C / wavelength


def mapped(name):
    factory, model = MAPPED[name]
    return factory(model=model)


@pytest.mark.parametrize("name", MAPPED)
def test_a_named_model_returns_that_optmat_model(name):
    _, model = MAPPED[name]
    index = mapped(name)
    assert isinstance(index, OptmatGlass)
    assert index.source == model


def test_bk7_has_the_index_that_defines_it():
    # SCHOTT defines N-BK7 by n_d = 1.51680 in air; optmat's model is absolute, so the
    # model gives n_d times the index of air, 1 + 2.7e-4 (the d-line
    # refractivity of dry air, 15-20 C), to the catalogue's five decimals.
    index = float(np.real(mapped("bk7").index(omega_of(587.5618e-9))))
    assert index == pytest.approx(1.51680 * (1.0 + 2.74e-4), abs=1e-5)


@pytest.mark.parametrize("name", MAPPED)
def test_the_frequency_derivative_is_the_analytic_one(name):
    model = mapped(name)
    shortest, longest = model.wavelength_range
    omega = omega_of(np.geomspace(shortest * 1.05, longest * 0.95, 9))
    step = 1e-5 * omega
    # A fourth-order central difference. Its rounding error is eps n / h, 1e-11 of n,
    # but dn/domega is a hundredth of n / omega, so relative to the derivative it is
    # 1e-9: measured 5e-10 to 2e-9 across the six models.
    numerical = (
        -np.real(model.index(omega + 2.0 * step))
        + 8.0 * np.real(model.index(omega + step))
        - 8.0 * np.real(model.index(omega - step))
        + np.real(model.index(omega - 2.0 * step))
    ) / (12.0 * step)
    np.testing.assert_allclose(
        np.real(model.dindex_domega(omega)), numerical, rtol=1e-6
    )
    assert np.all(np.imag(model.index(omega)) == 0.0)
    # Normal dispersion throughout the transparency range: the wavelengths ascend.
    assert np.all(np.diff(np.real(model.index(omega))) < 0.0)


@pytest.mark.parametrize("name", MAPPED)
def test_a_request_outside_the_source_range_warns_on_either_side(name):
    model = mapped(name)
    shortest, longest = model.wavelength_range
    for wavelength in (0.9 * shortest, 1.1 * longest):
        with pytest.warns(MaterialRangeWarning):
            model.index(omega_of(wavelength))
    assert model.citation
    assert 100e-9 < shortest < longest < 20e-6


def test_glass_finds_every_name_in_any_case():
    # optmat's adapters are equal by their model.
    assert glass("CaF2") == calcium_fluoride()
    caf2 = "malitson_1963/sellmeier"
    assert glass(" caf2 ", model=caf2) == calcium_fluoride(model=caf2)
    assert glass("MgF2", axis="o") == magnesium_fluoride(axis="o")
    assert glass("N-BK7") == bk7()
    assert glass("Al2O3", axis="e") == sapphire(axis="e")
    assert glass("BaF2") == barium_fluoride()
    silica = "malitson_1965/sellmeier"
    assert glass("silica", model=silica) == fused_silica(model=silica)
    assert glass("SiO2").source == "fit/SiO2"
    assert set(glass_names()) >= {"sio2", "fused silica", "bk7", "al2o3"}


def test_an_unknown_name_is_refused_by_name():
    with pytest.raises(MaterialError, match="unknown glass 'ZnSe'"):
        glass("ZnSe")
