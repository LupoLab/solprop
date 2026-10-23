# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The electronic Kerr coefficient from optmat, at the carrier (SEP 3, section 4.3)."""

import math
import warnings

import numpy as np
import pytest
from optmat.gas.nonlinear import hyperpolarizability_models

from solprop.constants import EPS0, ROOM_TEMPERATURE, C
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import (
    Gas,
    Mixture,
    Species,
    _optmat,
    chi3,
    chi3_from_n2,
    gamma,
    n2,
    n2_from_chi3,
    species,
)
from solprop.physdata.nonlinear_data import (
    CarrierKerr,
    OptmatKerr,
    carriers_agree,
    selected_carrier,
)
from solprop.physdata.selection import MaterialSelection

SELECTION = MaterialSelection()
GASES = [gas for gas in Gas if gas is not Gas.Air]
CARRIERS = (532e-9, 800e-9, 1030e-9, 2.0e-6)


def optmat_model(gas):
    return hyperpolarizability_models()[gas.value]


# N2O's Kerr fit is extrapolated at 532 nm (beyond its fitted dispersion), which both
# the adapter and the direct reference evaluation say; the identity holds there too.
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings("ignore::optmat.exceptions.ModelRangeWarning")
@pytest.mark.parametrize("gas", GASES)
def test_the_taylor_factor_six_converts_optmat_gamma_t_to_solprop_gamma3(gas):
    adapter = OptmatKerr(gas)
    model = optmat_model(gas)
    for carrier in CARRIERS:
        # gamma3 = gamma_T / (6 eps0): one division by six and one by eps0, from
        # optmat's SI gamma_T, which is itself the atomic-unit fit times one factor.
        expected = model.gamma_si("kerr", carrier) / (6.0 * EPS0)
        assert adapter.gamma3(carrier) == pytest.approx(float(expected), rel=4e-16)
    assert adapter.material_records[0].name == f"hyperpolarizability/{gas.value}"
    assert adapter.material_records[0].digest == model.identity.digest


def test_the_coefficient_is_dispersive_so_no_carrier_binds_one_value():
    argon = species("Ar", selection=SELECTION)
    assert argon.gamma3 is None
    values = [float(argon.kerr_coefficient(carrier) or 0.0) for carrier in CARRIERS]
    # The Kerr hyperpolarizability grows towards its resonances in the ultraviolet.
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)
    with pytest.raises(MaterialError, match="give the carrier wavelength"):
        argon.kerr_coefficient()


def test_a_callers_species_keeps_its_fixed_coefficient():
    # A caller's own number, 1.2e-63 m^5/V^2 (test data, argon's order of magnitude).
    supplied = Species("Ar", gamma("Ar"), gamma3=1.2e-63)
    assert supplied.kerr is None
    assert supplied.kerr_coefficient() == supplied.gamma3
    # A wavelength changes nothing for a fixed coefficient.
    assert supplied.kerr_coefficient(266e-9) == supplied.gamma3
    assert "kerr" in repr(species("Ar", selection=SELECTION))


def test_a_species_carries_a_fixed_coefficient_or_a_model_not_both():
    with pytest.raises(MaterialError, match="one or the other"):
        Species("Ar", gamma("Ar"), gamma3=1e-51, kerr=OptmatKerr("Ar"))


def test_air_has_no_model_of_its_own_and_its_constituents_do():
    with pytest.raises(MaterialError, match="no hyperpolarizability model of Air"):
        OptmatKerr("Air")
    assert species("Air", selection=SELECTION).kerr is None
    air = Mixture.air(1.0, selection=SELECTION)
    assert all(c.species.kerr is not None for c in air.components)


@pytest.mark.parametrize("bad", [0.0, -800e-9, math.inf, math.nan])
def test_an_invalid_carrier_is_refused(bad):
    with pytest.raises(MaterialError, match="positive, finite carrier"):
        OptmatKerr("Ar").gamma3(bad)


def test_the_carrier_record_reports_the_thg_discrepancy_where_it_is_in_range():
    record = OptmatKerr("Ar").at_carrier(800e-9)
    model = optmat_model(Gas.Ar)
    assert isinstance(record, CarrierKerr)
    assert record.gamma3 == OptmatKerr("Ar").gamma3(800e-9)
    assert record.gamma3_thg == pytest.approx(
        float(model.gamma_si("thg", 800e-9) / (6 * EPS0)), rel=4e-16
    )
    # SEP 3, section 9.1: argon's THG/Kerr ratio at 800 nm is 1.177.
    assert record.thg_ratio == pytest.approx(1.177, abs=5e-4)
    assert record.thg_reason == ""
    assert record.to_dict()["model"] == "hyperpolarizability/Ar"


def test_a_thg_evaluation_beyond_the_fitted_data_reports_no_ratio():
    # N2O's third harmonic of 800 nm lies beyond its fitted dispersion (section 9.1).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        record = OptmatKerr("N2O").at_carrier(800e-9)
    assert record.thg_ratio is None
    assert "beyond the fitted data" in record.thg_reason
    assert record.to_dict()["thg_ratio"] is None


def test_a_carrier_outside_the_fit_warns_once_per_context():
    adapter = OptmatKerr("Ar")
    with _optmat.material_context(), pytest.warns(MaterialRangeWarning) as caught:
        adapter.gamma3(266e-9)
        adapter.gamma3(300e-9)
    ranges = [w for w in caught if issubclass(w.category, MaterialRangeWarning)]
    assert len(ranges) == 1
    assert "hyperpolarizability/Ar" in str(ranges[0].message)


def test_the_selected_carrier_is_one_wavelength():
    assert selected_carrier(SELECTION, 800e-9) == 800e-9
    carried = MaterialSelection(carrier_wavelength=1030e-9)
    assert selected_carrier(carried, None) == 1030e-9
    assert selected_carrier(carried, 1030e-9 * (1 + 1e-15)) == 1030e-9 * (1 + 1e-15)
    with pytest.raises(MaterialError, match="differ"):
        selected_carrier(carried, 800e-9)
    with pytest.raises(MaterialError, match="give wavelength="):
        selected_carrier(SELECTION, None)
    assert carriers_agree(800e-9, 800e-9 * (1 - 5e-13))
    assert not carriers_agree(800e-9, 800e-9 * (1 - 5e-12))


def test_chi3_under_a_selection_is_the_carrier_coefficient_times_optmat_density():
    argon = species("Ar", selection=SELECTION)
    density = argon.equation_of_state.density(3.0, ROOM_TEMPERATURE)
    value = chi3(
        "Ar",
        pressure=3.0,
        temperature=ROOM_TEMPERATURE,
        selection=SELECTION,
        wavelength=800e-9,
    )
    assert value == float(argon.kerr_coefficient(800e-9) or 0.0) * density
    # selection=None is MaterialSelection(): the same coefficient, and no carrier
    # default anywhere.
    assert chi3("Ar", pressure=3.0, wavelength=800e-9) == value
    with pytest.raises(MaterialError, match="carrier wavelength"):
        chi3("Ar", pressure=1.0, selection=SELECTION)
    with pytest.raises(MaterialError, match="carrier wavelength"):
        chi3("Ar", pressure=1.0)
    with pytest.raises(MaterialError, match="no Kerr coefficient for Air"):
        chi3("Air", pressure=1.0, selection=SELECTION, wavelength=800e-9)


def test_the_n2_chi3_round_trip_uses_the_exact_linear_index():
    pressure, carrier = 30.0, 800e-9
    argon = species("Ar", selection=SELECTION)
    density = argon.equation_of_state.density(pressure, ROOM_TEMPERATURE)
    omega = 2 * math.pi * C / carrier
    index = float(np.sqrt(1 + density * argon.susceptibility.susceptibility(omega)))
    value = n2("Ar", pressure=pressure, wavelength=carrier, selection=SELECTION)
    susceptibility = chi3(
        "Ar", pressure=pressure, selection=SELECTION, wavelength=carrier
    )
    # n2 = 3 chi3 / (4 eps0 c n0^2) = N gamma_T / (8 eps0^2 c n0^2), and back.
    assert value == pytest.approx(n2_from_chi3(susceptibility, index), rel=1e-15)
    assert chi3_from_n2(value, index) == pytest.approx(susceptibility, rel=1e-15)
    gamma_t = float(optmat_model(Gas.Ar).gamma_si("kerr", carrier))
    assert value == pytest.approx(
        density * gamma_t / (8 * EPS0**2 * C * index**2), rel=1e-14
    )
    # optmat's own dilute n2 takes n0 = 1; at 30 bar of argon that is 1.6 % high,
    # which is why it is never imported.
    assert (index**2 - 1) == pytest.approx(1.57e-2, rel=0.02)
    with pytest.raises(MaterialError, match="differ"):
        n2(
            "Ar",
            pressure=1.0,
            wavelength=800e-9,
            selection=MaterialSelection(carrier_wavelength=1030e-9),
        )


def test_a_single_tone_cubic_polarisation_has_the_three_quarter_fundamental():
    argon = species("Ar", selection=SELECTION)
    carrier = 800e-9
    density = argon.equation_of_state.density(1.0, ROOM_TEMPERATURE)
    coefficient = argon.kerr_coefficient(carrier)
    assert coefficient is not None
    n_to, cycles, amplitude = 512, 16, 1e10
    omega = 2 * math.pi * C / carrier
    t = np.arange(n_to) * (2 * math.pi * cycles / omega) / n_to
    field = (amplitude * np.cos(omega * t))[None, :]
    polarisation = KerrResponse(coefficient, thg=True).evaluate(
        field, density, kind="field"
    )[0]
    projection = 2 * np.mean(polarisation * np.cos(omega * t))
    harmonic = 2 * np.mean(polarisation * np.cos(3 * omega * t))
    chi = coefficient * density
    # cos^3 = (3 cos + cos 3x)/4, exactly on a window of whole cycles: roundoff only.
    assert projection == pytest.approx(0.75 * EPS0 * chi * amplitude**3, rel=1e-12)
    assert harmonic == pytest.approx(0.25 * EPS0 * chi * amplitude**3, rel=1e-12)
    # The fundamental's index change is n2 I with I = n0 eps0 c E0^2 / 2.
    index = float(np.sqrt(1 + density * argon.susceptibility.susceptibility(omega)))
    delta_chi = projection / (EPS0 * amplitude)
    intensity = index * EPS0 * C * amplitude**2 / 2
    assert delta_chi / (2 * index) == pytest.approx(
        n2_from_chi3(chi, index) * intensity, rel=1e-12
    )


def test_the_mixture_path_evaluates_each_population_at_the_carrier():
    fill = Mixture.fill({"He": 0.5, "Ar": 0.5}, 2.0, selection=SELECTION)
    path = fill.gamma3_path(800e-9)
    z = np.array([0.0])
    expected = math.fsum(
        float(c.species.kerr_coefficient(800e-9) or 0.0) * float(c.density(z)[0])
        for c in fill.components
    )
    assert float(path(z)[0]) == pytest.approx(expected, rel=1e-15)
    with pytest.raises(MaterialError, match="give the carrier wavelength"):
        fill.gamma3_path()
    # A caller's fixed coefficients (test data) need no carrier, and ignore one.
    supplied = Mixture.fill(
        {
            Species("He", gamma("He"), gamma3=1.0e-64): 0.5,
            Species("Ar", gamma("Ar"), gamma3=1.2e-63): 0.5,
        },
        2.0,
    )
    assert float(supplied.gamma3_path()(z)[0]) == float(
        supplied.gamma3_path(266e-9)(z)[0]
    )


def test_n2_under_a_selection_takes_its_carrier_and_has_no_silent_default():
    # The coefficient disperses, so n2 needs a carrier: the caller's or the
    # selection's, never a default.
    with pytest.raises(MaterialError):
        n2("Ar", pressure=1.0, selection=SELECTION)
    carrier = MaterialSelection(carrier_wavelength=1030e-9)
    assert n2("Ar", pressure=1.0, selection=carrier) == n2(
        "Ar", pressure=1.0, wavelength=1030e-9, selection=SELECTION
    )
    # Without a selection it is MaterialSelection(), which has no carrier either.
    with pytest.raises(MaterialError, match="carrier wavelength"):
        n2("Ar", pressure=1.0)
    assert n2("Ar", pressure=1.0, wavelength=800e-9) == n2(
        "Ar", pressure=1.0, wavelength=800e-9, selection=SELECTION
    )
