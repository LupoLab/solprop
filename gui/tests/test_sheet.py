# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The live sheet's kernel requests: solprop's numbers, without building a run."""

import numpy as np
import pytest
from solprop import tools
from solprop.recipe import Recipe

from conftest import TINY
from solprop_gui.service.handlers import HANDLERS
from solprop_gui.service.sheet import (
    CURVE_POINTS,
    capillary_curve,
    capillary_sheet,
    design_numbers,
    gases,
    output_size,
)

CURVE_KEYS = ("beta2", "peak_power", "gamma", "fwhm", "shape", "wavelength")


def curve_of(numbers: dict[str, object]) -> dict[str, object]:
    payload = {key: numbers[key] for key in CURVE_KEYS}
    return capillary_curve({"text": numbers["resolved"], **payload})


def test_the_numbers_are_solprop_s_at_the_entrance() -> None:
    numbers = capillary_sheet({"text": Recipe.example("rdw").to_text()})
    # The example's gradient starts at 1.2 bar.
    direct = tools.capillary_params(
        125e-6, "He", 1.2, energy=275e-6, fwhm=7.5e-15, wavelength=800e-9, shape="gauss"
    )
    assert numbers["pressure"] == 1.2
    assert numbers["soliton_order"] == pytest.approx(direct.soliton_order)
    assert numbers["zdw"] == pytest.approx(direct.zdw)
    assert numbers["pinned"] == {}


def test_the_curve_has_its_zero_where_the_numbers_say() -> None:
    numbers = capillary_sheet({"text": Recipe.example("rdw").to_text()})
    curve = curve_of(numbers)
    wavelength, beta2 = np.asarray(curve["wavelength"]), np.asarray(curve["beta2"])
    assert wavelength.size == beta2.size == CURVE_POINTS
    # 100 nm to 1500 nm, from the recipe's window; sign change at the ZDW.
    assert wavelength[0] == pytest.approx(100e-9)
    assert wavelength[-1] == pytest.approx(1500e-9)
    near = np.argmin(np.abs(wavelength - float(numbers["zdw"])))
    assert beta2[near - 2] * beta2[near + 2] < 0
    assert 100e-9 < float(curve["rdw"]) < float(numbers["zdw"])
    assert float(curve["loss"]) > 0


def test_a_target_is_solved_and_the_curve_reads_the_solved_recipe() -> None:
    text = TINY.replace('energy = "100 uJ"\n', "")
    text += '\n[pins]\nenergy = { target = "soliton_order", order = 3 }\n'
    numbers = capillary_sheet({"text": text})
    assert numbers["soliton_order"] == pytest.approx(3.0)
    assert numbers["pinned"]["energy"] == pytest.approx(numbers["energy"])
    assert "[pins]" not in str(numbers["resolved"])
    assert curve_of(numbers)["beta2"] is not None


def test_a_peak_power_gives_the_energy_through_the_preview() -> None:
    text = TINY.replace('energy = "100 uJ"', 'peak_power = "5 GW"')
    numbers = capillary_sheet({"text": text})
    assert numbers["peak_power"] == pytest.approx(5e9, rel=1e-3)


def test_a_pump_in_normal_dispersion_has_no_wave() -> None:
    text = TINY.replace('pressure = "1 bar"', 'pressure = "20 bar"').replace(
        'gas = "He"', 'gas = "Ar"'
    )
    numbers = capillary_sheet({"text": text})
    assert float(numbers["beta2"]) > 0
    assert curve_of(numbers)["rdw"] is None
    assert "rdw" not in design_numbers(Recipe.loads(text), 100e-6)


def test_a_window_absent_from_the_recipe_is_around_the_pump() -> None:
    text = TINY.replace('wavelength_limits = ["200 nm", "2000 nm"]\n', "").replace(
        'trange = "300 fs"\n',
        'grid = { type = "FieldGrid.from_limits", length = "5 cm", '
        'wavelength_limits = ["200 nm", "2000 nm"], trange = "300 fs" }\n',
    )
    numbers = capillary_sheet({"text": text})
    wavelength = np.asarray(curve_of(numbers)["wavelength"])
    assert wavelength[0] == pytest.approx(0.15 * 800e-9)
    assert wavelength[-1] == pytest.approx(2.0 * 800e-9)


def test_an_anomalous_pump_with_no_phase_match_has_no_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "solprop_gui.service.sheet.lambda_rdw_all", lambda *_a, **_k: ()
    )
    assert "rdw" not in design_numbers(Recipe.loads(TINY), 100e-6)


def test_a_design_reports_its_wave_where_there_is_one() -> None:
    assert float(design_numbers(Recipe.loads(TINY), 100e-6)["rdw"]) > 0


def test_the_size_is_solprop_s() -> None:
    recipe = Recipe.loads(TINY)
    size = output_size({"text": TINY, "base": None})["output_bytes"]
    assert size == recipe.build().output_bytes()
    assert output_size({"text": TINY, "base": "."})["output_bytes"] == size


def test_the_gases_are_solprop_s_and_the_handlers_are_offered() -> None:
    names = gases({})
    assert {"He", "Ar", "N2"} <= set(names)
    assert {"capillary_sheet", "capillary_curve", "output_size", "gases"} <= set(
        HANDLERS
    )
