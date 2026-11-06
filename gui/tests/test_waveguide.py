# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Design a waveguide's kernel requests: curves and numbers with no pulse and no run."""

from typing import Any

import numpy as np
import pytest
from solprop import tools
from solprop.recipe import Recipe
from solprop.recipe.errors import RecipeError

from solprop_gui.service.handlers import HANDLERS
from solprop_gui.service.waveguide import (
    design_recipe,
    waveguide_curves,
    waveguide_numbers,
    waveguide_solve,
)

CAPILLARY: dict[str, Any] = {
    "guide": "capillary",
    "values": {
        "radius": "125 um",
        "gas": "He",
        "pressure": "1 bar",
        "wavelength": "800 nm",
    },
}
FIBRE: dict[str, Any] = {
    "guide": "antiresonant",
    "values": {
        "radius": "15 um",
        "gas": "Ar",
        "pressure": "1 bar",
        "wavelength": "800 nm",
        "lattice.wall_thickness": "180 nm",
    },
}


def test_a_design_is_a_recipe_of_one_mode() -> None:
    keywords = design_recipe(CAPILLARY).keywords()
    assert keywords["radius"] == pytest.approx(125e-6)
    assert keywords["pressure"] == pytest.approx(1.0)
    assert keywords["modes"] == 1
    assert design_recipe(FIBRE).keywords()["lattice"] is not None


def test_a_value_that_does_not_read_is_refused() -> None:
    values = {**CAPILLARY["values"], "radius": "not a length"}
    with pytest.raises(RecipeError):
        design_recipe({"guide": "capillary", "values": values})


def test_a_moved_pump_takes_the_window_with_it() -> None:
    moved = {**CAPILLARY["values"], "wavelength": "1600 nm"}
    reply: dict[str, Any] = waveguide_curves({"guide": "capillary", "values": moved})
    assert reply["x"][0] == pytest.approx(200.0)
    assert reply["x"][-1] == pytest.approx(3000.0)
    assert reply["pump"] == pytest.approx(1600.0)


@pytest.mark.parametrize("design", [CAPILLARY, FIBRE], ids=["capillary", "fibre"])
@pytest.mark.parametrize(
    ("quantity", "limited"),
    [
        ("beta2", True),
        ("dispersion", True),
        ("beta3", True),
        ("neff", False),
        ("group_index", False),
        ("loss", False),
        ("mismatch", True),
    ],
)
def test_every_quantity_has_a_curve(
    design: dict[str, Any], quantity: str, limited: bool
) -> None:
    reply: dict[str, Any] = waveguide_curves({**design, "quantity": quantity})
    assert reply["key"] == quantity
    assert (reply["limit"] is not None) == limited
    curve = reply["curves"][0]
    assert curve["value"] is None
    assert np.shape(curve["y"]) == np.shape(reply["x"])
    assert np.isfinite(curve["y"]).all()


def test_the_curve_is_the_modes_own() -> None:
    reply: dict[str, Any] = waveguide_curves(CAPILLARY)
    direct = tools.capillary_params(
        125e-6, "He", 1.0, energy=1e-6, fwhm=30e-15, wavelength=800e-9
    )
    assert reply["curves"][0]["zero"] == pytest.approx(direct.zdw * 1e9, rel=2e-3)
    at_pump = np.interp(800.0, reply["x"], reply["curves"][0]["y"])
    assert at_pump == pytest.approx(direct.beta2 * 1e30, rel=1e-2)


def test_a_sweep_has_a_curve_per_value_and_moves_the_zero() -> None:
    reply: dict[str, Any] = waveguide_curves(
        {**CAPILLARY, "sweep": {"name": "pressure", "values": ["0.5 bar", "2 bar"]}}
    )
    low, high = reply["curves"]
    assert (low["value"], high["value"]) == ("0.5 bar", "2 bar")
    assert low["zero"] < high["zero"]


def test_a_value_of_a_sweep_that_fails_says_so_and_keeps_the_rest() -> None:
    reply: dict[str, Any] = waveguide_curves(
        {**CAPILLARY, "sweep": {"name": "gas", "values": ["He", "NotAGas"]}}
    )
    good, bad = reply["curves"]
    assert "y" in good
    assert "error" in bad
    assert reply["limit"] is not None


def test_nothing_drawn_has_no_axis() -> None:
    reply: dict[str, Any] = waveguide_curves(
        {**CAPILLARY, "sweep": {"name": "gas", "values": ["NotAGas"]}}
    )
    assert reply["limit"] is None
    assert np.size(reply["x"]) == 0


def test_a_pulse_changes_the_phase_mismatch() -> None:
    bare: dict[str, Any] = waveguide_curves({**CAPILLARY, "quantity": "mismatch"})
    given: dict[str, Any] = waveguide_curves(
        {**CAPILLARY, "quantity": "mismatch", "pulse": True}
    )
    assert not np.allclose(bare["curves"][0]["y"], given["curves"][0]["y"])


def test_numbers_need_no_pulse() -> None:
    numbers: dict[str, Any] = waveguide_numbers({**CAPILLARY, "pulse": False})
    direct = tools.capillary_params(
        125e-6, "He", 1.0, energy=1e-6, fwhm=30e-15, wavelength=800e-9
    )
    assert numbers["beta2"] == pytest.approx(direct.beta2)
    assert numbers["gamma"] == pytest.approx(direct.gamma)
    assert numbers["aeff"] == pytest.approx(direct.effective_area)
    assert numbers["zdw"] == pytest.approx(direct.zdw)
    assert numbers["loss"] > 0
    assert "soliton_order" not in numbers
    assert Recipe.loads(numbers["resolved"]).keywords()["radius"] == pytest.approx(
        125e-6
    )


def test_a_pulse_adds_the_soliton_s_numbers() -> None:
    values = {**CAPILLARY["values"], "fwhm": "10 fs", "energy": "200 uJ"}
    numbers: dict[str, Any] = waveguide_numbers(
        {"guide": "capillary", "values": values, "pulse": True}
    )
    direct = tools.capillary_params(
        125e-6, "He", 1.0, energy=200e-6, fwhm=10e-15, wavelength=800e-9, shape="gauss"
    )
    assert numbers["soliton_order"] == pytest.approx(direct.soliton_order)
    assert numbers["fission_length"] == pytest.approx(direct.fission_length)
    assert numbers["dispersion_length"] == pytest.approx(direct.dispersion_length)
    assert numbers["rdw"] < 300e-9


def test_a_fibre_s_zero_is_on_its_curve() -> None:
    numbers: dict[str, Any] = waveguide_numbers({**FIBRE, "pulse": False})
    assert 460e-9 < numbers["zdw"] < 500e-9


def test_a_pressure_is_solved_for_a_zero() -> None:
    reply: dict[str, Any] = waveguide_solve({**CAPILLARY, "zdw": 450e-9})
    assert reply["pressure"] == pytest.approx(tools.pressure_zdw(125e-6, "He", 450e-9))
    assert "energy" not in reply
    held = {**CAPILLARY["values"], "pressure": f"{reply['pressure']} bar"}
    numbers: dict[str, Any] = waveguide_numbers({"guide": "capillary", "values": held})
    assert numbers["zdw"] == pytest.approx(450e-9, rel=1e-4)


def test_an_order_solves_the_energy_too() -> None:
    values = {**CAPILLARY["values"], "fwhm": "10 fs"}
    reply: dict[str, Any] = waveguide_solve(
        {"guide": "capillary", "values": values, "zdw": 450e-9, "soliton_order": 3.0}
    )
    held = {
        **values,
        "pressure": f"{reply['pressure']} bar",
        "energy": f"{reply['energy']} J",
    }
    numbers: dict[str, Any] = waveguide_numbers(
        {"guide": "capillary", "values": held, "pulse": True}
    )
    assert numbers["soliton_order"] == pytest.approx(3.0, rel=2e-2)


def test_the_requests_are_offered() -> None:
    for kind in ("waveguide_curves", "waveguide_numbers", "waveguide_solve"):
        assert kind in HANDLERS
