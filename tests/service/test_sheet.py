# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The live sheet's kernel requests: solprop's numbers, without building a run."""

import os
from typing import Any

import numpy as np
import pytest

from service.cases import TINY
from solprop import tools
from solprop.recipe import Recipe
from solprop.service.handlers import HANDLERS
from solprop.service.sheet import (
    CURVE_POINTS,
    _curve_range,
    _shapes_of,
    beam_curve,
    beam_sheet,
    capillary_curve,
    capillary_sheet,
    design_numbers,
    fibre_curve,
    fibre_sheet,
    gases,
    lens_beam,
    output_size,
)


def curve_of(numbers: dict[str, Any]) -> dict[str, Any]:
    return capillary_curve({"text": numbers["resolved"], "numbers": numbers})


def test_the_numbers_are_solprop_s_at_the_entrance() -> None:
    numbers: dict[str, Any] = capillary_sheet({"text": Recipe.example("rdw").to_text()})
    # The example's gradient starts at 1.2 bar.
    direct = tools.capillary_params(
        125e-6, "He", 1.2, energy=275e-6, fwhm=7.5e-15, wavelength=800e-9, shape="gauss"
    )
    assert numbers["pressure"] == 1.2
    assert numbers["soliton_order"] == pytest.approx(direct.soliton_order)
    assert numbers["zdw"] == pytest.approx(direct.zdw)
    assert numbers["pinned"] == {}


def test_the_curve_has_its_zero_where_the_numbers_say() -> None:
    numbers: dict[str, Any] = capillary_sheet({"text": Recipe.example("rdw").to_text()})
    curve = curve_of(numbers)
    plot = curve["plot"]
    wavelength, beta2 = np.asarray(plot["x"]), np.asarray(plot["y"])
    assert wavelength.size == beta2.size == CURVE_POINTS
    # 100 nm to 1500 nm, from the recipe's window; sign change at the ZDW.
    assert wavelength[0] == pytest.approx(100.0)
    assert wavelength[-1] == pytest.approx(1500.0)
    near = np.argmin(np.abs(wavelength - float(numbers["zdw"]) * 1e9))
    assert beta2[near - 2] * beta2[near + 2] < 0
    assert 100e-9 < float(curve["numbers"]["rdw"]) < float(numbers["zdw"])
    assert float(curve["numbers"]["loss"]) > 0
    assert dict(map(tuple, plot["marks"]))["Pump"] == pytest.approx(800.0)


def test_a_target_is_solved_and_the_curve_reads_the_solved_recipe() -> None:
    text = TINY.replace('energy = "100 uJ"\n', "")
    text += '\n[pins]\nenergy = { target = "soliton_order", order = 3 }\n'
    numbers: dict[str, Any] = capillary_sheet({"text": text})
    assert numbers["soliton_order"] == pytest.approx(3.0)
    assert numbers["pinned"]["energy"] == pytest.approx(numbers["energy"])
    assert "[pins]" not in str(numbers["resolved"])
    assert curve_of(numbers)["plot"]["y"] is not None


def test_a_peak_power_gives_the_energy_through_the_preview() -> None:
    text = TINY.replace('energy = "100 uJ"', 'peak_power = "5 GW"')
    numbers: dict[str, Any] = capillary_sheet({"text": text})
    assert numbers["peak_power"] == pytest.approx(5e9, rel=1e-3)


def test_a_pump_in_normal_dispersion_has_no_wave() -> None:
    text = TINY.replace('pressure = "1 bar"', 'pressure = "20 bar"').replace(
        'gas = "He"', 'gas = "Ar"'
    )
    numbers: dict[str, Any] = capillary_sheet({"text": text})
    assert float(numbers["beta2"]) > 0
    assert curve_of(numbers)["numbers"]["rdw"] is None
    assert "rdw" not in design_numbers(Recipe.loads(text), 100e-6)


def test_a_window_absent_from_the_recipe_is_around_the_pump() -> None:
    text = TINY.replace('wavelength_limits = ["200 nm", "2000 nm"]\n', "").replace(
        'trange = "300 fs"\n',
        'grid = { type = "FieldGrid.from_limits", length = "5 cm", '
        'reference_wavelength = "800 nm", '
        'wavelength_limits = ["200 nm", "2000 nm"], trange = "300 fs" }\n',
    )
    numbers: dict[str, Any] = capillary_sheet({"text": text})
    wavelength = np.asarray(curve_of(numbers)["plot"]["x"])
    assert wavelength[0] == pytest.approx(0.15 * 800.0)
    assert wavelength[-1] == pytest.approx(2.0 * 800.0)


def test_an_anomalous_pump_with_no_phase_match_has_no_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("solprop.service.sheet.lambda_rdw_all", lambda *_a, **_k: ())
    assert "rdw" not in design_numbers(Recipe.loads(TINY), 100e-6)


def test_a_design_reports_its_wave_where_there_is_one() -> None:
    assert float(design_numbers(Recipe.loads(TINY), 100e-6)["rdw"]) > 0


def test_a_filament_s_numbers_are_the_beam_s() -> None:
    numbers: dict[str, Any] = beam_sheet({"text": Recipe.example("filament").to_text()})
    assert numbers["power_ratio"] == pytest.approx(3.0, rel=0.01)
    assert numbers["waist"] == pytest.approx(300e-6)
    assert numbers["focus"] == 0.0
    assert numbers["rayleigh_range"] == pytest.approx(0.354, rel=0.01)
    assert 0.1 < numbers["collapse_distance"] < 0.2
    curve: dict[str, Any] = beam_curve({"numbers": numbers})
    radius = np.asarray(curve["plot"]["y"])
    assert radius[0] == pytest.approx(300.0)
    assert radius[-1] > radius[0]
    assert [mark[0] for mark in curve["plot"]["marks"]] == ["Collapse (Marburger)"]


def test_a_filament_with_a_focus_marks_it() -> None:
    text = (
        Recipe.example("filament")
        .to_text()
        .replace('waist = "300 um"', 'waist = "300 um"\npropz = "-10 cm"')
    )
    numbers: dict[str, Any] = beam_sheet({"text": text})
    assert numbers["focus"] == pytest.approx(0.1)
    curve: dict[str, Any] = beam_curve({"numbers": numbers})
    marks = dict(map(tuple, curve["plot"]["marks"]))
    assert marks["Focus"] == pytest.approx(10.0)


def test_a_medium_with_no_numbers_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise ValueError("no Kerr data")

    monkeypatch.setattr("solprop.service.sheet.beam_params", refuse)
    numbers: dict[str, Any] = beam_sheet({"text": Recipe.example("filament").to_text()})
    assert numbers["refusal"] == "no Kerr data"
    assert numbers["peak_power"] > 0
    assert "power_ratio" not in numbers
    assert beam_curve({"numbers": numbers})["plot"] is None


def test_a_lens_gives_the_textbook_waist() -> None:
    reply = lens_beam(
        {
            "text": Recipe.example("filament").to_text(),
            "focal_length": 1.0,
            "radius": 5e-3,
        }
    )
    assert reply["waist"] == pytest.approx(800e-9 / (np.pi * 5e-3), rel=1e-3)
    assert reply["distance"] == pytest.approx(1.0, rel=1e-3)


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


def fibre() -> dict[str, Any]:
    return capillary_sheet({"text": Recipe.example("antiresonant_fibre").to_text()})


def test_a_fibre_s_numbers_come_from_its_own_mode() -> None:
    numbers = fibre()
    recipe = Recipe.example("antiresonant_fibre")
    direct = tools.soliton_params(
        recipe.build().modes[0],
        "Ar",
        5.0,
        energy=1e-6,
        fwhm=30e-15,
        wavelength=800e-9,
        shape="gauss",
    )
    assert numbers["soliton_order"] == pytest.approx(direct.soliton_order)
    assert numbers["peak_power"] == pytest.approx(direct.peak_power)


def test_a_fibre_s_zero_is_the_one_on_its_curve() -> None:
    numbers = fibre()
    # At 5 bar the zero sits at 541 nm, between the wall resonance and the pump.
    assert 520e-9 < numbers["zdw"] < 560e-9
    curve = curve_of(numbers)
    plot = curve["plot"]
    marks = dict(plot["marks"])
    assert marks["ZDW"] == pytest.approx(numbers["zdw"] * 1e9)
    # The axis reads the pump's neighbourhood, not the resonance's swing.
    assert plot["limit"][1] < float(np.abs(plot["y"]).max())


def test_a_fibre_s_curve_has_its_loss_and_cross_section() -> None:
    curve = curve_of(fibre())
    assert np.shape(curve["loss"]["y"]) == (CURVE_POINTS,)
    assert curve["loss"]["pump"] == pytest.approx(800.0)
    kinds = [shape["kind"] for shape in curve["shapes"]]
    assert kinds.count("disc") + kinds.count("ring") == len(kinds)
    assert all(shape["fill"].startswith("#") for shape in curve["shapes"])
    assert kinds.count("ring") >= 1


def test_a_capillary_has_no_cross_section_beside_its_curve() -> None:
    numbers = capillary_sheet({"text": Recipe.example("rdw").to_text()})
    curve = curve_of(numbers)
    assert "loss" not in curve
    assert "shapes" not in curve


def test_a_fibre_with_nothing_crossing_zero_says_so() -> None:
    text = Recipe.example("antiresonant_fibre").to_text()
    narrow = text.replace('["150 nm", "1600 nm"]', '["700 nm", "900 nm"]')
    numbers: dict[str, Any] = capillary_sheet({"text": narrow})
    assert np.isnan(numbers["zdw"])
    assert dict(curve_of(numbers)["plot"]["marks"])["ZDW"] is None


def test_a_box_cross_section_is_sent_as_a_box() -> None:
    mode = Recipe.example("hollow_rectangular").build().modes[0]
    kinds = {shape["kind"] for shape in _shapes_of(mode)}
    assert "box" in kinds


def test_a_solid_core_fibre_s_numbers_are_the_tools() -> None:
    numbers: dict[str, Any] = fibre_sheet({"text": Recipe.example("soliton").to_text()})
    assert numbers["soliton_order"] == pytest.approx(3.0)
    assert numbers["peak_power"] == pytest.approx(96.7909090909)
    assert numbers["beta2"] == pytest.approx(-1.183e-26)
    assert numbers["energy"] == pytest.approx(19.36e-12, rel=0.01)
    assert numbers["dispersion_length"] == pytest.approx(0.8453, rel=1e-3)
    assert numbers["nonlinear_length"] == pytest.approx(1.0 / (0.11 * 96.7909))
    assert numbers["fission_length"] > 0
    assert numbers["pinned"] == {}


def test_normal_dispersion_has_no_soliton() -> None:
    text = (
        Recipe.example("soliton").to_text().replace("-0.01183 ps^2/m", "0.01183 ps^2/m")
    )
    numbers: dict[str, Any] = fibre_sheet({"text": text})
    assert "soliton_order" not in numbers
    assert "fission_length" not in numbers
    assert numbers["dispersion_length"] > 0


def test_no_dispersion_has_no_lengths_to_give() -> None:
    text = Recipe.example("soliton").to_text().replace("-0.01183 ps^2/m", "0 ps^2/m")
    numbers: dict[str, Any] = fibre_sheet({"text": text})
    assert numbers["peak_power"] == pytest.approx(96.7909090909)
    assert "dispersion_length" not in numbers


def test_a_given_full_width_is_taken_as_it_stands() -> None:
    text = Recipe.example("soliton").to_text()
    fwhm = 100e-15 / tools.pulse_width(1.0, "sech")
    given = text.replace('width = "100 fs"', f"fwhm = {fwhm}")
    numbers: dict[str, Any] = fibre_sheet({"text": given})
    assert numbers["dispersion_length"] == pytest.approx(0.8453, rel=1e-3)


def test_a_solid_core_curve_finds_the_zero_nearest_the_pump() -> None:
    text = Recipe.example("soliton").to_text()
    numbers: dict[str, Any] = fibre_sheet({"text": text})
    curve: dict[str, Any] = fibre_curve({"text": text, "numbers": numbers})
    plot = curve["plot"]
    assert plot["x"][0] == pytest.approx(500.0)
    assert plot["x"][-1] == pytest.approx(2000.0)
    assert curve["numbers"]["zdw"] is None
    assert dict(plot["marks"])["ZDW"] is None
    # A third-order term crosses zero at -beta2/beta3 from the pump.
    tilted = text.replace('"-0.01183 ps^2/m"]', '"-0.01183 ps^2/m", "1e-4 ps^3/m"]')
    numbers = fibre_sheet({"text": tilted})
    curve = fibre_curve({"text": tilted, "numbers": numbers})
    assert curve["numbers"]["zdw"] is not None
    assert curve["numbers"]["zdw"] < 835e-9
    assert dict(curve["plot"]["marks"])["ZDW"] == pytest.approx(
        curve["numbers"]["zdw"] * 1e9
    )


def test_a_curve_with_no_window_reaches_around_the_pump() -> None:
    assert _curve_range(_NoWindow(), 800e-9) == pytest.approx((120e-9, 1600e-9))  # pyright: ignore[reportArgumentType]


class _NoWindow:
    def keywords(self) -> dict[str, object]:
        return {}


def test_the_run_options_are_solprop_s() -> None:
    from solprop.backends import available  # noqa: PLC0415 - only here

    options: Any = HANDLERS["run_options"]({})
    # A kernel run in this process may have set its own budget: a run's is at least it.
    assert options["threads"] >= 1
    assert options["double"] == list(available("double"))
    assert "native" in options["double"] or "numpy_ref" in options["double"]
    assert isinstance(options["metal"], bool)


def test_the_threads_offered_are_a_run_s_not_the_kernel_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solprop.service.kernel import RUN_THREADS  # noqa: PLC0415 - only here

    monkeypatch.setenv("SOLPROP_THREADS", "1")
    monkeypatch.setenv(RUN_THREADS, "")
    unset: Any = HANDLERS["run_options"]({})
    assert unset["limit"] != "SOLPROP_THREADS"
    monkeypatch.setenv(RUN_THREADS, "1")
    one: Any = HANDLERS["run_options"]({})
    assert one["threads"] == 1
    # The kernel's own budget is left as it was.
    assert os.environ["SOLPROP_THREADS"] == "1"
    monkeypatch.delenv("SOLPROP_THREADS")
    monkeypatch.setenv(RUN_THREADS, "")
    HANDLERS["run_options"]({})
    assert "SOLPROP_THREADS" not in os.environ


def test_outside_a_kernel_the_threads_are_this_process_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solprop.backends import thread_budget  # noqa: PLC0415 - only here
    from solprop.service.kernel import RUN_THREADS  # noqa: PLC0415 - only here

    monkeypatch.delenv(RUN_THREADS, raising=False)
    options: Any = HANDLERS["run_options"]({})
    assert options["threads"] == thread_budget().threads


def test_a_graded_fibre_of_a_glass_has_no_aperture_to_give() -> None:
    from solprop.service.sheet import grin_curve, grin_sheet  # noqa: PLC0415

    text = (
        Recipe.example("graded_index_fibre")
        .to_text()
        .replace(
            'centre = { type = "IndexMaterial", index = 1.45 }',
            'centre = { type = "IndexMaterial", index = '
            '{ type = "Glass", name = "SiO2" } }',
        )
    )
    numbers: Any = grin_sheet({"text": text})
    assert "na" not in numbers
    assert numbers["peak_power"] > 0
    assert grin_curve({"text": text}) == {"numbers": {}}


def test_a_graded_fibre_s_aperture_and_modes() -> None:
    from solprop.service.sheet import grin_curve, grin_sheet  # noqa: PLC0415

    text = Recipe.example("graded_index_fibre").to_text()
    numbers: Any = grin_sheet({"text": text})
    aperture = (1.45**2 - 1.442731783804599**2) ** 0.5
    assert numbers["na"] == pytest.approx(aperture)
    v_number = 2 * np.pi * 25e-6 * aperture / 1.03e-6
    assert numbers["v_number"] == pytest.approx(v_number)
    assert numbers["mode_count"] == pytest.approx(v_number**2 / 4)
    curve: Any = grin_curve({"text": text})
    assert curve["plot"]["y"][0] == pytest.approx(1.45)
    assert curve["plot"]["y"][-1] == pytest.approx(1.442731783804599)


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_fibre_with_no_coefficient_gives_what_it_can(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solprop.service import sheet  # noqa: PLC0415 - only here

    monkeypatch.setattr(sheet, "nonlinear_coefficient", lambda *_args: None)
    numbers: Any = fibre_sheet({"text": Recipe.example("step_index_fibre").to_text()})
    assert "gamma" not in numbers
    assert numbers["peak_power"] > 0


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_solved_fibre_s_coefficient_and_cross_section() -> None:
    text = Recipe.example("step_index_fibre").to_text()
    numbers: Any = fibre_sheet({"text": text})
    assert numbers["gamma"] == pytest.approx(0.0721, rel=1e-2)
    curve: Any = fibre_curve({"text": text, "numbers": numbers})
    assert [shape["kind"] for shape in curve["shapes"]] == ["disc", "disc"]
