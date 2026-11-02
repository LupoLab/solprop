# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pins: recipe values set from a design target (SEP 41, S1 and S7)."""

import subprocess
import sys
import warnings

import numpy as np
import pytest

from solprop.modes import MarcatiliMode
from solprop.modes.base import beta2
from solprop.recipe import Recipe, RecipeError
from solprop.recipe.pins import solve_pin, solve_pulse
from solprop.tools import (
    ZDW,
    nonlinear_coefficient,
    pressure_rdw,
    pressure_zdw,
    pulse_peak_power,
    soliton_order,
)
from solprop.tools import _fill_n2 as fill_n2

ZDW_PIN = {
    "target": "zdw",
    "wavelength": "500 nm",
    "search": ["0.01 bar", "20 bar"],
}
ORDER_PIN = {"target": "soliton_order", "order": 4}


def pinned(pins, *, drop=(), **build):
    """Return the soliton self-compression example with pins in place of values."""
    document = Recipe.example("soliton_self_compression").to_dict()
    for name in drop:
        del document["build"][name]
    document["build"].update(build)
    document["pins"] = pins
    return Recipe.from_dict(document)


def carried(simulation):
    """Return the recipe a simulation carries: the resolved one, for a pinned recipe."""
    recipe = simulation._recipe
    assert isinstance(recipe, Recipe)
    return recipe


def quiet_build(recipe):
    """Build a recipe, ignoring the warnings its build gives by design."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return recipe.build()


def test_a_pinned_pressure_is_pressure_zdw_where_it_has_one_root():
    recipe = pinned({"pressure": ZDW_PIN}, drop=("pressure",))
    roots = recipe.pin_roots("pressure")
    assert len(roots) == 1
    expected = pressure_zdw(125e-6, "He", 500e-9)
    resolved = recipe.resolve()
    assert resolved.get("pressure") == pytest.approx(expected, rel=1e-7)
    assert "pins" not in resolved.to_dict()


def test_the_roots_of_a_path_that_is_not_pinned_are_refused():
    recipe = pinned({"pressure": ZDW_PIN}, drop=("pressure",))
    with pytest.raises(RecipeError, match="this path is not pinned") as caught:
        recipe.pin_roots("energy")
    assert caught.value.path == "energy"


def test_a_pin_beside_a_value_on_the_grid_is_solved_on_the_route_s_grid():
    # The absorber is built on the grid, so the pin's probe builds the route first;
    # it moves neither the guide's dispersion nor the pulse's order.
    absorber = {"type": "absorber", "boundary_N": 30}
    guide = pinned({"pressure": ZDW_PIN}, drop=("pressure",), absorber=absorber)
    assert guide.resolve().get("pressure") == pytest.approx(
        pressure_zdw(125e-6, "He", 500e-9), rel=1e-7
    )
    pulse = pinned({"energy": ORDER_PIN}, drop=("energy",), absorber=absorber)
    alone = pinned({"energy": ORDER_PIN}, drop=("energy",))
    assert pulse.resolve().get("energy") == pytest.approx(
        alone.resolve().get("energy"), rel=1e-12
    )


def test_a_soliton_order_pin_gives_the_order_on_the_launch_mode():
    recipe = pinned({"energy": ORDER_PIN}, drop=("energy",))
    simulation = quiet_build(recipe)
    energy = recipe.resolve().get("energy")
    assert isinstance(energy, float)
    mode = simulation.modes[0]
    assert isinstance(mode, MarcatiliMode)
    assert mode.mixture is not None
    gamma = nonlinear_coefficient(mode, fill_n2(mode.mixture, 800e-9)[0], 800e-9)
    dispersion = float(beta2(mode, np.array([2 * np.pi * 299_792_458.0 / 800e-9]))[0])
    power = pulse_peak_power(energy, 10e-15, "gauss")
    order = soliton_order(power, 10e-15, gamma, dispersion, "gauss")
    assert order == pytest.approx(4.0, rel=1e-9)


def test_the_pulse_pin_reads_the_guide_its_pin_settles():
    recipe = pinned(
        {"pressure": ZDW_PIN, "energy": ORDER_PIN}, drop=("pressure", "energy")
    )
    resolved = recipe.resolve()
    alone = pinned(
        {"energy": ORDER_PIN}, drop=("energy",), pressure=resolved.get("pressure")
    )
    assert resolved.get("energy") == pytest.approx(alone.resolve().get("energy"))
    assert recipe.pin_roots("energy").roots == pytest.approx((resolved.get("energy"),))


def test_a_run_records_the_authored_pins_and_the_resolved_values():
    recipe = pinned({"energy": ORDER_PIN}, drop=("energy",))
    simulation = quiet_build(recipe)
    simulation.translate()
    record = carried(simulation).record(simulation)
    assert "[pins.energy]" in record.authored
    assert "[pins" not in record.resolved
    assert "energy = " in record.resolved


@pytest.mark.parametrize(
    ("pins", "drop", "match"),
    [
        (
            {"pressure": ZDW_PIN, "radius": {**ZDW_PIN, "search": ["50 um", "200 um"]}},
            ("pressure", "radius"),
            "at most one parameter of the guide or fill",
        ),
        ({"pressure[0]": ZDW_PIN}, (), "an element of an array cannot be pinned"),
        ({"pressure": ZDW_PIN}, (), "cannot also be set in"),
        ({"length": ZDW_PIN}, ("length",), "on the capillary route is one of"),
        ({"pressure": {"target": "gdd"}}, ("pressure",), "target is one of"),
        (
            {"pressure": {"target": "zdw", "wavelength": "1 um"}},
            ("pressure",),
            "search",
        ),
        ({"pressure": {**ZDW_PIN, "colour": 1}}, ("pressure",), "has no key 'colour'"),
        ({"energy": {**ORDER_PIN, "root": "middle"}}, ("energy",), "root is"),
        ({"radius": ORDER_PIN}, ("radius",), "pulse's energy or peak_power"),
    ],
)
def test_a_pin_that_breaks_a_rule_is_refused_on_its_path(pins, drop, match):
    if any("[" in path for path in pins):
        recipe = pinned(pins, pressure=["0.8 bar", "0 bar"])
    else:
        recipe = pinned(pins, drop=drop)
    with pytest.raises(RecipeError, match=match) as caught:
        recipe.resolve()
    assert caught.value.path in pins


def test_an_unreachable_target_is_an_issue_with_the_attainable_range():
    pin = {**ZDW_PIN, "wavelength": "200 nm", "search": ["0.1 bar", "1 bar"]}
    # A constant cladding index: the attainable range is a search of its own over
    # the zero-dispersion wavelength, which a dispersive glass makes slow.
    recipe = pinned({"pressure": pin}, drop=("pressure",), cladding=1.45)
    roots = recipe.pin_roots("pressure")
    assert len(roots) == 0
    assert roots.attainable is not None
    issues = recipe.check()
    assert [issue.parameter for issue in issues] == ["pressure"]
    assert issues[0].severity == "error"
    with pytest.raises(RecipeError, match="pressure"):
        recipe.build()


def test_a_pinned_required_keyword_is_not_reported_missing():
    pin = {**ZDW_PIN, "search": ["50 um", "300 um"]}
    recipe = pinned({"radius": pin}, drop=("radius",))
    assert recipe.check() == []


def test_a_scan_re_solves_its_pins_at_each_point():
    recipe = pinned({"pressure": ZDW_PIN}, drop=("pressure",))
    document = recipe.to_dict()
    document["scan"] = {"name": "bore", "radius": ["100 um", "150 um"]}
    scan = Recipe.from_dict(document).scan()
    pressures = []
    for point in scan.scan.points():
        simulation = quiet_build_point(scan.build, point)
        pressures.append(carried(simulation).get("pressure"))
    expected = [pressure_zdw(radius, "He", 500e-9) for radius in (100e-6, 150e-6)]
    assert pressures == pytest.approx(expected, rel=1e-7)


def quiet_build_point(build, point):
    """Build one scan point, ignoring the warnings its build gives by design."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build(point)


_PROBE = """\
import sys
import warnings

from solprop.driver.simulation import Simulation

warnings.simplefilter("ignore")


def run(self, *args, **kwargs):
    print(self.metadata().configuration_hash())
    sys.exit(0)


Simulation.run = run
"""


def test_the_export_solves_the_pins_before_the_route_s_call(tmp_path):
    recipe = pinned(
        {"pressure": ZDW_PIN, "energy": ORDER_PIN}, drop=("pressure", "energy")
    )
    script = recipe.to_python()
    assert script.index("solve_pin(") < script.index("prop_capillary(")
    assert script.index("solve_pulse(") < script.index("prop_capillary(")
    path = tmp_path / "pinned.py"
    path.write_text(_PROBE + script)
    # A subprocess, as a person runs the script: without the tests' type hook.
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = quiet_build(recipe).metadata().configuration_hash()
    assert result.stdout.splitlines()[-1] == expected


@pytest.mark.parametrize(
    ("pin", "match"),
    [
        ("zdw", "a pin is a table"),
        ({"target": "zdw", "search": ["0.1 bar", "1 bar"]}, "needs 'wavelength'"),
        ({**ZDW_PIN, "wavelength": True}, "expected a number"),
        ({**ZDW_PIN, "wavelength": [1]}, "expected a number or a quantity"),
        ({**ZDW_PIN, "wavelength": float("inf")}, "finite"),
        ({**ZDW_PIN, "search": ["1 bar"]}, r"search is \[low, high\]"),
        ({**ZDW_PIN, "search": ["2 bar", "1 bar"]}, "low < high"),
        ({**ZDW_PIN, "root": True}, "root is"),
        (
            {"target": "rdw", "rdw": "200 nm", "fwhm": "10 fs", "search": [1, 2]},
            "fwhm goes with its peak_power",
        ),
        (
            {"target": "rdw", "rdw": "200 nm", "shape": 1, "search": [1, 2]},
            "is a string",
        ),
    ],
)
def test_a_malformed_guide_pin_is_refused_on_its_path(pin, match):
    recipe = pinned({"pressure": pin}, drop=("pressure",))
    with pytest.raises(RecipeError, match=match) as caught:
        recipe.resolve()
    assert caught.value.path == "pressure"


def test_an_order_is_a_plain_number():
    recipe = pinned({"energy": {**ORDER_PIN, "order": "four"}}, drop=("energy",))
    with pytest.raises(RecipeError, match="expected a number"):
        recipe.resolve()


def test_an_rdw_pin_reproduces_pressure_rdw():
    pin = {"target": "rdw", "rdw": "200 nm", "search": ["0.01 bar", "20 bar"]}
    recipe = pinned({"pressure": pin}, drop=("pressure",))
    expected = pressure_rdw(125e-6, "He", 200e-9, 800e-9)
    assert recipe.resolve().get("pressure") == pytest.approx(expected, rel=1e-6)


def test_a_root_index_chooses_among_the_roots():
    recipe = pinned({"pressure": {**ZDW_PIN, "root": 0}}, drop=("pressure",))
    assert recipe.resolve().get("pressure") == pytest.approx(
        pressure_zdw(125e-6, "He", 500e-9), rel=1e-7
    )


def test_a_soliton_order_in_normal_dispersion_is_refused():
    recipe = pinned({"energy": ORDER_PIN}, drop=("energy",), pressure="20 bar")
    with pytest.raises(RecipeError, match="normal"):
        recipe.resolve()


def test_a_sech_pulse_s_peak_power_is_pinned():
    recipe = pinned(
        {"peak_power": ORDER_PIN}, drop=("energy",), shape="sech", width="6 fs"
    )
    del_fwhm = recipe.to_dict()
    del del_fwhm["build"]["fwhm"]
    recipe = Recipe.from_dict(del_fwhm)
    sech = recipe.resolve()
    gauss = pinned({"peak_power": ORDER_PIN}, drop=("energy",)).resolve()
    assert isinstance(sech.get("peak_power"), float)
    assert sech.get("peak_power") != pytest.approx(gauss.get("peak_power"))


def test_a_pulse_in_the_list_is_pinned():
    document = Recipe.example("soliton_self_compression").to_dict()
    build = document["build"]
    for name in ("energy", "fwhm"):
        del build[name]
    build["pulses"] = [
        {"type": "GaussianPulse", "wavelength": "800 nm", "fwhm": "10 fs"}
    ]
    document["pins"] = {"pulses[0].energy": ORDER_PIN}
    recipe = Recipe.from_dict(document)
    alone = pinned({"energy": ORDER_PIN}, drop=("energy",)).resolve().get("energy")
    assert recipe.resolve().get("pulses[0].energy") == pytest.approx(alone)
    assert recipe.check() == []
    assert "solved when exported" in recipe.to_python()


def test_a_pin_on_a_pulse_the_list_does_not_hold_is_named_by_the_pin():
    document = Recipe.example("soliton_self_compression").to_dict()
    build = document["build"]
    for name in ("energy", "fwhm"):
        del build[name]
    build["pulses"] = [
        {
            "type": "GaussianPulse",
            "wavelength": "800 nm",
            "fwhm": "10 fs",
            "energy": 1e-6,
        }
    ]
    document["pins"] = {"pulses[3].energy": ORDER_PIN}
    recipe = Recipe.from_dict(document)
    (issue,) = recipe.check()
    assert issue.parameter == "pulses[3]"
    assert "pulses has 1 elements, so no [3]" in issue.message


def test_a_pulse_the_order_is_not_defined_for_is_refused():
    with pytest.raises(RecipeError, match="sech or Gaussian"):
        solve_pulse("capillary", {"pulses": [object()]}, "pulses[0].energy", 4.0)
    document = Recipe.example("soliton_self_compression").to_dict()
    build = document["build"]
    for name in ("energy", "fwhm"):
        del build[name]
    build["pulses"] = [
        {"type": "GaussianPulse", "wavelength": "800 nm", "fwhm": "10 fs", "m": 2}
    ]
    document["pins"] = {"pulses[0].energy": ORDER_PIN}
    with pytest.raises(RecipeError, match="super-Gaussian") as caught:
        Recipe.from_dict(document).resolve()
    assert caught.value.path == "pulses[0].energy"


def test_a_route_that_does_not_build_is_refused_on_the_pin():
    recipe = pinned({"pressure": ZDW_PIN}, drop=("pressure", "trange"))
    with pytest.raises(RecipeError, match="does not build while the pin is solved"):
        recipe.resolve()
    with pytest.raises(RecipeError, match="cannot pin"):
        solve_pin("gnlse", {}, "pressure", ZDW(500e-9), (0.1, 1.0))


def test_an_rdw_pin_with_its_pulse_exports_its_solver_call(tmp_path):
    pin = {
        "target": "rdw",
        "rdw": "200 nm",
        "peak_power": "1 GW",
        "fwhm": "10 fs",
        "shape": "gauss",
        "search": ["0.01 bar", "20 bar"],
    }
    recipe = pinned({"pressure": pin}, drop=("pressure",))
    expected = pressure_rdw(
        125e-6, "He", 200e-9, 800e-9, peak_power=1e9, fwhm=10e-15, shape="gauss"
    )
    assert recipe.resolve().get("pressure") == pytest.approx(expected, rel=1e-6)
    script = recipe.to_python()
    assert "RDW(keywords['wavelength'], 2e-07, peak_power=1000000000.0" in script
    path = tmp_path / "rdw.py"
    path.write_text(_PROBE + script)
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
