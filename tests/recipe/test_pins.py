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
from solprop.tools import _fill_n2 as fill_n2
from solprop.tools import (
    nonlinear_coefficient,
    pressure_zdw,
    pulse_peak_power,
    soliton_order,
)

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
