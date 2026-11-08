# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel's design requests: modes, sweeps and pins (no Qt), and a multimode run."""

import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from service.cases import MULTIMODE, TINY
from service.test_service import Collector
from solprop.exceptions import SolpropError
from solprop.recipe import Recipe
from solprop.service import handlers
from solprop.service.worker import RunSpec

PINNED = TINY.replace('pressure = "1 bar"\n', "") + (
    '\n[pins]\npressure = { target = "zdw", wavelength = "500 nm", '
    'search = ["0.1 bar", "10 bar"] }\n'
)
"""The tiny capillary with its pressure pinned to a zero-dispersion wavelength."""


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    """Ignore the material-range warnings of the wide bands."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def ask(kind: str, payload: Mapping[str, object]) -> Any:
    return handlers.HANDLERS[kind](payload)


def test_a_sweep_gives_one_mode_set_per_value_and_says_which_fail():
    payload = {
        "text": TINY,
        "sweep": {"name": "pressure", "values": ["0.5 bar", "2 bar", "x"]},
        "z": 0.0,
    }
    reply = ask("modes", payload)
    assert reply["sweep"] == "pressure"
    low, high, bad = reply["sets"]
    assert low["modes"][0]["zdw"] < high["modes"][0]["zdw"]  # more gas, redder ZDW
    assert "not a quantity" in bad["error"]
    wrong = ask(
        "modes", {"text": TINY, "sweep": {"name": "radius", "values": ["0 um"]}}
    )
    assert "error" in wrong["sets"][0]


def test_a_sweep_over_a_pinned_keyword_replaces_the_pin():
    pinned = PINNED + '\n[pins.energy]\ntarget = "soliton_order"\norder = 2\n'
    pinned = pinned.replace('energy = "100 uJ"\n', "")
    reply = ask("modes", {"text": pinned, "sweep": {"name": "pressure", "values": [1]}})
    assert "modes" in reply["sets"][0]
    alone = ask("modes", {"text": PINNED, "sweep": {"name": "pressure", "values": [1]}})
    assert "modes" in alone["sets"][0]


def test_a_route_without_modes_or_a_pump_says_so():
    beam = Recipe.example("cartesian_beam").to_text()
    reply = ask("modes", {"text": beam})
    assert "builds no mode set" in reply["sets"][0]["error"]
    no_pump = ask("modes", {"text": TINY.replace('wavelength = "800 nm"\n', "")})
    assert no_pump["pump"] is None and "wavelength" in no_pump["sets"][0]["error"]
    unsolvable = PINNED.replace('"500 nm"', '"50 nm"')
    assert ask("modes", {"text": unsolvable})["pump"] is None


def test_pins_are_described_and_their_roots_found():
    assert set(ask("pin_targets", {"route": "capillary"})["pressure"][0]) == {
        "target",
        "stage",
        "required",
    }
    description = ask("describe", {"route": "capillary"})
    by_name = {entry["name"]: entry for entry in description["parameters"]}
    assert [pin["target"] for pin in by_name["energy"]["pins"]] == ["soliton_order"]
    assert by_name["gas"]["pins"] == []
    found = ask("pin_roots", {"text": PINNED, "path": "pressure"})
    assert len(found["roots"]) >= 1 and found["attainable"] is None
    none = ask(
        "pin_roots", {"text": PINNED.replace('"500 nm"', '"50 nm"'), "path": "pressure"}
    )
    assert none["roots"] == [] and "wavelength" in none["quantity"]
    summary = ask("summarise", {"text": PINNED})
    assert summary["pinned"]["pressure"] == pytest.approx(found["roots"][0])


def test_a_multimode_run_with_a_launch_streams_rows(tmp_path: Path):
    spec = RunSpec("m1", str(tmp_path / "m1"), recipe=MULTIMODE, base=str(tmp_path))
    collector = Collector(spec).start()
    collector.collect()
    kinds = collector.kinds()
    assert kinds[-1] == "finished", collector.events[-1].data
    rows = [event.data for event in collector.events if event.kind == "row"]
    assert len(rows) == 3 and np.all(np.isfinite(rows[-1]["power"]))


DESIGN_PINS = {
    "pressure": {
        "target": "zdw",
        "wavelength": "500 nm",
        "search": ["0.1 bar", "10 bar"],
    },
    "energy": {"target": "soliton_order", "order": 3},
}
"""The design tool's two targets: the ZDW at 500 nm, then a third-order soliton."""


def test_a_design_solves_two_values_without_keeping_its_pins():
    solved = ask("design", {"text": TINY, "pins": DESIGN_PINS})
    pressure, energy = solved["values"]["pressure"], solved["values"]["energy"]
    # The pressure is tools.capillary_design's, and the pulse at it is a third-order
    # soliton (the recipe's own pulse shape, Gaussian by default).
    assert pressure == pytest.approx(1.27791, rel=1e-5)
    assert solved["design"]["soliton_order"] == pytest.approx(3.0, rel=1e-4)
    assert 50e-6 < energy < 500e-6 and solved["design_error"] is None
    # A pin already in the setup is replaced, not added to.
    pinned = ask("design", {"text": PINNED, "pins": DESIGN_PINS})
    assert pinned["values"]["pressure"] == pytest.approx(pressure)
    far = {
        **DESIGN_PINS,
        "pressure": {**DESIGN_PINS["pressure"], "wavelength": "50 nm"},
    }
    with pytest.raises(SolpropError):
        ask("design", {"text": TINY, "pins": far})
    no_width = TINY.replace('fwhm = "10 fs"', 'width = "6 fs"\nshape = "sech"')
    found = ask(
        "design", {"text": no_width, "pins": {"pressure": DESIGN_PINS["pressure"]}}
    )
    assert found["design"] == {} and found["design_error"]
    soliton = Recipe.example("soliton").to_text()
    assert ask("design", {"text": soliton, "pins": {}})["values"] == {}


def test_the_zdw_chart_is_the_pressure_per_radius_and_wavelength():
    chart = ask(
        "zdw_chart",
        {
            "text": TINY,
            "radii": ["20 um", 125e-6, "250 um"],
            "wavelengths": ["200 nm", "500 nm"],
            "pressure_max": 20,
        },
    )
    assert chart["radius"] == pytest.approx([20e-6, 125e-6, 250e-6])
    deep, shallow = (curve["pressure"] for curve in chart["curves"])
    assert deep[0] > deep[1] > deep[2]  # a wider core needs less gas
    assert shallow[0] is None and "500 nm" in chart["error"]
    assert shallow[1] == pytest.approx(1.27791, rel=1e-5)  # the design's pressure
    assert chart["setup"] == {"radius": 125e-6, "pressure": 1.0}
    gradient = TINY.replace('pressure = "1 bar"', 'pressure = ["1 bar", "0 bar"]')
    flat = ask("zdw_chart", {"text": gradient, "radii": [1e-4], "wavelengths": [3e-7]})
    assert flat["setup"]["pressure"] is None and flat["error"] is None
    with pytest.raises(ValueError, match="at most"):
        ask("zdw_chart", {"text": TINY, "radii": [1e-4] * 401, "wavelengths": [5e-7]})


def test_the_modes_along_a_gradient():
    gradient = TINY.replace('pressure = "1 bar"', 'pressure = ["1 bar", "0.1 bar"]')
    entrance = ask("modes", {"text": gradient})["sets"][0]
    halfway = ask("modes", {"text": gradient, "z": "2.5 cm"})["sets"][0]
    assert halfway["z"] == pytest.approx(0.025)
    assert halfway["modes"][0]["zdw"] < entrance["modes"][0]["zdw"]  # less gas


def test_a_route_s_description_carries_the_types_its_keywords_name():
    described = ask("describe", {"route": "waveguide"})
    types = described["types"]
    assert {"RectMode", "SlabMode", "GaussianPulse", "Constant"} <= set(types)
    rect = {entry["name"]: entry for entry in types["RectMode"]}
    assert rect["half_width_x"]["display"] == "um" and rect["half_width_x"]["required"]
    # Nested: a mode's radius path names the profiles, described too.
    assert "PiecewiseLinear" in ask("describe", {"route": "capillary"})["types"]


_RECT = '{ type = "RectMode", half_width_x = "125 um", half_width_y = "50 um" }'
_SLAB_MODES = '{ type = "ModeCollection.slab", half_gap = "63.5 um", orders = [1, 3] }'
SLAB = (
    Recipe.example("hollow_rectangular")
    .to_text()
    .replace(f"modes = [{_RECT}]", f"modes = {_SLAB_MODES}")
    .replace("n_theta = 12\n", "")  # a slab set is uniform in y: one y node
)
"""The waveguide example with a slab mode set in place of its rectangular mode."""


def test_a_slab_set_s_energy_is_per_metre_of_y():
    assert "ModeCollection.slab" in SLAB
    assert ask("summarise", {"text": SLAB})["per_width"]
    assert not ask("summarise", {"text": TINY})["per_width"]


SCAN = TINY.replace("n_save = 6", "n_save = 2") + (
    '\n[scan]\nname = "pressure"\npressure = ["0.5 bar", "1 bar"]\n'
)
"""The tiny capillary at two pressures."""


def test_a_scan_runs_point_by_point_and_stops(tmp_path: Path):
    spec = RunSpec("s1", str(tmp_path / "s1"), recipe=SCAN, base=str(tmp_path))
    collector = Collector(spec).start()
    collector.collect()
    kinds = collector.kinds()
    assert kinds[0] == "scan_started" and kinds[-1] == "finished"
    points = [event.data for event in collector.events if event.kind == "point"]
    assert points[0]["values"] == {"pressure": 0.5} and points[0]["status"] is None
    assert points[-1]["done"] == 2 and points[-1]["status"] == "done"
    final = collector.events[-1].data
    assert final["scan"] and final["failed"] == 0
    assert Path(str(final["path"])).name == "scan.h5"
    # Stopped after its first point, the rest are not run.
    spec = RunSpec("s2", str(tmp_path / "s2"), recipe=SCAN, base=str(tmp_path))
    collector = Collector(spec).start()
    collector.collect(until="point", then="stop")
    assert collector.kinds()[-1] == "stopped"


def test_a_figure_is_drawn_from_a_views_description(tmp_path: Path):
    x = np.linspace(1.0, 2.0, 5)
    pen = {"colour": "#1f77b4", "style": "-", "width": 1.2}
    spec = {
        "xlabel": "Radius (µm)",
        "ylabel": "Pressure (bar)",
        "logx": True,
        "logy": True,
        "curves": [
            {"x": x, "y": x**2, "label": "ZDW 400 nm", **pen},
            {"x": x, "y": x, "label": "", **pen},
        ],
        "lines": [
            {"x": 1.5, "label": "pump", **pen},
            {"x": 1.2, "label": "", **pen},
            {"y": 2.0, "label": "", **pen},
        ],
        "points": [{"x": 1.5, "y": 2.0, "label": "setup", "colour": "#c62828"}],
    }
    for name, head in (("f.pdf", b"%PDF"), ("f.svg", b"<?xml"), ("f.png", b"\x89PNG")):
        written = Path(ask("figure", {"path": str(tmp_path / name), "spec": spec}))
        assert written.read_bytes().startswith(head)
    bare = {"curves": [{"x": x, "y": x, "label": "", **pen}]}
    other = ask("figure", {"path": str(tmp_path / "f.tif"), "spec": bare})
    assert other.endswith("f.png")  # another kind is written as PNG
